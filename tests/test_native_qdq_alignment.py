"""Native CPU/CUDA QDQ must implement the same integer grids as Torch."""

import unittest

import numpy as np
from onnx import helper, TensorProto
import torch

from aimet_common import libpymo, libquant_info
from aimet_common.defs import QuantScheme
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode, TensorQuantizerParams
from aimet_onnx.rx_ptq.pipeline import create_onnx_ptq_sim
from aimet_onnx.rx_ptq.execution import ptq_execution_providers
from aimet_torch.v2.quantization.affine import QuantizeDequantize
import test_onnx_ptq_alignment as fixtures


def grid(bits, scale, offset=None):
    encoding = libpymo.TfEncoding()
    encoding.bw = bits
    encoding.delta = scale
    encoding.offset = -(1 << (bits - 1)) if offset is None else offset
    encoding.min = encoding.offset * scale
    encoding.max = (encoding.offset + (1 << bits) - 1) * scale
    return encoding


class NativeQdqAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def compare(self, values, bits, scales, *, symmetric=True, offset=None, expected=None):
        shape = (len(scales), 1) if len(scales) > 1 else ()
        native = QcQuantizeOp(libquant_info.QcQuantizeInfo(), bitwidth=bits,
                              quant_scheme=QuantScheme.post_training_tf,
                              op_mode=OpMode.quantizeDequantize,
                              use_symmetric_encodings=symmetric)
        if shape:
            native.tensor_quantizer_params = TensorQuantizerParams(values.shape, channel_axis=0)
            native.enable_per_channel_quantization()
        encodings = [grid(bits, scale, offset) for scale in scales]
        native.load_encodings(encodings)
        reference = QuantizeDequantize(shape, bits, symmetric, signed=symmetric)
        reference.set_range(torch.tensor([e.min for e in encodings]).reshape(shape),
                            torch.tensor([e.max for e in encodings]).reshape(shape))
        reference._reparametrize_to_scale_offset()
        with torch.no_grad():
            reference.scale.copy_(torch.tensor(scales).reshape(shape))
            if not symmetric:
                reference.offset.fill_(offset)
        reference.requires_grad_(False)
        if expected is None:
            expected = reference(torch.from_numpy(values)).detach().numpy()
        np.testing.assert_allclose(native.quantize_dequantize(values), expected,
                                   rtol=0, atol=0, equal_nan=True)

        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"], name="identity")],
                                  "qdq", [helper.make_tensor_value_info("x", TensorProto.FLOAT, values.shape)],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, values.shape)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
        for provider in providers:
            sim = create_onnx_ptq_sim(model, {"x": np.zeros_like(values)}, quant_scheme="tf",
                                     config_file=str(self.fixture.config), providers=[provider])
            for quantizer in sim.qc_quantize_op_dict.values():
                quantizer.set_bitwidth(bits)
                quantizer.use_symmetric_encodings = symmetric
                if shape:
                    quantizer.tensor_quantizer_params = TensorQuantizerParams(values.shape, channel_axis=0)
                    quantizer.enable_per_channel_quantization()
                quantizer.load_encodings([grid(bits, scale, offset) for scale in scales])
            actual = sim.session.run(None, {"x": values})[0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True,
                                       err_msg=provider)

    def test_exact_integer_codes_survive_int32_offset(self):
        codes = np.array([-257, -65, -7, -3, -1, 0, 1, 3, 7, 65, 257], dtype=np.float32)
        self.compare((codes * (2. ** -21)).reshape(1, -1), 32, [2. ** -21])

    def test_ptq_provider_policy_preserves_explicit_options(self):
        self.assertEqual(ptq_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]),
                         [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"])
        options = {"use_tf32": "1", "device_id": "0"}
        self.assertEqual(ptq_execution_providers([("CUDAExecutionProvider", options)]),
                         [("CUDAExecutionProvider", options)])
        self.assertEqual(options, {"use_tf32": "1", "device_id": "0"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_real_cuda_session_defaults_to_full_fp32_matmul(self):
        values = np.ones((1, 4), dtype=np.float32)
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"], name="identity")],
                                  "provider-policy", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        sim = create_onnx_ptq_sim(model, {"x": values}, quant_scheme="tf",
                                  config_file=str(self.fixture.config), providers=["CUDAExecutionProvider"])
        self.assertEqual(sim.session.get_provider_options()["CUDAExecutionProvider"]["use_tf32"], "0")

    def test_per_channel_int32_preserves_small_codes(self):
        codes = np.array([-7, -3, -1, 0, 1, 3, 7], dtype=np.float32)
        scales = [2. ** -21, 2. ** -20]
        self.compare(np.stack([codes * scale for scale in scales]), 32, scales)

    def test_nearest_rounding_is_ties_to_even(self):
        codes = np.array([-7.5, -3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5, 7.5], dtype=np.float32)
        for bits in (8, 16, 32):
            for symmetric in (True, False):
                with self.subTest(bits=bits, symmetric=symmetric):
                    scale = 2. ** -7 if bits == 8 else 2. ** -15 if bits == 16 else 2. ** -21
                    self.compare((codes * scale).reshape(1, -1), bits, [scale],
                                 symmetric=symmetric, offset=None if symmetric else -13)

    def test_decimal_scale_uses_the_same_float32_reciprocal(self):
        codes = np.arange(-63, 64, dtype=np.float32) + .5
        for scale in (1. / 32767, .013007, .027019):
            with self.subTest(scale=scale):
                self.compare((codes * np.float32(scale)).reshape(1, -1), 16, [scale])

    def test_nan_is_preserved_and_infinity_saturates(self):
        for bits in (8, 16, 32):
            with self.subTest(bits=bits):
                scale = 2. ** -7 if bits == 8 else 2. ** -15 if bits == 16 else 2. ** -21
                values = np.array([[np.nan, -np.inf, -scale, 0, scale, np.inf]], dtype=np.float32)
                encoding = grid(bits, scale)
                # Use the grid's saturation contract: upstream Torch CPU fake
                # quantization can wrap positive infinity to its negative bound.
                expected = np.array([[np.nan, encoding.min, -scale, 0, scale, encoding.max]], dtype=np.float32)
                self.compare(values, bits, [scale], expected=expected)


if __name__ == "__main__":
    unittest.main()
