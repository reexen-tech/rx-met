"""Signedness is independent of symmetry through calibration, export and reload."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from aimet_common import libquant_info
from aimet_common.defs import QuantScheme
from aimet_common.quant_utils import _convert_encoding_format_0_6_1_to_1_0_0
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer


def restore(record):
    layer = QuantizationMixin.from_module(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        layer.weight.fill_(1.)
    layer.param_quantizers['weight'] = None
    layer.input_quantizers[0] = None
    layer.output_quantizers[0] = QuantizeDequantize((), 16, True)
    model = torch.nn.Sequential(layer)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'compiler.encodings'
        path.write_text(json.dumps({'schema_version': 4,
            'activation_encodings': {'0': {'output': {'0': record}}}, 'param_encodings': {}}))
        load_quantizer_encodings(model, str(path), allow_overwrite=False,
                                 skip_if_not_found=False, verbose=False)
    return model


class IntegerDomainTest(unittest.TestCase):
    def test_default_integer_domain_does_not_depend_on_symmetry(self):
        for symmetric in (True, False):
            q = QuantizeDequantize((), 8, symmetric)
            self.assertEqual((q.qmin, q.qmax), (-128, 127))
            native = QcQuantizeOp(libquant_info.QcQuantizeInfo(),
                                  use_symmetric_encodings=symmetric, op_mode=OpMode.updateStats)
            self.assertTrue(native.signed)

    def test_all_domains_calibrate_and_preserve_native_qdq(self):
        values = torch.tensor([[-1., -.125, 0., .25, 2.]])
        for signed in (True, False):
            for symmetric in (True, False):
                for scheme in (QuantScheme.post_training_tf, QuantScheme.post_training_percentile):
                    with self.subTest(signed=signed, symmetric=symmetric, scheme=scheme):
                        kwargs = {'signed': signed}
                        if scheme == QuantScheme.post_training_percentile:
                            kwargs['encoding_analyzer'] = PercentileEncodingAnalyzer(())
                        q = QuantizeDequantize((), 8, symmetric, **kwargs)
                        with q.compute_encodings():
                            q(values)
                        q.requires_grad_(False)
                        native = QcQuantizeOp(libquant_info.QcQuantizeInfo(), bitwidth=8,
                            quant_scheme=scheme, use_symmetric_encodings=symmetric,
                            op_mode=OpMode.updateStats, signed=signed)
                        native.update_encoding_stats(values.numpy())
                        native.compute_encodings()
                        self.assertEqual(q.signed, signed)
                        self.assertEqual(native.signed, signed)
                        expected = q(values).detach().numpy()
                        np.testing.assert_array_equal(native.quantize_dequantize(values.numpy()), expected)
                        encoded = native.export_encodings('2.0.0')
                        self.assertEqual(encoded['output_dtype'], ('int8' if signed else 'uint8'))

    def test_compiler_reload_preserves_all_four_integer_grids(self):
        values = torch.tensor([[-40.], [-1.], [0.], [1.], [20.], [40.]])
        for signed in (True, False):
            for symmetric in (True, False):
                with self.subTest(signed=signed, symmetric=symmetric):
                    qmin, qmax = (-128, 127) if signed else (0, 255)
                    point = 0 if symmetric else -7 if signed else 7
                    record = dict(bitwidth=8, dtype='INT8' if signed else 'UINT8',
                        is_symmetric=str(symmetric), enc_type='PER_TENSOR', scale=.125,
                        n=3, zero_point=point, real_min=(qmin-point)*.125,
                        real_max=(qmax-point)*.125)
                    before = copy.deepcopy(record)
                    model = restore(record)
                    q = model[0].output_quantizers[0]
                    self.assertEqual((q.qmin, q.qmax), (qmin, qmax))
                    self.assertEqual(q.symmetric, symmetric)
                    self.assertEqual(q.get_scale().item(), .125)
                    self.assertEqual(q.get_offset().item(), -point)
                    codes = torch.clamp(torch.round(values / .125) + point, qmin, qmax)
                    torch.testing.assert_close(model(values), (codes-point)*.125, rtol=0, atol=0)
                    self.assertEqual(record, before)

    def test_torch_legacy_encoding_preserves_aimet_index_domain(self):
        q = QuantizeDequantize((), 8, True)
        q.set_range(torch.tensor(-16.), torch.tensor(15.875))
        q.requires_grad_(False)
        legacy = q.get_legacy_encodings()
        self.assertEqual(legacy[0]['offset'], -128)
        converted = _convert_encoding_format_0_6_1_to_1_0_0({'x': legacy})[0]
        native = QcQuantizeOp(libquant_info.QcQuantizeInfo(),
                             quant_scheme=QuantScheme.post_training_tf, op_mode=OpMode.updateStats)
        native._load_encodings_dict(converted)
        values = np.array([[-20., -1., 0., 1., 20.]], dtype=np.float32)
        np.testing.assert_array_equal(native.quantize_dequantize(values),
                                      q(torch.from_numpy(values)).detach().numpy())

    def test_asymmetric_domain_change_preserves_real_grid(self):
        q = QuantizeDequantize((), 8, False, signed=True)
        q.set_range(torch.tensor(-1.), torch.tensor(2.))
        q._reparametrize_to_scale_offset()
        q.requires_grad_(False)
        values = torch.linspace(-3, 4, 100)
        before = q(values).detach()
        q.signed = False
        torch.testing.assert_close(q(values), before, rtol=0, atol=0)
        q.signed = True
        torch.testing.assert_close(q(values), before, rtol=0, atol=0)

    def test_encoding_metadata_follows_symmetry_after_scale_offset_conversion(self):
        for signed in (True, False):
            q = QuantizeDequantize((), 8, True, signed=signed)
            q.set_range(torch.tensor(0. if not signed else -16.), torch.tensor(31.875 if not signed else 15.875))
            q._reparametrize_to_scale_offset()
            for symmetric in (False, True):
                q.symmetric = symmetric
                self.assertEqual(q.get_encodings().symmetry, symmetric)
                self.assertEqual(q.signed, signed)

    def test_int32_source_offsets_keep_nonzero_low_bits(self):
        from aimet_common.compiler_encodings import normalize_compiler_encodings
        for signed in (True, False):
            q = QuantizeDequantize((), 32, False, signed=signed)
            q.set_range(torch.tensor(-1.), torch.tensor(1.))
            q._reparametrize_to_scale_offset()
            with torch.no_grad():
                q.scale.fill_(.125)
                q.offset.fill_(-7.)
            raw = q.get_legacy_encodings()
            self.assertEqual(raw[0]['offset'], q.qmin - 7)
            self.assertEqual(q.get_encodings().to_qnn_encoding_dict('1.0.0')['offset'], [q.qmin-7])
            compiled = normalize_compiler_encodings({'param_encodings': {'w': raw}})
            self.assertEqual(compiled['param_encodings']['w']['zero_point'], [7])
            restored = QuantizeDequantize((), 8, True)
            restored.set_legacy_encodings(raw)
            self.assertEqual(restored.signed, signed)
            self.assertEqual(restored.get_scale().item(), .125)
            self.assertEqual(restored.get_offset().item(), -7)

    def test_unsupported_strict_unsigned_mode_is_rejected_before_native_execution(self):
        native = QcQuantizeOp(libquant_info.QcQuantizeInfo(), op_mode=OpMode.updateStats)
        native.use_strict_symmetric = True
        with self.assertRaisesRegex(ValueError, 'signed integer domain'):
            native.signed = False
        self.assertTrue(native.signed)
        native.use_strict_symmetric = False
        native.signed = False
        with self.assertRaisesRegex(ValueError, 'signed integer domain'):
            native.use_strict_symmetric = True

    def test_torch_compiler_reload_rejects_unsupported_narrow_range(self):
        record = dict(bitwidth=8, dtype='INT8', is_symmetric='True',
                      enc_type='PER_TENSOR', scale=.125, n=3, zero_point=0,
                      qmin=-127, qmax=127, real_min=-15.875, real_max=15.875)
        with self.assertRaisesRegex(ValueError, 'narrow integer ranges'):
            restore(record)


class IntegerDomainWorkflowTest(unittest.TestCase):
    def setUp(self):
        import test_onnx_ptq_alignment as fixtures
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_torch_shared_quantizer_conflict_is_rejected_before_mutation(self):
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        model = torch.nn.Sequential(*[QuantizationMixin.from_module(torch.nn.Linear(2, 2))
                                      for _ in range(2)])
        shared = QuantizeDequantize((), 8, True)
        model[0].output_quantizers[0] = shared
        model[1].input_quantizers[0] = shared
        path = self.fixture.path / 'conflict.json'
        path.write_text(json.dumps({'layer_name_config': {
            '0': {'output_signed': False}, '1': {'input_signed': True}}}))
        with self.assertRaisesRegex(ValueError, 'Conflicting shared quantizer signed'):
            apply_mixed_precision_bitwidth(model, str(path), verbose=False)
        self.assertTrue(shared.signed)

    def test_onnx_strict_raw_reload_rejects_signedness_mismatch(self):
        from aimet_onnx.quantsim import load_encodings_to_sim
        sim, _ = self.fixture.sim(bias=False)
        self.fixture.apply_config(sim, {'default_config': {'signed': False}})
        sim.compute_encodings([self.fixture.feed])
        sim.export(str(self.fixture.path), 'unsigned')
        fresh, _ = self.fixture.sim(bias=False)
        before = {name: q.signed for name, q in fresh.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(AssertionError, 'signed'):
            load_encodings_to_sim(fresh, str(self.fixture.path / 'unsigned.encodings'))
        self.assertEqual(before, {name: q.signed for name, q in fresh.qc_quantize_op_dict.items()})

    def test_json_signedness_is_independent_for_both_frontends(self):
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        for signed in (True, False):
            for symmetric in (True, False):
                with self.subTest(signed=signed, symmetric=symmetric):
                    settings = {f'{role}_{field}': value for role in ('weight', 'bias', 'input', 'output')
                        for field, value in (('bitwidth', 8), ('signed', signed), ('symmetric', symmetric))}
                    config = {'layer_type_config': {'QuantizedLinear': settings}}
                    native, _ = self.fixture.sim()
                    self.fixture.apply_config(native, config)
                    layer = QuantizationMixin.from_module(torch.nn.Linear(4, 2))
                    for name in layer.param_quantizers:
                        layer.param_quantizers[name] = QuantizeDequantize((), 16, True)
                    layer.input_quantizers[0] = QuantizeDequantize((), 16, True)
                    layer.output_quantizers[0] = QuantizeDequantize((), 16, True)
                    path = self.fixture.path / 'mixed.json'
                    apply_mixed_precision_bitwidth(torch.nn.Sequential(layer), str(path), verbose=False)
                    for q in list(layer.param_quantizers.values()) + list(layer.input_quantizers) + list(layer.output_quantizers):
                        self.assertEqual(q.signed, signed)
                        self.assertEqual(q.symmetric, symmetric)
                        self.assertEqual(q.bitwidth, 8)
                    for q in native.qc_quantize_op_dict.values():
                        if q.enabled:
                            self.assertEqual(q.signed, signed)
                            self.assertEqual(q.use_symmetric_encodings, symmetric)
                    changed = {role + '_symmetric': not symmetric for role in ('weight', 'bias', 'input', 'output')}
                    path.write_text(json.dumps({'layer_type_config': {'QuantizedLinear': changed}}))
                    apply_mixed_precision_bitwidth(torch.nn.Sequential(layer), str(path), verbose=False)
                    from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
                    apply_onnx_mixed_precision_bitwidth(native, str(path), verbose=False)
                    self.assertEqual(layer.input_quantizers[0].signed, signed)
                    self.assertEqual(native.qc_quantize_op_dict['x'].signed, signed)

    def test_real_onnx_sessions_export_and_reload_all_domains(self):
        import test_onnx_ptq_alignment as fixtures
        from aimet_onnx.rx_ptq import pipeline
        from aimet_onnx.quantsim import load_encodings_to_sim
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.append('CUDAExecutionProvider')
        feed = {'x': np.array([[-1., .25, 1., 2.]], dtype=np.float32)}
        for provider in providers:
            for signed in (True, False):
                for symmetric in (True, False):
                    with self.subTest(provider=provider, signed=signed, symmetric=symmetric):
                        root = self.fixture.path / f'{provider}-{signed}-{symmetric}'
                        model = fixtures.make_graph(bias=False)
                        sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.fixture.config),
                            quant_scheme='percentile', providers=[provider], work_dir=root / 'sim')
                        settings = {f'{role}_{field}': value for role in ('weight', 'input', 'output')
                            for field, value in (('signed', signed), ('symmetric', symmetric))}
                        config = {'layer_type_config': {'QuantizedLinear': settings}}
                        self.fixture.apply_config(sim, config)
                        sim.compute_encodings([feed])
                        pipeline.apply_power_of_2_workflow(sim, align_bias_scale=False)
                        expected = sim.session.run(None, feed)
                        files = pipeline.export_onnx_json(sim, model, root / 'export', 'model')
                        document = json.loads(files['encodings'].read_text())
                        record = document['param_encodings']['w']
                        self.assertEqual(record['dtype'], 'INT8' if signed else 'UINT8')
                        if symmetric:
                            self.assertTrue(all(point == 0 for point in record['zero_point']))
                        fresh = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.fixture.config),
                            quant_scheme='percentile', providers=[provider], work_dir=root / 'fresh')
                        self.fixture.apply_config(fresh, config)
                        load_encodings_to_sim(fresh, str(files['aimet_encodings']), allow_overwrite=False)
                        np.testing.assert_array_equal(fresh.session.run(None, feed)[0], expected[0])
                        for name, q in fresh.qc_quantize_op_dict.items():
                            if q.enabled:
                                self.assertEqual(q.signed, signed, name)


if __name__ == '__main__':
    unittest.main()
