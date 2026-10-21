"""Shared percentile regressions through real Torch and ONNX calibration paths."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch

from aimet_common import libquant_info
from aimet_common.defs import QuantScheme
from aimet_common.percentile import PercentileCalibrator
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode, TensorQuantizerParams
from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
from aimet_torch.v2.quantization.encoding_analyzer import (
    PercentileEncodingAnalyzer, TfEnhancedEncodingAnalyzer,
)


def quantizer(*, percentile=99.99, symmetric=True, shape=None, channel=None):
    params = TensorQuantizerParams(shape, channel_axis=channel) if shape else None
    q = QcQuantizeOp(libquant_info.QcQuantizeInfo(),
                    quant_scheme=QuantScheme.post_training_percentile,
                    op_mode=OpMode.updateStats, use_symmetric_encodings=symmetric,
                    tensor_quantizer_params=params)
    q.use_strict_symmetric = False
    q.use_unsigned_symmetric = False
    if channel is not None:
        q.enable_per_channel_quantization()
    q.set_percentile_value(percentile)
    return q


class SharedPercentileTest(unittest.TestCase):
    def test_later_large_batch_and_order(self):
        small = np.linspace(0, 1, 1024, dtype=np.float32)
        large = np.linspace(0, 100, 1024, dtype=np.float32)
        for percentile in (99.99, 100):
            for batches in ((small, large), (large, small)):
                with self.subTest(percentile=percentile, first_max=batches[0].max()):
                    q = quantizer(percentile=percentile)
                    reference = PercentileEncodingAnalyzer((), percentile=percentile)
                    for batch in batches:
                        q.update_encoding_stats(batch)
                        reference.update_stats(torch.from_numpy(batch))
                    encoding = q.compute_encodings()[0]
                    low, high = reference.compute_encodings(255, True)
                    self.assertAlmostEqual(encoding.min, low.item(), delta=2e-5)
                    self.assertAlmostEqual(encoding.max, high.item(), delta=2e-5)
                    self.assertGreater(encoding.max, 99)
                    self.assertEqual(q._percentile_calibrator.observer.stats[0].histogram.sum().item(), 2048)

    def test_constants_zero_and_reset_preserve_all_counts(self):
        for constant in (0, 3, -2):
            state = PercentileCalibrator(percentile=100)
            state.update_stats(torch.full((1024,), float(constant)))
            state.update_stats(torch.linspace(-100, 100, 1024))
            self.assertEqual(state.observer.stats[0].histogram.sum().item(), 2048)
            low, high = state.compute_encodings(255, True)
            self.assertLessEqual(low.item(), -100)
            self.assertGreaterEqual(high.item(), 100)
            state.reset_stats()
            with self.assertRaises(RuntimeError):
                state.compute_encodings(255, True)

    def test_per_channel_and_asymmetric_ranges_match_torch(self):
        for symmetric in (False, True):
            q = quantizer(symmetric=symmetric, shape=(2, 1024), channel=0)
            reference = PercentileEncodingAnalyzer((2, 1), percentile=99.99)
            for factor in (1, 100):
                values = np.stack((np.linspace(-factor, factor, 1024),
                                   np.linspace(0, factor * 3, 1024))).astype(np.float32)
                q.update_encoding_stats(values)
                reference.update_stats(torch.from_numpy(values))
            encodings = q.compute_encodings()
            low, high = reference.compute_encodings(255, symmetric)
            for index, encoding in enumerate(encodings):
                self.assertAlmostEqual(encoding.delta, ((high - low) / 255).flatten()[index].item(), places=6)

    def test_percentile_survives_quantizer_rebuild_and_freeze(self):
        q = quantizer(percentile=90, shape=(2, 1024), channel=0)
        q.enable_per_channel_quantization(False)
        values = np.arange(2048, dtype=np.float32).reshape(2, 1024)
        q.update_encoding_stats(values)
        self.assertEqual(q._percentile_calibrator.percentile, 90)
        original = [(e.delta, e.offset) for e in q.compute_encodings()]
        q.freeze_encodings()
        q.reset_encoding_stats()
        q.update_encoding_stats(values * 100)
        self.assertIsNone(q.compute_encodings())
        self.assertEqual(original, [(e.delta, e.offset) for e in q.get_encodings()])

    def test_nonleading_channel_axis_keeps_observations_separate(self):
        state = PercentileCalibrator((1, 2), percentile=100)
        first = torch.tensor([[1., 10.], [2., 20.], [3., 30.]])
        second = first * 100
        state.update_stats(first)
        state.update_stats(second)
        for index in range(2):
            channel = PercentileCalibrator(percentile=100)
            channel.update_stats(first[:, index])
            channel.update_stats(second[:, index])
            self.assertTrue(torch.equal(state.observer.stats[index].histogram,
                                        channel.observer.stats[0].histogram))

    def test_tiny_initial_range_does_not_discard_history(self):
        state = PercentileCalibrator(percentile=100)
        state.update_stats(torch.linspace(0, 1e-9, 1024))
        state.update_stats(torch.linspace(0, 100, 1024))
        self.assertEqual(state.observer.stats[0].histogram.sum().item(), 2048)

    def test_tf_enhanced_keeps_its_explicit_growth_limit(self):
        analyzer = TfEnhancedEncodingAnalyzer(())
        analyzer.update_stats(torch.linspace(0, 1, 1024))
        analyzer.update_stats(torch.linspace(0, 100, 1024))
        self.assertEqual(analyzer.observer.growth_limit, 3)
        self.assertLess(analyzer.observer.stats[0].bin_edges[-1].item(), 10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_same_shared_histogram_on_cpu_and_cuda(self):
        cpu, cuda = PercentileCalibrator(percentile=100), PercentileCalibrator(percentile=100)
        for maximum in (1, 100):
            values = torch.linspace(0, maximum, 1024)
            cpu.update_stats(values)
            cuda.update_stats(values.cuda())
        self.assertEqual(cuda.observer.stats[0].histogram.sum().item(), 2048)
        for actual, expected in zip(cuda.compute_encodings(255, True), cpu.compute_encodings(255, True)):
            torch.testing.assert_close(actual.cpu(), expected)

    def test_strict_and_unsigned_symmetric_native_grids(self):
        for strict, unsigned in ((True, False), (False, True)):
            q = quantizer(percentile=100)
            q.use_strict_symmetric = strict
            q.use_unsigned_symmetric = unsigned
            q.update_encoding_stats(np.linspace(0, 100, 1024, dtype=np.float32))
            encoding = q.compute_encodings()[0]
            self.assertGreaterEqual(encoding.max, 99.99)
            self.assertEqual(encoding.offset, 0 if unsigned else -127)
            self.assertEqual(encoding.min, 0 if unsigned else -encoding.max)

    def test_blockwise_calibration_matches_separate_blocks(self):
        q = quantizer(percentile=100, shape=(2, 4), channel=0)
        q.tensor_quantizer_params.block_axis = 1
        q._enable_blockwise_quantization(2)
        values = np.array([[1, 2, 10, 20], [-1, -2, -10, -20]], dtype=np.float32)
        q.update_encoding_stats(values)
        encodings = q.compute_encodings()
        for actual, block in zip(encodings, values.reshape(4, 2)):
            single = quantizer(percentile=100)
            single.update_encoding_stats(block)
            self.assertEqual(actual.delta, single.compute_encodings()[0].delta)


class OnnxPercentileIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.config = self.path / "config.json"
        self.config.write_text(json.dumps({
            "defaults": {"ops": {"is_output_quantized": "True", "is_symmetric": "True"},
                         "params": {"is_quantized": "True", "is_symmetric": "True"},
                         "per_channel_quantization": "False", "strict_symmetric": "False",
                         "unsigned_symmetric": "False"},
            "params": {}, "op_type": {}, "supergroups": [],
            "model_input": {"is_input_quantized": "True"}, "model_output": {},
        }))

    def sim(self, *, weight=False):
        nodes = [helper.make_node("Relu", ["x"], ["y"], name="relu")]
        weights = []
        if weight:
            weights = [numpy_helper.from_array(np.linspace(-100, 100, 1024, dtype=np.float32).reshape(1, -1), "w")]
            nodes = [helper.make_node("Mul", ["x", "w"], ["y"], name="multiply")]
        graph = helper.make_graph(nodes, "percentile", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1024])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1024])], weights)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        sim = QuantizationSimModel(model, quant_scheme="percentile", config_file=str(self.config),
                                   dummy_input={"x": np.ones((1, 1024), dtype=np.float32)})
        sim.set_percentile_value(100)
        return sim

    def feeds(self):
        return [{"x": np.linspace(0, maximum, 1024, dtype=np.float32).reshape(1, -1)}
                for maximum in (1, 100)]

    def test_iterable_callback_and_context_use_shared_statistics(self):
        for mode in ("iterable", "callback", "context"):
            with self.subTest(mode=mode):
                sim = self.sim()
                original = sim.session
                original_model = sim.model.model.SerializeToString()
                if mode == "iterable":
                    sim.compute_encodings(self.feeds())
                elif mode == "callback":
                    sim.compute_encodings(lambda session: [session.run(
                        [session.get_outputs()[0].name], feed) for feed in self.feeds()])
                else:
                    with compute_encodings(sim):
                        for feed in self.feeds():
                            outputs = sim.session.run(None, feed)
                            self.assertEqual(len(outputs), 1)
                self.assertIs(sim.session, original)
                self.assertEqual(sim.model.model.SerializeToString(), original_model)
                self.assertGreater(sim.qc_quantize_op_dict["x"].get_encodings()[0].max, 99.99)
                self.assertGreater(sim.qc_quantize_op_dict["y"].get_encodings()[0].max, 99.99)
                self.assertTrue(np.isfinite(sim.session.run(None, self.feeds()[1])[0]).all())

    def test_parameters_and_recalibration(self):
        sim = self.sim(weight=True)
        sim.compute_encodings(self.feeds())
        self.assertTrue(sim.qc_quantize_op_dict["w"].is_initialized())
        self.assertIsNotNone(sim.qc_quantize_op_dict["w"]._percentile_calibrator)
        sim.compute_encodings(self.feeds()[:1])
        self.assertLess(sim.qc_quantize_op_dict["x"].get_encodings()[0].max, 2)

    def test_failed_callback_restores_runtime_session(self):
        sim = self.sim()
        original = sim.session
        def fail(session):
            session.run(None, self.feeds()[0])
            raise ValueError("callback failed")
        with self.assertRaisesRegex(ValueError, "callback failed"):
            sim.compute_encodings(fail)
        self.assertIs(sim.session, original)
        sim.compute_encodings(self.feeds())
        self.assertTrue(sim.qc_quantize_op_dict["x"].is_initialized())

    def test_frozen_activation_retains_its_runtime_grid(self):
        sim = self.sim()
        sim.compute_encodings(self.feeds()[:1])
        q = sim.qc_quantize_op_dict["x"]
        expected = q.get_encodings()[0].delta
        q.freeze_encodings()
        with compute_encodings(sim):
            self.assertEqual(q.op_mode, OpMode.quantizeDequantize)
            sim.session.run(None, self.feeds()[1])
        self.assertEqual(q.get_encodings()[0].delta, expected)


if __name__ == "__main__":
    unittest.main()
