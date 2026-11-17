"""FP32 graph rewriting has an independent, bounded acceptance check."""

import json
import unittest

import numpy as np
import onnx

from aimet_onnx.rx_ptq import pipeline
from tests.support import onnx as fixtures


class LinearFloatAcceptanceTest(unittest.TestCase):
    def test_allows_float32_roundoff_without_claiming_exact_equality(self):
        expected = [[np.array([1.], dtype=np.float32)]]
        actual = [[np.nextafter(expected[0][0], np.float32(2.))]]
        report = pipeline._check_linear_float_equivalence(expected, actual)
        self.assertGreater(report[0]['max_abs'], 0)
        self.assertLess(report[0]['max_abs'], 1e-6)

    def test_checks_later_calibration_or_validation_batches(self):
        expected = [[np.array([1.], dtype=np.float32)] for _ in range(2)]
        actual = [[np.array([1.], dtype=np.float32)], [np.array([1.001], dtype=np.float32)]]
        with self.assertRaisesRegex(RuntimeError, 'batch 1'):
            pipeline._check_linear_float_equivalence(expected, actual)

    def test_rejects_nonfinite_or_mismatched_outputs(self):
        expected = [[np.array([1.], dtype=np.float32)]]
        with self.assertRaisesRegex(ValueError, 'NaN or Inf'):
            pipeline._check_linear_float_equivalence(expected, [[np.array([np.nan])]])
        with self.assertRaisesRegex(RuntimeError, 'shape differs'):
            pipeline._check_linear_float_equivalence(expected, [[np.ones(2)]])

    def test_workflow_exports_and_restores_the_same_normalized_graph(self):
        fixture = fixtures.OnnxFixture(self)
        for rank in (2, 3):
            with self.subTest(rank=rank):
                model = fixtures.make_graph(matmul_add=True)
                shape = [1, 4] if rank == 2 else [1, 2, 4]
                output_shape = shape[:-1] + [2]
                model.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info('x', 1, shape))
                model.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info('y', 1, output_shape))
                source = fixture.path / f'source-{rank}.onnx'
                onnx.save(model, source)
                calib, validation = fixture.path / f'calib-{rank}', fixture.path / f'val-{rank}'
                calib.mkdir()
                validation.mkdir()
                values = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
                for index, scale in enumerate((1., .5)):
                    np.save(calib / f'{index}.npy', values * scale)
                    np.save(validation / f'{index}.npy', values * scale * .75)
                artifacts = pipeline.run_onnx_ptq(
                    model_path=source, calibration_dir=calib, validation_dir=validation,
                    output_dir=fixture.path / f'workflow-{rank}', filename_prefix='linear',
                    config_file=str(fixture.config), bitwidth_config_file='', quant_scheme='tf')
                exported = onnx.load(str(artifacts['onnx']))
                self.assertEqual(sum(node.op_type == 'Gemm' for node in exported.graph.node), 1)
                self.assertFalse(any(node.op_type in ('MatMul', 'Add') for node in exported.graph.node))
                metadata = json.loads(artifacts['metadata'].read_text())
                self.assertEqual(len(metadata['linear_normalization']['groups']), 1)
                self.assertTrue(metadata['linear_normalization']['calibration'])
                self.assertTrue(metadata['linear_normalization']['evaluation'])
                self.assertEqual(metadata['clean_onnx_vs_normalized_fp32'][0]['max_abs'], 0.)
                self.assertEqual(metadata['restored_quantized_vs_pre_export'][0]['max_abs'], 0.)
                self.assertEqual(onnx.load(str(source)).SerializeToString(), model.SerializeToString())


if __name__ == '__main__':
    unittest.main()
