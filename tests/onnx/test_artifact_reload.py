"""ONNX export and strict restored-output validation."""

import json
import unittest
import numpy as np
import onnx
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow

from tests.support import onnx as fixtures
from tests.support.onnx import make_graph


class OnnxArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.path = self.fixture.path
        self.config = self.fixture.config
        self.feed = self.fixture.feed

    def test_required_weight_cannot_be_disabled_after_po2(self):
        sim, model = self.fixture.sim(bias=False)
        self.fixture.calibrate(sim, bias=False)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        sim.qc_quantize_op_dict['w'].enabled = False
        with self.assertRaisesRegex(RuntimeError, 'Required quantizers.*w'):
            pipeline.export_onnx_json(sim, model, self.path / 'disabled', 'model')

    def test_artifacts_restore_quantized_outputs_and_compiler_grids(self):
        for matmul_add, mixed in [(False, False), (True, False), (False, True)]:
            with self.subTest(matmul_add=matmul_add, mixed=mixed):
                artifacts, expected = self.fixture.artifacts(matmul_add=matmul_add, mixed=mixed)
                report = self.fixture.verify(artifacts, expected)
                self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_pipeline_records_actual_quantized_reload_validation(self):
        model_path = self.path / 'input.onnx'
        onnx.save(make_graph(), model_path)
        calibration_dir = self.path / 'calib'
        calibration_dir.mkdir()
        np.save(calibration_dir / 'sample0.npy', self.feed['x'])
        np.save(calibration_dir / 'sample1.npy', self.feed['x'] + 1.0)
        for align_bias in (False, True):
            with self.subTest(align_bias=align_bias):
                result = pipeline.run_onnx_ptq(model_path=model_path, calibration_dir=calibration_dir,
                    output_dir=self.path / f'pipeline-{align_bias}', filename_prefix='model',
                    config_file=str(self.config), bitwidth_config_file='', quant_scheme='tf',
                    align_bias_scale=align_bias)
                metadata = json.loads(result['metadata'].read_text())
                self.assertTrue(all(item['max_abs'] == 0 for item in
                    metadata['restored_quantized_vs_pre_export']))
                self.assertIn('b', metadata['coverage']['enabled_quantizer_names'])

    def test_missing_native_encoding_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        path = artifacts['aimet_encodings']
        raw = json.loads(path.read_text())
        raw['param_encodings'] = [item for item in raw['param_encodings'] if item['name'] != 'w']
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, 'Required quantizers.*w'):
            self.fixture.verify(artifacts, expected)

    def test_modified_native_scale_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        path = artifacts['aimet_encodings']
        raw = json.loads(path.read_text())
        raw['param_encodings'][0]['scale'][0] *= 2
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            self.fixture.verify(artifacts, expected)

    def test_missing_compiler_weight_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        path = artifacts['encodings']
        payload = json.loads(path.read_text())
        del payload['param_encodings']['w']
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            self.fixture.verify(artifacts, expected)

    def test_changed_quantized_output_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        with self.assertRaisesRegex(RuntimeError, 'quantized output'):
            self.fixture.verify(artifacts, [value + 1 for value in expected])

    def test_nonfinite_outputs_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for reference, actual in [(0., value), (value, 0.)]:
                with self.subTest(reference=reference, actual=actual):
                    with self.assertRaisesRegex(ValueError, 'NaN.*Inf'):
                        pipeline.compare_outputs([np.array([reference])], [np.array([actual])])


if __name__ == "__main__":
    unittest.main()
