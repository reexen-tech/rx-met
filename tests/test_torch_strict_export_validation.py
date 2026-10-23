"""Reject malformed schema-v4 files after a real calibrated Torch export."""
import copy
import json
import math
import unittest
from pathlib import Path

import torch

import test_torch_export_validation as fixtures
from aimet_common.quantization_grid import IntegerGrid
from aimet_torch.rx_export.validation import collect_expected_encodings, validate_exported_artifacts
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.utils_rx import apply_power_of_2_workflow


class StrictTorchExportValidationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TorchExportValidationTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def int32_export(self):
        sim = self.fixture.sim(calibrate=False)
        for quantizer in sim.model.modules():
            if isinstance(quantizer, QuantizerBase):
                quantizer.bitwidth = 32
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.fixture.x)
        apply_power_of_2_workflow(sim.model, align_bias_scale=False, verbose=False)
        expected = collect_expected_encodings(sim.model)
        with torch.no_grad():
            before = sim.model(self.fixture.x).clone()
        onnx_path, enc_path = self.fixture.export(sim)
        with torch.no_grad():
            torch.testing.assert_close(sim.model(self.fixture.x), before, atol=0, rtol=0)
        document = json.loads(Path(enc_path).read_text())
        return onnx_path, Path(enc_path), expected, document

    @staticmethod
    def output_record(document):
        for name, entry in document['activation_encodings'].items():
            record = entry.get('output', {}).get('0')
            if record and record['dtype'] == 'INT32':
                return name, record
        raise AssertionError('The calibrated fixture did not export an INT32 output')

    def test_int32_export_endpoints_follow_integer_grid_exactly(self):
        _, _, _, document = self.int32_export()
        _, record = self.output_record(document)
        grid = IntegerGrid.from_compiler_record(record)
        self.assertEqual(record['real_min'], grid.real_min)
        self.assertEqual(record['real_max'], grid.real_max)

    def test_int32_endpoint_corruption_is_rejected_without_repair(self):
        graph, path, expected, original = self.int32_export()
        for steps in (1, 256):
            with self.subTest(delta_in_steps=steps):
                document = copy.deepcopy(original)
                _, record = self.output_record(document)
                record['real_max'] += steps * record['scale']
                path.write_text(json.dumps(document))
                before = path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, r'real_max'):
                    validate_exported_artifacts(graph, str(path), expected)
                self.assertEqual(path.read_bytes(), before)

    def test_coherent_grid_change_still_differs_from_live_quantizer(self):
        graph, path, expected, document = self.int32_export()
        _, record = self.output_record(document)
        record['scale'] = math.nextafter(record['scale'], math.inf)
        grid = IntegerGrid.from_compiler_record(record)
        record['real_min'], record['real_max'] = grid.real_min.tolist(), grid.real_max.tolist()
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(RuntimeError, r'scale.*differs from QuantSim'):
            validate_exported_artifacts(graph, str(path), expected)

    def test_current_schema_and_required_grid_fields_are_not_inferred(self):
        graph, path, expected, original = self.int32_export()
        for field in ('schema_version', 'n', 'enc_type'):
            with self.subTest(field=field):
                document = copy.deepcopy(original)
                if field == 'schema_version':
                    document.pop(field)
                else:
                    _, record = self.output_record(document)
                    record.pop(field)
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(RuntimeError, field):
                    validate_exported_artifacts(graph, str(path), expected)

    def test_extra_input_port_is_rejected(self):
        graph, path, expected, document = self.int32_export()
        name, record = self.output_record(document)
        document['activation_encodings'][name].setdefault('input', {})['999'] = copy.deepcopy(record)
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(RuntimeError, r'999'):
            validate_exported_artifacts(graph, str(path), expected)

    def test_unrecognized_record_field_is_rejected(self):
        graph, path, expected, document = self.int32_export()
        _, record = self.output_record(document)
        record['unexpected_grid'] = 7
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(RuntimeError, r'unexpected_grid'):
            validate_exported_artifacts(graph, str(path), expected)

    def test_duplicate_json_key_is_rejected(self):
        graph, path, expected, document = self.int32_export()
        text = json.dumps(document)
        self.assertIn('"schema_version": 4', text)
        path.write_text(text.replace('"schema_version": 4',
                                     '"schema_version": 3, "schema_version": 4'))
        with self.assertRaisesRegex(RuntimeError, r'([Dd]uplicate|重复).*schema_version'):
            validate_exported_artifacts(graph, str(path), expected)

    def test_single_channel_linear_bias_retains_live_granularity(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 1)

            def forward(self, x):
                return self.linear(x)

        sim = self.fixture.sim(Linear())
        source = sim.model.linear.param_quantizers['bias'].get_encodings().to_qnn_encoding_dict('0.6.1')
        _, path = self.fixture.export(sim)
        document = json.loads(Path(path).read_text())
        bias = document['param_encodings']['linear.bias']
        self.assertEqual(bias['enc_type'], source[0]['enc_type'])
        self.assertEqual(bias['enc_type'], 'PER_CHANNEL')
        self.assertIsInstance(bias['scale'], list)
        self.assertEqual(len(bias['scale']), 1)


if __name__ == '__main__':
    unittest.main()
