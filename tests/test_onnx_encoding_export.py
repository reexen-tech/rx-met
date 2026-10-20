"""Frontend grid adapters and old export entry points share one compiler format."""
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx

from aimet_common.compiler_encodings import normalize_compiler_encodings, validate_compiler_encodings
from aimet_common.defs import QuantizationDataType
from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
from aimet_onnx.rx_ptq import pipeline


def raw_encoding(bits=8, offset=-128, scale=0.125):
    return dict(bitwidth=bits, dtype='int', is_symmetric='True', scale=scale,
                offset=offset, min=offset * scale, max=(offset + (1 << bits) - 1) * scale)


def graph():
    inputs = [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 2])]
    outputs = [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 2])]
    weight = onnx.numpy_helper.from_array(np.array([[0.5, -0.25], [-0.5, 0.25]], dtype=np.float32), 'w')
    nodes = [onnx.helper.make_node('MatMul', ['x', 'w'], ['y'], name='matmul')]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, 'adapter', inputs, outputs, [weight]),
                                   opset_imports=[onnx.helper.make_opsetid('', 12)])
    model.ir_version = 10
    return model


class OnnxGridAdapterTest(unittest.TestCase):
    def convert(self, entries, *, symmetric=True, unsigned=False, strict=False):
        quantizer = SimpleNamespace(enabled=True, use_symmetric_encodings=symmetric,
                                    use_unsigned_symmetric=unsigned, use_strict_symmetric=strict,
                                    data_type=QuantizationDataType.int,
                                    get_encodings=lambda: [SimpleNamespace(delta=e['scale'], min=e['min'],
                                        max=e['max'], offset=e['offset']) for e in entries],
                                    export_encodings=lambda version: entries)
        sim = SimpleNamespace(qc_quantize_op_dict={'w': quantizer}, quant_args={})
        payload, coverage = CompilerEncodingConverter(sim, graph()).convert()
        self.assertEqual(coverage['mapped_quantizers'], 1)
        validate_compiler_encodings(payload, require_po2=True)
        return payload['param_encodings']['w']

    def test_signed_symmetric_parameters_match_torch_convention(self):
        for bits in (8, 16, 32):
            with self.subTest(bitwidth=bits):
                raw = raw_encoding(bits=bits, offset=-(1 << (bits - 1)))
                onnx_entry = self.convert([raw])
                torch_raw = dict(raw, offset=0)
                torch_payload = normalize_compiler_encodings({'param_encodings': {'w': [torch_raw]}},
                                                             require_po2=True)
                self.assertEqual(onnx_entry, torch_payload['param_encodings']['w'])
                self.assertEqual(onnx_entry['zero_point'], [0])

    def test_asymmetric_parameters_keep_true_nonzero_additive_offset(self):
        raw = dict(raw_encoding(offset=-7), is_symmetric='False')
        result = self.convert([raw], symmetric=False)
        expected = normalize_compiler_encodings({'param_encodings': {'w': [raw]}}, require_po2=True)
        self.assertEqual(result, expected['param_encodings']['w'])
        self.assertEqual(result['dtype'], 'UINT8')
        self.assertEqual(result['zero_point'], [-7])

    def test_unsigned_symmetric_parameters(self):
        result = self.convert([raw_encoding(offset=0)], unsigned=True)
        self.assertEqual(result['dtype'], 'UINT8')
        self.assertEqual(result['zero_point'], [0])

    def test_strict_symmetric_parameters(self):
        raw = raw_encoding(offset=-127)
        raw['max'] = 127 * raw['scale']
        result = self.convert([raw], strict=True)
        self.assertEqual(result['zero_point'], [0])

    def test_mixed_sign_channels_preserve_effective_signed_grid(self):
        entries = [raw_encoding(), raw_encoding(offset=0, scale=0.25)]
        result = self.convert(entries, unsigned=True)
        self.assertEqual(result['dtype'], 'INT8')
        self.assertEqual(result['zero_point'], [0, 128])
        for index, entry in enumerate(entries):
            for unsigned_q in (0, 127, 255):
                signed_q = unsigned_q - 128
                actual = (signed_q + result['zero_point'][index]) * result['scale'][index]
                self.assertEqual(actual, (unsigned_q + entry['offset']) * entry['scale'])


class ExportEntryPointTest(unittest.TestCase):
    def test_old_onnx_entry_preserves_its_dictionary(self):
        expected = {'onnx': Path('model.onnx'), 'encodings': Path('model.encodings'),
                    'aimet_encodings': Path('model.aimet.encodings'), 'coverage': {'mapped_quantizers': 3}}
        sim, model = object(), object()
        with patch.object(pipeline, 'export_onnx_json', return_value=expected) as export:
            result = pipeline.export_onnx_compiler_artifacts(sim, model, 'output', 'model')
        self.assertIs(result, expected)
        export.assert_called_once_with(sim, model, 'output', 'model')

    def test_old_torch_entry_preserves_defaults_arguments_and_string_tuple(self):
        exporter = importlib.import_module('aimet_torch.rx_export.export_onnx_json')
        signature = inspect.signature(exporter.export_onnx_json)
        self.assertEqual(list(signature.parameters),
                         ['sim', 'export_dir', 'filename_prefix', 'dummy_input_shape', 'opset'])
        self.assertEqual(signature.parameters['dummy_input_shape'].default, exporter.STATIC_DUMMY_INPUT_SHAPE)
        self.assertEqual(signature.parameters['opset'].default, 18)
        bound = signature.bind(object(), 'output', 'model', dummy_input_shape=(2, 4), opset=13)
        self.assertEqual(bound.arguments['dummy_input_shape'], (2, 4))

    def test_public_packages_expose_new_and_legacy_names(self):
        for name, legacy in (('aimet_onnx.rx_ptq', 'export_onnx_compiler_artifacts'),
                             ('aimet_torch.rx_export', 'export_onnx_json')):
            with self.subTest(package=name):
                package = importlib.import_module(name)
                self.assertTrue(callable(package.export_onnx_json))
                self.assertTrue(callable(getattr(package, legacy)))

    def test_real_onnx_sim_exports_zero_point_and_clean_graph(self):
        with tempfile.TemporaryDirectory(prefix='onnx-encoding-export-') as temporary:
            path = Path(temporary)
            config = path / 'config.json'
            config.write_text(json.dumps({
                'defaults': {'ops': {'is_output_quantized': 'True', 'is_symmetric': 'True'},
                             'params': {'is_quantized': 'True', 'is_symmetric': 'True'},
                             'per_channel_quantization': 'True', 'strict_symmetric': 'False'},
                'params': {}, 'op_type': {}, 'supergroups': [], 'model_input': {}, 'model_output': {}}))
            model = graph()
            feed = {'x': np.array([[-1.0, 2.0]], dtype=np.float32)}
            sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(config),
                                               quant_scheme='tf', work_dir=path / 'sim')
            sim.compute_encodings(lambda session: session.run(None, feed))
            pipeline.apply_power_of_2_workflow(sim, align_bias_scale=False)
            result = pipeline.export_onnx_json(sim, model, path / 'export', 'model')
            self.assertTrue(all(isinstance(result[key], Path) and result[key].is_file()
                                for key in ('onnx', 'encodings', 'aimet_encodings')))
            encoded = json.loads(result['encodings'].read_text())
            self.assertEqual(set(encoded['param_encodings']['w']['zero_point']), {0})
            self.assertEqual(encoded['param_encodings']['w']['dtype'], 'INT8')
            clean = onnx.load(result['onnx'])
            self.assertFalse(any(node.op_type == 'QcQuantizeOp' for node in clean.graph.node))
            validate_compiler_encodings(encoded, require_po2=True)


if __name__ == '__main__':
    unittest.main()
