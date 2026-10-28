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
from aimet_common.defs import EncodingType, QuantizationDataType
from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
from aimet_onnx.rx_ptq import pipeline



def raw_encoding(bits=8, offset=-128, scale=0.125):
    return dict(bitwidth=bits, dtype='int', is_symmetric='True', enc_type='PER_TENSOR', scale=scale,
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
    def convert(self, entries, *, symmetric=True, unsigned=False, strict=False, enc_type=None, signed=True):
        enc_type = enc_type or (EncodingType.PER_CHANNEL if len(entries) > 1 else EncodingType.PER_TENSOR)
        quantizer = SimpleNamespace(enabled=True, use_symmetric_encodings=symmetric,
                                    use_unsigned_symmetric=unsigned, use_strict_symmetric=strict,
                                    data_type=QuantizationDataType.int,
                                    signed=signed,
                                    get_encodings=lambda: [SimpleNamespace(delta=e['scale'], min=e['min'],
                                        max=e['max'], offset=e['offset']) for e in entries],
                                    export_encodings=lambda version: entries,
                                    _encoding_type=lambda: enc_type)
        sim = SimpleNamespace(qc_quantize_op_dict={'w': quantizer}, quant_args={})
        model = graph()
        if enc_type == EncodingType.PER_CHANNEL and len(entries) == 1:
            weight = onnx.numpy_helper.to_array(model.graph.initializer[0])[:, :1].copy()
            model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(weight, 'w'))
            model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
        payload, coverage = CompilerEncodingConverter(sim, model).convert()
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

    def test_asymmetric_parameters_use_subtractive_zero_point(self):
        raw = dict(raw_encoding(offset=-7), is_symmetric='False')
        result = self.convert([raw], symmetric=False, signed=False)
        expected = normalize_compiler_encodings({'param_encodings': {'w': [raw]}}, require_po2=True)
        self.assertEqual(result, expected['param_encodings']['w'])
        self.assertEqual(result['dtype'], 'UINT8')
        self.assertEqual(result['zero_point'], [7])

    def test_unsigned_symmetric_parameters(self):
        result = self.convert([raw_encoding(offset=0)], unsigned=True, signed=False)
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
        self.assertEqual(result['zero_point'], [0, -128])
        self.assertEqual(result['is_symmetric'], 'False')
        for index, entry in enumerate(entries):
            for unsigned_q in (0, 127, 255):
                signed_q = unsigned_q - 128
                actual = (signed_q - result['zero_point'][index]) * result['scale'][index]
                self.assertEqual(actual, (unsigned_q + entry['offset']) * entry['scale'])

    def test_single_channel_granularity_comes_from_quantizer(self):
        for kind in (EncodingType.PER_TENSOR, EncodingType.PER_CHANNEL):
            with self.subTest(enc_type=kind.name):
                result = self.convert([raw_encoding()], enc_type=kind)
                self.assertEqual(result['scale'], [0.125])
                self.assertEqual(result['enc_type'], kind.name)


class TorchEncodingMetadataTest(unittest.TestCase):
    def test_legacy_export_retains_tensor_channel_and_block_granularity(self):
        import torch
        from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
        for shape, block_size, kind in (((), None, 'PER_TENSOR'), ((1,), None, 'PER_CHANNEL'),
                                         ((2,), None, 'PER_CHANNEL'), ((2, 2), (1, 2), 'PER_BLOCK')):
            with self.subTest(shape=shape, block_size=block_size):
                encoding = AffineEncoding(torch.full(shape, 0.125), torch.zeros(shape),
                                          8, True, True, block_size=block_size)
                records = encoding.to_qnn_encoding_dict('0.6.1')
                self.assertTrue(records)
                self.assertEqual({entry['enc_type'] for entry in records}, {kind})


class ExportEntryPointTest(unittest.TestCase):
    def test_old_onnx_entry_preserves_its_dictionary(self):
        expected = {'onnx': Path('model.onnx'), 'encodings': Path('model.json'),
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
            self.assertEqual(result['encodings'], result['onnx'].with_suffix('.json'))
            self.assertEqual(result['aimet_encodings'], result['onnx'].with_suffix('.aimet.encodings'))
            encoded = json.loads(result['encodings'].read_text())
            raw = json.loads(result['aimet_encodings'].read_text())
            self.assertEqual(encoded['schema_version'], 4)
            self.assertNotIn('schema_version', raw)
            self.assertEqual(raw, json.loads(result['onnx'].with_suffix('.encodings').read_text()))
            self.assertEqual(set(encoded['param_encodings']['w']['zero_point']), {0})
            self.assertEqual(encoded['param_encodings']['w']['dtype'], 'INT8')
            self.assertEqual(encoded['param_encodings']['w']['enc_type'], 'PER_CHANNEL')
            clean = onnx.load(result['onnx'])
            self.assertFalse(any(node.op_type == 'QcQuantizeOp' for node in clean.graph.node))
            validate_compiler_encodings(encoded, require_po2=True)


if __name__ == "__main__":
    unittest.main()
