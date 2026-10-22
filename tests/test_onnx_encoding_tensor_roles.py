"""Compiler grids must stay on floating data, never indices or control ports."""

import json
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
import test_onnx_ptq_alignment as fixtures


class OnnxEncodingTensorRolesTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path
        self.index = 0

    def model(self, nodes, inputs, outputs, constants=None):
        def infos(values):
            return [helper.make_tensor_value_info(name, dtype, shape)
                    for name, (dtype, shape) in values.items()]
        graph = helper.make_graph(nodes, 'tensor-roles', infos(inputs), infos(outputs),
            [numpy_helper.from_array(value, name) for name, value in (constants or {}).items()])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)

    def export(self, model, feed, *, disable=()):
        self.index += 1
        sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.fixture.config),
            quant_scheme='tf', work_dir=self.path / f'sim{self.index}')
        sim.compute_encodings([feed])
        for name in disable:
            sim.qc_quantize_op_dict[name].enabled = False
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        output = sim.session.run(None, feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / f'export{self.index}', 'model')
        payload = json.loads(artifacts['encodings'].read_text())
        return sim, artifacts, payload, output

    def test_gather_does_not_propagate_data_grid_to_computed_indices(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                model = self.model([
                    helper.make_node('Constant', [], ['ids'], name='ids',
                        value=numpy_helper.from_array(np.array([0, 2], dtype=np.int64))),
                    helper.make_node('Add', ['ids', 'shift'], ['positions'], name='index_add'),
                    helper.make_node('Gather', ['x', 'positions'], ['y'], name='gather', axis=1),
                ], {'x': (TensorProto.FLOAT, [2, 4])}, {'y': (TensorProto.FLOAT, [2, 2])},
                    {'shift': np.array(0, dtype=np.int64)})
                feed = {'x': np.arange(8, dtype=np.float32).reshape(2, 4) / 8}
                sim, artifacts, payload, output = self.export(model, feed, disable=('x',) if reverse else ())
                encodings = payload['activation_encodings']
                self.assertNotIn('index_add', encodings)
                self.assertEqual(set(encodings['gather']['input']), {'0'})
                self.assertIn('0', encodings['gather']['output'])
                self.assertNotIn('positions', artifacts['coverage']['propagated_tensors'])
                saved = onnx.load(artifacts['onnx'])
                self.assertEqual(list(saved.graph.node), list(model.graph.node))
                positions = next(value for value in saved.graph.value_info if value.name == 'positions')
                self.assertEqual(positions.type.tensor_type.elem_type, TensorProto.INT64)
                pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
                    config_file=str(self.fixture.config), quant_scheme='tf')

    def test_graph_input_indices_and_bool_controls_remain_unquantized(self):
        cases = (
            ('Gather', 'ids', TensorProto.INT64, np.array([0, 2], dtype=np.int64)),
            ('Compress', 'mask', TensorProto.BOOL, np.array([True, False, True, False])),
        )
        for op_type, control, dtype, values in cases:
            with self.subTest(op_type=op_type):
                model = self.model([
                    helper.make_node(op_type, ['x', control], ['y'], name='select', axis=1),
                ], {'x': (TensorProto.FLOAT, [2, 4]), control: (dtype, list(values.shape))},
                    {'y': (TensorProto.FLOAT, [2, 2])})
                feed = {'x': np.arange(8, dtype=np.float32).reshape(2, 4) / 8, control: values}
                sim, artifacts, payload, output = self.export(model, feed)
                self.assertNotIn(control, sim.qc_quantize_op_dict)
                self.assertNotIn(control, artifacts['coverage']['propagated_tensors'])
                self.assertEqual(set(payload['activation_encodings']['select']['input']), {'0'})
                pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
                    config_file=str(self.fixture.config), quant_scheme='tf')

    def test_topk_maxpool_and_dropout_do_not_encode_index_or_mask_outputs(self):
        cases = (
            (helper.make_node('TopK', ['x', 'k'], ['y', 'control'], name='op'),
             [2, 4], [2, 2], TensorProto.INT64, {'k': np.array([2], dtype=np.int64)}),
            (helper.make_node('MaxPool', ['x'], ['y', 'control'], name='op', kernel_shape=[2, 2]),
             [1, 1, 3, 3], [1, 1, 2, 2], TensorProto.INT64, {}),
            (helper.make_node('Dropout', ['x'], ['y', 'control'], name='op'),
             [2, 4], [2, 4], TensorProto.BOOL, {}),
        )
        for node, shape, output_shape, dtype, constants in cases:
            with self.subTest(op_type=node.op_type):
                model = self.model([node], {'x': (TensorProto.FLOAT, shape)},
                    {'y': (TensorProto.FLOAT, output_shape), 'control': (dtype, output_shape)}, constants)
                feed = {'x': np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)}
                sim, artifacts, payload, output = self.export(model, feed)
                self.assertNotIn('control', sim.qc_quantize_op_dict)
                self.assertEqual(set(payload['activation_encodings']['op']['output']), {'0'})
                self.assertNotIn('control', artifacts['coverage']['propagated_tensors'])
                self.assertEqual(output[1].dtype, np.dtype('bool' if dtype == TensorProto.BOOL else 'int64'))
                pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
                    config_file=str(self.fixture.config), quant_scheme='tf')

    def test_nonzero_does_not_inherit_its_float_input_grid(self):
        model = self.model([helper.make_node('NonZero', ['x'], ['y'], name='nonzero')],
            {'x': (TensorProto.FLOAT, [2, 4])}, {'y': (TensorProto.INT64, [2, 'count'])})
        feed = {'x': np.arange(8, dtype=np.float32).reshape(2, 4) / 8}
        sim, artifacts, payload, output = self.export(model, feed)
        self.assertNotIn('y', sim.qc_quantize_op_dict)
        self.assertNotIn('output', payload['activation_encodings']['nonzero'])
        self.assertNotIn('y', artifacts['coverage']['propagated_tensors'])
        self.assertEqual(output[0].dtype, np.dtype('int64'))
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
            config_file=str(self.fixture.config), quant_scheme='tf')

    def test_float_padding_control_does_not_inherit_data_grid(self):
        model = self.model([helper.make_node('Pad', ['x', 'pads', 'fill'], ['y'], name='pad')],
            {'x': (TensorProto.FLOAT, [1, 2]), 'fill': (TensorProto.FLOAT, [])},
            {'y': (TensorProto.FLOAT, [1, 4])}, {'pads': np.array([0, 1, 0, 1], dtype=np.int64)})
        feed = {'x': np.array([[-1., 1.]], dtype=np.float32), 'fill': np.array(.25, dtype=np.float32)}
        sim, artifacts, payload, output = self.export(model, feed, disable=('fill',))
        self.assertFalse(sim.qc_quantize_op_dict['fill'].enabled)
        self.assertEqual(set(payload['activation_encodings']['pad']['input']), {'0'})
        self.assertNotIn('fill', artifacts['coverage']['propagated_tensors'])
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
            config_file=str(self.fixture.config), quant_scheme='tf')

    def test_split_propagates_to_every_float_output(self):
        model = self.model([helper.make_node('Split', ['x', 'sizes'], ['first', 'second'], name='split', axis=1)],
            {'x': (TensorProto.FLOAT, [2, 4])},
            {'first': (TensorProto.FLOAT, [2, 1]), 'second': (TensorProto.FLOAT, [2, 3])},
            {'sizes': np.array([1, 3], dtype=np.int64)})
        feed = {'x': np.arange(8, dtype=np.float32).reshape(2, 4) / 8}
        _, artifacts, payload, output = self.export(model, feed, disable=('first', 'second'))
        encodings = payload['activation_encodings']['split']
        self.assertEqual(set(encodings['input']), {'0'})
        self.assertEqual(set(encodings['output']), {'0', '1'})
        self.assertEqual(encodings['input']['0'], encodings['output']['0'])
        self.assertEqual(encodings['input']['0'], encodings['output']['1'])
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
            config_file=str(self.fixture.config), quant_scheme='tf')

    def test_export_rejects_enabled_quantizers_on_non_float_tensors_or_ports(self):
        for port_alias in (False, True):
            with self.subTest(port_alias=port_alias):
                model = self.model([
                    helper.make_node('Constant', [], ['ids'], name='ids',
                        value=numpy_helper.from_array(np.array([0, 2], dtype=np.int64))),
                    helper.make_node('Gather', ['x', 'ids'], ['y'], name='gather', axis=1),
                ], {'x': (TensorProto.FLOAT, [2, 4])}, {'y': (TensorProto.FLOAT, [2, 2])})
                feed = {'x': np.arange(8, dtype=np.float32).reshape(2, 4) / 8}
                sim, _, _, _ = self.export(model, feed)
                name = 'invalid_port' if port_alias else 'ids'
                sim.qc_quantize_op_dict[name] = sim.qc_quantize_op_dict['x']
                if port_alias:
                    sim._rx_input_quantizers['gather', 1] = name
                with self.assertRaisesRegex(ValueError, 'non-floating.*ids.*INT64'):
                    CompilerEncodingConverter(sim, model).convert()

    def test_reload_rejects_sidecar_encoding_added_to_index_port(self):
        model = self.model([helper.make_node('Gather', ['x', 'ids'], ['y'], name='gather', axis=1)],
            {'x': (TensorProto.FLOAT, [2, 4]), 'ids': (TensorProto.INT64, [2])},
            {'y': (TensorProto.FLOAT, [2, 2])})
        feed = {'x': np.arange(8, dtype=np.float32).reshape(2, 4) / 8,
                'ids': np.array([0, 2], dtype=np.int64)}
        _, artifacts, payload, output = self.export(model, feed)
        inputs = payload['activation_encodings']['gather']['input']
        inputs['1'] = inputs['0'].copy()
        artifacts['encodings'].write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, 'Restored quantizers do not match'):
            pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
                config_file=str(self.fixture.config), quant_scheme='tf')


if __name__ == '__main__':
    unittest.main()
