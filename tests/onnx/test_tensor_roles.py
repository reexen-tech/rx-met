"""Floating data and control/index ports preserve distinct quantization roles."""

import json
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
import onnxruntime as ort
from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.rx_ptq.input_quantization import input_quantizer
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth

from tests.support import onnx as fixtures


class OnnxEncodingTensorRolesTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
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
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings.*(strict validation|do not match)'):
            pipeline.verify_onnx_ptq_artifacts(artifacts, feed, output,
                config_file=str(self.fixture.config), quant_scheme='tf')


class OnnxClipControlTest(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.OnnxFixture(self)
        self.path, self.config = fixture.path, fixture.config
        self.index = 0

    def model(self, source='initializer', *, shared=False, transparent=False, min_input=True):
        bounds = {'lo': np.array(1e-12, dtype=np.float32), 'hi': np.array(3e-12, dtype=np.float32)}
        inputs = [helper.make_tensor_value_info('x', TensorProto.FLOAT, [3])]
        nodes, constants = [], []
        feed = {'x': np.array([0., 2e-12, 8e-12], dtype=np.float32)}
        if source == 'input':
            inputs.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, []) for name in bounds)
            feed.update(bounds)
        elif source == 'constant':
            nodes.extend(helper.make_node('Constant', [], [name], name=name,
                value=numpy_helper.from_array(value)) for name, value in bounds.items())
        else:
            constants.extend(numpy_helper.from_array(value, name) for name, value in bounds.items())
        low = 'lo'
        if transparent:
            nodes.extend([helper.make_node('Identity', ['lo'], ['identity_bound'], name='bound_identity'),
                          helper.make_node('Cast', ['identity_bound'], ['cast_bound'],
                                           name='bound_cast', to=TensorProto.FLOAT)])
            low = 'cast_bound'
        nodes.extend([helper.make_node('Clip', ['x', low if min_input else '', 'hi'], ['clipped'], name='clip'),
                      helper.make_node('Log', ['clipped'], ['y'], name='log')])
        outputs = [helper.make_tensor_value_info('y', TensorProto.FLOAT, [3])]
        if shared:
            nodes.append(helper.make_node('Neg', [low], ['data'], name='data'))
            outputs.append(helper.make_tensor_value_info('data', TensorProto.FLOAT, []))
        model = helper.make_model(helper.make_graph(nodes, 'clip-controls', inputs, outputs, constants),
            opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        return model, feed

    def sim(self, model, feed):
        self.index += 1
        return pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.config),
            quant_scheme='tf', work_dir=self.path / f'sim{self.index}')

    def isolate_bounds(self, sim):
        # Isolate the threshold defect from deliberate data/output quantization.
        for name in ('x', 'clipped', 'y'):
            if name in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[name].enabled = False

    def test_epsilon_bounds_are_preserved_for_initializers_constants_and_inputs(self):
        for source in ('initializer', 'constant', 'input'):
            with self.subTest(source=source):
                model, feed = self.model(source)
                sim = self.sim(model, feed)
                self.isolate_bounds(sim)
                sim.compute_encodings([feed])
                expected = ort.InferenceSession(model.SerializeToString()).run(None, feed)
                actual = sim.session.run(None, feed)
                self.assertTrue(np.isfinite(actual[0]).all())
                np.testing.assert_array_equal(actual[0], expected[0])
                for name in ('lo', 'hi'):
                    q = sim.qc_quantize_op_dict.get(name)
                    self.assertTrue(q is None or not q.enabled, name)

    def test_shared_data_grid_is_not_changed_by_clip_json2(self):
        model, feed = self.model('input', shared=True)
        sim = self.sim(model, feed)
        config = self.path / 'mixed.json'
        config.write_text(json.dumps({'layer_name_config': {
            'clip': {'input_bitwidths': [8, 4, 4]},
            'data': {'input_bitwidth': 16, 'output_bitwidth': 16}}}))
        apply_onnx_mixed_precision_bitwidth(sim, str(config), verbose=False)
        self.assertEqual(sim.qc_quantize_op_dict['lo'].bitwidth, 16)
        self.assertIsNone(input_quantizer(sim, sim.connected_graph.get_all_ops()['clip'], 1))
        runtime = next(node for node in sim.model.model.graph.node if node.name == 'clip')
        self.assertEqual(runtime.input[1], 'lo')
        self.assertTrue(sim.qc_quantize_op_dict['lo'].enabled)

    def test_transparent_control_path_does_not_change_shared_data_branch(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                model, feed = self.model('input', transparent=True, shared=shared)
                sim = self.sim(model, feed)
                self.isolate_bounds(sim)
                sim.compute_encodings([feed])
                actual = sim.session.run(None, feed)
                expected = ort.InferenceSession(model.SerializeToString()).run(None, feed)
                np.testing.assert_array_equal(actual[0], expected[0])
                if shared:
                    self.assertTrue(sim.qc_quantize_op_dict['cast_bound'].enabled)
                    self.assertFalse(np.array_equal(actual[1], expected[1]))

    def test_export_reload_omits_control_encodings_and_rejects_corruption(self):
        model, feed = self.model('input')
        sim = self.sim(model, feed)
        self.isolate_bounds(sim)
        sim.compute_encodings([feed])
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        expected = sim.session.run(None, feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / 'export', 'clip')
        payload = json.loads(artifacts['encodings'].read_text())
        self.assertNotIn('lo', payload['param_encodings'])
        self.assertNotIn('hi', payload['param_encodings'])
        self.assertFalse(set(payload['activation_encodings'].get('clip', {}).get('input', {})) & {'1', '2'})
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
            config_file=str(self.config), quant_scheme='tf')
        raw = json.loads(artifacts['aimet_encodings'].read_text())
        raw['activation_encodings'].append({'name': 'lo', 'dtype': 'INT', 'bw': 8,
            'is_sym': True, 'enc_type': 'PER_TENSOR', 'scale': [1e-7], 'offset': [-128]})
        bad = self.path / 'corrupt.aimet.encodings'
        bad.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'control'):
            load_encodings_to_sim(sim, str(bad), strict=False)
        payload['activation_encodings']['clip'] = {'input': {'1': {
            'bitwidth': 8, 'dtype': 'INT8', 'is_symmetric': 'True', 'enc_type': 'PER_TENSOR',
            'scale': .125, 'zero_point': 0, 'n': 3, 'real_min': -16., 'real_max': 15.875}}}
        artifacts['encodings'].write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
                config_file=str(self.config), quant_scheme='tf')

    def test_shared_tensor_encoding_only_maps_to_its_data_consumer(self):
        model, feed = self.model('input', shared=True)
        sim = self.sim(model, feed)
        self.isolate_bounds(sim)
        sim.compute_encodings([feed])
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        payload, _ = CompilerEncodingConverter(sim, model).convert()
        self.assertIn('0', payload['activation_encodings']['data']['input'])
        self.assertFalse(set(payload['activation_encodings'].get('clip', {}).get('input', {})) & {'1', '2'})

    def test_json1_clip_settings_do_not_change_shared_data_grid(self):
        config = json.loads(self.config.read_text())
        config['op_type']['Clip'] = {'is_input_quantized': 'True', 'is_symmetric': 'False'}
        self.config.write_text(json.dumps(config))
        model, feed = self.model('input', shared=True)
        sim = self.sim(model, feed)
        self.assertTrue(sim.qc_quantize_op_dict['lo'].enabled)
        self.assertTrue(sim.qc_quantize_op_dict['lo'].use_symmetric_encodings)

    def test_optional_min_and_shared_min_max_preserve_values(self):
        for optional_min, same_bound in ((True, False), (False, True)):
            with self.subTest(optional_min=optional_min, same_bound=same_bound):
                model, feed = self.model('input', min_input=not optional_min)
                feed['x'] = np.array([1e-12, 2e-12, 8e-12], dtype=np.float32)
                clip = next(node for node in model.graph.node if node.name == 'clip')
                if same_bound:
                    clip.input[2] = 'lo'
                sim = self.sim(model, feed)
                self.isolate_bounds(sim)
                # hi is unused in the equal-bounds case, so it has no control role.
                if same_bound:
                    sim.qc_quantize_op_dict['hi'].enabled = False
                sim.compute_encodings([feed])
                wanted = ort.InferenceSession(model.SerializeToString()).run(None, feed)
                np.testing.assert_array_equal(sim.session.run(None, feed)[0], wanted[0])

    def test_same_tensor_data_and_control_ports_are_distinct(self):
        model = helper.make_model(helper.make_graph([
            helper.make_node('Clip', ['lo', 'lo', 'hi'], ['y'], name='clip')], 'shared-ports',
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, []) for name in ('lo', 'hi')],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [])]),
            opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        feed = {'lo': np.array(1e-12, np.float32), 'hi': np.array(3e-12, np.float32)}
        sim = self.sim(model, feed)
        sim.qc_quantize_op_dict['y'].enabled = False
        sim.compute_encodings([feed])
        node = next(node for node in sim.model.model.graph.node if node.name == 'clip')
        self.assertEqual(list(node.input), ['lo_updated', 'lo', 'hi'])
        self.assertTrue(sim.qc_quantize_op_dict['lo'].enabled)
        self.assertEqual(float(sim.session.run(None, feed)[0]), float(feed['lo']))

    def test_omitted_min_does_not_shift_shared_max_configuration(self):
        model, feed = self.model('input', min_input=False)
        model.graph.node.append(helper.make_node('Neg', ['hi'], ['data'], name='data'))
        model.graph.output.append(helper.make_tensor_value_info('data', TensorProto.FLOAT, []))
        sim = self.sim(model, feed)
        config = self.path / 'optional-mixed.json'
        config.write_text(json.dumps({'layer_name_config': {
            'clip': {'input_bitwidths': [8, 4, 4]},
            'data': {'input_bitwidth': 16, 'output_bitwidth': 16}}}))
        apply_onnx_mixed_precision_bitwidth(sim, str(config), verbose=False)
        self.assertEqual(sim.qc_quantize_op_dict['hi'].bitwidth, 16)
        op = sim.connected_graph.get_all_ops()['clip']
        self.assertIsNone(input_quantizer(sim, op, 1))
        self.assertIsNone(input_quantizer(sim, op, 2))
        self.assertEqual(sim.get_op_quantizers(op)[0], [sim.qc_quantize_op_dict['x']])

    def test_manual_grid_propagation_does_not_reinsert_control_quantizers(self):
        model, feed = self.model('input', transparent=True)
        model.graph.node.append(helper.make_node('Neg', ['lo'], ['data'], name='data'))
        model.graph.output.append(helper.make_tensor_value_info('data', TensorProto.FLOAT, []))
        sim = self.sim(model, feed)
        self.isolate_bounds(sim)
        sim.compute_encodings([feed])
        sim._insert_data_movement_op_output_quantizers()
        q = sim.qc_quantize_op_dict.get('identity_bound')
        self.assertTrue(q is None or not q.enabled)


if __name__ == "__main__":
    unittest.main()
