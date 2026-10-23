"""Clip bounds retain floating control semantics across PTQ and artifact reload."""

import json
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
from aimet_onnx.rx_ptq.input_quantization import input_quantizer
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
import test_onnx_ptq_alignment as fixtures


class OnnxClipControlTest(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.OnnxAlignmentTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
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


if __name__ == '__main__':
    unittest.main()
