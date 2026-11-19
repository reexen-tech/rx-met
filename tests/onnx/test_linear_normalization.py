"""Linear graph normalization preserves floating behavior before calibration."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator


MODULE_PATH = Path(__file__).resolve().parents[2] / 'src/aimet_onnx/rx_ptq/linear.py'
SPEC = importlib.util.spec_from_file_location('linear_normalization', MODULE_PATH)
linear = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linear)


def model_for(shape=(2, 3), reverse_bias=False):
    output_shape = [*shape[:-1], 4]
    weights = np.array([[.5, -.25, 1., .125], [-.5, .25, .75, -.125],
                        [.25, .5, -.25, 1.]], dtype=np.float32)
    bias = np.array([.125, -.25, .5, -.125], dtype=np.float32)
    graph = helper.make_graph([
        helper.make_node('MatMul', ['x', 'w'], ['middle'], name='fc'),
        helper.make_node('Add', ['b', 'middle'] if reverse_bias else ['middle', 'b'],
                         ['y'], name='bias_add'),
    ], 'linear', [helper.make_tensor_value_info('x', TensorProto.FLOAT, shape)],
       [helper.make_tensor_value_info('y', TensorProto.FLOAT, output_shape)],
       [numpy_helper.from_array(weights, 'w'), numpy_helper.from_array(bias, 'b')])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)


class LinearNormalizationTest(unittest.TestCase):
    def test_two_dimensional_linear_preserves_values_names_and_original_model(self):
        for reverse_bias in (False, True):
            with self.subTest(reverse_bias=reverse_bias):
                original = model_for(reverse_bias=reverse_bias)
                before = original.SerializeToString()
                normalized = linear.normalize_linear_graph(original)
                self.assertEqual(original.SerializeToString(), before)
                self.assertEqual([node.op_type for node in normalized.graph.node], ['Gemm'])
                gemm, = normalized.graph.node
                self.assertEqual(gemm.name, 'fc')
                self.assertEqual(list(gemm.input), ['x', 'w', 'b'])
                self.assertEqual(list(gemm.output), ['y'])
                self.assertEqual({attr.name: helper.get_attribute_value(attr) for attr in gemm.attribute},
                                 {'alpha': 1., 'beta': 1., 'transB': 0})
                self.assertEqual(list(original.graph.input), list(normalized.graph.input))
                self.assertEqual(list(original.graph.output), list(normalized.graph.output))
                feed = {'x': np.array([[1., -2., .5], [-1., .25, 2.]], dtype=np.float32)}
                np.testing.assert_array_equal(ReferenceEvaluator(original).run(None, feed)[0],
                                              ReferenceEvaluator(normalized).run(None, feed)[0])
                onnx.checker.check_model(normalized, full_check=True)

    def test_unsupported_linear_stays_unchanged_and_reports_export_boundary(self):
        cases = ('dynamic_weight', 'overridable_weight', 'overridable_bias', 'unknown_shape',
                 'zero_dimension', 'rank_one', 'non_fp32_input', 'non_fp32_weight',
                 'non_fp32_bias', 'custom_matmul', 'custom_add', 'side_consumer',
                 'intermediate_output', 'broadcast_bias', 'constant_node_bias', 'unnamed',
                 'legacy_broadcast')
        for case in cases:
            with self.subTest(case=case):
                model = model_for()
                if case == 'dynamic_weight':
                    del model.graph.initializer[0]
                    model.graph.input.append(helper.make_tensor_value_info('w', TensorProto.FLOAT, [3, 4]))
                elif case.startswith('overridable_'):
                    name, shape = ('w', [3, 4]) if case.endswith('weight') else ('b', [4])
                    model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
                elif case == 'unknown_shape':
                    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
                elif case == 'zero_dimension':
                    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 0
                elif case == 'rank_one':
                    model.graph.input[0].CopyFrom(helper.make_tensor_value_info('x', TensorProto.FLOAT, [3]))
                elif case == 'non_fp32_input':
                    model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
                elif case.startswith('non_fp32_'):
                    index = 0 if case.endswith('weight') else 1
                    tensor = model.graph.initializer[index]
                    tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float64), tensor.name))
                elif case.startswith('custom_'):
                    model.graph.node[0 if case.endswith('matmul') else 1].domain = 'custom.test'
                    model.opset_import.append(helper.make_opsetid('custom.test', 1))
                elif case == 'side_consumer':
                    model.graph.node.append(helper.make_node('Identity', ['middle'], ['side'], name='side'))
                elif case == 'intermediate_output':
                    model.graph.output.append(helper.make_tensor_value_info('middle', TensorProto.FLOAT, [2, 4]))
                elif case == 'broadcast_bias':
                    model.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.zeros((1, 4), np.float32), 'b'))
                elif case == 'constant_node_bias':
                    tensor = model.graph.initializer.pop()
                    model.graph.node.insert(0, helper.make_node('Constant', [], ['b'], name='bias', value=tensor))
                elif case == 'unnamed':
                    model.graph.node[0].name = ''
                elif case == 'legacy_broadcast':
                    model.opset_import[0].version = 6
                    model.graph.node[1].attribute.extend([helper.make_attribute('broadcast', 1),
                                                         helper.make_attribute('axis', 0)])
                before = model.SerializeToString()
                normalized = linear.normalize_linear_graph(model)
                self.assertEqual(normalized.SerializeToString(), before)
                with self.assertRaisesRegex(ValueError, 'MatMul.*Add.*Linear boundary'):
                    linear.validate_legacy_linear_boundaries(normalized)

    def test_tampered_alias_metadata_is_rejected(self):
        original = linear.normalize_linear_graph(model_for())
        for case in ('missing_field', 'wrong_node', 'duplicate', 'wrong_schema'):
            with self.subTest(case=case):
                model = onnx.load_model_from_string(original.SerializeToString())
                entry = next(value for value in model.metadata_props if value.key == 'rx_met.linear_normalization')
                payload = json.loads(entry.value)
                if case == 'missing_field':
                    del payload['groups'][0]['weight']
                elif case == 'wrong_node':
                    payload['groups'][0]['name'] = 'missing'
                elif case == 'duplicate':
                    payload['groups'].append(payload['groups'][0])
                else:
                    payload['schema_version'] = 2
                entry.value = json.dumps(payload)
                with self.assertRaisesRegex(ValueError, '[Ll]inear'):
                    linear.linear_groups(model)

    def test_persisted_aliases_reject_changed_gemm_parameters_and_reshape_targets(self):
        cases = ('alpha', 'beta', 'transA', 'transB', 'weight_shape', 'bias_shape',
                 'weight_dtype', 'bias_dtype', 'overridable_weight', 'input_shape_value',
                 'output_shape_value', 'shape_dtype', 'dynamic_shape')
        for case in cases:
            with self.subTest(case=case):
                model = linear.normalize_linear_graph(model_for((2, 5, 3)))
                group = linear.linear_groups(model)['fc']
                nodes = {node.name: node for node in model.graph.node}
                constants = {tensor.name: tensor for tensor in model.graph.initializer}
                if case in ('alpha', 'beta', 'transA', 'transB'):
                    gemm = nodes['fc']
                    retained = [attribute for attribute in gemm.attribute if attribute.name != case]
                    gemm.ClearField('attribute')
                    gemm.attribute.extend(retained)
                    gemm.attribute.append(helper.make_attribute(case, 2. if case in ('alpha', 'beta') else 1))
                elif case.endswith('_shape') and case != 'dynamic_shape':
                    name = 'w' if case.startswith('weight') else 'b'
                    shape = (4, 3) if name == 'w' else (1, 4)
                    constants[name].CopyFrom(numpy_helper.from_array(np.ones(shape, np.float32), name))
                elif case in ('weight_dtype', 'bias_dtype'):
                    name = 'w' if case.startswith('weight') else 'b'
                    constants[name].CopyFrom(numpy_helper.from_array(
                        numpy_helper.to_array(constants[name]).astype(np.float64), name))
                elif case == 'overridable_weight':
                    model.graph.input.append(helper.make_tensor_value_info('w', TensorProto.FLOAT, [3, 4]))
                else:
                    reshape = nodes[group['output_reshape'] if case == 'output_shape_value'
                                    else group['input_reshape']]
                    target = reshape.input[1]
                    if case == 'dynamic_shape':
                        model.graph.initializer.remove(constants[target])
                        model.graph.input.append(helper.make_tensor_value_info(target, TensorProto.INT64, [2]))
                    else:
                        values = (np.array([2, 1, 20], np.int64) if case == 'output_shape_value'
                                  else np.array([5, 6], np.int64))
                        if case == 'shape_dtype':
                            values = np.array([10, 3], np.int32)
                        constants[target].CopyFrom(numpy_helper.from_array(values, target))
                with self.assertRaisesRegex(ValueError, '[Ll]inear'):
                    linear.linear_groups(model)

    def test_generated_names_avoid_existing_graph_objects_and_preserve_metadata(self):
        original = model_for((2, 5, 3))
        original.graph.initializer.append(numpy_helper.from_array(np.array(1, np.int64), 'fc.linear_input'))
        original.graph.node.append(helper.make_node('Identity', ['x'], ['unused'],
                                                    name='fc.linear_output_reshape'))
        helper.set_model_props(original, {'owner': 'normalization-test'})
        normalized = linear.normalize_linear_graph(original)
        group = linear.linear_groups(normalized)['fc']
        self.assertNotEqual(group['gemm_input'], 'fc.linear_input')
        self.assertNotEqual(group['output_reshape'], 'fc.linear_output_reshape')
        self.assertIn(('owner', 'normalization-test'),
                      [(entry.key, entry.value) for entry in normalized.metadata_props])
        linear.validate_legacy_linear_boundaries(normalized)
        onnx.checker.check_model(normalized, full_check=True)

    def test_high_rank_linear_preserves_shape_and_persisted_aliases(self):
        for shape in ((2, 5, 3), (2, 1, 5, 3)):
            with self.subTest(shape=shape):
                original = model_for(shape)
                normalized = linear.normalize_linear_graph(original)
                self.assertEqual([node.op_type for node in normalized.graph.node],
                                 ['Reshape', 'Gemm', 'Reshape'])
                group, = linear.linear_groups(normalized).values()
                self.assertEqual(group['name'], 'fc')
                self.assertEqual(group['matmul'], 'fc')
                self.assertEqual(group['add'], 'bias_add')
                self.assertEqual(group['input'], 'x')
                self.assertEqual(group['output'], 'y')
                self.assertEqual(group['input_shape'], list(shape))
                self.assertEqual(group['output_shape'], [*shape[:-1], 4])
                first, gemm, last = normalized.graph.node
                self.assertEqual(group['gemm_input'], first.output[0])
                self.assertEqual(group['gemm_output'], gemm.output[0])
                self.assertEqual(group['input_reshape'], first.name)
                self.assertEqual(group['output_reshape'], last.name)
                self.assertEqual(group['generated_nodes'], [first.name, last.name])
                self.assertEqual(list(last.output), ['y'])
                feed = {'x': np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 8}
                np.testing.assert_array_equal(ReferenceEvaluator(original).run(None, feed)[0],
                                              ReferenceEvaluator(normalized).run(None, feed)[0])
                restored = onnx.load_model_from_string(normalized.SerializeToString())
                self.assertEqual(linear.linear_groups(restored), linear.linear_groups(normalized))
                self.assertEqual(linear.normalize_linear_graph(restored).SerializeToString(),
                                 normalized.SerializeToString())
                onnx.checker.check_model(normalized, full_check=True)


if __name__ == '__main__':
    unittest.main()
