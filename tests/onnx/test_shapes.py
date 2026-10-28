"""Persisted ONNX shape metadata is complete without changing computations."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from aimet_common.onnx_shapes import complete_onnx_shapes, save_onnx_with_shapes, validate_onnx_shapes
from aimet_onnx.rx_ptq import pipeline

from tests.support import onnx as fixtures


def simple_graph(batch=1):
    weight = numpy_helper.from_array(np.array([[.5, -.25], [-.5, .25]], dtype=np.float32), 'w')
    graph = helper.make_graph([
        helper.make_node('MatMul', ['x', 'w'], ['middle'], name='matmul'),
        helper.make_node('Relu', ['middle'], ['y'], name='relu'),
    ], 'shapes', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [batch, 2])],
       [helper.make_tensor_value_info('y', TensorProto.FLOAT, [batch, 2])], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)], ir_version=8)


class OnnxShapeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='onnx-shapes-')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_inference_changes_only_metadata_and_preserves_original(self):
        model = simple_graph()
        before = model.SerializeToString()
        result = complete_onnx_shapes(model)
        self.assertEqual(model.SerializeToString(), before)
        self.assertEqual([n.SerializeToString() for n in result.graph.node],
                         [n.SerializeToString() for n in model.graph.node])
        self.assertEqual([v.SerializeToString() for v in result.graph.initializer],
                         [v.SerializeToString() for v in model.graph.initializer])
        self.assertEqual([v.SerializeToString() for v in result.graph.input],
                         [v.SerializeToString() for v in model.graph.input])
        self.assertEqual([v.SerializeToString() for v in result.graph.output],
                         [v.SerializeToString() for v in model.graph.output])
        middle = next(value for value in result.graph.value_info if value.name == 'middle')
        self.assertEqual([dim.dim_value for dim in middle.type.tensor_type.shape.dim], [1, 2])
        self.assertEqual(middle.type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(validate_onnx_shapes(result)['intermediate_tensors'], 1)

    def test_shape_expression_before_reshape_is_resolved_without_folding_nodes(self):
        minus_one = numpy_helper.from_array(np.array([-1], dtype=np.int64), 'minus_one')
        index = numpy_helper.from_array(np.array(0, dtype=np.int64))
        nodes = [
            helper.make_node('Shape', ['x'], ['shape'], name='shape'),
            helper.make_node('Constant', [], ['index'], name='index', value=index),
            helper.make_node('Gather', ['shape', 'index'], ['batch'], name='batch', axis=0),
            helper.make_node('Unsqueeze', ['batch'], ['batch_vector'], name='unsqueeze', axes=[0]),
            helper.make_node('Concat', ['batch_vector', 'minus_one'], ['target'], name='target', axis=0),
            helper.make_node('Reshape', ['x', 'target'], ['flat'], name='reshape'),
            helper.make_node('Identity', ['flat'], ['y'], name='output'),
        ]
        graph = helper.make_graph(nodes, 'reshape-shapes',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 3, 4])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 24])], [minus_one])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)], ir_version=8)
        result = complete_onnx_shapes(model)
        flat = next(value for value in result.graph.value_info if value.name == 'flat')
        self.assertEqual([dim.dim_value for dim in flat.type.tensor_type.shape.dim], [1, 24])
        self.assertEqual([node.SerializeToString() for node in result.graph.node],
                         [node.SerializeToString() for node in model.graph.node])
        validate_onnx_shapes(result)

    def test_dynamic_dimensions_remain_symbolic(self):
        result = complete_onnx_shapes(simple_graph('batch'))
        middle = next(value for value in result.graph.value_info if value.name == 'middle')
        self.assertEqual(middle.type.tensor_type.shape.dim[0].dim_param, 'batch')
        self.assertGreater(validate_onnx_shapes(result)['symbolic_dimensions'], 0)

    def test_scalar_and_zero_length_tensors_have_explicit_shapes(self):
        zero = numpy_helper.from_array(np.array(0., dtype=np.float32))
        empty = numpy_helper.from_array(np.empty((0,), dtype=np.float32))
        graph = helper.make_graph([
            helper.make_node('Constant', [], ['zero'], name='scalar', value=zero),
            helper.make_node('Constant', [], ['empty'], name='empty', value=empty),
            helper.make_node('Add', ['x', 'zero'], ['y'], name='add'),
        ], 'scalars', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
           [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)], ir_version=8)
        result = complete_onnx_shapes(model)
        infos = {value.name: value for value in result.graph.value_info}
        self.assertTrue(infos['zero'].type.tensor_type.HasField('shape'))
        self.assertEqual(len(infos['zero'].type.tensor_type.shape.dim), 0)
        self.assertEqual(infos['empty'].type.tensor_type.shape.dim[0].dim_value, 0)
        validate_onnx_shapes(result)

    def test_missing_shape_dtype_or_dimension_is_rejected_without_repair(self):
        original = complete_onnx_shapes(simple_graph())
        for kind in ('entry', 'shape', 'dtype', 'dimension'):
            with self.subTest(kind=kind):
                model = copy.deepcopy(original)
                middle = next(value for value in model.graph.value_info if value.name == 'middle')
                if kind == 'entry':
                    model.graph.ClearField('value_info')
                elif kind == 'shape':
                    middle.type.tensor_type.ClearField('shape')
                elif kind == 'dtype':
                    middle.type.tensor_type.elem_type = TensorProto.UNDEFINED
                else:
                    middle.type.tensor_type.shape.dim[0].Clear()
                before = model.SerializeToString()
                with self.assertRaisesRegex(RuntimeError, 'middle.*(shape|dtype)'):
                    validate_onnx_shapes(model)
                self.assertEqual(before, model.SerializeToString())

    def test_control_flow_subgraph_shapes_are_checked(self):
        def branch(prefix):
            return helper.make_graph([
                helper.make_node('Identity', ['x'], [prefix + '_middle']),
                helper.make_node('Relu', [prefix + '_middle'], [prefix + '_output']),
            ], prefix, [], [helper.make_tensor_value_info(prefix + '_output', TensorProto.FLOAT, [1, 2])])
        graph = helper.make_graph([
            helper.make_node('If', ['condition'], ['middle'], name='choose',
                             then_branch=branch('yes'), else_branch=branch('no')),
            helper.make_node('Identity', ['middle'], ['y'], name='output'),
        ], 'branches', [helper.make_tensor_value_info('condition', TensorProto.BOOL, []),
                        helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
           [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        result = complete_onnx_shapes(model)
        validate_onnx_shapes(result)
        then_branch = next(attribute.g for attribute in result.graph.node[0].attribute
                           if attribute.name == 'then_branch')
        then_branch.ClearField('value_info')
        with self.assertRaisesRegex(RuntimeError, 'then_branch.*yes_middle'):
            validate_onnx_shapes(result)

    def test_external_weights_are_preserved_when_persisting_metadata(self):
        path = self.path / 'model.onnx'
        onnx.save_model(simple_graph(), str(path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location='weights.bin', size_threshold=0)
        before = onnx.load(str(path), load_external_data=False)
        weights = (self.path / 'weights.bin').read_bytes()
        report = save_onnx_with_shapes(path)
        after = onnx.load(str(path), load_external_data=False)
        self.assertEqual(report['intermediate_tensors'], 1)
        self.assertEqual([value.SerializeToString() for value in before.graph.initializer],
                         [value.SerializeToString() for value in after.graph.initializer])
        self.assertEqual(weights, (self.path / 'weights.bin').read_bytes())
        validate_onnx_shapes(after)
        onnx.checker.check_model(str(path), full_check=True)

    def test_unsupported_inference_fails_without_inventing_dimensions(self):
        model = simple_graph()
        model.graph.node[0].op_type = 'Opaque'
        model.graph.node[0].domain = 'unsupported.test'
        model.opset_import.append(helper.make_opsetid('unsupported.test', 1))
        before = model.SerializeToString()
        # ORT writes a diagnostic model when symbolic inference is incomplete.
        directory = Path.cwd()
        try:
            os.chdir(self.path)
            with self.assertRaisesRegex(RuntimeError, 'middle.*shape'):
                complete_onnx_shapes(model)
        finally:
            os.chdir(directory)
        self.assertEqual(before, model.SerializeToString())


class OnnxShapeArtifactTest(unittest.TestCase):
    def test_export_persists_shapes_and_reload_rejects_deleted_annotations(self):
        fixture = fixtures.OnnxFixture(self)
        sim, model = fixture.sim(second_layer=True)
        fixture.calibrate(sim)
        pipeline.apply_power_of_2_workflow(sim, verbose=False)
        expected = pipeline.run_onnx_feed(sim, fixture.feed)
        artifacts = pipeline.export_onnx_json(sim, model, fixture.path / 'export', 'model')
        stored = onnx.load(artifacts['onnx'])
        self.assertEqual(artifacts['onnx_shapes']['intermediate_tensors'], 1)
        validate_onnx_shapes(stored)
        fixture.verify(artifacts, expected)
        stored.graph.ClearField('value_info')
        onnx.save(stored, artifacts['onnx'])
        with self.assertRaisesRegex(RuntimeError, 'mid.*shape'):
            fixture.verify(artifacts, expected)


if __name__ == "__main__":
    unittest.main()
