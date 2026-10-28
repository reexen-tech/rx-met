"""Typed calibration feeds preserve values, dimensions and model dtypes."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.calibration import _validate_feed, iter_calibration_feeds



class CalibrationDtypeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='onnx-calibration-dtypes-')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_multi_input_npz_preserves_indices_masks_and_half_precision(self):
        ids = helper.make_tensor_value_info('ids', TensorProto.INT64, [2])
        mask = helper.make_tensor_value_info('mask', TensorProto.BOOL, [2, 2])
        values = helper.make_tensor_value_info('values', TensorProto.FLOAT16, [2, 2])
        weight = numpy_helper.from_array(np.arange(8, dtype=np.float16).reshape(4, 2), 'table')
        nodes = [helper.make_node('Gather', ['table', 'ids'], ['selected'], axis=0),
                 helper.make_node('Where', ['mask', 'values', 'selected'], ['y'])]
        graph = helper.make_graph(nodes, 'typed-inputs', [ids, mask, values],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2])], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        model_path = self.path / 'model.onnx'
        onnx.save(model, model_path)
        calib = self.path / 'calib'
        calib.mkdir()
        feed = {'ids': np.array([1, 3], dtype=np.int64),
                'mask': np.array([[True, False], [False, True]]),
                'values': np.full((2, 2), 10, dtype=np.float16)}
        np.savez(calib / 'sample.npz', **feed)
        _, _, feeds = pipeline.load_onnx_ptq_inputs(model_path, calib)
        for name in feed:
            self.assertEqual(feeds[0][name].dtype, feed[name].dtype)
        actual = pipeline.run_onnx_feed(model_path, feeds[0])[0]
        np.testing.assert_array_equal(actual, np.array([[10, 3], [6, 10]], dtype=np.float16))

    def test_float_indices_are_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(TypeError, 'float32.*int64'):
            _validate_feed({'ids': np.array([1.5], dtype=np.float32)}, {'ids': [1]},
                           'sample', {'ids': np.dtype('int64')})

    def test_integer_narrowing_checks_range(self):
        with self.assertRaisesRegex(ValueError, 'exceed model dtype'):
            _validate_feed({'ids': np.array([2 ** 40], dtype=np.int64)}, {'ids': [1]},
                           'sample', {'ids': np.dtype('int32')})

    def test_float_casting_checks_overflow(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            _validate_feed({'x': np.array([1e20], dtype=np.float32)}, {'x': [1]},
                           'sample', {'x': np.dtype('float16')})

    def test_float64_images_are_converted_to_declared_float32(self):
        feed = _validate_feed({'x': np.array([.25], dtype=np.float64)}, {'x': [1]},
                              'sample', {'x': np.dtype('float32')})
        self.assertEqual(feed['x'].dtype, np.float32)

    def test_shape_only_legacy_loader_preserves_dtype(self):
        np.save(self.path / 'sample.npy', np.array([2 ** 54 + 7], dtype=np.int64))
        feed = next(iter_calibration_feeds(self.path, {'ids': [1]}))
        self.assertEqual(feed['ids'].dtype, np.int64)
        self.assertEqual(feed['ids'][0], 2 ** 54 + 7)

    def test_scalar_input_remains_scalar(self):
        feed = _validate_feed({'flag': np.array(True)}, {'flag': []}, 'sample',
                              {'flag': np.dtype('bool')})
        self.assertEqual(feed['flag'].shape, ())


if __name__ == "__main__":
    unittest.main()
