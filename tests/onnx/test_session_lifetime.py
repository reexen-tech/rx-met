"""An ORT session owns only the temporary model directory it creates."""
import gc
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper
from onnxruntime import SessionOptions

from aimet_onnx import utils


def make_session(path=None):
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], 'lifetime', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = min(onnx.IR_VERSION, 8)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    return utils.OrtInferenceSession(model, ['CPUExecutionProvider'], options, path)


class SessionLifetimeTest(unittest.TestCase):
    def test_owned_directory_is_cleaned_during_module_teardown(self):
        session = make_session()
        directory = Path(session.model_dir)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        self.assertTrue((directory / 'model.onnx').is_file())
        failures = []
        # CPython can release the last session after clearing its module globals.
        with patch.dict(vars(utils), {'shutil': None, 'logger': None}), \
                patch.object(sys, 'unraisablehook', lambda event: failures.append(str(event.exc_value))):
            del session
            gc.collect()
        self.assertEqual(failures, [])
        self.assertFalse(directory.exists())

    def test_explicit_directory_remains_available_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            session = make_session(directory)
            self.assertIsNone(session.model_dir)
            del session
            gc.collect()
            self.assertTrue((Path(directory) / 'model.onnx').is_file())
