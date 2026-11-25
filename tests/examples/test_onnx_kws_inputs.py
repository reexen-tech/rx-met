"""The ONNX example prepares its own deterministic MFCC data from audio."""

import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper
from scipy.io import wavfile
import torch

from examples.common import kws_data
from examples import quick_start_onnx_ptq as example
from aimet_onnx.rx_ptq import load_onnx_ptq_inputs, load_onnx_validation_inputs


class OnnxKwsInputsTest(unittest.TestCase):
    def test_mfcc_runs_without_kws_model_dependencies(self):
        command = """
import sys
sys.modules.update({name: None for name in (
    'aimet_torch', 'quant_gru', 'quant_lstm', 'examples.common.kws_model',
)})
import torch
from examples.common import kws_data
with torch.no_grad():
    features = kws_data.make_speech_features()(torch.zeros(1, kws_data.TARGET_LEN))
assert features.shape == (1, *kws_data.FEATURE_SHAPE)
assert torch.isfinite(features).all()
"""
        result = subprocess.run([sys.executable, "-c", command],
                                cwd=Path(example.__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_audio_preparation_needs_only_onnx_and_speech_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "speech_commands"
            data.mkdir()
            waveform = (np.sin(np.arange(kws_data.TARGET_LEN) * .05) * 1000).astype(np.int16)
            for word in (*kws_data.WANTED_WORDS, "bird"):
                word_dir = data / word
                word_dir.mkdir()
                for split in ("train", "val"):
                    for index in range(1000):
                        path = word_dir / f"{word}_{index}.wav"
                        if kws_data.which_set(str(path), 10, 10) == split:
                            wavfile.write(path, kws_data.SAMPLE_RATE, waveform)
                            break
                    else:
                        self.fail(f"No fixture filename for {split}")
            model_path = root / "model.onnx"
            shape = [1, 49, 20]
            graph = helper.make_graph(
                [helper.make_node("Identity", ["input"], ["output"])], "mfcc",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
            )
            onnx.save(helper.make_model(graph), model_path)
            prepared = []
            for name in ("first", "second"):
                with patch.dict(sys.modules, {"examples.quick_start_kws": None}):
                    calibration, validation = example.prepare_kws_inputs(model_path, data, root / name, 1)
                model, _, feeds = load_onnx_ptq_inputs(model_path, calibration)
                val_feeds, labels = load_onnx_validation_inputs(model, validation)
                self.assertEqual(len(feeds), 12)
                self.assertEqual(len(val_feeds), 12)
                self.assertEqual(sorted(labels), list(range(12)))
                manifest = json.loads((calibration.parent / "manifest.json").read_text())
                self.assertFalse(manifest["augmentation"])
                self.assertFalse(set(manifest["calib_sources"]) & set(manifest["val_sources"]))
                for feed in (*feeds, *val_feeds):
                    self.assertEqual(feed["input"].shape, (1, 49, 20))
                    self.assertEqual(feed["input"].dtype, np.float32)
                    self.assertTrue(np.isfinite(feed["input"]).all())
                silence_index = labels.index(0)
                with torch.no_grad():
                    expected_silence = kws_data.make_speech_features()(torch.zeros(1, kws_data.TARGET_LEN)).numpy()
                np.testing.assert_allclose(val_feeds[silence_index]["input"], expected_silence, rtol=1e-6, atol=1e-6)
                prepared.append(feeds + val_feeds)
            for first, second in zip(*prepared):
                np.testing.assert_array_equal(first["input"], second["input"])
            self.assertFalse(list(root.rglob("*.pth")))

    def test_other_model_shapes_require_preprocessed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16000])
            graph = helper.make_graph([], "waveform", [value], [value])
            model_path = root / "waveform.onnx"
            onnx.save(helper.make_model(graph), model_path)
            with self.assertRaisesRegex(ValueError, "--calib-dir"):
                example.prepare_kws_inputs(model_path, root / "missing", root / "output", 1)
            self.assertFalse((root / "output").exists())
