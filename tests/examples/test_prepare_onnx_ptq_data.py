"""Validate and prepare real ONNX/parquet fixtures without a GPU or downloads."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from tests.support.paths import REPO_ROOT


ENTRY = REPO_ROOT / 'examples/prepare_onnx_ptq_data.py'
SPEC = importlib.util.spec_from_file_location('prepare_onnx_ptq_data', ENTRY)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareOnnxPtqDataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='onnx ptq inputs ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'original.onnx'
        self.parquet = self.root / 'images.parquet'
        self.output = self.root / 'prepared'
        self.make_model()
        buffer = io.BytesIO()
        Image.new('RGB', (32, 40), (64, 128, 192)).save(buffer, format='PNG')
        self.image = buffer.getvalue()
        self.make_parquet([self.image] * 6)

    def make_model(self, channels=3):
        shape = ['batch', channels, 224, 224]
        graph = helper.make_graph(
            [helper.make_node('Identity', ['image'], ['output'])], 'fixture',
            [helper.make_tensor_value_info('image', TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        onnx.save(model, self.source)

    def make_parquet(self, images):
        table = pa.table({
            'image': pa.array([{'bytes': image} for image in images],
                              type=pa.struct([('bytes', pa.binary())])),
            'label': pa.array([index % 2 for index in range(len(images))], type=pa.int64()),
        })
        pq.write_table(table, self.parquet)

    def run_prepare(self, calib=2, val=2, default_root=False):
        args = ['--onnx', str(self.source), '--parquet', str(self.parquet),
                '--calib-count', str(calib), '--val-count', str(val)]
        if not default_root:
            args.extend(['--out-root', str(self.output)])
        output = io.StringIO()
        defaults = (patch.object(prepare, 'DEFAULT_OUTPUT_ROOT', self.output)
                    if default_root else contextlib.nullcontext())
        with defaults, contextlib.redirect_stdout(output):
            prepare.main(args)
        return output.getvalue()

    def test_default_layout_and_preprocessed_inputs(self):
        before = {path: hashlib.sha256(path.read_bytes()).digest()
                  for path in (self.source, self.parquet)}
        self.run_prepare(default_root=True)
        model_path = self.output / 'models/mobilenetv2/mobilenetv2-12.onnx'
        data = self.output / 'datasets/imagenette2-320'
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        for value in list(model.graph.input) + list(model.graph.output):
            self.assertEqual(value.type.tensor_type.shape.dim[0].dim_value, 1)
        manifest = json.loads((data / 'manifest.json').read_text())
        self.assertEqual(manifest['onnx'], str(model_path))
        self.assertEqual(manifest['input_name'], 'image')
        self.assertEqual(manifest['input_shape'], [1, 3, 224, 224])
        self.assertFalse(set(manifest['calib_indices']) & set(manifest['val_indices']))
        self.assertEqual(manifest['validation_label_space'], 'imagenet-1k')
        self.assertEqual(manifest['val_labels'],
            [prepare.IMAGENETTE_TO_IMAGENET[index % 2] for index in manifest['val_indices']])
        for split in ('calib', 'val'):
            files = sorted((data / split).glob('*.npy'))
            self.assertEqual(len(files), 2)
            value = np.load(files[0])
            self.assertEqual(value.shape, (1, 3, 224, 224))
            self.assertEqual(value.dtype, np.float32)
            expected = ((np.array([64, 128, 192]) / 255 - [0.485, 0.456, 0.406])
                        / [0.229, 0.224, 0.225])
            np.testing.assert_allclose(value[0, :, 100, 100], expected, atol=1e-6)
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_default_output_follows_working_directory(self):
        before = Path.cwd()
        try:
            os.chdir(self.root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                prepare.main(['--onnx', str(self.source), '--parquet', str(self.parquet),
                              '--calib-count', '2', '--val-count', '2'])
        finally:
            os.chdir(before)
        prepared = self.root / 'output/onnx_ptq_quick_start'
        self.assertTrue((prepared / 'models/mobilenetv2/mobilenetv2-12.onnx').is_file())
        self.assertTrue((prepared / 'datasets/imagenette2-320/manifest.json').is_file())
        self.assertIn('quick_start_onnx_ptq.py', output.getvalue())
        self.assertNotIn('RX_MET_ONNX_PTQ_MODEL=', output.getvalue())
        self.assertNotIn('RX_MET_ONNX_PTQ_CALIB=', output.getvalue())
        self.assertNotIn('量化容器挂载', output.getvalue())

    def test_custom_output_prints_input_overrides(self):
        output = self.run_prepare()
        self.assertIn('RX_MET_ONNX_PTQ_MODEL=', output)
        self.assertIn('RX_MET_ONNX_PTQ_CALIB=', output)
        self.assertIn(str(self.output / 'models/mobilenetv2/mobilenetv2-12.onnx'), output)
        self.assertIn(str(self.output / 'datasets/imagenette2-320/calib'), output)

    def test_missing_files_leave_no_outputs(self):
        for path in (self.source, self.parquet):
            with self.subTest(path=path):
                saved = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(SystemExit, '不存在'):
                    self.run_prepare()
                self.assertFalse(self.output.exists())
                path.write_bytes(saved)

    def test_invalid_model_is_rejected_before_writing(self):
        self.source.write_bytes(b'not an ONNX model')
        with self.assertRaisesRegex(SystemExit, 'ONNX 模型不可用'):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_incompatible_input_shape_is_rejected(self):
        self.make_model(channels=1)
        with self.assertRaisesRegex(SystemExit, 'float32 NCHW'):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_parquet_and_schema_leave_no_outputs(self):
        cases = (
            None,
            pa.table({'image': [b'not a struct'], 'label': [0]}),
            pa.table({'image': [{'bytes': b'image'}], 'label': ['not an integer']}),
        )
        for table in cases:
            with self.subTest(table=table):
                if table is None:
                    self.parquet.write_bytes(b'not parquet')
                else:
                    pq.write_table(table, self.parquet)
                with self.assertRaisesRegex(SystemExit, 'parquet'):
                    self.run_prepare()
                self.assertFalse(self.output.exists())

    def test_bad_selected_image_leaves_no_outputs(self):
        self.make_parquet([b'not an image'] * 6)
        with self.assertRaisesRegex(SystemExit, '图片不可用'):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_insufficient_samples_leave_no_outputs(self):
        with self.assertRaisesRegex(SystemExit, '样本不够'):
            self.run_prepare(calib=4, val=4)
        self.assertFalse(self.output.exists())

    def test_rerun_replaces_generated_samples_without_stale_files(self):
        self.run_prepare()
        self.run_prepare(calib=1, val=1)
        data = self.output / 'datasets/imagenette2-320'
        for split in ('calib', 'val'):
            files = list((data / split).glob('*.npy'))
            self.assertEqual([path.name for path in files], [f'{split}_0000.npy'])
        manifest = json.loads((data / 'manifest.json').read_text())
        self.assertEqual(len(manifest['calib_indices']), 1)
        self.assertEqual(len(manifest['val_indices']), 1)

    def test_unmanaged_output_is_preserved(self):
        directory = self.output / 'datasets/imagenette2-320/calib'
        directory.mkdir(parents=True)
        existing = directory / 'my_sample.npy'
        existing.write_bytes(b'keep')
        with self.assertRaisesRegex(SystemExit, '非本脚本生成'):
            self.run_prepare()
        self.assertEqual(existing.read_bytes(), b'keep')
        self.assertFalse((self.output / 'models').exists())

    def test_source_model_cannot_be_overwritten(self):
        destination = self.output / 'models/mobilenetv2/mobilenetv2-12.onnx'
        destination.parent.mkdir(parents=True)
        self.source.rename(destination)
        self.source = destination
        before = self.source.read_bytes()
        with self.assertRaisesRegex(SystemExit, '不能覆盖原始 ONNX'):
            self.run_prepare()
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.output / 'datasets').exists())


if __name__ == '__main__':
    unittest.main()
