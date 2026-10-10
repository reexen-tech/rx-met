"""Customer KWS workflow regression checks; run in the rx-met CUDA environment."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from examples import quick_start_kws as kws
from tests.test_quant_lstm_integration import (
    assert_public_encoding_document, assert_recurrent_encoding_schema,
)


class KwsExampleConfigurationTest(unittest.TestCase):
    def test_checkpoint_rejects_wrong_model_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            original = nn.Linear(4, 2)
            kws.save_checkpoint(original, path, 'gru')
            restored = nn.Linear(4, 2)
            kws.load_checkpoint(restored, path, 'gru', 'cpu')
            torch.testing.assert_close(restored.weight, original.weight)
            with self.assertRaisesRegex(ValueError, 'rnn_type'):
                kws.load_checkpoint(restored, path, 'lstm', 'cpu')
            with self.assertRaisesRegex(ValueError, '模型类型'):
                kws.save_checkpoint(restored, path, 'lstm')
            self.assertEqual(torch.load(path, weights_only=True)['rnn_type'], 'gru')

    def test_calibration_uses_unaugmented_training_samples(self):
        class Dataset(TensorDataset):
            def __init__(self, root, split):
                values = {'train': 1., 'val': 2., 'test': 3.}
                super().__init__(torch.full((2, 4), values[split]), torch.zeros(2, dtype=torch.long))
                self.train_mode = split == 'train'
        with patch.object(kws, 'SpeechCommandsKWS', Dataset), patch.object(kws, 'NUM_WORKERS', 0):
            loaders = kws.build_dataloaders('unused')
        self.assertTrue(loaders['train'].dataset.train_mode)
        self.assertFalse(loaders['calib'].dataset.train_mode)
        self.assertTrue(torch.all(next(iter(loaders['calib']))[0] == 1))
        self.assertTrue(torch.all(next(iter(loaders['test']))[0] == 3))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class KwsExampleWorkflowTest(unittest.TestCase):
    def test_gru_and_lstm_customer_workflow(self):
        # Run the actual entry point. Small dimensions and random waveforms keep
        # this regression independent of Speech Commands and training accuracy.
        torch.manual_seed(23)
        data = TensorDataset(torch.randn(2, kws.TARGET_LEN) * 0.02, torch.tensor([2, 3]))
        loaders = {name: DataLoader(data, batch_size=2) for name in ('train', 'val', 'test', 'calib')}
        def new_model(rnn_type):
            return kws.AttMHRNN(rnn_type=rnn_type, rnn_units=4, heads=1,
                                dense_units=(4,), dropout=0).cuda()
        with tempfile.TemporaryDirectory() as directory:
            gru_metadata = None
            for rnn_type in ('gru', 'lstm'):
                with self.subTest(rnn_type=rnn_type), \
                     patch.object(kws, 'build_dataloaders', return_value=loaders), \
                     patch.object(kws, '_new_model', side_effect=new_model), \
                     patch.dict('os.environ', {'RX_MET_KWS_FP_MODEL': str(Path(directory) / rnn_type / 'model_fp_kws.pth')}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    kws.main(['--rnn_type', rnn_type, '--output-dir', directory])
                    output = Path(directory) / rnn_type
                    checkpoint = torch.load(output / 'att_mh_rnn_kws_qat.pth', weights_only=True)
                    self.assertEqual(checkpoint['rnn_type'], rnn_type)
                    graph = onnx.load(output / 'att_mh_rnn_kws.onnx')
                    onnx.checker.check_model(graph)
                    nodes = [node for node in graph.graph.node if node.op_type in ('GRU', 'LSTM')]
                    self.assertEqual([node.op_type for node in nodes], [rnn_type.upper()] * 2)
                    encodings = json.loads((output / 'att_mh_rnn_kws.encodings').read_text())
                    assert_public_encoding_document(self, encodings)
                    metadata = (set(encodings), encodings['version'], encodings['schema_version'])
                    if rnn_type == 'gru':
                        gru_metadata = metadata
                    else:
                        self.assertEqual(metadata, gru_metadata)
                    for node in nodes:
                        self.assertTrue(encodings['activation_encodings'][node.name]['is_' + rnn_type.upper()])
                        assert_recurrent_encoding_schema(self, encodings['activation_encodings'][node.name])
                        if rnn_type == 'lstm':
                            layer = encodings['activation_encodings'][node.name]
                            self.assertEqual(layer['internal_ops']['cell_state']['output'][0]['dtype'], 'INT16')
                            for field in ('internal_ops', 'internal_ops_reverse'):
                                for item in layer[field].values():
                                    scales = torch.as_tensor(item['output'][0]['scale'])
                                    torch.testing.assert_close(scales.log2(), scales.log2().round())


if __name__ == '__main__':
    unittest.main()
