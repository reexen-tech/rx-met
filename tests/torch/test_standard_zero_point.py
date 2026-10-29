"""Real Torch export and reload preserve asymmetric zero-point grids."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tests.support.compiler_encodings import shared
from tests.support.paths import REPO_ROOT


class TorchStandardZeroPointExportTest(unittest.TestCase):
    def test_asymmetric_export_and_legacy_reload_preserve_outputs(self):
        import torch
        from torch import nn
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.v2.quantsim import QuantizationSimModel
        from aimet_torch.rx_export import export_onnx_json
        from aimet_torch.staged_quantization_utils import load_quantizer_encodings

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        config = REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'
        torch.manual_seed(42)
        model = Model().eval()
        x = torch.tensor([[-.75, -.125, .25, 1.]])
        sim = QuantizationSimModel(prepare_model(copy.deepcopy(model)), x, quant_scheme='tf', config_file=str(config))
        layer = sim.model.linear
        for q in list(layer.input_quantizers) + list(layer.output_quantizers) + list(layer.param_quantizers.values()):
            if q is None:
                continue
            q.qmin, q.qmax = 0, 255
            q.symmetric = False
            q.set_range(torch.full(q.shape, -.875), torch.full(q.shape, 31.))
            q.requires_grad_(False)
        expected = sim.model(x).detach()
        with tempfile.TemporaryDirectory() as directory:
            _, path = export_onnx_json(sim, directory, 'linear', tuple(x.shape))
            document = json.loads(Path(path).read_text())
            self.assertEqual(document['schema_version'], 4)
            self.assertEqual(document['activation_encodings']['linear']['input']['0']['zero_point'], 7)
            self.assertEqual(document['param_encodings']['linear.bias']['zero_point'], [7, 7])
            for version in (None, 1, 2, 3, 4):
                saved = copy.deepcopy(document)
                if version != 4:
                    if version is None:
                        saved.pop('schema_version')
                    else:
                        saved['schema_version'] = version
                    for _, record, _ in shared._records(saved):
                        record['zero_point'] = shared._map_values(record['zero_point'], lambda value: -value)
                target = Path(directory) / f'reload{version}.json'
                target.write_text(json.dumps(saved))
                restored = QuantizationSimModel(prepare_model(copy.deepcopy(model)), x,
                    quant_scheme='tf', config_file=str(config))
                load_quantizer_encodings(restored.model, str(target), allow_overwrite=False,
                                         skip_if_not_found=False, verbose=False)
                torch.testing.assert_close(restored.model(x), expected, rtol=0, atol=0)
                self.assertEqual(restored.model.linear.input_quantizers[0].get_offset().item(), -7)
            broken = copy.deepcopy(document)
            broken['activation_encodings']['linear']['input']['0']['zero_point'] = -7
            target = Path(directory) / 'wrong-sign.json'
            target.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'linear.*range'):
                load_quantizer_encodings(restored.model, str(target), verbose=False)


if __name__ == '__main__':
    unittest.main()
