"""Synthetic QuantLSTM PTQ/QAT/export example; no dataset is required."""
import argparse
from pathlib import Path

import torch
from torch import nn
from quant_lstm import QuantLSTM
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.rx_export.export_onnx_json import export_onnx_json


class Model(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(4, 8, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(16 if bidirectional else 8, 2)

    def forward(self, value):
        sequence, _ = self.lstm(value)
        return self.head(sequence)


def make_sim(model, dummy):
    original = model.lstm
    model.lstm = QuantLSTM(original.input_size, original.hidden_size,
                          batch_first=True, bidirectional=original.bidirectional,
                          device=dummy.device)
    model.lstm.load_state_dict(original.state_dict())
    config = Path(__file__).parent / 'config'
    sim = QuantizationSimModel(prepare_model(model), dummy, quant_scheme='tf',
        config_file=str(config / 'mrnn_quantsim_config_custom_mixed_precision_v2.json'))
    apply_mixed_precision_bitwidth(sim.model, str(config / 'lstm_quant.json'))
    return sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/lstm'))
    args = parser.parse_args()
    torch.manual_seed(17)
    dummy = torch.randn(2, 5, 4, device='cuda')
    sim = make_sim(Model(args.bidirectional).cuda(), dummy)
    with torch.no_grad(), compute_encodings(sim.model):
        sim.model(dummy)
        sim.model(dummy * 0.7)
    assert sim.model.lstm.is_calibrated()
    params = sim.model.lstm.export_quant_params()
    sim.model.train()
    optimizer = torch.optim.Adam(sim.model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    sim.model(dummy).square().mean().backward()
    optimizer.step()
    assert params == sim.model.lstm.export_quant_params()
    sim.model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = args.output / 'model.pt'
    torch.save(sim.get_original_model(sim.model, qdq_weights=False).state_dict(), weights)
    onnx_path, encodings_path = export_onnx_json(sim, str(args.output), 'model', tuple(dummy.shape))
    restored_model = Model(args.bidirectional).cuda()
    restored_model.load_state_dict(torch.load(weights, weights_only=True))
    restored = make_sim(restored_model, dummy)
    load_quantizer_encodings(restored.model, encodings_path, allow_overwrite=False)
    restored.model.eval()
    torch.testing.assert_close(restored.model(dummy), sim.model(dummy))
    print('QuantLSTM PTQ/QAT/export/reload passed:', onnx_path, encodings_path)


if __name__ == '__main__':
    main()
