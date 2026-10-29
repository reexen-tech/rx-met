"""Small reference for the waveform/Linear/Softmax KWS expression."""

import torch
from torch import nn

from examples import quick_start_kws as kws


class ReferenceConvActBN(kws.ConvActBN):
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))


class ReferenceAttMHRNN(kws.AttMHRNN):
    """Reuse recurrent shapes while retaining the previous arithmetic boundary."""

    def __init__(self, **options):
        super().__init__(**options)
        self.speech_features = kws.make_speech_features()
        for index, block in enumerate(self.convs):
            reference = ReferenceConvActBN(block.conv.in_channels, block.conv.out_channels,
                                           block.conv.kernel_size)
            reference.load_state_dict(block.state_dict())
            self.convs[index] = reference
        for index, projection in enumerate(self.query_projs):
            self.query_projs[index] = nn.Linear(projection.conv.in_channels,
                                                projection.conv.out_channels)
        for index, layer in enumerate(self.mlp):
            if isinstance(layer, kws.LinearAsConv2d):
                self.mlp[index] = nn.Linear(layer.conv.in_channels, layer.conv.out_channels)
        self.fc = nn.Linear(self.fc.conv.in_channels, self.fc.conv.out_channels)
        del self.attention

    def forward(self, waveform):
        x = self.convs(self.speech_features(waveform).unsqueeze(1))
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.size(0), x.size(1), -1)
        recurrent_layers = self.grus if self.rnn_type == "gru" else self.lstms
        for recurrent in recurrent_layers:
            x, _ = recurrent(x)
        mid = x[:, self.mid_index, :].contiguous()
        heads = []
        for projection in self.query_projs:
            query = projection(mid)
            logits = torch.matmul(x, query.unsqueeze(-1)).squeeze(-1)
            weights = torch.softmax(logits, dim=-1)
            heads.append(torch.matmul(weights.unsqueeze(1), x).squeeze(1))
        return self.fc(self.mlp(self.dropout(torch.cat(heads, dim=-1))))


def save_reference_checkpoint(model, path, rnn_type):
    torch.save({"rnn_type": rnn_type, "model": model.state_dict()}, path)
