"""KWS classifier, quantization preparation, and checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from aimet_torch import model_preparer
from aimet_torch._base.nn.modules import custom
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM

from .kws_data import N_FRAMES, N_MFCC, NUM_CLASSES, make_speech_features


# ============================================================================
# 模型：kws_streaming/models/att_mh_rnn.py，支持 GRU / LSTM 两种循环层
# ============================================================================
class ConvActBN(nn.Module):
    """Keras Conv2D(activation=relu) + BatchNormalization。padding=same。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int]):
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=False)
        # Keras BN momentum=0.99 → PyTorch momentum = 1 - 0.99
        # FX expands this BN into Sub/Div/Mul/Add so both operands of
        # each operation are calibrated, including the frozen statistics.
        self.bn = QuantizableBatchNorm2d(out_ch, momentum=0.01)
        self.flatten_bn_spatial = out_ch == 1

    def forward(self, x):
        x = self.conv(x)
        if self.flatten_bn_spatial:
            # ReLU and BN preserve the single-channel spatial flattening.
            shape = x.shape
            x = self.bn(self.act(x.flatten(2).unsqueeze(2)))
            return x.reshape(shape)
        return self.bn(self.act(x))


class LinearAsConv2d(nn.Module):
    """A Linear with unchanged [out, in] weight order and a 1x1 kernel."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1, bias=True)

    def forward(self, x):
        return self.conv(x.unsqueeze(-1).unsqueeze(-1)).flatten(1)


class AttentionHead(nn.Module):
    """Stable attention pooling without dynamic MatMul or Softmax."""

    def __init__(self, time_steps: int):
        super().__init__()
        self.time_steps = time_steps
        self.score_product = custom.Multiply()
        self.score_sum = custom.Sum()
        self.score_max = custom.AMax()
        self.center_scores = custom.Subtract()
        self.exponential = custom.Exponential()
        self.context_product = custom.Multiply()
        self.numerator = custom.Sum()
        self.denominator = custom.Sum()
        self.denominator_clamp = custom.Clamp()
        self.normalize = custom.Divide()

    def forward(self, sequence, query):
        scores = self.score_sum(self.score_product(sequence, query), dim=2, keepdim=True)
        maximum = self.score_max(scores, dim=1, keepdim=True)
        exponentials = self.exponential(self.center_scores(scores, maximum))
        numerator = self.numerator(self.context_product(sequence, exponentials), dim=1, keepdim=True)
        denominator = self.denominator(exponentials, dim=1, keepdim=True)
        # Stable softmax has one exp(0)=1 and all other terms in [0,1].
        denominator = self.denominator_clamp(denominator, min=1.0, max=float(self.time_steps))
        return self.normalize(numerator, denominator)


class AttMHRNN(nn.Module):
    """
    kws_streaming.models.att_mh_rnn 的等价拓扑。

    输入外置 MFCC [B, 49, 20]，接 Conv / 双向 QuantGRU 或 QuantLSTM / 四头注意力。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        cnn_filters: tuple[int, ...] = (10, 1),
        cnn_kernels: tuple[tuple[int, int], ...] = ((5, 1), (5, 1)),
        rnn_layers: int = 2,
        rnn_units: int = 128,
        heads: int = 4,
        dense_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        in_time: int = N_FRAMES,
        in_freq: int = N_MFCC,
        rnn_type: str = "gru",
    ):
        super().__init__()
        if rnn_type not in ("gru", "lstm"):
            raise ValueError("rnn_type 必须是 gru 或 lstm")
        self.rnn_type = rnn_type
        convs: list[nn.Module] = []
        in_ch = 1
        for filters, kernel in zip(cnn_filters, cnn_kernels):
            convs.append(ConvActBN(in_ch, filters, kernel))
            in_ch = filters
        self.convs = nn.Sequential(*convs)

        # 两种算子均返回序列输出 [B, T, 2H]，后续注意力和分类头共用。
        # 多层网络由单层实例堆叠，QuantLSTM 每个实例使用 num_layers=1。
        rnn_class = QuantGRU if rnn_type == "gru" else QuantLSTM
        rnn_input_size = in_ch * in_freq
        recurrent_layers = []
        for _ in range(rnn_layers):
            recurrent_layers.append(
                rnn_class(
                    input_size=rnn_input_size,
                    hidden_size=rnn_units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            rnn_input_size = rnn_units * 2
        # 保留 GRU 的原有层名，让现有 GRU 权重的参数名保持兼容。
        if rnn_type == "gru":
            self.grus = nn.ModuleList(recurrent_layers)
        else:
            self.lstms = nn.ModuleList(recurrent_layers)

        feature_dim = rnn_units * 2
        self.mid_index = in_time // 2
        self.query_projs = nn.ModuleList(
            [LinearAsConv2d(feature_dim, feature_dim) for _ in range(heads)]
        )
        self.attention = nn.ModuleList([AttentionHead(in_time) for _ in range(heads)])
        self.dropout = nn.Dropout(p=dropout)

        mlp: list[nn.Module] = []
        prev = feature_dim * heads
        for i, units in enumerate(dense_units):
            mlp.append(LinearAsConv2d(prev, units))
            # act2: 'relu','linear'
            if i == 0:
                mlp.append(nn.ReLU(inplace=False))
            prev = units
        self.mlp = nn.Sequential(*mlp)
        self.fc = LinearAsConv2d(prev, num_classes)

    def forward(self, x):
        # MFCC is computed outside this quantized/exported model.
        x = x.unsqueeze(1)
        x = self.convs(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.size(0), x.size(1), -1)
        recurrent_layers = self.grus if self.rnn_type == "gru" else self.lstms
        for rnn in recurrent_layers:
            # GRU 的状态是 h，LSTM 的状态是 (h, c)；本网络只使用序列输出。
            x, _ = rnn(x)

        mid = x[:, self.mid_index, :].contiguous()
        heads = []
        for proj, attention in zip(self.query_projs, self.attention):
            query = proj(mid).unsqueeze(1)
            ctx = attention(x, query).squeeze(1)
            heads.append(ctx)
        x = torch.cat(heads, dim=-1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.fc(x)


def make_model(rnn_type: str, device: str | torch.device) -> AttMHRNN:
    return AttMHRNN(num_classes=NUM_CLASSES, rnn_type=rnn_type).to(device)


def prepare_model(model):
    """Keep named attention arithmetic as explicit quantizable FX operations."""
    return model_preparer.prepare_model(
        model, module_classes_to_exclude=[custom.AMax, custom.Multiply, custom.Sum,
                                         custom.Subtract, custom.Exponential,
                                         custom.Clamp, custom.Divide])


def configure_attention_quantizers(model) -> None:
    """A ReduceMax selection must use the same encoding as its source scores."""
    for name, module in model.named_modules():
        if not name.endswith(".score_sum"):
            continue
        head_name = name.rsplit(".", 1)[0]
        maximum = model.get_submodule(f"{head_name}.score_max")
        centered = model.get_submodule(f"{head_name}.center_scores")
        scores = module.output_quantizers[0]
        maximum.input_quantizers[0] = scores
        maximum.output_quantizers[0] = scores
        centered.input_quantizers[0] = scores
        centered.input_quantizers[1] = scores


def save_checkpoint(model, path: Path, rnn_type: str, speech_features=None) -> None:
    """模型类型随权重保存，并单独保存外置 MFCC 的常量。"""
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=True)
        if previous.get("rnn_type") != rnn_type:
            raise ValueError(f"{path} 的模型类型不匹配或缺少类型信息，请选择新的输出路径")
    path.parent.mkdir(parents=True, exist_ok=True)
    if speech_features is None:
        speech_features = getattr(model, "_external_frontend_state", None)
    if speech_features is None:
        speech_features = make_speech_features().state_dict()
    elif isinstance(speech_features, nn.Module):
        speech_features = speech_features.state_dict()
    torch.save({"rnn_type": rnn_type, "model_variant": "ada200-equivalent-v1",
                "model": model.state_dict(), "speech_features": speech_features}, path)


def load_checkpoint(model, path: Path, rnn_type: str, device, speech_features=None) -> None:
    """Strictly migrate legacy Linear weights or load equivalent core weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("rnn_type") != rnn_type:
        raise ValueError(f"检查点 {path} 的 rnn_type 与当前 {rnn_type} 不一致")
    source = dict(checkpoint["model"])
    if checkpoint.get("model_variant") == "ada200-equivalent-v1":
        frontend_state = checkpoint["speech_features"]
    elif checkpoint.get("model_variant") is None:
        frontend_state = {name.removeprefix("speech_features."): value
                          for name, value in source.items() if name.startswith("speech_features.")}
        source = {name: value for name, value in source.items()
                  if not name.startswith("speech_features.")}
    else:
        raise ValueError(f"Unsupported checkpoint model_variant: {checkpoint['model_variant']}")
    if speech_features is None:
        expected_frontend = make_speech_features()
        expected_frontend.load_state_dict(frontend_state, strict=True)
        default_state = make_speech_features().state_dict()
        if any(not torch.equal(value.cpu(), default_state[name])
               for name, value in frontend_state.items()):
            raise ValueError("Non-default MFCC buffers require an explicit speech_features instance")
    else:
        speech_features.load_state_dict(frontend_state, strict=True)
    target = model.state_dict()
    migrated = {}
    for name, value in source.items():
        target_name = name
        if name not in target and name.endswith((".weight", ".bias")):
            prefix, parameter = name.rsplit(".", 1)
            candidate = f"{prefix}.conv.{parameter}"
            if candidate in target:
                if parameter == "weight":
                    if value.ndim != 2 or tuple(target[candidate].shape) != (*value.shape, 1, 1):
                        raise ValueError(f"Invalid Linear-to-Conv weight shape for {name}: {tuple(value.shape)}")
                    value = value.unsqueeze(-1).unsqueeze(-1)
                target_name = candidate
        if target_name in migrated:
            raise ValueError(f"Duplicate migrated checkpoint parameter: {target_name}")
        migrated[target_name] = value
    model.load_state_dict(migrated, strict=True)
    model._external_frontend_state = {name: value.detach().cpu().clone()
                                      for name, value in frontend_state.items()}
