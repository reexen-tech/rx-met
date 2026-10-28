"""
RX-MET 量化演示 — kws_streaming att_mh_rnn + QuantGRU / QuantLSTM

流程用法：
    FP 训练 → prepare_model → QuantizationSimModel + 混合精度位宽
        → compute_encodings 校准 (PTQ)
        → apply_power_of_2_workflow (NPU 友好的 Po2 量化)
        → freeze_quantizer_parameters + QAT 微调
        → state_dict + ONNX + .json 编码文件三件套保存
        → 重建 sim → load_state_dict → load_quantizer_encodings → 精度对比

训练侧按 google-research/kws_streaming/models/att_mh_rnn.py：
    * Speech Commands v2 的 12 类协议（10 个 wanted word + _silence_ + _unknown_）
    * 文件名 hash 划分 train/validation/testing（validation/testing 各 10%）
    * 增强：time shift、背景噪声、时间拉伸（resample）
    * 完整浮点 SpeechFeatures（mfcc_tf）外置，分类网络输入 [49, 20] MFCC
    * Conv2D 10,1  kernel (5,1)  same + ReLU + BN
    * 2 层 Bidirectional GRU 或 LSTM（rnn_units=128）+ 4-head 注意力 + Dense
    * 上游原生支持 rnn_type=gru/lstm，本示例使用对应的 rx-met 量化算子

数据集 Speech Commands v0.02（官方包，解压后指向容器内路径）：

    mkdir -p /datasets/speech_commands_v0.02
    wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    tar -xf speech_commands_v0.02.tar.gz -C /datasets/speech_commands_v0.02

运行（先进入 examples/）：

    export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
    python quick_start_kws.py --rnn_type gru
    python quick_start_kws.py --rnn_type lstm

两种模式共用 config/quick_start_full_quant.json，分别读取 GRU_config / LSTM_config。
默认输出到运行目录的 output/quick_start_kws/，再分到 gru/ 或 lstm/ 下。
默认各训练 1 epoch，用于学习流程；
精度取决于训练轮数和配置，不代表上游论文成绩。完整说明见同目录 README.md。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import aimet_torch.v2 as aimet
from aimet_torch import model_preparer
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.utils_rx import (
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    freeze_quantizer_parameters,
    set_train_mode_freeze_bn,
)
from aimet_torch.v2 import quantsim
from aimet_torch._base.nn.modules import custom
from aimet_common.run_context import execution
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM


# ============================================================================
# 全局配置（kws_streaming base_parser / att_mh_rnn 默认值）
# ============================================================================
SEED = 42
DEVICE = torch.device("cuda")

DATA_ROOT = os.environ.get(
    "RX_MET_SPEECH_COMMANDS_ROOT", "/datasets/speech_commands_v0.02"
)

# 12 类：_silence_ / _unknown_ + 10 个 wanted words
WANTED_WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
LABELS = (SILENCE_LABEL, UNKNOWN_LABEL) + WANTED_WORDS
NUM_CLASSES = len(LABELS)

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
TARGET_LEN = SAMPLE_RATE * CLIP_DURATION_MS // 1000
WINDOW_SIZE_MS = 40.0
WINDOW_STRIDE_MS = 20.0
N_FFT = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)          # 640
HOP_LENGTH = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)   # 320
N_MELS = 40
N_MFCC = 20
MEL_FMIN = 20.0
MEL_FMAX = 7000.0
N_FRAMES = 1 + (TARGET_LEN - N_FFT) // HOP_LENGTH         # 49
WAVEFORM_SHAPE = (TARGET_LEN,)                            # preprocess=raw: [T]
FEATURE_SHAPE = (N_FRAMES, N_MFCC)

# kws_streaming 数据协议
KWS_RANDOM_SEED = 59185
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
SILENCE_PERCENTAGE = 10.0
UNKNOWN_PERCENTAGE = 10.0
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10
TIME_SHIFT_MS = 100.0
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME = 0.1
RESAMPLE = 0.15
MAX_ABS_INT16 = 32768.0

# 训练：quick_start 用 1 epoch 跑通流程；完整 kws 是分阶段 step（如 10k+10k+10k）
BATCH_SIZE = 64
NUM_WORKERS = 4
FP_EPOCHS = 1
QAT_EPOCHS = 1
FP_LR = 5e-4          # kws_streaming --learning_rate 第一段
QAT_LR = 1e-4

QUANT_SCHEME = "percentile"
PERCENTILE_VALUE = 99.99
DEFAULT_BW = 8
MAX_CALIB_BATCHES = 100

CONFIG_FILE = _HERE / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = _HERE / "config" / "quick_start_full_quant.json"
OUTPUT_DIR = Path(
    os.environ.get(
        "RX_MET_KWS_OUTPUT_DIR", str(Path.cwd() / "output" / "quick_start_kws")
    )
)


# ============================================================================
# 复现性 / 环境辅助
# ============================================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 数据集（对齐 kws_streaming/data/input_data.py）
# ============================================================================
def which_set(filename: str, validation_percentage: int, testing_percentage: int) -> str:
    """与 kws_streaming input_data_utils.which_set 相同的稳定 hash 划分。"""
    base_name = os.path.basename(filename)
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage_hash = (
        (int(hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    )
    if percentage_hash < validation_percentage:
        return "val"
    if percentage_hash < (testing_percentage + validation_percentage):
        return "test"
    return "train"


def _load_wav(path) -> tuple[torch.Tensor, int]:
    """读 wav 为 float32 PCM，int16 按 kws 的 32768 归一化。返回 [C, T]。"""
    sr, data = wavfile.read(str(path))
    if data.ndim == 1:
        data = data[:, None]
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max + 1)
        if data.dtype == np.int16:
            scale = MAX_ABS_INT16
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)


class SpeechCommandsKWS(Dataset):
    """
    kws_streaming AudioProcessor 的 PyTorch 版。

    返回 (wav[T], label)；外置 SpeechFeatures 在进入分类网络前提取 MFCC。
    """

    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        assert self.root.exists(), f"Data root not found: {root}"
        self.split = split
        self.train_mode = split == "train"
        self.label_to_idx = {name: i for i, name in enumerate(LABELS)}
        self.time_shift_samples = int(TIME_SHIFT_MS * SAMPLE_RATE / 1000)

        rng = random.Random(KWS_RANDOM_SEED)
        found_words: set[str] = set()
        items: list[tuple[Path, str]] = []
        unknown_pool: list[tuple[Path, str]] = []

        for word_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            word = word_dir.name.lower()
            if word.startswith("_"):
                continue
            found_words.add(word)
            for wav in sorted(word_dir.glob("*.wav")):
                part = which_set(str(wav), VALIDATION_PERCENTAGE, TESTING_PERCENTAGE)
                if part != split:
                    continue
                if word in WANTED_WORDS:
                    items.append((wav, word))
                else:
                    unknown_pool.append((wav, UNKNOWN_LABEL))

        missing = [w for w in WANTED_WORDS if w not in found_words]
        if missing:
            raise RuntimeError(f"数据集缺少 wanted words: {missing}")
        if not items:
            raise RuntimeError(f"No items for split={split} at {root}")

        silence_size = int(math.ceil(len(items) * SILENCE_PERCENTAGE / 100.0))
        unknown_size = int(math.ceil(len(items) * UNKNOWN_PERCENTAGE / 100.0))
        silence_wav = items[0][0]
        items.extend((silence_wav, SILENCE_LABEL) for _ in range(silence_size))
        rng.shuffle(unknown_pool)
        items.extend(unknown_pool[:unknown_size])
        rng.shuffle(items)
        self.items = items

        self.bg_noises: list[torch.Tensor] = []
        noise_dir = self.root / "_background_noise_"
        if noise_dir.exists():
            for w in sorted(noise_dir.glob("*.wav")):
                wav, sr = _load_wav(w)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
                self.bg_noises.append(wav.mean(dim=0) if wav.dim() == 2 else wav)

    def __len__(self) -> int:
        return len(self.items)

    def _to_mono_16k(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if wav.dim() == 2:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav

    def _pad_or_crop(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]
        if length < TARGET_LEN:
            return F.pad(wav, (0, TARGET_LEN - length))
        if length > TARGET_LEN:
            start = (length - TARGET_LEN) // 2
            return wav[..., start:start + TARGET_LEN]
        return wav

    def _time_shift(self, wav: torch.Tensor) -> torch.Tensor:
        if self.time_shift_samples <= 0:
            return wav
        shift = random.randint(-self.time_shift_samples, self.time_shift_samples)
        if shift > 0:
            return F.pad(wav, (shift, 0))[..., :TARGET_LEN]
        if shift < 0:
            return F.pad(wav, (0, -shift))[..., -shift:-shift + TARGET_LEN]
        return wav

    def _resample_stretch(self, wav: torch.Tensor) -> torch.Tensor:
        if RESAMPLE <= 0:
            return self._pad_or_crop(wav)
        factor = random.uniform(1.0 - RESAMPLE, 1.0 + RESAMPLE)
        new_len = max(1, int(round(wav.shape[-1] * factor)))
        stretched = F.interpolate(
            wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
        return self._pad_or_crop(stretched)

    def _mix_background(self, wav: torch.Tensor, is_silence: bool) -> torch.Tensor:
        foreground = torch.zeros_like(wav) if is_silence else wav
        if not self.bg_noises:
            return foreground
        if random.random() >= BACKGROUND_FREQUENCY:
            return foreground
        noise = random.choice(self.bg_noises)
        if noise.numel() <= TARGET_LEN:
            raise ValueError("Background sample is too short")
        start = random.randint(0, noise.numel() - TARGET_LEN)
        noise_seg = noise[start:start + TARGET_LEN].view_as(foreground)
        volume = random.uniform(0.0, BACKGROUND_VOLUME)
        return torch.clamp(foreground + volume * noise_seg, -1.0, 1.0)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav, sr = _load_wav(path)
        wav = self._to_mono_16k(wav, sr)
        wav = self._pad_or_crop(wav)
        if self.train_mode:
            wav = self._resample_stretch(wav)
            wav = self._time_shift(wav)
            wav = self._mix_background(wav, is_silence=(label == SILENCE_LABEL))
        elif label == SILENCE_LABEL:
            wav = torch.zeros_like(wav)
        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return wav.squeeze(0).contiguous(), y


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_dataloaders(root: str):
    train_ds = SpeechCommandsKWS(root, split="train")
    test_ds = SpeechCommandsKWS(root, split="test")
    val_ds = SpeechCommandsKWS(root, split="val")
    # 校准只使用训练集，关闭随机数据增强；测试集仅用于阶段评估。
    calib_ds = SpeechCommandsKWS(root, split="train")
    calib_ds.train_mode = False

    g_train = torch.Generator()
    g_train.manual_seed(SEED)
    g_calib = torch.Generator()
    g_calib.manual_seed(SEED)

    def _make(ds, *, shuffle, generator=None, drop_last=False):
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    print(
        f"kws_streaming 12 类划分  train={len(train_ds)}  val={len(val_ds)}  "
        f"test={len(test_ds)}  labels={list(LABELS)}"
    )
    return {
        "train": _make(train_ds, shuffle=True, generator=g_train, drop_last=True),
        "test": _make(test_ds, shuffle=False),
        "val": _make(val_ds, shuffle=False),
        "calib": _make(calib_ds, shuffle=True, generator=g_calib),
    }


# ============================================================================
# SpeechFeatures：kws_streaming _mfcc_tf（分帧 → Hann → RDFT → Mel → log → DCT）
# ============================================================================
def _hertz_to_mel(frequencies_hertz: np.ndarray) -> np.ndarray:
    # kws_streaming/layers/mel_table.py HertzToMel
    return 1127.0 * np.log(1.0 + (frequencies_hertz / 700.0))


def _spectrogram_to_mel_matrix(
    num_mel_bins: int,
    num_spectrogram_bins: int,
    audio_sample_rate: float,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
) -> np.ndarray:
    """kws_streaming/layers/mel_table.py SpectrogramToMelMatrix。"""
    nyquist_hertz = audio_sample_rate / 2.0
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = _hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(
        _hertz_to_mel(np.array(lower_edge_hertz)),
        _hertz_to_mel(np.array(upper_edge_hertz)),
        num_mel_bins + 2,
    )
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins), dtype=np.float64)
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix.astype(np.float32)


def _hann_window(window_length: int) -> np.ndarray:
    # kws windowing._hann_window_generator：周期 Hann，分母是 N 不是 N-1
    arg = 2.0 * np.pi / window_length
    return (0.5 - 0.5 * np.cos(arg * np.arange(window_length))).astype(np.float32)


def _compute_fft_size(frame_size: int) -> int:
    return 2 ** int(math.ceil(math.log(frame_size) / math.log(2.0)))


def _non_zero_mel_size(mel_weight_matrix: np.ndarray) -> int:
    """kws MagnitudeRDFTmel._get_non_zero_mel_size。"""
    non_zero_ind = mel_weight_matrix.shape[0]
    last_mel_ind = mel_weight_matrix.shape[1] - 1
    for i in reversed(range(mel_weight_matrix.shape[0])):
        if mel_weight_matrix[i, last_mel_ind] != 0.0:
            non_zero_ind = i
            break
    return int(np.minimum(non_zero_ind + 1, mel_weight_matrix.shape[0]))


def _rdft_matrices(frame_size: int, fft_mel_size: int) -> tuple[np.ndarray, np.ndarray]:
    """kws MagnitudeRDFT.build：实/虚 DFT，再截到 frame_size × fft_mel_size。"""
    fft_size = _compute_fft_size(frame_size)
    idx = np.arange(fft_size, dtype=np.float64)
    omega = 2.0 * np.pi * np.outer(idx, idx) / fft_size
    dft_real = np.cos(omega).astype(np.float32)
    dft_imag = (-np.sin(omega)).astype(np.float32)
    dft_real = dft_real[:fft_mel_size, :].T[:frame_size, :]
    dft_imag = dft_imag[:fft_mel_size, :].T[:frame_size, :]
    return dft_real, dft_imag


def _dct_matrix(feature_size: int, num_features: int) -> np.ndarray:
    """kws layers/dct.py DCT：2 * cos(...) / sqrt(2N)，只保留前 num_features 列。"""
    norm = 1.0 / np.sqrt(2.0 * feature_size)
    dct = 2.0 * np.cos(
        np.pi
        * np.outer(np.arange(feature_size) * 2.0 + 1.0, np.arange(feature_size))
        / (2.0 * feature_size)
    )
    return (dct[:, :num_features] * norm).astype(np.float32)


class DataFrame(nn.Module):
    """tf.signal.frame： [B, T] → [B, n_frames, frame_size]，pad_end=False。"""

    def __init__(self, frame_size: int, frame_step: int):
        super().__init__()
        self.frame_size = frame_size
        self.frame_step = frame_step
        self.unfold = nn.Unfold(kernel_size=(1, frame_size), stride=(1, frame_step))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        framed = self.unfold(x.unsqueeze(1).unsqueeze(1))
        return framed.transpose(1, 2).contiguous()


class Windowing(nn.Module):
    def __init__(self, frame_size: int):
        super().__init__()
        self.register_buffer("window", torch.from_numpy(_hann_window(frame_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.window


class MagnitudeRDFTmel(nn.Module):
    def __init__(
        self,
        frame_size: int,
        sample_rate: int,
        num_mel_bins: int,
        lower_edge_hertz: float,
        upper_edge_hertz: float,
        magnitude_squared: bool = False,
        mel_non_zero_only: bool = True,
    ):
        super().__init__()
        self.magnitude_squared = magnitude_squared
        fft_size = _compute_fft_size(frame_size)
        feature_size = fft_size // 2 + 1
        mel = _spectrogram_to_mel_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=feature_size,
            audio_sample_rate=float(sample_rate),
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        )
        fft_mel_size = _non_zero_mel_size(mel) if mel_non_zero_only else feature_size
        mel = mel[:fft_mel_size, :]
        real_dft, imag_dft = _rdft_matrices(frame_size, fft_mel_size)
        self.register_buffer("real_dft", torch.from_numpy(real_dft))
        self.register_buffer("imag_dft", torch.from_numpy(imag_dft))
        self.register_buffer("mel_weight", torch.from_numpy(mel))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        real_spectrum = torch.matmul(frames, self.real_dft)
        imag_spectrum = torch.matmul(frames, self.imag_dft)
        magnitude = real_spectrum * real_spectrum + imag_spectrum * imag_spectrum
        if not self.magnitude_squared:
            magnitude = torch.sqrt(magnitude)
        return torch.matmul(magnitude, self.mel_weight)


class LogMax(nn.Module):
    """tf.math.log(tf.math.maximum(x, log_epsilon))。"""

    def __init__(self, log_epsilon: float = 1e-12):
        super().__init__()
        self.log_epsilon = float(log_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.log_epsilon))


class DCT(nn.Module):
    def __init__(self, feature_size: int, num_features: int):
        super().__init__()
        self.register_buffer("dct", torch.from_numpy(_dct_matrix(feature_size, num_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.dct)


class Normalizer(nn.Module):
    def __init__(self, num_features: int, mean=None, stddev=None):
        super().__init__()
        mean_t = torch.zeros(num_features) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.ones(num_features) if stddev is None else torch.as_tensor(stddev, dtype=torch.float32)
        self.register_buffer("mean", mean_t)
        self.register_buffer("stddev", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.stddev


class SpeechFeatures(nn.Module):
    """kws SpeechFeatures._mfcc_tf + Normalizer。输入 [B, T]，输出 [B, n_frames, dct]。"""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size_ms: float = 40.0,
        window_stride_ms: float = 20.0,
        desired_samples: int = 16000,
        mel_num_bins: int = 40,
        mel_lower_edge_hertz: float = 20.0,
        mel_upper_edge_hertz: float = 7000.0,
        dct_num_features: int = 20,
        log_epsilon: float = 1e-12,
        fft_magnitude_squared: bool = False,
        mel_non_zero_only: bool = True,
    ):
        super().__init__()
        frame_size = int(round(sample_rate * window_size_ms / 1000.0))
        frame_step = int(round(sample_rate * window_stride_ms / 1000.0))
        self.n_frames = 1 + (desired_samples - frame_size) // frame_step
        self.dct_num_features = dct_num_features
        self.data_frame = DataFrame(frame_size=frame_size, frame_step=frame_step)
        self.windowing = Windowing(frame_size=frame_size)
        self.mag_rdft_mel = MagnitudeRDFTmel(
            frame_size=frame_size,
            sample_rate=sample_rate,
            num_mel_bins=mel_num_bins,
            lower_edge_hertz=mel_lower_edge_hertz,
            upper_edge_hertz=mel_upper_edge_hertz,
            magnitude_squared=fft_magnitude_squared,
            mel_non_zero_only=mel_non_zero_only,
        )
        self.log_max = LogMax(log_epsilon=log_epsilon)
        self.dct = DCT(feature_size=mel_num_bins, num_features=dct_num_features)
        self.normalizer = Normalizer(num_features=dct_num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.data_frame(x)
        x = self.windowing(x)
        x = self.mag_rdft_mel(x)
        x = self.log_max(x)
        x = self.dct(x)
        return self.normalizer(x)


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


def make_speech_features(device="cpu") -> SpeechFeatures:
    """Create the unchanged, external waveform-to-MFCC transform."""
    return SpeechFeatures(
        sample_rate=SAMPLE_RATE,
        window_size_ms=WINDOW_SIZE_MS,
        window_stride_ms=WINDOW_STRIDE_MS,
        desired_samples=TARGET_LEN,
        mel_num_bins=N_MELS,
        mel_lower_edge_hertz=MEL_FMIN,
        mel_upper_edge_hertz=MEL_FMAX,
        dct_num_features=N_MFCC,
    ).to(device).eval()


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


def extract_mfcc_feeds(feeds, speech_features, device="cpu"):
    """Apply the identical frontend before calibration/validation/ONNX feeds."""
    speech_features.to(device).eval()
    with torch.no_grad():
        return [{"input": speech_features(torch.from_numpy(feed["input"]).to(device))
                 .detach().cpu().numpy()} for feed in feeds]


class FeatureLoader:
    """Adapt waveform batches while keeping the frontend outside the core."""

    def __init__(self, loader, speech_features):
        self.loader = loader
        self.speech_features = speech_features

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        device = next(self.speech_features.buffers()).device
        for waveform, labels in self.loader:
            with torch.no_grad():
                features = self.speech_features(waveform.to(device)).detach()
            yield features, labels


# ============================================================================
# 通用工具
# ============================================================================
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).max(1).indices
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return correct / total if total else 0.0


@contextlib.contextmanager
def stage(name: str, timings: dict | None = None):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    t0 = time.time()
    try:
        yield
    finally:
        dur = time.time() - t0
        if timings is not None:
            timings[name] = dur
        if dur < 60:
            print(f"⏱️  耗时: {dur:.2f} 秒")
        else:
            print(f"⏱️  耗时: {dur:.2f} 秒 ({dur / 60:.2f} 分钟)")


# ============================================================================
# 训练循环
# ============================================================================
def train_floating_point(model, train_loader, test_loader, device,
                         save_path, rnn_type: str,
                         epochs: int = FP_EPOCHS, lr: float = FP_LR,
                         speech_features=None) -> float:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, batch_idx = 0.0, 0
        pbar = tqdm(train_loader, desc=f"FP Epoch {epoch}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
        print(f"[FP] Epoch {epoch}/{epochs} - Loss: {running_loss / max(batch_idx, 1):.4f}")
        save_checkpoint(model, save_path, rnn_type, speech_features=speech_features)

    load_checkpoint(model, save_path, rnn_type, device, speech_features=speech_features)
    return evaluate(model, test_loader, device)


def qat_finetune(sim, train_loader, device,
                 epochs: int = QAT_EPOCHS, lr: float = QAT_LR) -> None:
    optim = torch.optim.Adam(
        [p for p in sim.model.parameters() if p.requires_grad], lr=lr, eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        set_train_mode_freeze_bn(sim.model)
        running_loss, valid = 0.0, 0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(sim.model(x), y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            valid += 1
            pbar.set_postfix(
                loss=f"{running_loss / valid:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        print(
            f"[QAT] Epoch {epoch}/{epochs} - Loss: {running_loss / max(valid, 1):.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()


def print_accuracy_summary(fp_acc, ptq_acc, po2_acc, qat_acc, reload_acc):
    print("\n" + "=" * 70)
    print("精度汇总")
    print("=" * 70)
    print(f"  浮点模型精度:           {fp_acc * 100:.2f}%")
    print(f"  PTQ 量化精度:           {ptq_acc * 100:.2f}%")
    print(f"  Power-of-2 量化精度:    {po2_acc * 100:.2f}%")
    print(f"  QAT 微调后精度:         {qat_acc * 100:.2f}%")
    print(
        f"  重新加载后精度:         {reload_acc * 100:.2f}%  "
        f"(差距 {abs(qat_acc - reload_acc) * 100:.3f}%)"
    )
    print("=" * 70)


def print_stage_timings(timings: dict):
    print("\n" + "=" * 70)
    print("耗时汇总")
    print("=" * 70)
    total = sum(timings.values())
    for name, dur in timings.items():
        pct = (dur / total * 100) if total > 0 else 0
        if dur < 60:
            print(f"  {name:35s} {dur:8.2f} 秒 ({pct:5.1f}%)")
        else:
            print(f"  {name:35s} {dur:8.2f} 秒 ({dur / 60:6.2f} 分钟, {pct:5.1f}%)")
    print("-" * 70)
    if total < 3600:
        print(f"  {'总计':35s} {total:8.2f} 秒 ({total / 60:6.2f} 分钟)")
    else:
        print(f"  {'总计':35s} {total:8.2f} 秒 ({total / 3600:.2f} 小时)")
    print("=" * 70)


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


def _new_model(rnn_type: str = "gru") -> AttMHRNN:
    return AttMHRNN(num_classes=NUM_CLASSES, rnn_type=rnn_type).to(DEVICE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="使用 Speech Commands 演示 GRU / LSTM 的浮点训练、PTQ、Po2、QAT、导出和恢复。"
    )
    parser.add_argument("--rnn_type", "--rnn-type", choices=("gru", "lstm"), default="gru",
                        help="循环算子类型（默认 gru）；其余量化步骤相同")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_ROOT),
                        help="Speech Commands v0.02 目录；默认读取 RX_MET_SPEECH_COMMANDS_ROOT")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="输出根目录；自动追加 gru/ 或 lstm/，默认读取 RX_MET_KWS_OUTPUT_DIR")
    parser.add_argument("--quant-config", type=Path, default=BITWIDTH_CONFIG_FILE,
                        help="完整的量化配置 JSON；编辑其中的 GRU_config 或 LSTM_config")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ptq-only", action="store_true",
                        help="加载浮点检查点，执行校准、Po2、导出和恢复；不训练")
    parser.add_argument("--fp-checkpoint", type=Path,
                        default=Path(os.environ["RX_MET_KWS_FP_MODEL"]) if "RX_MET_KWS_FP_MODEL" in os.environ else None)
    parser.add_argument("--calib-dir", type=Path, help="复用 batch-1 的 input NPY/NPZ 样本")
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--export-feeds-dir", type=Path,
                        help="保存固定的 calib/、val/ 和 manifest.json，供 CPU/GPU 和 ONNX 复用")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--calib-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args(argv)
    if args.device == "cpu" and not args.ptq_only:
        parser.error("CPU 仅支持 --ptq-only；浮点训练和 QAT 请使用 CUDA")
    if args.ptq_only and args.fp_checkpoint is None:
        parser.error("--ptq-only 需要 --fp-checkpoint 或 RX_MET_KWS_FP_MODEL")
    if args.validation_dir is not None and args.calib_dir is None:
        parser.error("--validation-dir 需要 --calib-dir")
    for name in ("samples_per_class", "calib_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def load_ptq_feeds(directory: Path, limit=None):
    """Read the same static batch-1 feed files as the ONNX PTQ example."""
    from aimet_onnx.rx_ptq.calibration import iter_calibration_feeds
    feeds = list(iter_calibration_feeds(directory, {"input": [1, TARGET_LEN]},
                 input_dtypes={"input": np.dtype("float32")}))
    files = sorted(directory.glob("*.npz")) or sorted(directory.glob("*.npy"))
    manifest_path = directory.parent / "manifest.json"
    labels = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        key = "val" if directory.name == "val" else "calib"
        if f"{key}_labels" in manifest:
            by_name = dict(zip(manifest[f"{key}_files"], manifest[f"{key}_labels"], strict=True))
            if set(by_name) != {path.name for path in files}:
                raise ValueError(f"{directory} 样本与 manifest 不一致")
            labels = [by_name[path.name] for path in files]
    return feeds[:limit], labels[:limit] if labels is not None else None


def prepare_ptq_feeds(data_dir: Path, output_dir: Path, samples_per_class: int):
    """Select reproducible unaugmented samples and persist the ONNX input feeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"labels": list(LABELS), "seed": SEED, "batch_size": 1, "augmentation": False}
    for split, folder in (("train", "calib"), ("val", "val")):
        dataset = SpeechCommandsKWS(str(data_dir), split=split)
        dataset.train_mode = False
        rng = random.Random(SEED)
        selected = []
        for label in LABELS:
            indices = [index for index, (_, name) in enumerate(dataset.items) if name == label]
            if len(indices) < samples_per_class:
                raise ValueError(f"{split}/{label} 样本少于 {samples_per_class}")
            selected.extend(rng.sample(indices, samples_per_class))
        rng.shuffle(selected)
        directory = output_dir / folder
        directory.mkdir()
        files, labels, sources = [], [], []
        for number, index in enumerate(selected):
            waveform, label = dataset[index]
            filename = f"{number:05d}.npz"
            np.savez(directory / filename, input=waveform.unsqueeze(0).numpy())
            files.append(filename)
            labels.append(int(label))
            sources.append(str(dataset.items[index][0]))
        manifest.update({f"{folder}_files": files, f"{folder}_labels": labels,
                         f"{folder}_sources": sources})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir / "calib", output_dir / "val"


def _run_kws_ptq(args, run):
    from aimet_common.evaluation import classification_accuracy

    if not args.fp_checkpoint.is_file():
        raise FileNotFoundError(args.fp_checkpoint)
    run.note_inputs(checkpoint=args.fp_checkpoint)
    if args.calib_dir is None:
        calib_dir, validation_dir = prepare_ptq_feeds(
            args.data_dir, args.export_feeds_dir or run.root / "data", args.samples_per_class)
    else:
        calib_dir = args.calib_dir
        validation_dir = args.validation_dir or calib_dir.parent / "val"
    waveform_feeds, _ = load_ptq_feeds(calib_dir, args.calib_limit)
    waveform_validation, labels = load_ptq_feeds(validation_dir, args.validation_limit)
    if not waveform_feeds or not waveform_validation:
        raise ValueError("校准和验证样本均不能为空")
    run.note_inputs(calibration=calib_dir, validation=validation_dir)
    set_seed(SEED)
    model = _new_model(args.rnn_type).eval()
    frontend = make_speech_features(DEVICE)
    load_checkpoint(model, args.fp_checkpoint, args.rnn_type, DEVICE, speech_features=frontend)
    feeds = extract_mfcc_feeds(waveform_feeds, frontend, DEVICE)
    validation = extract_mfcc_feeds(waveform_validation, frontend, DEVICE)

    def outputs(module):
        module.eval()
        with torch.no_grad():
            result = [module(torch.from_numpy(feed["input"]).to(DEVICE)).detach().cpu().numpy()
                      for feed in validation]
        if not all(np.isfinite(value).all() for value in result):
            raise RuntimeError("KWS 输出含非有限值")
        return np.concatenate(result)

    stage_outputs = {"fp32": outputs(model)}
    prepared = prepare_model(model.cpu()).to(DEVICE).eval()
    dummy = torch.from_numpy(feeds[0]["input"]).to(DEVICE)
    sim_options = dict(dummy_input=dummy, quant_scheme=QUANT_SCHEME,
                       config_file=str(CONFIG_FILE), default_output_bw=DEFAULT_BW,
                       default_param_bw=DEFAULT_BW)
    sim = quantsim.QuantizationSimModel(prepared, **sim_options)
    sim.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(sim.model, config_file=str(args.quant_config), verbose=True)
    configure_attention_quantizers(sim.model)
    sim.model.eval()
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        for feed in tqdm(feeds, desc="PTQ calibration"):
            sim.model(torch.from_numpy(feed["input"]).to(DEVICE))
    stage_outputs["ptq"] = outputs(sim.model)
    apply_power_of_2_workflow(sim.model, align_bias_scale=True, verbose=True)
    stage_outputs["po2"] = outputs(sim.model)
    checkpoint = run.checkpoints / "att_mh_rnn_kws_ptq.pth"
    save_checkpoint(sim.get_original_model(sim.model, qdq_weights=False), checkpoint, args.rnn_type,
                    speech_features=frontend)
    sim.model.cpu().eval()
    onnx_path, enc_path = export_onnx_json(
        sim, export_dir=str(run.exports), filename_prefix="att_mh_rnn_kws_ptq",
        dummy_input_shape=(1, *FEATURE_SHAPE), opset=18)
    fresh = prepare_model(_new_model(args.rnn_type).eval())
    load_checkpoint(fresh, checkpoint, args.rnn_type, DEVICE, speech_features=frontend)
    restored = quantsim.QuantizationSimModel(fresh, **sim_options)
    restored.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(restored.model, config_file=str(args.quant_config), verbose=True)
    configure_attention_quantizers(restored.model)
    load_quantizer_encodings(restored.model, load_path=str(enc_path), skip_if_not_found=False,
                             allow_overwrite=False, verbose=True)
    stage_outputs["restored"] = outputs(restored.model)
    np.savez(run.exports / "outputs.npz", **{f"{name}_output0": value for name, value in stage_outputs.items()
                                           if name != "restored"})
    restore_delta = float(np.abs(stage_outputs["restored"] - stage_outputs["po2"]).max())
    metadata = {
        "mode": "ptq-only", "device": str(DEVICE), "rnn_type": args.rnn_type,
        "fp_checkpoint": str(args.fp_checkpoint.resolve()),
        "fp_checkpoint_sha256": hashlib.sha256(args.fp_checkpoint.read_bytes()).hexdigest(),
        "calibration_dir": str(calib_dir.resolve()), "calibration_count": len(feeds),
        "validation_dir": str(validation_dir.resolve()), "validation_count": len(validation),
        "calibration_sha256": hashlib.sha256(b"".join(feed["input"].tobytes() for feed in waveform_feeds)).hexdigest(),
        "validation_sha256": hashlib.sha256(b"".join(feed["input"].tobytes() for feed in waveform_validation)).hexdigest(),
        "restore_max_abs": restore_delta, "restore_atol": 1e-5, "restore_rtol": 1e-4,
        "onnx": str(onnx_path), "encodings": str(enc_path),
    }
    if labels is not None:
        metadata["accuracy"] = {name: classification_accuracy([[array]], labels)
                                for name, array in stage_outputs.items()}
    (run.exports / "att_mh_rnn_kws_ptq.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.testing.assert_allclose(stage_outputs["restored"], stage_outputs["po2"], atol=1e-5, rtol=1e-4)
    print(f"PTQ-only 完成；恢复误差: {restore_delta:.6g}")
    print("最终产物请配套使用 .onnx 模型和 .json 量化参数：")
    print(f"ONNX 模型 (.onnx): {onnx_path}")
    print(f"量化参数 (.json):  {enc_path}")


# ============================================================================
# 主流程：标准 RX-MET 量化 + 加载验证
# ============================================================================
def main(argv=None):
    args = parse_args(argv)
    with execution(args.output_dir / args.rnn_type, f'kws_{args.rnn_type}') as run:
        run.note_inputs(dataset=args.data_dir)
        _run_kws(args, run)


def _run_kws(args, run):
    global DEVICE
    DEVICE = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("此示例需要 CUDA GPU，请在 rx-met GPU 镜像中使用 --gpus 启动容器")
    if args.ptq_only:
        return _run_kws_ptq(args, run)
    output_dir = run.exports
    fp_model_path = args.fp_checkpoint or run.checkpoints / "model_fp_kws.pth"
    if not args.quant_config.is_file():
        raise FileNotFoundError(f"量化配置不存在: {args.quant_config}")
    set_seed(SEED)
    timings: dict = {}

    print("=" * 70)
    operator_name = "QuantGRU" if args.rnn_type == "gru" else "QuantLSTM"
    print(f"RX-MET 量化演示 — att_mh_rnn + 双向 {operator_name} + Speech Commands 12 类")
    print("=" * 70)
    print(f"使用设备: {DEVICE}")
    print(f"数据目录: {args.data_dir}")
    print(f"量化配置: {args.quant_config}（循环层查看 {args.rnn_type.upper()}_config）")
    print(f"输出目录: {output_dir}")
    print("默认浮点训练和 QAT 各 1 epoch，用于学习完整流程；精度需充分训练后评估。")
    print(f"特征: 外置完整 MFCC，分类网络输入 {FEATURE_SHAPE}")

    with stage("步骤 1: 加载数据和模型", timings):
        frontend = make_speech_features(DEVICE)
        loaders = {name: FeatureLoader(loader, frontend)
                   for name, loader in build_dataloaders(str(args.data_dir)).items()}
        model = _new_model(args.rnn_type)
        n_params = sum(p.numel() for p in model.parameters())
        n_feat_buf = sum(b.numel() for b in frontend.buffers())
        n_rnn = sum(1 for m in model.modules() if isinstance(m, (QuantGRU, QuantLSTM)))
        print(
            f"att_mh_rnn 参数量: {n_params / 1e3:.1f}K  {operator_name} 层: {n_rnn}（均为 bidirectional）"
        )
        print(f"外置 SpeechFeatures 常量 buffer: {n_feat_buf / 1e3:.1f}K（不进入量化或 ONNX）")

    with stage("步骤 2: 浮点训练 + 评估", timings):
        fp_accuracy = train_floating_point(
            model, loaders["train"], loaders["test"], DEVICE, fp_model_path, args.rnn_type,
            speech_features=frontend,
        )
        print(f"浮点精度: {fp_accuracy * 100:.2f}%")

    with stage("步骤 3: prepare_model + 创建 sim", timings):
        prepared_model = prepare_model(model)
        sample_input, _ = next(iter(loaders["train"]))
        dummy_input = sample_input.to(DEVICE)

        sim = quantsim.QuantizationSimModel(
            prepared_model,
            dummy_input=dummy_input,
            quant_scheme=QUANT_SCHEME,
            config_file=str(CONFIG_FILE),
            default_output_bw=DEFAULT_BW,
            default_param_bw=DEFAULT_BW,
        )
        sim.set_percentile_value(PERCENTILE_VALUE)
        apply_mixed_precision_bitwidth(
            sim.model, config_file=str(args.quant_config), verbose=True,
        )
        configure_attention_quantizers(sim.model)

    with stage("步骤 4: 校准 (PTQ)", timings):
        sim.model.to(DEVICE).eval()
        with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
            for idx, (x, _) in enumerate(tqdm(loaders["calib"], desc="校准", total=MAX_CALIB_BATCHES)):
                if idx >= MAX_CALIB_BATCHES:
                    break
                sim.model(x.to(DEVICE))

        ptq_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"PTQ 量化精度: {ptq_accuracy * 100:.2f}%")

    with stage("步骤 5: Power-of-2 量化", timings):
        apply_power_of_2_workflow(
            sim.model,
            align_bias_scale=True,
            verbose=True,
        )
        po2_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"Power-of-2 量化精度: {po2_accuracy * 100:.2f}%")

    with stage("步骤 6: QAT 微调", timings):
        freeze_quantizer_parameters(sim.model, verbose=True, freeze_bn_affine=True)
        qat_finetune(sim, loaders["train"], DEVICE)
        qat_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"QAT 微调后精度: {qat_accuracy * 100:.2f}%")

    with stage("步骤 7: 保存量化产物", timings):
        output_dir.mkdir(parents=True, exist_ok=True)
        qat_state_pth = run.checkpoints / "att_mh_rnn_kws_qat.pth"
        # 主权重与量化参数分开保存；避免检查点依赖 QuantSim 的内部 Python 状态。
        original_model = sim.get_original_model(sim.model, qdq_weights=False)
        save_checkpoint(original_model, qat_state_pth, args.rnn_type, speech_features=frontend)
        # 在导出前保留测试样本的输出，恢复后比较数值而不只比较准确率。
        reload_input = next(iter(loaders["test"]))[0].to(DEVICE)
        sim.model.eval()
        with torch.no_grad():
            expected_output = sim.model(reload_input).detach().clone()

        sim.model.cpu().eval()
        onnx_path, enc_path = export_onnx_json(
            sim,
            export_dir=str(output_dir),
            filename_prefix="att_mh_rnn_kws",
            dummy_input_shape=(1, *FEATURE_SHAPE),
            opset=18,
        )
        sim.model.to(DEVICE)
        print(f"训练态权重: {qat_state_pth}")
        print("最终产物请配套使用 .onnx 模型和 .json 量化参数：")
        print(f"ONNX 模型 (.onnx): {onnx_path}")
        print(f"量化参数 (.json):  {enc_path}")

    with stage("步骤 8: 重新加载并验证", timings):
        fresh_model = _new_model(args.rnn_type).eval()
        fresh_prepared = prepare_model(fresh_model)
        load_checkpoint(fresh_prepared, qat_state_pth, args.rnn_type, DEVICE, speech_features=frontend)
        fresh_sim = quantsim.QuantizationSimModel(
            fresh_prepared,
            dummy_input=dummy_input,
            quant_scheme=QUANT_SCHEME,
            config_file=str(CONFIG_FILE),
            default_output_bw=DEFAULT_BW,
            default_param_bw=DEFAULT_BW,
        )
        fresh_sim.set_percentile_value(PERCENTILE_VALUE)
        apply_mixed_precision_bitwidth(
            fresh_sim.model, config_file=str(args.quant_config), verbose=True,
        )
        configure_attention_quantizers(fresh_sim.model)

        load_quantizer_encodings(
            fresh_sim.model,
            load_path=str(enc_path),
            skip_if_not_found=False,
            allow_overwrite=False,
            verbose=True,
        )

        fresh_sim.model.eval()
        with torch.no_grad():
            restored_output = fresh_sim.model(reload_input)
        torch.testing.assert_close(restored_output, expected_output, atol=1e-5, rtol=1e-4)
        print(f"恢复输出最大绝对误差: {(restored_output - expected_output).abs().max().item():.6g}")
        reload_accuracy = evaluate(fresh_sim.model, loaders["test"], DEVICE)
        gap = abs(qat_accuracy - reload_accuracy) * 100
        print(f"QAT 保存前精度:    {qat_accuracy * 100:.2f}%")
        print(f"重新加载后精度:    {reload_accuracy * 100:.2f}%")
        if gap < 0.5:
            print(f"✅ 精度差异 {gap:.3f}%，加载验证通过")
        else:
            raise RuntimeError(f"恢复前后精度差异 {gap:.3f} 个百分点，检查点验证失败")

    print_accuracy_summary(fp_accuracy, ptq_accuracy, po2_accuracy, qat_accuracy, reload_accuracy)
    print_stage_timings(timings)
    print("\n量化流程完成（已通过加载验证）\n")


if __name__ == "__main__":
    main()
