"""Speech Commands data, MFCC preprocessing, and prepared input feeds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset

# KWS defaults from kws_streaming.
SEED = 42

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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def make_speech_features(device="cpu") -> SpeechFeatures:
    """Create the waveform-to-MFCC transform."""
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


def extract_mfcc_feeds(feeds, speech_features, device="cpu"):
    """Apply the identical frontend before calibration/validation/ONNX feeds."""
    speech_features.to(device).eval()
    with torch.no_grad():
        return [{"input": speech_features(torch.from_numpy(feed["input"]).to(device))
                 .detach().cpu().numpy()} for feed in feeds]


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
