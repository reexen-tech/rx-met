"""Input-name based calibration feed loading for direct ONNX PTQ."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import onnx


def _shape_from_value_info(value_info: onnx.ValueInfoProto) -> List[int]:
    dims: List[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            raise ValueError(
                f"动态或缺失输入 shape 暂不支持: {value_info.name}"
            )
    return dims


def inspect_model_inputs(model: onnx.ModelProto) -> Dict[str, List[int]]:
    """Return static model input shapes keyed by ONNX input name."""
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    result: Dict[str, List[int]] = {}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        result[value_info.name] = _shape_from_value_info(value_info)
    if not result:
        raise ValueError("ONNX 模型没有可供校准的 graph input")
    return result


def _validate_feed(
    feed: Dict[str, np.ndarray], input_shapes: Dict[str, List[int]], source: str
) -> Dict[str, np.ndarray]:
    required = set(input_shapes)
    missing = required - set(feed)
    extra = set(feed) - required
    if missing:
        raise KeyError(f"{source} 缺少模型输入: {sorted(missing)}")
    if extra:
        raise KeyError(f"{source} 含未知模型输入: {sorted(extra)}")

    normalized: Dict[str, np.ndarray] = {}
    for name, shape in input_shapes.items():
        array = np.asarray(feed[name])
        if list(array.shape) != shape:
            raise ValueError(
                f"{source}:{name} shape={tuple(array.shape)} != {tuple(shape)}"
            )
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        normalized[name] = np.ascontiguousarray(array)
    return normalized


def _iter_npz(
    root: Path, input_shapes: Dict[str, List[int]]
) -> Iterable[Dict[str, np.ndarray]]:
    files = sorted(root.glob("*.npz"))
    if not files:
        return
    for path in files:
        with np.load(path) as data:
            yield _validate_feed(
                {name: data[name] for name in data.files},
                input_shapes,
                str(path),
            )


def _sample_key(path: Path) -> str:
    name = path.stem
    name = re.sub(r"_(?:imu_)?input(?:_label.*)?_fp32$", "", name)
    name = re.sub(r"_audio(?:_data)?(?:_input)?(?:_label.*)?_fp32$", "", name)
    name = re.sub(r"_(?:imu_)?input.*$", "", name)
    name = re.sub(r"_audio.*$", "", name)
    return name


def _input_file_candidates(path: Path, input_name: str) -> bool:
    lower = path.name.lower()
    if input_name == "audio_data":
        return "audio" in lower
    if input_name == "input":
        return "imu" in lower or "input" in lower and "audio" not in lower
    token = input_name.lower().replace("_", "")
    return token in lower.replace("_", "")


def _iter_npy(
    root: Path, input_shapes: Dict[str, List[int]]
) -> Iterable[Dict[str, np.ndarray]]:
    files = sorted(root.glob("*.npy"))
    if not files:
        return
    input_names = list(input_shapes)
    if len(input_names) == 1:
        name = input_names[0]
        for path in files:
            yield _validate_feed(
                {name: np.load(path, allow_pickle=False)},
                input_shapes,
                str(path),
            )
        return

    grouped: Dict[str, Dict[str, Path]] = {}
    for path in files:
        key = _sample_key(path)
        for input_name in input_names:
            if _input_file_candidates(path, input_name):
                grouped.setdefault(key, {})[input_name] = path

    for key in sorted(grouped):
        paths = grouped[key]
        missing = set(input_names) - set(paths)
        if missing:
            raise KeyError(
                f"校准样本 {key} 缺少输入 {sorted(missing)}；"
                "多输入模型请使用同一 sample 的 npy，或改用 npz"
            )
        yield _validate_feed(
            {
                input_name: np.load(paths[input_name], allow_pickle=False)
                for input_name in input_names
            },
            input_shapes,
            f"{root}/{key}",
        )


def iter_calibration_feeds(
    calibration_dir: str | Path,
    input_shapes: Dict[str, List[int]],
    limit: int | None = None,
) -> Iterable[Dict[str, np.ndarray]]:
    """Load real calibration feeds from a directory of npz or npy files.

    ``npz`` files must contain one array per ONNX input name.  For a
    single-input model, individual ``npy`` files are accepted.  For the
    two-input watchhar layout, files containing ``imu`` and ``audio`` are
    paired by their sample prefix.
    """
    root = Path(calibration_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"校准目录不存在: {root}")

    iterator = _iter_npz(root, input_shapes)
    if not list(root.glob("*.npz")):
        iterator = _iter_npy(root, input_shapes)

    count = 0
    for feed in iterator:
        if limit is not None and count >= limit:
            break
        count += 1
        yield feed
    if count == 0:
        raise FileNotFoundError(
            f"{root} 下没有可用校准样本；需要 *.npz 或模型输入对应的 *.npy"
        )
