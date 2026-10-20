#!/usr/bin/env python3
"""Prepare MobileNetV2 PTQ feeds from the ImageNette parquet mirror.

The HF parquet is a single unlabeled mix of official train+val (13394 images).
This script does not train. It:

  1. validates the ONNX model and parquet
  2. applies ImageNet preprocessing used by ONNX Model Zoo MobileNetV2
  3. draws a fixed, non-overlapping calib / val split
  4. writes float32 NCHW ``.npy`` files for ``quick_start_onnx_ptq.py``
  5. freezes the ONNX batch dim to 1 (the zoo file uses dynamic ``batch_size``)

Host-side example (pyarrow / pillow / numpy / onnx):

    wget -O mobilenetv2-12.onnx \\
      https://hf-mirror.com/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx
    wget -O imagenette2-320.parquet \\
      https://hf-mirror.com/datasets/johnowhitaker/imagenette2-320/resolve/main/data/train-00000-of-00001.parquet

    python3 prepare_onnx_ptq_data.py \\
      --onnx /models/mobilenetv2-12.onnx \\
      --parquet /datasets/imagenette2-320.parquet

This is a MobileNetV2 + ImageNette preparation recipe, not a generic ONNX PTQ
preprocessing pipeline. Inputs may be mounted read-only. Results go to models/
and datasets/ under ./output/onnx_ptq_quick_start/ in the current working
directory, or under --out-root DIR. Source files are left unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
RESIZE_MIN = 256
CROP = 224
DEFAULT_OUTPUT_ROOT = Path("output") / "onnx_ptq_quick_start"


def _load_parquet_images(parquet_path: Path) -> tuple[object, list[int]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("需要 pyarrow：pip install pyarrow") from exc
    try:
        table = pq.read_table(parquet_path, columns=["image", "label"])
    except (OSError, pa.ArrowException) as exc:
        raise SystemExit(f"parquet 不可用: {parquet_path}: {exc}") from exc
    image_type = table.schema.field("image").type
    if not pa.types.is_struct(image_type) or image_type.get_field_index("bytes") < 0:
        raise SystemExit("parquet 的 image 列必须包含 bytes 字段")
    bytes_type = image_type.field("bytes").type
    if not (pa.types.is_binary(bytes_type) or pa.types.is_large_binary(bytes_type)):
        raise SystemExit("parquet 的 image.bytes 必须是 binary")
    if not pa.types.is_integer(table.schema.field("label").type) or table["label"].null_count:
        raise SystemExit("parquet 的 label 必须是非空整数")
    images = table["image"].combine_chunks()
    labels = [int(v) for v in table["label"].to_pylist()]
    if not labels:
        raise SystemExit("parquet 没有图片")
    return images.field("bytes"), labels


def _decode_rgb(raw: bytes):
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        return Image.open(io.BytesIO(raw)).convert("RGB")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "需要 Pillow 或 OpenCV 解码 JPEG：pip install pillow"
        ) from exc
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("JPEG 解码失败")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _center_crop_224(image) -> np.ndarray:
    if not isinstance(image, np.ndarray):  # PIL.Image
        from PIL import Image

        width, height = image.size
        scale = RESIZE_MIN / float(min(width, height))
        resized = image.resize(
            (
                max(CROP, int(round(width * scale))),
                max(CROP, int(round(height * scale))),
            ),
            Image.BILINEAR,
        )
        left = (resized.size[0] - CROP) // 2
        top = (resized.size[1] - CROP) // 2
        crop = resized.crop((left, top, left + CROP, top + CROP))
        array = np.asarray(crop, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))

    # OpenCV / numpy HWC uint8
    height, width = image.shape[:2]
    scale = RESIZE_MIN / float(min(width, height))
    new_w = max(CROP, int(round(width * scale)))
    new_h = max(CROP, int(round(height * scale)))
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("需要 Pillow 或 OpenCV：pip install pillow") from exc
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (new_w - CROP) // 2
    top = (new_h - CROP) // 2
    crop = resized[top : top + CROP, left : left + CROP]
    array = crop.astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def _preprocess(raw: bytes) -> np.ndarray:
    chw = _center_crop_224(_decode_rgb(raw))
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def _stratified_split(
    labels: list[int], calib_count: int, val_count: int, seed: int
) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    calib: list[int] = []
    val: list[int] = []
    labels_sorted = sorted(by_label)
    # Round-robin so both splits see every class before leftovers.
    while len(calib) < calib_count or len(val) < val_count:
        progressed = False
        for label in labels_sorted:
            bucket = by_label[label]
            if not bucket:
                continue
            if len(calib) < calib_count:
                calib.append(bucket.pop())
                progressed = True
            elif len(val) < val_count:
                val.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    if len(calib) < calib_count or len(val) < val_count:
        raise SystemExit(
            f"样本不够：需要 calib={calib_count} val={val_count}，"
            f"parquet 只有 {len(labels)} 张"
        )
    return sorted(calib), sorted(val)


def _write_npy(out_dir: Path, indices: list[int], image_bytes, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for rank, index in enumerate(indices):
        raw = image_bytes[index].as_py()
        if raw is None:
            raise SystemExit(f"parquet 第 {index} 张没有 image.bytes")
        try:
            tensor = _preprocess(raw)
        except Exception as exc:
            raise SystemExit(f"parquet 第 {index} 张图片不可用: {exc}") from exc
        name = f"{prefix}_{rank:04d}.npy"
        np.save(out_dir / name, tensor)
        written.append(name)
    return written


def _prepare_onnx_model(src: Path, batch: int = 1) -> tuple[object, str]:
    import onnx

    try:
        model = onnx.load(str(src))
        onnx.checker.check_model(model)
    except Exception as exc:
        raise SystemExit(f"ONNX 模型不可用: {src}: {exc}") from exc
    initializer = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer]
    if len(inputs) != 1:
        raise SystemExit("此准备脚本只支持单输入的 MobileNetV2 图像模型")
    tensor_type = inputs[0].type.tensor_type
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    if tensor_type.elem_type != onnx.TensorProto.FLOAT or len(shape) != 4 or shape[1:] != [3, CROP, CROP]:
        raise SystemExit("模型输入必须是 float32 NCHW [batch, 3, 224, 224]")
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.name in initializer:
            continue
        dims = value_info.type.tensor_type.shape.dim
        if not dims:
            continue
        dims[0].ClearField("dim_param")
        dims[0].dim_value = batch
    onnx.checker.check_model(model)
    return model, inputs[0].name


def _check_output_paths(source: Path, model_out: Path, data_dir: Path) -> None:
    if source.resolve() == model_out.resolve():
        raise SystemExit("预处理模型的输出路径不能覆盖原始 ONNX 文件")
    for split in ("calib", "val"):
        directory = data_dir / split
        if directory.exists():
            if not directory.is_dir():
                raise SystemExit(f"输出路径不是目录: {directory}")
            if any(not path.is_file() or not re.fullmatch(rf"{split}_[0-9]{{4,}}\.npy", path.name)
                   for path in directory.iterdir()):
                raise SystemExit(f"输出目录包含非本脚本生成的文件，请另选 --out-root: {directory}")


def _publish_split(source: Path, destination: Path, prefix: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = {path.name for path in source.iterdir()}
    shutil.copytree(source, destination, dirs_exist_ok=True)
    for path in destination.glob(f"{prefix}_*.npy"):
        if path.name not in names:
            path.unlink()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="MobileNetV2 fp32 ONNX")
    parser.add_argument("--parquet", type=Path, required=True, help="imagenette2-320 parquet")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="输出根目录，默认 ./output/onnx_ptq_quick_start；其下生成 models/ 和 datasets/")
    parser.add_argument("--calib-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.onnx = args.onnx.expanduser().resolve()
    args.parquet = args.parquet.expanduser().resolve()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX 不存在: {args.onnx}")
    if not args.parquet.is_file():
        raise SystemExit(f"parquet 不存在: {args.parquet}")
    if args.calib_count <= 0 or args.val_count <= 0:
        raise SystemExit("calib/val 数量必须 > 0")

    out_root = args.out_root.expanduser().resolve()
    onnx_out = out_root / "models" / "mobilenetv2" / "mobilenetv2-12.onnx"
    data_dir = out_root / "datasets" / "imagenette2-320"
    calib_dir = data_dir / "calib"
    val_dir = data_dir / "val"
    _check_output_paths(args.onnx, onnx_out, data_dir)
    model, input_name = _prepare_onnx_model(args.onnx)
    image_bytes, labels = _load_parquet_images(args.parquet)
    calib_idx, val_idx = _stratified_split(
        labels, args.calib_count, args.val_count, args.seed
    )
    if set(calib_idx) & set(val_idx):
        raise SystemExit("分层抽样结果重叠：calib 与 val 不能共用同一张图")

    print(f"parquet  {args.parquet}  rows={len(labels)}")
    print(f"onnx in  {args.onnx}")
    print(f"onnx out {onnx_out}  input={input_name}  shape=[1,3,{CROP},{CROP}]")
    print(f"split    seed={args.seed}  calib={len(calib_idx)}  val={len(val_idx)}")

    # Decode every selected image before publishing model or dataset outputs.
    with tempfile.TemporaryDirectory(prefix="rx-met-ptq-") as temporary:
        staged = Path(temporary)
        calib_files = _write_npy(staged / "calib", calib_idx, image_bytes, "calib")
        val_files = _write_npy(staged / "val", val_idx, image_bytes, "val")
        try:
            import onnx

            onnx_out.parent.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            onnx.save_model(model, str(onnx_out), save_as_external_data=False)
            _publish_split(staged / "calib", calib_dir, "calib")
            _publish_split(staged / "val", val_dir, "val")
        except OSError as exc:
            raise SystemExit(f"无法写入输出目录；请从可写目录运行或通过 --out-root 指定可写目录: {exc}") from exc
    manifest = {
        "source_parquet": str(args.parquet.resolve()),
        "source_onnx": str(args.onnx.resolve()),
        "onnx": str(onnx_out.resolve()),
        "input_name": input_name,
        "input_shape": [1, 3, CROP, CROP],
        "preprocess": {
            "resize_min": RESIZE_MIN,
            "center_crop": CROP,
            "mean": IMAGENET_MEAN.reshape(-1).tolist(),
            "std": IMAGENET_STD.reshape(-1).tolist(),
        },
        "seed": args.seed,
        "calib_indices": calib_idx,
        "val_indices": val_idx,
        "calib_files": calib_files,
        "val_files": val_files,
        "note": "parquet 无官方 train/val 标记；本脚本按 label 分层抽样，calib 与 val 不重叠。",
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote    {calib_dir} ({len(calib_files)} npy)")
    print(f"wrote    {val_dir} ({len(val_files)} npy)")
    print(f"manifest {manifest_path}")
    command = f"python3 {shlex.quote(str(Path(__file__).resolve().with_name('quick_start_onnx_ptq.py')))}"
    if out_root == DEFAULT_OUTPUT_ROOT.expanduser().resolve():
        print(f"在当前目录执行量化:\n  {command}")
    else:
        print("在当前容器内执行量化:\n"
              f"  RX_MET_ONNX_PTQ_MODEL={shlex.quote(str(onnx_out))} \\\n"
              f"  RX_MET_ONNX_PTQ_CALIB={shlex.quote(str(calib_dir))} \\\n"
              f"  {command}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
