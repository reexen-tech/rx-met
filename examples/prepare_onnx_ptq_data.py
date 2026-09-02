#!/usr/bin/env python3
"""Prepare MobileNetV2 PTQ feeds from the ImageNette parquet mirror.

The HF parquet is a single unlabeled mix of official train+val (13394 images).
This script does not train. It:

  1. reads the parquet
  2. applies ImageNet preprocessing used by ONNX Model Zoo MobileNetV2
  3. draws a fixed, non-overlapping calib / val split
  4. writes float32 NCHW ``.npy`` files for ``onnx_ptq_quick_start.py``
  5. freezes the ONNX batch dim to 1 (the zoo file uses dynamic ``batch_size``)

Host-side example (pyarrow / pillow / numpy / onnx):

    wget -O mobilenetv2-12.onnx \\
      https://hf-mirror.com/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx
    wget -O imagenette2-320.parquet \\
      https://hf-mirror.com/datasets/johnowhitaker/imagenette2-320/resolve/main/data/train-00000-of-00001.parquet

    python3 prepare_onnx_ptq_data.py \\
      --onnx mobilenetv2-12.onnx \\
      --parquet imagenette2-320.parquet \\
      --out-root /datasets/mobilenetv2
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
RESIZE_MIN = 256
CROP = 224


def _load_parquet_images(parquet_path: Path) -> tuple[object, list[int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("需要 pyarrow：pip install pyarrow") from exc
    table = pq.read_table(parquet_path, columns=["image", "label"])
    images = table["image"].combine_chunks()
    labels = [int(v) for v in table["label"].to_pylist()]
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
    if hasattr(image, "size"):  # PIL.Image
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
        tensor = _preprocess(raw)
        name = f"{prefix}_{rank:04d}.npy"
        np.save(out_dir / name, tensor)
        written.append(name)
    return written


def _freeze_onnx_batch(src: Path, dest: Path, batch: int = 1) -> str:
    import onnx

    model = onnx.load(str(src), load_external_data=False)
    initializer = {item.name for item in model.graph.initializer}
    input_name = None
    for value_info in list(model.graph.input) + list(model.graph.output):
        dims = value_info.type.tensor_type.shape.dim
        if not dims:
            continue
        dims[0].ClearField("dim_param")
        dims[0].dim_value = batch
    for value_info in model.graph.input:
        if value_info.name not in initializer:
            input_name = value_info.name
            break
    if input_name is None:
        raise SystemExit(f"{src} 没有 graph input")
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))
    return input_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="MobileNetV2 fp32 ONNX")
    parser.add_argument("--parquet", type=Path, required=True, help="imagenette2-320 parquet")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--calib-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX 不存在: {args.onnx}")
    if not args.parquet.is_file():
        raise SystemExit(f"parquet 不存在: {args.parquet}")
    if args.calib_count <= 0 or args.val_count <= 0:
        raise SystemExit("calib/val 数量必须 > 0")

    image_bytes, labels = _load_parquet_images(args.parquet)
    calib_idx, val_idx = _stratified_split(
        labels, args.calib_count, args.val_count, args.seed
    )
    if set(calib_idx) & set(val_idx):
        raise SystemExit("分层抽样结果重叠：calib 与 val 不能共用同一张图")

    out_root = args.out_root
    calib_dir = out_root / "calib"
    val_dir = out_root / "val"
    onnx_out = out_root / "mobilenetv2-12.onnx"
    input_name = _freeze_onnx_batch(args.onnx, onnx_out)

    print(f"parquet  {args.parquet}  rows={len(labels)}")
    print(f"onnx in  {args.onnx}")
    print(f"onnx out {onnx_out}  input={input_name}  shape=[1,3,{CROP},{CROP}]")
    print(f"split    seed={args.seed}  calib={len(calib_idx)}  val={len(val_idx)}")

    calib_files = _write_npy(calib_dir, calib_idx, image_bytes, "calib")
    val_files = _write_npy(val_dir, val_idx, image_bytes, "val")
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
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote    {calib_dir} ({len(calib_files)} npy)")
    print(f"wrote    {val_dir} ({len(val_files)} npy)")
    print(f"manifest {manifest_path}")
    print(
        "下一步:\n"
        f"  export RX_MET_ONNX_PTQ_EXAMPLE=mobilenetv2\n"
        f"  export RX_MET_ONNX_PTQ_MODEL={onnx_out}\n"
        f"  export RX_MET_ONNX_PTQ_CALIB={calib_dir}\n"
        "  python3 onnx_ptq_quick_start.py"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
