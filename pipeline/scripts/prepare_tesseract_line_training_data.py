from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Tesseract LSTM line training ground-truth files.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-labels", default="train.txt")
    parser.add_argument("--val-labels", default="val.txt")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--pad-x", type=int, default=10)
    parser.add_argument("--pad-y", type=int, default=6)
    parser.add_argument("--max-width", type=int, default=900)
    return parser.parse_args()


def read_label_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        rel_path, label = line.split("\t", 1)
        label = " ".join(label.replace("\t", " ").split())
        if rel_path and label:
            rows.append((rel_path, label))
    return rows


def read_bgr(path: Path) -> np.ndarray:
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def prepare_image(image_bgr: np.ndarray, height: int, pad_x: int, pad_y: int, max_width: int) -> Image.Image:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    source_height, source_width = gray.shape[:2]
    scale = height / max(1, source_height)
    width = max(12, int(round(source_width * scale)))
    if width > max_width:
        width = max_width
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((height + 2 * pad_y, width + 2 * pad_x), 255, dtype=np.uint8)
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = resized
    return Image.fromarray(canvas)


def write_split(
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    rows: list[tuple[str, str]],
    height: int,
    pad_x: int,
    pad_y: int,
    max_width: int,
) -> list[dict[str, str]]:
    split_dir = output_dir / "ground-truth" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for index, (rel_path, label) in enumerate(rows):
        stem = f"{split}_{index:05d}"
        image_path = split_dir / f"{stem}.tif"
        gt_path = split_dir / f"{stem}.gt.txt"
        box_path = split_dir / f"{stem}.box"
        source_path = dataset_dir / rel_path
        prepared = prepare_image(read_bgr(source_path), height, pad_x, pad_y, max_width)
        prepared.save(image_path)
        gt_path.write_text(label + "\n", encoding="utf-8")
        width, prepared_height = prepared.size
        box_path.write_text(
            f"WordStr 0 0 {width} {prepared_height} 0 # {label}\n"
            f"\t 0 0 {width} {prepared_height} 0\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "split": split,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "gt": str(gt_path.relative_to(output_dir)).replace("\\", "/"),
                "box": str(box_path.relative_to(output_dir)).replace("\\", "/"),
                "source": rel_path,
                "label": label,
            }
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_label_file(dataset_dir / args.train_labels)
    val_rows = read_label_file(dataset_dir / args.val_labels)
    manifest_rows = []
    manifest_rows.extend(
        write_split(dataset_dir, output_dir, "train", train_rows, args.height, args.pad_x, args.pad_y, args.max_width)
    )
    manifest_rows.extend(
        write_split(dataset_dir, output_dir, "val", val_rows, args.height, args.pad_x, args.pad_y, args.max_width)
    )

    with (output_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "image", "gt", "box", "source", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "height": args.height,
        "pad_x": args.pad_x,
        "pad_y": args.pad_y,
        "max_width": args.max_width,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
