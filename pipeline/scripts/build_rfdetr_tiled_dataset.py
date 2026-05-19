"""Build a tiled COCO dataset for RF-DETR price tag detection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_number(value: str) -> float:
    text = str(value).strip().replace(" ", "").replace(",", ".")
    return float(text)


def parse_timestamp_ms(value: str) -> int:
    return int(round(parse_number(value)))


def read_boxes(csv_path: Path) -> dict[int, list[Box]]:
    by_timestamp: dict[int, list[Box]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp_ms = parse_timestamp_ms(row["frame_timestamp"])
                box = Box(
                    x1=parse_number(row["x_min"]),
                    y1=parse_number(row["y_min"]),
                    x2=parse_number(row["x_max"]),
                    y2=parse_number(row["y_max"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if box.area > 16:
                by_timestamp[timestamp_ms].append(box)
    return dict(by_timestamp)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def clip_box_to_tile(
    box: Box,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visible_fraction: float,
) -> list[float] | None:
    ix1 = max(box.x1, tile_x)
    iy1 = max(box.y1, tile_y)
    ix2 = min(box.x2, tile_x + tile_size)
    iy2 = min(box.y2, tile_y + tile_size)
    width = ix2 - ix1
    height = iy2 - iy1
    if width <= 4 or height <= 4:
        return None
    visible_area = width * height
    if box.area <= 0 or visible_area / box.area < min_visible_fraction:
        return None
    return [
        round(ix1 - tile_x, 3),
        round(iy1 - tile_y, 3),
        round(width, 3),
        round(height, 3),
    ]


def read_frame(video: cv2.VideoCapture, timestamp_ms: int):
    video.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = video.read()
    if ok and frame is not None:
        return frame

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = max(0, int(round(timestamp_ms * fps / 1000.0)))
    if frame_count > 0:
        frame_index = min(frame_index, frame_count - 1)
    for candidate_index in range(frame_index, max(-1, frame_index - 240), -1):
        video.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
        ok, frame = video.read()
        if ok and frame is not None:
            return frame
    raise RuntimeError(f"Could not read frame at {timestamp_ms} ms")


def init_coco() -> dict:
    return {
        "info": {"description": "Lenta tiled price tag detection subset"},
        "licenses": [],
        "categories": [{"id": 1, "name": "price_tag", "supercategory": "price_tag"}],
        "images": [],
        "annotations": [],
    }


def export_video_split(
    *,
    video_name: str,
    video_path: Path,
    boxes_by_ts: dict[int, list[Box]],
    timestamps: list[int],
    split_dir: Path,
    coco: dict,
    image_id_start: int,
    annotation_id_start: int,
    tile_size: int,
    stride: int,
    min_visible_fraction: float,
    keep_empty_tiles: bool,
) -> tuple[int, int, dict[str, int]]:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    image_id = image_id_start
    annotation_id = annotation_id_start
    stats = {"frames": 0, "images": 0, "annotations": 0, "empty_images": 0}

    try:
        for timestamp_ms in timestamps:
            frame = read_frame(video, timestamp_ms)
            height, width = frame.shape[:2]
            x_starts = tile_starts(width, tile_size, stride)
            y_starts = tile_starts(height, tile_size, stride)
            frame_boxes = boxes_by_ts.get(timestamp_ms, [])
            stats["frames"] += 1

            for tile_y in y_starts:
                for tile_x in x_starts:
                    tile_annotations: list[list[float]] = []
                    for box in frame_boxes:
                        clipped = clip_box_to_tile(
                            box,
                            tile_x,
                            tile_y,
                            tile_size,
                            min_visible_fraction,
                        )
                        if clipped is not None:
                            tile_annotations.append(clipped)

                    if not tile_annotations and not keep_empty_tiles:
                        continue

                    file_name = f"{video_name}_{timestamp_ms:06d}_x{tile_x}_y{tile_y}.jpg"
                    tile = frame[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        continue
                    write_jpeg(split_dir / file_name, tile)

                    coco["images"].append(
                        {
                            "id": image_id,
                            "file_name": file_name,
                            "width": tile_size,
                            "height": tile_size,
                        }
                    )
                    stats["images"] += 1
                    if not tile_annotations:
                        stats["empty_images"] += 1

                    for bbox in tile_annotations:
                        area = round(bbox[2] * bbox[3], 3)
                        coco["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": bbox,
                                "area": area,
                                "iscrowd": 0,
                                "segmentation": [],
                            }
                        )
                        annotation_id += 1
                        stats["annotations"] += 1
                    image_id += 1
    finally:
        video.release()

    return image_id, annotation_id, stats


def discover_videos(data_root: Path) -> dict[str, tuple[Path, Path]]:
    videos: dict[str, tuple[Path, Path]] = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        video_path = child / f"{child.name}.mp4"
        csv_path = child / f"{child.name}.csv"
        if video_path.exists() and csv_path.exists():
            videos[child.name] = (video_path, csv_path)
    return videos


def split_train_valid(timestamps: list[int], valid_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if len(timestamps) <= 2:
        return timestamps, []
    rng = random.Random(seed)
    shuffled = sorted(timestamps)
    valid_count = max(1, int(round(len(shuffled) * valid_fraction)))
    valid_set = set(rng.sample(shuffled, valid_count))
    train = [ts for ts in shuffled if ts not in valid_set]
    valid = [ts for ts in shuffled if ts in valid_set]
    return train, valid


def write_coco(split_dir: Path, coco: dict) -> None:
    with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False, indent=2)


def write_jpeg(path: Path, image, quality: int = 95) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Could not encode tile: {path}")
    path.write_bytes(np.asarray(encoded).tobytes())


def guarded_rmtree(path: Path) -> None:
    resolved = path.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to delete suspicious output path: {resolved}")
    shutil.rmtree(resolved)


def mirror_split(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, target_dir / item.name)


def build_dataset(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        guarded_rmtree(output_dir)

    videos = discover_videos(data_root)
    if not videos:
        raise ValueError(f"No annotated videos found under {data_root}")
    if not args.all_train and args.holdout_video not in videos:
        raise ValueError(f"Holdout video {args.holdout_video!r} not found under {data_root}")

    for split in ("train", "valid", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    cocos = {split: init_coco() for split in ("train", "valid", "test")}
    next_image_id = {split: 1 for split in ("train", "valid", "test")}
    next_ann_id = {split: 1 for split in ("train", "valid", "test")}
    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "holdout_video": "" if args.all_train else args.holdout_video,
        "all_train": bool(args.all_train),
        "mirror_train_to_valid": bool(args.mirror_train_to_valid),
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "min_visible_fraction": args.min_visible_fraction,
        "valid_fraction": args.valid_fraction,
        "splits": {"train": [], "valid": [], "test": []},
    }
    stride = args.tile_size - args.tile_overlap

    for video_name, (video_path, csv_path) in videos.items():
        boxes_by_ts = read_boxes(csv_path)
        timestamps = sorted(boxes_by_ts)
        if not timestamps:
            continue

        if args.all_train:
            assignments = {"train": timestamps}
        elif video_name == args.holdout_video:
            assignments = {"test": timestamps}
        else:
            train_ts, valid_ts = split_train_valid(
                timestamps,
                args.valid_fraction,
                args.seed + sum(ord(ch) for ch in video_name),
            )
            assignments = {"train": train_ts, "valid": valid_ts}

        for split, split_timestamps in assignments.items():
            if not split_timestamps:
                continue
            keep_empty_tiles = split == "train"
            image_id, ann_id, stats = export_video_split(
                video_name=video_name,
                video_path=video_path,
                boxes_by_ts=boxes_by_ts,
                timestamps=split_timestamps,
                split_dir=output_dir / split,
                coco=cocos[split],
                image_id_start=next_image_id[split],
                annotation_id_start=next_ann_id[split],
                tile_size=args.tile_size,
                stride=stride,
                min_visible_fraction=args.min_visible_fraction,
                keep_empty_tiles=keep_empty_tiles,
            )
            next_image_id[split] = image_id
            next_ann_id[split] = ann_id
            summary["splits"][split].append({"video": video_name, **stats})

    for split, coco in cocos.items():
        write_coco(output_dir / split, coco)

    if args.all_train and args.mirror_train_to_valid:
        mirror_split(output_dir / "train", output_dir / "valid")
        train_coco = json.loads((output_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
        cocos["valid"] = train_coco
        annotated_images = {ann["image_id"] for ann in train_coco["annotations"]}
        summary["splits"]["valid"] = [
            {
                "video": "__mirrored_train__",
                "frames": 0,
                "images": len(train_coco["images"]),
                "annotations": len(train_coco["annotations"]),
                "empty_images": len(train_coco["images"]) - len(annotated_images),
            }
        ]

    summary["totals"] = {}
    for split, coco in cocos.items():
        annotated_images = {ann["image_id"] for ann in coco["annotations"]}
        summary["totals"][split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "empty_images": len(coco["images"]) - len(annotated_images),
        }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-video", default="")
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--mirror-train-to-valid", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=320)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--min-visible-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
