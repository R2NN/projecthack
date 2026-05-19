from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PRODUCT_NAME_ZONE = (0.02, 0.08, 0.62, 0.43)
DEFAULT_DATA_DIR_NAME = "\u0414\u0430\u043d\u043d\u044b\u0435"
DEFAULT_EXCLUDES = ("26_12-20", "Unlabeled")


@dataclass(frozen=True)
class SourceRow:
    source_id: str
    row_index: int
    csv_path: Path
    video_path: Path
    timestamp_ms: int
    bbox: tuple[float, float, float, float]
    product_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small real PaddleOCR product_name dataset from annotated Lenta videos."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repro_outputs") / "ppocr_finetune" / "real_product_name_v1",
    )
    parser.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDES))
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=Path("repro_outputs")
        / "third_party"
        / "PaddleOCR"
        / "ppocr"
        / "utils"
        / "dict"
        / "ppocrv5_cyrillic_dict.txt",
    )
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--tag-pad-ratio", type=float, default=0.015)
    parser.add_argument("--zone-pad-ratio", type=float, default=0.015)
    parser.add_argument("--preview-limit", type=int, default=1000)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    return parser.parse_args()


def find_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root

    preferred = Path("A:/lenta") / DEFAULT_DATA_DIR_NAME
    if preferred.exists():
        return preferred

    for child in Path("A:/lenta").iterdir():
        if child.is_dir() and child.name == DEFAULT_DATA_DIR_NAME:
            return child

    raise FileNotFoundError("Could not find the annotated data directory on A:/lenta")


def normalize_label(value: str) -> str:
    return " ".join((value or "").replace("\t", " ").split())


def parse_float(value: str) -> float:
    return float(str(value).strip().replace(",", "."))


def parse_timestamp_ms(value: str) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return int(round(parse_float(value)))


def video_path_for_source(source_dir: Path) -> Path:
    direct = source_dir / f"{source_dir.name}.mp4"
    if direct.exists():
        return direct
    videos = sorted(source_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No video found in {source_dir}")
    return videos[0]


def read_source_rows(data_root: Path, excludes: set[str]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for source_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if source_dir.name in excludes:
            continue
        csv_path = source_dir / f"{source_dir.name}.csv"
        if not csv_path.exists():
            continue
        video_path = video_path_for_source(source_dir)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row_index, row in enumerate(reader):
                product_name = normalize_label(row.get("product_name", ""))
                if not product_name:
                    continue
                try:
                    bbox = (
                        parse_float(row["x_min"]),
                        parse_float(row["y_min"]),
                        parse_float(row["x_max"]),
                        parse_float(row["y_max"]),
                    )
                    timestamp_ms = parse_timestamp_ms(row.get("frame_timestamp", "0"))
                except (KeyError, ValueError):
                    continue
                rows.append(
                    SourceRow(
                        source_id=source_dir.name,
                        row_index=row_index,
                        csv_path=csv_path,
                        video_path=video_path,
                        timestamp_ms=timestamp_ms,
                        bbox=bbox,
                        product_name=product_name,
                    )
                )
    return rows


def imwrite_jpeg(path: Path, image_bgr: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def read_frame_at_ms(cap: cv2.VideoCapture, timestamp_ms: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_ms))
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = int(round(max(0, timestamp_ms) * fps / 1000.0))
    if frame_count > 0:
        frame_index = min(frame_index, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    if frame_count > 1:
        for fallback_index in range(min(frame_index, frame_count - 1), max(-1, frame_index - 90), -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
    return None


def crop_bbox(image: np.ndarray, bbox: tuple[float, float, float, float], pad_ratio: float) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    pad = pad_ratio * max(right - left, bottom - top)
    ix1 = max(0, int(math.floor(left - pad)))
    iy1 = max(0, int(math.floor(top - pad)))
    ix2 = min(width, int(math.ceil(right + pad)))
    iy2 = min(height, int(math.ceil(bottom + pad)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return image[iy1:iy2, ix1:ix2].copy()


def crop_relative(image: np.ndarray, rel_box: tuple[float, float, float, float], pad_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = rel_box
    ix1 = max(0, int(round((x1 - pad_ratio) * width)))
    iy1 = max(0, int(round((y1 - pad_ratio) * height)))
    ix2 = min(width, int(round((x2 + pad_ratio) * width)))
    iy2 = min(height, int(round((y2 + pad_ratio) * height)))
    ix2 = max(ix1 + 1, ix2)
    iy2 = max(iy1 + 1, iy2)
    return image[iy1:iy2, ix1:ix2].copy()


def smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def split_line_boxes(product_crop_bgr: np.ndarray, max_lines: int = 6) -> list[tuple[int, int, int, int]]:
    height, width = product_crop_bgr.shape[:2]
    if height < 10 or width < 20:
        return [(0, 0, width, height)]

    scale = max(1.0, 180.0 / height)
    work_width = int(round(width * scale))
    work_height = int(round(height * scale))
    gray = cv2.cvtColor(product_crop_bgr, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(gray, (work_width, work_height), interpolation=cv2.INTER_CUBIC)
    work = cv2.GaussianBlur(work, (3, 3), 0)

    block_size = max(15, int(round(work_height / 5)) | 1)
    binary = cv2.adaptiveThreshold(
        work,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        13,
    )
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(16, work_width // 3), 1)),
        iterations=1,
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(14, work_height // 4))),
        iterations=1,
    )
    binary = cv2.subtract(binary, horizontal)
    binary = cv2.subtract(binary, vertical)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    text_mask = np.zeros_like(binary)
    for index in range(1, component_count):
        x, y, component_w, component_h, area = [int(value) for value in stats[index]]
        if area < 5:
            continue
        if component_h < max(3, int(round(work_height * 0.015))):
            continue
        if component_h > max(12, int(round(work_height * 0.24))):
            continue
        if component_w < 2:
            continue
        if component_w > int(round(work_width * 0.70)):
            continue
        aspect = component_w / max(1, component_h)
        if aspect > 8.0 and component_h < max(8, int(round(work_height * 0.08))):
            continue
        fill = area / max(1, component_w * component_h)
        if fill < 0.035:
            continue
        touches_edge = y <= 1 or y + component_h >= work_height - 1 or x <= 1 or x + component_w >= work_width - 1
        if touches_edge and (component_w > work_width * 0.10 or component_h > work_height * 0.12):
            continue
        text_mask[labels == index] = 255

    text_mask = cv2.dilate(
        text_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, work_width // 80), 2)),
        iterations=1,
    )

    projection = (text_mask > 0).sum(axis=1)
    projection = smooth_projection(projection, max(3, work_height // 55))
    nonzero = projection[projection > 0]
    if len(nonzero) == 0:
        return [(0, 0, width, height)]

    threshold = max(1.5, min(work_width * 0.08, float(np.percentile(nonzero, 40)) * 0.65))
    active = projection > threshold

    bands_work: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            bands_work.append((start, idx))
            start = None
    if start is not None:
        bands_work.append((start, len(active)))

    min_height = max(4, int(round(0.025 * work_height)))
    gap_merge = max(1, int(round(0.008 * work_height)))
    merged: list[tuple[int, int]] = []
    for top, bottom in bands_work:
        if bottom - top < min_height:
            continue
        if merged and top - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], bottom)
        else:
            merged.append((top, bottom))

    if not merged:
        return [(0, 0, width, height)]

    pad_y = max(2, int(round(0.016 * work_height)))
    pad_x = max(4, int(round(0.012 * work_width)))
    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in merged:
        roi = text_mask[max(0, top - pad_y) : min(work_height, bottom + pad_y), :]
        col_projection = (roi > 0).sum(axis=0)
        active_cols = np.flatnonzero(col_projection > 0)
        if len(active_cols) == 0:
            continue
        x1_work = max(0, int(active_cols[0]) - pad_x)
        x2_work = min(work_width, int(active_cols[-1]) + pad_x + 1)
        y1 = max(0, int(math.floor((top - pad_y) / scale)))
        y2 = min(height, int(math.ceil((bottom + pad_y) / scale)))
        x1 = max(0, int(math.floor(x1_work / scale)))
        x2 = min(width, int(math.ceil(x2_work / scale)))
        if y1 <= 1 and y2 - y1 < max(7, int(round(0.07 * height))):
            continue
        if y2 - y1 < max(7, int(round(0.055 * height))):
            continue
        if y2 - y1 >= 4 and x2 - x1 >= 8:
            boxes.append((x1, y1, x2, y2))

    if len(boxes) > max_lines:
        scored = []
        for x1, y1, x2, y2 in boxes:
            wy1 = int(round(y1 * scale))
            wy2 = int(round(y2 * scale))
            scored.append((float(projection[wy1:wy2].sum()), x1, y1, x2, y2))
        boxes = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in sorted(scored, reverse=True)[:max_lines]]
        boxes.sort(key=lambda item: item[1])

    return boxes or [(0, 0, width, height)]


def triangle_score(value: float, low: float, target: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value <= target:
        return (value - low) / max(1e-6, target - low)
    return (high - value) / max(1e-6, high - target)


def sigmoid_score(value: float, center: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def line_crop_quality(line_crop_bgr: np.ndarray) -> dict[str, float | int]:
    height, width = line_crop_bgr.shape[:2]
    gray = cv2.cvtColor(line_crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if height > 2 and width > 2 else 0.0
    contrast_raw = float(gray.std()) if height > 0 and width > 0 else 0.0
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(15, (height // 2) * 2 + 1),
        11,
    )
    ink_density = float((binary > 0).sum()) / max(1, height * width)
    aspect = width / max(1, height)

    sharpness = sigmoid_score(math.log1p(sharpness_raw), center=4.1, scale=0.8)
    contrast = triangle_score(contrast_raw, low=8.0, target=42.0, high=95.0)
    density = triangle_score(ink_density, low=0.018, target=0.18, high=0.58)
    height_score = triangle_score(float(height), low=6.0, target=22.0, high=80.0)
    aspect_score = triangle_score(aspect, low=1.1, target=7.0, high=24.0)
    quality = (
        0.26 * sharpness
        + 0.22 * contrast
        + 0.24 * density
        + 0.16 * height_score
        + 0.12 * aspect_score
    )
    return {
        "quality_score": round(max(0.0, min(1.0, quality)), 6),
        "width": int(width),
        "height": int(height),
        "sharpness_score": round(sharpness, 6),
        "contrast_score": round(contrast, 6),
        "ink_density": round(ink_density, 6),
    }


def split_wrappable_tokens(label: str) -> list[str]:
    tokens: list[str] = []
    for word in label.split():
        if "-" not in word or word.startswith("-") or word.endswith("-"):
            tokens.append(word)
            continue
        parts = [part for part in word.split("-") if part]
        if len(parts) <= 1:
            tokens.append(word)
            continue
        tokens.extend([f"{part}-" for part in parts[:-1]])
        tokens.append(parts[-1])
    return tokens


def wrap_label_into_lines(label: str, line_widths: list[int]) -> list[str]:
    line_count = len(line_widths)
    words = split_wrappable_tokens(label)
    if line_count <= 1 or len(words) <= 1:
        return [label]

    total_chars = sum(len(word) for word in words) + max(0, len(words) - 1)
    total_width = max(1, sum(max(1, width) for width in line_widths))
    targets = [max(4, int(round(total_chars * max(1, width) / total_width))) for width in line_widths]
    if len(words) < line_count:
        return words + [""] * (line_count - len(words))

    prefix = [0]
    for word in words:
        prefix.append(prefix[-1] + len(word))

    def segment_length(start: int, end: int) -> int:
        return prefix[end] - prefix[start] + max(0, end - start - 1)

    inf = 1e18
    dp = [[inf] * (len(words) + 1) for _ in range(line_count + 1)]
    back = [[-1] * (len(words) + 1) for _ in range(line_count + 1)]
    dp[0][0] = 0.0

    for line_idx in range(1, line_count + 1):
        min_end = line_idx
        max_end = len(words) - (line_count - line_idx)
        for end in range(min_end, max_end + 1):
            min_start = line_idx - 1
            max_start = end - 1
            for start in range(min_start, max_start + 1):
                if dp[line_idx - 1][start] >= inf:
                    continue
                length = segment_length(start, end)
                target = targets[line_idx - 1]
                cost = ((length - target) ** 2) / max(1.0, float(target))
                candidate = dp[line_idx - 1][start] + cost
                if candidate < dp[line_idx][end]:
                    dp[line_idx][end] = candidate
                    back[line_idx][end] = start

    if back[line_count][len(words)] < 0:
        target = max(8, int(math.ceil(total_chars / line_count)))
        return wrap_label_into_lines(label, [target] * line_count)

    spans: list[tuple[int, int]] = []
    end = len(words)
    for line_idx in range(line_count, 0, -1):
        start = back[line_idx][end]
        spans.append((start, end))
        end = start
    spans.reverse()
    return [" ".join(words[start:end]) for start, end in spans]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_label_file(path: Path, samples: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for rel_path, label in samples:
            file.write(f"{rel_path}\t{label}\n")


def rel_for_html(path: str) -> str:
    return path.replace("\\", "/")


def build_html_table(title: str, rows: list[dict[str, object]], columns: list[tuple[str, str]], limit: int) -> str:
    rendered_rows = []
    for row in rows[:limit]:
        cells = []
        for key, label in columns:
            value = row.get(key, "")
            if key.endswith("rel_path") or key in {"rel_path", "field_rel_path", "tag_rel_path"}:
                src = html.escape(rel_for_html(str(value)))
                cells.append(f'<td><img src="../{src}" loading="lazy"><div class="path">{src}</div></td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rendered_rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "\n".join(rendered_rows)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #f5f5f5; z-index: 1; }}
    img {{ max-width: 320px; max-height: 120px; background: #fff; }}
    .path {{ color: #666; font-size: 11px; margin-top: 4px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Rows shown: {min(limit, len(rows))} / {len(rows)}</p>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""


def copy_or_build_dict(dict_path: Path, output_dir: Path, labels: Iterable[str]) -> str:
    target = output_dir / "dict.txt"
    if dict_path.exists():
        shutil.copyfile(dict_path, target)
        return "copied"

    chars = sorted({char for label in labels for char in label if char != " "})
    with target.open("w", encoding="utf-8", newline="\n") as file:
        for char in chars:
            file.write(f"{char}\n")
    return "generated_from_labels"


def source_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        source_id = str(row["source_id"])
        item = grouped.setdefault(source_id, {"source_id": source_id, "field_samples": 0, "line_candidates": 0})
        item["field_samples"] = int(item["field_samples"]) + 1
        item["line_candidates"] = int(item["line_candidates"]) + int(row["line_count"])
    return list(grouped.values())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data_root = find_data_root(args.data_root)
    output_dir = args.output.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_source_rows(data_root, set(args.exclude))
    if not rows:
        raise RuntimeError("No annotated source rows found")

    rows_by_video: dict[Path, list[SourceRow]] = {}
    for row in rows:
        rows_by_video.setdefault(row.video_path, []).append(row)

    field_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    for video_path, video_rows in sorted(rows_by_video.items(), key=lambda item: str(item[0])):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            for row in video_rows:
                failed_rows.append(
                    {
                        "source_id": row.source_id,
                        "row_index": row.row_index,
                        "timestamp_ms": row.timestamp_ms,
                        "reason": f"could_not_open_video:{video_path}",
                    }
                )
            continue

        by_timestamp: dict[int, list[SourceRow]] = {}
        for row in video_rows:
            by_timestamp.setdefault(row.timestamp_ms, []).append(row)

        for timestamp_ms in sorted(by_timestamp):
            frame = read_frame_at_ms(cap, timestamp_ms)
            if frame is None:
                for row in by_timestamp[timestamp_ms]:
                    failed_rows.append(
                        {
                            "source_id": row.source_id,
                            "row_index": row.row_index,
                            "timestamp_ms": row.timestamp_ms,
                            "reason": "could_not_read_frame",
                        }
                    )
                continue

            for row in by_timestamp[timestamp_ms]:
                sample_id = f"{row.source_id}_row{row.row_index:04d}_ts{row.timestamp_ms:06d}"
                tag_crop = crop_bbox(frame, row.bbox, args.tag_pad_ratio)
                if tag_crop is None:
                    failed_rows.append(
                        {
                            "source_id": row.source_id,
                            "row_index": row.row_index,
                            "timestamp_ms": row.timestamp_ms,
                            "reason": "empty_bbox_crop",
                        }
                    )
                    continue

                normalized_tag = cv2.rotate(tag_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                product_crop = crop_relative(normalized_tag, PRODUCT_NAME_ZONE, args.zone_pad_ratio)

                tag_rel_path = f"full_tags/{row.source_id}/{sample_id}_tag.jpg"
                field_rel_path = f"field/images/{row.source_id}/{sample_id}_product_name.jpg"
                imwrite_jpeg(output_dir / tag_rel_path, normalized_tag, args.jpeg_quality)
                imwrite_jpeg(output_dir / field_rel_path, product_crop, args.jpeg_quality)

                line_boxes = split_line_boxes(product_crop)
                proposed_labels = wrap_label_into_lines(row.product_name, [x2 - x1 for x1, _y1, x2, _y2 in line_boxes])

                field_rows.append(
                    {
                        "rel_path": field_rel_path,
                        "label": row.product_name,
                        "source_id": row.source_id,
                        "row_index": row.row_index,
                        "timestamp_ms": row.timestamp_ms,
                        "line_count": len(line_boxes),
                        "tag_rel_path": tag_rel_path,
                        "csv_path": str(row.csv_path),
                    }
                )

                for line_index, (box, proposed_label) in enumerate(zip(line_boxes, proposed_labels)):
                    x1, y1, x2, y2 = box
                    line_crop = product_crop[y1:y2, x1:x2].copy()
                    line_rel_path = (
                        f"line_candidates/images/{row.source_id}/"
                        f"{sample_id}_line{line_index + 1:02d}_of_{len(line_boxes):02d}.jpg"
                    )
                    imwrite_jpeg(output_dir / line_rel_path, line_crop, args.jpeg_quality)
                    quality = line_crop_quality(line_crop)
                    line_rows.append(
                        {
                            "rel_path": line_rel_path,
                            "reviewed_label": proposed_label,
                            "proposed_label": proposed_label,
                            "full_product_name": row.product_name,
                            "source_id": row.source_id,
                            "row_index": row.row_index,
                            "timestamp_ms": row.timestamp_ms,
                            "line_index": line_index + 1,
                            "line_count": len(line_boxes),
                            "x_min": x1,
                            "x_max": x2,
                            "y_min": y1,
                            "y_max": y2,
                            "field_rel_path": field_rel_path,
                            "tag_rel_path": tag_rel_path,
                            "needs_review": 1,
                            "approved": "",
                            **quality,
                        }
                    )

        cap.release()

    random.shuffle(field_rows)
    split_index = int(round(len(field_rows) * args.train_ratio))
    split_index = min(max(1, split_index), max(1, len(field_rows) - 1))
    train_rows = field_rows[:split_index]
    val_rows = field_rows[split_index:]

    write_label_file(output_dir / "train.txt", ((str(row["rel_path"]), str(row["label"])) for row in train_rows))
    write_label_file(output_dir / "val.txt", ((str(row["rel_path"]), str(row["label"])) for row in val_rows))
    write_label_file(
        output_dir / "line_candidates_proposed.txt",
        ((str(row["rel_path"]), str(row["proposed_label"])) for row in line_rows),
    )

    field_columns = [
        "rel_path",
        "label",
        "source_id",
        "row_index",
        "timestamp_ms",
        "line_count",
        "tag_rel_path",
        "csv_path",
    ]
    line_columns = [
        "rel_path",
        "reviewed_label",
        "proposed_label",
        "full_product_name",
        "source_id",
        "row_index",
        "timestamp_ms",
        "line_index",
        "line_count",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "field_rel_path",
        "tag_rel_path",
        "needs_review",
        "approved",
        "quality_score",
        "width",
        "height",
        "sharpness_score",
        "contrast_score",
        "ink_density",
    ]
    write_csv(output_dir / "field" / "labels.csv", field_rows, field_columns)
    write_csv(output_dir / "line_candidates" / "review_queue.csv", line_rows, line_columns)
    priority_line_rows = sorted(line_rows, key=lambda item: float(item["quality_score"]), reverse=True)
    for rank, line_row in enumerate(priority_line_rows, start=1):
        line_row["priority_rank"] = rank
    write_csv(
        output_dir / "line_candidates" / "review_queue_priority.csv",
        priority_line_rows,
        ["priority_rank", *line_columns],
    )
    write_csv(output_dir / "failed_rows.csv", failed_rows, ["source_id", "row_index", "timestamp_ms", "reason"])
    write_csv(output_dir / "stats_by_source.csv", source_stats(field_rows), ["source_id", "field_samples", "line_candidates"])

    dict_mode = copy_or_build_dict(args.dict_path, output_dir, [str(row["label"]) for row in field_rows])

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "field_preview.html").write_text(
        build_html_table(
            "Exact product_name field crops",
            field_rows,
            [("rel_path", "crop"), ("label", "exact label"), ("source_id", "source"), ("line_count", "lines")],
            args.preview_limit,
        ),
        encoding="utf-8",
    )
    (preview_dir / "line_review.html").write_text(
        build_html_table(
            "Line crop review queue",
            priority_line_rows,
            [
                ("rel_path", "line crop"),
                ("reviewed_label", "label to review"),
                ("full_product_name", "full product_name"),
                ("quality_score", "quality"),
                ("priority_rank", "rank"),
                ("source_id", "source"),
                ("line_index", "line"),
                ("line_count", "lines"),
                ("field_rel_path", "field crop"),
            ],
            args.preview_limit,
        ),
        encoding="utf-8",
    )

    max_label_len = max((len(str(row["label"])) for row in field_rows), default=0)
    metadata = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "excluded_sources": sorted(set(args.exclude)),
        "field_samples_exact": len(field_rows),
        "train_samples_exact": len(train_rows),
        "val_samples_exact": len(val_rows),
        "line_candidates_need_review": len(line_rows),
        "line_priority_review_file": "line_candidates/review_queue_priority.csv",
        "failed_rows": len(failed_rows),
        "sources": sorted({row.source_id for row in rows}),
        "dict_mode": dict_mode,
        "max_exact_label_length": max_label_len,
        "product_name_zone": PRODUCT_NAME_ZONE,
        "tag_normalization": "bbox crop rotated 90 degrees counterclockwise before product_name zone crop",
        "note": (
            "train.txt/val.txt contain exact full product_name field crops. "
            "line_candidates/review_queue.csv contains visual line crops with heuristic labels "
            "and must be reviewed before using as supervised line-level OCR data."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Real product_name OCR dataset",
                "",
                "`train.txt` and `val.txt` are exact field-level samples from annotated CSV files.",
                "`line_candidates/review_queue.csv` contains real visual line crops with proposed labels.",
                "Review and correct `reviewed_label` before training a line-level recognizer on those rows.",
                "",
                f"Excluded sources: {', '.join(sorted(set(args.exclude)))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
