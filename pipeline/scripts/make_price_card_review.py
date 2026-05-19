"""Create a visual review sheet for recognized price_card values."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> float:
    return float(str(value).strip().replace(" ", "").replace(",", "."))


def parse_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(parse_float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_price(value: str) -> str:
    value = str(value).strip().replace(" ", "").replace(",", ".")
    if not value or value.lower() in {"нет", "null"}:
        return ""
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return ""
    number = float(match.group(0))
    return f"{number:.2f}" if "." in match.group(0) else str(int(number))


def read_rgb(path: str) -> Image.Image:
    if not path:
        return Image.new("RGB", (320, 120), "white")
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return Image.new("RGB", (320, 120), "white")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = image.copy()
    image.thumbnail((size[0] - 8, size[1] - 8), Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def safe_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def best_gt_match(
    pred_row: dict[str, str],
    gt_rows: list[dict[str, str]],
) -> tuple[dict[str, str] | None, float]:
    pred_box = parse_box(pred_row)
    best_row: dict[str, str] | None = None
    best_iou = 0.0
    for gt_row in gt_rows:
        try:
            current_iou = iou(pred_box, parse_box(gt_row))
        except ValueError:
            continue
        if current_iou > best_iou:
            best_iou = current_iou
            best_row = gt_row
    return best_row, best_iou


def best_track_gt_match(
    track_id: str,
    track_rows: dict[str, list[dict[str, str]]],
    fallback_row: dict[str, str],
    gt_rows: list[dict[str, str]],
    timestamp_tolerance_ms: int,
) -> tuple[dict[str, str] | None, float, str, int]:
    candidates = track_rows.get(track_id, [])
    if not candidates:
        gt_row, best_iou = best_gt_match(fallback_row, gt_rows)
        return gt_row, best_iou, "", 0

    best_row: dict[str, str] | None = None
    best_iou = 0.0
    best_timestamp = ""
    best_delta = 10**9
    best_score = -1.0
    for gt_row in gt_rows:
        try:
            gt_timestamp = int(round(parse_float(gt_row.get("frame_timestamp", "0"))))
            gt_box = parse_box(gt_row)
        except ValueError:
            continue
        for candidate in candidates:
            try:
                candidate_timestamp = int(round(parse_float(candidate.get("timestamp_ms", "0"))))
                current_iou = iou(parse_box(candidate), gt_box)
            except ValueError:
                continue
            delta = abs(candidate_timestamp - gt_timestamp)
            temporal_bonus = 1.0 if delta <= timestamp_tolerance_ms else max(0.0, 1.0 - delta / 2500.0)
            score = current_iou + 0.45 * temporal_bonus
            if score > best_score:
                best_score = score
                best_row = gt_row
                best_iou = current_iou
                best_timestamp = str(candidate_timestamp)
                best_delta = delta
    return best_row, best_iou, best_timestamp, best_delta


def best_source_gt_match(
    source_row: dict[str, str],
    gt_rows: list[dict[str, str]],
    max_timestamp_delta_ms: int,
) -> tuple[dict[str, str] | None, float, str, int, str, bool]:
    try:
        source_timestamp = int(round(parse_float(source_row.get("timestamp_ms") or source_row.get("frame_timestamp", "0"))))
        source_box = parse_box(source_row)
    except ValueError:
        return None, 0.0, "", 0, "", False

    candidates: list[tuple[float, dict[str, str], float, int]] = []
    for gt_row in gt_rows:
        try:
            gt_timestamp = int(round(parse_float(gt_row.get("frame_timestamp", "0"))))
            current_iou = iou(source_box, parse_box(gt_row))
        except ValueError:
            continue
        delta = abs(source_timestamp - gt_timestamp)
        candidates.append((current_iou, gt_row, current_iou, delta))

    if not candidates:
        return None, 0.0, str(source_timestamp), 0, "", False

    close_candidates = [item for item in candidates if item[3] <= max_timestamp_delta_ms]
    usable_candidates = close_candidates if close_candidates else candidates
    _, best_row, best_iou, best_delta = max(usable_candidates, key=lambda item: (item[2], -item[3]))

    top_items = sorted(candidates, key=lambda item: (item[3] <= max_timestamp_delta_ms, item[2], -item[3]), reverse=True)[:3]
    top_summary = " | ".join(
        f"{normalize_price(row.get('price_card', '')) or '?'}@{row.get('frame_timestamp', '')}:iou={candidate_iou:.2f},dt={delta}"
        for _, row, candidate_iou, delta in top_items
    )
    return best_row, best_iou, str(source_timestamp), best_delta, top_summary, bool(close_candidates)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("filename", "").strip(),
        row.get("frame_timestamp", "").strip(),
        row.get("x_min", "").strip(),
        row.get("y_min", "").strip(),
        row.get("x_max", "").strip(),
        row.get("y_max", "").strip(),
    )


def make_review(args: argparse.Namespace) -> None:
    pred_rows = read_rows(args.pred_csv)
    debug_rows = read_rows(args.debug_csv)
    manifest_rows = read_rows(args.zones_manifest)
    gt_rows = read_rows(args.gt_csv) if args.gt_csv else []

    debug_by_track = {row["track_id"]: row for row in debug_rows if row.get("field") == "price_card"}
    track_rows: dict[str, list[dict[str, str]]] = {}
    if args.track_candidates_csv:
        for row in read_rows(args.track_candidates_csv):
            track_rows.setdefault(row["track_id"], []).append(row)
    track_by_output_key = {
        row_key(row): row["track_id"]
        for row in manifest_rows
        if row.get("rank") == "1" and row.get("zone") == "price_card_number"
    }
    full_by_track_rank = {
        (row["track_id"], row["rank"]): row.get("full_tag", "")
        for row in manifest_rows
        if row.get("zone") == "price_card_number"
    }
    source_by_track_rank_zone = {
        (row["track_id"], row["rank"], row["zone"]): row
        for row in manifest_rows
        if row.get("zone") in {"price_card_number", "price_card_big_digits", "price_card"}
    }

    review_rows: list[dict[str, str]] = []
    cell_w, cell_h = 520, 330
    image_w, image_h = 500, 205
    columns = args.columns
    rows_count = math.ceil(len(pred_rows) / columns)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_count), "#f5f5f5")
    draw = ImageDraw.Draw(sheet)
    font = safe_font(18)
    small_font = safe_font(14)

    for index, pred_row in enumerate(pred_rows):
        track_id = pred_row.get("track_id", "") or track_by_output_key.get(row_key(pred_row), str(index + 1))
        debug_row = debug_by_track.get(track_id, {})
        pred_value = normalize_price(pred_row.get("price_card", ""))
        rank = debug_row.get("rank", "1")
        zone = debug_row.get("zone", "price_card_number")
        source_row = source_by_track_rank_zone.get((track_id, rank, zone), pred_row)
        if gt_rows:
            if args.match_mode == "track":
                gt_row, best_iou, match_timestamp, match_delta = best_track_gt_match(
                    track_id,
                    track_rows,
                    pred_row,
                    gt_rows,
                    args.timestamp_tolerance_ms,
                )
                top_gt_candidates = ""
                has_close_gt = match_delta <= args.max_timestamp_delta_ms
            else:
                gt_row, best_iou, match_timestamp, match_delta, top_gt_candidates, has_close_gt = best_source_gt_match(
                    source_row,
                    gt_rows,
                    args.max_timestamp_delta_ms,
                )
        else:
            gt_row, best_iou, match_timestamp, match_delta, top_gt_candidates, has_close_gt = None, 0.0, "", 0, "", False
        candidate_gt_value = normalize_price(gt_row.get("price_card", "")) if gt_row else ""
        gt_value = candidate_gt_value if has_close_gt else ""
        status = "NO_GT"
        color = "#777777"
        if gt_row and has_close_gt and best_iou >= args.iou_threshold:
            status = "OK" if pred_value == gt_value else "BAD"
            color = "#0a8f3c" if status == "OK" else "#c82828"
        elif gt_row:
            status = "LOW_IOU" if has_close_gt and best_iou < args.iou_threshold else "UNVERIFIED_TIME"

        full_tag_path = full_by_track_rank.get((track_id, rank), "")
        price_image_path = debug_row.get("image_path", "")
        image_path = price_image_path if price_image_path and Path(price_image_path).exists() else full_tag_path
        image = fit_image(read_rgb(image_path), (image_w, image_h))

        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        draw.rectangle((x + 8, y + 8, x + cell_w - 8, y + cell_h - 8), fill="white", outline="#d0d0d0")
        sheet.paste(image, (x + 10, y + 10))
        draw.text((x + 14, y + 220), f"T{track_id} pred: {pred_value or '<empty>'}", fill="#111111", font=font)
        draw.text((x + 14, y + 246), f"GT: {gt_value or '<none>'}  IoU: {best_iou:.2f}", fill=color, font=font)
        draw.text(
            (x + 14, y + 274),
            f"{status} | match_ts {match_timestamp or '-'} dt {match_delta} | {debug_row.get('engine', '')} r{rank}",
            fill=color,
            font=small_font,
        )
        draw.text(
            (x + 14, y + 296),
            debug_row.get("source_text", "")[:58],
            fill="#555555",
            font=small_font,
        )

        review_rows.append(
            {
                "track_id": track_id,
                "frame_timestamp": pred_row.get("frame_timestamp", ""),
                "pred_price_card": pred_value,
                "gt_price_card": gt_value,
                "candidate_gt_price_card": candidate_gt_value,
                "iou": f"{best_iou:.4f}",
                "match_timestamp": match_timestamp,
                "match_delta_ms": str(match_delta),
                "top_gt_candidates": top_gt_candidates,
                "status": status,
                "engine": debug_row.get("engine", ""),
                "rank": rank,
                "zone": debug_row.get("zone", ""),
                "source_text": debug_row.get("source_text", ""),
                "image_path": image_path,
            }
        )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output_image, quality=95)
    write_rows(
        args.output_csv,
        review_rows,
        [
            "track_id",
            "frame_timestamp",
                "pred_price_card",
                "gt_price_card",
                "candidate_gt_price_card",
                "iou",
                "match_timestamp",
                "match_delta_ms",
                "top_gt_candidates",
                "status",
            "engine",
            "rank",
            "zone",
            "source_text",
            "image_path",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--debug-csv", type=Path, required=True)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path)
    parser.add_argument("--track-candidates-csv", type=Path)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--timestamp-tolerance-ms", type=int, default=350)
    parser.add_argument("--max-timestamp-delta-ms", type=int, default=2500)
    parser.add_argument("--match-mode", choices=("source", "track"), default="source")
    return parser.parse_args()


if __name__ == "__main__":
    make_review(parse_args())
