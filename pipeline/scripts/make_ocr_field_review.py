"""Create visual review sheets for aggregated OCR fields."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


NO_VALUE_MARKERS = {"", "нет", "null", "none", "nan"}


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


def normalize_text(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return "" if value.lower() in NO_VALUE_MARKERS else value


def normalize_price(value: str) -> str:
    value = normalize_text(value).replace(" ", "").replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return ""
    number = float(match.group(0))
    return f"{number:.2f}" if "." in match.group(0) else str(int(number))


def normalize_discount(value: str) -> str:
    value = normalize_text(value)
    match = re.search(r"-?\s*(\d{1,2})\s*%", value)
    return f"-{int(match.group(1))}%" if match else ""


def normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", normalize_text(value))


def normalize_for_field(field: str, value: str) -> str:
    if field in {"price_card", "price_default", "price_discount"}:
        return normalize_price(value)
    if field == "discount_amount":
        return normalize_discount(value)
    if field in {"barcode", "id_sku", "qr_code_barcode"}:
        return normalize_digits(value)
    return normalize_text(value)


def values_match(field: str, pred_value: str, gt_value: str) -> bool:
    pred = normalize_for_field(field, pred_value)
    gt = normalize_for_field(field, gt_value)
    if not pred or not gt:
        return False
    if field in {"product_name", "additional_info"}:
        pred_words = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", pred.lower()))
        gt_words = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", gt.lower()))
        if not pred_words or not gt_words:
            return pred.lower() == gt.lower()
        return len(pred_words & gt_words) / max(1, min(len(pred_words), len(gt_words))) >= 0.55
    return pred == gt


def read_rgb(path: str) -> Image.Image:
    if not path:
        return Image.new("RGB", (320, 120), "white")
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return Image.new("RGB", (320, 120), "white")
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = image.copy()
    image.thumbnail((size[0] - 8, size[1] - 8), Image.Resampling.LANCZOS)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def safe_font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("filename", "").strip(),
        row.get("frame_timestamp", "").strip(),
        row.get("x_min", "").strip(),
        row.get("y_min", "").strip(),
        row.get("x_max", "").strip(),
        row.get("y_max", "").strip(),
    )


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

    candidates: list[tuple[dict[str, str], float, int]] = []
    for gt_row in gt_rows:
        try:
            gt_timestamp = int(round(parse_float(gt_row.get("frame_timestamp", "0"))))
            current_iou = iou(source_box, parse_box(gt_row))
        except ValueError:
            continue
        candidates.append((gt_row, current_iou, abs(source_timestamp - gt_timestamp)))

    if not candidates:
        return None, 0.0, str(source_timestamp), 0, "", False

    close_candidates = [item for item in candidates if item[2] <= max_timestamp_delta_ms]
    usable_candidates = close_candidates if close_candidates else candidates
    best_row, best_iou, best_delta = max(usable_candidates, key=lambda item: (item[1], -item[2]))
    top_items = sorted(candidates, key=lambda item: (item[2] <= max_timestamp_delta_ms, item[1], -item[2]), reverse=True)[:3]
    top_summary = " | ".join(
        f"{normalize_price(row.get('price_card', '')) or '?'}@{row.get('frame_timestamp', '')}:iou={candidate_iou:.2f},dt={delta}"
        for row, candidate_iou, delta in top_items
    )
    return best_row, best_iou, str(source_timestamp), best_delta, top_summary, bool(close_candidates)


def shorten(value: str, limit: int) -> str:
    value = normalize_text(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def status_for_match(
    field: str,
    pred_value: str,
    gt_value: str,
    has_close_gt: bool,
    best_iou: float,
    iou_threshold: float,
) -> tuple[str, str]:
    if not has_close_gt:
        return "UNVERIFIED_TIME", "#777777"
    if best_iou < iou_threshold:
        return "LOW_IOU", "#777777"
    if not gt_value:
        return "NO_GT_VALUE", "#777777"
    is_match = values_match(field, pred_value, gt_value)
    return ("OK", "#0a8f3c") if is_match else ("BAD", "#c82828")


def make_field_review(
    field: str,
    pred_rows: list[dict[str, str]],
    debug_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    debug_by_track = {
        row["track_id"]: row
        for row in debug_rows
        if row.get("field") == field and normalize_text(row.get("value", ""))
    }
    track_by_output_key = {
        row_key(row): row["track_id"]
        for row in manifest_rows
        if row.get("rank") == "1" and row.get("zone") == "price_card_number"
    }
    source_by_track_rank_zone = {
        (row["track_id"], row["rank"], row["zone"]): row
        for row in manifest_rows
        if row.get("track_id") and row.get("rank") and row.get("zone")
    }
    full_by_track_rank = {
        (row["track_id"], row["rank"]): row.get("full_tag", "")
        for row in manifest_rows
        if row.get("zone") == "price_card_number"
    }

    rows_to_render = [row for row in pred_rows if normalize_text(row.get(field, ""))]
    if args.max_items:
        rows_to_render = rows_to_render[: args.max_items]

    cell_w, cell_h = 560, 350
    image_w, image_h = 540, 205
    columns = args.columns
    rows_count = max(1, math.ceil(len(rows_to_render) / columns))
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_count), "#f5f5f5")
    draw = ImageDraw.Draw(sheet)
    font = safe_font(17)
    small_font = safe_font(13)
    review_rows: list[dict[str, str]] = []

    for index, pred_row in enumerate(rows_to_render):
        track_id = pred_row.get("track_id", "") or track_by_output_key.get(row_key(pred_row), str(index + 1))
        debug_row = debug_by_track.get(track_id, {})
        rank = debug_row.get("rank", "1")
        zone = debug_row.get("zone", "")
        source_row = source_by_track_rank_zone.get((track_id, rank, zone), pred_row)
        gt_row, best_iou, match_timestamp, match_delta, top_gt_candidates, has_close_gt = best_source_gt_match(
            source_row,
            gt_rows,
            args.max_timestamp_delta_ms,
        )

        pred_value = normalize_for_field(field, pred_row.get(field, ""))
        candidate_gt_value = normalize_for_field(field, gt_row.get(field, "")) if gt_row else ""
        gt_value = candidate_gt_value if has_close_gt else ""
        status, color = status_for_match(field, pred_value, gt_value, has_close_gt, best_iou, args.iou_threshold)

        image_path = debug_row.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            image_path = full_by_track_rank.get((track_id, rank), "")
        image = fit_image(read_rgb(image_path), (image_w, image_h))

        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        draw.rectangle((x + 8, y + 8, x + cell_w - 8, y + cell_h - 8), fill="white", outline="#d0d0d0")
        sheet.paste(image, (x + 10, y + 10))
        draw.text((x + 14, y + 220), f"T{track_id} {field}: {shorten(pred_value, 48) or '<empty>'}", fill="#111111", font=font)
        draw.text((x + 14, y + 246), f"GT: {shorten(gt_value, 50) or '<none>'}  IoU: {best_iou:.2f}", fill=color, font=font)
        draw.text(
            (x + 14, y + 274),
            f"{status} | ts {match_timestamp or '-'} dt {match_delta} | {debug_row.get('engine', '')} r{rank} {zone}",
            fill=color,
            font=small_font,
        )
        draw.text((x + 14, y + 298), shorten(debug_row.get("source_text", ""), 72), fill="#555555", font=small_font)

        review_rows.append(
            {
                "track_id": track_id,
                "field": field,
                "frame_timestamp": pred_row.get("frame_timestamp", ""),
                "pred_value": pred_value,
                "gt_value": gt_value,
                "candidate_gt_value": candidate_gt_value,
                "iou": f"{best_iou:.4f}",
                "match_timestamp": match_timestamp,
                "match_delta_ms": str(match_delta),
                "top_gt_candidates": top_gt_candidates,
                "status": status,
                "engine": debug_row.get("engine", ""),
                "rank": rank,
                "zone": zone,
                "source_text": debug_row.get("source_text", ""),
                "image_path": image_path,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / f"{field}_review.jpg"
    csv_path = args.output_dir / f"{field}_review.csv"
    sheet.save(sheet_path, quality=95)
    write_rows(
        csv_path,
        review_rows,
        [
            "track_id",
            "field",
            "frame_timestamp",
            "pred_value",
            "gt_value",
            "candidate_gt_value",
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
    print(f"{field}: {len(review_rows)} rows -> {sheet_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--debug-csv", type=Path, required=True)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-timestamp-delta-ms", type=int, default=2500)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_rows = read_rows(args.pred_csv)
    debug_rows = read_rows(args.debug_csv)
    manifest_rows = read_rows(args.zones_manifest)
    gt_rows = read_rows(args.gt_csv)
    for field in [item.strip() for item in args.fields.split(",") if item.strip()]:
        make_field_review(field, pred_rows, debug_rows, manifest_rows, gt_rows, args)


if __name__ == "__main__":
    main()
