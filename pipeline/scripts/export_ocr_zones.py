"""Export OCR-ready zone crops from tracked price-tag crops."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    mode: str
    target_fields: tuple[str, ...]


ZONES = [
    ZoneSpec("product_name", 0.02, 0.02, 0.62, 0.42, "text", ("product_name",)),
    ZoneSpec(
        "qr",
        0.58,
        0.00,
        1.00,
        0.43,
        "qr",
        (
            "qr_code_barcode",
            "price1_qr",
            "price2_qr",
            "price3_qr",
            "price4_qr",
            "wholesale_level_1_count",
            "wholesale_level_1_price",
            "wholesale_level_2_count",
            "wholesale_level_2_price",
            "action_price_qr",
            "action_code_qr",
        ),
    ),
    ZoneSpec("price_default_wide", 0.40, 0.23, 1.00, 0.52, "price", ("price_default",)),
    ZoneSpec("price_default", 0.50, 0.25, 0.99, 0.46, "price", ("price_default",)),
    ZoneSpec("price_default_compact", 0.53, 0.25, 0.99, 0.405, "price", ("price_default",)),
    ZoneSpec("price_default_digits", 0.58, 0.30, 0.99, 0.455, "price", ("price_default",)),
    ZoneSpec("price_card", 0.00, 0.40, 1.00, 0.88, "price", ("price_card", "price_discount")),
    ZoneSpec("price_card_number", 0.06, 0.40, 1.00, 0.84, "price", ("price_card",)),
    ZoneSpec("price_card_big_digits", 0.10, 0.43, 1.00, 0.82, "price", ("price_card",)),
    ZoneSpec("price_discount", 0.12, 0.40, 1.00, 0.84, "price", ("price_discount",)),
    ZoneSpec("discount_amount", 0.02, 0.43, 0.32, 0.76, "text", ("discount_amount",)),
    ZoneSpec("barcode", 0.34, 0.76, 0.99, 0.97, "barcode", ("barcode",)),
    ZoneSpec("barcode_digits", 0.34, 0.86, 0.99, 1.00, "digits", ("barcode",)),
    ZoneSpec("id_sku", 0.02, 0.73, 0.38, 0.84, "digits", ("id_sku",)),
    ZoneSpec("code", 0.02, 0.79, 0.40, 0.91, "text", ("code",)),
    ZoneSpec("print_datetime", 0.02, 0.88, 0.42, 1.00, "text", ("print_datetime",)),
    ZoneSpec("bottom_info", 0.02, 0.70, 0.52, 1.00, "text", ("id_sku", "print_datetime", "code")),
    ZoneSpec("special_symbols", 0.24, 0.72, 0.42, 0.95, "text", ("special_symbols",)),
    ZoneSpec("additional_info", 0.02, 0.26, 0.46, 0.46, "text", ("additional_info",)),
    ZoneSpec("color_reference", 0.02, 0.43, 0.98, 0.84, "color", ("color",)),
]


OUTPUT_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]


PREFERRED_FIELD_SOURCES = {
    "product_name": "product_name",
    "price_default": "price_default_compact",
    "price_card": "price_card_number",
    "price_discount": "price_discount",
    "barcode": "barcode",
    "discount_amount": "discount_amount",
    "id_sku": "bottom_info",
    "print_datetime": "bottom_info",
    "code": "bottom_info",
    "additional_info": "additional_info",
    "color": "color_reference",
    "special_symbols": "special_symbols",
    "qr_code_barcode": "qr",
    "price1_qr": "qr",
    "price2_qr": "qr",
    "price3_qr": "qr",
    "price4_qr": "qr",
    "wholesale_level_1_count": "qr",
    "wholesale_level_1_price": "qr",
    "wholesale_level_2_count": "qr",
    "wholesale_level_2_price": "qr",
    "action_price_qr": "qr",
    "action_code_qr": "qr",
}


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["track_id"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(row: dict[str, str]) -> str:
    return f"track_{int(row['track_id']):04d}_rank_{int(row['rank']):02d}_ts_{int(row['timestamp_ms']):06d}"


def image_path_for_row(row: dict[str, str], preferred_crop: str) -> Path:
    if preferred_crop == "rectified":
        keys = ("rectified_crop", "ocr_crop", "rotated_crop", "raw_crop")
    else:
        keys = ("ocr_crop", "rectified_crop", "rotated_crop", "raw_crop")

    for key in keys:
        value = row.get(key, "").strip()
        if value:
            path = Path(value)
            if path.exists():
                return path
    raise FileNotFoundError(f"No crop image found for track={row.get('track_id')} rank={row.get('rank')}")


def read_rgb(path: Path) -> np.ndarray:
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def crop_zone(image_rgb: np.ndarray, zone: ZoneSpec, pad: float) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x_min = int(round(max(0.0, zone.x_min - pad) * width))
    y_min = int(round(max(0.0, zone.y_min - pad) * height))
    x_max = int(round(min(1.0, zone.x_max + pad) * width))
    y_max = int(round(min(1.0, zone.y_max + pad) * height))
    x_max = max(x_min + 1, x_max)
    y_max = max(y_min + 1, y_max)
    return image_rgb[y_min:y_max, x_min:x_max]


def upscale(image_rgb: np.ndarray, min_height: int, min_width: int, max_scale: float) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = max(min_height / max(1, height), min_width / max(1, width), 1.0)
    scale = min(scale, max_scale)
    if scale <= 1.01:
        return image_rgb
    return cv2.resize(image_rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_CUBIC)


def sharpen_rgb(image_rgb: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), 1.0)
    return cv2.addWeighted(image_rgb, 1.45, blurred, -0.45, 0)


def enhance_for_ocr(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "color":
        return image_rgb

    min_height = 180 if mode in {"text", "price", "digits"} else 320
    min_width = 420 if mode in {"price", "barcode", "qr"} else 280
    enlarged = upscale(image_rgb, min_height=min_height, min_width=min_width, max_scale=4.0)

    lab = cv2.cvtColor(enlarged, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clip_limit = 2.2 if mode in {"barcode", "qr"} else 2.8
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge([lightness, channel_a, channel_b]), cv2.COLOR_LAB2RGB)
    enhanced = sharpen_rgb(enhanced)

    if mode == "price":
        return enhanced

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    if mode in {"text", "digits"} and min(gray.shape[:2]) >= 3:
        gray = cv2.medianBlur(gray, 3)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def binary_for_ocr(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    enhanced = enhance_for_ocr(image_rgb, mode)
    if mode == "color":
        return enhanced

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    if mode in {"qr", "barcode"}:
        min_side = min(gray.shape[:2])
        block_size = max(15, (min_side // 8) | 1)
        block_size = min(block_size, min_side if min_side % 2 == 1 else min_side - 1)
        block_size = max(3, block_size)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            3,
        )
    else:
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def crop_by_bbox(image_rgb: np.ndarray, bbox: tuple[int, int, int, int], pad_fraction: float) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    pad_x = int(round((x_max - x_min) * pad_fraction))
    pad_y = int(round((y_max - y_min) * pad_fraction))
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    return image_rgb[y_min:max(y_min + 1, y_max), x_min:max(x_min + 1, x_max)]


def tight_qr_crop(image_rgb: np.ndarray) -> np.ndarray | None:
    detector = cv2.QRCodeDetector()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, points = detector.detect(image_bgr)
    if ok and points is not None:
        pts = points.reshape(-1, 2)
        x_min, y_min = np.floor(pts.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(pts.max(axis=0)).astype(int)
        return crop_by_bbox(image_rgb, (x_min, y_min, x_max, y_max), 0.12)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox = None
    best_score = 0.0
    height, width = gray.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.14 or h < height * 0.14:
            continue
        aspect = w / max(1, h)
        square_score = 1.0 - min(abs(np.log(max(aspect, 1e-6))), 1.0)
        area_score = min((w * h) / max(1, width * height * 0.20), 1.0)
        right_score = (x + w * 0.5) / max(1, width)
        score = 0.45 * square_score + 0.35 * area_score + 0.20 * right_score
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + w, y + h)
    if best_bbox is None:
        return None
    return crop_by_bbox(image_rgb, best_bbox, 0.18)


def tight_barcode_crop(image_rgb: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox = None
    best_score = 0.0
    height, width = gray.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.20 or h < height * 0.10:
            continue
        aspect = w / max(1, h)
        if aspect < 2.0:
            continue
        lower_score = (y + h * 0.5) / max(1, height)
        area_score = min((w * h) / max(1, width * height * 0.18), 1.0)
        score = 0.55 * area_score + 0.45 * lower_score
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + w, y + h)
    if best_bbox is None:
        return None
    return crop_by_bbox(image_rgb, best_bbox, 0.16)


def tight_price_crop(image_rgb: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / max(1, width * height)
        if area_ratio < 0.004:
            continue
        if h < height * 0.16 and w < width * 0.06:
            continue
        center_x = (x + w * 0.5) / max(1, width)
        center_y = (y + h * 0.5) / max(1, height)
        aspect = w / max(1, h)
        looks_like_left_discount_circle = center_x < 0.23 and 0.55 <= aspect <= 1.65 and h > height * 0.34
        looks_like_bottom_barcode = center_y > 0.82 and h < height * 0.24
        if looks_like_left_discount_circle or looks_like_bottom_barcode:
            continue
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return None

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    if x_max - x_min < width * 0.20 or y_max - y_min < height * 0.22:
        return None
    return crop_by_bbox(image_rgb, (x_min, y_min, x_max, y_max), 0.08)


def tight_code_crop(image_rgb: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "qr":
        return tight_qr_crop(image_rgb)
    if mode == "barcode":
        return tight_barcode_crop(image_rgb)
    if mode == "price":
        return tight_price_crop(image_rgb)
    return None


def resize_full_tag(image_rgb: np.ndarray, full_width: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if width >= full_width:
        return image_rgb
    scale = full_width / max(1, width)
    return cv2.resize(image_rgb, (full_width, int(round(height * scale))), interpolation=cv2.INTER_CUBIC)


def make_zone_sheet(
    rows: list[dict[str, str]],
    output_path: Path,
    max_items: int,
    image_column: str = "zone_enhanced",
) -> None:
    preview_rows = rows[:max_items]
    if not preview_rows:
        return

    font = ImageFont.load_default()
    tile_width = 260
    tile_height = 190
    columns = 4
    sheet = Image.new(
        "RGB",
        (tile_width * columns, tile_height * int(np.ceil(len(preview_rows) / columns))),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(preview_rows):
        image_path_value = row.get(image_column, "")
        if not image_path_value:
            continue
        image_path = Path(image_path_value)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width - 12, tile_height - 32), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        draw.text((x + 6, y + 4), f"T{row['track_id']} R{row['rank']} {row['zone']}", fill=(20, 20, 20), font=font)
        sheet.paste(image, (x + 6, y + 24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_field_zone_map(output_dir: Path) -> None:
    rows = []
    for field_name in OUTPUT_COLUMNS:
        matching_zones = [
            zone.name
            for zone in ZONES
            if field_name in zone.target_fields
        ]
        rows.append(
            {
                "field": field_name,
                "zones": "|".join(matching_zones),
                "recommended_source": PREFERRED_FIELD_SOURCES.get(
                    field_name,
                    matching_zones[0] if matching_zones else "",
                ),
                "expected_when_missing": "нет" if field_name not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"} else "",
            }
        )
    write_rows(output_dir / "field_zone_map.csv", rows)
    with (output_dir / "expected_output_columns.json").open("w", encoding="utf-8") as file:
        json.dump(OUTPUT_COLUMNS, file, ensure_ascii=False, indent=2)


def resolve_filename(tracks_csv: Path) -> str:
    summary_path = tracks_csv.parent / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            video_path = str(summary.get("video", "")).strip()
            if video_path:
                return Path(video_path).name
        except (OSError, json.JSONDecodeError):
            pass
    video_id = tracks_csv.parent.name
    return video_id.removeprefix("unlabeled_") + ".mp4"


def submission_template_rows(manifest_rows: list[dict[str, str]], one_per_track: bool) -> list[dict[str, str]]:
    rows = []
    seen_tracks = set()
    for row in manifest_rows:
        if one_per_track:
            track_id = row["track_id"]
            if track_id in seen_tracks or row["rank"] != "1":
                continue
            seen_tracks.add(track_id)
        output_row = {column: "" for column in OUTPUT_COLUMNS}
        output_row["filename"] = row["filename"]
        output_row["frame_timestamp"] = row["frame_timestamp"]
        output_row["x_min"] = row["x_min"]
        output_row["y_min"] = row["y_min"]
        output_row["x_max"] = row["x_max"]
        output_row["y_max"] = row["y_max"]
        rows.append(output_row)
    return rows


def export_zones(
    tracks_csv: Path,
    output_dir: Path,
    top_k: int,
    zone_pad: float,
    full_width: int,
    preferred_crop: str,
    skip_contact_sheets: bool = False,
    selected_zones: set[str] | None = None,
) -> dict[str, object]:
    rows = read_rows(tracks_csv)
    if top_k > 0:
        rows = [row for row in rows if int(row["rank"]) <= top_k]

    manifest_rows = []
    zone_rows = []
    full_dir = output_dir / "full_tags"
    zones_dir = output_dir / "zones"
    filename = resolve_filename(tracks_csv)

    zones_to_export = [zone for zone in ZONES if not selected_zones or zone.name in selected_zones]

    for row in rows:
        source_path = image_path_for_row(row, preferred_crop=preferred_crop)
        source_rgb = read_rgb(source_path)
        full_rgb = resize_full_tag(source_rgb, full_width=full_width)

        stem = safe_stem(row)
        full_path = full_dir / f"{stem}_full.jpg"
        save_rgb(full_path, full_rgb)

        base_manifest = {
            "video_id": Path(filename).stem,
            "filename": filename,
            "track_id": row["track_id"],
            "rank": row["rank"],
            "timestamp_ms": row["timestamp_ms"],
            "frame_timestamp": row["timestamp_ms"],
            "x_min": row.get("x_min", ""),
            "y_min": row.get("y_min", ""),
            "x_max": row.get("x_max", ""),
            "y_max": row.get("y_max", ""),
            "crop_score": row.get("crop_score", ""),
            "ocr_score": row.get("ocr_score", ""),
            "ocr_crop_kind": row.get("ocr_crop_kind", ""),
            "source_crop": str(source_path),
            "full_tag": str(full_path),
        }
        manifest_rows.append(base_manifest)

        for zone in zones_to_export:
            zone_rgb = crop_zone(full_rgb, zone, pad=zone_pad)
            enhanced_rgb = enhance_for_ocr(zone_rgb, zone.mode)
            binary_rgb = binary_for_ocr(zone_rgb, zone.mode)
            tight_rgb = tight_code_crop(zone_rgb, zone.mode)
            tight_enhanced_rgb = enhance_for_ocr(tight_rgb, zone.mode) if tight_rgb is not None else None
            tight_binary_rgb = binary_for_ocr(tight_rgb, zone.mode) if tight_rgb is not None else None
            zone_dir = zones_dir / zone.name
            raw_path = zone_dir / f"{stem}_{zone.name}_raw.jpg"
            enhanced_path = zone_dir / f"{stem}_{zone.name}_ocr.jpg"
            binary_path = zone_dir / f"{stem}_{zone.name}_binary.jpg"
            tight_path = zone_dir / f"{stem}_{zone.name}_tight.jpg"
            tight_enhanced_path = zone_dir / f"{stem}_{zone.name}_tight_ocr.jpg"
            tight_binary_path = zone_dir / f"{stem}_{zone.name}_tight_binary.jpg"
            save_rgb(raw_path, zone_rgb)
            save_rgb(enhanced_path, enhanced_rgb)
            save_rgb(binary_path, binary_rgb)
            if tight_rgb is not None and tight_enhanced_rgb is not None and tight_binary_rgb is not None:
                save_rgb(tight_path, tight_rgb)
                save_rgb(tight_enhanced_path, tight_enhanced_rgb)
                save_rgb(tight_binary_path, tight_binary_rgb)
                tight_path_value = str(tight_path)
                tight_enhanced_path_value = str(tight_enhanced_path)
                tight_binary_path_value = str(tight_binary_path)
            else:
                tight_path_value = ""
                tight_enhanced_path_value = ""
                tight_binary_path_value = ""
            zone_rows.append(
                {
                    **base_manifest,
                    "zone": zone.name,
                    "zone_mode": zone.mode,
                    "target_fields": "|".join(zone.target_fields),
                    "zone_raw": str(raw_path),
                    "zone_enhanced": str(enhanced_path),
                    "zone_binary": str(binary_path),
                    "zone_tight": tight_path_value,
                    "zone_tight_enhanced": tight_enhanced_path_value,
                    "zone_tight_binary": tight_binary_path_value,
                    "zone_x_min": f"{zone.x_min:.3f}",
                    "zone_y_min": f"{zone.y_min:.3f}",
                    "zone_x_max": f"{zone.x_max:.3f}",
                    "zone_y_max": f"{zone.y_max:.3f}",
                }
            )

    write_rows(output_dir / "ocr_manifest.csv", manifest_rows)
    write_rows(output_dir / "ocr_zones_manifest.csv", zone_rows)
    write_rows(output_dir / "submission_template.csv", submission_template_rows(manifest_rows, one_per_track=True))
    write_rows(output_dir / "submission_crop_template.csv", submission_template_rows(manifest_rows, one_per_track=False))
    write_field_zone_map(output_dir)
    shutil.copy2(tracks_csv, output_dir / "source_tracks_top_crops.csv")

    if not skip_contact_sheets:
        for zone in zones_to_export:
            zone_manifest_rows = [row for row in zone_rows if row["zone"] == zone.name]
            make_zone_sheet(
                zone_manifest_rows,
                output_dir / f"contact_sheet_{zone.name}.jpg",
                max_items=40,
            )
            if zone.mode in {"qr", "barcode", "price"}:
                make_zone_sheet(
                    [row for row in zone_manifest_rows if row["zone_tight_enhanced"]],
                    output_dir / f"contact_sheet_{zone.name}_tight.jpg",
                    max_items=40,
                    image_column="zone_tight_enhanced",
                )
                make_zone_sheet(
                    zone_manifest_rows,
                    output_dir / f"contact_sheet_{zone.name}_binary.jpg",
                    max_items=40,
                    image_column="zone_binary",
                )

    summary = {
        "tracks_csv": str(tracks_csv),
        "output": str(output_dir),
        "input_rows": len(rows),
        "zone_count": len(zone_rows),
        "zones": [zone.name for zone in zones_to_export],
        "top_k": top_k,
        "zone_pad": zone_pad,
        "full_width": full_width,
        "preferred_crop": preferred_crop,
        "contact_sheets": not skip_contact_sheets,
        "expected_output_columns": OUTPUT_COLUMNS,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--zone-pad", type=float, default=0.015)
    parser.add_argument("--full-width", type=int, default=960)
    parser.add_argument("--preferred-crop", choices=["rectified", "ocr"], default="rectified")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--zones", default="")
    args = parser.parse_args()

    summary = export_zones(
        tracks_csv=Path(args.tracks_csv).resolve(),
        output_dir=Path(args.output).resolve(),
        top_k=args.top_k,
        zone_pad=args.zone_pad,
        full_width=args.full_width,
        preferred_crop=args.preferred_crop,
        skip_contact_sheets=args.skip_contact_sheets,
        selected_zones={zone.strip() for zone in args.zones.split(",") if zone.strip()} or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
