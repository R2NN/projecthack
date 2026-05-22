from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


QR_FIELD_ALIASES = {
    "barcode": "qr_code_barcode",
    "b": "qr_code_barcode",
    "price1": "price1_qr",
    "p1": "price1_qr",
    "price2": "price2_qr",
    "p2": "price2_qr",
    "price3": "price3_qr",
    "p3": "price3_qr",
    "price4": "price4_qr",
    "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count",
    "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price",
    "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count",
    "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price",
    "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr",
    "aP": "action_price_qr",
    "actionCode": "action_code_qr",
    "aC": "action_code_qr",
}

QR_FIELDS = [
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

PRICE_FIELDS = {
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}


@dataclass(frozen=True)
class FullTagCrop:
    path: Path
    track_id: str
    rank: int
    timestamp_ms: int


@dataclass(frozen=True)
class ImageCandidate:
    name: str
    image: Any


@dataclass(frozen=True)
class DecodeCandidate:
    track_id: str
    rank: int
    timestamp_ms: int
    decoder: str
    candidate_name: str
    preprocess: str
    image_path: str
    payload: str
    parsed: dict[str, str]


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_price(value: Any) -> str:
    match = re.search(r"\d+(?:[.,]\d+)?", str(value or ""))
    if not match:
        return ""
    number = float(match.group(0).replace(",", "."))
    if not math.isfinite(number) or number <= 0:
        return ""
    return f"{number:.2f}"


def normalize_int(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def normalize_barcode(value: Any) -> str:
    digits = normalize_int(value)
    return digits if 8 <= len(digits) <= 14 else ""


def parse_qr_payload(payload: str) -> dict[str, str]:
    payload = str(payload or "").strip()
    if not payload:
        return {}
    parsed: dict[str, str] = {}
    parsed_url = urlparse(payload)
    query = parsed_url.query if parsed_url.query else payload
    query = query.replace(";", "&").replace("|", "&").replace("\n", "&")
    for key, values in parse_qs(query, keep_blank_values=True).items():
        mapped_key = QR_FIELD_ALIASES.get(key)
        if mapped_key and values:
            parsed[mapped_key] = values[0]
    for key, value in re.findall(r"([A-Za-z0-9_]+)\s*[:=]\s*([^,&;|\s]+)", payload):
        mapped_key = QR_FIELD_ALIASES.get(key)
        if mapped_key:
            parsed[mapped_key] = value

    normalized: dict[str, str] = {}
    for field, value in parsed.items():
        if field in PRICE_FIELDS:
            cleaned = normalize_price(value)
        elif field == "qr_code_barcode":
            cleaned = normalize_barcode(value)
        elif field in {"wholesale_level_1_count", "wholesale_level_2_count", "action_code_qr"}:
            cleaned = normalize_int(value)
        else:
            cleaned = str(value or "").strip()
        if cleaned:
            normalized[field] = cleaned
    return normalized


def is_valid_lenta_payload(payload: str, parsed: dict[str, str]) -> bool:
    if not parsed.get("qr_code_barcode") or not parsed.get("price1_qr"):
        return False
    has_known_key = bool(re.search(r"(?:^|[?&;|])(?:barcode|b|price1|p1|price2|p2|price4|p4|aP|actionPrice)=", payload, re.I))
    has_long_shape = "barcode=" in payload and "price1=" in payload
    has_short_shape = bool(re.search(r"(?:^|[?&;|])b=", payload) and re.search(r"(?:^|[?&;|])p1=", payload))
    return bool(has_known_key and (has_long_shape or has_short_shape))


def load_image(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image)
    if ok:
        path.write_bytes(encoded.tobytes())


def crop_relative(image, x0: float, y0: float, x1: float, y1: float):
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(round(width * x0))))
    top = max(0, min(height - 1, int(round(height * y0))))
    right = max(left + 1, min(width, int(round(width * x1))))
    bottom = max(top + 1, min(height, int(round(height * y1))))
    return image[top:bottom, left:right].copy()


def detector_localized_candidates(name: str, image, pad_ratio: float = 0.16) -> list[ImageCandidate]:
    import cv2
    import numpy as np

    detector = cv2.QRCodeDetector()
    height, width = image.shape[:2]
    localized: list[ImageCandidate] = []
    for scale in (1, 2):
        probe = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            found, points = detector.detect(probe)
        except cv2.error:
            found, points = False, None
        if not found or points is None:
            continue
        probe_pts = points.reshape(-1, 2).astype(np.float32)
        side = 256
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
        try:
            matrix = cv2.getPerspectiveTransform(probe_pts, dst)
            warped = cv2.warpPerspective(probe, matrix, (side, side), borderValue=(255, 255, 255))
            localized.append(ImageCandidate(f"{name}_warp_s{scale}", warped))
        except cv2.error:
            pass
        pts = probe_pts / float(scale)
        x_min = max(0.0, float(pts[:, 0].min()))
        x_max = min(float(width), float(pts[:, 0].max()))
        y_min = max(0.0, float(pts[:, 1].min()))
        y_max = min(float(height), float(pts[:, 1].max()))
        pad = max(1.0, x_max - x_min, y_max - y_min) * pad_ratio
        left = max(0, int(math.floor(x_min - pad)))
        top = max(0, int(math.floor(y_min - pad)))
        right = min(width, int(math.ceil(x_max + pad)))
        bottom = min(height, int(math.ceil(y_max + pad)))
        if right > left and bottom > top:
            localized.append(ImageCandidate(f"{name}_detector_s{scale}", image[top:bottom, left:right].copy()))
    return localized


def build_full_tag_candidates(image) -> list[ImageCandidate]:
    broad = [
        ImageCandidate("upper_right", crop_relative(image, 0.56, 0.00, 1.00, 0.48)),
        ImageCandidate("upper_right_wide", crop_relative(image, 0.52, 0.00, 1.00, 0.54)),
        ImageCandidate("full_detector", image),
    ]
    ordered: list[ImageCandidate] = []
    for candidate in broad:
        ordered.extend(detector_localized_candidates(candidate.name, candidate.image))
        ordered.append(candidate)
    return ordered


def add_quiet_zone(image, border: int = 28):
    import cv2

    value = (255, 255, 255) if len(image.shape) == 3 else 255
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=value)


def direct_decode_variants(image) -> list[ImageCandidate]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    _, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_clahe = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants = [
        ImageCandidate("gray", gray),
        ImageCandidate("color", image),
        ImageCandidate("clahe", clahe),
        ImageCandidate("otsu_gray", otsu_gray),
        ImageCandidate("otsu_clahe", otsu_clahe),
    ]
    for block_size in (21, 31, 41):
        variants.append(
            ImageCandidate(
                f"adaptive_gaussian_{block_size}",
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5),
            )
        )
    out: list[ImageCandidate] = []
    for variant in variants:
        out.append(variant)
        out.append(ImageCandidate(f"{variant.name}_quiet", add_quiet_zone(variant.image)))
    return out


def decode_opencv(image) -> list[str]:
    import cv2

    detector = cv2.QRCodeDetector()
    payloads: list[str] = []
    try:
        text, _, _ = detector.detectAndDecode(image)
    except cv2.error:
        text = ""
    if text:
        payloads.append(str(text).strip())
    try:
        ok, decoded_info, _, _ = detector.detectAndDecodeMulti(image)
    except cv2.error:
        ok, decoded_info = False, []
    if ok:
        payloads.extend(str(text or "").strip() for text in decoded_info)
    return list(dict.fromkeys([payload for payload in payloads if payload]))


def decode_zxing(image) -> list[str]:
    import zxingcpp

    try:
        results = zxingcpp.read_barcodes(image, formats=zxingcpp.BarcodeFormat.QRCode)
    except TypeError:
        results = zxingcpp.read_barcodes(image)
    payloads = [str(getattr(result, "text", "") or "").strip() for result in results]
    return list(dict.fromkeys([payload for payload in payloads if payload]))


def decode_all_backends(image) -> list[tuple[str, str]]:
    decoded: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for decoder_name, decoder in (("opencv", decode_opencv), ("zxing", decode_zxing)):
        for payload in decoder(image):
            key = (decoder_name, payload)
            if key not in seen:
                decoded.append(key)
                seen.add(key)
    return decoded


def decode_one(crop: FullTagCrop, success_dir: Path) -> tuple[list[DecodeCandidate], list[dict[str, Any]]]:
    image = load_image(crop.path)
    candidates: list[DecodeCandidate] = []
    attempts: list[dict[str, Any]] = []
    if image is None:
        return candidates, attempts
    for candidate_image in build_full_tag_candidates(image):
        for variant in direct_decode_variants(candidate_image.image):
            error = ""
            try:
                decoded_payloads = decode_all_backends(variant.image)
            except Exception as exc:
                decoded_payloads = []
                error = f"{type(exc).__name__}: {exc}"
            valid_payloads = 0
            decoders = Counter(decoder_name for decoder_name, _ in decoded_payloads)
            for decoder_name, payload in decoded_payloads:
                parsed = parse_qr_payload(payload)
                if not is_valid_lenta_payload(payload, parsed):
                    continue
                valid_payloads += 1
                image_path = success_dir / f"track_{int(crop.track_id):04d}_{candidate_image.name}_{variant.name}_{decoder_name}.jpg"
                if not image_path.exists():
                    save_image(image_path, candidate_image.image)
                candidates.append(
                    DecodeCandidate(
                        track_id=crop.track_id,
                        rank=crop.rank,
                        timestamp_ms=crop.timestamp_ms,
                        decoder=decoder_name,
                        candidate_name=candidate_image.name,
                        preprocess=variant.name,
                        image_path=str(image_path),
                        payload=payload,
                        parsed=parsed,
                    )
                )
            attempts.append(
                {
                    "track_id": crop.track_id,
                    "rank": crop.rank,
                    "timestamp_ms": crop.timestamp_ms,
                    "candidate_name": candidate_image.name,
                    "preprocess": variant.name,
                    "decoder": ",".join(f"{name}:{count}" for name, count in sorted(decoders.items())),
                    "payload_count": len(decoded_payloads),
                    "valid_payload_count": valid_payloads,
                    "error": error,
                    "image_path": str(crop.path),
                }
            )
            if valid_payloads:
                return candidates, attempts
    return candidates, attempts


def build_full_tags(manifest_rows: list[dict[str, str]]) -> list[FullTagCrop]:
    by_track: dict[str, FullTagCrop] = {}
    for row in manifest_rows:
        if str(row.get("rank", "")) != "1":
            continue
        track_id = str(row.get("track_id", "")).strip()
        full_tag = Path(str(row.get("full_tag", "")))
        if not track_id or not full_tag.exists() or track_id in by_track:
            continue
        by_track[track_id] = FullTagCrop(
            path=full_tag,
            track_id=track_id,
            rank=1,
            timestamp_ms=int(float(row.get("timestamp_ms") or 0)),
        )
    return list(by_track.values())


def build_full_tags_from_dir(full_tags_dir: Path) -> list[FullTagCrop]:
    crops: list[FullTagCrop] = []
    pattern = re.compile(r"track_(\d+)_rank_(\d+)_ts_(\d+)_full\.jpg$", re.I)
    for path in sorted(full_tags_dir.glob("*_rank_*_full.jpg")):
        match = pattern.search(path.name)
        if not match:
            continue
        track_id, rank, timestamp_ms = match.groups()
        if int(rank) == 1:
            crops.append(FullTagCrop(path=path, track_id=str(int(track_id)), rank=int(rank), timestamp_ms=int(timestamp_ms)))
    return crops


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    def coord(name: str) -> str:
        try:
            return f"{float(str(row.get(name, '')).replace(',', '.')):.2f}"
        except ValueError:
            return str(row.get(name, "")).strip()

    return (
        Path(str(row.get("filename", ""))).name,
        str(row.get("frame_timestamp", "")).strip(),
        coord("x_min"),
        coord("y_min"),
        coord("x_max"),
        coord("y_max"),
    )


def build_track_lookup(manifest_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], str]:
    lookup: dict[tuple[str, str, str, str, str, str], str] = {}
    for row in manifest_rows:
        track_id = str(row.get("track_id", "")).strip()
        if track_id:
            lookup[row_key(row)] = track_id
    return lookup


def candidate_quality(candidate: DecodeCandidate) -> tuple[int, int, int]:
    return (
        sum(1 for field in ("qr_code_barcode", "price1_qr", "price2_qr", "price4_qr") if candidate.parsed.get(field)),
        len(candidate.parsed),
        -candidate.rank,
    )


def set_value(row: dict[str, str], changes: list[dict[str, str]], row_num: int, track_id: str, field: str, value: str, source: str) -> None:
    if field not in row or not value:
        return
    old = row.get(field, "")
    if str(old) == str(value):
        return
    row[field] = value
    changes.append({"row": str(row_num), "track_id": track_id, "field": field, "old": old, "new": value, "source": source})


def apply_qr_priority(row: dict[str, str], row_num: int, track_id: str, candidate: DecodeCandidate | None) -> tuple[dict[str, str], list[dict[str, str]]]:
    out = dict(row)
    changes: list[dict[str, str]] = []
    if candidate is None:
        return out, changes
    source = f"qr_priority:full_tag_direct:{candidate.decoder}:{candidate.candidate_name}:{candidate.preprocess}"
    for field in QR_FIELDS:
        set_value(out, changes, row_num, track_id, field, candidate.parsed.get(field, ""), source)
    set_value(out, changes, row_num, track_id, "barcode", candidate.parsed.get("qr_code_barcode", ""), source)
    set_value(out, changes, row_num, track_id, "price_default", candidate.parsed.get("price1_qr", ""), source)
    set_value(out, changes, row_num, track_id, "price_card", candidate.parsed.get("price4_qr", ""), source)
    return out, changes


def rel_url(path: str, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_html_report(
    path: Path,
    summary: dict[str, Any],
    selected_by_track: dict[str, DecodeCandidate],
    candidates: list[DecodeCandidate],
    changes: list[dict[str, str]],
) -> None:
    selected_payloads = {track_id: candidate.payload for track_id, candidate in selected_by_track.items()}
    candidate_rows = []
    for candidate in sorted(candidates, key=lambda item: (int(item.track_id), item.candidate_name, item.preprocess, item.decoder)):
        selected = selected_payloads.get(candidate.track_id) == candidate.payload
        parsed = ", ".join(f"{key}={value}" for key, value in candidate.parsed.items())
        candidate_rows.append(
            "<tr>"
            f"<td>{escape(candidate.track_id)}</td>"
            f"<td>{'yes' if selected else ''}</td>"
            f"<td>{escape(candidate.decoder)}</td>"
            f"<td>{escape(candidate.candidate_name)}</td>"
            f"<td>{escape(candidate.preprocess)}</td>"
            f"<td><code>{escape(candidate.payload)}</code></td>"
            f"<td>{escape(parsed)}</td>"
            f"<td><img src=\"{escape(rel_url(candidate.image_path, path.parent))}\" alt=\"qr crop\"></td>"
            "</tr>"
        )
    change_rows = []
    for change in changes:
        change_rows.append(
            "<tr>"
            f"<td>{escape(change.get('row', ''))}</td>"
            f"<td>{escape(change.get('track_id', ''))}</td>"
            f"<td>{escape(change.get('field', ''))}</td>"
            f"<td>{escape(change.get('old', ''))}</td>"
            f"<td>{escape(change.get('new', ''))}</td>"
            f"<td>{escape(change.get('source', ''))}</td>"
            "</tr>"
        )
    summary_items = "\n".join(
        f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>"
        for key, value in summary.items()
        if key != "html_report"
    )
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>QR priority debug</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 0 0 28px; }}
    dl {{ display: grid; grid-template-columns: 260px 1fr; gap: 6px 12px; }}
    dt {{ font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dee8; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f1f5f9; text-align: left; position: sticky; top: 0; }}
    code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    img {{ max-width: 180px; max-height: 140px; object-fit: contain; background: #fff; }}
  </style>
</head>
<body>
  <h1>QR priority debug</h1>
  <section>
    <h2>Summary</h2>
    <dl>{summary_items}</dl>
  </section>
  <section>
    <h2>Validated QR candidates</h2>
    <table>
      <thead>
        <tr><th>track</th><th>selected</th><th>decoder</th><th>candidate</th><th>variant</th><th>payload</th><th>parsed</th><th>crop</th></tr>
      </thead>
      <tbody>{''.join(candidate_rows)}</tbody>
    </table>
  </section>
  <section>
    <h2>Applied changes</h2>
    <table>
      <thead>
        <tr><th>row</th><th>track</th><th>field</th><th>old</th><th>new</th><th>source</th></tr>
      </thead>
      <tbody>{''.join(change_rows)}</tbody>
    </table>
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode QR from full-tag crops and apply QR-first values to final submission CSV.")
    parser.add_argument("--submission-csv", type=Path, required=True)
    parser.add_argument("--zones-manifest", type=Path)
    parser.add_argument("--full-tags-dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_rows(args.submission_csv)
    manifest_rows: list[dict[str, str]] = []
    if args.zones_manifest and args.zones_manifest.exists():
        manifest_rows, _ = read_rows(args.zones_manifest)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    success_dir = args.debug_dir / "fulltag_success_crops"
    full_tags = build_full_tags(manifest_rows)
    if not full_tags:
        full_tags_dir = args.full_tags_dir
        if full_tags_dir is None and args.zones_manifest is not None:
            candidate_dir = args.zones_manifest.parent / "full_tags"
            full_tags_dir = candidate_dir if candidate_dir.exists() else None
        if full_tags_dir is not None and full_tags_dir.exists():
            full_tags = build_full_tags_from_dir(full_tags_dir)

    candidates: list[DecodeCandidate] = []
    attempts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(decode_one, crop, success_dir) for crop in full_tags]
        for future in as_completed(futures):
            crop_candidates, crop_attempts = future.result()
            candidates.extend(crop_candidates)
            attempts.extend(crop_attempts)

    candidates_by_track: dict[str, list[DecodeCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_track[candidate.track_id].append(candidate)
    selected_by_track: dict[str, DecodeCandidate] = {}
    conflict_by_track: dict[str, int] = {}
    for track_id, track_candidates in candidates_by_track.items():
        payload_counts = Counter(candidate.payload for candidate in track_candidates)
        conflict_by_track[track_id] = len(payload_counts)
        selected_by_track[track_id] = max(track_candidates, key=lambda candidate: (payload_counts[candidate.payload], candidate_quality(candidate)))

    track_lookup = build_track_lookup(manifest_rows)
    output_rows: list[dict[str, str]] = []
    all_changes: list[dict[str, str]] = []
    matched_rows = 0
    for row_num, row in enumerate(rows, start=1):
        track_id = row.get("track_id", "").strip() or track_lookup.get(row_key(row), "")
        if track_id:
            matched_rows += 1
        new_row, changes = apply_qr_priority(row, row_num, track_id, selected_by_track.get(track_id))
        output_rows.append(new_row)
        all_changes.extend(changes)

    write_rows(args.output_csv, output_rows, fieldnames)
    candidate_rows = []
    for candidate in candidates:
        row = {
            "track_id": candidate.track_id,
            "rank": candidate.rank,
            "timestamp_ms": candidate.timestamp_ms,
            "decoder": candidate.decoder,
            "candidate_name": candidate.candidate_name,
            "preprocess": candidate.preprocess,
            "payload": candidate.payload,
            "image_path": candidate.image_path,
        }
        row.update({field: candidate.parsed.get(field, "") for field in QR_FIELDS})
        candidate_rows.append(row)
    write_rows(args.debug_dir / "qr_decode_candidates.csv", candidate_rows, ["track_id", "rank", "timestamp_ms", "decoder", "candidate_name", "preprocess", "payload", "image_path", *QR_FIELDS])
    write_rows(args.debug_dir / "qr_decode_attempts.csv", attempts, ["track_id", "rank", "timestamp_ms", "candidate_name", "preprocess", "decoder", "payload_count", "valid_payload_count", "error", "image_path"])
    write_rows(args.debug_dir / "qr_priority_changes.csv", all_changes, ["row", "track_id", "field", "old", "new", "source"])

    summary = {
        "submission_csv": str(args.submission_csv),
        "output_csv": str(args.output_csv),
        "zones_manifest": str(args.zones_manifest or ""),
        "full_tags_dir": str(args.full_tags_dir or ""),
        "html_report": str(args.debug_dir / "qr_priority_debug.html"),
        "full_tag_rank1_count": len(full_tags),
        "rows": len(rows),
        "rows_matched_to_tracks": matched_rows,
        "tracks_with_qr": len(selected_by_track),
        "rows_changed_by_qr": len({change["row"] for change in all_changes}),
        "changed_cells": len(all_changes),
        "tracks_with_conflicting_payloads": sum(1 for count in conflict_by_track.values() if count > 1),
    }
    write_html_report(args.debug_dir / "qr_priority_debug.html", summary, selected_by_track, candidates, all_changes)
    (args.debug_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
