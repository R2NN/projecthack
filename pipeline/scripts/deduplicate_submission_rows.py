"""Conservative postprocess deduplication for final submission rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


OCR_DIR = "ocr_final_quality_core_fast_fixed"
SUBMISSION_CSV = "ocr_aggregated_submission_product_lines.csv"
RECOVERY_CSV = "product_name_catalog_recovery.csv"
REPORT_CSV = "ocr_aggregated_submission_product_lines_dedup_report.csv"
SUMMARY_JSON = "ocr_aggregated_submission_product_lines_dedup_summary.json"

STOP_TOKENS = {
    "\u0431",
    "\u0433",
    "\u043c\u0435\u0434",
    "\u043c\u0435\u0434\u0430",
    "\u043d\u0435\u0442",
    "\u0440\u043e\u0441\u0441\u0438\u044f",
    "\u0441\u0442",
    "\u043a\u043e\u043d\u0444\u0438\u0442\u044e\u0440",
    "\u0434\u0435\u0441\u0435\u0440\u0442",
    "\u0444\u0440\u0443\u043a\u0442\u043e\u0432\u044b\u0439",
    "\u044d\u043a\u0441\u0442\u0440\u0430",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--recovery-csv", type=Path)
    parser.add_argument("--tracking-csv", type=Path)
    parser.add_argument("--report-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--barcode-pair-only", action="store_true", default=True)
    parser.add_argument("--max-exact-barcode-group-size", type=int, default=2)
    parser.add_argument("--same-price-iou", type=float, default=0.25)
    parser.add_argument("--same-price-center-distance", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_value(value: Any) -> str:
    text = str(value or "").strip()
    if text.casefold() in {"", "nan", "none", "null", "\u043d\u0435\u0442"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_barcode(row: dict[str, Any]) -> str:
    raw = clean_value(row.get("barcode")) or clean_value(row.get("qr_code_barcode"))
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits if len(digits) >= 6 else ""


def normalize_price(row: dict[str, Any]) -> str:
    raw = clean_value(row.get("price_card")) or clean_value(row.get("price4_qr"))
    if not raw:
        return ""
    try:
        value = Decimal(raw.replace(",", ".")).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return ""
    return str(value)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("\u0451", "\u0435")
    chars = [ch if ch.isalnum() else " " for ch in text]
    return " ".join("".join(chars).split())


def distinctive_tokens(row: dict[str, Any]) -> set[str]:
    tokens = normalize_text(row.get("product_name")).split()
    return {token for token in tokens if len(token) >= 4 and token not in STOP_TOKENS}


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def row_box(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values = [parse_float(row.get(key)) for key in ("x_min", "y_min", "x_max", "y_max")]
    if any(value is None for value in values):
        return None
    x_min, y_min, x_max, y_max = values
    return float(x_min), float(y_min), float(x_max), float(y_max)


def box_iou(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = first_area + second_area - inter
    return inter / denom if denom else 0.0


def center_distance(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return math.inf
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return math.hypot(((ax1 + ax2) - (bx1 + bx2)) / 2.0, ((ay1 + ay2) - (by1 + by2)) / 2.0)


def load_recovery_by_track(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not path.exists():
        return {}, []
    rows = read_rows(path)
    by_track = {clean_value(row.get("track_id")): row for row in rows if clean_value(row.get("track_id"))}
    track_ids = [clean_value(row.get("track_id")) for row in rows]
    return by_track, track_ids


def load_tracking_by_track(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {clean_value(row.get("track_id")): row for row in read_rows(path) if clean_value(row.get("track_id"))}


def float_field(row: dict[str, Any], key: str) -> float:
    value = parse_float(row.get(key))
    return value or 0.0


def is_recovery_accepted(track_id: str, recovery_by_track: dict[str, dict[str, str]]) -> bool:
    return clean_value(recovery_by_track.get(track_id, {}).get("accepted")) == "1"


def row_identity_score(
    row: dict[str, Any],
    recovery_by_track: dict[str, dict[str, str]],
    tracking_by_track: dict[str, dict[str, str]],
) -> float:
    track_id = clean_value(row.get("_dedup_track_id"))
    recovery = recovery_by_track.get(track_id, {})
    tracking = tracking_by_track.get(track_id, {})
    filled_fields = sum(
        1
        for key in ("product_name", "price_default", "price_card", "discount_amount", "barcode", "qr_code_barcode")
        if clean_value(row.get(key))
    )
    return (
        (5.0 if is_recovery_accepted(track_id, recovery_by_track) else 0.0)
        + (2.0 if normalize_barcode(row) else 0.0)
        + min(len(distinctive_tokens(row)), 8) * 0.12
        + float_field(recovery, "score")
        + float_field(recovery, "margin") * 0.05
        + float_field(tracking, "quality_score")
        + float_field(tracking, "crop_score") * 0.5
        + float_field(tracking, "ocr_score") * 0.25
        + min(float_field(tracking, "track_quality_candidate_count"), 10.0) * 0.04
        + filled_fields * 0.02
    )


def has_strong_identity(row: dict[str, Any], recovery_by_track: dict[str, dict[str, str]]) -> bool:
    track_id = clean_value(row.get("_dedup_track_id"))
    return bool(normalize_barcode(row) or is_recovery_accepted(track_id, recovery_by_track))


def mark_duplicate(
    drops: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    drop_index: int,
    keep_index: int,
    rule: str,
    details: dict[str, Any],
    recovery_by_track: dict[str, dict[str, str]],
    tracking_by_track: dict[str, dict[str, str]],
) -> None:
    if drop_index in drops:
        return
    drop_row = rows[drop_index]
    keep_row = rows[keep_index]
    drops[drop_index] = {
        "row_index": drop_index,
        "track_id": clean_value(drop_row.get("_dedup_track_id")),
        "kept_row_index": keep_index,
        "kept_track_id": clean_value(keep_row.get("_dedup_track_id")),
        "rule": rule,
        "barcode": normalize_barcode(drop_row),
        "price_card": normalize_price(drop_row),
        "product_name": clean_value(drop_row.get("product_name")),
        "score": f"{row_identity_score(drop_row, recovery_by_track, tracking_by_track):.6f}",
        "kept_score": f"{row_identity_score(keep_row, recovery_by_track, tracking_by_track):.6f}",
        **details,
    }


def find_duplicates(
    rows: list[dict[str, Any]],
    recovery_by_track: dict[str, dict[str, str]],
    tracking_by_track: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    drops: dict[int, dict[str, Any]] = {}
    skipped_groups: list[dict[str, Any]] = []
    by_barcode: dict[str, list[int]] = defaultdict(list)

    for index, row in enumerate(rows):
        barcode = normalize_barcode(row)
        if barcode:
            by_barcode[barcode].append(index)

    for barcode, indexes in by_barcode.items():
        if len(indexes) > args.max_exact_barcode_group_size:
            skipped_groups.append(
                {
                    "rule": "skipped_large_barcode_group",
                    "barcode": barcode,
                    "size": len(indexes),
                    "track_ids": [clean_value(rows[index].get("_dedup_track_id")) for index in indexes],
                }
            )
            continue
        if len(indexes) < 2:
            continue
        keep_index = max(
            indexes,
            key=lambda index: row_identity_score(rows[index], recovery_by_track, tracking_by_track),
        )
        for index in indexes:
            if index == keep_index:
                continue
            mark_duplicate(
                drops,
                rows,
                index,
                keep_index,
                "exact_barcode_pair",
                {"matched_on": barcode},
                recovery_by_track,
                tracking_by_track,
            )

    for first_index, first_row in enumerate(rows):
        if first_index in drops:
            continue
        first_price = normalize_price(first_row)
        if not first_price:
            continue
        for second_index in range(first_index + 1, len(rows)):
            if second_index in drops:
                continue
            second_row = rows[second_index]
            if first_price != normalize_price(second_row):
                continue
            if normalize_barcode(first_row) and normalize_barcode(second_row):
                continue
            first_strong = has_strong_identity(first_row, recovery_by_track)
            second_strong = has_strong_identity(second_row, recovery_by_track)
            if first_strong == second_strong:
                continue
            overlap_iou = box_iou(row_box(first_row), row_box(second_row))
            distance = center_distance(row_box(first_row), row_box(second_row))
            if overlap_iou < args.same_price_iou and distance > args.same_price_center_distance:
                continue
            keep_index, drop_index = (first_index, second_index) if first_strong else (second_index, first_index)
            common_tokens = distinctive_tokens(rows[keep_index]) & distinctive_tokens(rows[drop_index])
            if not common_tokens:
                continue
            mark_duplicate(
                drops,
                rows,
                drop_index,
                keep_index,
                "same_price_spatial_identity",
                {
                    "matched_on": first_price,
                    "iou": f"{overlap_iou:.6f}",
                    "center_distance": f"{distance:.3f}",
                    "common_tokens": "|".join(sorted(common_tokens)),
                },
                recovery_by_track,
                tracking_by_track,
            )

    return drops, skipped_groups


def default_tracking_csv(run_dir: Path) -> Path:
    return run_dir.parent / "base" / "tracking" / "best_per_track.csv"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    ocr_dir = run_dir / OCR_DIR
    input_csv = args.input_csv or ocr_dir / SUBMISSION_CSV
    output_csv = args.output_csv or input_csv
    recovery_csv = args.recovery_csv or ocr_dir / RECOVERY_CSV
    tracking_csv = args.tracking_csv or default_tracking_csv(run_dir)
    report_csv = args.report_csv or ocr_dir / REPORT_CSV
    summary_json = args.summary_json or ocr_dir / SUMMARY_JSON

    rows = read_rows(input_csv)
    fieldnames = list(rows[0].keys()) if rows else []
    recovery_by_track, recovery_track_ids = load_recovery_by_track(recovery_csv)
    tracking_by_track = load_tracking_by_track(tracking_csv)
    backup_csv = input_csv.with_name(input_csv.stem + "_before_dedup.csv")

    if output_csv == input_csv and backup_csv.exists() and recovery_track_ids and len(rows) < len(recovery_track_ids):
        summary = {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "rows_before": len(rows),
            "rows_after": len(rows),
            "dropped_rows": 0,
            "already_deduplicated": True,
            "dry_run": bool(args.dry_run),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    for index, row in enumerate(rows):
        if "track_id" in row and clean_value(row.get("track_id")):
            track_id = clean_value(row.get("track_id"))
        elif index < len(recovery_track_ids):
            track_id = recovery_track_ids[index]
        else:
            track_id = ""
        row["_dedup_track_id"] = track_id

    drops, skipped_groups = find_duplicates(rows, recovery_by_track, tracking_by_track, args)
    kept_rows = [{key: value for key, value in row.items() if key != "_dedup_track_id"} for index, row in enumerate(rows) if index not in drops]
    report_rows = list(drops.values())

    report_fieldnames = [
        "row_index",
        "track_id",
        "kept_row_index",
        "kept_track_id",
        "rule",
        "matched_on",
        "barcode",
        "price_card",
        "product_name",
        "score",
        "kept_score",
        "iou",
        "center_distance",
        "common_tokens",
    ]
    write_rows(report_csv, report_rows, report_fieldnames)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "tracking_csv": str(tracking_csv) if tracking_csv.exists() else "",
        "rows_before": len(rows),
        "rows_after": len(kept_rows),
        "dropped_rows": len(drops),
        "dropped_by_rule": dict(sorted((rule, sum(1 for row in report_rows if row["rule"] == rule)) for rule in {row["rule"] for row in report_rows})),
        "skipped_groups": skipped_groups,
        "dry_run": bool(args.dry_run),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.dry_run:
        if output_csv == input_csv and input_csv.exists():
            shutil.copy2(input_csv, backup_csv)
        write_rows(output_csv, kept_rows, fieldnames)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
