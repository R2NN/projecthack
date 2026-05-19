from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any


OCR_DIR = "ocr_final_quality_core_fast_fixed"
ZONES_DIR = "ocr_zones_core_fixed"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def surface_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\r", " ").replace("\n", " ")).strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", surface_text(value)).lower().replace("\u0451", "\u0435")
    text = re.sub(r"[^0-9a-z\u0400-\u04ff]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def parse_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("score", "") or 0.0)
    except ValueError:
        return 0.0


def read_candidate_rows(path: Path, label: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_rows(path):
        track_id = surface_text(row.get("track_id", ""))
        text = surface_text(row.get("value") or row.get("raw_text"))
        if not track_id or not text:
            continue
        copied: dict[str, Any] = dict(row)
        copied["ensemble_source"] = label
        copied["ensemble_original_score"] = row.get("score", "")
        copied["ensemble_original_engine"] = row.get("engine", "")
        copied["source"] = f"{label}:{row.get('source', '')}"
        copied["engine"] = f"ensemble_{label}"
        copied["field"] = copied.get("field") or "product_name"
        grouped.setdefault(track_id, []).append(copied)
    return grouped


def dedupe_ranked(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=parse_score, reverse=True)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        key = normalize_text(row.get("value") or row.get("raw_text"))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def interleave_by_track(
    primary: dict[str, list[dict[str, Any]]],
    secondary: dict[str, list[dict[str, Any]]],
    primary_label: str,
    secondary_label: str,
    per_engine_top: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    track_ids = sorted({*primary.keys(), *secondary.keys()}, key=lambda value: int(value) if value.isdigit() else value)
    for track_id in track_ids:
        lists = {
            primary_label: dedupe_ranked(primary.get(track_id, []), per_engine_top),
            secondary_label: dedupe_ranked(secondary.get(track_id, []), per_engine_top),
        }
        track_rows: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for index in range(per_engine_top):
            for label in (primary_label, secondary_label):
                if index >= len(lists[label]):
                    continue
                row = dict(lists[label][index])
                key = normalize_text(row.get("value") or row.get("raw_text"))
                if not key or key in seen_texts:
                    continue
                seen_texts.add(key)
                track_rows.append(row)
                if len(track_rows) >= pool_size:
                    break
            if len(track_rows) >= pool_size:
                break
        for order, row in enumerate(track_rows, start=1):
            row["ensemble_rank"] = order
            row["score"] = f"{1000.0 - order:.4f}"
            combined.append(row)
    return combined


def copy_small_ocr_files(primary_ocr_dir: Path, output_ocr_dir: Path) -> None:
    if output_ocr_dir.exists():
        shutil.rmtree(output_ocr_dir)
    output_ocr_dir.mkdir(parents=True, exist_ok=True)
    for path in primary_ocr_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_ocr_dir / path.name)


def copy_minimal_zones(zones_manifest: Path, output_run_dir: Path) -> None:
    target = output_run_dir / ZONES_DIR
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zones_manifest, target / "ocr_zones_manifest.csv")
    ocr_manifest = zones_manifest.parent / "ocr_manifest.csv"
    if ocr_manifest.exists():
        shutil.copy2(ocr_manifest, target / "ocr_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a product_name OCR ensemble run from two OCR candidate sets.")
    parser.add_argument("--primary-run-dir", type=Path, required=True)
    parser.add_argument("--secondary-run-dir", type=Path, required=True)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--primary-label", default="tesseract")
    parser.add_argument("--secondary-label", default="paddle")
    parser.add_argument("--per-engine-top", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_ocr_dir = args.primary_run_dir / OCR_DIR
    secondary_ocr_dir = args.secondary_run_dir / OCR_DIR
    output_ocr_dir = args.output_run_dir / OCR_DIR

    primary_candidates = primary_ocr_dir / "product_name_line_candidates.csv"
    secondary_candidates = secondary_ocr_dir / "product_name_line_candidates.csv"
    if not primary_candidates.exists():
        raise FileNotFoundError(primary_candidates)
    if not secondary_candidates.exists():
        raise FileNotFoundError(secondary_candidates)
    if not args.zones_manifest.exists():
        raise FileNotFoundError(args.zones_manifest)

    args.output_run_dir.mkdir(parents=True, exist_ok=True)
    copy_small_ocr_files(primary_ocr_dir, output_ocr_dir)
    copy_minimal_zones(args.zones_manifest, args.output_run_dir)

    primary_grouped = read_candidate_rows(primary_candidates, args.primary_label)
    secondary_grouped = read_candidate_rows(secondary_candidates, args.secondary_label)
    ensemble_candidates = interleave_by_track(
        primary_grouped,
        secondary_grouped,
        args.primary_label,
        args.secondary_label,
        args.per_engine_top,
        args.pool_size,
    )
    write_rows(output_ocr_dir / "product_name_line_candidates.csv", ensemble_candidates)

    summary = {
        "primary_run_dir": str(args.primary_run_dir),
        "secondary_run_dir": str(args.secondary_run_dir),
        "tracks_primary": len(primary_grouped),
        "tracks_secondary": len(secondary_grouped),
        "tracks_ensemble": len({row.get("track_id", "") for row in ensemble_candidates}),
        "primary_candidate_rows": sum(len(rows) for rows in primary_grouped.values()),
        "secondary_candidate_rows": sum(len(rows) for rows in secondary_grouped.values()),
        "ensemble_candidate_rows": len(ensemble_candidates),
        "per_engine_top": args.per_engine_top,
        "pool_size": args.pool_size,
    }
    (output_ocr_dir / "product_name_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
