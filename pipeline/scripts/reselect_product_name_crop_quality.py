"""Reselect Tesseract product_name by product crop image quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_product_name_tesseract as rpt
from reselect_product_name_tesseract import candidate_from_row


@dataclass(frozen=True)
class CropQuality:
    video_id: str
    filename: str
    track_id: str
    rank: int
    timestamp_ms: str
    frame_timestamp: str
    variant: str
    image_path: str
    score: float
    sharpness: float
    contrast: float
    text_density_score: float
    text_height_score: float
    edge_penalty: float
    border_penalty: float
    foreground_density: float
    median_component_height: float
    width: int
    height: int


def sigmoid_norm(value: float, center: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def triangle_score(value: float, low: float, target: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == target:
        return 1.0
    if value < target:
        return (value - low) / max(1e-6, target - low)
    return (high - value) / max(1e-6, high - target)


def image_foreground_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        11,
    )
    return adaptive


def compute_crop_quality(row: dict[str, str], variant: str, path_value: str) -> CropQuality | None:
    image = rpt.read_image(path_value)
    if image is None:
        return None
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_pad_y = max(2, int(round(height * 0.03)))
    inner_pad_x = max(2, int(round(width * 0.03)))
    inner = gray[inner_pad_y : max(inner_pad_y + 1, height - inner_pad_y), inner_pad_x : max(inner_pad_x + 1, width - inner_pad_x)]

    sharp_raw = float(cv2.Laplacian(inner, cv2.CV_64F).var())
    contrast_raw = float(inner.std())
    sharpness = sigmoid_norm(math.log1p(sharp_raw), center=5.1, scale=0.65)
    contrast = triangle_score(contrast_raw, low=12.0, target=48.0, high=92.0)

    mask = image_foreground_mask(gray)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    text_mask = np.zeros_like(mask)
    component_heights: list[int] = []
    large_border_area = 0
    for index in range(1, component_count):
        x, y, component_w, component_h, area = [int(value) for value in stats[index]]
        if area < 8:
            continue
        border_like = (
            (component_w > 0.58 * width and component_h < 0.12 * height)
            or (component_h > 0.48 * height and component_w < 0.12 * width)
            or (y < 2 and component_w > 0.18 * width)
            or (x < 2 and component_h > 0.18 * height)
        )
        if border_like:
            large_border_area += area
            continue
        if component_h < max(4, int(0.012 * height)) or component_h > 0.34 * height:
            continue
        if component_w < 2 or area / max(1, component_w * component_h) < 0.06:
            continue
        text_mask[labels == index] = 255
        component_heights.append(component_h)

    foreground_density = float((text_mask > 0).sum()) / max(1, width * height)
    text_density_score = triangle_score(foreground_density, low=0.015, target=0.145, high=0.42)
    median_component_height = float(np.median(component_heights)) if component_heights else 0.0
    text_height_ratio = median_component_height / max(1, height)
    text_height_score = triangle_score(text_height_ratio, low=0.045, target=0.17, high=0.34)

    edge = max(3, int(round(min(width, height) * 0.035)))
    edge_pixels = (
        int((text_mask[:edge, :] > 0).sum())
        + int((text_mask[-edge:, :] > 0).sum())
        + int((text_mask[:, :edge] > 0).sum())
        + int((text_mask[:, -edge:] > 0).sum())
    )
    edge_area = 2 * edge * width + 2 * edge * height
    edge_density = edge_pixels / max(1, edge_area)
    edge_penalty = min(1.0, edge_density / 0.08)
    border_penalty = min(1.0, large_border_area / max(1.0, 0.08 * width * height))

    score = (
        0.28 * sharpness
        + 0.22 * contrast
        + 0.22 * text_density_score
        + 0.18 * text_height_score
        - 0.07 * edge_penalty
        - 0.08 * border_penalty
    )
    score = max(0.0, min(1.0, score))
    return CropQuality(
        video_id=row.get("video_id", ""),
        filename=row.get("filename", ""),
        track_id=row.get("track_id", ""),
        rank=rpt.as_int(row.get("rank"), 999),
        timestamp_ms=row.get("timestamp_ms", ""),
        frame_timestamp=row.get("frame_timestamp", ""),
        variant=variant,
        image_path=path_value,
        score=score,
        sharpness=sharpness,
        contrast=contrast,
        text_density_score=text_density_score,
        text_height_score=text_height_score,
        edge_penalty=edge_penalty,
        border_penalty=border_penalty,
        foreground_density=foreground_density,
        median_component_height=median_component_height,
        width=width,
        height=height,
    )


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_quality_table(zones_manifest: Path, variants: list[str], top_k: int) -> list[CropQuality]:
    rows = [
        row
        for row in rpt.read_rows(zones_manifest)
        if row.get("zone") == rpt.PRODUCT_FIELD and rpt.as_int(row.get("rank"), 999) <= top_k
    ]
    quality_rows: list[CropQuality] = []
    for row in rows:
        for variant, path_value in rpt.row_image_variants(row, variants):
            quality = compute_crop_quality(row, variant, path_value)
            if quality:
                quality_rows.append(quality)
    return quality_rows


def source_variant(source: str) -> str:
    match = re.match(r"^(?:full|lines)_([^_]+)_", str(source or ""))
    return match.group(1) if match else ""


def best_quality_maps(quality_rows: list[CropQuality]) -> tuple[dict[str, CropQuality], dict[tuple[str, int], CropQuality], dict[tuple[str, int, str], CropQuality]]:
    best_track: dict[str, CropQuality] = {}
    best_rank: dict[tuple[str, int], CropQuality] = {}
    best_variant: dict[tuple[str, int, str], CropQuality] = {}
    for row in quality_rows:
        track_key = row.track_id
        rank_key = (row.track_id, row.rank)
        variant_key = (row.track_id, row.rank, row.variant)
        if track_key not in best_track or row.score > best_track[track_key].score:
            best_track[track_key] = row
        if rank_key not in best_rank or row.score > best_rank[rank_key].score:
            best_rank[rank_key] = row
        if variant_key not in best_variant or row.score > best_variant[variant_key].score:
            best_variant[variant_key] = row
    return best_track, best_rank, best_variant


def choose_best(items: list[rpt.ProductNameCandidate]) -> rpt.ProductNameCandidate | None:
    return rpt.choose_best(items)


def choose_for_policy(
    items: list[rpt.ProductNameCandidate],
    policy: str,
    best_track: dict[str, CropQuality],
    best_rank: dict[tuple[str, int], CropQuality],
    best_variant: dict[tuple[str, int, str], CropQuality],
    weight: float,
) -> rpt.ProductNameCandidate | None:
    if not items:
        return None
    if policy == "ocr_best":
        return choose_best(items)

    track_id = items[0].track_id
    if policy == "rank_only":
        chosen = best_track.get(track_id)
        if chosen:
            selected = [item for item in items if item.rank == chosen.rank]
            if selected:
                return choose_best(selected)
        return choose_best(items)

    if policy == "variant_only":
        chosen = best_track.get(track_id)
        if chosen:
            selected = [
                item
                for item in items
                if item.rank == chosen.rank and (not source_variant(item.source) or source_variant(item.source) == chosen.variant)
            ]
            if selected:
                return choose_best(selected)
        return choose_best(items)

    if policy == "weighted":
        def weighted_score(item: rpt.ProductNameCandidate) -> float:
            variant = source_variant(item.source)
            quality = best_variant.get((item.track_id, item.rank, variant)) or best_rank.get((item.track_id, item.rank))
            quality_score = quality.score if quality else 0.0
            rank_bonus = 0.04 if best_track.get(item.track_id) and item.rank == best_track[item.track_id].rank else 0.0
            return item.score + weight * quality_score + rank_bonus

        return max(items, key=weighted_score)

    if policy == "weighted_guarded":
        plausible = [item for item in items if rpt.candidate_is_plausible(item)]
        source_items = plausible if plausible else items

        def weighted_guarded_score(item: rpt.ProductNameCandidate) -> float:
            variant = source_variant(item.source)
            quality = best_variant.get((item.track_id, item.rank, variant)) or best_rank.get((item.track_id, item.rank))
            quality_score = quality.score if quality else 0.0
            border_penalty = quality.border_penalty if quality else 0.0
            edge_penalty = quality.edge_penalty if quality else 0.0
            return item.score + weight * quality_score - 0.12 * border_penalty - 0.08 * edge_penalty

        return max(source_items, key=weighted_guarded_score)

    raise ValueError(f"Unknown policy: {policy}")


def update_rows_with_best(
    best_by_track: dict[str, rpt.ProductNameCandidate],
    base_submission_csv: Path,
    base_debug_csv: Path,
    zones_manifest: Path,
    engine_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    submission_rows = rpt.read_rows(base_submission_csv)
    debug_rows = rpt.read_rows(base_debug_csv)
    track_lookup = rpt.build_track_lookup(rpt.read_rows(zones_manifest))
    changes: list[dict[str, Any]] = []
    for row in submission_rows:
        track_id = row.get("track_id") or row.get("track_id_export") or track_lookup.get(rpt.row_key(row), "")
        best = best_by_track.get(track_id)
        before = row.get(rpt.PRODUCT_FIELD, "")
        normalized_before = rpt.normalize_surface_text(before)
        proposed = best.value if best else normalized_before
        accepted = bool(proposed) if best else False
        final_value = proposed if accepted else normalized_before
        row[rpt.PRODUCT_FIELD] = final_value
        changes.append(
            {
                "track_id": track_id,
                "before": before,
                "after": final_value,
                "proposed": proposed,
                "accepted": int(accepted),
                "reason": "ocr_output" if accepted else "no_candidate",
                "changed": int(before != final_value),
                "score": f"{best.score:.4f}" if best else "",
                "confidence": f"{best.confidence:.4f}" if best else "",
                "source": best.source if best else "base_normalized",
                "line_count": best.line_count if best else 0,
                "raw_text": best.raw_text if best else before,
            }
        )

    final_by_track = {row["track_id"]: row for row in changes if row.get("track_id")}
    debug_by_track = {row.get("track_id", ""): row for row in debug_rows if row.get("field") == rpt.PRODUCT_FIELD}
    for track_id, change in final_by_track.items():
        row = debug_by_track.get(track_id)
        best = best_by_track.get(track_id)
        if row is None:
            if best:
                debug_rows.append(rpt.candidate_to_row(best))
            continue
        row["value"] = change["after"]
        row["score"] = change.get("score", row.get("score", ""))
        row["confidence"] = change.get("confidence", row.get("confidence", ""))
        row["engine"] = engine_name if change["accepted"] else "tesseract5_crop_quality_guarded"
        row["zone"] = rpt.PRODUCT_FIELD
        if best:
            row["image_kind"] = best.source
            row["image_path"] = best.image_path
            row["source_text"] = best.raw_text
    return changes, submission_rows, debug_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--base-submission-csv", type=Path, required=True)
    parser.add_argument("--base-debug-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["ocr_best", "rank_only", "variant_only", "weighted", "weighted_guarded"], default="weighted")
    parser.add_argument("--weight", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--image-variants", default="enhanced,raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    variants = [item.strip() for item in args.image_variants.split(",") if item.strip()]
    quality_rows = build_quality_table(args.zones_manifest, variants, args.top_k)
    best_track, best_rank, best_variant = best_quality_maps(quality_rows)
    candidates = [candidate_from_row(row) for row in rpt.read_rows(args.candidate_csv)]
    grouped: dict[str, list[rpt.ProductNameCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.track_id].append(candidate)

    best_by_track: dict[str, rpt.ProductNameCandidate] = {}
    for track_id, items in grouped.items():
        best = choose_for_policy(items, args.policy, best_track, best_rank, best_variant, args.weight)
        if best:
            best_by_track[track_id] = best

    changes, submission_rows, debug_rows = update_rows_with_best(
        best_by_track,
        args.base_submission_csv,
        args.base_debug_csv,
        args.zones_manifest,
        f"tesseract5_crop_quality_{args.policy}",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "crop_quality_manifest.csv", [asdict(row) for row in quality_rows])
    write_rows(args.output / "product_name_line_candidates.csv", [rpt.candidate_to_row(item) for item in candidates])
    write_rows(args.output / "product_name_line_best.csv", [rpt.candidate_to_row(item) for item in best_by_track.values()])
    write_rows(args.output / "product_name_line_changes.csv", changes)
    write_rows(args.output / "ocr_aggregated_submission_product_lines.csv", submission_rows, fieldnames=rpt.OUTPUT_COLUMNS)
    write_rows(args.output / "ocr_aggregated_debug_product_lines.csv", debug_rows)

    summary = {
        "engine": f"tesseract5_crop_quality_{args.policy}",
        "policy": args.policy,
        "weight": args.weight,
        "tracks": len(best_by_track),
        "candidate_rows": len(candidates),
        "crop_quality_rows": len(quality_rows),
        "changed_product_names": sum(1 for row in changes if row["changed"]),
        "accepted_proposals": sum(1 for row in changes if row["accepted"]),
        "elapsed_seconds": time.perf_counter() - started,
        "image_variants": variants,
        "top_k": args.top_k,
    }
    (args.output / "product_name_line_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
