"""Reselect Tesseract product_name candidates without rerunning OCR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_product_name_tesseract as rpt


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def candidate_from_row(row: dict[str, str]) -> rpt.ProductNameCandidate:
    return rpt.ProductNameCandidate(
        video_id=row.get("video_id", ""),
        filename=row.get("filename", ""),
        track_id=row.get("track_id", ""),
        rank=rpt.as_int(row.get("rank"), 999),
        timestamp_ms=row.get("timestamp_ms", ""),
        frame_timestamp=row.get("frame_timestamp", ""),
        value=row.get("value", ""),
        raw_text=row.get("raw_text", ""),
        confidence=parse_float(row.get("confidence")),
        score=parse_float(row.get("score")),
        source=row.get("source", ""),
        image_path=row.get("image_path", ""),
        line_count=rpt.as_int(row.get("line_count"), 0),
        elapsed_ms=parse_float(row.get("elapsed_ms")),
        return_code=rpt.as_int(row.get("return_code"), 0),
        stderr=row.get("stderr", ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--base-submission-csv", type=Path, required=True)
    parser.add_argument("--base-debug-csv", type=Path, required=True)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-policy", choices=["best", "line_split"], default="line_split")
    parser.add_argument("--accept-policy", choices=["all", "guard"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    candidates = [candidate_from_row(row) for row in rpt.read_rows(args.candidate_csv)]
    grouped: dict[str, list[rpt.ProductNameCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.track_id, []).append(candidate)

    best_by_track: dict[str, rpt.ProductNameCandidate] = {}
    for track_id, items in grouped.items():
        best = rpt.choose_best_for_policy(items, args.selection_policy)
        if best:
            best_by_track[track_id] = best

    submission_rows = rpt.read_rows(args.base_submission_csv)
    debug_rows = rpt.read_rows(args.base_debug_csv)
    track_lookup = rpt.build_track_lookup(rpt.read_rows(args.zones_manifest))
    changes: list[dict[str, Any]] = []
    for row in submission_rows:
        track_id = row.get("track_id") or row.get("track_id_export") or track_lookup.get(rpt.row_key(row), "")
        best = best_by_track.get(track_id)
        before = row.get(rpt.PRODUCT_FIELD, "")
        normalized_before = rpt.normalize_surface_text(before)
        proposed = best.value if best else normalized_before
        if not best:
            accepted = False
            accept_reason = "no_candidate"
        elif args.accept_policy == "guard":
            accepted, accept_reason = rpt.safe_replacement(normalized_before, proposed)
        else:
            accepted = bool(proposed)
            accept_reason = "ocr_output" if accepted else "empty"
        final_value = proposed if accepted else normalized_before
        row[rpt.PRODUCT_FIELD] = final_value
        changes.append(
            {
                "track_id": track_id,
                "before": before,
                "after": final_value,
                "proposed": proposed,
                "accepted": int(accepted),
                "reason": accept_reason,
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
        row["engine"] = "tesseract5_tessdata_best_linesplit_guarded" if change["accepted"] else "tesseract5_guarded"
        row["zone"] = rpt.PRODUCT_FIELD
        if best:
            row["image_kind"] = best.source
            row["image_path"] = best.image_path
            row["source_text"] = best.raw_text

    args.output.mkdir(parents=True, exist_ok=True)
    rpt.write_rows(args.output / "product_name_line_candidates.csv", [rpt.candidate_to_row(item) for item in candidates])
    rpt.write_rows(args.output / "product_name_line_best.csv", [rpt.candidate_to_row(item) for item in best_by_track.values()])
    rpt.write_rows(args.output / "product_name_line_changes.csv", changes)
    rpt.write_rows(args.output / "ocr_aggregated_submission_product_lines.csv", submission_rows, fieldnames=rpt.OUTPUT_COLUMNS)
    rpt.write_rows(args.output / "ocr_aggregated_debug_product_lines.csv", debug_rows)

    summary = {
        "engine": "tesseract5_tessdata_best_linesplit_guarded",
        "selection_policy": args.selection_policy,
        "source_candidate_csv": str(args.candidate_csv),
        "tracks": len(best_by_track),
        "candidates": len(candidates),
        "changed_product_names": sum(1 for row in changes if row["changed"]),
        "accepted_proposals": sum(1 for row in changes if row["accepted"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "product_name_line_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
