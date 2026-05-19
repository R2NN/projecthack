"""Merge selected fields from one OCR submission/debug pair into another."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("filename", "").strip(),
        row.get("frame_timestamp", "").strip(),
        row.get("x_min", "").strip(),
        row.get("y_min", "").strip(),
        row.get("x_max", "").strip(),
        row.get("y_max", "").strip(),
    )


def merge_submissions(base_rows: list[dict[str, str]], override_rows: list[dict[str, str]], fields: list[str]) -> list[dict[str, str]]:
    overrides = {row_key(row): row for row in override_rows}
    merged: list[dict[str, str]] = []
    for row in base_rows:
        new_row = dict(row)
        override = overrides.get(row_key(row))
        if override:
            for field in fields:
                value = override.get(field, "").strip()
                if value:
                    new_row[field] = value
        merged.append(new_row)
    return merged


def merge_debug(base_rows: list[dict[str, str]], override_rows: list[dict[str, str]], fields: list[str]) -> list[dict[str, str]]:
    filtered_base = [row for row in base_rows if row.get("field", "") not in fields]
    filtered_override = [row for row in override_rows if row.get("field", "") in fields]
    return filtered_base + filtered_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--base-debug", type=Path, required=True)
    parser.add_argument("--override-submission", type=Path, required=True)
    parser.add_argument("--override-debug", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    parser.add_argument("--output-debug", type=Path, required=True)
    parser.add_argument("--fields", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]

    base_submission = read_rows(args.base_submission)
    override_submission = read_rows(args.override_submission)
    submission_fieldnames = list(base_submission[0].keys()) if base_submission else []
    merged_submission = merge_submissions(base_submission, override_submission, fields)
    write_rows(args.output_submission, merged_submission, submission_fieldnames)

    base_debug = read_rows(args.base_debug)
    override_debug = read_rows(args.override_debug)
    debug_fieldnames = list(base_debug[0].keys()) if base_debug else list(override_debug[0].keys())
    merged_debug = merge_debug(base_debug, override_debug, fields)
    write_rows(args.output_debug, merged_debug, debug_fieldnames)


if __name__ == "__main__":
    main()
