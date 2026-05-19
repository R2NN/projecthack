from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


APPROVED_VALUES = {"1", "true", "yes", "y", "ok", "approved"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PaddleOCR train/val files from reviewed real line-crop OCR labels."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("line_candidates") / "review_queue_priority.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--require-approved", action="store_true")
    return parser.parse_args()


def is_approved(value: str) -> bool:
    return value.strip().lower() in APPROVED_VALUES


def write_label_file(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(f"{row['rel_path']}\t{row['reviewed_label'].strip()}\n")


def copy_images(dataset_dir: Path, output_dir: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        rel_path = Path(row["rel_path"])
        source = dataset_dir / rel_path
        target = output_dir / rel_path
        if not source.exists():
            raise FileNotFoundError(f"Missing source image: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    review_csv = args.review_csv
    if not review_csv.is_absolute():
        review_csv = dataset_dir / review_csv

    with review_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    selected: list[dict[str, str]] = []
    for row in rows:
        label = row.get("reviewed_label", "").strip()
        if not label:
            continue
        quality = float(row.get("quality_score") or 0.0)
        if quality < args.min_quality:
            continue
        if args.require_approved and not is_approved(row.get("approved", "")):
            continue
        selected.append(row)

    if not selected:
        raise RuntimeError("No reviewed line rows selected")

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    split_index = int(round(len(selected) * args.train_ratio))
    split_index = min(max(1, split_index), max(1, len(selected) - 1))
    train_rows = selected[:split_index]
    val_rows = selected[split_index:]

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_images(dataset_dir, output_dir, selected)
    write_label_file(output_dir / "train.txt", train_rows)
    write_label_file(output_dir / "val.txt", val_rows)

    dict_path = dataset_dir / "dict.txt"
    if dict_path.exists():
        shutil.copyfile(dict_path, output_dir / "dict.txt")

    with (output_dir / "source_dataset.txt").open("w", encoding="utf-8") as file:
        file.write(f"dataset_dir={dataset_dir}\n")
        file.write(f"review_csv={review_csv}\n")
        file.write(f"selected={len(selected)}\n")
        file.write(f"train={len(train_rows)}\n")
        file.write(f"val={len(val_rows)}\n")
        file.write(f"require_approved={int(args.require_approved)}\n")
        file.write(f"min_quality={args.min_quality}\n")

    print(f"selected={len(selected)} train={len(train_rows)} val={len(val_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
