from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_43_15_report import (  # noqa: E402
    is_missing,
    load_predictions,
    match_predictions,
    normalize_discount,
    normalize_price,
    normalize_text,
    parse_float,
    text_similarity,
)


GT_ALIASES = {"wholesale_level_1_count": "wholesale_level_1_coun"}
PRICE_FIELDS = {
    "price_default",
    "price_card",
    "price_discount",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}
COORD_FIELDS = {"x_min", "y_min", "x_max", "y_max"}
IDENTIFIER_FIELDS = {"barcode", "id_sku", "qr_code_barcode", "action_code_qr"}
NO_VALUE = "нет"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every sample submission field and build a visual review page.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, default=Path("data/43_15/43_15.csv"))
    parser.add_argument("--sample-csv", type=Path, default=Path("data/sample.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--coord-tolerance", type=float, default=2.0)
    return parser.parse_args()


def as_abs(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file).fieldnames or [])


def gt_value(row: pd.Series, field: str) -> Any:
    gt_field = field if field in row.index else GT_ALIASES.get(field, field)
    return row.get(gt_field, "")


def norm_no_value(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().lower().replace("ё", "е")
    return NO_VALUE if text == NO_VALUE else text


def price_or_no_value(value: Any) -> str:
    if norm_no_value(value) == NO_VALUE:
        return NO_VALUE
    return normalize_price(value)


def normalize_identifier(value: Any) -> str:
    if norm_no_value(value) == NO_VALUE:
        return NO_VALUE
    if is_missing(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or normalize_text(text)


def field_match(pred: pd.Series, truth: pd.Series, field: str, coord_tolerance: float) -> bool:
    expected = gt_value(truth, field)
    actual = pred.get(field, "")
    if is_missing(expected):
        return False
    if field == "product_name":
        return text_similarity(actual, expected) >= 0.80
    if field == "discount_amount":
        return normalize_discount(actual) == normalize_discount(expected) and normalize_discount(expected) != ""
    if field in PRICE_FIELDS:
        left = price_or_no_value(actual)
        right = price_or_no_value(expected)
        return left == right and right != ""
    if field in COORD_FIELDS:
        left = parse_float(actual)
        right = parse_float(expected)
        return not math.isnan(left) and not math.isnan(right) and abs(left - right) <= coord_tolerance
    if field == "frame_timestamp":
        left = parse_float(actual)
        right = parse_float(expected)
        return not math.isnan(left) and not math.isnan(right) and abs(left - right) <= 1
    if field in IDENTIFIER_FIELDS:
        return normalize_identifier(actual) == normalize_identifier(expected) and normalize_identifier(expected) != ""
    return normalize_text(actual) == normalize_text(expected) and normalize_text(expected) != ""


def value_for_csv(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value)


def rel_image(path_value: Any, report_path: Path) -> str:
    if is_missing(path_value):
        return ""
    path = Path(str(path_value))
    if not path.exists():
        return ""
    try:
        return Path(os.path.relpath(path.resolve(), report_path.parent.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_uri()
    except OSError:
        return ""


def img_tag(path_value: Any, report_path: Path, width: int = 180) -> str:
    src = local_review_image(path_value, report_path)
    if not src:
        return ""
    return f'<img src="{html.escape(src)}" style="max-width:{width}px;max-height:130px">'


def local_review_image(path_value: Any, report_path: Path) -> str:
    if is_missing(path_value):
        return ""
    source = Path(str(path_value))
    if not source.exists():
        return ""
    image_dir = report_path.parent / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return target.relative_to(report_path.parent).as_posix()


def build_full_matches(
    predictions: pd.DataFrame,
    gt: pd.DataFrame,
    matched_core: pd.DataFrame,
    fields: list[str],
    coord_tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, match in matched_core.iterrows():
        status = str(match.get("match_status", ""))
        row: dict[str, Any] = {
            "match_status": status,
            "track_id": match.get("track_id", ""),
            "pred_index": match.get("pred_index", ""),
            "gt_index": match.get("gt_index", ""),
            "match_score": match.get("match_score", ""),
            "iou": match.get("iou", ""),
            "full_tag": match.get("full_tag", ""),
            "source_crop": match.get("source_crop", ""),
        }
        correct = 0
        total = 0
        pred = None
        truth = None
        if status == "matched":
            pred = predictions.loc[int(match["pred_index"])]
            truth = gt.loc[int(match["gt_index"])]
        elif status == "extra_prediction":
            pred = predictions.loc[int(match["pred_index"])]
        elif status == "missed_ground_truth":
            truth = gt.loc[int(match["gt_index"])]

        mismatches: list[str] = []
        for field in fields:
            pred_value = pred.get(field, "") if pred is not None else ""
            gt_field_value = gt_value(truth, field) if truth is not None else ""
            matched = False
            if pred is not None and truth is not None and not is_missing(gt_field_value):
                total += 1
                matched = field_match(pred, truth, field, coord_tolerance)
                correct += int(matched)
                if not matched:
                    mismatches.append(f"{field}: {value_for_csv(pred_value)} -> {value_for_csv(gt_field_value)}")
            row[f"pred_{field}"] = value_for_csv(pred_value)
            row[f"gt_{field}"] = value_for_csv(gt_field_value)
            row[f"{field}_match"] = matched
            if field == "product_name":
                row["product_name_similarity"] = round(text_similarity(pred_value, gt_field_value), 4) if truth is not None else 0
        row["correct_fields"] = correct
        row["evaluated_fields"] = total
        row["field_accuracy"] = round(correct / total, 4) if total else 0.0
        row["mismatches"] = "; ".join(mismatches)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics(full_matches: pd.DataFrame, predictions: pd.DataFrame, gt: pd.DataFrame, fields: list[str]) -> dict[str, Any]:
    matched = full_matches[full_matches["match_status"] == "matched"].copy()
    field_metrics: dict[str, Any] = {}
    for field in fields:
        match_col = f"{field}_match"
        gt_col = f"gt_{field}"
        if match_col not in matched.columns:
            total = 0
            correct = 0
        else:
            usable = matched[matched[gt_col].apply(lambda value: not is_missing(value))]
            total = len(usable)
            correct = int(usable[match_col].sum()) if total else 0
        field_metrics[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "filled": int(predictions[field].apply(lambda value: not is_missing(value)).sum()) if field in predictions.columns else 0,
            "pred_total": len(predictions),
        }

    strict_fields = [
        field
        for field in fields
        if field not in {"x_min", "y_min", "x_max", "y_max", "frame_timestamp"}
    ]
    strict_match_cols = [f"{field}_match" for field in strict_fields if f"{field}_match" in matched.columns]
    fully_correct = int(matched[strict_match_cols].all(axis=1).sum()) if strict_match_cols and not matched.empty else 0
    summary = {
        "gt_tags": len(gt),
        "pred_tags": len(predictions),
        "matched_tags": len(matched),
        "extra_predictions": int((full_matches["match_status"] == "extra_prediction").sum()),
        "missed_gt": int((full_matches["match_status"] == "missed_ground_truth").sum()),
        "mean_full_field_accuracy_matched": round(float(matched["field_accuracy"].mean()), 4) if not matched.empty else 0.0,
        "fully_correct_non_box_rows": f"{fully_correct}/{len(matched)}",
        "mean_product_name_similarity": round(float(matched["product_name_similarity"].mean()), 4)
        if "product_name_similarity" in matched.columns and not matched.empty
        else 0.0,
    }
    return {"summary": summary, "fields": field_metrics}


def build_html(metrics: dict[str, Any], full_matches: pd.DataFrame, report_path: Path, fields: list[str]) -> str:
    cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(str(k))}</div><div class='value'>{html.escape(str(v))}</div></div>"
        for k, v in metrics["summary"].items()
    )
    field_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(v))}</td>"
            for v in [
                field,
                data["correct"],
                data["total"],
                data["accuracy"],
                f"{data['filled']}/{data['pred_total']}",
            ]
        )
        + "</tr>"
        for field, data in metrics["fields"].items()
    )
    key_cols = [
        "pred_product_name",
        "gt_product_name",
        "product_name_similarity",
        "pred_price_default",
        "gt_price_default",
        "pred_price_card",
        "gt_price_card",
        "pred_discount_amount",
        "gt_discount_amount",
        "pred_id_sku",
        "gt_id_sku",
        "pred_price1_qr",
        "gt_price1_qr",
        "pred_price2_qr",
        "gt_price2_qr",
        "pred_price4_qr",
        "gt_price4_qr",
    ]
    table_rows = []
    display = full_matches.copy()
    if "field_accuracy" in display.columns:
        display = display.sort_values(["match_status", "field_accuracy"], ascending=[True, True])
    for _, row in display.iterrows():
        cells = [
            f"<td>{img_tag(row.get('full_tag', ''), report_path, width=210)}</td>",
            f"<td>{html.escape(str(row.get('match_status', '')))}</td>",
            f"<td>{html.escape(str(row.get('track_id', '')))}</td>",
            f"<td>{html.escape(str(row.get('field_accuracy', '')))}</td>",
        ]
        for col in key_cols:
            cells.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        cells.append(f"<td class='mismatch'>{html.escape(str(row.get('mismatches', '')))}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    headers = ["image", "status", "track_id", "row_acc"] + key_cols + ["all_mismatches"]
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Full Submission Field Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f6f8fb; }}
    header {{ padding: 20px 24px; background: #101828; color: white; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 16px 24px; }}
    .card {{ background: white; border: 1px solid #dbe3ef; border-radius: 6px; padding: 10px 12px; min-width: 150px; }}
    .label {{ color: #667085; font-size: 12px; }}
    .value {{ font-weight: 700; font-size: 18px; margin-top: 4px; }}
    section {{ padding: 0 24px 24px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5eaf2; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #eef3f9; text-align: left; position: sticky; top: 67px; z-index: 1; }}
    .mismatch {{ min-width: 360px; max-width: 560px; white-space: normal; }}
    img {{ border: 1px solid #d0d7e2; border-radius: 4px; background: white; }}
  </style>
</head>
<body>
  <header><h1>Full Submission Field Review</h1></header>
  <div class="cards">{cards}</div>
  <section>
    <h2>Field Metrics</h2>
    <table>
      <thead><tr><th>field</th><th>correct</th><th>total</th><th>accuracy</th><th>filled</th></tr></thead>
      <tbody>{field_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Manual Review</h2>
    <table>
      <thead><tr>{"".join(f"<th>{html.escape(h)}</th>" for h in headers)}</tr></thead>
      <tbody>{"".join(table_rows)}</tbody>
    </table>
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_dir = as_abs(root, args.run_dir)
    gt_csv = as_abs(root, args.gt_csv)
    sample_csv = as_abs(root, args.sample_csv)
    output_dir = as_abs(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = read_fieldnames(sample_csv)
    gt = pd.read_csv(gt_csv, encoding="utf-8-sig")
    predictions, _, _ = load_predictions(run_dir)
    matched_core = match_predictions(predictions, gt, args.iou_threshold)
    full_matches = build_full_matches(predictions, gt, matched_core, fields, args.coord_tolerance)
    metrics = compute_metrics(full_matches, predictions, gt, fields)

    full_matches_csv = output_dir / "full_field_matches.csv"
    field_metrics_csv = output_dir / "full_field_metrics.csv"
    metrics_json = output_dir / "full_metrics.json"
    report_html = output_dir / "full_submission_review.html"

    full_matches.to_csv(full_matches_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"field": field, **data} for field, data in metrics["fields"].items()]
    ).to_csv(field_metrics_csv, index=False, encoding="utf-8-sig")
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_html.write_text(build_html(metrics, full_matches, report_html, fields), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"full_matches_csv={full_matches_csv}")
    print(f"field_metrics_csv={field_metrics_csv}")
    print(f"metrics_json={metrics_json}")
    print(f"report_html={report_html}")


if __name__ == "__main__":
    main()
