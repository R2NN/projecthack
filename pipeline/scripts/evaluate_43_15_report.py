from __future__ import annotations

import argparse
import html
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


KEY_FIELDS = ["product_name", "price_default", "price_card", "discount_amount"]
ZONE_ORDER = ["product_name", "price_default_wide", "price_card_number", "discount_amount", "qr", "barcode"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 43_15 pipeline output and build a visual report.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, default=Path("repro_outputs/quality_43_15"))
    parser.add_argument("--gt-csv", type=Path, default=Path("data/43_15/43_15.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/quality_43_15/evaluation"))
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    return parser.parse_args()


def as_abs(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def parse_float(value: Any) -> float:
    if is_missing(value):
        return math.nan
    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", "."}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_price(value: Any) -> str:
    number = parse_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.2f}"


def normalize_discount(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().replace("−", "-").replace(" ", "")
    match = re.search(r"-?\d+", text)
    if not match:
        return normalize_text(text)
    number = match.group(0)
    if not number.startswith("-"):
        number = "-" + number
    return f"{number}%"


def text_similarity(left: Any, right: Any) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def boxes_iou(left: pd.Series, right: pd.Series) -> float:
    lx1, ly1, lx2, ly2 = (parse_float(left[c]) for c in ["x_min", "y_min", "x_max", "y_max"])
    rx1, ry1, rx2, ry2 = (parse_float(right[c]) for c in ["x_min", "y_min", "x_max", "y_max"])
    if any(math.isnan(v) for v in [lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2]):
        return 0.0
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    larea = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    rarea = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    denom = larea + rarea - inter
    return inter / denom if denom > 0 else 0.0


def center_distance(left: pd.Series, right: pd.Series) -> float:
    lx1, ly1, lx2, ly2 = (parse_float(left[c]) for c in ["x_min", "y_min", "x_max", "y_max"])
    rx1, ry1, rx2, ry2 = (parse_float(right[c]) for c in ["x_min", "y_min", "x_max", "y_max"])
    if any(math.isnan(v) for v in [lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2]):
        return math.inf
    return math.hypot(((lx1 + lx2) / 2) - ((rx1 + rx2) / 2), ((ly1 + ly2) / 2) - ((ry1 + ry2) / 2))


def load_predictions(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_csv = run_dir / "ocr_final_quality_core_fast_fixed" / "ocr_aggregated_submission_product_lines.csv"
    debug_csv = run_dir / "ocr_final_quality_core_fast_fixed" / "ocr_aggregated_debug_product_lines.csv"
    manifest_csv = run_dir / "ocr_zones_core_fixed" / "ocr_zones_manifest.csv"
    ocr_manifest_csv = run_dir / "ocr_zones_core_fixed" / "ocr_manifest.csv"
    if not final_csv.exists():
        raise FileNotFoundError(final_csv)
    if not debug_csv.exists():
        raise FileNotFoundError(debug_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(manifest_csv)

    final = pd.read_csv(final_csv, encoding="utf-8-sig")
    debug = pd.read_csv(debug_csv, encoding="utf-8-sig")
    zones = pd.read_csv(manifest_csv, encoding="utf-8-sig")
    ocr_manifest = pd.read_csv(ocr_manifest_csv, encoding="utf-8-sig") if ocr_manifest_csv.exists() else pd.DataFrame()

    if "track_id" not in final.columns:
        rank1 = (
            ocr_manifest.sort_values(["track_id", "rank"])
            .drop_duplicates("track_id", keep="first")
            [["track_id", "full_tag", "source_crop", "x_min", "y_min", "x_max", "y_max", "timestamp_ms"]]
        )
        assigned = []
        used: set[int] = set()
        for _, pred_row in final.iterrows():
            best_idx = None
            best_dist = math.inf
            for idx, manifest_row in rank1.iterrows():
                if idx in used:
                    continue
                dist = center_distance(pred_row, manifest_row)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None:
                assigned.append({})
                continue
            used.add(best_idx)
            assigned.append(rank1.loc[best_idx].to_dict())
        assigned_df = pd.DataFrame(assigned)
        for col in ["track_id", "full_tag", "source_crop", "timestamp_ms"]:
            if col in assigned_df.columns:
                final[col] = assigned_df[col]
    return final, debug, zones


def best_debug_by_field(debug: pd.DataFrame) -> pd.DataFrame:
    if debug.empty:
        return pd.DataFrame(columns=["track_id", "field", "value", "score", "confidence", "engine", "zone", "image_path", "source_text"])
    work = debug.copy()
    work["score_sort"] = work.get("score", 0).fillna(0)
    work["confidence_sort"] = work.get("confidence", 0).fillna(0)
    return (
        work.sort_values(["track_id", "field", "score_sort", "confidence_sort"], ascending=[True, True, False, False])
        .drop_duplicates(["track_id", "field"], keep="first")
        .drop(columns=["score_sort", "confidence_sort"])
    )


def field_match(pred: pd.Series, gt: pd.Series, field: str) -> bool:
    if field in {"price_default", "price_card"}:
        return normalize_price(pred.get(field)) == normalize_price(gt.get(field)) and normalize_price(gt.get(field)) != ""
    if field == "discount_amount":
        return normalize_discount(pred.get(field)) == normalize_discount(gt.get(field)) and normalize_discount(gt.get(field)) != ""
    if field == "product_name":
        return text_similarity(pred.get(field), gt.get(field)) >= 0.80
    return normalize_text(pred.get(field)) == normalize_text(gt.get(field))


def pair_score(pred: pd.Series, gt: pd.Series) -> float:
    score = 0.0
    iou = boxes_iou(pred, gt)
    distance = center_distance(pred, gt)
    if iou > 0:
        score += 2.0 * iou
    if math.isfinite(distance):
        score += max(0.0, 1.5 * (1.0 - min(distance, 350.0) / 350.0))
    if normalize_price(pred.get("price_card")) and normalize_price(pred.get("price_card")) == normalize_price(gt.get("price_card")):
        score += 5.0
    if normalize_price(pred.get("price_default")) and normalize_price(pred.get("price_default")) == normalize_price(gt.get("price_default")):
        score += 3.0
    else:
        pred_default = parse_float(pred.get("price_default"))
        gt_default = parse_float(gt.get("price_default"))
        if not math.isnan(pred_default) and not math.isnan(gt_default) and abs(pred_default - gt_default) <= 1.0:
            score += 1.2
    if normalize_discount(pred.get("discount_amount")) and normalize_discount(pred.get("discount_amount")) == normalize_discount(gt.get("discount_amount")):
        score += 1.5
    score += 2.0 * text_similarity(pred.get("product_name"), gt.get("product_name"))
    return score


def match_predictions(predictions: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    candidate_pairs: list[dict[str, Any]] = []
    for pred_idx, pred_row in predictions.iterrows():
        for gt_idx, gt_row in gt.iterrows():
            iou = boxes_iou(pred_row, gt_row)
            score = pair_score(pred_row, gt_row)
            any_value_hit = any(field_match(pred_row, gt_row, field) for field in KEY_FIELDS)
            if score >= 2.0 or iou >= iou_threshold or any_value_hit:
                candidate_pairs.append(
                    {
                        "pred_idx": pred_idx,
                        "gt_idx": gt_idx,
                        "score": score,
                        "iou": iou,
                        "center_distance": center_distance(pred_row, gt_row),
                    }
                )
    matches = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for pair in sorted(candidate_pairs, key=lambda item: (item["score"], item["iou"]), reverse=True):
        pred_idx = int(pair["pred_idx"])
        gt_idx = int(pair["gt_idx"])
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append(pair)

    rows: list[dict[str, Any]] = []
    for match in matches:
        pred = predictions.loc[match["pred_idx"]]
        truth = gt.loc[match["gt_idx"]]
        row: dict[str, Any] = {
            "match_status": "matched",
            "match_score": round(match["score"], 4),
            "iou": round(match["iou"], 4),
            "center_distance_px": round(match["center_distance"], 1) if math.isfinite(match["center_distance"]) else "",
            "gt_index": int(match["gt_idx"]),
            "pred_index": int(match["pred_idx"]),
            "track_id": pred.get("track_id", ""),
            "full_tag": pred.get("full_tag", ""),
            "source_crop": pred.get("source_crop", ""),
            "pred_frame_timestamp": pred.get("frame_timestamp", ""),
            "gt_frame_timestamp": truth.get("frame_timestamp", ""),
        }
        correct = 0
        total = 0
        for field in KEY_FIELDS:
            has_gt = not is_missing(truth.get(field)) and normalize_text(truth.get(field)) != "нет"
            if has_gt:
                total += 1
            matched = field_match(pred, truth, field) if has_gt else False
            correct += int(matched)
            row[f"pred_{field}"] = pred.get(field, "")
            row[f"gt_{field}"] = truth.get(field, "")
            row[f"{field}_match"] = matched
            if field == "product_name":
                row["product_name_similarity"] = round(text_similarity(pred.get(field), truth.get(field)), 4)
        row["recognized_fields"] = correct
        row["evaluated_fields"] = total
        row["field_accuracy"] = round(correct / total, 4) if total else 0.0
        row["pass_80_proxy"] = bool(total and correct / total >= 0.80)
        rows.append(row)

    for pred_idx, pred in predictions.iterrows():
        if pred_idx in used_pred:
            continue
        rows.append(
            {
                "match_status": "extra_prediction",
                "match_score": 0,
                "iou": 0,
                "center_distance_px": "",
                "gt_index": "",
                "pred_index": int(pred_idx),
                "track_id": pred.get("track_id", ""),
                "full_tag": pred.get("full_tag", ""),
                "source_crop": pred.get("source_crop", ""),
                "pred_frame_timestamp": pred.get("frame_timestamp", ""),
                "gt_frame_timestamp": "",
                **{f"pred_{field}": pred.get(field, "") for field in KEY_FIELDS},
                **{f"gt_{field}": "" for field in KEY_FIELDS},
                **{f"{field}_match": False for field in KEY_FIELDS},
                "product_name_similarity": 0,
                "recognized_fields": 0,
                "evaluated_fields": 0,
                "field_accuracy": 0,
                "pass_80_proxy": False,
            }
        )
    for gt_idx, truth in gt.iterrows():
        if gt_idx in used_gt:
            continue
        rows.append(
            {
                "match_status": "missed_ground_truth",
                "match_score": 0,
                "iou": 0,
                "center_distance_px": "",
                "gt_index": int(gt_idx),
                "pred_index": "",
                "track_id": "",
                "full_tag": "",
                "source_crop": "",
                "pred_frame_timestamp": "",
                "gt_frame_timestamp": truth.get("frame_timestamp", ""),
                **{f"pred_{field}": "" for field in KEY_FIELDS},
                **{f"gt_{field}": truth.get(field, "") for field in KEY_FIELDS},
                **{f"{field}_match": False for field in KEY_FIELDS},
                "product_name_similarity": 0,
                "recognized_fields": 0,
                "evaluated_fields": sum(
                    1 for field in KEY_FIELDS if not is_missing(truth.get(field)) and normalize_text(truth.get(field)) != "нет"
                ),
                "field_accuracy": 0,
                "pass_80_proxy": False,
            }
        )
    return pd.DataFrame(rows)


def build_zone_table(predictions: pd.DataFrame, debug: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    best_debug = best_debug_by_field(debug)
    debug_map = {
        (str(row["track_id"]), str(row["field"])): row
        for _, row in best_debug.iterrows()
        if not is_missing(row.get("track_id")) and not is_missing(row.get("field"))
    }
    rank1_zones = zones[zones["rank"] == 1].copy() if "rank" in zones.columns else zones.copy()
    rows: list[dict[str, Any]] = []
    for _, zone_row in rank1_zones.iterrows():
        zone = str(zone_row.get("zone", ""))
        if zone not in ZONE_ORDER:
            continue
        track_id = str(zone_row.get("track_id", ""))
        target_fields = str(zone_row.get("target_fields", ""))
        target_field = target_fields.split("|")[0] if target_fields and target_fields != "nan" else zone
        value = ""
        source_text = ""
        confidence = ""
        engine = ""
        if (track_id, target_field) in debug_map:
            dbg = debug_map[(track_id, target_field)]
            value = dbg.get("value", "")
            source_text = dbg.get("source_text", "")
            confidence = dbg.get("confidence", "")
            engine = dbg.get("engine", "")
        else:
            pred_match = predictions[predictions.get("track_id").astype(str) == track_id] if "track_id" in predictions.columns else pd.DataFrame()
            if not pred_match.empty and target_field in pred_match.columns:
                value = pred_match.iloc[0].get(target_field, "")
        rows.append(
            {
                "track_id": track_id,
                "zone": zone,
                "target_fields": target_fields,
                "prediction": value,
                "source_text": source_text,
                "confidence": confidence,
                "engine": engine,
                "crop_image": zone_row.get("zone_enhanced") or zone_row.get("zone_raw") or "",
                "full_tag": zone_row.get("full_tag", ""),
            }
        )
    return pd.DataFrame(rows)


def rel_image(path_value: Any, report_path: Path) -> str:
    if is_missing(path_value):
        return ""
    path = Path(str(path_value))
    try:
        if path.exists():
            return path.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri() if path.exists() else ""
    except OSError:
        return ""
    return ""


def img_tag(path_value: Any, report_path: Path, width: int = 180) -> str:
    src = rel_image(path_value, report_path)
    if not src:
        return ""
    return f'<img src="{html.escape(src)}" loading="lazy" style="max-width:{width}px;max-height:130px">'


def style_bool(value: Any) -> str:
    return "ok" if str(value).lower() == "true" else "bad"


def build_html(metrics: dict[str, Any], matched: pd.DataFrame, zone_table: pd.DataFrame, report_path: Path) -> str:
    metric_cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(str(key))}</div><div class='value'>{html.escape(str(value))}</div></div>"
        for key, value in metrics["summary_cards"].items()
    )
    field_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in [field, data["correct"], data["total"], data["accuracy"]])
        + "</tr>"
        for field, data in metrics["field_metrics"].items()
    )

    match_rows = []
    display_cols = [
        "match_status",
        "track_id",
        "full_tag",
        "pred_product_name",
        "gt_product_name",
        "product_name_similarity",
        "pred_price_default",
        "gt_price_default",
        "price_default_match",
        "pred_price_card",
        "gt_price_card",
        "price_card_match",
        "pred_discount_amount",
        "gt_discount_amount",
        "discount_amount_match",
        "field_accuracy",
        "pass_80_proxy",
        "iou",
    ]
    for _, row in matched.iterrows():
        cells = []
        for col in display_cols:
            if col == "full_tag":
                cells.append(f"<td>{img_tag(row.get(col), report_path, 210)}</td>")
            elif col.endswith("_match") or col == "pass_80_proxy":
                cells.append(f"<td class='{style_bool(row.get(col))}'>{html.escape(str(row.get(col)))}</td>")
            else:
                cells.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        match_rows.append("<tr>" + "".join(cells) + "</tr>")

    zone_rows = []
    for _, row in zone_table.iterrows():
        zone_rows.append(
            "<tr>"
            + f"<td>{html.escape(str(row.get('track_id', '')))}</td>"
            + f"<td>{html.escape(str(row.get('zone', '')))}</td>"
            + f"<td>{img_tag(row.get('crop_image'), report_path, 170)}</td>"
            + f"<td>{html.escape(str(row.get('prediction', '')))}</td>"
            + f"<td>{html.escape(str(row.get('source_text', '')))}</td>"
            + f"<td>{html.escape(str(row.get('engine', '')))}</td>"
            + "</tr>"
        )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>43_15 Quality Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1, h2 {{ margin: 0 0 14px; }}
    h2 {{ margin-top: 28px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .card {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 12px; }}
    .label {{ font-size: 12px; color: #657282; }}
    .value {{ font-size: 24px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dde3ea; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #e5e9ef; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #edf1f5; z-index: 1; text-align: left; }}
    tr:hover {{ background: #fbfcfd; }}
    .ok {{ color: #0b7a3b; font-weight: 650; }}
    .bad {{ color: #b42318; font-weight: 650; }}
    .note {{ color: #5b6776; max-width: 1100px; line-height: 1.45; }}
    img {{ border: 1px solid #d5dce5; border-radius: 4px; background: #fff; }}
  </style>
</head>
<body>
  <h1>43_15: прогнозы, кропы и метрики</h1>
  <p class="note">Отчет построен по свежему прогону handoff-пайплайна. Метрика pass_80_proxy считается по четырем полям, которые реально извлекает этот режим OCR: product_name, price_default, price_card и discount_amount. Название товара засчитывается при similarity >= 0.80, цены и скидка требуют точного нормализованного совпадения.</p>
  <div class="cards">{metric_cards}</div>
  <h2>Field Metrics</h2>
  <table>
    <thead><tr><th>field</th><th>correct</th><th>total</th><th>accuracy</th></tr></thead>
    <tbody>{field_rows}</tbody>
  </table>
  <h2>Matched Price Tags</h2>
  <table>
    <thead><tr>{"".join(f"<th>{html.escape(col)}</th>" for col in display_cols)}</tr></thead>
    <tbody>{"".join(match_rows)}</tbody>
  </table>
  <h2>Crop Categories / OCR Zones</h2>
  <table>
    <thead><tr><th>track_id</th><th>zone</th><th>crop</th><th>prediction</th><th>source_text</th><th>engine</th></tr></thead>
    <tbody>{"".join(zone_rows)}</tbody>
  </table>
</body>
</html>
"""


def compute_metrics(matched: pd.DataFrame, predictions: pd.DataFrame, gt: pd.DataFrame) -> dict[str, Any]:
    matched_only = matched[matched["match_status"] == "matched"]
    field_metrics = {}
    for field in KEY_FIELDS:
        total = int(matched_only[f"gt_{field}"].apply(lambda value: not is_missing(value) and normalize_text(value) != "нет").sum())
        correct = int(matched_only[f"{field}_match"].sum())
        field_metrics[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
    default_total = 0
    default_within_1rub = 0
    for _, row in matched_only.iterrows():
        pred_default = parse_float(row.get("pred_price_default"))
        gt_default = parse_float(row.get("gt_price_default"))
        if math.isnan(gt_default):
            continue
        default_total += 1
        if not math.isnan(pred_default) and abs(pred_default - gt_default) <= 1.0:
            default_within_1rub += 1
    field_metrics["price_default_within_1rub"] = {
        "correct": default_within_1rub,
        "total": default_total,
        "accuracy": round(default_within_1rub / default_total, 4) if default_total else 0.0,
    }

    pass_80 = int(matched_only["pass_80_proxy"].sum())
    summary_cards = {
        "gt_tags": len(gt),
        "pred_tags": len(predictions),
        "matched_tags": len(matched_only),
        "extra_predictions": int((matched["match_status"] == "extra_prediction").sum()),
        "missed_gt": int((matched["match_status"] == "missed_ground_truth").sum()),
        "pass_80_proxy": f"{pass_80}/{len(gt)}",
        "pass_80_proxy_rate": round(pass_80 / len(gt), 4) if len(gt) else 0.0,
        "mean_field_accuracy_matched": round(float(matched_only["field_accuracy"].mean()), 4) if not matched_only.empty else 0.0,
        "mean_product_name_similarity": round(float(matched_only["product_name_similarity"].mean()), 4)
        if not matched_only.empty
        else 0.0,
    }
    fill_rates = {
        field: {
            "filled": int(predictions[field].apply(lambda value: not is_missing(value)).sum()) if field in predictions.columns else 0,
            "total": len(predictions),
        }
        for field in KEY_FIELDS
    }
    return {"summary_cards": summary_cards, "field_metrics": field_metrics, "fill_rates": fill_rates}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_dir = as_abs(root, args.run_dir)
    gt_csv = as_abs(root, args.gt_csv)
    output_dir = as_abs(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(gt_csv, encoding="utf-8-sig")
    predictions, debug, zones = load_predictions(run_dir)
    matched = match_predictions(predictions, gt, args.iou_threshold)
    zone_table = build_zone_table(predictions, debug, zones)
    metrics = compute_metrics(matched, predictions, gt)

    matched_csv = output_dir / "matched_predictions.csv"
    zone_csv = output_dir / "crop_category_predictions.csv"
    metrics_json = output_dir / "metrics.json"
    report_html = output_dir / "quality_43_15_report.html"

    matched.to_csv(matched_csv, index=False, encoding="utf-8-sig")
    zone_table.to_csv(zone_csv, index=False, encoding="utf-8-sig")
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_html.write_text(build_html(metrics, matched, zone_table, report_html), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"matched_csv={matched_csv}")
    print(f"zone_csv={zone_csv}")
    print(f"metrics_json={metrics_json}")
    print(f"report_html={report_html}")


if __name__ == "__main__":
    main()
