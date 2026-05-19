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


MODEL_DEFAULTS = {
    "eslav": Path("repro_outputs/quality_43_15/ocr_final_quality_core_fast_fixed"),
    "cyrillic": Path("repro_outputs/quality_43_15_cyrillic/ocr_final_quality_core_fast_fixed"),
    "tesseract": Path("repro_outputs/quality_43_15_tesseract/ocr_final_quality_core_fast_fixed"),
    "tesseract_crop_quality": Path(
        "repro_outputs/quality_43_15_tesseract_crop_quality_weighted_w005/ocr_final_quality_core_fast_fixed"
    ),
    "ppocr_synth": Path("repro_outputs/quality_43_15_ppocr_synth_cyrillic_v1/ocr_final_quality_core_fast_fixed"),
    "ppocr_real_lines_q075": Path(
        "repro_outputs/quality_43_15_ppocr_real_lines_q075/ocr_final_quality_core_fast_fixed"
    ),
    "ppocr_real_lines_q085": Path(
        "repro_outputs/quality_43_15_ppocr_real_lines_q085/ocr_final_quality_core_fast_fixed"
    ),
    "tesseract_finetune_q075": Path(
        "repro_outputs/quality_43_15_tesseract_finetune_q075/ocr_final_quality_core_fast_fixed"
    ),
    "tesseract_finetune_q075_raw": Path(
        "repro_outputs/quality_43_15_tesseract_finetune_q075_raw/ocr_final_quality_core_fast_fixed"
    ),
    "tesseract_catalog_v5": Path(
        "repro_outputs/quality_43_15_tesseract_catalog_recovery_v5_best/ocr_final_quality_core_fast_fixed"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare product_name OCR outputs across OCR experiments.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--matched-csv", type=Path, default=Path("repro_outputs/quality_43_15/evaluation/matched_predictions.csv"))
    parser.add_argument("--zones-manifest", type=Path, default=Path("repro_outputs/quality_43_15/ocr_zones_core_fixed/ocr_zones_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/quality_43_15/model_compare"))
    parser.add_argument("--models", default="eslav,cyrillic,tesseract")
    parser.add_argument("--stem", default="")
    parser.add_argument("--eslav-dir", type=Path, default=MODEL_DEFAULTS["eslav"])
    parser.add_argument("--cyrillic-dir", type=Path, default=MODEL_DEFAULTS["cyrillic"])
    parser.add_argument("--tesseract-dir", type=Path, default=MODEL_DEFAULTS["tesseract"])
    parser.add_argument("--tesseract-crop-quality-dir", type=Path, default=MODEL_DEFAULTS["tesseract_crop_quality"])
    parser.add_argument("--ppocr-synth-dir", type=Path, default=MODEL_DEFAULTS["ppocr_synth"])
    parser.add_argument("--ppocr-real-lines-q075-dir", type=Path, default=MODEL_DEFAULTS["ppocr_real_lines_q075"])
    parser.add_argument("--ppocr-real-lines-q085-dir", type=Path, default=MODEL_DEFAULTS["ppocr_real_lines_q085"])
    parser.add_argument("--tesseract-finetune-q075-dir", type=Path, default=MODEL_DEFAULTS["tesseract_finetune_q075"])
    parser.add_argument(
        "--tesseract-finetune-q075-raw-dir",
        type=Path,
        default=MODEL_DEFAULTS["tesseract_finetune_q075_raw"],
    )
    parser.add_argument("--tesseract-catalog-v5-dir", type=Path, default=MODEL_DEFAULTS["tesseract_catalog_v5"])
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


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("\u0451", "\u0435")
    text = re.sub(r"[^0-9a-z\u0400-\u04ff]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def similarity(left: Any, right: Any) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def read_changes(path: Path, prefix: str) -> pd.DataFrame:
    changes_path = path / "product_name_line_changes.csv"
    changes = pd.read_csv(changes_path, encoding="utf-8-sig")
    keep = ["track_id", "before", "after", "proposed", "accepted", "reason", "score", "confidence", "source", "raw_text"]
    for col in keep:
        if col not in changes.columns:
            changes[col] = ""
    changes = changes[keep].copy()
    changes["track_id"] = changes["track_id"].astype(str)
    return changes.rename(columns={col: f"{prefix}_{col}" for col in keep if col != "track_id"})


def rel_image(path_value: Any, report_path: Path) -> str:
    if is_missing(path_value):
        return ""
    path = Path(str(path_value))
    if not path.exists():
        return ""
    try:
        return path.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def img_tag(path_value: Any, report_path: Path) -> str:
    src = rel_image(path_value, report_path)
    if not src:
        return ""
    return f'<img src="{html.escape(src)}" loading="lazy" style="max-width:220px;max-height:150px">'


def build_html(summary: dict[str, Any], rows: pd.DataFrame, report_path: Path, models: list[str]) -> str:
    cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(str(key))}</div><div class='value'>{html.escape(str(value))}</div></div>"
        for key, value in summary.items()
    )
    columns = ["track_id", "full_tag", "gt_product_name"]
    for model in models:
        columns.extend([f"{model}_after", f"{model}_similarity_to_gt", f"{model}_confidence", f"{model}_source"])
    columns.extend(["winner_by_gt", "winner_margin"])

    body = []
    for _, row in rows.iterrows():
        cells = []
        for col in columns:
            if col == "full_tag":
                cells.append(f"<td>{img_tag(row.get(col), report_path)}</td>")
                continue
            value = "" if is_missing(row.get(col)) else str(row.get(col))
            css = ""
            if col == "winner_by_gt" and value:
                css = f" class='{html.escape(value)}'"
            cells.append(f"<td{css}>{html.escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    style_classes = "\n".join(f"    .{html.escape(model)} {{ color: #0b63ce; font-weight: 650; }}" for model in models)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Product Name OCR Comparison</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1 {{ margin: 0 0 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .card {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 12px; }}
    .label {{ color: #657282; font-size: 12px; }}
    .value {{ font-size: 24px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dde3ea; }}
    th, td {{ border-bottom: 1px solid #e5e9ef; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #edf1f5; z-index: 1; text-align: left; }}
    img {{ border: 1px solid #d5dce5; border-radius: 4px; background: #fff; }}
{style_classes}
    .tie {{ color: #657282; font-weight: 650; }}
    .unknown {{ color: #8a5a00; font-weight: 650; }}
  </style>
</head>
<body>
  <h1>Product Name OCR Comparison</h1>
  <p>Same product_name crops and same GT matching. Similarity is normalized character SequenceMatcher score.</p>
  <div class="cards">{cards}</div>
  <table>
    <thead><tr>{"".join(f"<th>{html.escape(col)}</th>" for col in columns)}</tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    model_dirs = {
        "eslav": as_abs(root, args.eslav_dir),
        "cyrillic": as_abs(root, args.cyrillic_dir),
        "tesseract": as_abs(root, args.tesseract_dir),
        "tesseract_crop_quality": as_abs(root, args.tesseract_crop_quality_dir),
        "ppocr_synth": as_abs(root, args.ppocr_synth_dir),
        "ppocr_real_lines_q075": as_abs(root, args.ppocr_real_lines_q075_dir),
        "ppocr_real_lines_q085": as_abs(root, args.ppocr_real_lines_q085_dir),
        "tesseract_finetune_q075": as_abs(root, args.tesseract_finetune_q075_dir),
        "tesseract_finetune_q075_raw": as_abs(root, args.tesseract_finetune_q075_raw_dir),
        "tesseract_catalog_v5": as_abs(root, args.tesseract_catalog_v5_dir),
    }
    output_dir = as_abs(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: pd.DataFrame | None = None
    for model in models:
        model_rows = read_changes(model_dirs[model], model)
        rows = model_rows if rows is None else rows.merge(model_rows, on="track_id", how="outer")
    if rows is None:
        raise ValueError("No models selected")

    zones = pd.read_csv(as_abs(root, args.zones_manifest), encoding="utf-8-sig")
    rank1 = zones[(zones["zone"] == "product_name") & (zones["rank"].astype(str) == "1")].copy()
    rank1["track_id"] = rank1["track_id"].astype(str)
    rows = rows.merge(rank1[["track_id", "full_tag", "zone_enhanced", "zone_raw"]], on="track_id", how="left")

    matched = pd.read_csv(as_abs(root, args.matched_csv), encoding="utf-8-sig")
    matched = matched[matched["match_status"] == "matched"].copy()
    matched["track_id"] = matched["track_id"].apply(lambda value: str(int(float(value))) if not is_missing(value) else "")
    gt_by_track = matched[["track_id", "gt_product_name"]].drop_duplicates("track_id")
    rows = rows.merge(gt_by_track, on="track_id", how="left")

    for model in models:
        rows[f"{model}_similarity_to_gt"] = rows.apply(
            lambda row, model_name=model: round(similarity(row.get(f"{model_name}_after"), row.get("gt_product_name")), 4),
            axis=1,
        )
        rows[f"{model}_strict80"] = rows[f"{model}_similarity_to_gt"] >= 0.80

    def winner(row: pd.Series) -> tuple[str, float]:
        if is_missing(row.get("gt_product_name")):
            return "unknown", 0.0
        scored = [(model, float(row.get(f"{model}_similarity_to_gt", 0.0))) for model in models]
        scored.sort(key=lambda item: item[1], reverse=True)
        if len(scored) > 1 and abs(scored[0][1] - scored[1][1]) < 0.02:
            return "tie", round(scored[0][1] - scored[1][1], 4)
        return scored[0][0], round(scored[0][1] - (scored[1][1] if len(scored) > 1 else 0.0), 4)

    winners = rows.apply(winner, axis=1, result_type="expand")
    rows["winner_by_gt"] = winners[0]
    rows["winner_margin"] = winners[1]
    rows = rows.sort_values(["winner_by_gt", "track_id"], kind="stable")

    known = rows[rows["winner_by_gt"] != "unknown"].copy()
    summary: dict[str, Any] = {
        "tracks_compared": len(rows),
        "tracks_with_gt_match": len(known),
    }
    for model in models:
        summary[f"{model}_wins"] = int((known["winner_by_gt"] == model).sum())
    summary["ties"] = int((known["winner_by_gt"] == "tie").sum())
    for model in models:
        summary[f"mean_{model}_similarity"] = round(float(known[f"{model}_similarity_to_gt"].mean()), 4) if len(known) else 0.0
        summary[f"{model}_strict80"] = f"{int(known[f'{model}_strict80'].sum())}/{len(known)}"

    stem = args.stem.strip() or "product_name_" + "_".join(models)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_summary.json"
    html_path = output_dir / f"{stem}.html"
    rows.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(build_html(summary, rows, html_path, models), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"html={html_path}")


if __name__ == "__main__":
    main()
