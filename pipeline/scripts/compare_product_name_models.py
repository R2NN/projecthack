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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare product_name OCR outputs for two PaddleOCR recognition models.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--eslav-dir",
        type=Path,
        default=Path("repro_outputs/quality_43_15/ocr_final_quality_core_fast_fixed"),
    )
    parser.add_argument(
        "--cyrillic-dir",
        type=Path,
        default=Path("repro_outputs/quality_43_15_cyrillic/ocr_final_quality_core_fast_fixed"),
    )
    parser.add_argument(
        "--matched-csv",
        type=Path,
        default=Path("repro_outputs/quality_43_15/evaluation/matched_predictions.csv"),
    )
    parser.add_argument(
        "--zones-manifest",
        type=Path,
        default=Path("repro_outputs/quality_43_15/ocr_zones_core_fixed/ocr_zones_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("repro_outputs/quality_43_15/model_compare"))
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
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def similarity(left: Any, right: Any) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def read_changes(path: Path, prefix: str) -> pd.DataFrame:
    changes = pd.read_csv(path / "product_name_line_changes.csv", encoding="utf-8-sig")
    keep = ["track_id", "before", "after", "proposed", "accepted", "reason", "score", "confidence", "source", "raw_text"]
    missing = [col for col in keep if col not in changes.columns]
    for col in missing:
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


def build_html(summary: dict[str, Any], rows: pd.DataFrame, report_path: Path) -> str:
    cards = "\n".join(
        f"<div class='card'><div class='label'>{html.escape(str(key))}</div><div class='value'>{html.escape(str(value))}</div></div>"
        for key, value in summary.items()
    )
    columns = [
        "track_id",
        "full_tag",
        "gt_product_name",
        "eslav_after",
        "cyrillic_after",
        "eslav_similarity_to_gt",
        "cyrillic_similarity_to_gt",
        "winner_by_gt",
        "eslav_confidence",
        "cyrillic_confidence",
        "eslav_raw_text",
        "cyrillic_raw_text",
    ]
    body = []
    for _, row in rows.iterrows():
        cells = []
        for col in columns:
            if col == "full_tag":
                cells.append(f"<td>{img_tag(row.get(col), report_path)}</td>")
            else:
                value = "" if is_missing(row.get(col)) else str(row.get(col))
                css = ""
                if col == "winner_by_gt" and value:
                    css = f" class='{html.escape(value)}'"
                cells.append(f"<td{css}>{html.escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Product Name OCR: eslav vs cyrillic</title>
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
    .eslav {{ color: #0b63ce; font-weight: 650; }}
    .cyrillic {{ color: #0b7a3b; font-weight: 650; }}
    .tie {{ color: #657282; font-weight: 650; }}
    .unknown {{ color: #8a5a00; font-weight: 650; }}
  </style>
</head>
<body>
  <h1>Product Name OCR: eslav vs cyrillic</h1>
  <p>Сравнение построено на одних и тех же product_name кропах и track_id. Winner считается только там, где есть автоматически сматченный GT из отчета.</p>
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
    eslav_dir = as_abs(root, args.eslav_dir)
    cyrillic_dir = as_abs(root, args.cyrillic_dir)
    matched_csv = as_abs(root, args.matched_csv)
    zones_manifest = as_abs(root, args.zones_manifest)
    output_dir = as_abs(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eslav = read_changes(eslav_dir, "eslav")
    cyrillic = read_changes(cyrillic_dir, "cyrillic")
    rows = eslav.merge(cyrillic, on="track_id", how="outer")

    zones = pd.read_csv(zones_manifest, encoding="utf-8-sig")
    rank1 = zones[(zones["zone"] == "product_name") & (zones["rank"].astype(str) == "1")].copy()
    rank1["track_id"] = rank1["track_id"].astype(str)
    rows = rows.merge(rank1[["track_id", "full_tag", "zone_enhanced", "zone_raw"]], on="track_id", how="left")

    matched = pd.read_csv(matched_csv, encoding="utf-8-sig")
    matched = matched[matched["match_status"] == "matched"].copy()
    matched["track_id"] = matched["track_id"].apply(lambda value: str(int(float(value))) if not is_missing(value) else "")
    gt_by_track = matched[["track_id", "gt_product_name", "product_name_similarity"]].drop_duplicates("track_id")
    rows = rows.merge(gt_by_track, on="track_id", how="left")

    rows["eslav_similarity_to_gt"] = rows.apply(lambda row: round(similarity(row.get("eslav_after"), row.get("gt_product_name")), 4), axis=1)
    rows["cyrillic_similarity_to_gt"] = rows.apply(
        lambda row: round(similarity(row.get("cyrillic_after"), row.get("gt_product_name")), 4),
        axis=1,
    )

    def winner(row: pd.Series) -> str:
        if is_missing(row.get("gt_product_name")):
            return "unknown"
        left = float(row.get("eslav_similarity_to_gt", 0))
        right = float(row.get("cyrillic_similarity_to_gt", 0))
        if abs(left - right) < 0.02:
            return "tie"
        return "cyrillic" if right > left else "eslav"

    rows["winner_by_gt"] = rows.apply(winner, axis=1)
    rows["similarity_delta_cyrillic_minus_eslav"] = rows["cyrillic_similarity_to_gt"] - rows["eslav_similarity_to_gt"]
    rows = rows.sort_values(["winner_by_gt", "track_id"], kind="stable")

    known = rows[rows["winner_by_gt"] != "unknown"]
    summary = {
        "tracks_compared": len(rows),
        "tracks_with_gt_match": len(known),
        "cyrillic_wins": int((known["winner_by_gt"] == "cyrillic").sum()),
        "eslav_wins": int((known["winner_by_gt"] == "eslav").sum()),
        "ties": int((known["winner_by_gt"] == "tie").sum()),
        "mean_eslav_similarity": round(float(known["eslav_similarity_to_gt"].mean()), 4) if len(known) else 0.0,
        "mean_cyrillic_similarity": round(float(known["cyrillic_similarity_to_gt"].mean()), 4) if len(known) else 0.0,
    }

    csv_path = output_dir / "product_name_eslav_vs_cyrillic.csv"
    json_path = output_dir / "product_name_eslav_vs_cyrillic_summary.json"
    html_path = output_dir / "product_name_eslav_vs_cyrillic.html"
    rows.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(build_html(summary, rows, html_path), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"html={html_path}")


if __name__ == "__main__":
    main()
