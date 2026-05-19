from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def track_id(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


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


def as_abs(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


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


def img_tag(path_value: Any, report_path: Path, css_class: str) -> str:
    src = rel_image(path_value, report_path)
    if not src:
        return "<span class='muted'>no image</span>"
    return f'<img class="{css_class}" src="{html.escape(src)}" loading="lazy">'


def zones_by_track(zones_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in zones_rows:
        if row.get("zone") != "product_name" or str(row.get("rank", "")).strip() != "1":
            continue
        result[track_id(row.get("track_id"))] = row
    return result


def matched_by_track(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("match_status") == "matched":
            result[track_id(row.get("track_id"))] = row
    return result


def load_top_candidates(value: str, limit: int = 3) -> str:
    if not value:
        return ""
    try:
        items = json.loads(value)
    except json.JSONDecodeError:
        return ""
    rendered = []
    for item in items[:limit]:
        meta = []
        if item.get("id"):
            meta.append(f"id={item.get('id')}")
        if item.get("weight_volume"):
            meta.append(str(item.get("weight_volume")))
        if item.get("price_promo_rub"):
            meta.append(f"promo={item.get('price_promo_rub')}")
        elif item.get("price_regular_rub"):
            meta.append(f"regular={item.get('price_regular_rub')}")
        suffix = f" ({', '.join(meta)})" if meta else ""
        rendered.append(f"{float(item.get('score', 0.0)):.3f} · {item.get('name', '')}{suffix}")
    return "\n".join(rendered)


def make_rows(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    recovery_rows = read_rows(as_abs(root, args.recovery_csv))
    zones = zones_by_track(read_rows(as_abs(root, args.zones_manifest)))
    baseline_gt = matched_by_track(read_rows(as_abs(root, args.baseline_matched_csv)))
    catalog_gt = matched_by_track(read_rows(as_abs(root, args.catalog_matched_csv)))

    rows: list[dict[str, Any]] = []
    for item in recovery_rows:
        tid = track_id(item.get("track_id"))
        zone = zones.get(tid, {})
        base_match = baseline_gt.get(tid, {})
        cat_match = catalog_gt.get(tid, {})
        gt_name = cat_match.get("gt_product_name") or base_match.get("gt_product_name") or ""
        before = item.get("ocr_text", "")
        after = item.get("final_text", "")
        accepted = str(item.get("accepted", "")).strip() == "1"
        before_sim = similarity(before, gt_name) if gt_name else 0.0
        after_sim = similarity(after, gt_name) if gt_name else 0.0
        delta = after_sim - before_sim
        if not gt_name:
            verdict = "no_gt"
        elif abs(delta) < 0.0001:
            verdict = "same"
        elif delta > 0:
            verdict = "better"
        else:
            verdict = "worse"
        rows.append(
            {
                "track_id": tid,
                "accepted": int(accepted),
                "verdict": verdict,
                "delta": round(delta, 4),
                "before_similarity": round(before_sim, 4),
                "after_similarity": round(after_sim, 4),
                "ocr_prediction": before,
                "catalog_prediction": after,
                "gt_product_name": gt_name,
                "catalog_score": item.get("score", ""),
                "catalog_margin": item.get("margin", ""),
                "catalog_reason": item.get("reason", ""),
                "catalog_candidate": item.get("candidate", ""),
                "second_candidate": item.get("second_candidate", ""),
                "tail_glass_packaging": item.get("tail_glass_packaging", ""),
                "top_candidates": load_top_candidates(item.get("top_candidates_json", "")),
                "full_tag": zone.get("full_tag", ""),
                "product_crop": zone.get("zone_enhanced") or zone.get("zone_raw", ""),
                "price_card": cat_match.get("pred_price_card") or base_match.get("pred_price_card") or "",
                "gt_price_card": cat_match.get("gt_price_card") or base_match.get("gt_price_card") or "",
            }
        )
    return sorted(rows, key=lambda row: (row["verdict"] != "better", row["accepted"] == 0, -float(row["delta"]), int(row["track_id"] or 0)))


def td_text(value: Any, class_name: str = "") -> str:
    css = f' class="{class_name}"' if class_name else ""
    text = "" if is_missing(value) else str(value)
    return f"<td{css}>{html.escape(text)}</td>"


def build_html(rows: list[dict[str, Any]], report_path: Path) -> str:
    total = len(rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    better = sum(1 for row in rows if row["verdict"] == "better")
    worse = sum(1 for row in rows if row["verdict"] == "worse")
    same = sum(1 for row in rows if row["verdict"] == "same")
    no_gt = sum(1 for row in rows if row["verdict"] == "no_gt")

    cards = {
        "rows": total,
        "catalog replacements": accepted,
        "better": better,
        "worse": worse,
        "same": same,
        "no gt": no_gt,
    }
    card_html = "".join(
        f"<div class='card'><div class='label'>{html.escape(key)}</div><div class='value'>{value}</div></div>"
        for key, value in cards.items()
    )

    body = []
    for row in rows:
        accepted_label = "replaced" if row["accepted"] else "unchanged"
        verdict = str(row["verdict"])
        body.append(
            "<tr "
            f"data-verdict='{html.escape(verdict)}' "
            f"data-accepted='{row['accepted']}'>"
            + td_text(row["track_id"], "track")
            + f"<td>{img_tag(row['full_tag'], report_path, 'tag-img')}</td>"
            + f"<td>{img_tag(row['product_crop'], report_path, 'crop-img')}</td>"
            + td_text(accepted_label, "accepted" if row["accepted"] else "unchanged")
            + td_text(verdict, verdict)
            + td_text(row["delta"], "delta")
            + td_text(row["before_similarity"])
            + td_text(row["after_similarity"])
            + td_text(row["ocr_prediction"], "text-cell")
            + td_text(row["catalog_prediction"], "text-cell")
            + td_text(row["gt_product_name"], "text-cell gt")
            + td_text(row["catalog_score"])
            + td_text(row["catalog_margin"])
            + td_text(row["catalog_reason"])
            + td_text(row["tail_glass_packaging"])
            + td_text(row["catalog_candidate"], "text-cell small")
            + td_text(row["second_candidate"], "text-cell small")
            + td_text(row["top_candidates"], "text-cell small pre")
            + td_text(row["price_card"])
            + td_text(row["gt_price_card"])
            + "</tr>"
        )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Catalog Recovery Review</title>
  <style>
    body {{ margin: 24px; font-family: Segoe UI, Arial, sans-serif; background: #f5f7fa; color: #1f2933; }}
    h1 {{ margin: 0 0 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }}
    .card {{ background: #fff; border: 1px solid #dbe3ed; border-radius: 8px; padding: 10px; }}
    .label {{ color: #64748b; font-size: 12px; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    .toolbar {{ display: flex; gap: 8px; align-items: center; margin: 14px 0; flex-wrap: wrap; }}
    input, select {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; font: inherit; background: white; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe3ed; }}
    th, td {{ border-bottom: 1px solid #e6ebf2; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; z-index: 1; background: #eef3f8; text-align: left; }}
    .tag-img {{ width: 230px; max-height: 190px; object-fit: contain; border: 1px solid #d6dee8; border-radius: 4px; background: #fff; }}
    .crop-img {{ width: 230px; max-height: 95px; object-fit: contain; border: 1px solid #d6dee8; border-radius: 4px; background: #fff; }}
    .text-cell {{ min-width: 230px; max-width: 360px; line-height: 1.35; }}
    .small {{ min-width: 180px; max-width: 300px; font-size: 12px; }}
    .pre {{ white-space: pre-wrap; }}
    .gt {{ font-weight: 600; }}
    .better {{ color: #087443; font-weight: 650; }}
    .worse {{ color: #b42318; font-weight: 650; }}
    .same {{ color: #64748b; font-weight: 650; }}
    .no_gt {{ color: #8a5a00; font-weight: 650; }}
    .accepted {{ color: #0b63ce; font-weight: 650; }}
    .unchanged {{ color: #64748b; }}
    .delta {{ font-weight: 650; }}
    .muted {{ color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Catalog Recovery Review</h1>
  <div class="cards">{card_html}</div>
  <div class="toolbar">
    <input id="search" placeholder="Поиск по тексту / track_id" size="36">
    <select id="filter">
      <option value="all">Все строки</option>
      <option value="replaced">Только замененные</option>
      <option value="unchanged">Только без замены</option>
      <option value="better">Стало лучше</option>
      <option value="worse">Стало хуже</option>
      <option value="same">Без изменения</option>
      <option value="no_gt">Без GT</option>
    </select>
  </div>
  <table id="table">
    <thead>
      <tr>
        <th>track</th>
        <th>ценник</th>
        <th>product crop</th>
        <th>замена</th>
        <th>итог</th>
        <th>delta</th>
        <th>sim до</th>
        <th>sim после</th>
        <th>наш Tesseract pred</th>
        <th>после каталога</th>
        <th>правильный ответ</th>
        <th>score</th>
        <th>margin</th>
        <th>reason</th>
        <th>ст/б from OCR</th>
        <th>best catalog</th>
        <th>second catalog</th>
        <th>top candidates</th>
        <th>pred card price</th>
        <th>gt card price</th>
      </tr>
    </thead>
    <tbody>{"".join(body)}</tbody>
  </table>
  <script>
    const search = document.getElementById('search');
    const filter = document.getElementById('filter');
    const rows = Array.from(document.querySelectorAll('#table tbody tr'));
    function applyFilter() {{
      const q = search.value.trim().toLowerCase();
      const mode = filter.value;
      for (const row of rows) {{
        const accepted = row.dataset.accepted === '1';
        const verdict = row.dataset.verdict;
        const text = row.textContent.toLowerCase();
        let ok = !q || text.includes(q);
        if (mode === 'replaced') ok = ok && accepted;
        else if (mode === 'unchanged') ok = ok && !accepted;
        else if (mode !== 'all') ok = ok && verdict === mode;
        row.style.display = ok ? '' : 'none';
      }}
    }}
    search.addEventListener('input', applyFilter);
    filter.addEventListener('change', applyFilter);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual report for catalog product-name recovery.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--recovery-csv",
        type=Path,
        default=Path(
            "repro_outputs/quality_43_15_tesseract_catalog_recovery_v5_best/"
            "ocr_final_quality_core_fast_fixed/product_name_catalog_recovery.csv"
        ),
    )
    parser.add_argument(
        "--baseline-matched-csv",
        type=Path,
        default=Path("repro_outputs/quality_43_15_tesseract_crop_quality_weighted_w005/evaluation/matched_predictions.csv"),
    )
    parser.add_argument(
        "--catalog-matched-csv",
        type=Path,
        default=Path("repro_outputs/quality_43_15_tesseract_catalog_recovery_v5_best/evaluation/matched_predictions.csv"),
    )
    parser.add_argument(
        "--zones-manifest",
        type=Path,
        default=Path("repro_outputs/quality_43_15/ocr_zones_core_fixed/ocr_zones_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repro_outputs/quality_43_15_tesseract_catalog_recovery_v5_best/manual_review"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = as_abs(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = make_rows(root, args)
    csv_path = output_dir / "catalog_recovery_visual_review.csv"
    html_path = output_dir / "catalog_recovery_visual_review.html"
    write_rows(csv_path, rows)
    html_path.write_text(build_html(rows, html_path), encoding="utf-8")
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"html={html_path}")


if __name__ == "__main__":
    main()
