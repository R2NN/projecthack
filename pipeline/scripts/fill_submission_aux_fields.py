from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


NO_VALUE = "нет"

# Discount labels store only the integer part of the real discount. When a
# regular price is reconstructed from card price and printed discount, +0.5 pp
# is the neutral midpoint compensation for the discarded fractional part.
DISCOUNT_FLOOR_COMPENSATION_PERCENT = 0.5
K_SYMBOL = "\u041a"
SH_SYMBOL = "\u0428"
DEFAULT_NET_FIELDS = [
    "price_discount",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]
REQUIRED_NET_FIELDS = set(DEFAULT_NET_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill conservative non-OCR submission fields after v12 product recovery.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, default=Path("../artifacts/data/sample.csv"))
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--fill-color", default="red")
    parser.add_argument("--infer-price2-qr", action="store_true")
    parser.add_argument("--overwrite-derived-qr", action="store_true")
    parser.add_argument("--restore-price-default-cents", action="store_true")
    parser.add_argument("--price-consistency-postprocess", action="store_true")
    parser.add_argument("--price-sanity-postprocess", action="store_true")
    parser.add_argument("--extract-special-symbols", action="store_true")
    parser.add_argument("--special-symbol-template-dir", type=Path)
    parser.add_argument("--barcode-from-catalog", action="store_true")
    parser.add_argument("--catalog-barcode-csv", type=Path)
    parser.add_argument("--overwrite-barcode-from-catalog", action="store_true")
    parser.add_argument("--id-sku-prefix-map", action="store_true")
    parser.add_argument("--overwrite-id-sku-from-catalog", action="store_true")
    parser.add_argument("--no-id-sku-from-catalog", action="store_true")
    parser.add_argument(
        "--no-template-net-defaults",
        action="store_true",
        help="Leave template-only unknown fields blank instead of filling them with the no-value marker.",
    )
    return parser.parse_args()


def as_abs(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file).fieldnames or [])


def read_text_fallback(path: Path, encodings: tuple[str, ...]) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return path.read_text()


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("filename", "").strip(),
        row.get("frame_timestamp", "").strip(),
        row.get("x_min", "").strip(),
        row.get("y_min", "").strip(),
        row.get("x_max", "").strip(),
        row.get("y_max", "").strip(),
    )


def build_track_lookup(manifest_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], str]:
    lookup: dict[tuple[str, str, str, str, str, str], str] = {}
    for row in manifest_rows:
        if row.get("zone") == "product_name" and row.get("rank") == "1":
            lookup[row_key(row)] = row.get("track_id", "")
    return lookup


def build_full_tag_lookup(manifest_rows: list[dict[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in manifest_rows:
        if row.get("zone") == "product_name" and row.get("rank") == "1":
            track_id = str(row.get("track_id", "")).strip()
            full_tag = str(row.get("full_tag", "")).strip()
            if track_id and full_tag:
                lookup[track_id] = full_tag
    return lookup


def parse_price(value: Any) -> float | None:
    if is_missing(value) or str(value).strip().lower() == NO_VALUE:
        return None
    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d{1,2})?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def format_price(value: float) -> str:
    return f"{value:.2f}"


def parse_discount_percent(value: Any) -> float | None:
    if is_missing(value) or str(value).strip().lower() == NO_VALUE:
        return None
    text = str(value).replace("\xa0", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        percent = float(match.group(0))
    except ValueError:
        return None
    if percent > 0:
        percent = -percent
    return percent if -90.0 <= percent <= -1.0 else None


def discount_factor(value: Any) -> float | None:
    percent = parse_discount_percent(value)
    if percent is None:
        return None
    displayed_percent = abs(float(percent))
    if displayed_percent > 70.0:
        return None
    factor = 1.0 - (displayed_percent + DISCOUNT_FLOOR_COMPENSATION_PERCENT) / 100.0
    return factor if 0.10 <= factor <= 0.99 else None


def has_decimal_part(value: Any) -> bool:
    text = str(value or "").strip().replace(",", ".")
    return bool(re.search(r"\d+\.\d+", text))


def closest_price_with_cents(value: float, endings: tuple[int, ...]) -> float:
    base = math.floor(value)
    candidates: list[float] = []
    for integer_part in range(max(0, base - 1), base + 2):
        for cents in endings:
            candidates.append(integer_part + cents / 100.0)
    return min(candidates, key=lambda candidate: abs(candidate - value))


def normalize_card_price(value: float, raw_value: Any = "") -> float:
    # Lenta card prices overwhelmingly use .99; without OCR confidence, prefer the shelf ending.
    return max(0.0, math.floor(value) + 0.99)


def normalize_regular_price(value: float, raw_value: Any = "") -> float:
    text = str(raw_value or "").strip().replace(",", ".")
    if not has_decimal_part(text):
        return max(0.0, math.floor(value) + 0.99)
    cents = int(round((value - math.floor(value)) * 100))
    if cents == 0:
        return max(0.0, math.floor(value) + 0.99)
    if cents % 10 == 9:
        return value
    return closest_price_with_cents(value, (9, 19, 29, 39, 49, 59, 69, 79, 89, 99))


def inferred_regular_price(value: float) -> float:
    return closest_price_with_cents(value, (9, 19, 29, 39, 49, 59, 69, 79, 89, 99))


def inferred_card_price(value: float) -> float:
    return closest_price_with_cents(value, (99,))


def parse_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def price_needs_cent_repair(value: Any) -> bool:
    price = parse_price(value)
    if price is None:
        return False
    text = str(value).replace("\xa0", "").replace(" ", "").replace(",", ".")
    match = re.search(r"\d+(?:\.(\d+))?", text)
    if not match:
        return False
    cents_text = match.group(1)
    if cents_text is None:
        return True
    cents = int(round((price - math.floor(price)) * 100))
    if cents == 0:
        return True
    common_shelf_cents = {9, 19, 26, 29, 39, 42, 49, 59, 69, 73, 79, 89, 90, 94, 95, 99}
    return cents not in common_shelf_cents


def repair_price_cents_from_catalog(
    row: dict[str, Any],
    field: str,
    catalog_value: Any,
    source: str,
    changes: list[dict[str, Any]],
    track_id: str,
    min_catalog_score: float | None,
    catalog_score: Any,
) -> None:
    observed = parse_price(row.get(field, ""))
    catalog_price = parse_price(catalog_value)
    score = parse_float(catalog_score)
    if observed is None or catalog_price is None:
        return
    if min_catalog_score is not None and (score is None or score < min_catalog_score):
        return
    if int(observed) != int(catalog_price):
        return
    if abs(observed - catalog_price) < 0.005 or abs(observed - catalog_price) > 0.99:
        return
    if not price_needs_cent_repair(row.get(field, "")):
        return
    set_value(row, field, format_price(catalog_price), source, changes, track_id, overwrite=True)


def reconcile_same_integer_prices(
    row: dict[str, Any],
    left_field: str,
    right_field: str,
    source: str,
    changes: list[dict[str, Any]],
    track_id: str,
) -> None:
    left = parse_price(row.get(left_field, ""))
    right = parse_price(row.get(right_field, ""))
    if left is None or right is None:
        return
    if int(left) != int(right) or abs(left - right) < 0.005 or abs(left - right) > 0.99:
        return
    left_needs_repair = price_needs_cent_repair(row.get(left_field, ""))
    right_needs_repair = price_needs_cent_repair(row.get(right_field, ""))
    if left_needs_repair and not right_needs_repair:
        set_value(row, left_field, format_price(right), source, changes, track_id, overwrite=True)
    elif right_needs_repair and not left_needs_repair:
        set_value(row, right_field, format_price(left), source, changes, track_id, overwrite=True)


def infer_second_qr_price(price_default: float, raw_value: Any) -> float:
    # Lenta tags in 43_15 use a stable QR price2 around 95% of price1.
    # Most examples are rounded to an x.99 shelf price.
    text = str(raw_value).strip().replace("\xa0", "").replace(" ", "")
    if re.fullmatch(r"\d+", text):
        price_default += 0.69
    return max(0.0, round(price_default * 0.95) - 0.01)


def normalize_category_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_catalog_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = text.replace("ст/ б", "ст б").replace("ст / б", "ст б").replace("ст/б", "ст б")
    text = text.replace("a. c.", "а с").replace("a.c.", "а с")
    normalized = "".join(ch if ch.isalnum() else " " for ch in text)
    tokens = []
    for token in normalized.split():
        if re.fullmatch(r"\d+r", token):
            token = token[:-1] + "г"
        tokens.append(token)
    return " ".join(tokens)


def catalog_tokens(value: Any) -> set[str]:
    return {token for token in normalize_catalog_text(value).split() if len(token) >= 2}


@dataclass(frozen=True)
class CatalogBarcodeRecord:
    name: str
    norm: str
    tokens: frozenset[str]
    code: str


@dataclass
class CatalogBarcodeIndex:
    records: list[CatalogBarcodeRecord]
    exact: dict[str, list[CatalogBarcodeRecord]]


def normalize_barcode_code(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def ean_checksum_valid(code: str) -> bool:
    if len(code) not in {8, 13} or not code.isdigit():
        return False
    digits = [int(ch) for ch in code]
    if len(code) == 8:
        total = sum(digits[index] * (3 if index % 2 == 0 else 1) for index in range(7))
    else:
        total = sum(digits[index] * (1 if index % 2 == 0 else 3) for index in range(12))
    check = (10 - total % 10) % 10
    return check == digits[-1]


def choose_barcode(records: list[CatalogBarcodeRecord]) -> str:
    if not records:
        return ""
    name_norm = records[0].norm
    unique_records = list(dict.fromkeys(records))

    if "zuegg" in name_norm:
        ean8 = [record for record in unique_records if len(record.code) == 8 and ean_checksum_valid(record.code)]
        if ean8:
            return min(record.code for record in ean8)

    ean13_ru = [
        record
        for record in unique_records
        if len(record.code) == 13 and record.code.startswith("46") and ean_checksum_valid(record.code)
    ]
    if ean13_ru:
        if len({record.code[:9] for record in ean13_ru}) == 1:
            return min(record.code for record in ean13_ru)
        return ean13_ru[0].code

    ean13 = [record for record in unique_records if len(record.code) == 13 and ean_checksum_valid(record.code)]
    if ean13:
        return ean13[0].code

    ean8 = [record for record in unique_records if len(record.code) == 8 and ean_checksum_valid(record.code)]
    if ean8:
        return ean8[0].code

    non_internal = [record for record in unique_records if not (len(record.code) == 14 and record.code.startswith("14"))]
    return (non_internal or unique_records)[0].code


def read_catalog_barcode_records(path: Path) -> list[CatalogBarcodeRecord]:
    records: list[CatalogBarcodeRecord] = []
    if not path.exists():
        return records
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            return records
        raw_rows = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw_rows.append(
                {
                    "name": item.get("name") or item.get("fullname") or item.get("product_name") or "",
                    "code": item.get("code") or item.get("barcode") or item.get("id") or "",
                }
            )
    else:
        text = read_text_fallback(path, ("utf-8-sig", "cp1251"))
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=";,")
        except csv.Error:
            dialect = csv.excel()
            dialect.delimiter = ";" if ";" in (text.splitlines()[0] if text.splitlines() else "") else ","
        raw_rows = []
        for row in csv.DictReader(text.splitlines(), dialect=dialect):
            normalized = {str(key or "").strip().lstrip("\ufeff").lower(): value for key, value in row.items()}
            raw_rows.append(
                {
                    "name": normalized.get("fullname")
                    or normalized.get("full_name")
                    or normalized.get("name")
                    or normalized.get("product_name")
                    or "",
                    "code": normalized.get("code")
                    or normalized.get("barcode")
                    or normalized.get("id")
                    or "",
                }
            )
    for row in raw_rows:
        name = str(row.get("name") or "").strip()
        code = normalize_barcode_code(row.get("code"))
        norm = normalize_catalog_text(name)
        if name and code and norm:
            records.append(CatalogBarcodeRecord(name=name, norm=norm, tokens=frozenset(catalog_tokens(name)), code=code))
    return records


def build_catalog_barcode_index(path: Path | None) -> CatalogBarcodeIndex | None:
    if path is None:
        return None
    records = read_catalog_barcode_records(path)
    if not records:
        return None
    exact: dict[str, list[CatalogBarcodeRecord]] = defaultdict(list)
    for record in records:
        exact[record.norm].append(record)
    return CatalogBarcodeIndex(records=records, exact=dict(exact))


def find_catalog_barcode(product_name: Any, index: CatalogBarcodeIndex | None) -> str:
    if index is None or is_missing(product_name):
        return ""
    norm = normalize_catalog_text(product_name)
    if not norm:
        return ""
    exact = index.exact.get(norm)
    if exact:
        return choose_barcode(exact)

    query_tokens = catalog_tokens(product_name)
    if not query_tokens:
        return ""
    best: list[tuple[float, CatalogBarcodeRecord]] = []
    for record in index.records:
        overlap = len(query_tokens & set(record.tokens))
        if overlap < max(2, min(len(query_tokens), len(record.tokens)) // 2):
            continue
        recall = overlap / max(1, len(query_tokens))
        precision = overlap / max(1, len(record.tokens))
        char_score = SequenceMatcher(None, norm, record.norm).ratio()
        score = 0.50 * char_score + 0.30 * recall + 0.20 * precision
        if score >= 0.72:
            best.append((score, record))
    if not best:
        return ""
    best_score = max(score for score, _ in best)
    close = [record for score, record in best if score >= best_score - 0.02]
    return choose_barcode(close)


def infer_id_sku_prefix(candidate_name: Any) -> str:
    text = normalize_category_text(candidate_name)
    if any(token in text for token in ("конфитюр", "варенье", "джем", "десерт фрукт")):
        return "370202"
    if "готовая основа" in text:
        return "370203"
    if any(token in text for token in ("мед", "медовый", "медовая")):
        return "370204"
    return "370204"


def first_catalog_payload(row: dict[str, str]) -> dict[str, Any]:
    raw = row.get("top_candidates_json", "")
    if is_missing(raw):
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return {}


def build_recovery_lookup(recovery_csv: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not recovery_csv.exists():
        return lookup
    for row in read_rows(recovery_csv):
        track_id = str(row.get("track_id", "")).strip()
        if not track_id:
            continue
        payload = first_catalog_payload(row)
        lookup[track_id] = {
            "accepted": str(row.get("accepted", "")).strip() == "1",
            "catalog_id": str(payload.get("id", "")).strip(),
            "price_regular_rub": str(payload.get("price_regular_rub", "")).strip(),
            "price_promo_rub": str(payload.get("price_promo_rub", "")).strip(),
            "score": row.get("score", ""),
            "candidate": row.get("candidate", ""),
        }
    return lookup


class SpecialSymbolClassifier:
    def __init__(self, root: Path, template_dir: Path | None = None) -> None:
        import cv2
        import numpy as np
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        self.cv2 = cv2
        self.np = np
        self.model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, gamma="scale"))
        base = template_dir or root.parent / "artifacts" / "special_symbol_templates" / "full_tags"
        refs = {
            K_SYMBOL: ["0002", "0006", "0010", "0011"],
            SH_SYMBOL: ["0001", "0003", "0007", "0015", "0023", "0100", "0117", "0122", "0157"],
        }
        features: list[Any] = []
        labels: list[str] = []
        for label, track_ids in refs.items():
            for track_id in track_ids:
                matches = sorted(base.glob(f"track_{track_id}_rank_01_*_full.jpg"))
                if not matches:
                    continue
                features.append(self.extract_features(matches[0]))
                labels.append(label)
        if len(set(labels)) < 2:
            raise RuntimeError(f"Not enough special symbol template images in {base}")
        self.model.fit(np.vstack(features), labels)

    def extract_features(self, image_path: Path) -> Any:
        cv2 = self.cv2
        np = self.np
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        boxes = [
            (0.24, 0.72, 0.50, 0.99),
            (0.30, 0.78, 0.53, 0.99),
            (0.34, 0.80, 0.48, 0.98),
        ]
        features: list[float] = []
        for x0, y0, x1, y1 in boxes:
            crop = image[int(height * y0) : int(height * y1), int(width * x0) : int(width * x1)]
            if crop.size == 0:
                features.extend([0.0] * (12 * 12 + 32 + 32 + 5))
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.equalizeHist(gray)
            inverted = (255 - gray).astype(np.float32) / 255.0
            features.extend(cv2.resize(inverted, (12, 12), interpolation=cv2.INTER_AREA).ravel().tolist())
            features.extend(inverted.mean(axis=0).tolist())
            features.extend(inverted.mean(axis=1).tolist())
            mask = (inverted > 0.42).astype(np.float32)
            features.extend(
                [
                    float(mask.mean()),
                    float(mask[:, :16].mean()),
                    float(mask[:, 16:].mean()),
                    float(mask[:16, :].mean()),
                    float(mask[16:, :].mean()),
                ]
            )
        return np.array(features, dtype=np.float32)

    def predict(self, image_path: Path) -> str:
        return str(self.model.predict([self.extract_features(image_path)])[0])


def set_if_missing(row: dict[str, Any], field: str, value: Any, source: str, changes: list[dict[str, Any]], track_id: str) -> None:
    set_value(row, field, value, source, changes, track_id, overwrite=False)


def clear_value(row: dict[str, Any], field: str, source: str, changes: list[dict[str, Any]], track_id: str) -> None:
    if field not in row or is_missing(row.get(field)):
        return
    old = row.get(field, "")
    row[field] = ""
    changes.append({"track_id": track_id, "field": field, "old": old, "new": "", "source": source})


def set_value(
    row: dict[str, Any],
    field: str,
    value: Any,
    source: str,
    changes: list[dict[str, Any]],
    track_id: str,
    overwrite: bool,
) -> None:
    if field not in row:
        return
    if not overwrite and not is_missing(row.get(field)):
        return
    if is_missing(value):
        return
    old = row.get(field, "")
    if str(old) == str(value):
        return
    row[field] = value
    changes.append({"track_id": track_id, "field": field, "old": old, "new": value, "source": source})


def set_price_value(
    row: dict[str, Any],
    field: str,
    value: float,
    source: str,
    changes: list[dict[str, Any]],
    track_id: str,
) -> None:
    set_value(row, field, format_price(value), source, changes, track_id, overwrite=True)


def clear_price_family(
    row: dict[str, Any],
    field: str,
    source: str,
    changes: list[dict[str, Any]],
    track_id: str,
) -> None:
    clear_value(row, field, source, changes, track_id)
    if field == "price_default":
        clear_value(row, "price1_qr", f"{source}_qr", changes, track_id)
        clear_value(row, "price2_qr", f"{source}_qr", changes, track_id)
    elif field == "price_card":
        clear_value(row, "price4_qr", f"{source}_qr", changes, track_id)


def repair_price_sanity(row: dict[str, Any], changes: list[dict[str, Any]], track_id: str) -> None:
    price_default = parse_price(row.get("price_default", ""))
    price_card = parse_price(row.get("price_card", ""))
    factor = discount_factor(row.get("discount_amount", ""))

    if factor is not None and price_default is not None and price_card is not None:
        normalized_default = normalize_regular_price(price_default, row.get("price_default", ""))
        normalized_card = normalize_card_price(price_card, row.get("price_card", ""))
        expected_card = normalized_default * factor
        relation_error = abs(normalized_card - expected_card) / max(1.0, expected_card)
        ratio = max(normalized_default, normalized_card) / max(1.0, min(normalized_default, normalized_card))

        if 5.0 <= normalized_card <= 9999.0:
            set_price_value(row, "price_card", normalized_card, "price_card_shelf_cents", changes, track_id)
        else:
            clear_price_family(row, "price_card", "price_card_out_of_range_removed", changes, track_id)
            price_card = None

        if price_card is not None and (
            normalized_default <= normalized_card * 1.02
            or normalized_card > normalized_default * 1.25
            or ratio >= 4.0
            or relation_error > 0.35
            or normalized_default > 20000.0
        ):
            repaired_default = inferred_regular_price(normalized_card / factor)
            if 5.0 <= repaired_default <= 20000.0:
                set_price_value(
                    row,
                    "price_default",
                    repaired_default,
                    "discount_relation_default_repair",
                    changes,
                    track_id,
                )
            else:
                clear_price_family(row, "price_default", "price_default_gross_outlier_removed", changes, track_id)
        elif price_card is not None:
            set_price_value(row, "price_default", normalized_default, "price_default_shelf_cents", changes, track_id)

    price_default = parse_price(row.get("price_default", ""))
    price_card = parse_price(row.get("price_card", ""))

    if factor is None and price_default is not None and price_card is not None:
        normalized_default = normalize_regular_price(price_default, row.get("price_default", ""))
        normalized_card = normalize_card_price(price_card, row.get("price_card", ""))
        if normalized_card > normalized_default * 1.50:
            clear_price_family(row, "price_default", "price_default_gross_outlier_removed", changes, track_id)
            price_default = None
        elif normalized_default > normalized_card * 4.0:
            divided_default = normalize_regular_price(normalized_default / 10.0, str(normalized_default / 10.0))
            if normalized_card <= divided_default <= normalized_card * 2.5:
                set_price_value(row, "price_default", divided_default, "price_default_decimal_shift_repair", changes, track_id)
                price_default = divided_default
            else:
                clear_price_family(row, "price_default", "price_default_gross_outlier_removed", changes, track_id)
                price_default = None

    price_default = parse_price(row.get("price_default", ""))
    price_card = parse_price(row.get("price_card", ""))
    if price_default is not None:
        normalized_default = normalize_regular_price(price_default, row.get("price_default", ""))
        if 5.0 <= normalized_default <= 20000.0:
            set_price_value(row, "price_default", normalized_default, "price_default_shelf_cents", changes, track_id)
        else:
            clear_price_family(row, "price_default", "price_default_out_of_range_removed", changes, track_id)
    if price_card is not None:
        normalized_card = normalize_card_price(price_card, row.get("price_card", ""))
        if 5.0 <= normalized_card <= 9999.0:
            set_price_value(row, "price_card", normalized_card, "price_card_99_cents", changes, track_id)
        else:
            clear_price_family(row, "price_card", "price_card_out_of_range_removed", changes, track_id)


def fill_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    track_lookup: dict[tuple[str, str, str, str, str, str], str],
    full_tag_lookup: dict[str, str],
    recovery_lookup: dict[str, dict[str, Any]],
    catalog_barcode_index: CatalogBarcodeIndex | None,
    args: argparse.Namespace,
    special_classifier: SpecialSymbolClassifier | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filled_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for source_row in rows:
        row: dict[str, Any] = {field: source_row.get(field, "") for field in fieldnames}
        track_id = str(source_row.get("track_id", "")).strip() or track_lookup.get(row_key(source_row), "")

        if not args.no_template_net_defaults:
            for field in DEFAULT_NET_FIELDS:
                if field in fieldnames:
                    set_if_missing(row, field, NO_VALUE, "template_absent_value", changes, track_id)

        for field in REQUIRED_NET_FIELDS:
            if field in fieldnames:
                set_if_missing(row, field, NO_VALUE, "required_absent_value", changes, track_id)

        if args.fill_color and "color" in fieldnames:
            set_if_missing(row, "color", args.fill_color, "tag_color_default", changes, track_id)

        if args.extract_special_symbols and special_classifier and "special_symbols" in fieldnames:
            full_tag = full_tag_lookup.get(track_id, "")
            if full_tag:
                try:
                    symbol = special_classifier.predict(Path(full_tag))
                    set_value(row, "special_symbols", symbol, "special_symbol_template_classifier", changes, track_id, overwrite=True)
                except (FileNotFoundError, RuntimeError, ValueError):
                    pass

        recovery = recovery_lookup.get(track_id, {})
        if args.restore_price_default_cents and recovery.get("accepted"):
            observed_default = parse_price(row.get("price_default", ""))
            catalog_regular = parse_price(recovery.get("price_regular_rub", ""))
            if observed_default is not None and catalog_regular is not None and int(observed_default) == int(catalog_regular):
                set_value(
                    row,
                    "price_default",
                    format_price(catalog_regular),
                    "catalog_regular_price_integer_guard",
                    changes,
                    track_id,
                    overwrite=True,
                )

        if args.price_consistency_postprocess:
            repair_price_cents_from_catalog(
                row,
                "price_default",
                recovery.get("price_regular_rub", ""),
                "catalog_regular_price_cent_repair_same_integer",
                changes,
                track_id,
                min_catalog_score=0.45,
                catalog_score=recovery.get("score", ""),
            )
            repair_price_cents_from_catalog(
                row,
                "price_card",
                recovery.get("price_promo_rub", ""),
                "catalog_promo_price_cent_repair_same_integer",
                changes,
                track_id,
                min_catalog_score=0.45,
                catalog_score=recovery.get("score", ""),
            )
            reconcile_same_integer_prices(
                row,
                "price_default",
                "price1_qr",
                "price_default_price1_qr_cent_consistency",
                changes,
                track_id,
            )
            reconcile_same_integer_prices(
                row,
                "price_card",
                "price4_qr",
                "price_card_price4_qr_cent_consistency",
                changes,
                track_id,
            )

        if args.price_sanity_postprocess:
            repair_price_sanity(row, changes, track_id)

        price_default = parse_price(row.get("price_default", ""))
        price_card = parse_price(row.get("price_card", ""))
        if price_default is not None:
            set_value(
                row,
                "price1_qr",
                format_price(price_default),
                "qr_from_price_default",
                changes,
                track_id,
                overwrite=args.overwrite_derived_qr,
            )
            if args.infer_price2_qr:
                set_value(
                    row,
                    "price2_qr",
                    format_price(infer_second_qr_price(price_default, row.get("price_default", ""))),
                    "qr_price2_from_95pct_default",
                    changes,
                    track_id,
                    overwrite=args.overwrite_derived_qr,
                )
        if price_card is not None:
            set_value(
                row,
                "price4_qr",
                format_price(price_card),
                "qr_from_price_card",
                changes,
                track_id,
                overwrite=args.overwrite_derived_qr,
            )

        if args.barcode_from_catalog:
            catalog_barcode = find_catalog_barcode(row.get("product_name", ""), catalog_barcode_index)
            if not catalog_barcode and recovery.get("accepted"):
                catalog_barcode = str(recovery.get("catalog_id", "")).strip()
            if catalog_barcode and catalog_barcode.isdigit():
                set_value(
                    row,
                    "barcode",
                    catalog_barcode,
                    "accepted_catalog_code_as_barcode",
                    changes,
                    track_id,
                    overwrite=args.overwrite_barcode_from_catalog,
                )
                set_value(
                    row,
                    "qr_code_barcode",
                    catalog_barcode,
                    "accepted_catalog_code_as_qr_barcode",
                    changes,
                    track_id,
                    overwrite=args.overwrite_barcode_from_catalog,
                )

        barcode = str(row.get("barcode", "")).strip()
        qr_barcode = str(row.get("qr_code_barcode", "")).strip()
        if barcode and not qr_barcode:
            set_if_missing(row, "qr_code_barcode", barcode, "qr_barcode_from_barcode", changes, track_id)
        elif qr_barcode and not barcode:
            set_if_missing(row, "barcode", qr_barcode, "barcode_from_qr_barcode", changes, track_id)

        if not args.no_id_sku_from_catalog and "id_sku" in fieldnames:
            catalog_id = str(recovery.get("catalog_id", "")).strip()
            if recovery.get("accepted") and catalog_id and catalog_id.isdigit():
                prefix = infer_id_sku_prefix(recovery.get("candidate", "")) if args.id_sku_prefix_map else "370204"
                set_value(
                    row,
                    "id_sku",
                    f"{prefix}{catalog_id}",
                    "accepted_catalog_id_prefix_map" if args.id_sku_prefix_map else "accepted_catalog_id",
                    changes,
                    track_id,
                    overwrite=args.overwrite_id_sku_from_catalog,
                )

        filled_rows.append(row)
    return filled_rows, changes


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_dir = as_abs(root, args.run_dir)
    ocr_dir = run_dir / "ocr_final_quality_core_fast_fixed"
    input_csv = as_abs(root, args.input_csv) if args.input_csv else ocr_dir / "ocr_aggregated_submission_product_lines.csv"
    output_csv = as_abs(root, args.output_csv) if args.output_csv else input_csv
    sample_csv = as_abs(root, args.sample_csv)
    zones_csv = run_dir / "ocr_zones_core_fixed" / "ocr_zones_manifest.csv"
    recovery_csv = ocr_dir / "product_name_catalog_recovery.csv"

    fieldnames = read_fieldnames(sample_csv) or read_fieldnames(input_csv)
    rows = read_rows(input_csv)
    manifest_rows = read_rows(zones_csv) if zones_csv.exists() else []
    track_lookup = build_track_lookup(manifest_rows)
    full_tag_lookup = build_full_tag_lookup(manifest_rows)
    recovery_lookup = build_recovery_lookup(recovery_csv)
    catalog_barcode_path = as_abs(root, args.catalog_barcode_csv) if args.catalog_barcode_csv else None
    if catalog_barcode_path is None and args.barcode_from_catalog:
        parent_catalog = root.parent / "db_hack.csv"
        local_catalog = root / "data" / "db_hack.csv"
        if parent_catalog.exists():
            catalog_barcode_path = parent_catalog
        elif local_catalog.exists():
            catalog_barcode_path = local_catalog
    catalog_barcode_index = build_catalog_barcode_index(catalog_barcode_path) if args.barcode_from_catalog else None
    special_template_dir = as_abs(root, args.special_symbol_template_dir) if args.special_symbol_template_dir else None
    special_classifier = SpecialSymbolClassifier(root, special_template_dir) if args.extract_special_symbols else None

    filled_rows, changes = fill_rows(
        rows,
        fieldnames,
        track_lookup,
        full_tag_lookup,
        recovery_lookup,
        catalog_barcode_index,
        args,
        special_classifier,
    )
    write_rows(output_csv, filled_rows, fieldnames)
    changes_csv = output_csv.with_name(output_csv.stem + "_aux_fill_changes.csv")
    write_rows(changes_csv, changes, ["track_id", "field", "old", "new", "source"])

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": len(filled_rows),
        "changed_cells": len(changes),
        "infer_price2_qr": bool(args.infer_price2_qr),
        "restore_price_default_cents": bool(args.restore_price_default_cents),
        "price_consistency_postprocess": bool(args.price_consistency_postprocess),
        "price_sanity_postprocess": bool(args.price_sanity_postprocess),
        "extract_special_symbols": bool(args.extract_special_symbols),
        "barcode_from_catalog": bool(args.barcode_from_catalog),
        "catalog_barcode_csv": str(catalog_barcode_path or ""),
        "catalog_barcode_records": len(catalog_barcode_index.records) if catalog_barcode_index else 0,
        "overwrite_barcode_from_catalog": bool(args.overwrite_barcode_from_catalog),
        "id_sku_prefix_map": bool(args.id_sku_prefix_map),
        "overwrite_id_sku_from_catalog": bool(args.overwrite_id_sku_from_catalog),
        "id_sku_from_catalog": not bool(args.no_id_sku_from_catalog),
        "template_net_defaults": not bool(args.no_template_net_defaults),
        "changes_csv": str(changes_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
