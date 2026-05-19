from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_CACHE_VERSION = 3
PRODUCT_FIELD = "product_name"
OUTPUT_COLUMNS = [
    "video_id",
    "filename",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "product_name",
    "price_default",
    "price_card",
    "discount_amount",
    "frame_timestamp",
    "source_crop",
    "full_tag",
    "track_id",
]

STOPWORDS = {
    "a",
    "c",
    "co",
    "и",
    "в",
    "во",
    "с",
    "со",
    "из",
    "для",
    "на",
    "по",
    "к",
    "от",
    "без",
    "или",
    "the",
    "new",
    "арт",
    "россия",
    "италия",
    "германия",
    "испания",
    "франция",
    "китай",
}

CATEGORY_HINTS = {
    "мед",
    "конфитюр",
    "десерт",
    "вино",
    "набор",
    "шашлык",
    "вода",
    "кетчуп",
    "салфетки",
    "сосиски",
    "сыр",
    "молоко",
    "йогурт",
    "масло",
    "чай",
    "кофе",
}

CATEGORY_PRIORITY = (
    "\u043c\u0435\u0434",
    "\u043a\u043e\u043d\u0444\u0438\u0442\u044e\u0440",
    "\u0432\u0430\u0440\u0435\u043d\u044c\u0435",
    "\u0434\u0435\u0441\u0435\u0440\u0442",
    "\u0432\u0438\u043d\u043e",
    "\u043d\u0430\u0431\u043e\u0440",
    "\u0448\u0430\u0448\u043b\u044b\u043a",
    "\u0432\u043e\u0434\u0430",
    "\u043a\u0435\u0442\u0447\u0443\u043f",
    "\u0441\u0430\u043b\u0444\u0435\u0442\u043a\u0438",
    "\u0441\u043e\u0441\u0438\u0441\u043a\u0438",
    "\u0441\u044b\u0440",
    "\u043c\u043e\u043b\u043e\u043a\u043e",
    "\u0439\u043e\u0433\u0443\u0440\u0442",
    "\u043c\u0430\u0441\u043b\u043e",
    "\u0447\u0430\u0439",
    "\u043a\u043e\u0444\u0435",
)

KNOWN_COUNTRIES = (
    "Россия",
    "Италия",
    "Германия",
    "Испания",
    "Франция",
    "Китай",
    "Беларусь",
    "Турция",
    "Армения",
    "Грузия",
    "Чили",
)


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    norm: str
    tokens: frozenset[str]
    token_list: tuple[str, ...]
    brand_tokens: frozenset[str]
    units: frozenset[str]
    category: str
    trigrams: frozenset[str]
    catalog_id: str = ""
    weight_volume: str = ""
    price_regular_rub: str = ""
    price_promo_rub: str = ""
    url: str = ""


@dataclass
class CatalogIndex:
    entries: list[CatalogEntry]
    token_index: dict[str, tuple[int, ...]]
    brand_index: dict[str, tuple[int, ...]]
    unit_index: dict[str, tuple[int, ...]]
    trigram_index: dict[str, tuple[int, ...]]
    category_index: dict[str, tuple[int, ...]]
    source_path: str = ""
    source_size: int = 0
    source_mtime_ns: int = 0
    cache_version: int = CATALOG_CACHE_VERSION
    match_cache: dict[tuple[Any, ...], "MatchResult"] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class MatchResult:
    accepted: bool
    candidate: str
    score: float
    margin: float
    second_candidate: str
    second_score: float
    reason: str
    top_candidates: list[dict[str, Any]]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file).fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
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


def build_track_lookup(manifest_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], str]:
    lookup: dict[tuple[str, str, str, str, str, str], str] = {}
    for row in manifest_rows:
        if row.get("zone") == PRODUCT_FIELD and row.get("rank") == "1":
            lookup[row_key(row)] = row.get("track_id", "")
    return lookup


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def surface_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\r", " ").replace("\n", " ")).strip()


def tokenize(value: Any) -> list[str]:
    text = normalize_text(value)
    return re.findall(r"[0-9a-zа-я]{2,}", text, flags=re.IGNORECASE)


def content_tokens(value: Any) -> set[str]:
    return {token for token in tokenize(value) if token not in STOPWORDS and len(token) >= 3}


def trigrams(value: str) -> frozenset[str]:
    compact = normalize_text(value).replace(" ", "")
    if len(compact) < 3:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + 3] for index in range(len(compact) - 2))


def normalize_unit_value(number_text: str, unit: str) -> str:
    number = number_text.replace(" ", "").replace(",", ".")
    try:
        value = float(number)
    except ValueError:
        return ""
    unit = unit.lower()
    if unit in {"кг", "kg"}:
        return f"{int(round(value * 1000))}g"
    if unit in {"г", "гр", "g"}:
        return f"{int(round(value))}g"
    if unit in {"л", "l"}:
        return f"{int(round(value * 1000))}ml"
    if unit in {"мл", "ml"}:
        return f"{int(round(value))}ml"
    return ""


def extract_units(value: Any) -> frozenset[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace(",", ".")
    units: set[str] = set()
    for match in re.finditer(r"(\d+(?:\.\d+)?|\d+\s+\d{2,3})\s*(кг|kg|гр|г|g|мл|ml|л|l)\b", text):
        unit = normalize_unit_value(match.group(1), match.group(2))
        if unit:
            units.add(unit)
    return frozenset(units)


def catalog_weight_display(value: Any) -> str:
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if re.search(r"\bарт\b|art\.?", text, flags=re.IGNORECASE):
        return ""
    match = re.search(
        r"(\d+(?:[,.]\d+)?|\d+\s*[xх]\s*\d+(?:[,.]\d+)?)\s*(кг|kg|гр|г|g|мл|ml|л|l)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    number = re.sub(r"\s+", "", match.group(1)).replace(".", ",")
    unit = match.group(2).lower()
    unit_map = {
        "kg": "кг",
        "кг": "кг",
        "гр": "г",
        "g": "г",
        "г": "г",
        "ml": "мл",
        "мл": "мл",
        "l": "л",
        "л": "л",
    }
    return f"{number}{unit_map.get(unit, unit)}"


def append_catalog_weight(name: str, weight_volume: Any) -> str:
    weight = catalog_weight_display(weight_volume)
    if not weight:
        return name
    if extract_units(name) & extract_units(weight):
        return name
    return f"{name} {weight}"


def visible_country(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    best_country = ""
    best_score = 0.0
    for match in re.finditer(r"\(([^)]{3,32})\)", text):
        inner = normalize_text(match.group(1))
        if not inner:
            continue
        for country in KNOWN_COUNTRIES:
            current = ratio(inner, normalize_text(country))
            if current > best_score:
                best_score = current
                best_country = country
    if best_score >= 0.72:
        return best_country

    tokens = tokenize(text)
    for token in tokens:
        if token.startswith(("росси", "россм", "росск", "росг", "россж")) and not token.startswith("россо"):
            return "Россия"
        if token.startswith("герман"):
            return "Германия"
        if token.startswith("итал"):
            return "Италия"
        if token.startswith("испан"):
            return "Испания"

    for token in tokens:
        for country in KNOWN_COUNTRIES:
            if ratio(token, normalize_text(country)) >= 0.78:
                return country
    return ""


def unit_display_from_normalized(unit: str) -> str:
    if unit.endswith("ml"):
        return unit[:-2] + "мл"
    if unit.endswith("g"):
        return unit[:-1] + "г"
    return unit


def visible_unit(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace(",", ".")
    matches = list(re.finditer(r"(\d+(?:\.\d+)?|\d+\s+\d{2,3})\s*(кг|kg|гр|г|g|мл|ml|л|l)\b", text))
    for match in reversed(matches):
        normalized = normalize_unit_value(match.group(1), match.group(2))
        if not normalized:
            continue
        if normalized.endswith("g"):
            try:
                grams = int(normalized[:-1])
            except ValueError:
                continue
            if 1 <= grams <= 1500:
                return unit_display_from_normalized(normalized)
            continue
        if normalized.endswith("ml"):
            try:
                ml = int(normalized[:-2])
            except ValueError:
                continue
            if 1 <= ml <= 3000:
                return unit_display_from_normalized(normalized)
    bare = re.search(r"(?:\)|\s)(\d{2,4})\s*$", text)
    if bare:
        value_int = int(bare.group(1))
        if 10 <= value_int <= 1500:
            return f"{value_int}г"
    return ""


def visible_glass_packaging(value: Any) -> bool:
    text = normalize_text(value)
    return bool(re.search(r"(?:^|\s)ст\s*(?:б|b|6)(?:\s|$)", text, flags=re.IGNORECASE))


def order_tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    return re.findall(r"[0-9a-zа-я]+", text, flags=re.IGNORECASE)


def word_order_position(word: str, tokens: list[str]) -> float | None:
    word_tokens = order_tokens(word)
    if not word_tokens or not tokens:
        return None
    best_position: float | None = None
    best_score = 0.0
    for word_token in word_tokens:
        if len(word_token) <= 1:
            continue
        for index, token in enumerate(tokens):
            current = ratio(word_token, token)
            if current > best_score:
                best_score = current
                best_position = float(index)
    return best_position if best_position is not None and best_score >= 0.64 else None


def fill_missing_positions(positions: list[float | None]) -> list[float]:
    filled = [0.0] * len(positions)
    known = [index for index, value in enumerate(positions) if value is not None]
    if not known:
        return [float(index) for index in range(len(positions))]

    for index, value in enumerate(positions):
        if value is not None:
            filled[index] = value
            continue
        prev_known = max((known_index for known_index in known if known_index < index), default=None)
        next_known = min((known_index for known_index in known if known_index > index), default=None)
        if prev_known is not None and next_known is not None:
            left = float(positions[prev_known])
            right = float(positions[next_known])
            ratio_between = (index - prev_known) / max(1, next_known - prev_known)
            filled[index] = left + (right - left) * ratio_between
        elif prev_known is not None:
            filled[index] = float(positions[prev_known]) + 0.01 * (index - prev_known)
        elif next_known is not None:
            filled[index] = float(positions[next_known]) - 0.01 * (next_known - index)
        else:
            filled[index] = float(index)
    return filled


def reorder_base_by_ocr(base: str, ocr_text: str) -> str:
    words = base.split()
    if len(words) < 4:
        return base
    tokens = order_tokens(ocr_text)
    positions = [word_order_position(word, tokens) for word in words]
    matched = sum(1 for value in positions if value is not None)
    if matched < 3 or matched / len(words) < 0.45:
        return base
    filled = fill_missing_positions(positions)
    reordered = [word for _, original_index, word in sorted((filled[index], index, word) for index, word in enumerate(words))]
    return " ".join(reordered)


def order_source_quality(base: str, ocr_text: str) -> float:
    words = base.split()
    if len(words) < 4:
        return 0.0
    tokens = order_tokens(ocr_text)
    if not tokens:
        return 0.0
    positions = [word_order_position(word, tokens) for word in words]
    matched = sum(1 for value in positions if value is not None)
    matched_ratio = matched / len(words)
    if matched < 3 or matched_ratio < 0.45:
        return 0.0
    distinct_positions = len({round(float(value), 2) for value in positions if value is not None})
    noise_penalty = max(0, len(tokens) - matched) * 0.015
    return matched + matched_ratio + distinct_positions * 0.05 - noise_penalty


def select_hybrid_order_source(
    catalog_name: str,
    before: str,
    match_source_text: str,
    tail: dict[str, str],
    candidate_pool_size: int,
) -> tuple[str, float]:
    base, _, _ = split_tail(catalog_name)
    sources: list[str] = []

    def add_source(value: Any) -> None:
        text = surface_text(value)
        if text and text not in sources:
            sources.append(text)

    add_source(match_source_text)
    add_source(before)
    if candidate_pool_size > 0:
        try:
            pool = json.loads(tail.get("candidate_texts_json", "[]"))
        except json.JSONDecodeError:
            pool = []
        for text in pool[:candidate_pool_size]:
            add_source(text)

    if not sources:
        return match_source_text, 0.0

    scored = [(order_source_quality(base, source), source) for source in sources]
    current_score = order_source_quality(base, match_source_text)
    best_score, best_source = max(scored, key=lambda item: (item[0], len(item[1])))
    if best_score >= current_score + 0.25:
        return best_source, best_score
    return match_source_text, current_score


def trailing_unit_display(value: str) -> str:
    match = re.search(
        r"(\d+(?:[,.]\d+)?|\d+\s*[xх]\s*\d+(?:[,.]\d+)?)\s*(кг|kg|гр|г|g|мл|ml|л|l)\s*$",
        value,
        flags=re.IGNORECASE,
    )
    return catalog_weight_display(match.group(0)) if match else ""


def split_tail(value: str) -> tuple[str, str, str]:
    base = surface_text(value)
    unit = trailing_unit_display(base)
    if unit:
        base = re.sub(
            r"\s*(\d+(?:[,.]\d+)?|\d+\s*[xх]\s*\d+(?:[,.]\d+)?)\s*(кг|kg|гр|г|g|мл|ml|л|l)\s*$",
            "",
            base,
            flags=re.IGNORECASE,
        ).strip()
    country = ""
    country_match = re.search(r"\(([^)]{3,32})\)\s*$", base)
    if country_match:
        candidate = visible_country(country_match.group(0))
        if candidate:
            country = candidate
            base = base[: country_match.start()].strip()
    return base, country, unit


def restore_visible_tail(
    catalog_name: str,
    ocr_text: str,
    country_override: str = "",
    unit_override: str = "",
    glass_packaging: bool = False,
    preserve_order: bool = False,
) -> str:
    base, existing_country, existing_unit = split_tail(catalog_name)
    if preserve_order:
        base = reorder_base_by_ocr(base, ocr_text)
    country = existing_country or country_override or visible_country(ocr_text)
    unit = existing_unit or unit_override or visible_unit(ocr_text)
    if glass_packaging and not visible_glass_packaging(base):
        base = f"{base} ст/б"
    result = base
    if country:
        result = f"{result} ({country})"
    if unit:
        result = f"{result} {unit}"
    return result


def extract_brand_tokens(value: Any) -> frozenset[str]:
    brands: set[str] = set()
    for raw in re.findall(r"[A-Za-zА-ЯЁ0-9]{3,}", unicodedata.normalize("NFKC", str(value or ""))):
        norm = normalize_text(raw)
        if not norm or norm in STOPWORDS:
            continue
        has_latin = bool(re.search(r"[A-Za-z]", raw))
        has_cyrillic_upper = bool(re.search(r"[А-ЯЁ]", raw))
        if has_latin or (has_cyrillic_upper and raw.upper() == raw):
            brands.add(norm)
    return frozenset(brands)


def category_for(tokens: set[str]) -> str:
    for token in CATEGORY_PRIORITY:
        if token in tokens:
            return token
    return ""


def format_catalog_price(value: Any) -> str:
    if is_missing(value):
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return surface_text(value)


def parse_price_value(value: Any) -> float | None:
    if is_missing(value):
        return None
    text = surface_text(value).replace(",", ".")
    match = re.search(r"\d+(?:\.\d{1,2})?", text)
    if not match:
        return None
    try:
        price = float(match.group(0))
    except ValueError:
        return None
    if 1 <= price <= 100000:
        return price
    return None


def candidate_prices(candidate: CatalogEntry) -> list[float]:
    prices: list[float] = []
    for value in (candidate.price_promo_rub, candidate.price_regular_rub):
        price = parse_price_value(value)
        if price is not None and price not in prices:
            prices.append(price)
    return prices


def price_score_adjustment(
    observed_price: float | None,
    candidate: CatalogEntry,
    exact_tolerance: float,
    near_tolerance: float,
    bonus: float,
    penalty: float,
) -> tuple[float, dict[str, Any]]:
    if observed_price is None:
        return 0.0, {}
    prices = candidate_prices(candidate)
    if not prices:
        return 0.0, {"observed_price_card": round(observed_price, 2)}

    best_price = min(prices, key=lambda price: abs(price - observed_price))
    diff = abs(best_price - observed_price)
    adjustment = 0.0
    if diff <= exact_tolerance:
        adjustment = bonus
    elif diff <= near_tolerance:
        span = max(near_tolerance - exact_tolerance, 1e-6)
        adjustment = bonus * 0.5 * (1.0 - (diff - exact_tolerance) / span)
    else:
        adjustment = -penalty
    return adjustment, {
        "observed_price_card": round(observed_price, 2),
        "catalog_price_match": round(best_price, 2),
        "catalog_price_diff": round(diff, 2),
        "price_adjustment": round(adjustment, 6),
    }


def catalog_item_name(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("name", "fullname", "full_name", "product_name"):
            name = surface_text(item.get(key, ""))
            if name:
                return name
        return ""
    return surface_text(item)


def catalog_item_display_name(item: Any) -> str:
    if isinstance(item, dict):
        return append_catalog_weight(catalog_item_name(item), item.get("weight_volume"))
    return catalog_item_name(item)


def make_entry(item: Any) -> CatalogEntry:
    name = catalog_item_display_name(item)
    token_set = content_tokens(name)
    token_list = tuple(tokenize(name))
    return CatalogEntry(
        name=surface_text(name),
        norm=normalize_text(name),
        tokens=frozenset(token_set),
        token_list=token_list,
        brand_tokens=extract_brand_tokens(name),
        units=extract_units(name),
        category=category_for(token_set),
        trigrams=trigrams(name),
        catalog_id=surface_text(item.get("id") or item.get("code") or item.get("barcode") or "") if isinstance(item, dict) else "",
        weight_volume=catalog_weight_display(item.get("weight_volume", "")) if isinstance(item, dict) else "",
        price_regular_rub=format_catalog_price(item.get("price_regular_rub", "")) if isinstance(item, dict) else "",
        price_promo_rub=format_catalog_price(item.get("price_promo_rub", "")) if isinstance(item, dict) else "",
        url=surface_text(item.get("url", "")) if isinstance(item, dict) else "",
    )


@lru_cache(maxsize=20000)
def make_text_entry(text: str) -> CatalogEntry:
    return make_entry(surface_text(text))


@lru_cache(maxsize=300000)
def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def fuzzy_token_score(query_tokens: set[str], candidate_tokens: frozenset[str]) -> tuple[float, int]:
    if not query_tokens or not candidate_tokens:
        return 0.0, 0
    scores: list[float] = []
    matched = 0
    for query_token in query_tokens:
        best = max(ratio(query_token, candidate_token) for candidate_token in candidate_tokens)
        scores.append(best)
        matched += int(best >= 0.78)
    return sum(scores) / len(scores), matched


def score_entry(query: CatalogEntry, candidate: CatalogEntry) -> tuple[float, dict[str, Any]]:
    if not query.norm or not candidate.norm:
        return 0.0, {}

    exact_overlap = query.tokens & candidate.tokens
    token_recall = len(exact_overlap) / max(1, len(query.tokens))
    token_precision = len(exact_overlap) / max(1, len(candidate.tokens))
    token_min = len(exact_overlap) / max(1, min(len(query.tokens), len(candidate.tokens)))
    fuzzy_mean, fuzzy_count = fuzzy_token_score(set(query.tokens), candidate.tokens)
    char_score = ratio(query.norm, candidate.norm)
    tri_overlap = len(query.trigrams & candidate.trigrams)
    tri_union = len(query.trigrams | candidate.trigrams)
    trigram_score = tri_overlap / max(1, tri_union)

    brand_overlap = query.brand_tokens & candidate.brand_tokens
    brand_score = 0.0
    if brand_overlap:
        brand_score = 0.10
    elif query.brand_tokens and candidate.brand_tokens:
        brand_score = -0.04

    unit_score = 0.0
    if query.units and candidate.units:
        if query.units & candidate.units:
            unit_score = 0.10
        else:
            unit_score = -0.13

    category_score = 0.0
    if query.category and candidate.category:
        category_score = 0.07 if query.category == candidate.category else -0.08

    score = (
        0.24 * token_recall
        + 0.13 * token_precision
        + 0.10 * token_min
        + 0.18 * fuzzy_mean
        + 0.18 * char_score
        + 0.07 * trigram_score
        + brand_score
        + unit_score
        + category_score
    )
    score = max(0.0, min(1.0, score))
    details = {
        "score": round(score, 6),
        "token_recall": round(token_recall, 4),
        "token_precision": round(token_precision, 4),
        "token_min": round(token_min, 4),
        "fuzzy_token_mean": round(fuzzy_mean, 4),
        "fuzzy_token_count": fuzzy_count,
        "char_score": round(char_score, 4),
        "trigram_score": round(trigram_score, 4),
        "brand_overlap": " ".join(sorted(brand_overlap)),
        "unit_overlap": " ".join(sorted(query.units & candidate.units)),
        "query_units": " ".join(sorted(query.units)),
        "candidate_units": " ".join(sorted(candidate.units)),
        "query_category": query.category,
        "candidate_category": candidate.category,
        "exact_overlap": " ".join(sorted(exact_overlap)),
    }
    return score, details


def is_tail_or_packaging_token(token: str) -> bool:
    if re.search(r"\d", token):
        return True
    if token in {"ст", "б", "b"}:
        return True
    if token.startswith(("росси", "россм", "росск", "росг", "россж", "герман", "итал", "испан")):
        return True
    return False


def unexplained_query_tokens(query: CatalogEntry, candidate: CatalogEntry, min_ratio: float = 0.65) -> list[str]:
    unexplained: list[str] = []
    for token in sorted(query.tokens):
        if len(token) < 5 or is_tail_or_packaging_token(token):
            continue
        best = max((ratio(token, candidate_token) for candidate_token in candidate.tokens), default=0.0)
        if best < min_ratio:
            unexplained.append(token)
    return unexplained


def cheap_score(query: CatalogEntry, candidate: CatalogEntry) -> float:
    exact_overlap = len(query.tokens & candidate.tokens)
    brand_overlap = len(query.brand_tokens & candidate.brand_tokens)
    unit_overlap = len(query.units & candidate.units)
    tri_overlap = len(query.trigrams & candidate.trigrams)
    category_bonus = 1 if query.category and query.category == candidate.category else 0
    return 4.0 * exact_overlap + 5.0 * brand_overlap + 3.0 * unit_overlap + 0.05 * tri_overlap + category_bonus


def build_index_mapping(entries: list[CatalogEntry], attr: str) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        values = getattr(entry, attr)
        if isinstance(values, str):
            if values:
                grouped[values].append(index)
            continue
        for value in values:
            grouped[value].append(index)
    return {key: tuple(indices) for key, indices in grouped.items()}


def build_catalog_index(entries: list[CatalogEntry], source_path: Path) -> CatalogIndex:
    stat = source_path.stat()
    return CatalogIndex(
        entries=entries,
        token_index=build_index_mapping(entries, "tokens"),
        brand_index=build_index_mapping(entries, "brand_tokens"),
        unit_index=build_index_mapping(entries, "units"),
        trigram_index=build_index_mapping(entries, "trigrams"),
        category_index=build_index_mapping(entries, "category"),
        source_path=str(source_path.resolve()),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )


def is_catalog_cache_valid(index: CatalogIndex, source_path: Path) -> bool:
    stat = source_path.stat()
    return (
        getattr(index, "cache_version", None) == CATALOG_CACHE_VERSION
        and getattr(index, "source_path", "") == str(source_path.resolve())
        and getattr(index, "source_size", 0) == stat.st_size
        and getattr(index, "source_mtime_ns", 0) == stat.st_mtime_ns
    )


def save_catalog_cache(index: CatalogIndex, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index.match_cache = {}
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temp_path.open("wb") as file:
        pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(cache_path)


def load_catalog_cache(cache_path: Path, source_path: Path) -> CatalogIndex | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as file:
            index = pickle.load(file)
    except (OSError, pickle.PickleError, AttributeError, EOFError, ValueError):
        return None
    if not isinstance(index, CatalogIndex) or not is_catalog_cache_valid(index, source_path):
        return None
    index.match_cache = {}
    return index


def preselect_catalog(query: CatalogEntry, catalog: CatalogIndex, preselect: int) -> list[CatalogEntry]:
    if preselect <= 0:
        return []

    candidate_indices: set[int] = set()

    def add(index_mapping: dict[str, tuple[int, ...]], values: Any) -> None:
        if isinstance(values, str):
            value_iterable = [values] if values else []
        else:
            value_iterable = values
        for value in value_iterable:
            candidate_indices.update(index_mapping.get(value, ()))

    add(catalog.token_index, query.tokens)
    add(catalog.brand_index, query.brand_tokens)
    add(catalog.unit_index, query.units)
    add(catalog.trigram_index, query.trigrams)
    add(catalog.category_index, query.category)

    scored = [
        (cheap_score(query, catalog.entries[index]), index, catalog.entries[index])
        for index in candidate_indices
    ]
    scored = [item for item in scored if item[0] > 0.0]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected_indices = {index for _, index, _ in scored[:preselect]}
    selected = [entry for _, _, entry in scored[:preselect]]

    if len(selected) < preselect:
        for index, entry in enumerate(catalog.entries):
            if index in selected_indices or index in candidate_indices:
                continue
            selected.append(entry)
            if len(selected) >= preselect:
                break
    return selected


def match_catalog(
    ocr_text: str,
    catalog: CatalogIndex,
    preselect: int,
    accept_score: float,
    accept_margin: float,
    strong_score: float,
    observed_price: float | None = None,
    price_bonus: float = 0.0,
    price_tolerance_rub: float = 1.0,
    price_near_rub: float = 8.0,
    price_penalty: float = 0.0,
    price_can_accept: bool = False,
) -> MatchResult:
    cache_key = (
        surface_text(ocr_text),
        preselect,
        round(accept_score, 6),
        round(accept_margin, 6),
        round(strong_score, 6),
        None if observed_price is None else round(observed_price, 2),
        round(price_bonus, 6),
        round(price_tolerance_rub, 6),
        round(price_near_rub, 6),
        round(price_penalty, 6),
        bool(price_can_accept),
    )
    cached = catalog.match_cache.get(cache_key)
    if cached is not None:
        return cached

    query = make_text_entry(surface_text(ocr_text))
    if not query.norm or len(query.norm) < 4:
        result = MatchResult(False, "", 0.0, 0.0, "", 0.0, "empty_query", [])
        catalog.match_cache[cache_key] = result
        return result

    preselected = preselect_catalog(query, catalog, preselect)
    scored: list[tuple[float, CatalogEntry, dict[str, Any]]] = []
    for candidate in preselected:
        score, details = score_entry(query, candidate)
        if observed_price is not None and price_bonus:
            text_score = score
            adjustment, price_details = price_score_adjustment(
                observed_price,
                candidate,
                price_tolerance_rub,
                price_near_rub,
                price_bonus,
                price_penalty,
            )
            score = max(0.0, min(1.0, score + adjustment))
            details["text_score"] = round(text_score, 6)
            details.update(price_details)
        scored.append((score, candidate, details))
    scored.sort(key=lambda item: item[0], reverse=True)

    top = scored[:5]
    if not top:
        result = MatchResult(False, "", 0.0, 0.0, "", 0.0, "no_candidates", [])
        catalog.match_cache[cache_key] = result
        return result

    best_score, best, best_details = top[0]
    second_score = top[1][0] if len(top) > 1 else 0.0
    second_name = top[1][1].name if len(top) > 1 else ""
    margin = best_score - second_score
    accept_score_for_threshold = best_score
    if not price_can_accept and "text_score" in best_details:
        accept_score_for_threshold = float(best_details["text_score"])
    accepted = accept_score_for_threshold >= strong_score or (
        accept_score_for_threshold >= accept_score and margin >= accept_margin
    )
    reason = "accepted" if accepted else "below_threshold"
    if accept_score_for_threshold >= accept_score and margin < accept_margin and accept_score_for_threshold < strong_score:
        reason = "low_margin"
    if not accepted and best_score >= accept_score and accept_score_for_threshold < accept_score:
        reason = "price_only_below_text_threshold"
    unexplained = unexplained_query_tokens(query, best)
    if accepted and unexplained:
        accepted = False
        reason = "unexplained_ocr_token:" + "|".join(unexplained[:3])

    top_payload = []
    for score, candidate, details in top:
        payload = {"name": candidate.name, "score": round(score, 6)}
        if candidate.catalog_id:
            payload["id"] = candidate.catalog_id
        if candidate.weight_volume:
            payload["weight_volume"] = candidate.weight_volume
        if candidate.price_regular_rub:
            payload["price_regular_rub"] = candidate.price_regular_rub
        if candidate.price_promo_rub:
            payload["price_promo_rub"] = candidate.price_promo_rub
        payload.update(details)
        top_payload.append(payload)
    result = MatchResult(
        accepted=accepted,
        candidate=best.name,
        score=best_score,
        margin=margin,
        second_candidate=second_name,
        second_score=second_score,
        reason=reason,
        top_candidates=top_payload,
    )
    catalog.match_cache[cache_key] = result
    return result


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


def read_catalog_csv(path: Path) -> list[dict[str, str]]:
    text = read_text_fallback(path, ("utf-8-sig", "cp1251"))
    first_line = text.splitlines()[0] if text.splitlines() else ""
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=";,")
    except csv.Error:
        dialect = csv.excel()
        dialect.delimiter = ";" if ";" in first_line else ","

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    records: list[dict[str, str]] = []
    for row in reader:
        normalized = {str(key or "").strip().lstrip("\ufeff").lower(): value for key, value in row.items()}
        name = surface_text(
            normalized.get("fullname")
            or normalized.get("full_name")
            or normalized.get("name")
            or normalized.get("product_name")
            or ""
        )
        code = surface_text(
            normalized.get("code")
            or normalized.get("barcode")
            or normalized.get("id")
            or normalized.get("id_sku")
            or ""
        )
        if name:
            records.append({"name": name, "id": code, "code": code})
    return records


def load_catalog_data(path: Path) -> list[Any]:
    if path.suffix.lower() == ".csv":
        return read_catalog_csv(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Catalog JSON/CSV must be a list of records")
    return data


def load_catalog(path: Path, cache_path: Path | None = None, rebuild_cache: bool = False) -> CatalogIndex:
    if cache_path is not None and not rebuild_cache:
        cached = load_catalog_cache(cache_path, path)
        if cached is not None:
            return cached

    data = load_catalog_data(path)
    seen: set[tuple[str, str]] = set()
    entries: list[CatalogEntry] = []
    for item in data:
        name = catalog_item_name(item)
        if not name:
            continue
        weight = catalog_weight_display(item.get("weight_volume", "")) if isinstance(item, dict) else ""
        key = (name.lower(), weight)
        if key in seen:
            continue
        seen.add(key)
        entries.append(make_entry(item))
    index = build_catalog_index(sorted(entries, key=lambda entry: entry.name.lower()), path)
    if cache_path is not None:
        save_catalog_cache(index, cache_path)
    return index


def parse_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("score", "") or 0.0)
    except ValueError:
        return 0.0


def build_tail_lookup(candidate_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in candidate_rows:
        track_id = str(row.get("track_id", "")).strip()
        if track_id:
            grouped.setdefault(track_id, []).append(row)

    lookup: dict[str, dict[str, str]] = {}
    for track_id, rows in grouped.items():
        sorted_rows = sorted(rows, key=parse_score, reverse=True)
        country = ""
        unit = ""
        glass_packaging = any(visible_glass_packaging(row.get("value") or row.get("raw_text") or "") for row in rows)
        candidate_texts: list[str] = []
        for row in sorted_rows:
            text = row.get("value") or row.get("raw_text") or ""
            normalized_text = surface_text(text)
            if normalized_text and normalized_text not in candidate_texts:
                candidate_texts.append(normalized_text)
            if not country:
                country = visible_country(text)
            if not unit:
                unit = visible_unit(text)
            if country and unit and len(candidate_texts) >= 12:
                break
        lookup[track_id] = {
            "country": country,
            "unit": unit,
            "glass_packaging": "1" if glass_packaging else "",
            "candidate_texts_json": json.dumps(candidate_texts, ensure_ascii=False),
        }
    return lookup


def copy_run_scaffold(source_run_dir: Path, output_run_dir: Path) -> None:
    output_run_dir.mkdir(parents=True, exist_ok=True)
    source_zones = source_run_dir / "ocr_zones_core_fixed"
    target_zones = output_run_dir / "ocr_zones_core_fixed"
    if source_zones.exists():
        if target_zones.exists():
            shutil.rmtree(target_zones)
        shutil.copytree(source_zones, target_zones)

    source_ocr = source_run_dir / "ocr_final_quality_core_fast_fixed"
    target_ocr = output_run_dir / "ocr_final_quality_core_fast_fixed"
    if target_ocr.exists():
        shutil.rmtree(target_ocr)
    shutil.copytree(source_ocr, target_ocr)


def best_match_from_ocr_pool(
    before: str,
    tail: dict[str, str],
    catalog: CatalogIndex,
    args: argparse.Namespace,
    observed_price: float | None = None,
) -> tuple[MatchResult, str]:
    if args.catalog_decoder_mode == "vote":
        voted_result, voted_text = vote_match_from_ocr_pool(before, tail, catalog, args, observed_price)
        if voted_result.accepted:
            return voted_result, voted_text

    texts = [before]
    if args.candidate_pool_size > 0:
        try:
            pool = json.loads(tail.get("candidate_texts_json", "[]"))
        except json.JSONDecodeError:
            pool = []
        for text in pool[: args.candidate_pool_size]:
            normalized_text = surface_text(text)
            if normalized_text and normalized_text not in texts:
                texts.append(normalized_text)

    best_result: MatchResult | None = None
    best_text = before
    for text in texts:
        result = match_catalog(
            text,
            catalog,
            args.preselect,
            args.accept_score,
            args.accept_margin,
            args.strong_score,
            observed_price=observed_price,
            price_bonus=args.price_bonus if args.price_aware_rerank else 0.0,
            price_tolerance_rub=args.price_tolerance_rub,
            price_near_rub=args.price_near_rub,
            price_penalty=args.price_penalty,
            price_can_accept=args.price_can_accept,
        )
        if best_result is None:
            best_result = result
            best_text = text
            continue
        current_key = (int(result.accepted), result.score, result.margin)
        best_key = (int(best_result.accepted), best_result.score, best_result.margin)
        if current_key > best_key:
            best_result = result
            best_text = text
    if best_result is None:
        return match_catalog(
            before,
            catalog,
            args.preselect,
            args.accept_score,
            args.accept_margin,
            args.strong_score,
            observed_price=observed_price,
            price_bonus=args.price_bonus if args.price_aware_rerank else 0.0,
            price_tolerance_rub=args.price_tolerance_rub,
            price_near_rub=args.price_near_rub,
            price_penalty=args.price_penalty,
            price_can_accept=args.price_can_accept,
        ), before
    return best_result, best_text


def candidate_text_pool(before: str, tail: dict[str, str], candidate_pool_size: int) -> list[str]:
    texts = [before]
    if candidate_pool_size <= 0:
        return texts
    try:
        pool = json.loads(tail.get("candidate_texts_json", "[]"))
    except json.JSONDecodeError:
        pool = []
    for text in pool[:candidate_pool_size]:
        normalized_text = surface_text(text)
        if normalized_text and normalized_text not in texts:
            texts.append(normalized_text)
    return texts


def candidate_explains_source(source_text: str, candidate_name: str, min_ratio: float = 0.65) -> bool:
    query = make_text_entry(surface_text(source_text))
    candidate = make_text_entry(surface_text(candidate_name))
    return not unexplained_query_tokens(query, candidate, min_ratio=min_ratio)


def vote_match_from_ocr_pool(
    before: str,
    tail: dict[str, str],
    catalog: CatalogIndex,
    args: argparse.Namespace,
    observed_price: float | None = None,
) -> tuple[MatchResult, str]:
    texts = candidate_text_pool(before, tail, args.candidate_pool_size)
    evidence: dict[str, dict[str, Any]] = {}
    best_source_by_candidate: dict[str, tuple[float, str]] = {}

    for source_index, text in enumerate(texts):
        result = match_catalog(
            text,
            catalog,
            args.preselect,
            args.accept_score,
            args.accept_margin,
            args.strong_score,
            observed_price=observed_price,
            price_bonus=args.price_bonus if args.price_aware_rerank else 0.0,
            price_tolerance_rub=args.price_tolerance_rub,
            price_near_rub=args.price_near_rub,
            price_penalty=args.price_penalty,
            price_can_accept=args.price_can_accept,
        )
        for rank, payload in enumerate(result.top_candidates[: args.vote_top_k], start=1):
            score = float(payload.get("score", 0.0) or 0.0)
            if score < args.vote_min_score:
                continue
            name = surface_text(payload.get("name", ""))
            if not name or not candidate_explains_source(text, name, args.vote_explain_ratio):
                continue
            weight = max(0.0, score - args.vote_min_score) / math.sqrt(rank)
            current = evidence.setdefault(
                name,
                {
                    "vote": 0.0,
                    "support": 0,
                    "best_score": 0.0,
                    "best_text_score": 0.0,
                    "best_payload": payload,
                    "sources": [],
                },
            )
            current["vote"] += weight
            current["support"] += 1
            current["sources"].append(
                {
                    "source_index": source_index,
                    "rank": rank,
                    "score": round(score, 6),
                    "text": text,
                }
            )
            if score > current["best_score"]:
                current["best_score"] = score
                current["best_text_score"] = float(payload.get("text_score", score) or 0.0)
                current["best_payload"] = payload
            if name not in best_source_by_candidate or score > best_source_by_candidate[name][0]:
                best_source_by_candidate[name] = (score, text)

    ranked = sorted(
        evidence.items(),
        key=lambda item: (item[1]["vote"], item[1]["support"], item[1]["best_score"]),
        reverse=True,
    )
    if not ranked:
        return MatchResult(False, "", 0.0, 0.0, "", 0.0, "vote_no_candidates", []), before

    best_name, best_data = ranked[0]
    second_name = ranked[1][0] if len(ranked) > 1 else ""
    second_vote = float(ranked[1][1]["vote"]) if len(ranked) > 1 else 0.0
    best_vote = float(best_data["vote"])
    margin = best_vote - second_vote
    best_score = float(best_data["best_score"])
    best_text_score = float(best_data.get("best_text_score", best_score) or 0.0)
    support = int(best_data["support"])
    accepted = (
        support >= args.vote_min_support
        and best_vote >= args.vote_accept_score
        and margin >= args.vote_accept_margin
        and best_score >= args.accept_score
        and (args.price_can_accept or best_text_score >= args.accept_score)
    )
    reason = "vote_accepted" if accepted else "vote_below_threshold"
    if support < args.vote_min_support:
        reason = "vote_low_support"
    elif margin < args.vote_accept_margin:
        reason = "vote_low_margin"
    elif best_vote < args.vote_accept_score:
        reason = "vote_low_score"
    elif best_score < args.accept_score:
        reason = "vote_low_best_match"
    elif not args.price_can_accept and best_text_score < args.accept_score:
        reason = "vote_price_only_below_text_threshold"

    top_payload: list[dict[str, Any]] = []
    for name, data in ranked[:5]:
        payload = dict(data["best_payload"])
        payload["name"] = name
        payload["vote"] = round(float(data["vote"]), 6)
        payload["support"] = int(data["support"])
        payload["best_score"] = round(float(data["best_score"]), 6)
        payload["best_text_score"] = round(float(data.get("best_text_score", data["best_score"])), 6)
        payload["sources"] = data["sources"][:5]
        top_payload.append(payload)

    return (
        MatchResult(
            accepted=accepted,
            candidate=best_name,
            score=best_score,
            margin=margin,
            second_candidate=second_name,
            second_score=second_vote,
            reason=reason,
            top_candidates=top_payload,
        ),
        best_source_by_candidate.get(best_name, (0.0, before))[1],
    )


def update_outputs(source_run_dir: Path, output_run_dir: Path, catalog: CatalogIndex, args: argparse.Namespace) -> dict[str, Any]:
    copy_run_scaffold(source_run_dir, output_run_dir)
    ocr_dir = output_run_dir / "ocr_final_quality_core_fast_fixed"

    changes_path = ocr_dir / "product_name_line_changes.csv"
    final_path = ocr_dir / "ocr_aggregated_submission_product_lines.csv"
    debug_path = ocr_dir / "ocr_aggregated_debug_product_lines.csv"
    candidates_path = ocr_dir / "product_name_line_candidates.csv"
    zones_path = output_run_dir / "ocr_zones_core_fixed" / "ocr_zones_manifest.csv"
    final_fieldnames = read_fieldnames(final_path)

    changes = read_rows(changes_path)
    tail_lookup = build_tail_lookup(read_rows(candidates_path)) if candidates_path.exists() else {}
    track_lookup = build_track_lookup(read_rows(zones_path)) if zones_path.exists() else {}
    source_final_rows = read_rows(final_path)
    price_by_track: dict[str, float] = {}
    if args.price_aware_rerank:
        for final_row in source_final_rows:
            track_id_for_price = str(final_row.get("track_id", "")).strip() or track_lookup.get(row_key(final_row), "")
            if not track_id_for_price:
                continue
            price = parse_price_value(final_row.get(args.price_field, ""))
            if price is not None:
                price_by_track[track_id_for_price] = price
    recovery_rows: list[dict[str, Any]] = []
    recovered_by_track: dict[str, dict[str, Any]] = {}

    for row in changes:
        track_id = str(row.get("track_id", "")).strip()
        before = surface_text(row.get("after") or row.get("proposed") or row.get("raw_text"))
        tail = tail_lookup.get(track_id, {})
        observed_price = price_by_track.get(track_id) if args.price_aware_rerank else None
        result, match_source_text = best_match_from_ocr_pool(before, tail, catalog, args, observed_price)
        order_source_text = match_source_text
        order_source_score = 0.0
        if result.accepted and args.preserve_ocr_order and args.hybrid_order_source_pool:
            order_source_text, order_source_score = select_hybrid_order_source(
                result.candidate,
                before,
                match_source_text,
                tail,
                args.candidate_pool_size,
            )
        final_value = (
            restore_visible_tail(
                result.candidate,
                order_source_text,
                tail.get("country", ""),
                tail.get("unit", ""),
                bool(tail.get("glass_packaging")) and not args.disable_glass_packaging_tail,
                args.preserve_ocr_order,
            )
            if result.accepted
            else before
        )
        recovered_by_track[track_id] = {
            "before": before,
            "after": final_value,
            "accepted": result.accepted,
            "candidate": result.candidate,
            "score": result.score,
            "margin": result.margin,
            "reason": result.reason,
            "second_candidate": result.second_candidate,
            "second_score": result.second_score,
            "top_candidates": result.top_candidates,
            "match_source_text": match_source_text,
            "order_source_text": order_source_text,
            "order_source_score": order_source_score,
        }
        row["catalog_before"] = before
        row["catalog_after"] = final_value
        row["catalog_candidate"] = result.candidate
        row["catalog_score"] = f"{result.score:.6f}"
        row["catalog_margin"] = f"{result.margin:.6f}"
        row["catalog_reason"] = result.reason
        row["catalog_second_candidate"] = result.second_candidate
        row["catalog_second_score"] = f"{result.second_score:.6f}"
        row["catalog_top_candidates_json"] = json.dumps(result.top_candidates, ensure_ascii=False)
        if result.accepted:
            row["after"] = final_value
            row["proposed"] = final_value
            row["reason"] = "catalog_recovery"
            row["changed"] = int(before != final_value)
        recovery_rows.append(
            {
                "track_id": track_id,
                "ocr_text": before,
                "final_text": final_value,
                "accepted": int(result.accepted),
                "score": f"{result.score:.6f}",
                "margin": f"{result.margin:.6f}",
                "reason": result.reason,
                "candidate": result.candidate,
                "second_candidate": result.second_candidate,
                "second_score": f"{result.second_score:.6f}",
                "match_source_text": match_source_text,
                "order_source_text": order_source_text,
                "order_source_score": f"{order_source_score:.6f}" if order_source_score else "",
                "tail_country": tail.get("country", ""),
                "tail_unit": tail.get("unit", ""),
                "tail_glass_packaging": tail.get("glass_packaging", ""),
                "observed_price_card": f"{observed_price:.2f}" if observed_price is not None else "",
                "top_candidates_json": json.dumps(result.top_candidates, ensure_ascii=False),
            }
        )

    final_rows = read_rows(final_path)
    for row in final_rows:
        track_id = str(row.get("track_id", "")).strip() or track_lookup.get(row_key(row), "")
        if track_id:
            row["track_id"] = track_id
        recovered = recovered_by_track.get(track_id)
        if recovered and recovered["accepted"]:
            row[PRODUCT_FIELD] = recovered["after"]

    debug_rows = read_rows(debug_path)
    for row in debug_rows:
        if row.get("field") != PRODUCT_FIELD:
            continue
        track_id = str(row.get("track_id", "")).strip()
        recovered = recovered_by_track.get(track_id)
        if not recovered:
            continue
        row["source_text"] = recovered["before"]
        row["catalog_candidate"] = recovered["candidate"]
        row["catalog_score"] = f"{recovered['score']:.6f}"
        row["catalog_margin"] = f"{recovered['margin']:.6f}"
        row["catalog_reason"] = recovered["reason"]
        if recovered["accepted"]:
            row["value"] = recovered["after"]
            row["engine"] = "tesseract5_catalog_recovery"

    write_rows(changes_path, changes)
    write_rows(final_path, final_rows, final_fieldnames or OUTPUT_COLUMNS)
    write_rows(debug_path, debug_rows)
    write_rows(ocr_dir / "product_name_catalog_recovery.csv", recovery_rows)

    accepted = sum(1 for row in recovery_rows if str(row["accepted"]) == "1")
    changed = sum(1 for row in recovery_rows if row["ocr_text"] != row["final_text"])
    summary = {
        "engine": "tesseract5_catalog_recovery",
        "source_run_dir": str(source_run_dir),
        "catalog_size": len(catalog),
        "tracks": len(recovery_rows),
        "accepted_recoveries": accepted,
        "changed_product_names": changed,
        "accept_score": args.accept_score,
        "accept_margin": args.accept_margin,
        "strong_score": args.strong_score,
        "preselect": args.preselect,
        "catalog_decoder_mode": args.catalog_decoder_mode,
        "candidate_pool_size": args.candidate_pool_size,
        "vote_top_k": args.vote_top_k,
        "vote_min_score": args.vote_min_score,
        "vote_min_support": args.vote_min_support,
        "vote_accept_score": args.vote_accept_score,
        "vote_accept_margin": args.vote_accept_margin,
        "vote_explain_ratio": args.vote_explain_ratio,
        "price_aware_rerank": bool(args.price_aware_rerank),
        "price_field": args.price_field,
        "price_bonus": args.price_bonus,
        "price_tolerance_rub": args.price_tolerance_rub,
        "price_near_rub": args.price_near_rub,
        "price_penalty": args.price_penalty,
        "price_can_accept": bool(args.price_can_accept),
        "hybrid_order_source_pool": bool(args.hybrid_order_source_pool),
        "disable_glass_packaging_tail": bool(args.disable_glass_packaging_tail),
        "catalog_cache_path": str(getattr(args, "catalog_cache_resolved", "") or ""),
        "catalog_match_cache_size": len(catalog.match_cache),
    }
    (ocr_dir / "product_name_catalog_recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover product_name OCR text using a Lenta product-name catalog.")
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("repro_outputs/quality_43_15_tesseract_crop_quality_weighted_w005"),
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=Path("repro_outputs/quality_43_15_tesseract_catalog_recovery_v1"),
    )
    parser.add_argument("--catalog-json", type=Path, default=Path("data/lenta_all_products_prices_weight.json"))
    parser.add_argument(
        "--catalog-cache",
        type=Path,
        default=Path("repro_outputs/cache/lenta_catalog_product_name_index_v3.pkl"),
    )
    parser.add_argument("--rebuild-catalog-cache", action="store_true")
    parser.add_argument("--disable-catalog-cache", action="store_true")
    parser.add_argument("--preselect", type=int, default=900)
    parser.add_argument("--accept-score", type=float, default=0.63)
    parser.add_argument("--accept-margin", type=float, default=0.035)
    parser.add_argument("--strong-score", type=float, default=0.76)
    parser.add_argument("--candidate-pool-size", type=int, default=0)
    parser.add_argument("--preserve-ocr-order", action="store_true")
    parser.add_argument("--hybrid-order-source-pool", action="store_true")
    parser.add_argument("--disable-glass-packaging-tail", action="store_true")
    parser.add_argument("--catalog-decoder-mode", choices=["best", "vote"], default="best")
    parser.add_argument("--vote-top-k", type=int, default=3)
    parser.add_argument("--vote-min-score", type=float, default=0.50)
    parser.add_argument("--vote-min-support", type=int, default=2)
    parser.add_argument("--vote-accept-score", type=float, default=0.20)
    parser.add_argument("--vote-accept-margin", type=float, default=0.05)
    parser.add_argument("--vote-explain-ratio", type=float, default=0.65)
    parser.add_argument("--price-aware-rerank", action="store_true")
    parser.add_argument("--price-field", default="price_card")
    parser.add_argument("--price-bonus", type=float, default=0.08)
    parser.add_argument("--price-tolerance-rub", type=float, default=1.0)
    parser.add_argument("--price-near-rub", type=float, default=8.0)
    parser.add_argument("--price-penalty", type=float, default=0.02)
    parser.add_argument("--price-can-accept", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_run_dir = args.source_run_dir if args.source_run_dir.is_absolute() else root / args.source_run_dir
    output_run_dir = args.output_run_dir if args.output_run_dir.is_absolute() else root / args.output_run_dir
    catalog_json = args.catalog_json if args.catalog_json.is_absolute() else root / args.catalog_json
    catalog_cache = None
    if not args.disable_catalog_cache:
        catalog_cache = args.catalog_cache if args.catalog_cache.is_absolute() else root / args.catalog_cache
    args.catalog_cache_resolved = catalog_cache

    catalog = load_catalog(catalog_json, catalog_cache, args.rebuild_catalog_cache)
    summary = update_outputs(source_run_dir, output_run_dir, catalog, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
