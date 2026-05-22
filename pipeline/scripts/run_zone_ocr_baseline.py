"""Run local OCR engines on prepared price-tag zones and aggregate top crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


# Shelf labels print discount as an integer percent after dropping the
# fractional part. For reverse-calculating regular price from card price, +0.5
# percentage points is the least-biased midpoint estimate of the hidden fraction
# in [printed_discount, printed_discount + 1).
DISCOUNT_FLOOR_COMPENSATION_PERCENT = 0.5


OUTPUT_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

OCR_TEXT_ZONES = {
    "product_name",
    "price_default_wide",
    "price_default",
    "price_default_compact",
    "price_default_digits",
    "price_card",
    "price_card_number",
    "price_card_big_digits",
    "price_discount",
    "discount_amount",
    "barcode_digits",
    "id_sku",
    "code",
    "print_datetime",
    "bottom_info",
    "special_symbols",
    "additional_info",
}

DECODER_ZONES = {"qr", "barcode"}

FIELD_ZONE_PRIORITY = {
    "product_name": {"product_name": 0.25},
    "price_default": {
        "price_default_compact": 0.32,
        "price_default_digits": 0.28,
        "price_default": 0.18,
        "price_default_wide": 0.10,
    },
    "price_card": {
        "price_card_number": 0.28,
        "price_card_big_digits": 0.25,
        "price_card": 0.18,
        "price_discount": 0.08,
    },
    "price_discount": {"price_discount": 0.22, "price_card": 0.08},
    "barcode": {"barcode": 0.35, "barcode_digits": 0.25},
    "discount_amount": {"discount_amount": 0.30},
    "id_sku": {"bottom_info": 0.25, "id_sku": 0.18},
    "print_datetime": {"bottom_info": 0.24, "print_datetime": 0.18},
    "code": {"bottom_info": 0.22, "code": 0.18},
    "additional_info": {"additional_info": 0.18},
    "special_symbols": {"special_symbols": 0.18},
}

ENGINE_PRIORITY = {
    "zxing": 0.30,
    "opencv_qr": 0.26,
    "paddleocr": 0.0,
    "rapidocr": 0.0,
    "easyocr": 0.0,
}

ENGINE_PLAN_PRESETS = {
    "all": {},
    "lenta_fast": {
        "product_name": ["paddleocr"],
        "additional_info": ["paddleocr"],
        "price_default_wide": ["rapidocr"],
        "price_default": ["rapidocr"],
        "price_default_compact": ["rapidocr"],
        "price_default_digits": ["rapidocr"],
        "price_card": ["easyocr", "rapidocr"],
        "price_card_number": ["easyocr", "rapidocr"],
        "price_card_big_digits": ["easyocr", "rapidocr"],
        "price_discount": ["easyocr", "rapidocr"],
        "discount_amount": ["rapidocr"],
        "barcode_digits": ["rapidocr"],
        "id_sku": ["rapidocr"],
        "code": ["rapidocr"],
        "print_datetime": ["rapidocr"],
        "bottom_info": ["rapidocr"],
        "special_symbols": ["rapidocr"],
    },
}

QR_FIELD_ALIASES = {
    "barcode": "qr_code_barcode",
    "b": "qr_code_barcode",
    "price1": "price1_qr",
    "p1": "price1_qr",
    "price2": "price2_qr",
    "p2": "price2_qr",
    "price3": "price3_qr",
    "p3": "price3_qr",
    "price4": "price4_qr",
    "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count",
    "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price",
    "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count",
    "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price",
    "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr",
    "aP": "action_price_qr",
    "actionCode": "action_code_qr",
    "aC": "action_code_qr",
}


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    box: list[list[float]]


@dataclass(frozen=True)
class OCRResult:
    engine: str
    image_kind: str
    image_path: str
    text: str
    confidence: float
    elapsed_ms: float
    tokens: list[OCRToken]


@dataclass(frozen=True)
class FieldCandidate:
    video_id: str
    filename: str
    track_id: str
    rank: int
    timestamp_ms: str
    frame_timestamp: str
    field: str
    value: str
    score: float
    confidence: float
    valid: bool
    engine: str
    zone: str
    image_kind: str
    image_path: str
    source_text: str


@dataclass(frozen=True)
class OCRTask:
    order: int
    row: dict[str, str]
    engine_name: str
    image_kind: str
    image_path: str


@dataclass(frozen=True)
class OCRTaskOutput:
    order: int
    row: dict[str, str]
    result: OCRResult


@dataclass(frozen=True)
class DecoderTask:
    order: int
    row: dict[str, str]
    decoder_name: str
    image_kind: str
    image_path: str


@dataclass(frozen=True)
class DecoderTaskOutput:
    order: int
    row: dict[str, str]
    results: list[OCRResult]


class RapidOCREngine:
    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def recognize(self, image_bgr: np.ndarray) -> tuple[list[OCRToken], float]:
        start = time.perf_counter()
        result, _ = self.engine(image_bgr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        tokens: list[OCRToken] = []
        for item in result or []:
            box, text, confidence = item
            tokens.append(OCRToken(str(text), float(confidence), to_box(box)))
        return tokens, elapsed_ms


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, gpu: bool) -> None:
        import easyocr

        self.engine = easyocr.Reader(["ru", "en"], gpu=gpu, verbose=False)

    def recognize(self, image_bgr: np.ndarray) -> tuple[list[OCRToken], float]:
        start = time.perf_counter()
        result = self.engine.readtext(image_bgr, detail=1, paragraph=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        tokens = [OCRToken(str(text), float(confidence), to_box(box)) for box, text, confidence in result]
        return tokens, elapsed_ms


class PaddleOCREngine:
    name = "paddleocr"

    def __init__(self, cache_dir: str | None) -> None:
        if cache_dir:
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", cache_dir)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        os.environ.setdefault("FLAGS_use_pir_api", "0")

        from paddleocr import PaddleOCR

        self.engine = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image_bgr: np.ndarray) -> tuple[list[OCRToken], float]:
        start = time.perf_counter()
        result = self.engine.predict(image_bgr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        tokens: list[OCRToken] = []
        for page in result or []:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for index, text in enumerate(texts):
                confidence = float(scores[index]) if index < len(scores) else 0.0
                box = to_box(polys[index]) if index < len(polys) else []
                tokens.append(OCRToken(str(text), confidence, box))
        return tokens, elapsed_ms


def to_box(box: Any) -> list[list[float]]:
    array = np.asarray(box, dtype=np.float32)
    if array.size == 0:
        return []
    array = array.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in array.tolist()]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(csv_path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_image_bgr(path: str) -> np.ndarray:
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def unique_rows_by_track_rank_zone(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    filtered = [row for row in rows if int(row["rank"]) <= top_k]
    filtered.sort(key=lambda row: (row["video_id"], int(row["track_id"]), int(row["rank"]), row["zone"]))
    return filtered


def row_image_variants(row: dict[str, str], variants: list[str]) -> list[tuple[str, str]]:
    key_by_variant = {
        "raw": "zone_raw",
        "enhanced": "zone_enhanced",
        "binary": "zone_binary",
        "tight": "zone_tight",
        "tight_enhanced": "zone_tight_enhanced",
        "tight_binary": "zone_tight_binary",
    }
    fallback = {
        "tight": ["zone_tight", "zone_enhanced", "zone_raw"],
        "tight_enhanced": ["zone_tight_enhanced", "zone_enhanced", "zone_raw"],
        "tight_binary": ["zone_tight_binary", "zone_binary", "zone_enhanced"],
    }
    seen: set[str] = set()
    found: list[tuple[str, str]] = []
    for variant in variants:
        keys = fallback.get(variant, [key_by_variant.get(variant, "")])
        for key in keys:
            path = row.get(key, "").strip()
            if path and path not in seen and Path(path).exists():
                found.append((variant, path))
                seen.add(path)
                break
    return found


def join_tokens(tokens: list[OCRToken]) -> str:
    tokens = sorted(tokens, key=lambda token: token_sort_key(token.box))
    parts = [clean_text(token.text) for token in tokens if clean_text(token.text)]
    return " ".join(parts).strip()


def token_sort_key(box: list[list[float]]) -> tuple[float, float]:
    if not box:
        return (0.0, 0.0)
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return (float(min(ys)), float(min(xs)))


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def latinize_digit_noise(text: str) -> str:
    table = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "О": "0",
            "о": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "З": "3",
            "з": "3",
            "Б": "6",
            "б": "6",
            "S": "5",
            "s": "5",
            "g": "9",
        }
    )
    return text.translate(table)


def token_bounds(token: OCRToken) -> tuple[float, float, float, float]:
    if not token.box:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point[0] for point in token.box]
    ys = [point[1] for point in token.box]
    return min(xs), min(ys), max(xs), max(ys)


def image_size_from_tokens(tokens: list[OCRToken]) -> tuple[float, float]:
    x_max = 1.0
    y_max = 1.0
    for token in tokens:
        _, _, x2, y2 = token_bounds(token)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)
    return x_max, y_max


def numeric_tokens(tokens: list[OCRToken]) -> list[dict[str, float | str]]:
    image_width, image_height = image_size_from_tokens(tokens)
    parsed = []
    for token in tokens:
        raw = latinize_digit_noise(token.text)
        if "%" in raw:
            continue
        digits = re.sub(r"\D", "", raw)
        if not digits:
            continue
        x1, y1, x2, y2 = token_bounds(token)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        parsed.append(
            {
                "digits": digits,
                "raw_text": raw,
                "confidence": float(token.confidence),
                "x1": x1 / image_width,
                "y1": y1 / image_height,
                "x2": x2 / image_width,
                "y2": y2 / image_height,
                "cx": ((x1 + x2) / 2.0) / image_width,
                "cy": ((y1 + y2) / 2.0) / image_height,
                "width": width / image_width,
                "height": height / image_height,
                "area": (width * height) / (image_width * image_height),
            }
        )
    return parsed


def normalize_price_digits(digits: str) -> str:
    digits = re.sub(r"\D", "", digits)
    if not digits:
        return ""
    if len(digits) >= 5:
        return f"{int(digits[:-2])}.{digits[-2:]}"
    return str(int(digits))


def format_price(main_digits: str, cents_digits: str = "") -> str:
    main_digits = re.sub(r"\D", "", main_digits)
    cents_digits = re.sub(r"\D", "", cents_digits)
    if not main_digits:
        return ""
    if cents_digits and len(cents_digits) == 2:
        if len(main_digits) == 4 and main_digits[-1] == cents_digits[0]:
            main_digits = main_digits[:-1]
        return f"{int(main_digits)}.{cents_digits}"
    return normalize_price_digits(main_digits)


def extract_price(tokens: list[OCRToken], source_text: str, field: str, zone: str) -> tuple[str, bool, float]:
    numbers = numeric_tokens(tokens)
    if numbers:
        if field in {"price_card", "price_discount"}:
            value, confidence = extract_large_price(numbers)
            if value:
                return value, is_valid_price(value), confidence
        else:
            value, confidence = extract_default_price(numbers)
            if value:
                return value, is_valid_price(value), confidence

    text = latinize_digit_noise(source_text)
    match = re.search(r"(\d{1,4})[\s,.]+(\d{2})(?!\d)", text)
    if match:
        value = f"{int(match.group(1))}.{match.group(2)}"
        return value, is_valid_price(value), 0.55

    digits = re.sub(r"\D", "", text)
    if len(digits) >= 3:
        value = normalize_price_digits(digits[:5] if len(digits) > 5 else digits)
        return value, is_valid_price(value), 0.45
    return "", False, 0.0


def extract_large_price(numbers: list[dict[str, float | str]]) -> tuple[str, float]:
    candidates = [
        number
        for number in numbers
        if len(str(number["digits"])) >= 3 and float(number["cy"]) > 0.25
    ]
    if not candidates:
        candidates = [number for number in numbers if len(str(number["digits"])) >= 3]
    if not candidates:
        return "", 0.0

    def main_score(number: dict[str, float | str]) -> float:
        digits = str(number["digits"])
        return (
            float(number["height"]) * 2.0
            + float(number["area"]) * 3.0
            + float(number["confidence"])
            + min(len(digits), 4) * 0.05
            + float(number["cy"]) * 0.15
        )

    main = max(candidates, key=main_score)
    main_digits = str(main["digits"])
    cents = find_cents_for_main(main, numbers)
    confidence = float(main["confidence"])
    if cents:
        confidence = (confidence + float(cents["confidence"])) / 2.0
        return format_price(main_digits, str(cents["digits"])[:2]), confidence
    return normalize_price_digits(main_digits), confidence


def extract_default_price(numbers: list[dict[str, float | str]]) -> tuple[str, float]:
    broad_candidates = [number for number in numbers if len(str(number["digits"])) >= 2]
    candidates = [
        number
        for number in numbers
        if len(str(number["digits"])) >= 2 and 0.30 < float(number["cy"]) < 0.85
    ]
    if not candidates:
        candidates = broad_candidates
    if not candidates:
        candidates = numbers

    non_weight_candidates = [
        number
        for number in candidates
        if not looks_like_weight_token(str(number.get("raw_text", "")))
    ]
    if not non_weight_candidates:
        non_weight_candidates = [
            number
            for number in broad_candidates
            if not looks_like_weight_token(str(number.get("raw_text", "")))
        ]
    if non_weight_candidates:
        candidates = non_weight_candidates

    def default_score(number: dict[str, float | str]) -> float:
        digits = str(number["digits"])
        raw_text = str(number.get("raw_text", ""))
        weight_penalty = 0.75 if looks_like_weight_token(raw_text) else 0.0
        right_bonus = max(0.0, float(number["cx"]) - 0.35)
        lower_bonus = max(0.0, float(number["cy"]) - 0.30)
        return (
            float(number["confidence"])
            + right_bonus * 0.35
            + lower_bonus * 0.25
            + min(len(digits), 4) * 0.04
            - weight_penalty
        )

    main = max(candidates, key=default_score)
    cents = find_cents_for_main(main, numbers)
    confidence = float(main["confidence"])
    if cents:
        confidence = (confidence + float(cents["confidence"])) / 2.0
        return format_price(str(main["digits"]), str(cents["digits"])[:2]), confidence
    return normalize_price_digits(str(main["digits"])), confidence


def looks_like_weight_token(text: str) -> bool:
    return bool(re.search(r"\d+\s*(?:г|гр|g|r)\b", text, flags=re.IGNORECASE))


def find_cents_for_main(
    main: dict[str, float | str], numbers: list[dict[str, float | str]]
) -> dict[str, float | str] | None:
    cents_candidates = []
    for number in numbers:
        if number is main:
            continue
        digits = str(number["digits"])
        if len(digits) != 2:
            continue
        if float(number["cx"]) <= float(main["cx"]):
            continue
        if abs(float(number["cy"]) - float(main["cy"])) > 0.22:
            continue
        if float(number["height"]) > float(main["height"]) * 1.05:
            continue
        cents_candidates.append(number)
    if not cents_candidates:
        return None
    return max(cents_candidates, key=lambda item: float(item["confidence"]) + float(item["height"]))


def is_valid_price(value: str) -> bool:
    if not re.fullmatch(r"\d{1,5}(?:\.\d{2})?", value):
        return False
    try:
        return float(value) >= 100.0
    except ValueError:
        return False


def parse_discount_from_text(value: Any) -> tuple[int | None, float]:
    text = latinize_digit_noise(str(value or ""))
    clean = re.search(r"(?<!\d)-?\s*(\d{1,2})\s*%(?!\d)", text)
    if clean:
        percent = int(clean.group(1))
        if 1 <= percent <= 90:
            return percent, 0.0

    noisy = re.search(r"(?<!\d)([1-6]\d)9\s*%(?!\d)", text)
    if noisy:
        percent = int(noisy.group(1))
        if 10 <= percent <= 69:
            return percent, 0.35
    return None, 0.0


def discount_plausibility_penalty(percent: int | None) -> float:
    if percent is None:
        return 0.0
    percent = abs(int(percent))
    if percent <= 70:
        return 0.0
    if percent <= 80:
        return 0.45 + (percent - 70) * 0.03
    return 1.20 + (percent - 80) * 0.12


def parse_discount_candidate(value: Any, source_text: Any = "") -> tuple[int | None, float]:
    source_percent, source_penalty = parse_discount_from_text(source_text)
    if source_percent is not None:
        return source_percent, source_penalty
    return parse_discount_from_text(value)


def extract_discount(source_text: str) -> tuple[str, bool, float]:
    percent, parse_penalty = parse_discount_candidate("", source_text)
    if percent is None:
        return "", False, 0.0
    confidence = max(0.25, 0.9 - parse_penalty - discount_plausibility_penalty(percent) * 0.2)
    return f"-{percent}%", True, confidence


def extract_barcode(source_text: str) -> tuple[str, bool, float]:
    text = latinize_digit_noise(source_text)
    sequences = re.findall(r"\d{8,14}", re.sub(r"\s+", "", text))
    if not sequences:
        digits = re.sub(r"\D", "", text)
        if 8 <= len(digits) <= 14:
            sequences = [digits]
        elif len(digits) > 14:
            sequences = re.findall(r"\d{13}", digits)
    if not sequences:
        return "", False, 0.0
    value = max(sequences, key=lambda sequence: (len(sequence) == 13, len(sequence)))
    return value, len(value) == 13, 0.95 if len(value) == 13 else 0.75


def extract_id_sku(source_text: str) -> tuple[str, bool, float]:
    text = latinize_digit_noise(source_text)
    sku_match = re.search(r"(?:sku|id[_\s-]*sku|арт\w*)\D{0,8}(\d{5,15})", text, flags=re.IGNORECASE)
    if sku_match:
        return sku_match.group(1), True, 0.88
    candidates = re.findall(r"\d{6,15}", text)
    candidates = [candidate for candidate in candidates if not looks_like_date_fragment(candidate)]
    if not candidates:
        return "", False, 0.0
    value = max(candidates, key=len)
    return value, len(value) >= 6, 0.65


def looks_like_date_fragment(value: str) -> bool:
    return len(value) in {6, 8} and value[:2].isdigit() and value[2:4].isdigit()


def extract_datetime(source_text: str) -> tuple[str, bool, float]:
    text = latinize_digit_noise(source_text)
    match = re.search(
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?",
        text,
    )
    if not match:
        return "", False, 0.0
    date = match.group(1).replace("-", ".").replace("/", ".")
    time_part = match.group(2) or ""
    return f"{date} {time_part}".strip(), True, 0.88


def extract_code(source_text: str) -> tuple[str, bool, float]:
    text = latinize_digit_noise(source_text)
    match = re.search(r"\b(\d{1,3}\s*[_-]\s*\d{4,}(?:\s*[-/]\s*\d{2,})?)\b", text)
    if not match:
        code_match = re.search(r"(?:код|code)\D{0,8}([0-9_\-/\s]{5,})", text, flags=re.IGNORECASE)
        if not code_match:
            return "", False, 0.0
        value = clean_text(code_match.group(1))
    else:
        value = match.group(1)
    value = re.sub(r"\s+", "", value)
    return value, bool(re.search(r"\d", value)), 0.82


def extract_text_field(source_text: str) -> tuple[str, bool, float]:
    text = clean_text(source_text)
    text = re.sub(r"[|_]{2,}", " ", text)
    text = clean_text(text)
    if len(re.sub(r"\W", "", text, flags=re.UNICODE)) < 2:
        return "", False, 0.0
    return text, True, 0.70


def parse_qr_payload(payload: str) -> dict[str, str]:
    payload = payload.strip()
    if not payload:
        return {}

    parsed: dict[str, str] = {}
    parsed_url = urlparse(payload)
    query = parsed_url.query if parsed_url.query else payload
    query = query.replace(";", "&").replace("|", "&").replace("\n", "&")
    for key, values in parse_qs(query, keep_blank_values=True).items():
        mapped_key = QR_FIELD_ALIASES.get(key)
        if mapped_key and values:
            parsed[mapped_key] = values[0]

    for key, value in re.findall(r"([A-Za-z0-9_]+)\s*[:=]\s*([^,&;|\s]+)", payload):
        mapped_key = QR_FIELD_ALIASES.get(key)
        if mapped_key:
            parsed[mapped_key] = value
    return parsed


def decode_with_zxing(image_bgr: np.ndarray) -> list[str]:
    import zxingcpp

    results = zxingcpp.read_barcodes(image_bgr)
    return [str(result.text) for result in results if getattr(result, "text", "")]


def decode_qr_with_opencv(image_bgr: np.ndarray) -> list[str]:
    detector = cv2.QRCodeDetector()
    values: list[str] = []
    value, _, _ = detector.detectAndDecode(image_bgr)
    if value:
        values.append(value)
    ok, decoded, _, _ = detector.detectAndDecodeMulti(image_bgr)
    if ok:
        values.extend([item for item in decoded if item])
    return list(dict.fromkeys(values))


def build_ocr_result(
    engine_name: str,
    image_kind: str,
    image_path: str,
    tokens: list[OCRToken],
    elapsed_ms: float,
) -> OCRResult:
    confidence = sum(token.confidence for token in tokens) / max(1, len(tokens))
    return OCRResult(
        engine=engine_name,
        image_kind=image_kind,
        image_path=image_path,
        text=join_tokens(tokens),
        confidence=confidence,
        elapsed_ms=elapsed_ms,
        tokens=tokens,
    )


def candidates_from_result(row: dict[str, str], result: OCRResult) -> list[FieldCandidate]:
    zone = row["zone"]
    target_fields = [field for field in row["target_fields"].split("|") if field]
    candidates: list[FieldCandidate] = []
    for field in target_fields:
        value = ""
        valid = False
        confidence = result.confidence
        if field in {"price_default", "price_card", "price_discount"}:
            value, valid, confidence = extract_price(result.tokens, result.text, field, zone)
            if value and not valid:
                continue
        elif field == "discount_amount":
            value, valid, confidence = extract_discount(result.text)
        elif field == "barcode":
            value, valid, confidence = extract_barcode(result.text)
            if value and not valid:
                continue
        elif field == "id_sku":
            value, valid, confidence = extract_id_sku(result.text)
        elif field == "print_datetime":
            value, valid, confidence = extract_datetime(result.text)
        elif field == "code":
            value, valid, confidence = extract_code(result.text)
        elif field in {"product_name", "additional_info", "special_symbols"}:
            value, valid, confidence = extract_text_field(result.text)
        else:
            continue
        if not value:
            continue
        candidates.append(make_candidate(row, field, value, valid, confidence, result))
    return candidates


def candidates_from_decoder(row: dict[str, str], engine: str, image_kind: str, image_path: str, payload: str) -> list[FieldCandidate]:
    result = OCRResult(
        engine=engine,
        image_kind=image_kind,
        image_path=image_path,
        text=payload,
        confidence=1.0,
        elapsed_ms=0.0,
        tokens=[],
    )
    zone = row["zone"]
    candidates: list[FieldCandidate] = []
    if zone == "barcode":
        value, valid, confidence = extract_barcode(payload)
        if value:
            candidates.append(make_candidate(row, "barcode", value, valid, confidence, result))
    if zone == "qr":
        parsed = parse_qr_payload(payload)
        for field, value in parsed.items():
            valid = bool(value)
            candidates.append(make_candidate(row, field, value, valid, 1.0, result))
        value, valid, confidence = extract_barcode(payload)
        if value:
            candidates.append(make_candidate(row, "qr_code_barcode", value, valid, confidence, result))
    return candidates


def target_fields_for_row(row: dict[str, str]) -> list[str]:
    return [field for field in row.get("target_fields", "").split("|") if field]


def cascade_key(row: dict[str, str], field: str) -> tuple[str, str, str]:
    return (row.get("video_id", ""), row.get("track_id", ""), field)


def candidate_satisfies_early_stop(candidate: FieldCandidate) -> bool:
    field = candidate.field
    value = candidate.value
    if field == "price_default":
        return candidate.valid and is_valid_price(value) and "." in value and candidate.confidence >= 0.60
    if field in {"price_card", "price_discount"}:
        return candidate.valid and is_valid_price(value) and "." in value and candidate.confidence >= 0.92
    if field == "discount_amount":
        return candidate.valid and candidate.confidence >= 0.80
    if field in {"barcode", "qr_code_barcode"}:
        return candidate.valid and len(re.sub(r"\D", "", value)) == 13
    if field in {
        "price1_qr",
        "price2_qr",
        "price3_qr",
        "price4_qr",
        "wholesale_level_1_count",
        "wholesale_level_1_price",
        "wholesale_level_2_count",
        "wholesale_level_2_price",
        "action_price_qr",
        "action_code_qr",
    }:
        return candidate.valid and bool(value)
    if field in {"product_name", "additional_info", "special_symbols", "id_sku", "print_datetime", "code"}:
        return candidate.valid and candidate.confidence >= 0.70 and candidate.score >= 1.05
    return candidate.valid


def make_candidate(
    row: dict[str, str],
    field: str,
    value: str,
    valid: bool,
    confidence: float,
    result: OCRResult,
) -> FieldCandidate:
    zone = row["zone"]
    rank = int(row["rank"])
    score = float(confidence)
    score += FIELD_ZONE_PRIORITY.get(field, {}).get(zone, 0.0)
    score += ENGINE_PRIORITY.get(result.engine, 0.0)
    score += max(0.0, 0.12 - (rank - 1) * 0.04)
    score += 0.35 if valid else 0.0
    if field == "barcode" and len(re.sub(r"\D", "", value)) == 13:
        score += 0.15
    if field in {"price_card", "price_default", "price_discount"} and "." in value:
        score += 0.08
    if field in {"product_name", "additional_info", "special_symbols"} and re.search(r"[А-Яа-яЁё]", value):
        score += 0.18
    return FieldCandidate(
        video_id=row["video_id"],
        filename=row["filename"],
        track_id=row["track_id"],
        rank=rank,
        timestamp_ms=row["timestamp_ms"],
        frame_timestamp=row["frame_timestamp"],
        field=field,
        value=value,
        score=score,
        confidence=float(confidence),
        valid=valid,
        engine=result.engine,
        zone=zone,
        image_kind=result.image_kind,
        image_path=result.image_path,
        source_text=result.text,
    )


def load_engines(names: list[str], gpu: bool, paddle_cache: str | None) -> dict[str, Any]:
    engines: dict[str, Any] = {}
    for name in names:
        if name == "rapidocr":
            engines[name] = RapidOCREngine()
        elif name == "easyocr":
            engines[name] = EasyOCREngine(gpu=gpu)
        elif name == "paddleocr":
            engines[name] = PaddleOCREngine(cache_dir=paddle_cache)
        else:
            raise ValueError(f"Unknown OCR engine: {name}")
    return engines


_THREAD_LOCAL = threading.local()


def configure_parallel_cpu_threads(jobs: int) -> None:
    if jobs <= 1:
        return
    for key in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def get_thread_engine(engine_name: str, gpu: bool, paddle_cache: str | None) -> Any:
    engines = getattr(_THREAD_LOCAL, "engines", None)
    if engines is None:
        engines = {}
        _THREAD_LOCAL.engines = engines
    if engine_name not in engines:
        engines[engine_name] = load_engines([engine_name], gpu=gpu, paddle_cache=paddle_cache)[engine_name]
    return engines[engine_name]


def run_ocr_task(task: OCRTask, engine: Any | None = None, gpu: bool = False, paddle_cache: str | None = None) -> OCRTaskOutput:
    try:
        image = read_image_bgr(task.image_path)
        task_engine = engine if engine is not None else get_thread_engine(task.engine_name, gpu, paddle_cache)
        tokens, elapsed_ms = task_engine.recognize(image)
    except Exception as exc:  # noqa: BLE001
        tokens = [OCRToken(f"ERROR: {exc}", 0.0, [])]
        elapsed_ms = 0.0
    result = build_ocr_result(task.engine_name, task.image_kind, task.image_path, tokens, elapsed_ms)
    return OCRTaskOutput(task.order, task.row, result)


def run_decoder_task(task: DecoderTask) -> DecoderTaskOutput:
    try:
        image = read_image_bgr(task.image_path)
        if task.decoder_name == "zxing":
            payloads = decode_with_zxing(image)
        elif task.decoder_name == "opencv_qr":
            payloads = decode_qr_with_opencv(image)
        else:
            raise ValueError(f"Unknown decoder: {task.decoder_name}")
    except Exception as exc:  # noqa: BLE001
        payloads = [f"ERROR: {exc}"]
    results = [
        OCRResult(task.decoder_name, task.image_kind, task.image_path, payload, 1.0, 0.0, [])
        for payload in payloads
    ]
    return DecoderTaskOutput(task.order, task.row, results)


def append_ocr_output(
    output: OCRTaskOutput,
    raw_rows: list[dict[str, Any]],
    all_candidates: list[FieldCandidate],
) -> None:
    raw_rows.append(result_to_row(output.row, output.result))
    if output.result.text.startswith("ERROR:"):
        return
    all_candidates.extend(candidates_from_result(output.row, output.result))


def append_decoder_output(
    output: DecoderTaskOutput,
    raw_rows: list[dict[str, Any]],
    all_candidates: list[FieldCandidate],
) -> None:
    for result in output.results:
        raw_rows.append(result_to_row(output.row, result))
        if result.text.startswith("ERROR:"):
            continue
        all_candidates.extend(
            candidates_from_decoder(
                output.row,
                result.engine,
                result.image_kind,
                result.image_path,
                result.text,
            )
        )


def parse_engine_plan(value: str) -> dict[str, list[str]]:
    value = value.strip()
    if not value or value in ENGINE_PLAN_PRESETS:
        return ENGINE_PLAN_PRESETS.get(value or "all", {})

    plan: dict[str, list[str]] = {}
    for item in value.split(";"):
        if not item.strip() or "=" not in item:
            continue
        key, raw_engines = item.split("=", 1)
        engines = [engine.strip() for engine in raw_engines.split("+") if engine.strip()]
        if key.strip() and engines:
            plan[key.strip()] = engines
    return plan


def engines_for_row(row: dict[str, str], engine_names: list[str], plan: dict[str, list[str]]) -> list[str]:
    if not plan:
        return engine_names

    zone = row.get("zone", "").strip()
    target_fields = [field for field in row.get("target_fields", "").split("|") if field]
    requested = list(plan.get(zone, []))
    for field in target_fields:
        for engine in plan.get(field, []):
            if engine not in requested:
                requested.append(engine)
    if not requested:
        return engine_names
    return [engine for engine in engine_names if engine in requested]


def required_engines_for_rows(
    rows: list[dict[str, str]], engine_names: list[str], plan: dict[str, list[str]]
) -> list[str]:
    required: list[str] = []
    for row in rows:
        if row.get("zone") not in OCR_TEXT_ZONES:
            continue
        for engine in engines_for_row(row, engine_names, plan):
            if engine not in required:
                required.append(engine)
    return required


def aggregate_submission(
    candidates: list[FieldCandidate],
    template_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[FieldCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.video_id, candidate.track_id, candidate.field)].append(candidate)

    debug_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, str]] = []
    for template in template_rows:
        row = {column: template.get(column, "") for column in OUTPUT_COLUMNS}
        video_id = Path(template.get("filename", "")).stem
        track_id = template.get("track_id", "")
        if not track_id:
            track_id = template.get("track_id_export", "")
        selected_timestamp = template.get("frame_timestamp", "")
        discount_best = select_best_candidate(
            "discount_amount",
            grouped.get((video_id, track_id, "discount_amount"), []),
            selected_timestamp,
        )
        discount_percent = parse_discount_percent_value(discount_best.value if discount_best else "")
        price_card_best = select_best_candidate(
            "price_card",
            grouped.get((video_id, track_id, "price_card"), []),
            selected_timestamp,
            discount_percent,
        )
        price_card_value, _ = parse_candidate_price(price_card_best.value if price_card_best else "")
        for field in OUTPUT_COLUMNS:
            if field in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                continue
            choices = grouped.get((video_id, track_id, field), [])
            if not choices:
                continue
            if field == "price_card":
                best = price_card_best
            else:
                best = select_best_candidate(field, choices, selected_timestamp, discount_percent, price_card_value)
            if best is None:
                continue
            row[field] = output_value_for_field(field, best.value)
            debug_rows.append(candidate_to_row(best))
        if not row.get("price_default") and price_card_value is not None and discount_percent is not None:
            inferred_default = infer_default_from_card_discount(price_card_value, discount_percent)
            if inferred_default is not None:
                row["price_default"] = f"{inferred_default:.2f}"
                if not row.get("price1_qr"):
                    row["price1_qr"] = f"{inferred_default:.2f}"
        if not row.get("discount_amount"):
            card_for_discount, _ = parse_candidate_price(row.get("price_card", ""))
            default_for_discount, _ = parse_candidate_price(row.get("price_default", ""))
            inferred_discount = infer_discount_from_prices(card_for_discount, default_for_discount)
            if inferred_discount:
                row["discount_amount"] = inferred_discount
        output_rows.append(row)
    return output_rows, debug_rows


def output_value_for_field(field: str, value: str) -> str:
    if field in {"price_card", "price_discount"} and re.fullmatch(r"\d{3,4}", value):
        return f"{value}.99"
    return value


def parse_candidate_price(value: str) -> tuple[float | None, bool]:
    text = str(value or "").strip().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None, False
    token = match.group(0)
    try:
        return float(token), "." in token
    except ValueError:
        return None, "." in token


def normalize_shelf_price(value: float) -> float:
    return max(0.0, int(value) + 0.99)


def closest_price_with_cents(value: float, endings: tuple[int, ...]) -> float:
    base = math.floor(value)
    candidates: list[float] = []
    for integer_part in range(max(0, base - 1), base + 2):
        for cents in endings:
            candidates.append(integer_part + cents / 100.0)
    return min(candidates, key=lambda candidate: abs(candidate - value))


def infer_default_from_card_discount(card_value: float | None, discount_percent: int | None) -> float | None:
    if card_value is None or discount_percent is None:
        return None
    displayed_percent = abs(float(discount_percent))
    if displayed_percent < 5.0 or displayed_percent > 70.0:
        return None
    adjusted_percent = displayed_percent + DISCOUNT_FLOOR_COMPENSATION_PERCENT
    factor = 1.0 - adjusted_percent / 100.0
    if factor < 0.10 or factor >= 1.0:
        return None
    inferred = float(card_value) / factor
    if inferred <= float(card_value) * 1.02 or inferred > 20000.0:
        return None
    return closest_price_with_cents(inferred, (9, 19, 29, 39, 49, 59, 69, 79, 89, 99))


def infer_discount_from_prices(card_value: float | None, default_value: float | None) -> str:
    if card_value is None or default_value is None:
        return ""
    card = float(card_value)
    default = float(default_value)
    if default <= card * 1.02 or default > card * 4.0:
        return ""
    raw_percent = (1.0 - card / default) * 100.0
    percent = int(math.floor(raw_percent + 1e-9))
    if 5 <= percent <= 70:
        return f"-{percent}%"
    return ""


def max_digit_run(text: str) -> int:
    return max((len(run) for run in re.findall(r"\d+", text or "")), default=0)


def has_price_marker(text: str) -> bool:
    return bool(re.search(r"[\u00b0\u00ba]|(?:^|\s)99(?:\s|$)|rub", text or "", re.IGNORECASE))


def parse_discount_percent_value(value: str) -> int | None:
    percent, _ = parse_discount_candidate(value)
    return percent


def discount_fragment_penalty(
    field: str,
    value: float,
    source_text: str,
    atom_kind: str,
    discount_percent: int | None,
) -> float:
    if field not in {"price_card", "price_discount"}:
        return 0.0
    whole = str(int(value))
    source = source_text or ""
    penalty = 0.0

    if discount_percent is not None:
        discount_digits = str(abs(discount_percent))
        if len(discount_digits) >= 2 and whole.startswith(discount_digits) and len(whole) > len(discount_digits):
            penalty += 0.95
            if atom_kind in {"source_99", "parser_value"}:
                penalty += 0.25

    for match in re.finditer(r"(?<!\d)(\d{2})\s*[#%]\s*(\d)(?!\d)", source):
        if whole.startswith(match.group(1) + match.group(2)):
            penalty += 1.35

    return penalty


def source_price_atoms(text: str) -> list[tuple[float, str]]:
    atoms: list[tuple[float, str]] = []
    for match in re.finditer(r"(?<!\d)(\d{2,4})\s*[\u00b0\u00ba]\s*(\d{2})?(?!\d)", text or ""):
        whole = int(match.group(1))
        cents = int(match.group(2)) if match.group(2) and 0 <= int(match.group(2)) <= 99 else 99
        if 10 <= whole <= 9999:
            atoms.append((whole + cents / 100.0, "source_marker"))
    for match in re.finditer(r"(?<!\d)(\d{2,4})\s+99(?!\d)", text or ""):
        whole = int(match.group(1))
        if 10 <= whole <= 9999:
            atoms.append((whole + 0.99, "source_99"))
    return atoms


def timestamp_price_bonus(candidate: FieldCandidate, selected_timestamp_ms: str | None) -> tuple[float, int]:
    try:
        diff = abs(int(float(candidate.timestamp_ms or 0)) - int(float(selected_timestamp_ms or 0)))
    except (TypeError, ValueError):
        return 0.0, 10**9
    if diff <= 250:
        return 0.36, diff
    if diff <= 500:
        return 0.25, diff
    if diff <= 1000:
        return 0.05, diff
    if diff <= 2000:
        return -0.22, diff
    if diff <= 4000:
        return -0.48, diff
    return -0.85, diff


def reject_price_atom(field: str, value: float, source_text: str, explicit_decimal: bool) -> bool:
    if value < 10.0 or value > 9999.0:
        return True
    if field in {"price_card", "price_discount"} and value > 5000.0 and not has_price_marker(source_text):
        return True
    if explicit_decimal and max_digit_run(source_text) >= 5 and not re.search(r"[,.]", source_text or ""):
        return True
    if field in {"price_card", "price_discount"} and explicit_decimal and value < 200.0:
        whole = int(value)
        if re.search(rf"(?<!\d){whole}\d\s*[\u00b0\u00ba]", source_text or ""):
            return True
    return False


def select_best_candidate(
    field: str,
    choices: list[FieldCandidate],
    selected_timestamp_ms: str | None = None,
    discount_percent: int | None = None,
    card_value: float | None = None,
) -> FieldCandidate | None:
    if not choices:
        return None
    if field in {"price_default", "price_card", "price_discount"}:
        valid_choices = [choice for choice in choices if choice.valid and is_valid_price(choice.value)]
        if valid_choices:
            return select_consensus_price(field, valid_choices, selected_timestamp_ms, discount_percent, card_value)
    if field == "discount_amount":
        valid_choices = [choice for choice in choices if choice.valid]
        if valid_choices:
            return select_consensus_discount(valid_choices, selected_timestamp_ms)
        return None
    if field == "barcode":
        valid_choices = [choice for choice in choices if choice.valid and len(re.sub(r"\D", "", choice.value)) == 13]
        if valid_choices:
            return max(valid_choices, key=lambda candidate: candidate.score)
        return None
    return max(choices, key=lambda candidate: candidate.score)


def select_consensus_price(
    field: str,
    choices: list[FieldCandidate],
    selected_timestamp_ms: str | None = None,
    discount_percent: int | None = None,
    card_value: float | None = None,
) -> FieldCandidate | None:
    grouped: dict[str, list[tuple[FieldCandidate, float]]] = defaultdict(list)
    for choice in choices:
        time_bonus, _ = timestamp_price_bonus(choice, selected_timestamp_ms)
        common_score = choice.score + time_bonus
        if choice.image_kind == "tight_enhanced":
            common_score += 0.02
        if choice.engine == "rapidocr":
            common_score += 0.02
        if has_price_marker(choice.source_text):
            common_score += 0.08

        for source_value, kind in source_price_atoms(choice.source_text):
            normalized = normalize_shelf_price(source_value)
            if reject_price_atom(field, normalized, choice.source_text, False):
                continue
            if field == "price_default" and card_value is not None and normalized <= card_value * 1.02:
                continue
            discount_penalty = discount_fragment_penalty(field, normalized, choice.source_text, kind, discount_percent)
            grouped[f"{normalized:.2f}"].append(
                (
                    replace(choice, value=f"{normalized:.2f}"),
                    common_score + (0.44 if kind == "source_marker" else 0.34) - discount_penalty,
                )
            )

        parsed, explicit_decimal = parse_candidate_price(choice.value)
        if parsed is None:
            continue
        normalized = normalize_shelf_price(parsed)
        if reject_price_atom(field, normalized, choice.source_text, explicit_decimal):
            continue
        if field == "price_default" and card_value is not None and normalized <= card_value * 1.02:
            continue
        score = common_score
        if explicit_decimal and field in {"price_card", "price_discount"} and abs(parsed - normalized) > 0.02:
            score -= 0.28
        if max_digit_run(choice.source_text) >= 5 and not re.search(r"[\u00b0\u00ba]|\s99(?!\d)", choice.source_text):
            score -= 0.45
        if normalized > 5000.0:
            score -= 1.0
        elif normalized > 3000.0:
            score -= 0.35
        score -= discount_fragment_penalty(field, normalized, choice.source_text, "parser_value", discount_percent)
        grouped[f"{normalized:.2f}"].append((replace(choice, value=f"{normalized:.2f}"), score))

    if not grouped:
        return None

    def price_value_score(value: str, value_choices: list[tuple[FieldCandidate, float]]) -> float:
        best_score = max(score for _, score in value_choices)
        support = min(len(value_choices), 8) * 0.07
        timestamps = {candidate.timestamp_ms for candidate, _ in value_choices}
        close_support = sum(
            1 for candidate, _ in value_choices if timestamp_price_bonus(candidate, selected_timestamp_ms)[1] <= 500
        )
        marker_support = sum(
            1 for candidate, _ in value_choices if has_price_marker(candidate.source_text)
        )
        return best_score + support + min(len(timestamps), 4) * 0.05 + min(close_support, 4) * 0.10 + min(marker_support, 4) * 0.04

    best_value = max(grouped, key=lambda value: price_value_score(value, grouped[value]))
    return max(grouped[best_value], key=lambda item: item[1])[0]


def select_consensus_discount(
    choices: list[FieldCandidate],
    selected_timestamp_ms: str | None = None,
) -> FieldCandidate | None:
    grouped: dict[str, list[tuple[FieldCandidate, float]]] = defaultdict(list)
    for choice in choices:
        percent, parse_penalty = parse_discount_candidate(choice.value, choice.source_text)
        if percent is None:
            continue
        time_bonus, _ = timestamp_price_bonus(choice, selected_timestamp_ms)
        score = choice.score + time_bonus - parse_penalty - discount_plausibility_penalty(percent)
        if choice.image_kind == "tight_enhanced":
            score += 0.05
        grouped[f"-{percent}%"].append((replace(choice, value=f"-{percent}%"), score))

    if not grouped:
        return None

    def discount_score(value_choices: list[tuple[FieldCandidate, float]]) -> float:
        best_score = max(score for _, score in value_choices)
        support = min(len(value_choices), 6) * 0.09
        close_support = sum(
            1 for candidate, _ in value_choices if timestamp_price_bonus(candidate, selected_timestamp_ms)[1] <= 500
        )
        return best_score + support + min(close_support, 4) * 0.10

    best_value = max(grouped, key=lambda value: discount_score(grouped[value]))
    return max(grouped[best_value], key=lambda item: item[1])[0]


def build_template_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows_by_track: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_rows:
        key = (row["video_id"], row["track_id"])
        if key in rows_by_track and int(row["rank"]) != 1:
            continue
        rows_by_track.setdefault(
            key,
            {
                "track_id": row["track_id"],
                "filename": row["filename"],
                "frame_timestamp": row["frame_timestamp"],
                "x_min": row["x_min"],
                "y_min": row["y_min"],
                "x_max": row["x_max"],
                "y_max": row["y_max"],
            },
        )
    return list(rows_by_track.values())


def enrich_template_track_ids(
    template_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    template_by_key = {}
    for row in manifest_rows:
        if row["rank"] != "1":
            continue
        key = (
            row["filename"],
            row["frame_timestamp"],
            row["x_min"],
            row["y_min"],
            row["x_max"],
            row["y_max"],
        )
        template_by_key[key] = row["track_id"]

    for row in template_rows:
        row.setdefault("track_id", "")
        key = (
            row.get("filename", ""),
            row.get("frame_timestamp", ""),
            row.get("x_min", ""),
            row.get("y_min", ""),
            row.get("x_max", ""),
            row.get("y_max", ""),
        )
        if not row["track_id"]:
            row["track_id"] = template_by_key.get(key, "")
    return template_rows


def result_to_row(row: dict[str, str], result: OCRResult) -> dict[str, Any]:
    return {
        "video_id": row["video_id"],
        "filename": row["filename"],
        "track_id": row["track_id"],
        "rank": row["rank"],
        "timestamp_ms": row["timestamp_ms"],
        "frame_timestamp": row["frame_timestamp"],
        "zone": row["zone"],
        "zone_mode": row["zone_mode"],
        "target_fields": row["target_fields"],
        "engine": result.engine,
        "image_kind": result.image_kind,
        "image_path": result.image_path,
        "text": result.text,
        "confidence": f"{result.confidence:.4f}",
        "elapsed_ms": f"{result.elapsed_ms:.1f}",
        "tokens_json": json.dumps(
            [
                {
                    "text": token.text,
                    "confidence": token.confidence,
                    "box": token.box,
                }
                for token in result.tokens
            ],
            ensure_ascii=False,
        ),
    }


def candidate_to_row(candidate: FieldCandidate) -> dict[str, Any]:
    return {
        "video_id": candidate.video_id,
        "filename": candidate.filename,
        "track_id": candidate.track_id,
        "rank": candidate.rank,
        "timestamp_ms": candidate.timestamp_ms,
        "frame_timestamp": candidate.frame_timestamp,
        "field": candidate.field,
        "value": candidate.value,
        "score": f"{candidate.score:.4f}",
        "confidence": f"{candidate.confidence:.4f}",
        "valid": int(candidate.valid),
        "engine": candidate.engine,
        "zone": candidate.zone,
        "image_kind": candidate.image_kind,
        "image_path": candidate.image_path,
        "source_text": candidate.source_text,
    }


def candidate_from_row(row: dict[str, str]) -> FieldCandidate:
    return FieldCandidate(
        video_id=row["video_id"],
        filename=row["filename"],
        track_id=row["track_id"],
        rank=int(row["rank"]),
        timestamp_ms=row["timestamp_ms"],
        frame_timestamp=row["frame_timestamp"],
        field=row["field"],
        value=row["value"],
        score=float(row["score"]),
        confidence=float(row["confidence"]),
        valid=row.get("valid", "0") in {"1", "true", "True"},
        engine=row["engine"],
        zone=row["zone"],
        image_kind=row["image_kind"],
        image_path=row["image_path"],
        source_text=row["source_text"],
    )


def summarize(candidates: list[FieldCandidate], output_rows: list[dict[str, str]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    filled_by_field = {
        field: sum(1 for row in output_rows if row.get(field, "").strip())
        for field in OUTPUT_COLUMNS
        if field not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
    }
    candidates_by_engine: dict[str, int] = defaultdict(int)
    valid_by_field: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        candidates_by_engine[candidate.engine] += 1
        if candidate.valid:
            valid_by_field[candidate.field] += 1
    return {
        "tracks": len(output_rows),
        "raw_ocr_rows": len(raw_rows),
        "field_candidates": len(candidates),
        "filled_by_field": filled_by_field,
        "valid_candidates_by_field": dict(sorted(valid_by_field.items())),
        "candidates_by_engine": dict(sorted(candidates_by_engine.items())),
    }


def run_rows_parallel(
    manifest_rows: list[dict[str, str]],
    engine_names: list[str],
    decoder_names: list[str],
    image_variants: list[str],
    decoder_variants: list[str],
    engine_plan: dict[str, list[str]],
    gpu: bool,
    paddle_cache: str | None,
    jobs: int,
    parallel_engine_mode: str,
) -> tuple[list[dict[str, Any]], list[FieldCandidate], dict[str, int]]:
    parallel_ocr_engines = {"rapidocr"}
    required_engine_names = required_engines_for_rows(manifest_rows, engine_names, engine_plan)
    main_engine_names = [name for name in required_engine_names if name not in parallel_ocr_engines]
    if parallel_engine_mode == "shared":
        main_engine_names.extend(name for name in required_engine_names if name in parallel_ocr_engines)
    engines = load_engines(main_engine_names, gpu=gpu, paddle_cache=paddle_cache)

    serial_ocr_tasks: list[OCRTask] = []
    parallel_ocr_tasks: list[OCRTask] = []
    decoder_tasks: list[DecoderTask] = []
    recognition_calls = 0
    decoder_calls = 0
    cascade_executed_rows = 0
    total_rows = len(manifest_rows)
    order = 0

    for index, row in enumerate(manifest_rows, start=1):
        zone = row["zone"]
        row_ran = False
        if zone in OCR_TEXT_ZONES:
            row_engines = engines_for_row(row, engine_names, engine_plan)
            for image_kind, image_path in row_image_variants(row, image_variants):
                for engine_name in row_engines:
                    task = OCRTask(order, row, engine_name, image_kind, image_path)
                    order += 1
                    recognition_calls += 1
                    row_ran = True
                    if engine_name in parallel_ocr_engines:
                        parallel_ocr_tasks.append(task)
                    else:
                        serial_ocr_tasks.append(task)
        if zone in DECODER_ZONES and decoder_names:
            for image_kind, image_path in row_image_variants(row, decoder_variants):
                if "zxing" in decoder_names:
                    decoder_tasks.append(DecoderTask(order, row, "zxing", image_kind, image_path))
                    order += 1
                    decoder_calls += 1
                    row_ran = True
                if zone == "qr" and "opencv_qr" in decoder_names:
                    decoder_tasks.append(DecoderTask(order, row, "opencv_qr", image_kind, image_path))
                    order += 1
                    decoder_calls += 1
                    row_ran = True
        if row_ran:
            cascade_executed_rows += 1
        if index % 100 == 0:
            print(f"Scheduled {index}/{total_rows} zone rows")

    ordered_outputs: list[OCRTaskOutput | DecoderTaskOutput] = []
    if parallel_ocr_tasks:
        print(f"Running {len(parallel_ocr_tasks)} rapidocr calls with {jobs} workers")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = []
            for task in parallel_ocr_tasks:
                shared_engine = engines[task.engine_name] if parallel_engine_mode == "shared" else None
                futures.append(executor.submit(run_ocr_task, task, shared_engine, gpu, paddle_cache))
            for task in serial_ocr_tasks:
                ordered_outputs.append(run_ocr_task(task, engines[task.engine_name], gpu, paddle_cache))
            completed = 0
            for future in as_completed(futures):
                ordered_outputs.append(future.result())
                completed += 1
                if completed % 100 == 0:
                    print(f"Completed {completed}/{len(parallel_ocr_tasks)} parallel rapidocr calls")
    else:
        for task in serial_ocr_tasks:
            ordered_outputs.append(run_ocr_task(task, engines[task.engine_name], gpu, paddle_cache))

    if decoder_tasks:
        print(f"Running {len(decoder_tasks)} decoder calls with {jobs} workers")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(run_decoder_task, task) for task in decoder_tasks]
            completed = 0
            for future in as_completed(futures):
                ordered_outputs.append(future.result())
                completed += 1
                if completed % 200 == 0:
                    print(f"Completed {completed}/{len(decoder_tasks)} decoder calls")

    raw_rows: list[dict[str, Any]] = []
    all_candidates: list[FieldCandidate] = []
    for output in sorted(ordered_outputs, key=lambda item: item.order):
        if isinstance(output, OCRTaskOutput):
            append_ocr_output(output, raw_rows, all_candidates)
        else:
            append_decoder_output(output, raw_rows, all_candidates)

    return raw_rows, all_candidates, {
        "cascade_executed_rows": cascade_executed_rows,
        "cascade_skipped_rows": 0,
        "recognition_calls": recognition_calls,
        "decoder_calls": decoder_calls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission-template", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--engines", default="rapidocr,easyocr,paddleocr")
    parser.add_argument("--engine-plan", default="all")
    parser.add_argument("--decoders", default="zxing,opencv_qr")
    parser.add_argument("--image-variants", default="enhanced,tight_enhanced")
    parser.add_argument("--decoder-variants", default="tight,tight_enhanced,tight_binary,enhanced,binary,raw")
    parser.add_argument("--zones", default="")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--parallel-engine-mode", choices=["shared", "thread_local"], default="shared")
    parser.add_argument("--fast-no-decoders", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--paddle-cache", default=os.environ.get("PADDLE_CACHE_DIR", str(Path("runtime") / "cache" / "paddle")))
    parser.add_argument("--aggregate-only-candidates", type=Path)
    parser.add_argument("--aggregate-only-raw", type=Path)
    parser.add_argument("--cascade-early-stop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_rows(args.zones_manifest)
    manifest_rows = unique_rows_by_track_rank_zone(manifest_rows, top_k=args.top_k)

    if args.max_tracks:
        allowed_tracks = sorted({(row["video_id"], int(row["track_id"])) for row in manifest_rows})[: args.max_tracks]
        allowed_keys = {(video_id, str(track_id)) for video_id, track_id in allowed_tracks}
        manifest_rows = [row for row in manifest_rows if (row["video_id"], row["track_id"]) in allowed_keys]

    selected_zones = {zone.strip() for zone in args.zones.split(",") if zone.strip()}
    if selected_zones:
        manifest_rows = [row for row in manifest_rows if row["zone"] in selected_zones]

    raw_rows: list[dict[str, Any]] = []
    all_candidates: list[FieldCandidate] = []
    if args.aggregate_only_candidates:
        all_candidates = [candidate_from_row(row) for row in read_rows(args.aggregate_only_candidates)]
        raw_rows = read_rows(args.aggregate_only_raw) if args.aggregate_only_raw and args.aggregate_only_raw.exists() else []
    else:
        engine_names = [name.strip() for name in args.engines.split(",") if name.strip()]
        decoder_names = [name.strip() for name in args.decoders.split(",") if name.strip()]
        if args.fast_no_decoders:
            decoder_names = []
        image_variants = [name.strip() for name in args.image_variants.split(",") if name.strip()]
        decoder_variants = [name.strip() for name in args.decoder_variants.split(",") if name.strip()]
        engine_plan = parse_engine_plan(args.engine_plan)

        satisfied_fields: set[tuple[str, str, str]] = set()
        cascade_skipped_rows = 0
        cascade_executed_rows = 0
        recognition_calls = 0
        decoder_calls = 0
        jobs = max(1, int(args.jobs))
        configure_parallel_cpu_threads(jobs)
        if jobs > 1 and args.cascade_early_stop:
            raise ValueError("--jobs > 1 is not compatible with --cascade-early-stop")
        if jobs > 1:
            raw_rows, all_candidates, stats = run_rows_parallel(
                manifest_rows,
                engine_names,
                decoder_names,
                image_variants,
                decoder_variants,
                engine_plan,
                gpu=args.gpu,
                paddle_cache=args.paddle_cache,
                jobs=jobs,
                parallel_engine_mode=args.parallel_engine_mode,
            )
            cascade_executed_rows = stats["cascade_executed_rows"]
            cascade_skipped_rows = stats["cascade_skipped_rows"]
            recognition_calls = stats["recognition_calls"]
            decoder_calls = stats["decoder_calls"]
        else:
            required_engine_names = required_engines_for_rows(manifest_rows, engine_names, engine_plan)
            engines = load_engines(required_engine_names, gpu=args.gpu, paddle_cache=args.paddle_cache)

            total_rows = len(manifest_rows)
            for index, row in enumerate(manifest_rows, start=1):
                zone = row["zone"]
                row_target_keys = [cascade_key(row, field) for field in target_fields_for_row(row)]
                if args.cascade_early_stop and row_target_keys and all(key in satisfied_fields for key in row_target_keys):
                    cascade_skipped_rows += 1
                    continue
                row_candidates: list[FieldCandidate] = []
                row_ran = False
                if zone in OCR_TEXT_ZONES:
                    row_engines = engines_for_row(row, engine_names, engine_plan)
                    for image_kind, image_path in row_image_variants(row, image_variants):
                        image = read_image_bgr(image_path)
                        for engine_name in row_engines:
                            row_ran = True
                            recognition_calls += 1
                            engine = engines[engine_name]
                            try:
                                tokens, elapsed_ms = engine.recognize(image)
                            except Exception as exc:  # noqa: BLE001
                                tokens = [OCRToken(f"ERROR: {exc}", 0.0, [])]
                                elapsed_ms = 0.0
                            result = build_ocr_result(engine_name, image_kind, image_path, tokens, elapsed_ms)
                            raw_rows.append(result_to_row(row, result))
                            if not result.text.startswith("ERROR:"):
                                candidates = candidates_from_result(row, result)
                                all_candidates.extend(candidates)
                                row_candidates.extend(candidates)
                if zone in DECODER_ZONES and decoder_names:
                    for image_kind, image_path in row_image_variants(row, decoder_variants):
                        image = read_image_bgr(image_path)
                        if "zxing" in decoder_names:
                            row_ran = True
                            decoder_calls += 1
                            for payload in decode_with_zxing(image):
                                result = OCRResult("zxing", image_kind, image_path, payload, 1.0, 0.0, [])
                                raw_rows.append(result_to_row(row, result))
                                candidates = candidates_from_decoder(row, "zxing", image_kind, image_path, payload)
                                all_candidates.extend(candidates)
                                row_candidates.extend(candidates)
                        if zone == "qr" and "opencv_qr" in decoder_names:
                            row_ran = True
                            decoder_calls += 1
                            for payload in decode_qr_with_opencv(image):
                                result = OCRResult("opencv_qr", image_kind, image_path, payload, 1.0, 0.0, [])
                                raw_rows.append(result_to_row(row, result))
                                candidates = candidates_from_decoder(row, "opencv_qr", image_kind, image_path, payload)
                                all_candidates.extend(candidates)
                                row_candidates.extend(candidates)
                if row_ran:
                    cascade_executed_rows += 1
                if args.cascade_early_stop:
                    for candidate in row_candidates:
                        if candidate_satisfies_early_stop(candidate):
                            satisfied_fields.add(cascade_key(row, candidate.field))
                if index % 100 == 0:
                    print(f"Processed {index}/{total_rows} zone rows")

    if args.submission_template and args.submission_template.exists():
        template_rows = read_rows(args.submission_template)
        template_rows = enrich_template_track_ids(template_rows, manifest_rows)
        if not any(row.get("track_id", "") for row in template_rows):
            template_rows = build_template_rows(manifest_rows)
    else:
        template_rows = build_template_rows(manifest_rows)
    allowed_template_keys = {(row["video_id"], row["track_id"]) for row in manifest_rows}
    template_rows = [
        row
        for row in template_rows
        if (Path(row.get("filename", "")).stem, row.get("track_id", "")) in allowed_template_keys
    ]

    output_rows, debug_rows = aggregate_submission(all_candidates, template_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "ocr_raw_results.csv", raw_rows)
    write_rows(args.output / "ocr_field_candidates.csv", [candidate_to_row(item) for item in all_candidates])
    write_rows(args.output / "ocr_aggregated_debug.csv", debug_rows)
    write_rows(args.output / "ocr_aggregated_submission.csv", output_rows, fieldnames=OUTPUT_COLUMNS)

    summary = summarize(all_candidates, output_rows, raw_rows)
    if not args.aggregate_only_candidates:
        summary.update(
            {
                "cascade_early_stop": bool(args.cascade_early_stop),
                "cascade_executed_rows": cascade_executed_rows,
                "cascade_skipped_rows": cascade_skipped_rows,
                "recognition_calls": recognition_calls,
                "decoder_calls": decoder_calls,
                "jobs": max(1, int(args.jobs)),
                "parallel_engine_mode": args.parallel_engine_mode,
                "fast_no_decoders": bool(args.fast_no_decoders),
            }
        )
    with (args.output / "ocr_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
