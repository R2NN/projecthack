"""Run Tesseract 5 OCR on product_name crops as an isolated experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PRODUCT_FIELD = "product_name"
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


@dataclass(frozen=True)
class ProductNameCandidate:
    video_id: str
    filename: str
    track_id: str
    rank: int
    timestamp_ms: str
    frame_timestamp: str
    value: str
    raw_text: str
    confidence: float
    score: float
    source: str
    image_path: str
    line_count: int
    elapsed_ms: float
    return_code: int
    stderr: str


@dataclass(frozen=True)
class TesseractTask:
    task_id: int
    row: dict[str, str]
    image_path: str
    psm: int
    source: str
    line_count: int
    line_group_key: str = ""
    line_index: int = 0


@dataclass(frozen=True)
class TesseractTaskResult:
    task: TesseractTask
    text: str
    confidence: float
    elapsed_ms: float
    return_code: int
    stderr: str


@dataclass(frozen=True)
class CandidateEvent:
    kind: str
    task_id: int = -1
    line_group_key: str = ""


@dataclass(frozen=True)
class LineGroup:
    row: dict[str, str]
    source: str
    image_path: str


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\x0c", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_cyrillic(char: str) -> bool:
    return "\u0400" <= char <= "\u04ff"


def normalize_surface_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value))
    text = text.replace("|", " ").replace("«", '"').replace("»", '"')
    text = text.replace("{", "(").replace("[", "(").replace("}", ")").replace("]", ")")
    text = re.sub(r"\b(\d{2,4})\s*[rR]\b", lambda match: f"{match.group(1)}\u0433", text)
    text = re.sub(r"\b(\d{1,3})\s*%\b", r"\1%", text)
    text = re.sub(r"\s+([),.%])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return clean_text(text)


def guard_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", normalize_surface_text(value)).lower().replace("\u0451", "\u0435")
    latin_to_cyr = str.maketrans(
        {
            "a": "\u0430",
            "c": "\u0441",
            "e": "\u0435",
            "k": "\u043a",
            "m": "\u043c",
            "o": "\u043e",
            "p": "\u0440",
            "r": "\u0433",
            "t": "\u0442",
            "x": "\u0445",
            "y": "\u0443",
        }
    )
    text = text.translate(latin_to_cyr)
    return re.sub(r"[^0-9a-z\u0400-\u04ff]+", "", text)


def char_ngrams(value: Any, n: int = 3) -> set[str]:
    key = guard_key(value)
    if len(key) <= n:
        return {key} if key else set()
    return {key[index : index + n] for index in range(len(key) - n + 1)}


def digit_groups(value: Any) -> set[str]:
    normalized = str(value or "").translate(str.maketrans({"O": "0", "o": "0", "\u041e": "0", "\u043e": "0", "S": "5", "s": "5"}))
    return set(re.findall(r"\d{2,}", normalized))


def has_repeated_noise(value: Any) -> bool:
    tokens = re.findall(r"[A-Za-z\u0400-\u04ff]{3,}", normalize_surface_text(value).lower())
    if len(tokens) < 5:
        return False
    duplicates = len(tokens) - len(set(tokens))
    return duplicates >= 2


def quality_score(text: str, confidence: float, source: str, rank: int, line_count: int) -> float:
    text = normalize_surface_text(text)
    letters = [char for char in text if char.isalpha()]
    cyrillic = [char for char in letters if is_cyrillic(char)]
    latin = [char for char in letters if "A" <= char.upper() <= "Z"]
    weird_symbols = len(re.findall(r"[^\w\s()/.%,'\"\-]", text, flags=re.UNICODE))
    digit_letter_mixes = len(re.findall(r"[A-Za-z\u0400-\u04ff]\d|\d[A-Za-z\u0400-\u04ff]", text))
    short_noise_tokens = len(re.findall(r"\b[A-Za-z\u0400-\u04ff]{1,2}\b", text))

    score = confidence
    score += min(len(text), 80) / 155.0
    if letters:
        score += len(cyrillic) / max(1, len(letters)) * 0.30
    if re.search(r"\b\d{2,4}\s*(?:\u0433|\u0433\u0440|\u043c\u043b|\u043b|\u043a\u0433|g|gr|ml|l|kg)\b", text, flags=re.IGNORECASE):
        score += 0.20
    if line_count >= 2:
        score += 0.08
    if source.startswith("line"):
        score += 0.04
    if "binary" in source:
        score += 0.02
    score += max(0.0, 0.10 - (rank - 1) * 0.035)
    score -= weird_symbols * 0.09
    score -= digit_letter_mixes * 0.15
    score -= short_noise_tokens * 0.025
    if latin and cyrillic and len(latin) / max(1, len(letters)) > 0.55:
        score -= 0.25
    if len(text) < 5:
        score -= 0.80
    if has_repeated_noise(text):
        score -= 0.35
    return score


def safe_replacement(before: str, after: str) -> tuple[bool, str]:
    before = normalize_surface_text(before)
    after = normalize_surface_text(after)
    if not after:
        return False, "empty"
    if not before:
        return True, "new_from_ocr"
    if before == after:
        return True, "normalized_only"
    if has_repeated_noise(after):
        return False, "repeated_noise"
    if len(before) >= 8 and len(after) > max(22, int(len(before) * 1.45)):
        return False, "too_much_longer"

    before_digits = digit_groups(before)
    after_digits = digit_groups(after)
    if after_digits and not before_digits:
        return False, "new_digits"
    if before_digits and not after_digits.issubset(before_digits):
        return False, "changed_digits"

    before_grams = char_ngrams(before)
    after_grams = char_ngrams(after)
    if before_grams and after_grams:
        candidate_overlap = len(after_grams & before_grams) / max(1, len(after_grams))
        source_overlap = len(after_grams & before_grams) / max(1, len(before_grams))
        if candidate_overlap < 0.48 or source_overlap < 0.30:
            return False, "low_overlap"
    before_score = quality_score(before, 0.70, "base", 1, 1)
    after_score = quality_score(after, 0.70, "candidate", 1, 1)
    if after_score < before_score:
        return False, "no_quality_gain"
    return True, "ok"


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


def row_image_variants(row: dict[str, str], variants: list[str]) -> list[tuple[str, str]]:
    keys = {
        "raw": "zone_raw",
        "enhanced": "zone_enhanced",
        "binary": "zone_binary",
        "tight": "zone_tight",
        "tight_enhanced": "zone_tight_enhanced",
        "tight_binary": "zone_tight_binary",
    }
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for variant in variants:
        path = row.get(keys.get(variant, ""), "").strip()
        if path and path not in seen and Path(path).exists():
            found.append((variant, path))
            seen.add(path)
    return found


def read_image(path: str) -> np.ndarray | None:
    if not path:
        return None
    image_bytes = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)


def save_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    ext = ".png" if suffix == ".png" else ".jpg"
    params = [] if ext == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 96]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())
    return str(path)


def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    scale = 1
    if "up3" in mode:
        scale = 3
    elif "up2" in mode:
        scale = 2
    work = image
    if scale > 1:
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if "clahe" in mode or "binary" in mode:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if "binary" in mode:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return binary
    if "gray" in mode or "clahe" in mode:
        return gray
    return work


def add_white_border(image: np.ndarray, border_px: int) -> np.ndarray:
    if border_px <= 0:
        return image
    if image.ndim == 2:
        value: int | list[int] = 255
    else:
        value = [255, 255, 255]
    return cv2.copyMakeBorder(
        image,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=value,
    )


def split_line_crops(image: np.ndarray, max_lines: int) -> list[np.ndarray]:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        11,
    )
    height, width = image.shape[:2]
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    text_mask = np.zeros_like(binary)
    for index in range(1, component_count):
        x, y, component_w, component_h, area = [int(value) for value in stats[index]]
        if area < 25:
            continue
        if x < 0.08 * width and component_w < 0.12 * width:
            continue
        border_like = (
            (component_w > 0.50 * width and component_h < 0.12 * height)
            or (component_h > 0.42 * height and component_w < 0.12 * width)
            or (y < 3 and component_w > 0.20 * width)
        )
        if border_like:
            continue
        if component_h < max(10, int(0.025 * height)) or component_h > 0.24 * height:
            continue
        if component_w < 3 or area / max(1, component_w * component_h) < 0.08:
            continue
        text_mask[labels == index] = 255

    projection = text_mask.sum(axis=1) / 255.0
    if projection.max(initial=0.0) <= 0:
        return []
    window = max(9, height // 80)
    smoothed = np.convolve(projection, np.ones(window) / window, mode="same")
    threshold = max(8.0, float(smoothed.max()) * 0.06)
    active = smoothed > threshold

    base_intervals: list[tuple[int, int]] = []
    start_y: int | None = None
    for y, is_active in enumerate(active):
        if is_active and start_y is None:
            start_y = y
        elif not is_active and start_y is not None:
            base_intervals.append((start_y, y))
            start_y = None
    if start_y is not None:
        base_intervals.append((start_y, len(active) - 1))

    min_line_height = max(32, int(height * 0.045))
    max_line_height = max(70, int(height * 0.19))

    def split_interval(y1: int, y2: int) -> list[tuple[int, int]]:
        if y2 - y1 <= max_line_height:
            return [(y1, y2)]
        pad = max(min_line_height, int((y2 - y1) * 0.12))
        lo = y1 + pad
        hi = y2 - pad
        if hi <= lo:
            return [(y1, y2)]
        valley_y = int(lo + np.argmin(smoothed[lo:hi]))
        valley = float(smoothed[valley_y])
        left_peak = float(smoothed[y1:valley_y].max(initial=0.0))
        right_peak = float(smoothed[valley_y:y2].max(initial=0.0))
        if (
            min(valley_y - y1, y2 - valley_y) >= min_line_height
            and valley < min(left_peak, right_peak) * 0.55
        ):
            return split_interval(y1, valley_y) + split_interval(valley_y, y2)
        return [(y1, y2)]

    intervals: list[tuple[int, int]] = []
    for y1, y2 in base_intervals:
        if y2 - y1 < min_line_height:
            continue
        intervals.extend(split_interval(y1, y2))
    intervals = [(y1, y2) for y1, y2 in intervals if y2 - y1 >= min_line_height]
    if len(intervals) <= 1:
        return []

    crops: list[np.ndarray] = []
    for y1, y2 in intervals[:max_lines]:
        pad_y = max(4, int(round((y2 - y1) * 0.22)))
        y1 = max(0, y1 - pad_y)
        y2 = min(height, y2 + pad_y)
        row_mask = text_mask[y1:y2, :]
        column_projection = row_mask.sum(axis=0) / 255.0
        xs = np.where(column_projection > 1)[0]
        if xs.size:
            x1 = max(0, int(xs.min()) - 14)
            x2 = min(width, int(xs.max()) + 15)
        else:
            x1, x2 = 0, width
        if x2 - x1 < 20 or y2 - y1 < 8:
            continue
        crops.append(image[y1:y2, x1:x2])
    return crops


def parse_tesseract_tsv(tsv_text: str) -> tuple[str, float]:
    if not tsv_text.strip():
        return "", 0.0
    lines: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    confidences: list[float] = []
    for raw_line in tsv_text.splitlines()[1:]:
        # Tesseract TSV is not always valid CSV: OCR text may contain a bare quote.
        # Splitting into the first 11 tab-separated columns keeps the text column intact.
        parts = raw_line.split("\t", 11)
        if len(parts) < 12 or parts[0] != "5":
            continue
        text = clean_text(parts[11])
        if not text:
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            conf = -1.0
        if 0 <= conf < 5 and len(guard_key(text)) <= 3:
            continue
        if conf >= 0:
            confidences.append(conf / 100.0)
        key = (as_int(parts[2]), as_int(parts[3]), as_int(parts[4]))
        lines[key].append(text)
    joined_lines = [" ".join(tokens) for _key, tokens in sorted(lines.items()) if tokens]
    text = normalize_surface_text(" ".join(joined_lines))
    confidence = sum(confidences) / max(1, len(confidences))
    return text, confidence


def run_tesseract(
    image_path: str,
    tesseract_exe: Path,
    tessdata_dir: Path,
    language: str,
    psm: int,
    timeout_seconds: int,
    omp_thread_limit: int,
) -> tuple[str, float, float, int, str]:
    command = [
        str(tesseract_exe),
        image_path,
        "stdout",
        "-l",
        language,
        "--oem",
        "1",
        "--psm",
        str(psm),
        "-c",
        "preserve_interword_spaces=1",
        "-c",
        "user_defined_dpi=300",
        "tsv",
    ]
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(tessdata_dir)
    if omp_thread_limit > 0:
        env["OMP_THREAD_LIMIT"] = str(omp_thread_limit)
    start = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout_seconds,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    text, confidence = parse_tesseract_tsv(result.stdout)
    return text, confidence, elapsed_ms, result.returncode, clean_text(result.stderr)


def run_tesseract_task(
    task: TesseractTask,
    tesseract_exe: Path,
    tessdata_dir: Path,
    language: str,
    timeout_seconds: int,
    omp_thread_limit: int,
) -> TesseractTaskResult:
    text, confidence, elapsed_ms, return_code, stderr = run_tesseract(
        task.image_path,
        tesseract_exe,
        tessdata_dir,
        language,
        task.psm,
        timeout_seconds,
        omp_thread_limit,
    )
    return TesseractTaskResult(
        task=task,
        text=text,
        confidence=confidence,
        elapsed_ms=elapsed_ms,
        return_code=return_code,
        stderr=stderr,
    )


def candidate_to_row(candidate: ProductNameCandidate) -> dict[str, Any]:
    return {
        "video_id": candidate.video_id,
        "filename": candidate.filename,
        "track_id": candidate.track_id,
        "rank": candidate.rank,
        "timestamp_ms": candidate.timestamp_ms,
        "frame_timestamp": candidate.frame_timestamp,
        "field": PRODUCT_FIELD,
        "value": candidate.value,
        "raw_text": candidate.raw_text,
        "confidence": f"{candidate.confidence:.4f}",
        "score": f"{candidate.score:.4f}",
        "engine": "tesseract5_tessdata_best",
        "source": candidate.source,
        "image_path": candidate.image_path,
        "line_count": candidate.line_count,
        "elapsed_ms": f"{candidate.elapsed_ms:.1f}",
        "return_code": candidate.return_code,
        "stderr": candidate.stderr,
    }


def choose_best(candidates: list[ProductNameCandidate]) -> ProductNameCandidate | None:
    if not candidates:
        return None
    grouped: dict[str, list[ProductNameCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.value].append(candidate)

    def value_score(value: str, items: list[ProductNameCandidate]) -> float:
        return max(item.score for item in items) + min(len(items), 5) * 0.045

    best_value = max(grouped, key=lambda value: value_score(value, grouped[value]))
    return max(grouped[best_value], key=lambda item: item.score)


def candidate_is_plausible(candidate: ProductNameCandidate) -> bool:
    text = normalize_surface_text(candidate.value)
    key = guard_key(text)
    if len(key) < 5:
        return False
    letters = [char for char in text if char.isalpha()]
    cyrillic = [char for char in letters if is_cyrillic(char)]
    weird_symbols = len(re.findall(r"[^\w\s()/.%,'\"\-]", text, flags=re.UNICODE))
    if len(letters) < 4:
        return False
    if letters and len(cyrillic) / max(1, len(letters)) < 0.25:
        return False
    if weird_symbols >= 6:
        return False
    if has_repeated_noise(text):
        return False
    if candidate.source.startswith("lines_"):
        if candidate.line_count < 2:
            return False
        if candidate.confidence < 0.55 or candidate.score < 1.18:
            return False
        if weird_symbols >= 4:
            return False
    return candidate.confidence >= 0.25 or candidate.score >= 0.85


def choose_best_line_split(candidates: list[ProductNameCandidate]) -> ProductNameCandidate | None:
    if not candidates:
        return None
    tiers = [
        lambda item: item.source.startswith("lines_") and item.line_count >= 2 and candidate_is_plausible(item),
        lambda item: item.source.startswith("full_") and "_psm6" in item.source and candidate_is_plausible(item),
        lambda item: item.source.startswith("full_") and "_psm11" in item.source and candidate_is_plausible(item),
        lambda item: candidate_is_plausible(item),
        lambda _item: True,
    ]
    for selector in tiers:
        selected = [item for item in candidates if selector(item)]
        if selected:
            return choose_best(selected)
    return None


def choose_best_for_policy(candidates: list[ProductNameCandidate], policy: str) -> ProductNameCandidate | None:
    if policy == "line_split":
        return choose_best_line_split(candidates)
    return choose_best(candidates)


def add_candidate(
    candidates: list[ProductNameCandidate],
    row: dict[str, str],
    text: str,
    confidence: float,
    elapsed_ms: float,
    return_code: int,
    stderr: str,
    source: str,
    image_path: str,
    line_count: int,
) -> None:
    normalized = normalize_surface_text(text)
    if not normalized:
        return
    rank = as_int(row.get("rank"), 999)
    candidates.append(
        ProductNameCandidate(
            video_id=row.get("video_id", ""),
            filename=row.get("filename", ""),
            track_id=row.get("track_id", ""),
            rank=rank,
            timestamp_ms=row.get("timestamp_ms", ""),
            frame_timestamp=row.get("frame_timestamp", ""),
            value=normalized,
            raw_text=text,
            confidence=confidence,
            score=quality_score(normalized, confidence, source, rank, line_count),
            source=source,
            image_path=image_path,
            line_count=line_count,
            elapsed_ms=elapsed_ms,
            return_code=return_code,
            stderr=stderr,
        )
    )


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    variants = comma_list(args.image_variants)
    preprocess_modes = comma_list(args.preprocess_modes)
    psms = [as_int(item) for item in comma_list(args.psms)]
    line_variants = set(comma_list(args.line_variants))
    line_psms = [as_int(item) for item in comma_list(args.line_psms)]

    manifest_rows = [
        row
        for row in read_rows(args.zones_manifest)
        if row.get("zone") == PRODUCT_FIELD and as_int(row.get("rank"), 999) <= args.top_k
    ]
    manifest_rows.sort(key=lambda row: (row.get("video_id", ""), as_int(row.get("track_id")), as_int(row.get("rank"))))
    if args.limit_rows > 0:
        manifest_rows = manifest_rows[: args.limit_rows]

    prep_dir = args.output / "product_name_tesseract_preprocessed"
    line_dir = args.output / "product_name_tesseract_line_crops"
    all_candidates: list[ProductNameCandidate] = []
    started = time.perf_counter()
    tasks: list[TesseractTask] = []
    events: list[CandidateEvent] = []
    line_groups: dict[str, LineGroup] = {}

    for index, row in enumerate(manifest_rows, start=1):
        for variant, source_path in row_image_variants(row, variants):
            image = read_image(source_path)
            if image is None:
                continue
            prepared_paths: list[tuple[str, str]] = []
            for mode in preprocess_modes:
                if mode == "none":
                    prepared_image = image
                else:
                    prepared_image = preprocess_image(image, mode)
                full_image = add_white_border(prepared_image, args.full_border_px)
                if mode == "none" and args.full_border_px <= 0:
                    prepared_paths.append((mode, source_path))
                else:
                    prep_path = prep_dir / (
                        f"track_{as_int(row.get('track_id')):04d}_rank_{as_int(row.get('rank')):02d}_{variant}_{mode}.png"
                    )
                    prepared_paths.append((mode, save_image(prep_path, full_image)))
                for psm in psms:
                    task = TesseractTask(
                        task_id=len(tasks),
                        row=row,
                        image_path=prepared_paths[-1][1],
                        psm=psm,
                        source=f"full_{variant}_{mode}_psm{psm}",
                        line_count=1,
                    )
                    tasks.append(task)
                    events.append(CandidateEvent(kind="full", task_id=task.task_id))

                if variant in line_variants and mode == args.line_preprocess_mode:
                    line_paths: list[str] = []
                    for line_index, line_crop in enumerate(split_line_crops(prepared_image, args.max_lines), start=1):
                        line_crop = add_white_border(line_crop, args.line_border_px)
                        line_path = line_dir / (
                            f"track_{as_int(row.get('track_id')):04d}_rank_{as_int(row.get('rank')):02d}_{variant}_{mode}_line_{line_index:02d}.png"
                        )
                        saved_line = save_image(line_path, line_crop)
                        line_paths.append(saved_line)
                        for psm in line_psms:
                            group_key = f"{index}:{variant}:{mode}:{psm}"
                            task = TesseractTask(
                                task_id=len(tasks),
                                row=row,
                                image_path=saved_line,
                                psm=psm,
                                source=f"line_{variant}_{mode}_psm{psm}",
                                line_count=1,
                                line_group_key=group_key,
                                line_index=line_index,
                            )
                            tasks.append(task)
                    for psm in line_psms:
                        group_key = f"{index}:{variant}:{mode}:{psm}"
                        line_groups[group_key] = LineGroup(
                            row=row,
                            source=f"lines_{variant}_{mode}_psm{psm}",
                            image_path=line_paths[0] if line_paths else source_path,
                        )
                        events.append(CandidateEvent(kind="lines", line_group_key=group_key))
        if index % 20 == 0:
            print(f"Prepared {index}/{len(manifest_rows)} product_name zone rows")

    print(f"Running {len(tasks)} Tesseract calls with jobs={max(1, args.jobs)} and OMP_THREAD_LIMIT={args.tesseract_omp_thread_limit}")
    result_by_task: dict[int, TesseractTaskResult] = {}
    if args.jobs <= 1:
        for index, task in enumerate(tasks, start=1):
            result = run_tesseract_task(
                task,
                args.tesseract_exe,
                args.tessdata_dir,
                args.language,
                args.timeout_seconds,
                args.tesseract_omp_thread_limit,
            )
            result_by_task[task.task_id] = result
            if index % 100 == 0:
                print(f"Ran {index}/{len(tasks)} Tesseract calls")
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    run_tesseract_task,
                    task,
                    args.tesseract_exe,
                    args.tessdata_dir,
                    args.language,
                    args.timeout_seconds,
                    args.tesseract_omp_thread_limit,
                )
                for task in tasks
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                result_by_task[result.task.task_id] = result
                if index % 100 == 0:
                    print(f"Ran {index}/{len(tasks)} Tesseract calls")

    tesseract_calls = len(tasks)
    failed_calls = sum(1 for result in result_by_task.values() if result.return_code not in {0})
    line_results_by_group: dict[str, list[TesseractTaskResult]] = defaultdict(list)
    for result in result_by_task.values():
        if result.task.line_group_key:
            line_results_by_group[result.task.line_group_key].append(result)

    for event in events:
        if event.kind == "full":
            result = result_by_task[event.task_id]
            add_candidate(
                all_candidates,
                result.task.row,
                result.text,
                result.confidence,
                result.elapsed_ms,
                result.return_code,
                result.stderr,
                result.task.source,
                result.task.image_path,
                result.task.line_count,
            )
            continue
        if event.kind != "lines":
            continue
        group = line_groups.get(event.line_group_key)
        if group is None:
            continue
        line_results = sorted(line_results_by_group.get(event.line_group_key, []), key=lambda item: item.task.line_index)
        line_texts: list[str] = []
        line_scores: list[float] = []
        elapsed_ms = 0.0
        for result in line_results:
            cleaned = normalize_surface_text(result.text)
            if cleaned:
                line_texts.append(cleaned)
                line_scores.append(result.confidence)
            elapsed_ms += result.elapsed_ms
        joined = normalize_surface_text(" ".join(line_texts))
        if not joined:
            continue
        add_candidate(
            all_candidates,
            group.row,
            joined,
            sum(line_scores) / max(1, len(line_scores)),
            elapsed_ms,
            0,
            "",
            group.source,
            group.image_path,
            len(line_texts),
        )

    grouped: dict[str, list[ProductNameCandidate]] = defaultdict(list)
    for candidate in all_candidates:
        grouped[candidate.track_id].append(candidate)
    best_by_track: dict[str, ProductNameCandidate] = {}
    for track_id, candidates in grouped.items():
        best = choose_best_for_policy(candidates, args.selection_policy)
        if best:
            best_by_track[track_id] = best

    submission_rows = read_rows(args.base_submission_csv)
    debug_rows = read_rows(args.base_debug_csv)
    track_lookup = build_track_lookup(read_rows(args.zones_manifest))
    changes: list[dict[str, Any]] = []
    for row in submission_rows:
        track_id = row.get("track_id") or row.get("track_id_export") or track_lookup.get(row_key(row), "")
        best = best_by_track.get(track_id)
        before = row.get(PRODUCT_FIELD, "")
        normalized_before = normalize_surface_text(before)
        proposed = best.value if best else normalized_before
        if not best:
            accepted = False
            accept_reason = "no_candidate"
        elif args.accept_policy == "guard":
            accepted, accept_reason = safe_replacement(normalized_before, proposed)
        else:
            accepted = bool(proposed)
            accept_reason = "ocr_output" if accepted else "empty"
        final_value = proposed if accepted else normalized_before
        row[PRODUCT_FIELD] = final_value
        changes.append(
            {
                "track_id": track_id,
                "before": before,
                "after": final_value,
                "proposed": proposed,
                "accepted": int(accepted),
                "reason": accept_reason,
                "changed": int(before != final_value),
                "score": f"{best.score:.4f}" if best else "",
                "confidence": f"{best.confidence:.4f}" if best else "",
                "source": best.source if best else "base_normalized",
                "line_count": best.line_count if best else 0,
                "raw_text": best.raw_text if best else before,
            }
        )

    final_by_track = {row["track_id"]: row for row in changes if row.get("track_id")}
    debug_by_track = {row.get("track_id", ""): row for row in debug_rows if row.get("field") == PRODUCT_FIELD}
    for track_id, change in final_by_track.items():
        row = debug_by_track.get(track_id)
        best = best_by_track.get(track_id)
        if row is None:
            if best:
                debug_rows.append(candidate_to_row(best))
            continue
        row["value"] = change["after"]
        row["score"] = change.get("score", row.get("score", ""))
        row["confidence"] = change.get("confidence", row.get("confidence", ""))
        row["engine"] = "tesseract5_tessdata_best" if change["accepted"] else "tesseract5_guarded"
        row["zone"] = PRODUCT_FIELD
        if best:
            row["image_kind"] = best.source
            row["image_path"] = best.image_path
            row["source_text"] = best.raw_text

    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "product_name_line_candidates.csv", [candidate_to_row(item) for item in all_candidates])
    write_rows(args.output / "product_name_line_best.csv", [candidate_to_row(item) for item in best_by_track.values()])
    write_rows(args.output / "product_name_line_changes.csv", changes)
    write_rows(args.output / "ocr_aggregated_submission_product_lines.csv", submission_rows, fieldnames=OUTPUT_COLUMNS)
    write_rows(args.output / "ocr_aggregated_debug_product_lines.csv", debug_rows)

    elapsed_total = time.perf_counter() - started
    summary = {
        "engine": "tesseract5_tessdata_best",
        "tesseract_exe": str(args.tesseract_exe),
        "tessdata_dir": str(args.tessdata_dir),
        "language": args.language,
        "tracks": len(best_by_track),
        "zone_rows": len(manifest_rows),
        "candidates": len(all_candidates),
        "changed_product_names": sum(1 for row in changes if row["changed"]),
        "accepted_proposals": sum(1 for row in changes if row["accepted"]),
        "elapsed_seconds": elapsed_total,
        "avg_seconds_per_zone_row": elapsed_total / max(1, len(manifest_rows)),
        "tesseract_calls": tesseract_calls,
        "failed_calls": failed_calls,
        "jobs": max(1, args.jobs),
        "tesseract_omp_thread_limit": args.tesseract_omp_thread_limit,
        "top_k": args.top_k,
        "image_variants": variants,
        "preprocess_modes": preprocess_modes,
        "psms": psms,
        "line_variants": sorted(line_variants),
        "line_preprocess_mode": args.line_preprocess_mode,
        "line_psms": line_psms,
        "selection_policy": args.selection_policy,
        "full_border_px": args.full_border_px,
        "line_border_px": args.line_border_px,
        "accept_policy": args.accept_policy,
    }
    with (args.output / "product_name_line_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones-manifest", type=Path, required=True)
    parser.add_argument("--base-submission-csv", type=Path, required=True)
    parser.add_argument("--base-debug-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tesseract-exe", type=Path, default=Path(os.environ.get("TESSERACT_EXE", "tesseract")))
    parser.add_argument("--tessdata-dir", type=Path, default=Path(os.environ.get("TESSDATA_DIR", "")))
    parser.add_argument("--language", default="rus+eng")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--image-variants", default="enhanced,raw")
    parser.add_argument("--preprocess-modes", default="none,up2,binary_up2")
    parser.add_argument("--psms", default="6,11")
    parser.add_argument("--line-variants", default="enhanced")
    parser.add_argument("--line-preprocess-mode", default="up2")
    parser.add_argument("--line-psms", default="7")
    parser.add_argument("--max-lines", type=int, default=5)
    parser.add_argument("--selection-policy", choices=["best", "line_split"], default="best")
    parser.add_argument("--full-border-px", type=int, default=0)
    parser.add_argument("--line-border-px", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--accept-policy", choices=["all", "guard"], default="all")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--tesseract-omp-thread-limit", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
