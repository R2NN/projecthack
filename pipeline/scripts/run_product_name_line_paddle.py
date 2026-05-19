"""Run GPU PaddleOCR on product_name full crops and line crops, then rerank."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import types
from collections import defaultdict
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


def install_modelscope_stub() -> None:
    """Avoid PaddleX importing real ModelScope, which imports torch and conflicts with Paddle CUDA DLLs."""
    ms = types.ModuleType("modelscope")

    def disabled_snapshot_download(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("ModelScope is disabled for local PaddleOCR runs")

    ms.snapshot_download = disabled_snapshot_download
    hub = types.ModuleType("modelscope.hub")
    errors = types.ModuleType("modelscope.hub.errors")

    class StubNotExistError(Exception):
        pass

    errors.NotExistError = StubNotExistError
    hub.errors = errors
    ms.hub = hub
    sys.modules["modelscope"] = ms
    sys.modules["modelscope.hub"] = hub
    sys.modules["modelscope.hub.errors"] = errors


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


def clean_text(value: str) -> str:
    value = str(value or "").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", value).strip()


def read_image_bgr(path: str) -> np.ndarray | None:
    if not path:
        return None
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return image


def save_image(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())
    return str(path)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("filename", "").strip(),
        row.get("frame_timestamp", "").strip(),
        row.get("x_min", "").strip(),
        row.get("y_min", "").strip(),
        row.get("x_max", "").strip(),
        row.get("y_max", "").strip(),
    )


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


def normalize_surface_text(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"(?<=[а-яё])(?=[А-ЯЁ]{2,})", " ", text)
    text = re.sub(r"\b(\d{2,4})\s*[rR]\b", r"\1г", text)
    text = re.sub(r"\b([S5][O0Оо][O0Оо])\s*[rR]\b", "500г", text)
    text = re.sub(
        r"\b[PР][oо0][cс][cс][aаuиия]{1,4}\b",
        "Россия",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\(\s*Россия\s*\)", "(Россия)", text, flags=re.IGNORECASE)
    text = text.replace("{", "(").replace("[", "(").replace("}", ")").replace("]", ")")
    text = re.sub(r"\s+([),.])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return clean_text(text)


def guard_key(value: str) -> str:
    table = str.maketrans(
        {
            "a": "а",
            "c": "с",
            "d": "д",
            "e": "е",
            "h": "н",
            "k": "к",
            "m": "м",
            "o": "о",
            "p": "р",
            "r": "г",
            "t": "т",
            "x": "х",
            "y": "у",
        }
    )
    value = normalize_surface_text(value).lower().translate(table)
    return re.sub(r"[^a-zа-яё0-9]+", "", value)


def char_ngrams(value: str, n: int = 3) -> set[str]:
    value = guard_key(value)
    if len(value) <= n:
        return {value} if value else set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def digit_groups(value: str) -> set[str]:
    normalized = value.translate(str.maketrans({"O": "0", "o": "0", "О": "0", "о": "0", "S": "5", "s": "5"}))
    return set(re.findall(r"\d{2,}", normalized))


def protected_latin_brands(value: str) -> set[str]:
    protected = {"ZUEGG", "PREMIUM", "CLUB", "PERONI", "PERON"}
    return {token for token in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", value) if token in protected}


def guard_word(value: str) -> str:
    return guard_key(value)


def word_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", value)


def all_replacement_words_supported(before: str, after: str) -> bool:
    before_tokens = [guard_word(token) for token in word_tokens(before)]
    before_tokens = [token for token in before_tokens if token]
    if not before_tokens:
        return True
    for raw_token in word_tokens(after):
        token = guard_word(raw_token)
        if len(token) < 3:
            continue
        best = max((sequence_ratio(token, source) for source in before_tokens), default=0.0)
        if best < 0.72:
            return False
    return True


def sequence_ratio(left: str, right: str) -> float:
    matcher = __import__("difflib").SequenceMatcher(None, left, right)
    return float(matcher.ratio())


def has_repeated_noise(value: str) -> bool:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", normalize_surface_text(value).lower())
    if len(tokens) < 5:
        return False
    duplicates = len(tokens) - len(set(tokens))
    return duplicates >= 2


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
    if len(before) >= 8 and len(after) > max(18, int(len(before) * 1.35)):
        return False, "too_much_longer"

    before_digits = digit_groups(before)
    after_digits = digit_groups(after)
    if after_digits and not before_digits:
        return False, "new_digits"
    if before_digits and not after_digits.issubset(before_digits):
        return False, "changed_digits"

    missing_brands = protected_latin_brands(before) - protected_latin_brands(after)
    if missing_brands:
        return False, "dropped_brand"
    if not all_replacement_words_supported(before, after):
        return False, "new_words"

    before_grams = char_ngrams(before)
    after_grams = char_ngrams(after)
    if before_grams and after_grams:
        candidate_overlap = len(after_grams & before_grams) / max(1, len(after_grams))
        source_overlap = len(after_grams & before_grams) / max(1, len(before_grams))
        if candidate_overlap < 0.55 or source_overlap < 0.35:
            return False, "low_overlap"

    before_score = quality_score(before, 0.70, "base", 1, 1)
    after_score = quality_score(after, 0.70, "candidate", 1, 1)
    if after_score < before_score + 0.06:
        return False, "no_quality_gain"
    return True, "ok"


def quality_score(text: str, confidence: float, source: str, rank: int, line_count: int) -> float:
    text = normalize_surface_text(text)
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    cyrillic = re.findall(r"[А-Яа-яЁё]", text)
    latin = re.findall(r"[A-Za-z]", text)
    digits_inside_words = len(re.findall(r"[A-Za-zА-Яа-яЁё]\d|\d[A-Za-zА-Яа-яЁё]", text))
    mixed_words = len(re.findall(r"\b(?=[A-Za-zА-Яа-яЁё]*[A-Za-z])(?=[A-Za-zА-Яа-яЁё]*[А-Яа-яЁё])[A-Za-zА-Яа-яЁё]{4,}\b", text))
    weird_symbols = len(re.findall(r"[^\w\s()/.%-]", text, flags=re.UNICODE))

    score = confidence
    score += min(len(text), 70) / 140.0
    score += len(cyrillic) / max(1, len(letters)) * 0.45
    score += 0.25 if re.search(r"\b\d{2,4}\s*(?:г|мл|л|кг)\b", text, flags=re.IGNORECASE) else 0.0
    score += 0.18 if re.search(r"\b(?:Россия|Германия|Испания|Франция)\b", text, flags=re.IGNORECASE) else 0.0
    score += 0.16 if re.search(r"\b(?:Мед|ПАСЕКА|Конфитюр|Джем|Сироп|PREMIUM|ZUEGG)\b", text, flags=re.IGNORECASE) else 0.0
    score += 0.10 if source.startswith("lines") and line_count >= 2 else 0.0
    score += max(0.0, 0.12 - (rank - 1) * 0.04)
    score -= digits_inside_words * 0.28
    score -= mixed_words * 0.13
    score -= weird_symbols * 0.09
    if latin and cyrillic and len(latin) / max(1, len(letters)) > 0.45:
        score -= 0.35
    if len(text) < 5:
        score -= 0.70
    return score


def parse_paddle_result(result: list[Any]) -> tuple[str, float]:
    texts: list[str] = []
    scores: list[float] = []
    for page in result or []:
        if not isinstance(page, dict):
            continue
        page_texts = page.get("rec_texts", [])
        page_scores = page.get("rec_scores", [])
        for index, text in enumerate(page_texts):
            text = clean_text(str(text))
            if not text:
                continue
            texts.append(text)
            scores.append(float(page_scores[index]) if index < len(page_scores) else 0.0)
    return clean_text(" ".join(texts)), sum(scores) / max(1, len(scores))


def recognize(ocr: Any, image_bgr: np.ndarray) -> tuple[str, float, float]:
    start = time.perf_counter()
    result = ocr.predict(image_bgr)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    text, confidence = parse_paddle_result(result)
    return text, confidence, elapsed_ms


def split_line_crops(image_bgr: np.ndarray, max_lines: int) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel_w = max(9, image_bgr.shape[1] // 24)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    projection = connected.sum(axis=1) / 255.0
    threshold = max(3.0, image_bgr.shape[1] * 0.018)
    active = projection > threshold

    intervals: list[tuple[int, int]] = []
    start_y: int | None = None
    for y, is_active in enumerate(active):
        if is_active and start_y is None:
            start_y = y
        elif not is_active and start_y is not None:
            intervals.append((start_y, y))
            start_y = None
    if start_y is not None:
        intervals.append((start_y, len(active) - 1))

    merged: list[tuple[int, int]] = []
    for y1, y2 in intervals:
        if y2 - y1 < max(6, image_bgr.shape[0] * 0.035):
            continue
        if merged and y1 - merged[-1][1] <= max(4, image_bgr.shape[0] * 0.025):
            merged[-1] = (merged[-1][0], y2)
        else:
            merged.append((y1, y2))

    if len(merged) <= 1:
        return []

    height, width = image_bgr.shape[:2]
    crops: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for y1, y2 in merged[:max_lines]:
        pad_y = max(4, int(round((y2 - y1) * 0.22)))
        y1 = max(0, y1 - pad_y)
        y2 = min(height, y2 + pad_y)
        row_mask = connected[y1:y2, :]
        column_projection = row_mask.sum(axis=0) / 255.0
        xs = np.where(column_projection > 1)[0]
        if xs.size:
            x1 = max(0, int(xs.min()) - 10)
            x2 = min(width, int(xs.max()) + 11)
        else:
            x1, x2 = 0, width
        if x2 - x1 < 20 or y2 - y1 < 8:
            continue
        crops.append((image_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)))
    return crops


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
        "engine": "paddleocr_gpu_line",
        "source": candidate.source,
        "image_path": candidate.image_path,
        "line_count": candidate.line_count,
        "elapsed_ms": f"{candidate.elapsed_ms:.1f}",
    }


def choose_best(candidates: list[ProductNameCandidate]) -> ProductNameCandidate | None:
    if not candidates:
        return None
    grouped: dict[str, list[ProductNameCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.value].append(candidate)

    def value_score(value: str, items: list[ProductNameCandidate]) -> float:
        return max(item.score for item in items) + min(len(items), 4) * 0.06

    best_value = max(grouped, key=lambda value: value_score(value, grouped[value]))
    return max(grouped[best_value], key=lambda item: item.score)


def product_candidate_satisfies_early_stop(
    candidate: ProductNameCandidate,
    min_score: float,
    min_confidence: float,
) -> bool:
    text = normalize_surface_text(candidate.value)
    if len(text) < 8:
        return False
    if candidate.confidence < min_confidence or candidate.score < min_score:
        return False
    if has_repeated_noise(text):
        return False
    if sum(1 for char in text if char.isalpha()) < 5:
        return False
    return True


def build_track_lookup(manifest_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], str]:
    lookup: dict[tuple[str, str, str, str, str, str], str] = {}
    for row in manifest_rows:
        if row.get("zone") == PRODUCT_FIELD and row.get("rank") == "1":
            lookup[row_key(row)] = row.get("track_id", "")
    return lookup


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", args.paddle_cache)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    install_modelscope_stub()

    import paddle
    from paddleocr import PaddleOCR

    if args.gpu:
        paddle.set_device("gpu:0")
    device = paddle.device.get_device()
    if args.gpu and not device.startswith("gpu"):
        raise RuntimeError(f"PaddleOCR did not switch to GPU, current device is {device}")

    manifest_rows = [
        row
        for row in read_rows(args.zones_manifest)
        if row.get("zone") == PRODUCT_FIELD and int(row.get("rank", "999")) <= args.top_k
    ]
    manifest_rows.sort(key=lambda row: (row["video_id"], int(row["track_id"]), int(row["rank"])))
    variants = [item.strip() for item in args.image_variants.split(",") if item.strip()]

    ocr_kwargs: dict[str, Any] = {
        "lang": args.paddle_lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if args.text_recognition_model_name:
        ocr_kwargs["text_recognition_model_name"] = args.text_recognition_model_name
    if args.text_recognition_model_dir:
        ocr_kwargs["text_recognition_model_dir"] = args.text_recognition_model_dir
    ocr = PaddleOCR(**ocr_kwargs)

    line_dir = args.output / "product_name_line_crops"
    all_candidates: list[ProductNameCandidate] = []
    early_stopped_tracks: set[str] = set()
    cascade_skipped_rows = 0
    cascade_executed_rows = 0
    started = time.perf_counter()
    for index, row in enumerate(manifest_rows, start=1):
        track_id = row["track_id"]
        if args.cascade_early_stop and track_id in early_stopped_tracks:
            cascade_skipped_rows += 1
            continue
        row_start_index = len(all_candidates)
        row_ran = False
        for variant, image_path in row_image_variants(row, variants):
            image = read_image_bgr(image_path)
            if image is None:
                continue
            row_ran = True
            full_text, full_conf, full_elapsed = recognize(ocr, image)
            normalized = normalize_surface_text(full_text)
            if normalized:
                all_candidates.append(
                    ProductNameCandidate(
                        video_id=row["video_id"],
                        filename=row["filename"],
                        track_id=row["track_id"],
                        rank=int(row["rank"]),
                        timestamp_ms=row["timestamp_ms"],
                        frame_timestamp=row["frame_timestamp"],
                        value=normalized,
                        raw_text=full_text,
                        confidence=full_conf,
                        score=quality_score(normalized, full_conf, f"full_{variant}", int(row["rank"]), 1),
                        source=f"full_{variant}",
                        image_path=image_path,
                        line_count=1,
                        elapsed_ms=full_elapsed,
                    )
                )

            line_texts: list[str] = []
            line_scores: list[float] = []
            line_elapsed = 0.0
            line_paths: list[str] = []
            for line_index, (line_crop, _bbox) in enumerate(split_line_crops(image, args.max_lines), start=1):
                line_path = line_dir / f"track_{int(row['track_id']):04d}_rank_{int(row['rank']):02d}_{variant}_line_{line_index:02d}.jpg"
                line_paths.append(save_image(line_path, line_crop))
                text, conf, elapsed_ms = recognize(ocr, line_crop)
                line_elapsed += elapsed_ms
                if clean_text(text):
                    line_texts.append(clean_text(text))
                    line_scores.append(conf)
            line_joined = normalize_surface_text(" ".join(line_texts))
            if line_joined:
                line_conf = sum(line_scores) / max(1, len(line_scores))
                all_candidates.append(
                    ProductNameCandidate(
                        video_id=row["video_id"],
                        filename=row["filename"],
                        track_id=row["track_id"],
                        rank=int(row["rank"]),
                        timestamp_ms=row["timestamp_ms"],
                        frame_timestamp=row["frame_timestamp"],
                        value=line_joined,
                        raw_text=" | ".join(line_texts),
                        confidence=line_conf,
                        score=quality_score(line_joined, line_conf, f"lines_{variant}", int(row["rank"]), len(line_texts)),
                        source=f"lines_{variant}",
                        image_path=line_paths[0] if line_paths else image_path,
                        line_count=len(line_texts),
                        elapsed_ms=line_elapsed,
                    )
                )
        if row_ran:
            cascade_executed_rows += 1
        if args.cascade_early_stop:
            row_candidates = all_candidates[row_start_index:]
            best = choose_best(row_candidates)
            if best and product_candidate_satisfies_early_stop(
                best,
                min_score=args.early_stop_min_score,
                min_confidence=args.early_stop_min_confidence,
            ):
                early_stopped_tracks.add(track_id)
        if index % 25 == 0:
            print(f"Processed {index}/{len(manifest_rows)} product_name zone rows")

    best_by_track: dict[str, ProductNameCandidate] = {}
    grouped: dict[str, list[ProductNameCandidate]] = defaultdict(list)
    for candidate in all_candidates:
        grouped[candidate.track_id].append(candidate)
    for track_id, candidates in grouped.items():
        best = choose_best(candidates)
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
        accepted, accept_reason = safe_replacement(normalized_before, proposed)
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
        row["engine"] = "paddleocr_gpu_line" if change["accepted"] else "paddleocr_gpu_norm"
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
        "device": device,
        "tracks": len(best_by_track),
        "zone_rows": len(manifest_rows),
        "candidates": len(all_candidates),
        "changed_product_names": sum(1 for row in changes if row["changed"]),
        "accepted_proposals": sum(1 for row in changes if row["accepted"]),
        "elapsed_seconds": elapsed_total,
        "avg_seconds_per_zone_row": elapsed_total / max(1, len(manifest_rows)),
        "cascade_early_stop": bool(args.cascade_early_stop),
        "cascade_executed_rows": cascade_executed_rows,
        "cascade_skipped_rows": cascade_skipped_rows,
        "early_stopped_tracks": len(early_stopped_tracks),
        "paddle_lang": args.paddle_lang,
        "text_recognition_model_name": args.text_recognition_model_name or "auto",
        "text_recognition_model_dir": args.text_recognition_model_dir or "",
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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--image-variants", default="enhanced,raw")
    parser.add_argument("--max-lines", type=int, default=5)
    parser.add_argument("--paddle-cache", default="A:\\paddlex-cache")
    parser.add_argument("--paddle-lang", default="ru")
    parser.add_argument("--text-recognition-model-name", default="")
    parser.add_argument("--text-recognition-model-dir", default="")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cascade-early-stop", action="store_true")
    parser.add_argument("--early-stop-min-score", type=float, default=1.25)
    parser.add_argument("--early-stop-min-confidence", type=float, default=0.70)
    return parser.parse_args()


if __name__ == "__main__":
    main()
