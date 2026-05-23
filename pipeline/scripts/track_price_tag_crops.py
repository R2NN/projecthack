"""Track price tag detections in a video and export top OCR crops per track."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from predict_rfdetr_price_tags import (
    box_iou_matrix,
    filter_predictions,
    padded_crop,
    predict_image,
    save_rgb,
)


MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def resolve_device(requested: str) -> str:
    requested = (requested or os.environ.get("LENTA_RESOLVED_DEVICE") or os.environ.get("LENTA_INFERENCE_DEVICE") or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}. Use auto, cpu, or cuda.")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("GPU_NOT_AVAILABLE: torch is not installed in this Python environment.")
        return "cpu"
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("GPU_NOT_AVAILABLE: CUDA device was requested but torch.cuda.is_available() is false.")
    return "cuda" if cuda_available else "cpu"


@dataclass
class Candidate:
    timestamp_ms: int
    frame_index: int
    box_xyxy: np.ndarray
    confidence: float
    sharpness: float
    sharpness_score: float
    centrality: float
    size_score: float
    edge_score: float
    tag_likeness: float
    rotated_aspect: float
    quality_score: float
    structure_score: float
    red_fraction: float
    white_fraction: float
    layout_score: float
    color_coverage: float
    center_price_fraction: float
    price_center_score: float
    qr_like_score: float
    yellow_fraction: float
    dark_fraction: float
    non_tag_fraction: float
    crop_score: float
    ocr_score: float
    rotated_ocr_score: float
    ocr_text_score: float
    ocr_price_score: float
    ocr_sharpness_score: float
    ocr_contrast_score: float
    ocr_exposure_score: float
    ocr_resolution_score: float
    rectified_crop: np.ndarray | None
    rectified_success: bool
    rectified_score: float
    rectified_ocr_score: float
    ocr_crop_kind: str
    refinement_success: bool
    refinement_score: float
    refinement_method: str
    passes_quality_filters: bool
    quality_reject_reason: str
    raw_crop: np.ndarray
    rotated_crop: np.ndarray


@dataclass
class Track:
    track_id: int
    predicted_box: np.ndarray
    last_timestamp_ms: int
    candidates: list[Candidate] = field(default_factory=list)


@dataclass(frozen=True)
class MotionProbe:
    timestamp_ms: int
    shift_x: float
    shift_y: float
    speed_px_s: float
    smoothed_speed_px_s: float
    response: float
    sharpness: float
    is_stop: bool


def find_data_root(workspace: Path) -> Path:
    for child in workspace.iterdir():
        if child.is_dir() and (child / "sample.csv").exists():
            return child
    raise FileNotFoundError("Could not find extracted data directory with sample.csv")


def resolve_video_path(video_name: str, video_path: str) -> Path:
    if video_path:
        resolved = Path(video_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Video does not exist: {resolved}")
        return resolved

    data_root = find_data_root(Path.cwd())
    resolved = data_root / video_name / f"{video_name}.mp4"
    if not resolved.exists():
        raise FileNotFoundError(f"Video does not exist: {resolved}")
    return resolved


def sampled_timestamps(video_path: Path, step_ms: int, start_ms: int, end_ms: int | None) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = int(round(frame_count / fps * 1000)) if fps > 0 and frame_count > 0 else 0
    cap.release()

    stop_ms = min(end_ms, duration_ms) if end_ms is not None and duration_ms else end_ms
    if stop_ms is None:
        stop_ms = duration_ms
    if not stop_ms or stop_ms <= start_ms:
        raise RuntimeError(f"Could not determine a valid video duration for {video_path}")

    return list(range(max(0, start_ms), stop_ms + 1, step_ms))


def read_extra_timestamps(path: str) -> list[int]:
    if not path:
        return []

    timestamps_path = Path(path).resolve()
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Extra timestamps file does not exist: {timestamps_path}")

    timestamps = []
    with timestamps_path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        has_header = "timestamp" in sample.splitlines()[0].lower() if sample.splitlines() else False
        if has_header:
            reader = csv.DictReader(file)
            for row in reader:
                value = row.get("timestamp_ms") or row.get("frame_timestamp") or next(iter(row.values()))
                timestamps.append(int(round(float(str(value).replace(",", ".").strip()))))
        else:
            for line in file:
                value = line.strip()
                if value:
                    timestamps.append(int(round(float(value.replace(",", ".")))))
    return timestamps


def video_duration_ms(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not determine video duration: {video_path}")
    return int(round(frame_count / fps * 1000))


def read_frame_at(cap: cv2.VideoCapture, timestamp_ms: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def gray_for_motion(image_rgb: np.ndarray, width: int = 480) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    height, original_width = gray.shape[:2]
    scale = width / original_width
    resized = cv2.resize(gray, (width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32), scale


def estimate_shift(prev_gray: np.ndarray | None, current_gray: np.ndarray | None, scale: float) -> tuple[float, float, float]:
    if prev_gray is None or current_gray is None or prev_gray.shape != current_gray.shape:
        return 0.0, 0.0, 0.0

    window = cv2.createHanningWindow((prev_gray.shape[1], prev_gray.shape[0]), cv2.CV_32F)
    (dx_small, dy_small), response = cv2.phaseCorrelate(prev_gray * window, current_gray * window)
    if response < 0.02:
        return 0.0, 0.0, float(response)

    dx = float(dx_small / scale)
    dy = float(dy_small / scale)
    if abs(dx) > 1200 or abs(dy) > 1200:
        return 0.0, 0.0, float(response)
    return dx, dy, float(response)


def analyze_motion(
    video_path: Path,
    start_ms: int,
    end_ms: int | None,
    probe_step_ms: int,
    stop_speed_threshold: float,
    adaptive_stop_percentile: float,
    stop_response_min: float,
    smooth_window: int,
) -> tuple[list[MotionProbe], list[tuple[int, int]], float]:
    duration_ms = video_duration_ms(video_path)
    stop_ms = min(end_ms, duration_ms) if end_ms is not None else duration_ms
    probe_timestamps = list(range(max(0, start_ms), stop_ms + 1, probe_step_ms))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    prev_gray = None
    prev_scale = 1.0
    raw_rows = []
    for timestamp_ms in probe_timestamps:
        image_rgb = read_frame_at(cap, timestamp_ms)
        if image_rgb is None:
            continue
        current_gray, current_scale = gray_for_motion(image_rgb)
        dx, dy, response = estimate_shift(prev_gray, current_gray, prev_scale)
        speed = float(np.hypot(dx, dy) / max(probe_step_ms / 1000.0, 1e-6)) if prev_gray is not None else 0.0
        sharpness = float(cv2.Laplacian(current_gray, cv2.CV_32F).var())
        raw_rows.append(
            {
                "timestamp_ms": timestamp_ms,
                "shift_x": dx,
                "shift_y": dy,
                "speed_px_s": speed,
                "response": response,
                "sharpness": sharpness,
            }
        )
        prev_gray = current_gray
        prev_scale = current_scale
    cap.release()

    speeds = np.asarray([row["speed_px_s"] for row in raw_rows], dtype=np.float32)
    if stop_speed_threshold > 0:
        effective_stop_speed_threshold = float(stop_speed_threshold)
    elif len(speeds) > 1:
        effective_stop_speed_threshold = float(np.percentile(speeds[1:], adaptive_stop_percentile))
    else:
        effective_stop_speed_threshold = 0.0

    probes: list[MotionProbe] = []
    half_window = max(0, smooth_window // 2)
    for idx, row in enumerate(raw_rows):
        lo = max(0, idx - half_window)
        hi = min(len(raw_rows), idx + half_window + 1)
        smoothed_speed = float(np.median(speeds[lo:hi])) if len(speeds) else 0.0
        is_stop = smoothed_speed <= effective_stop_speed_threshold and (
            idx == 0
            or row["response"] >= stop_response_min
            or smoothed_speed <= effective_stop_speed_threshold * 0.45
        )
        probes.append(
            MotionProbe(
                timestamp_ms=int(row["timestamp_ms"]),
                shift_x=float(row["shift_x"]),
                shift_y=float(row["shift_y"]),
                speed_px_s=float(row["speed_px_s"]),
                smoothed_speed_px_s=smoothed_speed,
                response=float(row["response"]),
                sharpness=float(row["sharpness"]),
                is_stop=bool(is_stop),
            )
        )

    windows: list[tuple[int, int]] = []
    window_start = None
    last_stop_ts = None
    for probe in probes:
        if probe.is_stop:
            if window_start is None:
                window_start = probe.timestamp_ms
            last_stop_ts = probe.timestamp_ms
            continue
        if window_start is not None and last_stop_ts is not None:
            windows.append((window_start, last_stop_ts))
        window_start = None
        last_stop_ts = None
    if window_start is not None and last_stop_ts is not None:
        windows.append((window_start, last_stop_ts))

    return probes, windows, effective_stop_speed_threshold


def filter_stop_windows(windows: list[tuple[int, int]], min_stop_duration_ms: int) -> list[tuple[int, int]]:
    return [(start, end) for start, end in windows if end - start >= min_stop_duration_ms]


def motion_aware_timestamps(
    video_path: Path,
    start_ms: int,
    end_ms: int | None,
    probe_step_ms: int,
    stop_frame_step_ms: int,
    moving_frame_step_ms: int,
    stop_speed_threshold: float,
    adaptive_stop_percentile: float,
    stop_response_min: float,
    min_stop_duration_ms: int,
    smooth_window: int,
) -> tuple[list[int], list[MotionProbe], list[tuple[int, int]], float]:
    duration_ms = video_duration_ms(video_path)
    stop_ms = min(end_ms, duration_ms) if end_ms is not None else duration_ms
    probes, windows, effective_stop_speed_threshold = analyze_motion(
        video_path=video_path,
        start_ms=start_ms,
        end_ms=stop_ms,
        probe_step_ms=probe_step_ms,
        stop_speed_threshold=stop_speed_threshold,
        adaptive_stop_percentile=adaptive_stop_percentile,
        stop_response_min=stop_response_min,
        smooth_window=smooth_window,
    )
    windows = filter_stop_windows(windows, min_stop_duration_ms)

    selected = set()
    if moving_frame_step_ms > 0:
        selected.update(range(max(0, start_ms), stop_ms + 1, moving_frame_step_ms))
    for window_start, window_end in windows:
        padded_start = max(max(0, start_ms), window_start - probe_step_ms)
        padded_end = min(stop_ms, window_end + probe_step_ms)
        selected.update(range(padded_start, padded_end + 1, stop_frame_step_ms))
        selected.add((padded_start + padded_end) // 2)
    selected.add(max(0, start_ms))
    selected.add(stop_ms)

    timestamps = sorted(ts for ts in selected if max(0, start_ms) <= ts <= stop_ms)
    return timestamps, probes, windows, effective_stop_speed_threshold


def save_motion_diagnostics(
    output_dir: Path,
    probes: list[MotionProbe],
    windows: list[tuple[int, int]],
    selected_timestamps: list[int],
    effective_stop_speed_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(selected_timestamps)
    with (output_dir / "motion_timeline.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp_ms",
                "shift_x",
                "shift_y",
                "speed_px_s",
                "smoothed_speed_px_s",
                "response",
                "sharpness",
                "is_stop",
                "selected_for_detection",
            ],
        )
        writer.writeheader()
        for probe in probes:
            writer.writerow(
                {
                    "timestamp_ms": probe.timestamp_ms,
                    "shift_x": f"{probe.shift_x:.3f}",
                    "shift_y": f"{probe.shift_y:.3f}",
                    "speed_px_s": f"{probe.speed_px_s:.3f}",
                    "smoothed_speed_px_s": f"{probe.smoothed_speed_px_s:.3f}",
                    "response": f"{probe.response:.6f}",
                    "sharpness": f"{probe.sharpness:.3f}",
                    "is_stop": int(probe.is_stop),
                    "selected_for_detection": int(probe.timestamp_ms in selected),
                }
            )

    with (output_dir / "stop_windows.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "effective_stop_speed_threshold": effective_stop_speed_threshold,
                "windows": [
                    {"start_ms": start, "end_ms": end, "duration_ms": end - start}
                    for start, end in windows
                ],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "selected_timestamps.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp_ms"])
        writer.writeheader()
        for timestamp_ms in selected_timestamps:
            writer.writerow({"timestamp_ms": timestamp_ms})


def crop_sharpness(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def roi_readability_score(roi_gray: np.ndarray) -> tuple[float, float, float, float]:
    if roi_gray.size < 400:
        return 0.0, 0.0, 0.0, 0.0

    roi = roi_gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    laplacian_var = float(cv2.Laplacian(clahe, cv2.CV_32F).var())
    sharpness_score = min(1.0, np.log1p(laplacian_var) / 8.4)

    contrast = float(np.percentile(clahe, 92) - np.percentile(clahe, 8))
    contrast_score = min(1.0, contrast / 115.0)

    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    edge_mask = gradient > 32.0
    edge_density = float(np.mean(edge_mask))
    edge_density_score = min(1.0, edge_density / 0.18)

    dark_pixels = float(np.mean(clahe < 105))
    dark_text_score = min(1.0, dark_pixels / 0.28)

    score = (
        0.38 * sharpness_score
        + 0.26 * contrast_score
        + 0.22 * edge_density_score
        + 0.14 * dark_text_score
    )
    return float(score), sharpness_score, contrast_score, edge_density_score


def rotated_crop_ocr_metrics(rotated_crop: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    gray = cv2.cvtColor(rotated_crop, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    if height < 20 or width < 20:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pad_y = max(1, round(height * 0.04))
    pad_x = max(1, round(width * 0.04))
    gray = gray[pad_y : height - pad_y, pad_x : width - pad_x]
    height, width = gray.shape[:2]
    if height < 20 or width < 20:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    top_text = gray[: max(1, int(height * 0.58)), : max(1, int(width * 0.72))]
    price_text = gray[int(height * 0.42) : max(int(height * 0.92), int(height * 0.42) + 1), : max(1, int(width * 0.78))]
    full_score, full_sharpness, full_contrast, _ = roi_readability_score(gray)
    text_score, text_sharpness, text_contrast, _ = roi_readability_score(top_text)
    price_score, price_sharpness, price_contrast, _ = roi_readability_score(price_text)

    min_dimension = min(height, width)
    area = height * width
    resolution_score = min(1.0, min_dimension / 180.0) * min(1.0, np.sqrt(area) / 430.0)

    overexposed = float(np.mean(gray >= 252))
    underexposed = float(np.mean(gray <= 12))
    exposure_penalty = max(0.0, overexposed - 0.30) * 1.8 + max(0.0, underexposed - 0.08) * 1.4
    exposure_score = max(0.0, 1.0 - exposure_penalty)

    sharpness_score = max(full_sharpness, text_sharpness, price_sharpness)
    contrast_score = max(full_contrast, text_contrast, price_contrast)
    ocr_score = (
        0.36 * text_score
        + 0.27 * price_score
        + 0.14 * full_score
        + 0.11 * resolution_score
        + 0.08 * exposure_score
        + 0.04 * sharpness_score
    )
    return (
        float(ocr_score),
        float(text_score),
        float(price_score),
        float(sharpness_score),
        float(contrast_score),
        float(exposure_score),
        float(resolution_score),
    )


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def expand_quad(points: np.ndarray, width: int, height: int, scale: float = 1.025) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    expanded = center + (points - center) * scale
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype(np.float32)


def warp_quad(image_rgb: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    ordered = order_quad_points(quad)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    target_width = int(round(max(top_width, bottom_width)))
    target_height = int(round(max(left_height, right_height)))
    if target_width < 40 or target_height < 30:
        return None

    target_width = min(max(target_width, 64), 1200)
    target_height = min(max(target_height, 48), 800)
    destination = np.asarray(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image_rgb, matrix, (target_width, target_height), flags=cv2.INTER_CUBIC)


def refine_single_tag_crop(rotated_crop: np.ndarray) -> tuple[np.ndarray, bool, float, str]:
    height, width = rotated_crop.shape[:2]
    if height < 60 or width < 80:
        return rotated_crop, False, 0.0, "too_small"

    hsv = cv2.cvtColor(rotated_crop, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    price_mask = (((hue < 16) | (hue > 170)) & (saturation > 35) & (value > 55)).astype(np.uint8) * 255
    price_mask[: int(height * 0.30), :] = 0

    kernel_width = max(5, int(round(width / 55)) | 1)
    kernel_height = max(3, int(round(height / 55)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    price_mask = cv2.morphologyEx(price_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    price_mask = cv2.morphologyEx(price_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(price_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rotated_crop, False, 0.0, "no_price_component"

    image_area = float(width * height)
    best_box = None
    best_score = -1.0
    best_method = "red_component"
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if area_ratio < 0.012:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.14 or h < height * 0.08:
            continue
        if y + h / 2 < height * 0.45:
            continue

        pad_x = int(round(max(6.0, w * 0.08)))
        pad_bottom = int(round(max(3.0, h * 0.08)))
        x1 = max(0, x - pad_x)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_bottom)
        estimated_height = int(round(max(2.15 * h, (x2 - x1) / 1.70)))
        y1 = max(0, y2 - estimated_height)

        crop_width = x2 - x1
        crop_height = y2 - y1
        if crop_width < 50 or crop_height < 45:
            continue
        crop_aspect = crop_width / max(1, crop_height)
        if crop_aspect < 0.70 or crop_aspect > 2.45:
            continue

        price_center_x = (x + w / 2) / max(1, width)
        crop_center_x = (x1 + x2) / (2 * max(1, width))
        frame_center_score = max(0.0, 1.0 - abs(crop_center_x - 0.5) / 0.50)
        component_center_score = max(0.0, 1.0 - abs(price_center_x - 0.5) / 0.50)
        crop_fill = (crop_width * crop_height) / image_area
        size_score = min(1.0, crop_fill / 0.55)
        aspect_score = 1.0 - min(1.0, abs(np.log(max(crop_aspect, 1e-3) / 1.65)) / 1.0)
        edge_touch = int(x1 <= 1) + int(x2 >= width - 2) + int(y1 <= 1) + int(y2 >= height - 2)
        edge_penalty = 0.06 * edge_touch
        score = (
            0.34 * min(1.0, area_ratio / 0.18)
            + 0.24 * aspect_score
            + 0.20 * size_score
            + 0.14 * frame_center_score
            + 0.08 * component_center_score
            - edge_penalty
        )

        if score > best_score:
            best_score = float(score)
            best_box = (x1, y1, x2, y2)

    if best_box is None or best_score < 0.18:
        return rotated_crop, False, max(0.0, best_score), "no_valid_price_component"

    x1, y1, x2, y2 = best_box
    crop_area_fraction = ((x2 - x1) * (y2 - y1)) / image_area
    if crop_area_fraction > 0.86:
        return rotated_crop, False, float(best_score), "already_tight"

    refined_crop = rotated_crop[y1:y2, x1:x2]
    if refined_crop.size == 0:
        return rotated_crop, False, 0.0, "empty_refined_crop"

    return refined_crop, True, float(np.clip(best_score, 0.0, 1.0)), best_method


def contour_quad(contour: np.ndarray) -> tuple[np.ndarray, str]:
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    for source_contour in (hull, contour):
        for epsilon_ratio in (0.018, 0.025, 0.035, 0.05, 0.07, 0.10):
            approx = cv2.approxPolyDP(source_contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32), "quad"

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32), "min_area_rect"


def rectify_price_tag_crop(rotated_crop: np.ndarray) -> tuple[np.ndarray | None, bool, float, str]:
    height, width = rotated_crop.shape[:2]
    if height < 40 or width < 60:
        return None, False, 0.0, "too_small"

    hsv = cv2.cvtColor(rotated_crop, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    white_mask = (saturation < 85) & (value > 125)
    orange_mask = ((hue < 25) | (hue > 170)) & (saturation > 35) & (value > 65)
    yellow_mask = (hue >= 18) & (hue <= 48) & (saturation > 35) & (value > 85)
    mask = (white_mask | orange_mask | yellow_mask).astype(np.uint8) * 255

    kernel_size = max(3, int(round(min(width, height) / 35)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False, 0.0, "no_contours"

    image_area = float(width * height)
    best = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if area_ratio < 0.16 or area_ratio > 0.98:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(1, h)
        if aspect < 0.55 or aspect > 3.4:
            continue

        extent = area / max(1.0, float(w * h))
        center_x = x + w / 2
        center_y = y + h / 2
        center_distance = np.hypot((center_x - width / 2) / max(1.0, width / 2), (center_y - height / 2) / max(1.0, height / 2))
        center_score = max(0.0, 1.0 - float(center_distance / np.sqrt(2.0)))
        edge_touch = int(x <= 2) + int(y <= 2) + int(x + w >= width - 3) + int(y + h >= height - 3)
        edge_penalty = 0.06 * edge_touch
        aspect_score = 1.0 - min(1.0, abs(np.log(max(aspect, 1e-3) / 1.85)) / 1.2)

        score = 0.42 * area_ratio + 0.22 * extent + 0.20 * center_score + 0.16 * aspect_score - edge_penalty
        if score > best_score:
            best = contour
            best_score = score

    if best is None:
        return None, False, 0.0, "no_valid_contour"

    quad, method = contour_quad(best)
    quad = expand_quad(quad, width, height)
    rectified = warp_quad(rotated_crop, quad)
    if rectified is None:
        return None, False, 0.0, "warp_failed"

    rect_height, rect_width = rectified.shape[:2]
    rect_aspect = rect_width / max(1, rect_height)
    if rect_aspect < 0.65 or rect_aspect > 3.8:
        return None, False, float(best_score), f"bad_aspect_{method}"

    return rectified, True, float(best_score), method


def quality_metrics(
    image_rgb: np.ndarray,
    box: np.ndarray,
    confidence: float,
    raw_crop: np.ndarray,
    rotated_crop: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    sharpness = crop_sharpness(raw_crop)
    sharpness_score = min(1.0, np.log1p(sharpness) / 8.0)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    dx = abs(cx - width / 2) / max(1.0, width / 2)
    dy = abs(cy - height / 2) / max(1.0, height / 2)
    centrality = max(0.0, 1.0 - float(np.hypot(dx, dy) / np.sqrt(2.0)))

    size_score = min(1.0, area / 80_000.0)
    margin = min(x1, y1, width - x2, height - y2)
    edge_score = max(0.0, min(1.0, margin / 180.0))

    hsv = cv2.cvtColor(rotated_crop, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    bright_fraction = float(np.mean(value > 135))
    orange_fraction = float(np.mean(((hue < 25) | (hue > 170)) & (saturation > 45) & (value > 80)))
    white_fraction = float(np.mean((saturation < 55) & (value > 145)))
    tag_likeness = min(1.0, 0.45 * bright_fraction + 0.40 * orange_fraction + 0.15 * white_fraction)
    rotated_aspect = float(rotated_crop.shape[1] / max(1, rotated_crop.shape[0]))

    quality_score = (
        0.34 * confidence
        + 0.25 * sharpness_score
        + 0.16 * centrality
        + 0.08 * size_score
        + 0.05 * edge_score
        + 0.12 * tag_likeness
    )
    return sharpness, sharpness_score, centrality, size_score, edge_score, tag_likeness, rotated_aspect, float(quality_score)


def price_tag_structure_metrics(crop_rgb: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    height, width = crop_rgb.shape[:2]
    if height < 30 or width < 40:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    price_mask = ((hue < 16) | (hue > 170)) & (saturation > 35) & (value > 55)
    white_mask = (saturation < 82) & (value > 135)
    tag_mask = (price_mask | white_mask).astype(np.uint8) * 255

    red_fraction = float(np.mean(price_mask))
    white_fraction = float(np.mean(white_mask))

    top = slice(0, max(1, int(height * 0.62)))
    bottom = slice(int(height * 0.35), height)
    center_columns = slice(int(width * 0.18), max(int(width * 0.82), int(width * 0.18) + 1))
    bottom_center = price_mask[bottom, center_columns]
    top_white_fraction = float(np.mean(white_mask[top, :]))
    bottom_red_fraction = float(np.mean(price_mask[bottom, :]))
    center_price_fraction = float(np.mean(bottom_center)) if bottom_center.size else 0.0
    bottom_price_mask = price_mask[bottom, :]
    _, price_xs = np.where(bottom_price_mask)
    if len(price_xs):
        price_center_x = float(np.mean(price_xs) / max(1, width))
        price_center_score = max(0.0, 1.0 - abs(price_center_x - 0.5) / 0.35)
    else:
        price_center_score = 0.0
    red_presence = min(1.0, red_fraction / 0.13)
    white_presence = min(1.0, white_fraction / 0.22)
    center_price_presence = min(1.0, center_price_fraction / 0.14)
    layout_score = 0.52 * min(1.0, bottom_red_fraction / 0.20) + 0.48 * min(1.0, top_white_fraction / 0.24)

    kernel_size = max(3, int(round(min(width, height) / 32)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tag_mask = cv2.morphologyEx(tag_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(tag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_coverage = 0.0
    aspect_score = 0.0
    center_score = 0.0
    edge_penalty = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        color_coverage = area / max(1.0, float(width * height))
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(1, h)
        aspect_score = 1.0 - min(1.0, abs(np.log(max(aspect, 1e-3) / 1.75)) / 1.35)
        center_x = x + w / 2
        center_y = y + h / 2
        center_distance = np.hypot(
            (center_x - width / 2) / max(1.0, width / 2),
            (center_y - height / 2) / max(1.0, height / 2),
        )
        center_score = max(0.0, 1.0 - float(center_distance / np.sqrt(2.0)))
        horizontal_fill = w / max(1, width)
        vertical_fill = h / max(1, height)
        touches_left = x <= max(2, width * 0.025)
        touches_right = x + w >= width - max(2, width * 0.025)
        touches_top = y <= max(2, height * 0.025)
        touches_bottom = y + h >= height - max(2, height * 0.025)
        edge_touches = int(touches_left) + int(touches_right) + int(touches_top) + int(touches_bottom)
        if horizontal_fill < 0.68 or vertical_fill < 0.62:
            edge_penalty += 0.12
        if edge_touches >= 3 and color_coverage < 0.72:
            edge_penalty += 0.10

    coverage_score = min(1.0, color_coverage / 0.50)
    structure_score = (
        0.23 * red_presence
        + 0.17 * white_presence
        + 0.23 * layout_score
        + 0.14 * center_price_presence
        + 0.08 * coverage_score
        + 0.07 * aspect_score
        + 0.05 * center_score
        + 0.03 * price_center_score
        - edge_penalty
    )
    return (
        float(np.clip(structure_score, 0.0, 1.0)),
        red_fraction,
        white_fraction,
        float(layout_score),
        float(color_coverage),
        center_price_fraction,
        float(price_center_score),
    )


def qr_like_score(crop_rgb: np.ndarray) -> float:
    height, width = crop_rgb.shape[:2]
    if height < 50 or width < 70:
        return 0.0

    region = crop_rgb[
        int(height * 0.02) : max(int(height * 0.55), int(height * 0.02) + 1),
        int(width * 0.50) : max(int(width * 0.99), int(width * 0.50) + 1),
    ]
    if min(region.shape[:2]) < 16:
        return 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7,
    )
    kernel_size = max(3, int(min(region.shape[:2]) / 18) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_height, region_width = region.shape[:2]
    best_score = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        aspect = w / max(1, h)
        square_score = max(0.0, 1.0 - abs(np.log(max(aspect, 1e-3))) / 0.38)
        relative_area = (w * h) / max(1.0, float(region_height * region_width))
        size_score = min(1.0, relative_area / 0.16)
        fill_ratio = cv2.contourArea(contour) / max(1.0, float(w * h))
        fill_score = max(0.0, 1.0 - abs(fill_ratio - 0.35) / 0.35)
        score = size_score * (0.70 * square_score + 0.30 * fill_score)
        best_score = max(best_score, float(score))

    return float(np.clip(best_score, 0.0, 1.0))


def background_noise_metrics(crop_rgb: np.ndarray) -> tuple[float, float, float]:
    height, width = crop_rgb.shape[:2]
    if height < 20 or width < 20:
        return 1.0, 1.0, 1.0

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    price_mask = ((hue < 16) | (hue > 170)) & (saturation > 35) & (value > 55)
    white_mask = (saturation < 82) & (value > 135)
    yellow_mask = (hue >= 16) & (hue <= 48) & (saturation > 35) & (value > 55)
    dark_mask = value < 75
    non_tag_mask = ~(price_mask | white_mask)
    return (
        float(np.mean(yellow_mask)),
        float(np.mean(dark_mask)),
        float(np.mean(non_tag_mask)),
    )


def center_distance_norm(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = np.asarray([(box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2], dtype=np.float32)
    center_b = np.asarray([(box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2], dtype=np.float32)
    area_a = max(1.0, float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = max(1.0, float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    scale = max(np.sqrt(area_a), np.sqrt(area_b), 1.0)
    return float(np.linalg.norm(center_a - center_b) / scale)


def match_track(
    tracks: list[Track],
    box: np.ndarray,
    timestamp_ms: int,
    used_track_ids: set[int],
    max_gap_ms: int,
    min_iou: float,
    max_center_distance: float,
) -> Track | None:
    best_track = None
    best_score = -1.0
    for track in tracks:
        if track.track_id in used_track_ids:
            continue
        if timestamp_ms - track.last_timestamp_ms > max_gap_ms:
            continue

        iou = float(box_iou_matrix(box[None, :], track.predicted_box[None, :])[0, 0])
        distance = center_distance_norm(box, track.predicted_box)
        if iou < min_iou and distance > max_center_distance:
            continue

        distance_score = max(0.0, 1.0 - distance / max_center_distance)
        score = iou + 0.45 * distance_score
        if score > best_score:
            best_score = score
            best_track = track
    return best_track


def add_candidate(track: Track, candidate: Candidate, max_candidates: int) -> None:
    track.candidates.append(candidate)
    track.candidates.sort(key=lambda item: (item.passes_quality_filters, item.crop_score), reverse=True)
    del track.candidates[max_candidates:]


def make_crop_contact_sheet(rows: list[dict[str, str]], output_path: Path, tile_width: int = 240, tile_height: int = 190) -> None:
    if not rows:
        return

    font = ImageFont.load_default()
    columns = 4
    rows_count = int(np.ceil(len(rows) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(rows):
        image = Image.open(row.get("ocr_crop") or row["rotated_crop"]).convert("RGB")
        image.thumbnail((tile_width - 16, tile_height - 36), Image.Resampling.LANCZOS)
        x = (idx % columns) * tile_width + (tile_width - image.width) // 2
        y = (idx // columns) * tile_height + 20
        sheet.paste(image, (x, y))
        label = (
            f"T{row['track_id']} #{row['rank']} "
            f"crop={float(row.get('crop_score', 0.0)):.2f} "
            f"ocr={float(row['ocr_score']):.2f}"
        )
        draw.text(((idx % columns) * tile_width + 8, (idx // columns) * tile_height + 4), label, fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def make_paginated_crop_contact_sheets(
    rows: list[dict[str, str]],
    output_dir: Path,
    prefix: str,
    page_size: int,
) -> list[str]:
    if not rows or page_size <= 0:
        return []

    output_paths = []
    pages = int(np.ceil(len(rows) / page_size))
    for page_index in range(pages):
        page_rows = rows[page_index * page_size : (page_index + 1) * page_size]
        page_path = output_dir / f"{prefix}_page_{page_index + 1:03d}.jpg"
        make_crop_contact_sheet(page_rows, page_path)
        output_paths.append(str(page_path))
        if page_index == 0:
            make_crop_contact_sheet(page_rows, output_dir / f"{prefix}.jpg")
    return output_paths


def best_rows_per_track(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_track: dict[str, dict[str, str]] = {}
    for row in rows:
        track_id = row["track_id"]
        current = best_by_track.get(track_id)
        if current is None or float(row["crop_score"]) > float(current["crop_score"]):
            best_by_track[track_id] = row
    return [best_by_track[track_id] for track_id in sorted(best_by_track, key=lambda value: int(value))]


def save_rows_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["track_id"])
        writer.writeheader()
        writer.writerows(rows)


def track_candidate_diagnostic_rows(tracks: list[Track], exported_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    exported_track_ids = {row["track_id"] for row in exported_rows}
    selected_keys = {
        (row["track_id"], row["timestamp_ms"], row["frame_index"])
        for row in exported_rows
    }
    rows = []
    for track in sorted(tracks, key=lambda item: item.track_id):
        track_id = str(track.track_id)
        track_exported = int(track_id in exported_track_ids)
        for rank, candidate in enumerate(track.candidates, start=1):
            key = (track_id, str(candidate.timestamp_ms), str(candidate.frame_index))
            rows.append(
                {
                    "track_id": track_id,
                    "candidate_rank": str(rank),
                    "track_exported": str(track_exported),
                    "selected_for_ocr_export": str(int(key in selected_keys)),
                    "track_candidate_count": str(len(track.candidates)),
                    "timestamp_ms": str(candidate.timestamp_ms),
                    "frame_index": str(candidate.frame_index),
                    "quality_score": f"{candidate.quality_score:.6f}",
                    "crop_score": f"{candidate.crop_score:.6f}",
                    "confidence": f"{candidate.confidence:.6f}",
                    "sharpness": f"{candidate.sharpness:.3f}",
                    "sharpness_score": f"{candidate.sharpness_score:.6f}",
                    "centrality": f"{candidate.centrality:.6f}",
                    "size_score": f"{candidate.size_score:.6f}",
                    "edge_score": f"{candidate.edge_score:.6f}",
                    "tag_likeness": f"{candidate.tag_likeness:.6f}",
                    "structure_score": f"{candidate.structure_score:.6f}",
                    "red_fraction": f"{candidate.red_fraction:.6f}",
                    "white_fraction": f"{candidate.white_fraction:.6f}",
                    "layout_score": f"{candidate.layout_score:.6f}",
                    "color_coverage": f"{candidate.color_coverage:.6f}",
                    "center_price_fraction": f"{candidate.center_price_fraction:.6f}",
                    "price_center_score": f"{candidate.price_center_score:.6f}",
                    "qr_like_score": f"{candidate.qr_like_score:.6f}",
                    "yellow_fraction": f"{candidate.yellow_fraction:.6f}",
                    "dark_fraction": f"{candidate.dark_fraction:.6f}",
                    "non_tag_fraction": f"{candidate.non_tag_fraction:.6f}",
                    "rectified_success": str(int(candidate.rectified_success)),
                    "rectified_score": f"{candidate.rectified_score:.6f}",
                    "ocr_crop_kind": candidate.ocr_crop_kind,
                    "refinement_success": str(int(candidate.refinement_success)),
                    "refinement_score": f"{candidate.refinement_score:.6f}",
                    "refinement_method": candidate.refinement_method,
                    "passes_quality_filters": str(int(candidate.passes_quality_filters)),
                    "quality_reject_reason": candidate.quality_reject_reason,
                    "x_min": f"{candidate.box_xyxy[0]:.2f}",
                    "y_min": f"{candidate.box_xyxy[1]:.2f}",
                    "x_max": f"{candidate.box_xyxy[2]:.2f}",
                    "y_max": f"{candidate.box_xyxy[3]:.2f}",
                }
            )
    return rows


def color_for_track(track_id: int) -> tuple[int, int, int]:
    palette = [
        (229, 57, 53),
        (30, 136, 229),
        (67, 160, 71),
        (251, 140, 0),
        (142, 36, 170),
        (0, 137, 123),
        (216, 27, 96),
        (94, 53, 177),
    ]
    return palette[track_id % len(palette)]


def make_frame_overlays(video_path: Path, rows: list[dict[str, str]], output_dir: Path) -> list[dict[str, str]]:
    if not rows:
        return []

    rows_by_timestamp: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_timestamp.setdefault(int(row["timestamp_ms"]), []).append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlays: {video_path}")

    overlay_rows = []
    for timestamp_ms, timestamp_rows in sorted(rows_by_timestamp.items()):
        image_rgb = read_frame_at(cap, timestamp_ms)
        if image_rgb is None:
            continue

        image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(image)
        for row in timestamp_rows:
            track_id = int(row["track_id"])
            color = color_for_track(track_id)
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])
            draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=4)
            label = f"T{track_id} #{row['rank']} {float(row['crop_score']):.2f}"
            text_bbox = draw.textbbox((x_min, y_min), label, font=font)
            draw.rectangle(
                (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2),
                fill=color,
            )
            draw.text((x_min, y_min), label, fill=(255, 255, 255), font=font)

        overlay_path = output_dir / f"overlay_ts_{timestamp_ms:06d}.jpg"
        image.save(overlay_path, quality=92)
        overlay_rows.append(
            {
                "timestamp_ms": str(timestamp_ms),
                "overlay_path": str(overlay_path),
                "box_count": str(len(timestamp_rows)),
            }
        )

    cap.release()
    save_rows_csv(overlay_rows, output_dir / "overlay_index.csv")
    return overlay_rows


def export_tracks(
    tracks: list[Track],
    output_dir: Path,
    top_k: int,
    min_track_detections: int,
    keep_singletons_score: float,
    min_export_crop_score: float,
) -> list[dict[str, str]]:
    rows = []
    for track in sorted(tracks, key=lambda item: item.track_id):
        if not track.candidates:
            continue

        export_candidates = [
            candidate
            for candidate in track.candidates
            if candidate.passes_quality_filters and candidate.crop_score >= min_export_crop_score
        ]
        if not export_candidates:
            continue

        best_confidence = max(candidate.confidence for candidate in export_candidates)
        if len(export_candidates) < min_track_detections and best_confidence < keep_singletons_score:
            continue

        track_dir = output_dir / "tracks" / f"track_{track.track_id:04d}"
        for rank, candidate in enumerate(export_candidates[:top_k], start=1):
            base_name = f"top_{rank:02d}_ts_{candidate.timestamp_ms:06d}_ocr_{candidate.ocr_score:.3f}_q_{candidate.quality_score:.3f}"
            raw_path = track_dir / f"{base_name}_raw.jpg"
            rotated_path = track_dir / f"{base_name}_rot_ccw.jpg"
            rectified_path = track_dir / f"{base_name}_rectified.jpg"
            save_rgb(raw_path, candidate.raw_crop)
            save_rgb(rotated_path, candidate.rotated_crop)
            if candidate.rectified_crop is not None and candidate.rectified_success:
                save_rgb(rectified_path, candidate.rectified_crop)
                rectified_path_value = str(rectified_path)
            else:
                rectified_path_value = ""
            ocr_crop_path = rectified_path_value if candidate.ocr_crop_kind == "rectified" and rectified_path_value else str(rotated_path)
            rows.append(
                {
                    "track_id": str(track.track_id),
                    "rank": str(rank),
                    "track_candidate_count": str(len(track.candidates)),
                    "track_quality_candidate_count": str(len(export_candidates)),
                    "timestamp_ms": str(candidate.timestamp_ms),
                    "frame_index": str(candidate.frame_index),
                    "quality_score": f"{candidate.quality_score:.6f}",
                    "crop_score": f"{candidate.crop_score:.6f}",
                    "confidence": f"{candidate.confidence:.6f}",
                    "sharpness": f"{candidate.sharpness:.3f}",
                    "sharpness_score": f"{candidate.sharpness_score:.6f}",
                    "centrality": f"{candidate.centrality:.6f}",
                    "size_score": f"{candidate.size_score:.6f}",
                    "edge_score": f"{candidate.edge_score:.6f}",
                    "tag_likeness": f"{candidate.tag_likeness:.6f}",
                    "rotated_aspect": f"{candidate.rotated_aspect:.6f}",
                    "structure_score": f"{candidate.structure_score:.6f}",
                    "red_fraction": f"{candidate.red_fraction:.6f}",
                    "white_fraction": f"{candidate.white_fraction:.6f}",
                    "layout_score": f"{candidate.layout_score:.6f}",
                    "color_coverage": f"{candidate.color_coverage:.6f}",
                    "center_price_fraction": f"{candidate.center_price_fraction:.6f}",
                    "price_center_score": f"{candidate.price_center_score:.6f}",
                    "qr_like_score": f"{candidate.qr_like_score:.6f}",
                    "yellow_fraction": f"{candidate.yellow_fraction:.6f}",
                    "dark_fraction": f"{candidate.dark_fraction:.6f}",
                    "non_tag_fraction": f"{candidate.non_tag_fraction:.6f}",
                    "ocr_score": f"{candidate.ocr_score:.6f}",
                    "rotated_ocr_score": f"{candidate.rotated_ocr_score:.6f}",
                    "rectified_ocr_score": f"{candidate.rectified_ocr_score:.6f}",
                    "rectified_success": int(candidate.rectified_success),
                    "rectified_score": f"{candidate.rectified_score:.6f}",
                    "ocr_crop_kind": candidate.ocr_crop_kind,
                    "refinement_success": int(candidate.refinement_success),
                    "refinement_score": f"{candidate.refinement_score:.6f}",
                    "refinement_method": candidate.refinement_method,
                    "passes_quality_filters": int(candidate.passes_quality_filters),
                    "quality_reject_reason": candidate.quality_reject_reason,
                    "ocr_text_score": f"{candidate.ocr_text_score:.6f}",
                    "ocr_price_score": f"{candidate.ocr_price_score:.6f}",
                    "ocr_sharpness_score": f"{candidate.ocr_sharpness_score:.6f}",
                    "ocr_contrast_score": f"{candidate.ocr_contrast_score:.6f}",
                    "ocr_exposure_score": f"{candidate.ocr_exposure_score:.6f}",
                    "ocr_resolution_score": f"{candidate.ocr_resolution_score:.6f}",
                    "x_min": f"{candidate.box_xyxy[0]:.2f}",
                    "y_min": f"{candidate.box_xyxy[1]:.2f}",
                    "x_max": f"{candidate.box_xyxy[2]:.2f}",
                    "y_max": f"{candidate.box_xyxy[3]:.2f}",
                    "raw_crop": str(raw_path),
                    "rotated_crop": str(rotated_path),
                    "rectified_crop": rectified_path_value,
                    "ocr_crop": ocr_crop_path,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-name", default="25_12-20")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--checkpoint", default="runs/rfdetr_small_price_tag_tiled1280_e8/checkpoint_best_total.pth")
    parser.add_argument("--output", default="")
    parser.add_argument("--variant", choices=sorted(MODEL_CLASSES), default="small")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=os.environ.get("LENTA_RESOLVED_DEVICE", os.environ.get("LENTA_INFERENCE_DEVICE", "auto")))
    parser.add_argument("--frame-step-ms", type=int, default=500)
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--extra-timestamps-file", default="")
    parser.add_argument("--sampling-mode", choices=["fixed", "stops"], default="fixed")
    parser.add_argument("--motion-probe-step-ms", type=int, default=500)
    parser.add_argument("--stop-frame-step-ms", type=int, default=250)
    parser.add_argument("--moving-frame-step-ms", type=int, default=1500)
    parser.add_argument("--stop-speed-threshold", type=float, default=0.0)
    parser.add_argument("--adaptive-stop-percentile", type=float, default=60.0)
    parser.add_argument("--stop-response-min", type=float, default=0.03)
    parser.add_argument("--min-stop-duration-ms", type=int, default=650)
    parser.add_argument("--motion-smooth-window", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=320)
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--selection-mode", choices=["candidate", "track"], default="candidate")
    parser.add_argument("--contact-sheet-page-size", type=int, default=80)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--disable-overlays", action="store_true")
    parser.add_argument("--max-track-candidates", type=int, default=12)
    parser.add_argument("--max-gap-ms", type=int, default=1800)
    parser.add_argument("--track-iou", type=float, default=0.18)
    parser.add_argument("--max-center-distance", type=float, default=1.35)
    parser.add_argument("--min-track-detections", type=int, default=2)
    parser.add_argument("--keep-singletons-score", type=float, default=0.45)
    parser.add_argument("--min-area", type=float, default=8_000.0)
    parser.add_argument("--max-area", type=float, default=180_000.0)
    parser.add_argument("--min-aspect", type=float, default=0.25)
    parser.add_argument("--max-aspect", type=float, default=1.60)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--min-tag-likeness", type=float, default=0.25)
    parser.add_argument("--min-rotated-aspect", type=float, default=0.75)
    parser.add_argument("--max-rotated-aspect", type=float, default=2.35)
    parser.add_argument("--min-structure-score", type=float, default=0.0)
    parser.add_argument("--min-red-fraction", type=float, default=0.0)
    parser.add_argument("--min-white-fraction", type=float, default=0.0)
    parser.add_argument("--min-layout-score", type=float, default=0.0)
    parser.add_argument("--min-color-coverage", type=float, default=0.0)
    parser.add_argument("--min-center-price-fraction", type=float, default=0.0)
    parser.add_argument("--min-price-center-score", type=float, default=0.0)
    parser.add_argument("--max-yellow-fraction", type=float, default=1.0)
    parser.add_argument("--max-dark-fraction", type=float, default=1.0)
    parser.add_argument("--max-non-tag-fraction", type=float, default=1.0)
    parser.add_argument("--min-qr-like-score", type=float, default=0.0)
    parser.add_argument("--min-export-crop-score", type=float, default=0.0)
    parser.add_argument("--qr-like-weight", type=float, default=0.12)
    parser.add_argument("--disable-inner-refine", action="store_true")
    parser.add_argument("--reject-no-valid-refinement", action="store_true")
    parser.add_argument("--disable-rectify", action="store_true")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video_name, args.video_path)
    video_stem = video_path.stem
    output_dir = Path(args.output or f"analysis_outputs/tracked_ocr_crops_{video_stem}_tiled1280").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = MODEL_CLASSES[args.variant](pretrain_weights=str(Path(args.checkpoint).resolve()), num_classes=1, device=device)

    motion_probes: list[MotionProbe] = []
    stop_windows: list[tuple[int, int]] = []
    effective_stop_speed_threshold = 0.0
    if args.sampling_mode == "stops":
        timestamps, motion_probes, stop_windows, effective_stop_speed_threshold = motion_aware_timestamps(
            video_path=video_path,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            probe_step_ms=args.motion_probe_step_ms,
            stop_frame_step_ms=args.stop_frame_step_ms,
            moving_frame_step_ms=args.moving_frame_step_ms,
            stop_speed_threshold=args.stop_speed_threshold,
            adaptive_stop_percentile=args.adaptive_stop_percentile,
            stop_response_min=args.stop_response_min,
            min_stop_duration_ms=args.min_stop_duration_ms,
            smooth_window=args.motion_smooth_window,
        )
        save_motion_diagnostics(output_dir, motion_probes, stop_windows, timestamps, effective_stop_speed_threshold)
    else:
        timestamps = sampled_timestamps(video_path, args.frame_step_ms, args.start_ms, args.end_ms)
    extra_timestamps = read_extra_timestamps(args.extra_timestamps_file)
    if extra_timestamps:
        timestamps = sorted(set(timestamps).union(extra_timestamps))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    tracks: list[Track] = []
    next_track_id = 1
    prev_gray = None
    prev_scale = 1.0
    processed_frames = 0
    total_detections = 0
    accepted_candidates = 0
    reject_counts = {
        "empty_crop": 0,
        "tag_likeness": 0,
        "rotated_aspect": 0,
        "structure_score": 0,
        "red_fraction": 0,
        "white_fraction": 0,
        "layout_score": 0,
        "color_coverage": 0,
        "center_price_fraction": 0,
        "price_center_score": 0,
        "yellow_fraction": 0,
        "dark_fraction": 0,
        "non_tag_fraction": 0,
        "qr_like_score": 0,
        "no_valid_refinement": 0,
    }
    motion_responses = []

    for frame_index, timestamp_ms in enumerate(timestamps):
        image_rgb = read_frame_at(cap, timestamp_ms)
        if image_rgb is None:
            continue
        processed_frames += 1

        current_gray, current_scale = gray_for_motion(image_rgb)
        dx, dy, response = estimate_shift(prev_gray, current_gray, prev_scale)
        motion_responses.append(response)
        for track in tracks:
            if timestamp_ms - track.last_timestamp_ms <= args.max_gap_ms:
                track.predicted_box = track.predicted_box + np.asarray([dx, dy, dx, dy], dtype=np.float32)

        prev_gray = current_gray
        prev_scale = current_scale

        boxes, confidences = predict_image(
            model,
            image_rgb,
            threshold=args.threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
        order = np.argsort(-confidences)
        boxes = boxes[order]
        confidences = confidences[order]
        boxes, confidences = filter_predictions(
            boxes,
            confidences,
            min_area=args.min_area,
            max_area=args.max_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            nms_iou=args.nms_iou,
        )
        total_detections += len(boxes)

        used_track_ids: set[int] = set()
        for box, confidence in zip(boxes, confidences):
            raw_crop = padded_crop(image_rgb, box, args.crop_pad)
            if raw_crop is None:
                reject_counts["empty_crop"] += 1
                continue
            rotated_crop = cv2.rotate(raw_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            quality_reasons: list[str] = []
            refinement_success = False
            refinement_score = 0.0
            refinement_method = "disabled"
            if not args.disable_inner_refine:
                rotated_crop, refinement_success, refinement_score, refinement_method = refine_single_tag_crop(rotated_crop)
                if args.reject_no_valid_refinement and refinement_method in {"no_price_component", "no_valid_price_component"}:
                    reject_counts["no_valid_refinement"] += 1
                    quality_reasons.append("no_valid_refinement")
                    if args.selection_mode == "candidate":
                        continue
            (
                sharpness,
                sharpness_score,
                centrality,
                size_score,
                edge_score,
                tag_likeness,
                rotated_aspect,
                quality_score,
            ) = quality_metrics(
                image_rgb,
                box,
                float(confidence),
                raw_crop,
                rotated_crop,
            )
            if tag_likeness < args.min_tag_likeness:
                reject_counts["tag_likeness"] += 1
                quality_reasons.append("tag_likeness")
                if args.selection_mode == "candidate":
                    continue
            if rotated_aspect < args.min_rotated_aspect or rotated_aspect > args.max_rotated_aspect:
                reject_counts["rotated_aspect"] += 1
                quality_reasons.append("rotated_aspect")
                if args.selection_mode == "candidate":
                    continue

            rectified_crop = None
            rectified_success = False
            rectified_score = 0.0
            if not args.disable_rectify:
                rectified_crop, rectified_success, rectified_score, _ = rectify_price_tag_crop(rotated_crop)

            (
                rotated_ocr_score,
                rotated_text_score,
                rotated_price_score,
                rotated_sharpness_score,
                rotated_contrast_score,
                rotated_exposure_score,
                rotated_resolution_score,
            ) = rotated_crop_ocr_metrics(rotated_crop)
            if rectified_crop is not None and rectified_success:
                (
                    rectified_ocr_score,
                    rectified_text_score,
                    rectified_price_score,
                    rectified_sharpness_score,
                    rectified_contrast_score,
                    rectified_exposure_score,
                    rectified_resolution_score,
                ) = rotated_crop_ocr_metrics(rectified_crop)
            else:
                rectified_ocr_score = 0.0
                rectified_text_score = 0.0
                rectified_price_score = 0.0
                rectified_sharpness_score = 0.0
                rectified_contrast_score = 0.0
                rectified_exposure_score = 0.0
                rectified_resolution_score = 0.0

            use_rectified = bool(rectified_crop is not None and rectified_success and rectified_ocr_score >= rotated_ocr_score * 0.995)
            if use_rectified:
                ocr_score = rectified_ocr_score
                ocr_text_score = rectified_text_score
                ocr_price_score = rectified_price_score
                ocr_sharpness_score = rectified_sharpness_score
                ocr_contrast_score = rectified_contrast_score
                ocr_exposure_score = rectified_exposure_score
                ocr_resolution_score = rectified_resolution_score
                ocr_crop_kind = "rectified"
            else:
                ocr_score = rotated_ocr_score
                ocr_text_score = rotated_text_score
                ocr_price_score = rotated_price_score
                ocr_sharpness_score = rotated_sharpness_score
                ocr_contrast_score = rotated_contrast_score
                ocr_exposure_score = rotated_exposure_score
                ocr_resolution_score = rotated_resolution_score
                ocr_crop_kind = "rotated"

            structure_crop = rectified_crop if use_rectified and rectified_crop is not None else rotated_crop
            (
                structure_score,
                red_fraction,
                white_fraction,
                layout_score,
                color_coverage,
                center_price_fraction,
                price_center_score,
            ) = price_tag_structure_metrics(structure_crop)
            if structure_score < args.min_structure_score:
                reject_counts["structure_score"] += 1
                quality_reasons.append("structure_score")
                if args.selection_mode == "candidate":
                    continue
            if red_fraction < args.min_red_fraction:
                reject_counts["red_fraction"] += 1
                quality_reasons.append("red_fraction")
                if args.selection_mode == "candidate":
                    continue
            if white_fraction < args.min_white_fraction:
                reject_counts["white_fraction"] += 1
                quality_reasons.append("white_fraction")
                if args.selection_mode == "candidate":
                    continue
            if layout_score < args.min_layout_score:
                reject_counts["layout_score"] += 1
                quality_reasons.append("layout_score")
                if args.selection_mode == "candidate":
                    continue
            if color_coverage < args.min_color_coverage:
                reject_counts["color_coverage"] += 1
                quality_reasons.append("color_coverage")
                if args.selection_mode == "candidate":
                    continue
            if center_price_fraction < args.min_center_price_fraction:
                reject_counts["center_price_fraction"] += 1
                quality_reasons.append("center_price_fraction")
                if args.selection_mode == "candidate":
                    continue
            if price_center_score < args.min_price_center_score:
                reject_counts["price_center_score"] += 1
                quality_reasons.append("price_center_score")
                if args.selection_mode == "candidate":
                    continue

            yellow_fraction, dark_fraction, non_tag_fraction = background_noise_metrics(structure_crop)
            if yellow_fraction > args.max_yellow_fraction:
                reject_counts["yellow_fraction"] += 1
                quality_reasons.append("yellow_fraction")
                if args.selection_mode == "candidate":
                    continue
            if dark_fraction > args.max_dark_fraction:
                reject_counts["dark_fraction"] += 1
                quality_reasons.append("dark_fraction")
                if args.selection_mode == "candidate":
                    continue
            if non_tag_fraction > args.max_non_tag_fraction:
                reject_counts["non_tag_fraction"] += 1
                quality_reasons.append("non_tag_fraction")
                if args.selection_mode == "candidate":
                    continue

            qr_score = qr_like_score(structure_crop)
            if qr_score < args.min_qr_like_score:
                reject_counts["qr_like_score"] += 1
                quality_reasons.append("qr_like_score")
                if args.selection_mode == "candidate":
                    continue
            base_crop_score = float(
                np.clip(
                    ocr_score
                    * (
                        0.46
                        + 0.40 * structure_score
                        + 0.08 * min(1.0, center_price_fraction / 0.14)
                        + 0.06 * price_center_score
                    )
                    + 0.05 * quality_score
                    + 0.03 * tag_likeness
                    + 0.02 * edge_score,
                    0.0,
                    1.0,
                )
            )
            crop_score = float(
                np.clip(
                    (1.0 - args.qr_like_weight) * base_crop_score + args.qr_like_weight * qr_score,
                    0.0,
                    1.0,
                )
            )
            accepted_candidates += 1

            candidate = Candidate(
                timestamp_ms=timestamp_ms,
                frame_index=frame_index,
                box_xyxy=box.astype(np.float32),
                confidence=float(confidence),
                sharpness=sharpness,
                sharpness_score=sharpness_score,
                centrality=centrality,
                size_score=size_score,
                edge_score=edge_score,
                tag_likeness=tag_likeness,
                rotated_aspect=rotated_aspect,
                quality_score=quality_score,
                structure_score=structure_score,
                red_fraction=red_fraction,
                white_fraction=white_fraction,
                layout_score=layout_score,
                color_coverage=color_coverage,
                center_price_fraction=center_price_fraction,
                price_center_score=price_center_score,
                qr_like_score=qr_score,
                yellow_fraction=yellow_fraction,
                dark_fraction=dark_fraction,
                non_tag_fraction=non_tag_fraction,
                crop_score=crop_score,
                ocr_score=ocr_score,
                rotated_ocr_score=rotated_ocr_score,
                ocr_text_score=ocr_text_score,
                ocr_price_score=ocr_price_score,
                ocr_sharpness_score=ocr_sharpness_score,
                ocr_contrast_score=ocr_contrast_score,
                ocr_exposure_score=ocr_exposure_score,
                ocr_resolution_score=ocr_resolution_score,
                rectified_crop=rectified_crop,
                rectified_success=rectified_success,
                rectified_score=rectified_score,
                rectified_ocr_score=rectified_ocr_score,
                ocr_crop_kind=ocr_crop_kind,
                refinement_success=refinement_success,
                refinement_score=refinement_score,
                refinement_method=refinement_method,
                passes_quality_filters=not quality_reasons,
                quality_reject_reason="|".join(quality_reasons),
                raw_crop=raw_crop,
                rotated_crop=rotated_crop,
            )

            track = match_track(
                tracks,
                box,
                timestamp_ms,
                used_track_ids,
                args.max_gap_ms,
                args.track_iou,
                args.max_center_distance,
            )
            if track is None:
                track = Track(track_id=next_track_id, predicted_box=box.astype(np.float32), last_timestamp_ms=timestamp_ms)
                tracks.append(track)
                next_track_id += 1

            add_candidate(track, candidate, args.max_track_candidates)
            track.predicted_box = box.astype(np.float32)
            track.last_timestamp_ms = timestamp_ms
            used_track_ids.add(track.track_id)

        print(
            f"frame {processed_frames}/{len(timestamps)} ts={timestamp_ms}ms "
            f"detections={len(boxes)} tracks={len(tracks)}"
        )

    cap.release()

    rows = export_tracks(
        tracks,
        output_dir,
        top_k=args.top_k,
        min_track_detections=args.min_track_detections,
        keep_singletons_score=args.keep_singletons_score,
        min_export_crop_score=args.min_export_crop_score,
    )

    csv_path = output_dir / "tracks_top_crops.csv"
    save_rows_csv(rows, csv_path)
    all_candidate_rows = track_candidate_diagnostic_rows(tracks, rows)
    all_candidates_csv_path = output_dir / "track_candidates_all.csv"
    save_rows_csv(all_candidate_rows, all_candidates_csv_path)
    contact_sheet_paths = (
        []
        if args.skip_contact_sheets
        else make_paginated_crop_contact_sheets(
            rows,
            output_dir,
            "top_crops_contact_sheet",
            args.contact_sheet_page_size,
        )
    )

    best_rows = best_rows_per_track(rows)
    best_csv_path = output_dir / "best_per_track.csv"
    save_rows_csv(best_rows, best_csv_path)
    best_contact_sheet_paths = (
        []
        if args.skip_contact_sheets
        else make_paginated_crop_contact_sheets(
            best_rows,
            output_dir,
            "best_per_track_contact_sheet",
            args.contact_sheet_page_size,
        )
    )
    overlay_rows = [] if args.disable_overlays else make_frame_overlays(video_path, best_rows, output_dir / "overlays")

    exported_track_ids = sorted({row["track_id"] for row in rows})
    rectified_success_count = sum(int(row.get("rectified_success", 0)) for row in rows)
    rectified_used_count = sum(1 for row in rows if row.get("ocr_crop_kind") == "rectified")
    refinement_success_count = sum(int(row.get("refinement_success", 0)) for row in rows)
    summary = {
        "video": str(video_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output": str(output_dir),
        "processed_frames": processed_frames,
        "sampling_mode": args.sampling_mode,
        "selection_mode": args.selection_mode,
        "reject_counts_meaning": "quality_fail_counts" if args.selection_mode == "track" else "candidate_reject_counts",
        "frame_step_ms": args.frame_step_ms,
        "selected_timestamps": len(timestamps),
        "extra_timestamps_file": str(Path(args.extra_timestamps_file).resolve()) if args.extra_timestamps_file else "",
        "extra_timestamps_count": len(extra_timestamps),
        "motion_probe_step_ms": args.motion_probe_step_ms if args.sampling_mode == "stops" else None,
        "stop_frame_step_ms": args.stop_frame_step_ms if args.sampling_mode == "stops" else None,
        "moving_frame_step_ms": args.moving_frame_step_ms if args.sampling_mode == "stops" else None,
        "stop_speed_threshold": args.stop_speed_threshold if args.sampling_mode == "stops" else None,
        "adaptive_stop_percentile": args.adaptive_stop_percentile if args.sampling_mode == "stops" else None,
        "effective_stop_speed_threshold": round(effective_stop_speed_threshold, 3) if args.sampling_mode == "stops" else None,
        "stop_windows": [
            {"start_ms": start, "end_ms": end, "duration_ms": end - start}
            for start, end in stop_windows
        ],
        "raw_detections_after_filter": total_detections,
        "accepted_candidates": accepted_candidates,
        "reject_counts": reject_counts,
        "tracks_total": len(tracks),
        "tracks_exported": len(exported_track_ids),
        "track_candidates_total": len(all_candidate_rows),
        "track_candidates_all_csv": str(all_candidates_csv_path),
        "saved_crops": len(rows),
        "best_per_track_crops": len(best_rows),
        "rectified_success_crops": rectified_success_count,
        "rectified_used_as_ocr_crops": rectified_used_count,
        "refinement_success_crops": refinement_success_count,
        "contact_sheet_pages": contact_sheet_paths,
        "best_per_track_contact_sheet_pages": best_contact_sheet_paths,
        "overlay_frames": len(overlay_rows),
        "overlay_index": str(output_dir / "overlays" / "overlay_index.csv") if overlay_rows else "",
        "min_tag_likeness": args.min_tag_likeness,
        "min_structure_score": args.min_structure_score,
        "min_red_fraction": args.min_red_fraction,
        "min_white_fraction": args.min_white_fraction,
        "min_layout_score": args.min_layout_score,
        "min_color_coverage": args.min_color_coverage,
        "min_center_price_fraction": args.min_center_price_fraction,
        "min_price_center_score": args.min_price_center_score,
        "max_yellow_fraction": args.max_yellow_fraction,
        "max_dark_fraction": args.max_dark_fraction,
        "max_non_tag_fraction": args.max_non_tag_fraction,
        "min_qr_like_score": args.min_qr_like_score,
        "min_export_crop_score": args.min_export_crop_score,
        "qr_like_weight": args.qr_like_weight,
        "reject_no_valid_refinement": bool(args.reject_no_valid_refinement),
        "mean_motion_response": round(float(np.mean(motion_responses)), 4) if motion_responses else 0.0,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
