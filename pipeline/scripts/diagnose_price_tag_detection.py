"""Diagnose which labeled price tags are missed by the detector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from predict_rfdetr_price_tags import box_iou_matrix, filter_predictions, padded_crop, predict_image, save_rgb


MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def parse_number(value: str) -> float:
    return float(str(value).replace("\xa0", "").replace(",", ".").strip())


def find_data_root(workspace: Path) -> Path:
    for child in workspace.iterdir():
        if child.is_dir() and (child / "sample.csv").exists():
            return child
    raise FileNotFoundError("Could not find extracted data directory with sample.csv")


def read_labeled_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_frame(video_path: Path, timestamp_ms: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {timestamp_ms} ms from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def row_box(row: dict[str, str]) -> list[float]:
    return [
        parse_number(row["x_min"]),
        parse_number(row["y_min"]),
        parse_number(row["x_max"]),
        parse_number(row["y_max"]),
    ]


def greedy_match(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_threshold: float) -> tuple[dict[int, int], np.ndarray]:
    ious = box_iou_matrix(gt_boxes, pred_boxes)
    matches: dict[int, int] = {}
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return matches, ious

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    while True:
        gt_idx, pred_idx = np.unravel_index(np.argmax(ious), ious.shape)
        best_iou = float(ious[gt_idx, pred_idx])
        if best_iou < iou_threshold:
            break
        if gt_idx not in used_gt and pred_idx not in used_pred:
            matches[int(gt_idx)] = int(pred_idx)
            used_gt.add(int(gt_idx))
            used_pred.add(int(pred_idx))
        ious[gt_idx, pred_idx] = -1.0
    return matches, box_iou_matrix(gt_boxes, pred_boxes)


def classify_issue(
    matched: bool,
    nearest_filtered_iou: float,
    nearest_raw_iou: float,
    nearest_raw_score: float,
    threshold: float,
) -> str:
    if matched:
        return "matched"
    if nearest_raw_iou >= 0.50 and nearest_raw_score < threshold:
        return "low_confidence"
    if nearest_raw_iou >= 0.50:
        return "post_filter_or_nms_removed"
    if nearest_filtered_iou >= 0.30:
        return "localization_shift"
    if nearest_raw_iou >= 0.25:
        return "weak_localization_or_low_confidence"
    return "not_detected"


def draw_overlay(
    image_rgb: np.ndarray,
    rows: list[dict[str, str]],
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    matches: dict[int, int],
    max_predictions: int,
) -> np.ndarray:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    matched_pred_ids = set(matches.values())
    for gt_idx, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
        matched = gt_idx in matches
        color = (35, 220, 80) if matched else (255, 45, 45)
        label = f"GT{gt_idx + 1} {'ok' if matched else 'miss'}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=7)
        draw.text((x1, max(0, y1 - 16)), label, fill=color, font=font)
        product = rows[gt_idx].get("product_name", "")[:35]
        if not matched:
            draw.text((x1, y2 + 4), product, fill=color, font=font)

    order = np.argsort(-pred_scores)[:max_predictions]
    for rank, pred_idx in enumerate(order, start=1):
        x1, y1, x2, y2 = [float(value) for value in pred_boxes[pred_idx]]
        color = (255, 210, 35) if int(pred_idx) in matched_pred_ids else (60, 160, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1, y2 + 18), f"P{rank} {float(pred_scores[pred_idx]):.2f}", fill=color, font=font)

    return np.asarray(image)


def make_missed_sheet(crop_paths: list[Path], output_path: Path, tile_width: int = 260, tile_height: int = 210) -> None:
    if not crop_paths:
        return
    columns = 4
    rows_count = int(np.ceil(len(crop_paths) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, path in enumerate(crop_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_width - 18, tile_height - 42), Image.Resampling.LANCZOS)
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_height
        sheet.paste(image, (x + 9, y + 28))
        draw.text((x + 8, y + 8), path.stem[:35], fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-name", default="43_15")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=sorted(MODEL_CLASSES), default="small")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--raw-threshold", type=float, default=0.03)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=320)
    parser.add_argument("--min-area", type=float, default=8_000.0)
    parser.add_argument("--max-area", type=float, default=180_000.0)
    parser.add_argument("--min-aspect", type=float, default=0.25)
    parser.add_argument("--max-aspect", type=float, default=1.60)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--crop-pad", type=float, default=0.12)
    parser.add_argument("--max-draw-predictions", type=int, default=80)
    args = parser.parse_args()

    workspace = Path.cwd()
    data_root = find_data_root(workspace)
    video_dir = data_root / args.video_name
    csv_path = video_dir / f"{args.video_name}.csv"
    video_path = video_dir / f"{args.video_name}.mp4"
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_timestamp: dict[int, list[dict[str, str]]] = {}
    for row in read_labeled_rows(csv_path):
        timestamp_ms = int(round(parse_number(row["frame_timestamp"])))
        rows_by_timestamp.setdefault(timestamp_ms, []).append(row)

    model = MODEL_CLASSES[args.variant](pretrain_weights=str(Path(args.checkpoint).resolve()), num_classes=1, device="cuda")

    gt_rows = []
    frame_rows = []
    missed_crop_paths = []
    issue_counts: dict[str, int] = {}
    total_gt = 0
    total_pred = 0
    total_matched = 0
    skipped_timestamps = 0
    skipped_gt = 0

    for timestamp_ms, rows in sorted(rows_by_timestamp.items()):
        try:
            image_rgb = read_frame(video_path, timestamp_ms)
        except RuntimeError as error:
            skipped_timestamps += 1
            skipped_gt += len(rows)
            frame_rows.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "gt_count": len(rows),
                    "pred_count": 0,
                    "matched_iou50": 0,
                    "precision_iou50": "",
                    "recall_iou50": "",
                    "skipped": 1,
                    "skip_reason": str(error),
                }
            )
            continue
        height, width = image_rgb.shape[:2]
        gt_boxes = np.asarray([row_box(row) for row in rows], dtype=np.float32)
        raw_boxes, raw_scores = predict_image(
            model,
            image_rgb,
            threshold=min(args.raw_threshold, args.threshold),
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )

        threshold_mask = raw_scores >= args.threshold
        threshold_boxes = raw_boxes[threshold_mask]
        threshold_scores = raw_scores[threshold_mask]
        order = np.argsort(-threshold_scores)
        threshold_boxes = threshold_boxes[order]
        threshold_scores = threshold_scores[order]
        pred_boxes, pred_scores = filter_predictions(
            threshold_boxes,
            threshold_scores,
            min_area=args.min_area,
            max_area=args.max_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            nms_iou=args.nms_iou,
        )

        matches, filtered_ious = greedy_match(gt_boxes.copy(), pred_boxes, args.iou_threshold)
        raw_ious = box_iou_matrix(gt_boxes, raw_boxes)
        matched_count = len(matches)
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        total_matched += matched_count

        overlay = draw_overlay(
            image_rgb=image_rgb,
            rows=rows,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            matches=matches,
            max_predictions=args.max_draw_predictions,
        )
        save_rgb(output_dir / f"frame_{timestamp_ms:06d}_overlay.jpg", overlay)

        frame_rows.append(
            {
                "timestamp_ms": timestamp_ms,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "matched_iou50": matched_count,
                "precision_iou50": f"{matched_count / len(pred_boxes):.4f}" if len(pred_boxes) else "0.0000",
                "recall_iou50": f"{matched_count / len(gt_boxes):.4f}" if len(gt_boxes) else "0.0000",
                "skipped": 0,
                "skip_reason": "",
            }
        )

        for gt_idx, row in enumerate(rows):
            gt_box = gt_boxes[gt_idx]
            x1, y1, x2, y2 = [float(value) for value in gt_box]
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            matched = gt_idx in matches
            matched_pred_idx = matches.get(gt_idx)
            matched_iou = 0.0
            matched_score = 0.0
            if matched_pred_idx is not None:
                matched_iou = float(filtered_ious[gt_idx, matched_pred_idx])
                matched_score = float(pred_scores[matched_pred_idx])

            nearest_filtered_iou = 0.0
            nearest_filtered_score = 0.0
            if len(pred_boxes):
                nearest_pred_idx = int(np.argmax(filtered_ious[gt_idx]))
                nearest_filtered_iou = float(filtered_ious[gt_idx, nearest_pred_idx])
                nearest_filtered_score = float(pred_scores[nearest_pred_idx])

            nearest_raw_iou = 0.0
            nearest_raw_score = 0.0
            if len(raw_boxes):
                nearest_raw_idx = int(np.argmax(raw_ious[gt_idx]))
                nearest_raw_iou = float(raw_ious[gt_idx, nearest_raw_idx])
                nearest_raw_score = float(raw_scores[nearest_raw_idx])

            issue = classify_issue(
                matched=matched,
                nearest_filtered_iou=nearest_filtered_iou,
                nearest_raw_iou=nearest_raw_iou,
                nearest_raw_score=nearest_raw_score,
                threshold=args.threshold,
            )
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

            crop_path = ""
            if not matched:
                crop = padded_crop(image_rgb, gt_box, args.crop_pad)
                if crop is not None:
                    crop_path_obj = output_dir / "missed_gt_crops" / (
                        f"ts_{timestamp_ms:06d}_gt_{gt_idx + 1:02d}_{issue}.jpg"
                    )
                    save_rgb(crop_path_obj, crop)
                    missed_crop_paths.append(crop_path_obj)
                    crop_path = str(crop_path_obj)

            gt_rows.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "gt_index": gt_idx + 1,
                    "matched": int(matched),
                    "issue": issue,
                    "product_name": row.get("product_name", ""),
                    "barcode": row.get("barcode", ""),
                    "id_sku": row.get("id_sku", ""),
                    "price_card": row.get("price_card", ""),
                    "x_min": f"{x1:.1f}",
                    "y_min": f"{y1:.1f}",
                    "x_max": f"{x2:.1f}",
                    "y_max": f"{y2:.1f}",
                    "width": f"{box_width:.1f}",
                    "height": f"{box_height:.1f}",
                    "area": f"{box_area:.1f}",
                    "aspect": f"{box_width / max(1.0, box_height):.4f}",
                    "center_x_norm": f"{((x1 + x2) * 0.5) / width:.4f}",
                    "center_y_norm": f"{((y1 + y2) * 0.5) / height:.4f}",
                    "matched_iou": f"{matched_iou:.4f}",
                    "matched_score": f"{matched_score:.4f}",
                    "nearest_filtered_iou": f"{nearest_filtered_iou:.4f}",
                    "nearest_filtered_score": f"{nearest_filtered_score:.4f}",
                    "nearest_raw_iou": f"{nearest_raw_iou:.4f}",
                    "nearest_raw_score": f"{nearest_raw_score:.4f}",
                    "missed_crop": crop_path,
                }
            )

    with (output_dir / "per_gt.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(gt_rows[0].keys()) if gt_rows else ["timestamp_ms"])
        writer.writeheader()
        writer.writerows(gt_rows)

    with (output_dir / "per_frame.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["timestamp_ms"])
        writer.writeheader()
        writer.writerows(frame_rows)

    make_missed_sheet(missed_crop_paths, output_dir / "missed_gt_contact_sheet.jpg")

    summary = {
        "video_name": args.video_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": args.threshold,
        "raw_threshold": args.raw_threshold,
        "iou_threshold": args.iou_threshold,
        "gt_count": total_gt,
        "pred_count": total_pred,
        "matched_iou50": total_matched,
        "precision_iou50": round(total_matched / total_pred, 4) if total_pred else 0.0,
        "recall_iou50": round(total_matched / total_gt, 4) if total_gt else 0.0,
        "skipped_timestamps": skipped_timestamps,
        "skipped_gt": skipped_gt,
        "issue_counts": issue_counts,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
