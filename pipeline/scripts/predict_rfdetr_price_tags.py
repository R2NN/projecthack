"""Visualize RF-DETR price tag detections on a prepared COCO dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


@dataclass(frozen=True)
class GroundTruth:
    image_id: int
    file_name: str
    boxes_xyxy: np.ndarray


def load_split(split_dir: Path) -> list[GroundTruth]:
    annotations_path = split_dir / "_annotations.coco.json"
    with annotations_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    image_by_id = {image["id"]: image["file_name"] for image in coco["images"]}
    boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in image_by_id}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        boxes_by_image[ann["image_id"]].append([x, y, x + w, y + h])

    return [
        GroundTruth(
            image_id=image_id,
            file_name=image_by_id[image_id],
            boxes_xyxy=np.asarray(boxes_by_image[image_id], dtype=np.float32),
        )
        for image_id in sorted(image_by_id)
    ]


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match_ious(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> list[float]:
    ious = box_iou_matrix(gt_boxes, pred_boxes)
    matches: list[float] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    while ious.size:
        gt_idx, pred_idx = np.unravel_index(np.argmax(ious), ious.shape)
        best = float(ious[gt_idx, pred_idx])
        if best <= 0:
            break
        if gt_idx not in used_gt and pred_idx not in used_pred:
            matches.append(best)
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
        ious[gt_idx, pred_idx] = -1

    return matches


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.asarray([], dtype=np.int64)

    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        ious = box_iou_matrix(boxes[[current]], boxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def filter_predictions(
    boxes: np.ndarray,
    scores: np.ndarray,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores

    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas = widths * heights
    aspects = np.divide(widths, heights, out=np.zeros_like(widths), where=heights > 0)
    mask = (areas >= min_area) & (areas <= max_area) & (aspects >= min_aspect) & (aspects <= max_aspect)
    boxes = boxes[mask]
    scores = scores[mask]
    keep = nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep]


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("tile overlap must be smaller than tile size")
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_image(
    model,
    image_rgb: np.ndarray,
    threshold: float,
    tile_size: int,
    tile_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tile_size <= 0:
        image = Image.fromarray(image_rgb)
        detections = model.predict(image, threshold=threshold)
        return np.asarray(detections.xyxy, dtype=np.float32), np.asarray(detections.confidence, dtype=np.float32)

    height, width = image_rgb.shape[:2]
    boxes = []
    scores = []
    for y0 in tile_starts(height, tile_size, tile_overlap):
        for x0 in tile_starts(width, tile_size, tile_overlap):
            tile = image_rgb[y0 : min(height, y0 + tile_size), x0 : min(width, x0 + tile_size)]
            detections = model.predict(Image.fromarray(tile), threshold=threshold)
            tile_boxes = np.asarray(detections.xyxy, dtype=np.float32)
            tile_scores = np.asarray(detections.confidence, dtype=np.float32)
            if len(tile_boxes) == 0:
                continue
            tile_boxes[:, [0, 2]] += x0
            tile_boxes[:, [1, 3]] += y0
            boxes.append(tile_boxes)
            scores.append(tile_scores)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(boxes, axis=0), np.concatenate(scores, axis=0)


def padded_crop(image_rgb: np.ndarray, box: np.ndarray, pad_ratio: float) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    x1 = int(max(0, round(x1 - pad_x)))
    y1 = int(max(0, round(y1 - pad_y)))
    x2 = int(min(width, round(x2 + pad_x)))
    y2 = int(min(height, round(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image_rgb[y1:y2, x1:x2]


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def draw_overlay(
    image_rgb: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    confidences: np.ndarray,
    max_draw: int,
) -> np.ndarray:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for idx, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 60, 60), width=6)
        draw.text((x1, max(0, y1 - 14)), f"GT {idx + 1}", fill=(255, 60, 60), font=font)

    order = np.argsort(-confidences)[:max_draw]
    for rank, det_idx in enumerate(order, start=1):
        x1, y1, x2, y2 = [float(v) for v in pred_boxes[det_idx]]
        score = float(confidences[det_idx])
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 90), width=5)
        draw.text((x1, y2 + 4), f"P{rank} {score:.2f}", fill=(0, 220, 90), font=font)

    return np.asarray(image)


def make_contact_sheet(images: list[np.ndarray], output_path: Path, width: int = 1000) -> None:
    if not images:
        return
    resized = []
    for image in images:
        h, w = image.shape[:2]
        scale = width / w
        resized.append(cv2.resize(image, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate(resized, axis=0)
    save_rgb(output_path, sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/rfdetr_25_12_20_price_tag")
    parser.add_argument("--checkpoint", default="runs/rfdetr_small_price_tag_25_12_20/checkpoint_best_regular.pth")
    parser.add_argument("--output", default="analysis_outputs/rfdetr_small_price_tag_25_12_20")
    parser.add_argument("--variant", choices=sorted(MODEL_CLASSES), default="small")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-draw", type=int, default=20)
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--min-area", type=float, default=8_000.0)
    parser.add_argument("--max-area", type=float, default=180_000.0)
    parser.add_argument("--min-aspect", type=float, default=0.25)
    parser.add_argument("--max-aspect", type=float, default=1.60)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--skip-crops", action="store_true")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=320)
    parser.add_argument("--splits", default="train,valid,test")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MODEL_CLASSES[args.variant](pretrain_weights=str(checkpoint_path), num_classes=1, device="cuda")

    summary: dict[str, dict[str, float | int]] = {}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        rows = load_split(split_dir)
        overlays: list[np.ndarray] = []
        split_matches: list[float] = []
        total_gt = 0
        total_pred = 0

        for row in rows:
            image_path = split_dir / row.file_name
            image = Image.open(image_path).convert("RGB")
            image_rgb = np.asarray(image)

            pred_boxes, confidences = predict_image(
                model,
                image_rgb,
                threshold=args.threshold,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
            order = np.argsort(-confidences)
            pred_boxes = pred_boxes[order]
            confidences = confidences[order]
            pred_boxes, confidences = filter_predictions(
                pred_boxes,
                confidences,
                min_area=args.min_area,
                max_area=args.max_area,
                min_aspect=args.min_aspect,
                max_aspect=args.max_aspect,
                nms_iou=args.nms_iou,
            )

            matches = greedy_match_ious(row.boxes_xyxy, pred_boxes)
            split_matches.extend(matches)
            total_gt += len(row.boxes_xyxy)
            total_pred += len(pred_boxes)

            overlay = draw_overlay(image_rgb, row.boxes_xyxy, pred_boxes, confidences, args.max_draw)
            overlays.append(overlay)
            save_rgb(output_dir / split / f"{Path(row.file_name).stem}_overlay.jpg", overlay)

            if not args.skip_crops:
                crop_dir = output_dir / "crops" / split / Path(row.file_name).stem
                for det_idx, box in enumerate(pred_boxes[: args.max_draw], start=1):
                    crop = padded_crop(image_rgb, box, args.crop_pad)
                    if crop is None:
                        continue
                    score = float(confidences[det_idx - 1])
                    base_name = f"det_{det_idx:02d}_{score:.3f}"
                    save_rgb(crop_dir / f"{base_name}_raw.jpg", crop)
                    rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    save_rgb(crop_dir / f"{base_name}_rot_ccw.jpg", rotated)

        mean_iou = float(np.mean(split_matches)) if split_matches else 0.0
        true_positive_50 = sum(iou >= 0.5 for iou in split_matches)
        recall_50 = true_positive_50 / total_gt if total_gt else 0.0
        precision_50 = true_positive_50 / total_pred if total_pred else 0.0
        summary[split] = {
            "images": len(rows),
            "gt_boxes": total_gt,
            "pred_boxes": total_pred,
            "matched_boxes": len(split_matches),
            "mean_matched_iou": round(mean_iou, 4),
            "recall_at_iou_0_50": round(recall_50, 4),
            "precision_at_iou_0_50": round(precision_50, 4),
        }
        make_contact_sheet(overlays, output_dir / f"{split}_contact_sheet.jpg")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps({"output": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
