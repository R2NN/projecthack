"""Train RF-DETR for Lenta price tag detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=sorted(MODEL_CLASSES), default="small")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--class-name", default="price_tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "train_config.json"
    config_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    model_cls = MODEL_CLASSES[args.model_size]
    model = model_cls(num_classes=1, resolution=args.resolution)
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        dataset_file="roboflow",
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        class_names=[args.class_name],
        run_test=args.run_test,
        tensorboard=False,
        wandb=False,
    )


if __name__ == "__main__":
    main()
