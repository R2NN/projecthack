from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PaddleOCR recognizer fine-tune config.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--save-model-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", default="")
    parser.add_argument("--epoch-num", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00012)
    parser.add_argument("--max-text-length", type=int, default=40)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-epoch-step", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_nrtr_max_text_length(config: dict[str, Any], value: int) -> None:
    for head in config["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:
            head["NRTRHead"]["max_text_length"] = value


def without_rec_con_aug(transforms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in transforms if "RecConAug" not in item]


def main() -> None:
    args = parse_args()
    with args.base_config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_dir = args.dataset_dir.resolve()
    save_model_dir = args.save_model_dir.resolve()

    config["Global"]["use_gpu"] = not args.cpu
    config["Global"]["epoch_num"] = args.epoch_num
    config["Global"]["print_batch_step"] = 10
    config["Global"]["save_model_dir"] = str(save_model_dir)
    config["Global"]["save_epoch_step"] = args.save_epoch_step
    config["Global"]["eval_batch_step"] = [0, args.eval_every_steps]
    config["Global"]["pretrained_model"] = args.pretrained_model
    config["Global"]["character_dict_path"] = str(dataset_dir / "dict.txt")
    config["Global"]["max_text_length"] = args.max_text_length
    config["Global"]["use_visualdl"] = False
    config["Global"]["distributed"] = False
    config["Global"]["d2s_train_image_shape"] = [3, 48, 320]

    config["Optimizer"]["lr"]["learning_rate"] = args.learning_rate
    warmup_epoch = config["Optimizer"]["lr"].get("warmup_epoch")
    if isinstance(warmup_epoch, int) and warmup_epoch >= args.epoch_num:
        config["Optimizer"]["lr"]["warmup_epoch"] = max(0, args.epoch_num // 3)

    set_nrtr_max_text_length(config, args.max_text_length)

    config["Train"]["dataset"]["data_dir"] = str(dataset_dir)
    config["Train"]["dataset"]["label_file_list"] = [str(dataset_dir / "train.txt")]
    config["Train"]["dataset"]["transforms"] = without_rec_con_aug(config["Train"]["dataset"]["transforms"])
    config["Train"]["sampler"]["first_bs"] = args.batch_size
    config["Train"]["loader"]["batch_size_per_card"] = args.batch_size
    config["Train"]["loader"]["num_workers"] = args.num_workers

    config["Eval"]["dataset"]["data_dir"] = str(dataset_dir)
    config["Eval"]["dataset"]["label_file_list"] = [str(dataset_dir / "val.txt")]
    config["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    config["Eval"]["loader"]["num_workers"] = args.num_workers

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    print(args.output_config)


if __name__ == "__main__":
    main()
