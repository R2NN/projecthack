from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def resolve(root: Path, value: str | None, default: Path) -> Path:
    raw = value or str(default)
    path = Path(raw)
    return path if path.is_absolute() else root / path


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check required runtime artifacts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    artifacts = resolve(root, os.environ.get("ARTIFACTS_DIR"), root / "artifacts")
    required = {
        "detector_checkpoint": resolve(
            root,
            env_first("MODEL_PATH", "LENTA_DETECTOR_CHECKPOINT", "DETECTOR_CHECKPOINT"),
            artifacts / "models" / "rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth",
        ),
        "catalog_csv": resolve(root, env_first("CATALOG_PATH", "LENTA_CATALOG_PATH"), artifacts / "data" / "db_hack.csv"),
        "sample_csv": resolve(root, env_first("SAMPLE_CSV", "LENTA_SAMPLE_CSV"), artifacts / "data" / "sample.csv"),
        "special_symbol_templates": resolve(
            root,
            env_first("SPECIAL_SYMBOL_TEMPLATE_DIR", "LENTA_SPECIAL_SYMBOL_TEMPLATE_DIR"),
            artifacts / "special_symbol_templates" / "full_tags",
        ),
    }
    results = {}
    ok = True
    for name, path in required.items():
        exists = path.exists()
        if name == "special_symbol_templates":
            count = len(list(path.glob("track_*_rank_01_*_full.jpg"))) if exists else 0
            exists = exists and count >= 2
            results[name] = {"path": str(path), "exists": exists, "template_count": count}
        else:
            results[name] = {"path": str(path), "exists": exists, "bytes": path.stat().st_size if exists else 0}
        ok = ok and exists
    print(json.dumps({"ok": ok, "artifacts": results}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
