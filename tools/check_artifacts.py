from __future__ import annotations

import os
from pathlib import Path


def exists(path: Path) -> str:
    return "ok" if path.exists() else "missing"


def main() -> int:
    pipeline_root = Path(os.environ.get("LENTA_PIPELINE_ROOT", "pipeline"))
    catalog_path = Path(os.environ.get("LENTA_CATALOG_PATH", "artifacts/db_hack.csv"))
    checkpoint_path = Path(
        os.environ.get(
            "LENTA_DETECTOR_CHECKPOINT",
            "artifacts/models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth",
        )
    )
    script_path = Path(os.environ.get("LENTA_PIPELINE_SCRIPT", str(pipeline_root / "run_inference_no_ensemble.ps1")))

    checks = {
        "pipeline": pipeline_root,
        "inference script": script_path,
        "catalog": catalog_path,
        "detector checkpoint": checkpoint_path,
    }
    failed = False
    for name, path in checks.items():
        status = exists(path)
        print(f"{name:20} {status:8} {path}")
        failed = failed or status != "ok"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
