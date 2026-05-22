from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def command_exists(command: str) -> bool:
    return bool(shutil.which(command) or Path(command).exists())


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = Path(os.environ.get("RUNTIME_DIR", root / "runtime"))
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", root / "artifacts"))
    checks: dict[str, object] = {
        "python": sys.executable,
        "runtime_dir": str(runtime),
        "artifacts_dir": str(artifacts),
        "free_space_runtime_gb": round(shutil.disk_usage(runtime if runtime.exists() else root).free / (1024**3), 2),
        "powershell": command_exists(os.environ.get("LENTA_POWERSHELL", "powershell.exe" if os.name == "nt" else "pwsh")),
        "tesseract": command_exists(os.environ.get("LENTA_TESSERACT_EXE", os.environ.get("TESSERACT_EXE", "tesseract"))),
        "ffmpeg": command_exists("ffmpeg"),
        "modules": {name: module_exists(name) for name in ["cv2", "numpy", "pandas", "zxingcpp", "rfdetr"]},
    }
    artifact_check = subprocess.run(
        [sys.executable, str(root / "tools" / "check_artifacts.py"), "--root", str(root)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        checks["artifacts"] = json.loads(artifact_check.stdout)
    except json.JSONDecodeError:
        checks["artifacts"] = {"ok": False, "raw": artifact_check.stdout}
    ok = bool(checks["powershell"] and checks["tesseract"] and all(checks["modules"].values()) and checks["artifacts"]["ok"])
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
