from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def resolve_path(raw_value: str, root: Path = ROOT) -> Path:
    path = Path(raw_value)
    return path if path.is_absolute() else root / path


def command_exists(command: str) -> bool:
    return bool(Path(command).exists() or shutil.which(command))


def resolve_command(command: str) -> str:
    if Path(command).exists():
        return str(Path(command))
    return shutil.which(command) or command


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def tesseract_languages(tesseract: str, tessdata_dir: Path | None) -> set[str]:
    env = os.environ.copy()
    if tessdata_dir is not None:
        env["TESSDATA_PREFIX"] = str(tessdata_dir)
    try:
        result = subprocess.run(
            [resolve_command(tesseract), "--list-langs"],
            text=True,
            capture_output=True,
            timeout=8,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip() and "List of" not in line}


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def easyocr_model_status() -> dict[str, object]:
    module_root = Path(
        env_first(
            "EASYOCR_MODULE_PATH",
            "MODULE_PATH",
            default=str(Path.home() / ".EasyOCR"),
        )
    )
    model_dir = module_root / "model"
    expected = ["craft_mlt_25k.pth", "cyrillic_g2.pth"]
    files = {name: (model_dir / name).is_file() for name in expected}
    return {
        "ok": all(files.values()),
        "module_path": str(module_root),
        "model_dir": str(model_dir),
        "files": files,
    }


def add_check(
    checks: dict[str, dict[str, object]],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
    name: str,
    ok: bool,
    code: str,
    message: str,
    warning: bool = False,
    **extra: object,
) -> None:
    checks[name] = {"ok": ok, "code": "" if ok else code, "message": message, **extra}
    if ok:
        return
    target = warnings if warning else errors
    target.append({"code": code, "message": message})


def main() -> None:
    load_env_file(Path(os.environ.get("ENV_FILE", ROOT / ".env")))

    artifacts = resolve_path(env_first("ARTIFACTS_DIR", "LENTA_ARTIFACTS_ROOT", default=str(ROOT / "artifacts")))
    runtime = resolve_path(env_first("RUNTIME_DIR", "LENTA_WEB_RUNTIME_ROOT", default=str(ROOT / "runtime")))
    model = resolve_path(
        env_first(
            "MODEL_PATH",
            "LENTA_DETECTOR_CHECKPOINT",
            "DETECTOR_CHECKPOINT",
            default=str(artifacts / "models" / "rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth"),
        )
    )
    catalog = resolve_path(env_first("CATALOG_PATH", "LENTA_CATALOG_PATH", default=str(artifacts / "data" / "db_hack.csv")))
    sample_csv = resolve_path(env_first("SAMPLE_CSV", default=str(artifacts / "data" / "sample.csv")))
    templates = resolve_path(env_first("SPECIAL_SYMBOL_TEMPLATE_DIR", default=str(artifacts / "special_symbol_templates" / "full_tags")))
    pipeline_script = resolve_path(env_first("LENTA_PIPELINE_SCRIPT", default=str(ROOT / "pipeline" / "run_inference.ps1")))
    powershell = env_first("LENTA_POWERSHELL", default="powershell.exe" if os.name == "nt" else "pwsh")
    tesseract = env_first("LENTA_TESSERACT_EXE", "TESSERACT_EXE", default="tesseract")
    tessdata_raw = env_first("LENTA_TESSDATA_DIR", "TESSDATA_DIR", default="")
    tessdata = resolve_path(tessdata_raw) if tessdata_raw else None
    requested_device = env_first("INFERENCE_DEVICE", "LENTA_INFERENCE_DEVICE", default="auto").lower()
    min_free_gb = float(env_first("MIN_FREE_GB", default="5"))

    checks: dict[str, dict[str, object]] = {}
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    runtime.mkdir(parents=True, exist_ok=True)
    try:
        probe = runtime / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        runtime_writable = True
    except OSError:
        runtime_writable = False
    disk = shutil.disk_usage(runtime if runtime.exists() else ROOT)
    free_gb = round(disk.free / (1024**3), 2)

    add_check(checks, errors, warnings, "model", model.is_file(), "MODEL_NOT_FOUND", f"put checkpoint to {model}", path=str(model))
    add_check(checks, errors, warnings, "catalog", catalog.is_file(), "CATALOG_NOT_FOUND", f"put db_hack.csv to {catalog}", path=str(catalog))
    add_check(checks, errors, warnings, "sample_csv", sample_csv.is_file(), "SAMPLE_CSV_NOT_FOUND", f"put sample.csv to {sample_csv}", path=str(sample_csv))
    add_check(checks, errors, warnings, "special_symbol_templates", templates.is_dir(), "TEMPLATE_DIR_NOT_FOUND", f"put templates to {templates}", path=str(templates))
    add_check(checks, errors, warnings, "pipeline_script", pipeline_script.is_file(), "PIPELINE_SCRIPT_NOT_FOUND", f"pipeline script is missing: {pipeline_script}", path=str(pipeline_script))
    add_check(
        checks,
        errors,
        warnings,
        "runtime",
        runtime_writable and free_gb >= min_free_gb,
        "RUNTIME_NOT_READY",
        f"runtime must be writable and have at least {min_free_gb:g} GB free",
        path=str(runtime),
        free_gb=free_gb,
    )
    add_check(checks, errors, warnings, "powershell", command_exists(powershell), "POWERSHELL_NOT_FOUND", f"PowerShell executable is missing: {powershell}", executable=powershell)
    add_check(
        checks,
        errors,
        warnings,
        "ffmpeg",
        command_exists("ffmpeg"),
        "FFMPEG_NOT_FOUND",
        "ffmpeg is missing from PATH; Docker image installs it, local OpenCV may still read MP4 without it",
        warning=True,
    )
    add_check(checks, errors, warnings, "tesseract", command_exists(tesseract), "TESSERACT_NOT_FOUND", f"install tesseract or set TESSERACT_EXE: {tesseract}", executable=tesseract)

    languages = tesseract_languages(tesseract, tessdata) if command_exists(tesseract) else set()
    add_check(
        checks,
        errors,
        warnings,
        "tessdata",
        {"rus", "eng"}.issubset(languages),
        "TESSERACT_NOT_READY",
        "rus+eng tessdata missing",
        tessdata_dir=str(tessdata or ""),
        languages=sorted(languages),
    )

    module_names = ["cv2", "numpy", "pandas", "zxingcpp", "rfdetr", "easyocr", "rapidocr", "sklearn", "psutil"]
    modules = {name: module_exists(name) for name in module_names}
    add_check(
        checks,
        errors,
        warnings,
        "python_ocr_stack",
        all(modules.values()),
        "OCR_STACK_NOT_READY",
        "install pipeline requirements in this environment",
        modules=modules,
        python=sys.executable,
    )
    easyocr_status = easyocr_model_status()
    easyocr_ready = bool(easyocr_status.pop("ok"))
    add_check(
        checks,
        errors,
        warnings,
        "easyocr_models",
        easyocr_ready,
        "EASYOCR_MODEL_NOT_FOUND",
        f"preload EasyOCR rus+eng models to {easyocr_status['model_dir']} or rebuild the Docker image",
        **easyocr_status,
    )

    cuda_available = torch_cuda_available()
    resolved_device = "cuda" if requested_device == "cuda" or (requested_device == "auto" and cuda_available) else "cpu"
    if requested_device == "cuda" and not cuda_available:
        add_check(
            checks,
            errors,
            warnings,
            "compute",
            False,
            "GPU_NOT_AVAILABLE",
            "CUDA was requested but torch.cuda.is_available() is false",
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
    else:
        add_check(
            checks,
            errors,
            warnings,
            "compute",
            True,
            "",
            "compute device resolved",
            requested_device=requested_device,
            resolved_device=resolved_device,
            cuda_available=cuda_available,
        )
        if resolved_device == "cpu":
            warnings.append({"code": "CPU_MODE", "message": "CPU mode is available, but inference will be slower than GPU."})

    payload = {
        "ok": not errors,
        "app": "shelf-vision",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
