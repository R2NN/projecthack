from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from email import policy
from email.parser import BytesParser
from html import escape
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from worker_queue import enqueue_task, get_task_by_source_job, init_queue_db, queue_summary


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent
A_PIPELINE_ROOT = Path("A:/lenta_pipeline/handoff_v19_speed_full_pipeline_20260518_040946")
LOCAL_PIPELINE_ROOT = PROJECT_ROOT / "handoff_v19_speed_full_pipeline_20260518_040946"
PIPELINE_ROOT = Path(os.environ.get("LENTA_PIPELINE_ROOT", str(A_PIPELINE_ROOT if A_PIPELINE_ROOT.exists() else LOCAL_PIPELINE_ROOT)))
PIPELINE_SCRIPT = Path(os.environ.get("LENTA_PIPELINE_SCRIPT", str(PIPELINE_ROOT / "run_inference_no_ensemble.ps1")))
PIPELINE_POWERSHELL = os.environ.get("LENTA_POWERSHELL", "powershell.exe" if sys.platform == "win32" else "pwsh")
DEFAULT_PIPELINE_PYTHON = Path("A:/rfdetr_envs/lenta-rfdetr-gpu/Scripts/python.exe")
PIPELINE_PYTHON = os.environ.get(
    "LENTA_PIPELINE_PYTHON",
    str(DEFAULT_PIPELINE_PYTHON if sys.platform == "win32" and DEFAULT_PIPELINE_PYTHON.exists() else sys.executable),
)
DEFAULT_TESSERACT_EXE = Path("A:/tesseract_env/Library/bin/tesseract.exe")
DEFAULT_TESSDATA_DIR = Path("A:/tesseract_env/Library/share/tessdata")
TESSERACT_EXE = os.environ.get(
    "LENTA_TESSERACT_EXE",
    str(
        DEFAULT_TESSERACT_EXE
        if sys.platform == "win32" and DEFAULT_TESSERACT_EXE.exists()
        else shutil.which("tesseract") or "/usr/bin/tesseract"
    ),
)
TESSDATA_DIR = Path(
    os.environ.get(
        "LENTA_TESSDATA_DIR",
        str(
            DEFAULT_TESSDATA_DIR
            if sys.platform == "win32" and DEFAULT_TESSDATA_DIR.exists()
            else "/usr/share/tesseract-ocr/5/tessdata"
        ),
    )
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "LENTA_DETECTOR_CHECKPOINT",
        str(PIPELINE_ROOT / "models" / "rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth"),
    )
)
FALLBACK_CHECKPOINT = PIPELINE_ROOT / "models" / "rfdetr_small_price_tag_except_26_12_20_tiled1280_e8_checkpoint_best_total.pth"
RUNTIME_ROOT = Path(os.environ.get("LENTA_WEB_RUNTIME_ROOT", "A:/lenta_web_runtime" if Path("A:/").exists() else str(WEB_ROOT / "runtime")))
JOBS_ROOT = Path(os.environ.get("LENTA_WEB_JOBS_ROOT", str(RUNTIME_ROOT / "jobs")))
UNCERTAIN_ROOT = Path(os.environ.get("LENTA_UNCERTAIN_ROOT", str(RUNTIME_ROOT / "uncertain_predictions")))
RETRAIN_CONFIG_PATH = Path(os.environ.get("LENTA_RETRAIN_CONFIG", str(RUNTIME_ROOT / "retrain_config.json")))
RESULTS_DB = Path(os.environ.get("LENTA_RESULTS_DB", str(RUNTIME_ROOT / "lenta_results.sqlite")))
WORKER_QUEUE_DB = Path(os.environ.get("LENTA_WORKER_QUEUE_DB", str(RUNTIME_ROOT / "worker_queue.sqlite")))
JOB_EXECUTION_MODE = os.environ.get("LENTA_JOB_EXECUTION_MODE", "local").strip().lower()
QUEUE_MODE_ENABLED = JOB_EXECUTION_MODE in {"queue", "worker", "workers", "distributed"}
A_CATALOG = Path("A:/lenta_data/db_hack.csv")
DEFAULT_CATALOG = Path(os.environ.get("LENTA_CATALOG_PATH", str(A_CATALOG if A_CATALOG.exists() else PROJECT_ROOT / "db_hack.csv")))
SAMPLE_CSV = PROJECT_ROOT / "Данные" / "sample.csv"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
AUTO_RETRAIN_DEFAULT_ENABLED = False
REVIEW_MAX_ITEMS = 160
UNCERTAIN_MAX_ITEMS = 240

STEP_DEFS = [
    ("01_detection_tracking", "Детекция и трекинг ценников", 121.0),
    ("02_export_ocr_zones", "Подготовка OCR-зон", 41.0),
    ("03_numeric_ocr", "OCR цен, QR и barcode", 294.0),
    ("04_tesseract_product_name", "OCR названий товаров", 300.0),
    ("05_crop_quality_reselect", "Выбор лучших кропов", 7.0),
    ("06_catalog_recovery_db_hack", "Восстановление по db_hack", 126.0),
    ("07_fill_aux_barcode", "Заполнение barcode и QR-полей", 42.0),
    ("08_deduplicate_rows", "Удаление дублей", 1.0),
    ("09_export_final_submission", "Сборка CSV", 1.0),
]
STEP_IDS = [step_id for step_id, _, _ in STEP_DEFS]
STEP_LABELS = {step_id: label for step_id, label, _ in STEP_DEFS}
STEP_AVG_SECONDS = {step_id: seconds for step_id, _, seconds in STEP_DEFS}
TOTAL_VIDEO_SECONDS = sum(seconds for _, _, seconds in STEP_DEFS)
REFERENCE_VIDEO_DURATION_SECONDS = 32.0
SCALABLE_STEP_IDS = {
    "01_detection_tracking",
    "02_export_ocr_zones",
    "03_numeric_ocr",
    "04_tesseract_product_name",
    "05_crop_quality_reselect",
    "06_catalog_recovery_db_hack",
    "07_fill_aux_barcode",
}
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()
DB_LOCK = threading.Lock()
JOB_STATE_FILENAME = "state.json"


def json_bytes(payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> tuple[int, bytes, str]:
    return status.value, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def db_connect() -> sqlite3.Connection:
    RESULTS_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(RESULTS_DB), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def init_results_db() -> None:
    with DB_LOCK, db_connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              created_at REAL NOT NULL,
              finished_at REAL,
              seconds REAL,
              total_videos INTEGER DEFAULT 0,
              rows_count INTEGER DEFAULT 0,
              review_count INTEGER DEFAULT 0,
              uncertain_count INTEGER DEFAULT 0,
              job_root TEXT,
              csv_path TEXT,
              error TEXT
            );
            CREATE TABLE IF NOT EXISTS videos (
              job_id TEXT NOT NULL,
              filename TEXT NOT NULL,
              duration_seconds REAL,
              run_root TEXT,
              final_csv TEXT,
              rows_count INTEGER DEFAULT 0,
              PRIMARY KEY (job_id, filename)
            );
            CREATE TABLE IF NOT EXISTS result_rows (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT NOT NULL,
              video_filename TEXT NOT NULL,
              row_index INTEGER NOT NULL,
              product_name TEXT,
              price_default REAL,
              price_card REAL,
              price_discount TEXT,
              barcode TEXT,
              id_sku TEXT,
              color TEXT,
              frame_timestamp TEXT,
              x_min REAL,
              y_min REAL,
              x_max REAL,
              y_max REAL,
              row_json TEXT NOT NULL,
              UNIQUE (job_id, video_filename, row_index)
            );
            CREATE INDEX IF NOT EXISTS idx_result_rows_job ON result_rows(job_id);
            CREATE INDEX IF NOT EXISTS idx_result_rows_barcode ON result_rows(barcode);
            CREATE INDEX IF NOT EXISTS idx_result_rows_video ON result_rows(video_filename);
            CREATE TABLE IF NOT EXISTS review_items (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT NOT NULL,
              video_filename TEXT,
              row_index INTEGER,
              track_id TEXT,
              product_name TEXT,
              price_default TEXT,
              price_card TEXT,
              discount_amount TEXT,
              barcode TEXT,
              frame_timestamp TEXT,
              crop_score TEXT,
              ocr_score TEXT,
              image_url TEXT,
              source_url TEXT,
              item_json TEXT NOT NULL,
              UNIQUE (job_id, video_filename, row_index)
            );
            CREATE INDEX IF NOT EXISTS idx_review_items_job ON review_items(job_id);
            """
        )


def to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() == "нет":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def upsert_job_record(job_id: str, **values: object) -> None:
    init_results_db()
    now = time.time()
    with DB_LOCK, db_connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, total_videos, job_root)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
              status = excluded.status,
              total_videos = COALESCE(excluded.total_videos, jobs.total_videos),
              job_root = COALESCE(excluded.job_root, jobs.job_root)
            """,
            (
                job_id,
                str(values.get("status", "queued")),
                float(values.get("created_at", now) or now),
                int(values.get("total_videos", 0) or 0),
                str(values.get("job_root", "") or ""),
            ),
        )
        update_fields = {key: value for key, value in values.items() if key not in {"created_at", "total_videos", "job_root"}}
        if update_fields:
            columns = ", ".join(f"{key} = ?" for key in update_fields)
            connection.execute(
                f"UPDATE jobs SET {columns} WHERE job_id = ?",
                [*update_fields.values(), job_id],
            )


def persist_completed_job(
    job_id: str,
    manifest: list[dict[str, str]],
    final_csvs: list[Path],
    combined_csv: Path,
    row_count: int,
    review: dict[str, object],
    uncertain: dict[str, object],
    started: float,
) -> None:
    init_results_db()
    with DB_LOCK, db_connect() as connection:
        connection.execute("DELETE FROM result_rows WHERE job_id = ?", (job_id,))
        connection.execute("DELETE FROM review_items WHERE job_id = ?", (job_id,))
        for video_manifest, final_csv in zip(manifest, final_csvs):
            rows = read_csv_rows(final_csv)
            filename = video_manifest.get("filename", "")
            connection.execute(
                """
                INSERT INTO videos (job_id, filename, run_root, final_csv, rows_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id, filename) DO UPDATE SET
                  run_root = excluded.run_root,
                  final_csv = excluded.final_csv,
                  rows_count = excluded.rows_count
                """,
                (job_id, filename, video_manifest.get("run_root", ""), str(final_csv), len(rows)),
            )
            for index, row in enumerate(rows, start=1):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO result_rows (
                      job_id, video_filename, row_index, product_name, price_default, price_card,
                      price_discount, barcode, id_sku, color, frame_timestamp, x_min, y_min, x_max, y_max, row_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        filename,
                        index,
                        row.get("product_name", ""),
                        to_float_or_none(row.get("price_default")),
                        to_float_or_none(row.get("price_card")),
                        row.get("price_discount", ""),
                        row.get("barcode", ""),
                        row.get("id_sku", ""),
                        row.get("color", ""),
                        row.get("frame_timestamp", ""),
                        to_float_or_none(row.get("x_min")),
                        to_float_or_none(row.get("y_min")),
                        to_float_or_none(row.get("x_max")),
                        to_float_or_none(row.get("y_max")),
                        json.dumps(row, ensure_ascii=False),
                    ),
                )
        for item in review.get("items", []):
            if not isinstance(item, dict):
                continue
            connection.execute(
                """
                INSERT OR REPLACE INTO review_items (
                  job_id, video_filename, row_index, track_id, product_name, price_default,
                  price_card, discount_amount, barcode, frame_timestamp, crop_score, ocr_score,
                  image_url, source_url, item_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    str(item.get("video", "")),
                    int(item.get("row", 0) or 0),
                    str(item.get("track_id", "")),
                    str(item.get("product_name", "")),
                    str(item.get("price_default", "")),
                    str(item.get("price_card", "")),
                    str(item.get("discount_amount", "")),
                    str(item.get("barcode", "")),
                    str(item.get("frame_timestamp", "")),
                    str(item.get("crop_score", "")),
                    str(item.get("ocr_score", "")),
                    str(item.get("image_url", "")),
                    str(item.get("source_url", "")),
                    json.dumps(item, ensure_ascii=False),
                ),
            )
        connection.execute(
            """
            UPDATE jobs SET
              status = 'done',
              finished_at = ?,
              seconds = ?,
              rows_count = ?,
              review_count = ?,
              uncertain_count = ?,
              csv_path = ?,
              error = ''
            WHERE job_id = ?
            """,
            (
                time.time(),
                round(time.time() - started, 3),
                row_count,
                int(review.get("count", 0) or 0),
                int(uncertain.get("count", 0) or 0),
                str(combined_csv),
                job_id,
            ),
        )


def mark_job_failed_in_db(job_id: str, error: str) -> None:
    upsert_job_record(job_id, status="failed", finished_at=time.time(), error=error)


def business_metrics_payload() -> dict[str, object]:
    init_results_db()
    with DB_LOCK, db_connect() as connection:
        jobs = connection.execute(
            """
            SELECT
              COUNT(*) AS total_jobs,
              COALESCE(SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END), 0) AS done_jobs,
              COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed_jobs,
              COALESCE(SUM(total_videos), 0) AS total_videos,
              COALESCE(SUM(rows_count), 0) AS total_rows,
              COALESCE(SUM(review_count), 0) AS review_items,
              COALESCE(SUM(uncertain_count), 0) AS uncertain_items,
              AVG(seconds) AS avg_job_seconds
            FROM jobs
            """
        ).fetchone()
        rows = connection.execute(
            """
            SELECT
              COUNT(*) AS rows_total,
              SUM(CASE WHEN barcode IS NOT NULL AND barcode != '' AND lower(barcode) != 'нет' THEN 1 ELSE 0 END) AS rows_with_barcode,
              SUM(CASE WHEN product_name IS NOT NULL AND product_name != '' AND lower(product_name) != 'нет' THEN 1 ELSE 0 END) AS rows_with_product_name,
              SUM(CASE WHEN price_card IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_price_card,
              SUM(CASE WHEN price_default IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_price_default,
              SUM(CASE WHEN price_discount IS NOT NULL AND price_discount != '' AND lower(price_discount) != 'нет' THEN 1 ELSE 0 END) AS discounted_rows,
              AVG(price_card) AS avg_card_price,
              AVG(price_default) AS avg_default_price,
              SUM(CASE WHEN price_card IS NOT NULL AND price_default IS NOT NULL AND price_card > price_default THEN 1 ELSE 0 END) AS suspicious_card_gt_default
            FROM result_rows
            """
        ).fetchone()
        videos = connection.execute(
            """
            SELECT video_filename, COUNT(*) AS rows_count
            FROM result_rows
            GROUP BY video_filename
            ORDER BY rows_count DESC
            LIMIT 20
            """
        ).fetchall()
    row_total = int(rows["rows_total"] or 0)

    def rate(value: object) -> float:
        return round((float(value or 0) / row_total) * 100, 2) if row_total else 0.0

    return {
        "database": str(RESULTS_DB),
        "jobs": dict(jobs),
        "quality": {
            "rows_total": row_total,
            "barcode_fill_rate_pct": rate(rows["rows_with_barcode"]),
            "product_name_fill_rate_pct": rate(rows["rows_with_product_name"]),
            "price_card_fill_rate_pct": rate(rows["rows_with_price_card"]),
            "price_default_fill_rate_pct": rate(rows["rows_with_price_default"]),
            "discounted_rows": int(rows["discounted_rows"] or 0),
            "avg_card_price": round(float(rows["avg_card_price"] or 0), 2),
            "avg_default_price": round(float(rows["avg_default_price"] or 0), 2),
            "suspicious_card_gt_default": int(rows["suspicious_card_gt_default"] or 0),
        },
        "top_videos_by_detected_tags": [dict(row) for row in videos],
    }


def sanitize_name(value: str, default: str = "video") -> str:
    stem = Path(value or default).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._-")
    return stem[:80] or default


def safe_upload_path(upload_dir: Path, filename: str) -> Path:
    stem = sanitize_name(filename)
    suffix = Path(filename).suffix.lower() or ".mp4"
    if suffix != ".mp4":
        suffix = ".mp4"
    candidate = upload_dir / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def sample_columns() -> list[str]:
    if SAMPLE_CSV.exists():
        with SAMPLE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if header:
                return header
    return [
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


def parse_multipart(headers, body: bytes) -> list[tuple[str, str, bytes]]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data")

    message_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    files: list[tuple[str, str, bytes]] = []

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename() or ""
        payload = part.get_payload(decode=True) or b""
        if field_name == "videos" and filename and payload:
            files.append((filename, filename, payload))
    return files


def save_uploaded_videos(headers, body: bytes, upload_dir: Path) -> list[Path]:
    items = parse_multipart(headers, body)
    saved_paths: list[Path] = []
    for _, filename, payload in items:
        saved_path = safe_upload_path(upload_dir, filename)
        saved_path.write_bytes(payload)
        if saved_path.stat().st_size > 0:
            saved_paths.append(saved_path)
    return saved_paths


def mp4_duration_seconds(path: Path) -> float:
    try:
        with path.open("rb") as handle:
            overlap = b""
            offset = 0
            while True:
                chunk = handle.read(2 * 1024 * 1024)
                if not chunk:
                    break
                data = overlap + chunk
                index = data.find(b"mvhd")
                if index >= 4:
                    type_pos = offset - len(overlap) + index
                    handle.seek(max(0, type_pos - 4))
                    header = handle.read(48)
                    if len(header) < 32 or header[4:8] != b"mvhd":
                        break
                    version = header[8]
                    if version == 0 and len(header) >= 28:
                        timescale = int.from_bytes(header[20:24], "big")
                        duration = int.from_bytes(header[24:28], "big")
                    elif version == 1 and len(header) >= 40:
                        timescale = int.from_bytes(header[28:32], "big")
                        duration = int.from_bytes(header[32:40], "big")
                    else:
                        return 0.0
                    if timescale > 0 and duration > 0:
                        return duration / timescale
                    return 0.0
                overlap = data[-64:]
                offset += len(chunk)
    except OSError:
        return 0.0
    return 0.0


def video_duration_scale(duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 1.0
    return max(0.35, duration_seconds / REFERENCE_VIDEO_DURATION_SECONDS)


def step_seconds_for_scale(scale: float) -> dict[str, float]:
    return {
        step_id: seconds * scale if step_id in SCALABLE_STEP_IDS else seconds
        for step_id, seconds in STEP_AVG_SECONDS.items()
    }


def total_seconds_for_scale(scale: float) -> float:
    return sum(step_seconds_for_scale(scale).values())


def choose_checkpoint() -> Path:
    explicit = os.environ.get("LENTA_DETECTOR_CHECKPOINT")
    if explicit:
        return Path(explicit)
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    return FALLBACK_CHECKPOINT


def parse_seconds(value: object) -> float:
    text = str(value or "").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def read_tail(path: Path, max_chars: int = 240_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def read_timings(run_root: Path) -> list[dict[str, str]]:
    timings_csv = run_root / "timings.csv"
    if not timings_csv.exists():
        return []
    try:
        with timings_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value or "").replace(",", "."))
    except ValueError:
        return default


def coord_key(row: dict[str, str]) -> tuple[str, int, int, int, int, int]:
    return (
        str(row.get("filename", "")).strip(),
        int(round(safe_float(row.get("frame_timestamp", "")))),
        int(round(safe_float(row.get("x_min", "")))),
        int(round(safe_float(row.get("y_min", "")))),
        int(round(safe_float(row.get("x_max", "")))),
        int(round(safe_float(row.get("y_max", "")))),
    )


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return ""


def artifact_url(job_id: str, rel_path: str) -> str:
    return f"/api/jobs/{job_id}/artifact?path={quote(rel_path, safe='')}" if rel_path else ""


def ensure_retrain_config() -> dict[str, object]:
    if RETRAIN_CONFIG_PATH.exists():
        try:
            payload = json.loads(RETRAIN_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    payload = {
        "enabled": AUTO_RETRAIN_DEFAULT_ENABLED,
        "schedule": "weekly",
        "uncertain_root": str(UNCERTAIN_ROOT),
        "note": "Auto retraining is intentionally disabled until enabled manually.",
    }
    write_json(RETRAIN_CONFIG_PATH, payload)
    return payload


def last_progress_match(log_text: str, patterns: list[str]) -> tuple[int, int] | None:
    for pattern in patterns:
        matches = list(re.finditer(pattern, log_text, flags=re.IGNORECASE))
        if not matches:
            continue
        current = int(matches[-1].group(1))
        total = max(1, int(matches[-1].group(2)))
        return current, total
    return None


def step_fraction(step_id: str, log_text: str, current_elapsed: float, step_seconds: dict[str, float]) -> tuple[float, str]:
    progress_match: tuple[int, int] | None = None
    detail = ""
    if step_id == "01_detection_tracking":
        progress_match = last_progress_match(log_text, [r"frame\s+(\d+)/(\d+)"])
        if progress_match:
            current, total = progress_match
            detail = f"кадр {current}/{total}"
    elif step_id == "03_numeric_ocr":
        progress_match = last_progress_match(
            log_text,
            [
                r"Completed\s+(\d+)/(\d+)\s+parallel rapidocr calls",
                r"Completed\s+(\d+)/(\d+)\s+decoder calls",
                r"Processed\s+(\d+)/(\d+)\s+zone rows",
                r"Scheduled\s+(\d+)/(\d+)\s+zone rows",
            ],
        )
        if progress_match:
            current, total = progress_match
            detail = f"OCR-вызовы {current}/{total}"
    elif step_id == "04_tesseract_product_name":
        progress_match = last_progress_match(
            log_text,
            [
                r"Ran\s+(\d+)/(\d+)\s+Tesseract calls",
                r"Prepared\s+(\d+)/(\d+)\s+product_name zone rows",
            ],
        )
        if progress_match:
            current, total = progress_match
            detail = f"Tesseract {current}/{total}"

    if progress_match:
        current, total = progress_match
        return min(0.98, max(0.01, current / total)), detail

    avg_seconds = step_seconds.get(step_id, STEP_AVG_SECONDS.get(step_id, 60.0))
    elapsed_fraction = current_elapsed / max(1.0, avg_seconds)
    return min(0.85, max(0.03, elapsed_fraction * 0.85)), detail


def current_step_from_log(log_text: str, completed_steps: set[str]) -> str:
    starts = re.findall(r"^START\s+(.+?)\s*$", log_text, flags=re.MULTILINE)
    for step_id in reversed(starts):
        if step_id not in completed_steps:
            return step_id
    for step_id in STEP_IDS:
        if step_id not in completed_steps:
            return step_id
    return STEP_IDS[-1]


def analyze_run_progress(run_root: Path, log_path: Path, elapsed_seconds: float, duration_scale: float = 1.0) -> dict[str, object]:
    step_seconds = step_seconds_for_scale(duration_scale)
    total_video_seconds = sum(step_seconds.values())
    timings = read_timings(run_root)
    completed_steps = {row.get("step", "") for row in timings if row.get("status") == "ok"}
    completed_actual_seconds = sum(parse_seconds(row.get("seconds")) for row in timings if row.get("status") == "ok")
    log_text = read_tail(log_path)

    if len(completed_steps) >= len(STEP_IDS):
        return {
            "progress": 1.0,
            "stage": "CSV готов",
            "detail": "",
            "eta_seconds": 0.0,
            "current_step": STEP_IDS[-1],
        }

    current_step = current_step_from_log(log_text, completed_steps)
    current_elapsed = max(0.0, elapsed_seconds - completed_actual_seconds)
    fraction, detail = step_fraction(current_step, log_text, current_elapsed, step_seconds)
    completed_weight = sum(step_seconds.get(step_id, 0.0) for step_id in completed_steps)
    current_weight = step_seconds.get(current_step, 30.0)
    progress = (completed_weight + current_weight * fraction) / max(1.0, total_video_seconds)

    current_index = STEP_IDS.index(current_step) if current_step in STEP_IDS else 0
    future_steps = STEP_IDS[current_index + 1 :]
    future_seconds = sum(step_seconds.get(step_id, 0.0) for step_id in future_steps)
    avg_current_remaining = current_weight * max(0.0, 1.0 - fraction)
    if fraction > 0.08 and current_elapsed > 8:
        observed_current_remaining = max(0.0, current_elapsed / fraction - current_elapsed)
        current_remaining = observed_current_remaining * 0.7 + avg_current_remaining * 0.3
    else:
        current_remaining = avg_current_remaining

    return {
        "progress": min(0.99, max(0.0, progress)),
        "stage": STEP_LABELS.get(current_step, current_step),
        "detail": detail,
        "eta_seconds": current_remaining + future_seconds,
        "current_step": current_step,
    }


def eta_range(seconds: float, progress: float) -> tuple[int, int]:
    if seconds <= 0:
        return 0, 0
    spread = 0.35 if progress < 0.25 else 0.25
    low = max(5, int(seconds * (1.0 - spread)))
    high = max(low + 5, int(seconds * (1.0 + spread)))
    return low, high


def set_job_state(job_id: str, **updates: object) -> None:
    updates.setdefault("updated_at", time.time())
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        snapshot = dict(job)
    job_root_value = snapshot.get("job_root", "")
    if job_root_value:
        try:
            state_path = Path(str(job_root_value)) / JOB_STATE_FILENAME
            write_json(state_path, snapshot)
        except OSError:
            pass


def load_job_state_from_disk(job_id: str) -> dict[str, object] | None:
    state_path = JOBS_ROOT / job_id / JOB_STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    with JOBS_LOCK:
        JOBS[job_id] = dict(payload)
    return dict(payload)


def get_job_state(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            return dict(job)
    return load_job_state_from_disk(job_id)


def get_live_job(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def read_json_request(handler: SimpleHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    if content_length > 2 * 1024 * 1024:
        raise ValueError("JSON request is too large")
    raw = handler.rfile.read(content_length)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON object expected")
    return payload


def make_upload_job(files: list[dict[str, object]]) -> dict[str, object]:
    if not files:
        raise ValueError("No MP4 files were selected")
    started = time.time()
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_root = JOBS_ROOT / job_id
    upload_dir = job_root / "uploads"
    outputs_dir = job_root / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    upload_files: list[dict[str, object]] = []
    total_bytes = 0
    for index, item in enumerate(files, start=1):
        original_name = Path(str(item.get("name", f"video_{index}.mp4"))).name
        size = int(item.get("size", 0) or 0)
        if size <= 0:
            raise ValueError(f"Empty file: {original_name}")
        if size > MAX_UPLOAD_BYTES:
            raise ValueError(f"File is too large: {original_name}")
        total_bytes += size
        path = safe_upload_path(upload_dir, original_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        upload_files.append(
            {
                "upload_id": str(index - 1),
                "name": original_name,
                "path": str(path),
                "size": size,
                "received": 0,
                "complete": False,
            }
        )
    set_job_state(
        job_id,
        status="uploading",
        started_at=started,
        total_videos=len(upload_files),
        completed_videos=0,
        job_root=str(job_root),
        upload_dir=str(upload_dir),
        outputs_dir=str(outputs_dir),
        upload_files=upload_files,
        upload_total_bytes=total_bytes,
        upload_received_bytes=0,
        stage="Загрузка видео",
        detail="",
        rows=0,
    )
    upsert_job_record(
        job_id,
        status="uploading",
        created_at=started,
        total_videos=len(upload_files),
        job_root=str(job_root),
    )
    return {
        "job_id": job_id,
        "files": [
            {
                "upload_id": item["upload_id"],
                "name": item["name"],
                "size": item["size"],
            }
            for item in upload_files
        ],
        "status_url": f"/api/jobs/{job_id}/status",
        "start_url": f"/api/jobs/{job_id}/start",
    }


def find_upload_file(job: dict[str, object], upload_id: str) -> dict[str, object]:
    upload_files = job.get("upload_files", [])
    if not isinstance(upload_files, list):
        raise ValueError("Job has no upload files")
    for item in upload_files:
        if isinstance(item, dict) and str(item.get("upload_id", "")) == upload_id:
            return item
    raise ValueError("Upload file was not found")


def write_upload_chunk(job_id: str, upload_id: str, offset: int, body: bytes) -> dict[str, object]:
    if not body:
        raise ValueError("Empty upload chunk")
    if len(body) > MAX_UPLOAD_CHUNK_BYTES:
        raise ValueError("Upload chunk is too large")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError("Job not found")
        if str(job.get("status", "")) != "uploading":
            raise ValueError("Job is not accepting uploads")
        upload_file = find_upload_file(job, upload_id)
        path = Path(str(upload_file.get("path", "")))
        expected_size = int(upload_file.get("size", 0) or 0)
    if offset < 0 or offset + len(body) > expected_size:
        raise ValueError("Chunk offset is out of range")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(body)
    current_size = path.stat().st_size
    is_complete = current_size >= expected_size
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError("Job not found")
        upload_file = find_upload_file(job, upload_id)
        upload_file["received"] = min(expected_size, max(int(upload_file.get("received", 0) or 0), current_size))
        upload_file["complete"] = is_complete
        upload_files = [item for item in job.get("upload_files", []) if isinstance(item, dict)]
        job["upload_received_bytes"] = sum(min(int(item.get("size", 0) or 0), int(item.get("received", 0) or 0)) for item in upload_files)
        job["detail"] = f"{job['upload_received_bytes']} / {job.get('upload_total_bytes', 0)} bytes"
    return {"received": current_size, "complete": is_complete}


def enqueue_inference_task(job_id: str, uploads: list[Path], job_root: Path, outputs_dir: Path, started: float) -> str:
    payload = {
        "job_id": job_id,
        "uploads": [str(path) for path in uploads],
        "job_root": str(job_root),
        "outputs_dir": str(outputs_dir),
        "started": started,
    }
    return enqueue_task(
        WORKER_QUEUE_DB,
        "video_inference",
        payload,
        source_job_id=job_id,
        priority=100,
        max_attempts=2,
        task_id=f"video_inference_{job_id}",
    )


def start_uploaded_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError("Job not found")
        if QUEUE_MODE_ENABLED and job.get("queue_task_id"):
            return {
                "job_id": job_id,
                "status": "queued",
                "execution_mode": "queue",
                "queue_task_id": str(job.get("queue_task_id", "")),
                "status_url": f"/api/jobs/{job_id}/status",
                "csv_url": f"/api/jobs/{job_id}/csv",
            }
        if str(job.get("status", "")) not in {"uploading", "queued"}:
            raise ValueError("Job cannot be started")
        upload_files = [dict(item) for item in job.get("upload_files", []) if isinstance(item, dict)]
    uploads: list[Path] = []
    for item in upload_files:
        path = Path(str(item.get("path", "")))
        expected_size = int(item.get("size", 0) or 0)
        if not path.exists() or path.stat().st_size < expected_size:
            raise ValueError(f"Upload is incomplete: {item.get('name', '')}")
        uploads.append(path)
    if not uploads:
        raise ValueError("No uploaded videos to process")
    started = time.time()
    video_durations = [round(mp4_duration_seconds(path), 3) for path in uploads]
    job_root = Path(str(job.get("job_root", JOBS_ROOT / job_id)))
    outputs_dir = Path(str(job.get("outputs_dir", job_root / "outputs")))
    set_job_state(
        job_id,
        status="queued",
        started_at=started,
        total_videos=len(uploads),
        video_durations=video_durations,
        completed_videos=0,
        job_root=str(job_root),
        outputs_dir=str(outputs_dir),
        stage="Видео загружено",
        detail="",
        rows=0,
    )
    upsert_job_record(job_id, status="queued", created_at=started, total_videos=len(uploads), job_root=str(job_root))
    if QUEUE_MODE_ENABLED:
        task_id = enqueue_inference_task(job_id, uploads, job_root, outputs_dir, started)
        set_job_state(
            job_id,
            status="queued",
            execution_mode="queue",
            queue_task_id=task_id,
            stage="В очереди worker",
            detail="Ожидаем свободный обработчик",
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "execution_mode": "queue",
            "queue_task_id": task_id,
            "status_url": f"/api/jobs/{job_id}/status",
            "csv_url": f"/api/jobs/{job_id}/csv",
        }
    worker = threading.Thread(
        target=run_job,
        args=(job_id, uploads, job_root, outputs_dir, started),
        daemon=True,
        name=f"lenta-job-{job_id}",
    )
    worker.start()
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/jobs/{job_id}/status",
        "csv_url": f"/api/jobs/{job_id}/csv",
    }


def build_job_status(job_id: str) -> dict[str, object] | None:
    job = get_job_state(job_id)
    if job is None:
        return None
    if str(job.get("execution_mode", "")) == "queue":
        disk_job = load_job_state_from_disk(job_id)
        if disk_job and float(disk_job.get("updated_at", 0) or 0) >= float(job.get("updated_at", 0) or 0):
            job = disk_job

    status = str(job.get("status", "queued"))
    queue_task = None
    if str(job.get("execution_mode", "")) == "queue":
        try:
            queue_task = get_task_by_source_job(WORKER_QUEUE_DB, job_id)
        except sqlite3.Error:
            queue_task = None
        if queue_task and status not in {"done", "failed"}:
            task_status = str(queue_task.get("status", ""))
            if task_status == "running":
                status = "running"
                job["status"] = "running"
                job.setdefault("stage", "Worker обрабатывает видео")
                worker_id = str(queue_task.get("lease_owner", ""))
                if worker_id:
                    job.setdefault("detail", f"Задачу забрал {worker_id}")
            elif task_status == "failed":
                status = "failed"
                job["status"] = "failed"
                job["error"] = str(queue_task.get("error", "Worker task failed"))
    total_videos = max(1, int(job.get("total_videos", 1)))
    completed_videos = max(0, int(job.get("completed_videos", 0)))
    current_index = max(1, int(job.get("current_video_index", min(total_videos, completed_videos + 1))))
    raw_durations = job.get("video_durations", [])
    video_durations = [float(value or 0.0) for value in raw_durations] if isinstance(raw_durations, list) else []
    while len(video_durations) < total_videos:
        video_durations.append(0.0)
    video_scales = [video_duration_scale(duration) for duration in video_durations[:total_videos]]
    video_totals = [total_seconds_for_scale(scale) for scale in video_scales]
    total_estimated_seconds = sum(video_totals) or TOTAL_VIDEO_SECONDS * total_videos
    elapsed_seconds = max(0.0, time.time() - float(job.get("started_at", time.time())))
    progress = 0.0
    eta_seconds = sum(video_totals[completed_videos:])
    stage = str(job.get("stage", "Ожидание запуска"))
    detail = str(job.get("detail", ""))

    if status == "running":
        current_run_root = Path(str(job.get("current_run_root", "")))
        current_log_path = Path(str(job.get("current_log_path", "")))
        current_started_at = float(job.get("current_video_started_at", job.get("started_at", time.time())))
        current_elapsed = max(0.0, time.time() - current_started_at)
        current_scale = video_scales[min(current_index - 1, len(video_scales) - 1)]
        current_total = video_totals[min(current_index - 1, len(video_totals) - 1)]
        run_progress = analyze_run_progress(current_run_root, current_log_path, current_elapsed, current_scale)
        video_progress = float(run_progress["progress"])
        completed_weight = sum(video_totals[:completed_videos])
        progress = min(0.99, (completed_weight + video_progress * current_total) / max(1.0, total_estimated_seconds))
        remaining_current = float(run_progress["eta_seconds"])
        remaining_videos = sum(video_totals[current_index:])
        eta_seconds = remaining_current + remaining_videos
        stage = str(run_progress["stage"])
        detail = str(run_progress["detail"])
    elif status == "done":
        progress = 1.0
        eta_seconds = 0.0
        stage = "CSV готов"
    elif status == "failed":
        progress = min(0.99, completed_videos / total_videos)
        eta_seconds = 0.0
        stage = "Ошибка обработки"

    eta_low, eta_high = eta_range(eta_seconds, progress)
    return {
        "job_id": job_id,
        "status": status,
        "progress": round(progress * 100, 1),
        "stage": stage,
        "detail": detail,
        "eta_seconds": int(eta_seconds),
        "eta_low_seconds": eta_low,
        "eta_high_seconds": eta_high,
        "elapsed_seconds": int(elapsed_seconds),
        "current_video": str(job.get("current_video", "")),
        "current_video_index": current_index,
        "total_videos": total_videos,
        "rows": int(job.get("rows", 0)),
        "csv_url": f"/api/jobs/{job_id}/csv" if status == "done" else "",
        "review_url": f"/api/jobs/{job_id}/review" if status == "done" else "",
        "review_html_url": f"/api/jobs/{job_id}/review.html" if status == "done" else "",
        "review_zip_url": f"/api/jobs/{job_id}/review.zip" if status == "done" else "",
        "execution_mode": str(job.get("execution_mode", "local")),
        "queue_task_id": str(job.get("queue_task_id", "")),
        "queue_task_status": str(queue_task.get("status", "")) if queue_task else "",
        "queue_worker": str(queue_task.get("lease_owner", "")) if queue_task else "",
        "review_count": int(job.get("review_count", 0)),
        "uncertain_count": int(job.get("uncertain_count", 0)),
        "auto_retrain_enabled": bool(ensure_retrain_config().get("enabled", False)),
        "error": str(job.get("error", "")),
    }


def resolve_executable(value: str) -> str:
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    resolved = shutil.which(value)
    if resolved:
        return resolved
    return value


def run_pipeline(video_path: Path, video_id: str, run_root: Path, log_path: Path) -> Path:
    checkpoint = choose_checkpoint()
    if not PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(f"Pipeline script is missing: {PIPELINE_SCRIPT}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint is missing: {checkpoint}")
    if not DEFAULT_CATALOG.exists():
        raise FileNotFoundError(f"Catalog is missing: {DEFAULT_CATALOG}")
    pipeline_python = resolve_executable(PIPELINE_PYTHON)
    tesseract_exe = resolve_executable(TESSERACT_EXE)
    if not TESSDATA_DIR.exists():
        raise FileNotFoundError(f"Tessdata directory is missing: {TESSDATA_DIR}")

    command = [
        PIPELINE_POWERSHELL,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PIPELINE_SCRIPT),
        "-Python",
        pipeline_python,
        "-TesseractExe",
        tesseract_exe,
        "-TessdataDir",
        str(TESSDATA_DIR),
        "-VideoPath",
        str(video_path),
        "-VideoId",
        video_id,
        "-RunRoot",
        str(run_root),
        "-DetectorCheckpoint",
        str(checkpoint),
        "-CatalogPath",
        str(DEFAULT_CATALOG),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    pipeline_python_dir = str(Path(pipeline_python).resolve().parent)
    env["PATH"] = pipeline_python_dir + os.pathsep + env.get("PATH", "")
    env["PYTHONEXECUTABLE"] = pipeline_python

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(PIPELINE_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Inference failed for {video_path.name}. Log tail:\n{tail}")

    final_csv = run_root / "final_submission.csv"
    if not final_csv.exists():
        raise FileNotFoundError(f"Pipeline finished without final CSV: {final_csv}")
    return final_csv


def combine_csvs(csv_paths: list[Path], output_path: Path) -> int:
    columns = sample_columns()
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({column: row.get(column, "") for column in columns})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def build_zone_lookup(run_root: Path) -> dict[tuple[str, int, int, int, int, int], dict[str, str]]:
    zones_csv = run_root / "base" / "ocr_zones_core_fixed" / "ocr_zones_manifest.csv"
    lookup: dict[tuple[str, int, int, int, int, int], dict[str, str]] = {}
    for row in read_csv_rows(zones_csv):
        if row.get("zone") == "product_name" and row.get("rank") == "1":
            lookup.setdefault(coord_key(row), row)
    return lookup


def build_review_package(job_id: str, job_root: Path, video_manifests: list[dict[str, str]], final_csvs: list[Path]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    review_dir = job_root / "review"
    crops_dir = review_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for video_manifest, final_csv in zip(video_manifests, final_csvs):
        run_root = Path(video_manifest["run_root"])
        zone_lookup = build_zone_lookup(run_root)
        fallback_zones = list(zone_lookup.values())
        for index, row in enumerate(read_csv_rows(final_csv), start=1):
            zone = zone_lookup.get(coord_key(row)) or (fallback_zones[index - 1] if index - 1 < len(fallback_zones) else {})
            full_tag = Path(zone.get("full_tag", "")) if zone.get("full_tag") else Path()
            source_crop = Path(zone.get("source_crop", "")) if zone.get("source_crop") else Path()
            full_rel = relative_to_root(full_tag, job_root) if full_tag.exists() else ""
            source_rel = relative_to_root(source_crop, job_root) if source_crop.exists() else ""
            if not full_rel and not source_rel:
                continue
            image_source = full_tag if full_rel else source_crop
            image_file = ""
            if image_source.exists():
                crop_name = (
                    f"{len(items) + 1:04d}_"
                    f"{sanitize_name(video_manifest.get('filename', 'video'))}_"
                    f"{sanitize_name(str(zone.get('track_id', 'track')), 'track')}"
                    f"{image_source.suffix.lower() or '.jpg'}"
                )
                target_crop = crops_dir / crop_name
                try:
                    shutil.copyfile(image_source, target_crop)
                    image_file = relative_to_root(target_crop, review_dir)
                except OSError:
                    image_file = ""
            items.append(
                {
                    "video": video_manifest.get("filename", ""),
                    "row": index,
                    "track_id": zone.get("track_id", ""),
                    "product_name": row.get("product_name", ""),
                    "price_default": row.get("price_default", ""),
                    "price_card": row.get("price_card", ""),
                    "discount_amount": row.get("discount_amount", ""),
                    "barcode": row.get("barcode", ""),
                    "frame_timestamp": row.get("frame_timestamp", ""),
                    "crop_score": zone.get("crop_score", ""),
                    "ocr_score": zone.get("ocr_score", ""),
                    "image_url": artifact_url(job_id, full_rel or source_rel),
                    "source_url": artifact_url(job_id, source_rel),
                    "image_file": image_file,
                }
            )
            if len(items) >= REVIEW_MAX_ITEMS:
                break
        if len(items) >= REVIEW_MAX_ITEMS:
            break
    html_path = review_dir / "review_report.html"
    offline_html_path = review_dir / "review_report_offline.html"
    zip_path = review_dir / "review_package.zip"
    payload = {
        "job_id": job_id,
        "items": items,
        "count": len(items),
        "html_path": str(html_path),
        "offline_html_path": str(offline_html_path),
        "zip_path": str(zip_path),
    }
    manifest_path = review_dir / "review_manifest.json"
    write_json(manifest_path, payload)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_review_html(job_id, items), encoding="utf-8")
    offline_html_path.write_text(build_review_html(job_id, items, offline=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(offline_html_path, "review_report.html")
        archive.write(manifest_path, "review_manifest.json")
        for item in items:
            image_file = str(item.get("image_file", ""))
            if not image_file:
                continue
            image_path = (review_dir / image_file).resolve()
            if review_dir.resolve() in image_path.parents and image_path.exists():
                archive.write(image_path, image_file)
    return payload


def build_review_html(job_id: str, items: list[dict[str, object]], offline: bool = False) -> str:
    cards = []
    for item in items:
        title = str(item.get("product_name", "")) or "Без названия"
        price = " / ".join(str(item.get(key, "")) for key in ("price_default", "price_card") if item.get(key))
        meta = " · ".join(
            value
            for value in [
                str(item.get("video", "")),
                f"row {item.get('row', '')}",
                f"track {item.get('track_id', '')}" if item.get("track_id") else "",
                str(item.get("barcode", "")),
                price,
            ]
            if value
        )
        image_url = str(item.get("image_file" if offline else "image_url", ""))
        cards.append(
            f"""
            <article class="card">
              <img src="{escape(image_url)}" alt="">
              <div class="body">
                <h2>{escape(title)}</h2>
                <p>{escape(meta)}</p>
                <dl>
                  <dt>Скидка</dt><dd>{escape(str(item.get("discount_amount", "")))}</dd>
                  <dt>Timestamp</dt><dd>{escape(str(item.get("frame_timestamp", "")))}</dd>
                  <dt>Crop score</dt><dd>{escape(str(item.get("crop_score", "")))}</dd>
                  <dt>OCR score</dt><dd>{escape(str(item.get("ocr_score", "")))}</dd>
                </dl>
              </div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Review crops · {escape(job_id)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #152238; }}
    header {{ padding: 24px 32px; background: #071f4a; color: white; }}
    main {{ padding: 24px 32px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #dce3ef; border-radius: 8px; overflow: hidden; }}
    img {{ width: 100%; height: 190px; object-fit: contain; background: #101828; display: block; }}
    .body {{ padding: 14px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 8px; font-size: 16px; line-height: 1.35; }}
    p {{ margin: 0 0 10px; color: #506079; font-size: 13px; }}
    dl {{ display: grid; grid-template-columns: 96px 1fr; gap: 5px 10px; margin: 0; font-size: 12px; }}
    dt {{ color: #6b778c; }}
    dd {{ margin: 0; color: #172033; }}
  </style>
</head>
<body>
  <header>
    <h1>Review crops</h1>
    <div>job: {escape(job_id)} · crops: {len(items)}</div>
  </header>
  <main>
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""


def is_uncertain_candidate(row: dict[str, str]) -> bool:
    confidence = safe_float(row.get("confidence", ""), 1.0)
    score = safe_float(row.get("score", ""), 1.0)
    field = row.get("field", "")
    if field == "product_name":
        return confidence < 0.45 or score < 0.45
    if field in {"price_default", "price_card", "discount_amount", "barcode", "qr_code_barcode"}:
        return confidence < 0.55 or score < 0.95 or row.get("valid") in {"0", "False", "false"}
    return confidence < 0.45


def collect_uncertain_predictions(job_id: str, job_root: Path, video_manifests: list[dict[str, str]]) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    store_root = UNCERTAIN_ROOT / job_id
    for video_manifest in video_manifests:
        run_root = Path(video_manifest["run_root"])
        candidate_csvs = [
            run_root / "base" / "ocr_numeric_quality_core_fast_fixed" / "ocr_field_candidates.csv",
            run_root / "tesseract" / "ocr_final_quality_core_fast_fixed" / "product_name_line_candidates.csv",
        ]
        for candidate_csv in candidate_csvs:
            for row in read_csv_rows(candidate_csv):
                if len(rows) >= UNCERTAIN_MAX_ITEMS:
                    break
                image_path = Path(row.get("image_path", ""))
                if not image_path.exists() or not is_uncertain_candidate(row):
                    continue
                target_dir = store_root / sanitize_name(video_manifest.get("filename", "video")) / "images"
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / image_path.name
                if not target.exists():
                    shutil.copy2(image_path, target)
                stored_rel = relative_to_root(target, store_root)
                rows.append(
                    {
                        "job_id": job_id,
                        "video": video_manifest.get("filename", ""),
                        "track_id": row.get("track_id", ""),
                        "field": row.get("field", ""),
                        "value": row.get("value", ""),
                        "confidence": row.get("confidence", ""),
                        "score": row.get("score", ""),
                        "engine": row.get("engine", ""),
                        "source": row.get("source", row.get("zone", "")),
                        "stored_image": stored_rel,
                    }
                )
            if len(rows) >= UNCERTAIN_MAX_ITEMS:
                break
    manifest = {
        "job_id": job_id,
        "enabled_for_auto_retrain": bool(ensure_retrain_config().get("enabled", False)),
        "count": len(rows),
        "items": rows,
    }
    write_json(store_root / "uncertain_manifest.json", manifest)
    csv_path = store_root / "uncertain_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["job_id", "video", "track_id", "field", "value", "confidence", "score", "engine", "source", "stored_image"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"root": str(store_root), "count": len(rows), "manifest": str(store_root / "uncertain_manifest.json")}


def delete_uploaded_videos(uploads: list[Path]) -> None:
    for upload in uploads:
        try:
            if upload.exists():
                upload.unlink()
        except OSError:
            pass


def run_job(job_id: str, uploads: list[Path], job_root: Path, outputs_dir: Path, started: float) -> None:
    final_csvs: list[Path] = []
    manifest: list[dict[str, str]] = []
    set_job_state(
        job_id,
        status="queued",
        stage="В очереди на GPU",
        detail="Ждем завершения предыдущей обработки",
    )

    try:
        with PIPELINE_LOCK:
            set_job_state(job_id, status="running", stage="Запуск пайплайна", detail="")
            for index, saved_path in enumerate(uploads, start=1):
                video_id = sanitize_name(saved_path.stem, f"video_{index}")
                run_root = outputs_dir / video_id
                log_path = outputs_dir / f"{video_id}.log"
                set_job_state(
                    job_id,
                    current_video=saved_path.name,
                    current_video_index=index,
                    current_run_root=str(run_root),
                    current_log_path=str(log_path),
                    current_video_started_at=time.time(),
                    stage="Запуск пайплайна",
                    detail="",
                )
                final_csv = run_pipeline(saved_path, video_id, run_root, log_path)
                final_csvs.append(final_csv)
                manifest.append(
                    {
                        "filename": saved_path.name,
                        "saved_path": str(saved_path),
                        "run_root": str(run_root),
                        "log_path": str(log_path),
                        "final_csv": str(final_csv),
                    }
                )
                set_job_state(job_id, completed_videos=index)

            set_job_state(job_id, stage="Сборка общего CSV", detail="")
            combined_csv = job_root / "combined_submission.csv"
            row_count = combine_csvs(final_csvs, combined_csv)
            review = build_review_package(job_id, job_root, manifest, final_csvs)
            uncertain = collect_uncertain_predictions(job_id, job_root, manifest)
            persist_completed_job(job_id, manifest, final_csvs, combined_csv, row_count, review, uncertain, started)
            delete_uploaded_videos(uploads)
            (job_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "rows": row_count,
                        "seconds": round(time.time() - started, 3),
                        "full_videos_deleted": True,
                        "review_count": review.get("count", 0),
                        "review_html": review.get("html_path", ""),
                        "review_zip": review.get("zip_path", ""),
                        "uncertain_count": uncertain.get("count", 0),
                        "uncertain_root": uncertain.get("root", ""),
                        "videos": manifest,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            set_job_state(
                job_id,
                status="done",
                stage="CSV готов",
                detail="",
                rows=row_count,
                csv_path=str(combined_csv),
                review_count=int(review.get("count", 0)),
                review_path=str(job_root / "review" / "review_manifest.json"),
                review_html_path=str(review.get("html_path", "")),
                review_zip_path=str(review.get("zip_path", "")),
                uncertain_count=int(uncertain.get("count", 0)),
                uncertain_root=str(uncertain.get("root", "")),
                completed_videos=len(uploads),
                finished_at=time.time(),
            )
    except Exception as exc:
        delete_uploaded_videos(uploads)
        mark_job_failed_in_db(job_id, str(exc))
        error_path = job_root / "error.txt"
        error_path.write_text(str(exc), encoding="utf-8", errors="replace")
        set_job_state(
            job_id,
            status="failed",
            stage="Ошибка обработки",
            detail="",
            error=str(exc),
            error_log=str(error_path),
            finished_at=time.time(),
        )


class LentaHandler(SimpleHTTPRequestHandler):
    server_version = "LentaWeb/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_payload(self, status: int, payload: bytes, content_type: str, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            checkpoint = choose_checkpoint()
            status, payload, content_type = json_bytes(
                {
                    "ok": PIPELINE_SCRIPT.exists() and checkpoint.exists() and DEFAULT_CATALOG.exists(),
                    "pipeline_root": str(PIPELINE_ROOT),
                    "runtime_root": str(RUNTIME_ROOT),
                    "jobs_root": str(JOBS_ROOT),
                    "uncertain_root": str(UNCERTAIN_ROOT),
                    "auto_retrain_enabled": bool(ensure_retrain_config().get("enabled", False)),
                    "results_db": str(RESULTS_DB),
                    "execution_mode": "queue" if QUEUE_MODE_ENABLED else "local",
                    "worker_queue_db": str(WORKER_QUEUE_DB),
                    "pipeline_python": resolve_executable(PIPELINE_PYTHON),
                    "tesseract": resolve_executable(TESSERACT_EXE),
                    "tessdata": str(TESSDATA_DIR),
                    "checkpoint": str(checkpoint),
                    "catalog": str(DEFAULT_CATALOG),
                }
            )
            self.send_payload(status, payload, content_type)
            return
        if path == "/api/queue":
            status, payload, content_type = json_bytes(
                {
                    "enabled": QUEUE_MODE_ENABLED,
                    "execution_mode": "queue" if QUEUE_MODE_ENABLED else "local",
                    "queue_db": str(WORKER_QUEUE_DB),
                    "summary": queue_summary(WORKER_QUEUE_DB),
                }
            )
            self.send_payload(status, payload, content_type)
            return
        if path == "/api/metrics":
            status, payload, content_type = json_bytes(business_metrics_payload())
            self.send_payload(status, payload, content_type)
            return
        status_match = re.fullmatch(r"/api/jobs/([^/]+)/status", path)
        if status_match:
            job_id = status_match.group(1)
            payload_obj = build_job_status(job_id)
            if payload_obj is None:
                status, payload, content_type = json_bytes({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            else:
                status, payload, content_type = json_bytes(payload_obj)
            self.send_payload(status, payload, content_type)
            return
        review_match = re.fullmatch(r"/api/jobs/([^/]+)/review", path)
        if review_match:
            job_id = review_match.group(1)
            job = get_job_state(job_id)
            if job is None:
                status, payload, content_type = json_bytes({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            review_path = Path(str(job.get("review_path", "")))
            if not review_path.exists():
                review_path = Path(str(job.get("job_root", ""))) / "review" / "review_manifest.json"
            if not review_path.exists():
                status, payload, content_type = json_bytes({"job_id": job_id, "items": [], "count": 0})
                self.send_payload(status, payload, content_type)
                return
            self.send_payload(HTTPStatus.OK.value, review_path.read_bytes(), "application/json; charset=utf-8")
            return
        review_html_match = re.fullmatch(r"/api/jobs/([^/]+)/review\.html", path)
        if review_html_match:
            job_id = review_html_match.group(1)
            job = get_job_state(job_id)
            if job is None:
                status, payload, content_type = json_bytes({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            html_path = Path(str(job.get("review_html_path", "")))
            if not html_path.exists():
                html_path = Path(str(job.get("job_root", ""))) / "review" / "review_report.html"
            if not html_path.exists():
                status, payload, content_type = json_bytes({"error": "Review report is missing"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            self.send_payload(
                HTTPStatus.OK.value,
                html_path.read_bytes(),
                "text/html; charset=utf-8",
                {"Content-Disposition": f'inline; filename="review_{job_id}.html"'},
            )
            return
        review_zip_match = re.fullmatch(r"/api/jobs/([^/]+)/review\.zip", path)
        if review_zip_match:
            job_id = review_zip_match.group(1)
            job = get_job_state(job_id)
            if job is None:
                status, payload, content_type = json_bytes({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            zip_path = Path(str(job.get("review_zip_path", "")))
            if not zip_path.exists():
                zip_path = Path(str(job.get("job_root", ""))) / "review" / "review_package.zip"
            if not zip_path.exists():
                status, payload, content_type = json_bytes({"error": "Review package is missing"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            self.send_payload(
                HTTPStatus.OK.value,
                zip_path.read_bytes(),
                "application/zip",
                {"Content-Disposition": f'attachment; filename="review_{job_id}.zip"'},
            )
            return
        artifact_match = re.fullmatch(r"/api/jobs/([^/]+)/artifact", path)
        if artifact_match:
            job_id = artifact_match.group(1)
            job = get_job_state(job_id)
            if job is None:
                status, payload, content_type = json_bytes({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            rel_path = parse_qs(urlparse(self.path).query).get("path", [""])[0].replace("\\", "/")
            job_root = Path(str(job.get("job_root", ""))).resolve()
            artifact_path = (job_root / rel_path).resolve()
            if not rel_path or job_root not in artifact_path.parents or not artifact_path.exists():
                status, payload, content_type = json_bytes({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            content_type = mimetypes.guess_type(str(artifact_path))[0] or "application/octet-stream"
            self.send_payload(HTTPStatus.OK.value, artifact_path.read_bytes(), content_type)
            return
        csv_match = re.fullmatch(r"/api/jobs/([^/]+)/csv", path)
        if csv_match:
            job_id = csv_match.group(1)
            job = get_job_state(job_id)
            if job is None:
                status, payload, content_type = json_bytes({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            if job.get("status") != "done":
                status, payload, content_type = json_bytes({"error": "CSV is not ready"}, HTTPStatus.ACCEPTED)
                self.send_payload(status, payload, content_type)
                return
            csv_path = Path(str(job.get("csv_path", "")))
            if not csv_path.exists():
                status, payload, content_type = json_bytes({"error": "CSV file is missing"}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, payload, content_type)
                return
            payload = csv_path.read_bytes()
            self.send_payload(
                HTTPStatus.OK.value,
                payload,
                "text/csv; charset=utf-8",
                {
                    "Content-Disposition": f'attachment; filename="lenta_submission_{job_id}.csv"',
                    "X-Lenta-Job-Id": job_id,
                    "X-Lenta-Rows": str(job.get("rows", 0)),
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/upload/create":
            try:
                payload = read_json_request(self)
                files = payload.get("files", [])
                if not isinstance(files, list):
                    raise ValueError("files must be a list")
                result = make_upload_job([item for item in files if isinstance(item, dict)])
                status, response, content_type = json_bytes(result, HTTPStatus.ACCEPTED)
                self.send_payload(status, response, content_type)
            except Exception as exc:
                status, response, content_type = json_bytes({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_payload(status, response, content_type)
            return

        chunk_match = re.fullmatch(r"/api/jobs/([^/]+)/upload-chunk", path)
        if chunk_match:
            job_id = chunk_match.group(1)
            try:
                query = parse_qs(urlparse(self.path).query)
                upload_id = query.get("upload_id", [""])[0]
                offset = int(query.get("offset", ["0"])[0])
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("Empty upload chunk")
                if content_length > MAX_UPLOAD_CHUNK_BYTES:
                    raise ValueError("Upload chunk is too large")
                body = self.rfile.read(content_length)
                result = write_upload_chunk(job_id, upload_id, offset, body)
                status, response, content_type = json_bytes(result)
                self.send_payload(status, response, content_type)
            except KeyError as exc:
                status, response, content_type = json_bytes({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, response, content_type)
            except Exception as exc:
                status, response, content_type = json_bytes({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_payload(status, response, content_type)
            return

        start_match = re.fullmatch(r"/api/jobs/([^/]+)/start", path)
        if start_match:
            job_id = start_match.group(1)
            try:
                result = start_uploaded_job(job_id)
                status, response, content_type = json_bytes(result, HTTPStatus.ACCEPTED)
                self.send_payload(status, response, content_type, {"X-Lenta-Job-Id": job_id})
            except KeyError as exc:
                status, response, content_type = json_bytes({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                self.send_payload(status, response, content_type)
            except Exception as exc:
                mark_job_failed_in_db(job_id, str(exc))
                status, response, content_type = json_bytes({"error": str(exc), "job_id": job_id}, HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_payload(status, response, content_type)
            return

        if path != "/api/infer":
            status, payload, content_type = json_bytes({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            self.send_payload(status, payload, content_type)
            return

        started = time.time()
        job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        job_root = JOBS_ROOT / job_id
        upload_dir = job_root / "uploads"
        outputs_dir = job_root / "outputs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Empty request body")
            if content_length > MAX_UPLOAD_BYTES:
                raise ValueError("Uploaded videos are too large for this local server limit")
            body = self.rfile.read(content_length)
            uploads = save_uploaded_videos(self.headers, body, upload_dir)
            if not uploads:
                raise ValueError("No MP4 files were uploaded")
            video_durations = [round(mp4_duration_seconds(path), 3) for path in uploads]

            set_job_state(
                job_id,
                status="queued",
                started_at=started,
                total_videos=len(uploads),
                video_durations=video_durations,
                completed_videos=0,
                job_root=str(job_root),
                upload_dir=str(upload_dir),
                outputs_dir=str(outputs_dir),
                stage="Видео загружено",
                detail="",
                rows=0,
            )
            upsert_job_record(
                job_id,
                status="queued",
                created_at=started,
                total_videos=len(uploads),
                job_root=str(job_root),
            )
            task_id = ""
            if QUEUE_MODE_ENABLED:
                task_id = enqueue_inference_task(job_id, uploads, job_root, outputs_dir, started)
                set_job_state(
                    job_id,
                    execution_mode="queue",
                    queue_task_id=task_id,
                    stage="В очереди worker",
                    detail="Ожидаем свободный обработчик",
                )
            else:
                worker = threading.Thread(
                    target=run_job,
                    args=(job_id, uploads, job_root, outputs_dir, started),
                    daemon=True,
                    name=f"lenta-job-{job_id}",
                )
                worker.start()
            status, payload, content_type = json_bytes(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "execution_mode": "queue" if QUEUE_MODE_ENABLED else "local",
                    "queue_task_id": task_id,
                    "status_url": f"/api/jobs/{job_id}/status",
                    "csv_url": f"/api/jobs/{job_id}/csv",
                },
                HTTPStatus.ACCEPTED,
            )
            self.send_payload(
                status,
                payload,
                content_type,
                {
                    "X-Lenta-Job-Id": job_id,
                },
            )
        except Exception as exc:
            mark_job_failed_in_db(job_id, str(exc))
            error_path = job_root / "error.txt"
            error_path.write_text(str(exc), encoding="utf-8", errors="replace")
            status, payload, content_type = json_bytes(
                {
                    "error": str(exc),
                    "job_id": job_id,
                    "seconds": round(time.time() - started, 3),
                    "error_log": str(error_path),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            self.send_payload(status, payload, content_type)


def prune_old_jobs(max_jobs: int = 25) -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = sorted([path for path in JOBS_ROOT.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)
    for old_job in jobs[max_jobs:]:
        shutil.rmtree(old_job, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Lenta web inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()

    init_results_db()
    init_queue_db(WORKER_QUEUE_DB)
    prune_old_jobs()
    httpd = ThreadingHTTPServer((args.host, args.port), LentaHandler)
    print(f"Serving Lenta web inference on http://{args.host}:{args.port}")
    print(f"Pipeline root: {PIPELINE_ROOT}")
    print(f"Detector checkpoint: {choose_checkpoint()}")
    print(f"Execution mode: {'queue' if QUEUE_MODE_ENABLED else 'local'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
