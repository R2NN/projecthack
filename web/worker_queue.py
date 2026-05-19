from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_LEASE_SECONDS = 60 * 60


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


@contextmanager
def queue_connection(db_path: Path | str):
    connection = connect(db_path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_queue_db(db_path: Path | str) -> None:
    with queue_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS worker_tasks (
              task_id TEXT PRIMARY KEY,
              task_type TEXT NOT NULL,
              source_job_id TEXT,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL,
              priority INTEGER NOT NULL DEFAULT 100,
              attempts INTEGER NOT NULL DEFAULT 0,
              max_attempts INTEGER NOT NULL DEFAULT 3,
              lease_owner TEXT,
              lease_until REAL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              started_at REAL,
              finished_at REAL,
              error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_worker_tasks_status ON worker_tasks(status, priority, created_at);
            CREATE INDEX IF NOT EXISTS idx_worker_tasks_source_job ON worker_tasks(source_job_id);

            CREATE TABLE IF NOT EXISTS worker_heartbeats (
              worker_id TEXT PRIMARY KEY,
              host TEXT,
              pid INTEGER,
              last_seen REAL NOT NULL,
              metadata_json TEXT
            );
            """
        )


def _row_to_task(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    task = dict(row)
    try:
        task["payload"] = json.loads(str(task.pop("payload_json", "{}")))
    except json.JSONDecodeError:
        task["payload"] = {}
    return task


def enqueue_task(
    db_path: Path | str,
    task_type: str,
    payload: dict[str, Any],
    *,
    source_job_id: str = "",
    priority: int = 100,
    max_attempts: int = 3,
    task_id: str | None = None,
) -> str:
    init_queue_db(db_path)
    now = time.time()
    task_id = task_id or f"task_{uuid.uuid4().hex}"
    with queue_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO worker_tasks (
              task_id, task_type, source_job_id, payload_json, status, priority,
              attempts, max_attempts, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 'queued', ?, 0, ?, ?, ?)
            """,
            (
                task_id,
                task_type,
                source_job_id,
                json.dumps(payload, ensure_ascii=False),
                int(priority),
                int(max_attempts),
                now,
                now,
            ),
        )
    return task_id


def heartbeat(db_path: Path | str, worker_id: str, metadata: dict[str, Any] | None = None) -> None:
    init_queue_db(db_path)
    now = time.time()
    with queue_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO worker_heartbeats (worker_id, host, pid, last_seen, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
              host = excluded.host,
              pid = excluded.pid,
              last_seen = excluded.last_seen,
              metadata_json = excluded.metadata_json
            """,
            (
                worker_id,
                socket.gethostname(),
                os.getpid(),
                now,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )


def claim_next_task(
    db_path: Path | str,
    worker_id: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    task_types: list[str] | None = None,
) -> dict[str, Any] | None:
    init_queue_db(db_path)
    now = time.time()
    params: list[Any] = [now]
    task_filter = ""
    if task_types:
        placeholders = ", ".join("?" for _ in task_types)
        task_filter = f" AND task_type IN ({placeholders})"
        params.extend(task_types)

    with queue_connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            f"""
            SELECT *
            FROM worker_tasks
            WHERE
              (
                status = 'queued'
                OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
              )
              AND attempts < max_attempts
              {task_filter}
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is None:
            connection.commit()
            return None

        task_id = str(row["task_id"])
        connection.execute(
            """
            UPDATE worker_tasks
            SET
              status = 'running',
              attempts = attempts + 1,
              lease_owner = ?,
              lease_until = ?,
              started_at = COALESCE(started_at, ?),
              updated_at = ?,
              error = ''
            WHERE task_id = ?
            """,
            (worker_id, now + int(lease_seconds), now, now, task_id),
        )
        claimed = connection.execute("SELECT * FROM worker_tasks WHERE task_id = ?", (task_id,)).fetchone()
        connection.commit()
    return _row_to_task(claimed)


def complete_task(db_path: Path | str, task_id: str, worker_id: str, result: dict[str, Any] | None = None) -> None:
    now = time.time()
    with queue_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE worker_tasks
            SET status = 'done', finished_at = ?, updated_at = ?, lease_until = NULL, error = ?
            WHERE task_id = ? AND lease_owner = ?
            """,
            (now, now, json.dumps(result or {}, ensure_ascii=False), task_id, worker_id),
        )


def fail_task(db_path: Path | str, task_id: str, worker_id: str, error: str) -> None:
    now = time.time()
    with queue_connection(db_path) as connection:
        row = connection.execute(
            "SELECT attempts, max_attempts FROM worker_tasks WHERE task_id = ? AND lease_owner = ?",
            (task_id, worker_id),
        ).fetchone()
        if row is None:
            return
        final_status = "failed" if int(row["attempts"]) >= int(row["max_attempts"]) else "queued"
        connection.execute(
            """
            UPDATE worker_tasks
            SET status = ?, finished_at = CASE WHEN ? = 'failed' THEN ? ELSE finished_at END,
                updated_at = ?, lease_until = NULL, error = ?
            WHERE task_id = ? AND lease_owner = ?
            """,
            (final_status, final_status, now, now, error[:4000], task_id, worker_id),
        )


def get_task_by_source_job(db_path: Path | str, source_job_id: str) -> dict[str, Any] | None:
    init_queue_db(db_path)
    with queue_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM worker_tasks
            WHERE source_job_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (source_job_id,),
        ).fetchone()
    return _row_to_task(row)


def queue_summary(db_path: Path | str) -> dict[str, Any]:
    init_queue_db(db_path)
    with queue_connection(db_path) as connection:
        status_rows = connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM worker_tasks
            GROUP BY status
            """
        ).fetchall()
        workers = connection.execute(
            """
            SELECT worker_id, host, pid, last_seen, metadata_json
            FROM worker_heartbeats
            ORDER BY last_seen DESC
            LIMIT 25
            """
        ).fetchall()
        recent = connection.execute(
            """
            SELECT task_id, task_type, source_job_id, status, attempts, max_attempts,
                   lease_owner, created_at, updated_at, finished_at, error
            FROM worker_tasks
            ORDER BY updated_at DESC
            LIMIT 20
            """
        ).fetchall()
    now = time.time()
    return {
        "counts": {str(row["status"]): int(row["count"]) for row in status_rows},
        "workers": [
            {
                "worker_id": row["worker_id"],
                "host": row["host"],
                "pid": row["pid"],
                "last_seen": row["last_seen"],
                "seconds_since_seen": round(now - float(row["last_seen"]), 3),
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in workers
        ],
        "recent_tasks": [dict(row) for row in recent],
    }

