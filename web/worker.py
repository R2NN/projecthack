from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import server
from worker_queue import claim_next_task, complete_task, fail_task, heartbeat


def run_video_inference_task(task: dict[str, object], worker_id: str) -> None:
    payload = task.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("Task payload must be a JSON object")
    job_id = str(payload.get("job_id", ""))
    uploads = [Path(str(path)) for path in payload.get("uploads", [])]
    job_root = Path(str(payload.get("job_root", "")))
    outputs_dir = Path(str(payload.get("outputs_dir", "")))
    started = float(payload.get("started", time.time()) or time.time())
    if not job_id or not uploads or not job_root or not outputs_dir:
        raise ValueError("Task payload is missing required fields")
    server.set_job_state(
        job_id,
        status="running",
        execution_mode="queue",
        queue_task_id=str(task.get("task_id", "")),
        queue_worker=worker_id,
        job_root=str(job_root),
        outputs_dir=str(outputs_dir),
        stage="Worker запустил пайплайн",
        detail=f"worker={worker_id}",
    )
    server.run_job(job_id, uploads, job_root, outputs_dir, started)
    final_state = server.get_job_state(job_id) or {}
    if str(final_state.get("status", "")) == "failed":
        raise RuntimeError(str(final_state.get("error", "Pipeline failed")))


def handle_task(task: dict[str, object], worker_id: str, dry_run: bool) -> dict[str, object]:
    task_type = str(task.get("task_type", ""))
    if dry_run or task_type == "dry_run":
        time.sleep(0.05)
        return {"dry_run": True, "worker_id": worker_id}
    if task_type == "video_inference":
        run_video_inference_task(task, worker_id)
        return {"job_id": str(task.get("source_job_id", "")), "worker_id": worker_id}
    raise ValueError(f"Unsupported task type: {task_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shelf Vision worker process")
    parser.add_argument("--queue-db", default=str(server.WORKER_QUEUE_DB))
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=int, default=60 * 60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    worker_id = args.worker_id or f"{os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or 'worker'}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    print(f"Worker started: {worker_id}")
    while True:
        heartbeat(args.queue_db, worker_id, {"dry_run": args.dry_run})
        task = claim_next_task(args.queue_db, worker_id, lease_seconds=args.lease_seconds)
        if task is None:
            if args.once:
                print("No tasks available")
                return
            time.sleep(max(0.2, args.poll_seconds))
            continue
        task_id = str(task.get("task_id", ""))
        print(f"Claimed task {task_id} ({task.get('task_type')})")
        try:
            result = handle_task(task, worker_id, args.dry_run)
            complete_task(args.queue_db, task_id, worker_id, result)
            print(f"Completed task {task_id}")
        except Exception as exc:
            fail_task(args.queue_db, task_id, worker_id, str(exc))
            print(f"Failed task {task_id}: {exc}")
        if args.once:
            return


if __name__ == "__main__":
    main()
