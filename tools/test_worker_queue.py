from __future__ import annotations

import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

from worker_queue import claim_next_task, complete_task, enqueue_task, fail_task, queue_summary  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "queue.sqlite"
        task_ids = [
            enqueue_task(db_path, "dry_run", {"index": index}, source_job_id=f"job_{index}")
            for index in range(6)
        ]

        claimed_by_a = claim_next_task(db_path, "worker-a", lease_seconds=30)
        claimed_by_b = claim_next_task(db_path, "worker-b", lease_seconds=30)
        assert claimed_by_a is not None
        assert claimed_by_b is not None
        assert claimed_by_a["task_id"] != claimed_by_b["task_id"]

        complete_task(db_path, str(claimed_by_a["task_id"]), "worker-a", {"ok": True})
        fail_task(db_path, str(claimed_by_b["task_id"]), "worker-b", "temporary failure")

        reclaimed = claim_next_task(db_path, "worker-c", lease_seconds=30)
        assert reclaimed is not None
        assert reclaimed["task_id"] == claimed_by_b["task_id"]
        complete_task(db_path, str(reclaimed["task_id"]), "worker-c", {"retry": True})

        remaining = []
        while True:
            task = claim_next_task(db_path, "worker-d", lease_seconds=30)
            if task is None:
                break
            remaining.append(str(task["task_id"]))
            complete_task(db_path, str(task["task_id"]), "worker-d", {"ok": True})

        summary = queue_summary(db_path)
        assert summary["counts"].get("done") == len(task_ids)
        assert len(set(remaining)) == len(remaining)
        print("worker queue mini-test passed")


if __name__ == "__main__":
    main()
