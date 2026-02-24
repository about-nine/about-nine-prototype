import os
import time
import traceback

from firebase_admin import firestore

from backend.services.firestore import get_firestore
from backend.services.analysis_service import analyze_talk_pipeline


def _now_ms() -> int:
    return int(time.time() * 1000)


def _claim_job(db, job_ref) -> bool:
    @firestore.transactional
    def _claim(transaction):
        snap = job_ref.get(transaction=transaction)
        if not snap.exists:
            return False
        data = snap.to_dict() or {}
        if data.get("status") != "queued":
            return False
        attempts = int(data.get("attempts") or 0) + 1
        transaction.update(
            job_ref,
            {
                "status": "running",
                "started_at": _now_ms(),
                "updated_at": _now_ms(),
                "attempts": attempts,
            },
        )
        return True

    txn = db.transaction()
    return _claim(txn)


def _complete_job(job_ref, status: str, error: str | None = None, trace: str | None = None):
    payload = {
        "status": status,
        "updated_at": _now_ms(),
    }
    if status in {"complete", "failed"}:
        payload["completed_at"] = _now_ms()
    if error:
        payload["last_error"] = error
    if trace:
        payload["last_trace"] = trace
    job_ref.update(payload)


def _process_job(db, job_ref, job):
    talk_id = job.get("talk_id") or job_ref.id
    try:
        result = analyze_talk_pipeline(talk_id, force=True)
        if result.get("success"):
            _complete_job(job_ref, "complete")
        else:
            msg = result.get("message") or "analysis_failed"
            _complete_job(job_ref, "failed", error=msg)
    except Exception as exc:  # noqa: BLE001
        _complete_job(job_ref, "failed", error=str(exc), trace=traceback.format_exc())


def run_worker():
    poll_seconds = float(os.getenv("ANALYSIS_WORKER_POLL_SECONDS", "3"))
    batch_size = int(os.getenv("ANALYSIS_WORKER_BATCH", "5"))
    run_once = os.getenv("ANALYSIS_WORKER_ONCE", "").lower() in {"1", "true", "yes"}

    db = get_firestore()
    jobs_ref = db.collection("analysis_jobs")

    while True:
        try:
            query = jobs_ref.where("status", "==", "queued").limit(batch_size)
            snaps = list(query.stream())
        except Exception:  # noqa: BLE001
            snaps = []

        if not snaps:
            if run_once:
                return
            time.sleep(poll_seconds)
            continue

        for snap in snaps:
            job_ref = snap.reference
            if not _claim_job(db, job_ref):
                continue
            job = snap.to_dict() or {}
            _process_job(db, job_ref, job)

        if run_once:
            return


if __name__ == "__main__":
    run_worker()
