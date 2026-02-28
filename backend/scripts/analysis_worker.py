import os
import time
import traceback

from firebase_admin import firestore

from backend.services.firestore import get_firestore
from backend.services.analysis_service import analyze_talk_pipeline

import threading

JOB_TIMEOUT_SEC = int(os.getenv("ANALYSIS_JOB_TIMEOUT", "1800"))  # 기본 30분

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
    try:
        job_ref.update(payload)
    except Exception as e:
        print(f"⚠️ _complete_job failed (doc may not exist): {e}")


MAX_ATTEMPTS = 3

def _process_job(db, job_ref, job):
    talk_id = job.get("talk_id") or job_ref.id
    attempts = int(job.get("attempts") or 1)

    # 업로드 완료 여부 확인
    talk_snap = db.collection("talk_history").document(talk_id).get()
    if talk_snap.exists:
        upload_status = (talk_snap.to_dict() or {}).get("recording_uploading_status")
        if upload_status and upload_status != "uploaded":
            print(f"⏳ [{talk_id}] Not uploaded yet ({upload_status}), requeueing")
            job_ref.update({"status": "queued", "updated_at": _now_ms()})
            return

    result_box = {}

    def _run():
        try:
            result_box["result"] = analyze_talk_pipeline(talk_id, force=True)
        except Exception as exc:
            result_box["error"] = (str(exc), traceback.format_exc())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=JOB_TIMEOUT_SEC)

    if t.is_alive():
        print(f"❌ [{talk_id}] Job timed out after {JOB_TIMEOUT_SEC}s")
        if attempts < MAX_ATTEMPTS:
            job_ref.update({"status": "queued", "updated_at": _now_ms(), "last_error": "job_timeout"})
        else:
            _complete_job(job_ref, "failed", error="job_timeout")
        return

    if "error" in result_box:
        err, trace = result_box["error"]
        if attempts < MAX_ATTEMPTS:
            job_ref.update({"status": "queued", "updated_at": _now_ms(), "last_error": err, "last_trace": trace})
        else:
            _complete_job(job_ref, "failed", error=err, trace=trace)
        return

    result = result_box.get("result", {})
    if result.get("success"):
        _complete_job(job_ref, "complete")
    else:
        msg = result.get("message") or "analysis_failed"
        if attempts < MAX_ATTEMPTS:
            job_ref.update({"status": "queued", "updated_at": _now_ms(), "last_error": msg})
        else:
            _complete_job(job_ref, "failed", error=msg)


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
            try:
                if not _claim_job(db, job_ref):
                    continue
                updated_snap = job_ref.get()
                job = updated_snap.to_dict() or {}
                _process_job(db, job_ref, job)
            except Exception as e:
                print(f"⚠️ job loop error (skipping): {e}")
                continue

        if run_once:
            return


if __name__ == "__main__":
    run_worker()
