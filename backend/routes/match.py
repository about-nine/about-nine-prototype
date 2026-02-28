# routes/match.py - matching, recommendations, and analysis queue endpoints
import time
from flask import Blueprint, request, jsonify, session
from firebase_admin import firestore
from backend.services.matching_service import (
    filter_users_for_list,
    BYPASS_USER_FILTERS,
    recommend_for_user,
)
from backend.services.firestore import get_firestore

match_bp = Blueprint("match", __name__, url_prefix="/api/match")
analysis_bp = Blueprint("analysis", __name__, url_prefix="/api/analysis")


def _enqueue_analysis_job(db, talk_id: str, requested_by: str | None = None) -> str:
    jobs_ref = db.collection("analysis_jobs").document(talk_id)
    now_ms = int(time.time() * 1000)

    @firestore.transactional
    def _enqueue(transaction):
        snap = jobs_ref.get(transaction=transaction)
        if snap.exists:
            data = snap.to_dict() or {}
            status = data.get("status")
            if status in {"queued", "running"}:
                return status
        transaction.set(
            jobs_ref,
            {
                "talk_id": talk_id,
                "status": "queued",
                "created_at": now_ms,
                "updated_at": now_ms,
                "requested_by": requested_by,
                "attempts": 0,
            },
            merge=True,
        )
        return "queued"

    try:
        txn = db.transaction()
        return _enqueue(txn)
    except Exception:
        # Best-effort enqueue; caller will still return "started".
        return "queued"


def enqueue_analysis_for_talk(talk_id: str, requested_by: str | None = None):
    db = get_firestore()
    talk_ref = db.collection("talk_history").document(talk_id)
    now_ms = int(time.time() * 1000)

    @firestore.transactional
    def _claim(transaction):
        snap = talk_ref.get(transaction=transaction)
        if not snap.exists:
            return "missing"
        talk = snap.to_dict() or {}
        analysis = talk.get("analysis") or {}
        if analysis.get("chemistry_score") is not None:
            return "complete"
        status = talk.get("analysis_status")
        started = int(talk.get("analysis_started_at") or 0)

        # 🔥 3분 이상 "running"이면 재시도 허용
        if status == "running" and started and now_ms - started < 180_000:
            return "running"

        transaction.update(
            talk_ref,
            {"analysis_status": "running", "analysis_started_at": now_ms},
        )
        return "start"

    try:
        txn = db.transaction()
        state = _claim(txn)
    except Exception:
        state = "start"

    if state == "missing":
        return {"success": False, "message": "talk_history not found"}, 404
    if state == "complete":
        return {"success": True, "talk_id": talk_id, "status": "complete"}, 200
    if state == "running":
        return {"success": True, "talk_id": talk_id, "status": "running"}, 200

    _enqueue_analysis_job(db, talk_id, requested_by=requested_by)
    return {"success": True, "talk_id": talk_id, "status": "started"}, 200


@analysis_bp.route("/talk", methods=["POST"])
def analyze_talk_api():
    data = request.get_json() or {}
    talk_id = data.get("talk_id")

    if not talk_id:
        return jsonify(success=False, message="talk_id required"), 400

    payload, status = enqueue_analysis_for_talk(
        talk_id,
        requested_by=session.get("user_id"),
    )
    return jsonify(payload), status


@match_bp.route("/list", methods=["GET"])
def list_candidates():
    uid = session.get("user_id")
    if not uid:
        return jsonify(success=False), 401

    users, debug, error = filter_users_for_list(uid, bypass_filters=BYPASS_USER_FILTERS)
    if error:
        status, message = error
        return jsonify(success=False, message=message), status

    if debug and not debug.get("bypass"):
        print(f"\n=== MATCH LIST DEBUG ===")
        print(f"My ID: {uid}")
        print(f"My profile: id={debug.get('me', {}).get('id')}, talk_count={debug.get('me', {}).get('talk_profile', {}).get('talk_count')}")
        total_count = debug.get("total_count", 0)
        filtered_stats = debug.get("filtered_stats") or {}
        print(f"\nTotal users in DB: {total_count}")
        print(f"Filter results:")
        for key, value in filtered_stats.items():
            print(f"  {key}: {value}")
        print(f"Final result: {len(users)} users")
        print("======================\n")

    return jsonify(success=True, users=users)


@match_bp.route("/recommend", methods=["GET"])
def recommend():
    uid = session.get("user_id") or request.headers.get("X-User-ID")
    if not uid:
        return jsonify(success=False, message="not logged in"), 401
    users = recommend_for_user(uid)
    db = get_firestore()
    results = []
    for user_id, _score in users:
        doc = db.collection("users").document(user_id).get()
        if not doc.exists:
            continue
        data = doc.to_dict() or {}
        if "id" not in data:
            data["id"] = doc.id
        results.append(data)
    return jsonify(users=results)
