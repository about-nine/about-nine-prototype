# match.py
import threading
import time
from flask import Blueprint, request, jsonify, session
from firebase_admin import firestore
from backend.services.recommend_service import recommend_for_user
from backend.services.firestore import get_firestore

match_bp = Blueprint("match", __name__, url_prefix="/api/match")


def _run_analysis_background(talk_id: str):
    """백그라운드 스레드에서 분석 실행"""
    try:
        from backend.services.analysis_service import analyze_talk_pipeline
        analyze_talk_pipeline(talk_id, force=True)
    except Exception as e:
        # 에러는 analysis_service 내부에서 Firestore에 기록됨
        print(f"⚠️ Background analysis failed for {talk_id}: {e}")


@match_bp.route("/analyze-talk", methods=["POST"])
def analyze_talk():
    data = request.get_json() or {}
    talk_id = data.get("talk_id")

    if not talk_id:
        return jsonify(success=False, message="talk_id required"), 400

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
        return jsonify(success=False, message="talk_history not found"), 404
    if state == "complete":
        return jsonify(success=True, talk_id=talk_id, status="complete")
    if state == "running":
        return jsonify(success=True, talk_id=talk_id, status="running")

    # 백그라운드 스레드로 분석 시작 → 즉시 응답
    thread = threading.Thread(
        target=_run_analysis_background,
        args=(talk_id,),
        daemon=True,
    )
    thread.start()

    return jsonify(success=True, talk_id=talk_id, status="started")


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
