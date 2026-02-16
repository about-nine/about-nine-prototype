# match.py
import threading
import time
from flask import Blueprint, request, jsonify, session
from firebase_admin import firestore
from backend.services.recommend_service import recommend_for_user
from backend.services.firestore import get_firestore

match_bp = Blueprint("match", __name__, url_prefix="/api/match")


def _run_analysis_background(talk_id: str):
    """백그라운드 스레드에서 분석 실행 (로깅 포함)"""
    from backend.services.firestore import get_firestore
    db = get_firestore()
    talk_ref = db.collection("talk_history").document(talk_id)
    
    print(f"🚀 [{talk_id}] Starting background analysis...")
    
    try:
        from backend.services.analysis_service import analyze_talk_pipeline
        
        print(f"🔬 [{talk_id}] Calling analyze_talk_pipeline...")
        result = analyze_talk_pipeline(talk_id, force=True)
        
        if result.get("success"):
            print(f"✅ [{talk_id}] Analysis complete")
        else:
            print(f"⚠️ [{talk_id}] Analysis failed: {result.get('message')}")
        
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"❌ [{talk_id}] Background analysis exception: {e}")
        print(err_trace)
        
        try:
            talk_ref.update({
                "analysis_error": str(e),
                "analysis_trace": err_trace,
                "analysis_failed_at": int(time.time() * 1000),
                "analysis_status": "failed",
            })
        except Exception:
            pass


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
        return jsonify(success=False, message="talk_history not found"), 404
    if state == "complete":
        return jsonify(success=True, talk_id=talk_id, status="complete")
    if state == "running":
        return jsonify(success=True, talk_id=talk_id, status="running")

    # 백그라운드 스레드로 분석 시작
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