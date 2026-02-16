# match.py
import threading
from flask import Blueprint, request, jsonify, session
from backend.services.recommend_service import recommend_for_user
from backend.services.firestore import get_firestore

match_bp = Blueprint("match", __name__, url_prefix="/api/match")


def _run_analysis_background(talk_id: str):
    """백그라운드 스레드에서 분석 실행"""
    try:
        from backend.services.analysis_service import analyze_talk_pipeline
        analyze_talk_pipeline(talk_id)
    except Exception as e:
        # 에러는 analysis_service 내부에서 Firestore에 기록됨
        print(f"⚠️ Background analysis failed for {talk_id}: {e}")


@match_bp.route("/analyze-talk", methods=["POST"])
def analyze_talk():
    data = request.get_json() or {}
    talk_id = data.get("talk_id")

    if not talk_id:
        return jsonify(success=False, message="talk_id required"), 400

    # 이미 분석 완료됐는지 확인
    try:
        db = get_firestore()
        snap = db.collection("talk_history").document(talk_id).get()
        if snap.exists:
            talk = snap.to_dict() or {}
            analysis = talk.get("analysis") or {}
            if analysis.get("chemistry_score") is not None:
                return jsonify(success=True, talk_id=talk_id, status="complete")

            # 이미 running 중이면 중복 실행 방지
            if talk.get("analysis_status") == "running":
                started = talk.get("analysis_started_at") or 0
                elapsed = int(__import__("time").time() * 1000) - started
                # 3분 이상 running이면 stuck → 재실행 허용
                if elapsed < 180_000:
                    return jsonify(success=True, talk_id=talk_id, status="running")
    except Exception:
        pass

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