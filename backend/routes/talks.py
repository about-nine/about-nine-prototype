"""
talks.py - 대화 관련 엔드포인트

- calculate-round: 다음 라운드 계산
- save-talk-history: 대화 저장 (클라이언트에서 호출하지 않음, talk-end.html에서 직접 저장)
- get-talk-history: 대화 기록 조회
"""

from datetime import datetime
from flask import Blueprint, jsonify, session, request
from firebase_admin import firestore
from google.api_core.exceptions import AlreadyExists
from backend.services.firestore import get_firestore
from backend.services.rtdb import get_rtdb
from backend.services.user_profile_service import update_user_embedding, update_user_stats
from backend.utils.request import get_json
import random
import os
import time

talks_bp = Blueprint("talks", __name__, url_prefix="/api/talks")

# =========================
# 질문 풀
# =========================
QUESTIONS = {
    "food": [
        "What are you craving right now?",
        "What do you want to eat when you're stressed?",
        "If you could only eat one food for three years, what would it be?",
        "What's your soul food?",
        "What do you want to eat when you need comfort?",
        "Which food best represents your taste?",
        "What did you have for dinner most recently?",
        "What would you want to cook for your partner?",
        "What tastes even better when you're in a good mood?",
        "What would you eat to cure a hangover?",
        "What would you want as your last meal?",
        "What would you want to eat on a first date?",
        "What tastes better when you eat alone?",
        "Which one appeals to you the least?",
        "What do you want for lunch tomorrow?",
        "What would you eat right after ending a diet?",
        "What would you want to cook together?",
        "What would you serve at a housewarming party?",
        "Which one would make you like someone more if they chose it?",
        "Which one do you think we'd both choose?"
    ],
    "visual": [
        "Which painting resonates with you the most?",
        "Which painting would you choose as a gift for someone you care about?",
        "If you were opening a café, which painting would you hang?",
        "Which painting would you want to see on your daily commute?",
        "Which one caught your eye within 3 seconds?",
        "Which painting would suit a hotel lobby?",
        "Which painting would you hang in your bedroom?",
        "Which painting would you look at when you need energy?",
        "Which painting would you want to see when you're feeling down?",
        "Which painting would you want to show someone on a first date?",
        "Which painting do you think your parents would like?",
        "Which painting best represents who you are?",
        "Which choice would surprise your friends?",
        "Which painting would make you more attracted to someone if they chose it?",
        "Which painting would worry you a little if someone chose it?",
        "Which painting would you want to see right after a breakup?",
        "Which painting would you look at before a new beginning?",
        "Which painting feels most valuable to you?"
    ]
}

# =========================
# 이미지 카테고리
# =========================
IMAGE_CATEGORIES = {
    "food": ["italian", "pizza", "others", "dessert", "bread"],
    "visual": ["abstract", "landscape", "portrait"]
}

# =========================
# 이미지 파일 스캔
# =========================
def get_image_files(topic, category):
    """실제 파일 시스템에서 이미지 파일 목록 가져오기"""
    base_path = os.path.join("frontend", "images", topic, category)
    
    if not os.path.exists(base_path):
        print(f"⚠️ Path not found: {base_path}")
        return []
    
    files = [f for f in os.listdir(base_path) 
             if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    return files


# =========================
# 🔥 Calculate Round
# =========================
@talks_bp.route("/calculate-round", methods=["POST"])
def calculate_round():
    """
    두 사용자 간의 다음 대화 라운드 계산
    
    Request:
    {
        "partner_id": "user_xxx"
    }
    
    Response:
    {
        "success": true,
        "round": 1,
        "topic": "food",
        "question": "What are you craving?",
        "options": [...]
    }
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code
    
    partner_id = data.get("partner_id")
    if not partner_id:
        return jsonify(success=False, message="partner_id required"), 400

    try:
        db = get_firestore()

        # 🔥 talk_history (top-level)에서 완료된 대화 수 계산
        completed_count = count_completed_talks(db, user_id, partner_id)

        # 다음 라운드 (최대 3)
        next_round = min(completed_count + 1, 3)

        # Round별 Topic
        topics = {
            1: "food",
            2: "visual",
            3: "life" 
        }
        
        topic = topics[next_round]

        print(f"📊 Round 계산: {user_id} ↔ {partner_id}")
        print(f"   완료된 대화: {completed_count}개")
        print(f"   다음 Round: {next_round} ({topic})")

        response = {
            "success": True,
            "round": next_round,
            "topic": topic,
            "completed_talks": completed_count
        }

        # food/visual만 질문/옵션 제공
        if topic in ["food", "visual"]:
            # 이미 받은 질문들
            used_questions = get_used_questions(db, user_id, partner_id, topic)
            
            # 새 질문 선택
            question = select_new_question(topic, used_questions)
            
            # 랜덤 옵션 선택
            options = select_random_options(topic)
            
            response["question"] = question
            response["options"] = options
            
            print(f"   질문: {question}")
            print(f"   옵션: {len(options)}개")

        return jsonify(response)

    except Exception as e:
        print(f"❌ calculate-round 실패: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=str(e)), 500


# =========================
# Save Talk History (from match_request)
# =========================
@talks_bp.route("/save-history", methods=["POST"])
def save_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    request_id = data.get("request_id")
    if not request_id:
        return jsonify(success=False, message="request_id required"), 400

    rtdb = get_rtdb()
    if not rtdb:
        return jsonify(success=False, message="rtdb not configured"), 500

    match_request_ref = rtdb.child("match_requests").child(request_id)
    match_request = match_request_ref.get()
    if not match_request:
        # Fallback: match_request might be cleaned up; reuse existing talk_history if present.
        db = get_firestore()
        existing = (
            db.collection("talk_history")
            .where("match_request_id", "==", request_id)
            .limit(1)
            .stream()
        )
        existing_doc = next(existing, None)
        if existing_doc:
            return jsonify(success=True, talk_id=existing_doc.id)
        return jsonify(success=False, message="match_request not found"), 404

    def _build_patch_from_match(match_request_data, existing_data=None):
        existing_data = existing_data or {}
        patch = {}

        call_started = match_request_data.get("call_started_at")
        call_ended = match_request_data.get("ended_at")
        existing_ts = existing_data.get("timestamp") or 0

        if call_started:
            patch["call_started_at"] = call_started
        if call_ended:
            patch["call_ended_at"] = call_ended
            if call_started:
                patch["duration"] = int((call_ended - call_started) / 1000)
            if not existing_ts or call_ended > existing_ts:
                patch["timestamp"] = call_ended
        elif call_started and not existing_ts:
            patch["timestamp"] = call_started

        files = match_request_data.get("recording_file_list") or []
        if isinstance(files, list) and files:
            existing_files = existing_data.get("recording_files") or []
            if not isinstance(existing_files, list) or len(files) >= len(existing_files):
                patch["recording_files"] = files

        status = match_request_data.get("recording_uploading_status")
        if status:
            patch["recording_uploading_status"] = status

        uid_map = match_request_data.get("uid_mapping") or {}
        if isinstance(uid_map, dict) and uid_map:
            existing_map = existing_data.get("uid_mapping") or {}
            if not isinstance(existing_map, dict):
                existing_map = {}
            merged = {**existing_map, **uid_map}
            patch["uid_mapping"] = merged

        initiator = match_request_data.get("initiator")
        receiver = match_request_data.get("receiver")
        if initiator:
            initiator_selection = match_request_data.get("initiator_selection")
            if initiator_selection is not None:
                patch[f"selections.{initiator}"] = initiator_selection
        if receiver:
            receiver_selection = match_request_data.get("receiver_selection")
            if receiver_selection is not None:
                patch[f"selections.{receiver}"] = receiver_selection

        return patch

    existing_talk_id = match_request.get("talk_id")
    if existing_talk_id:
        db = get_firestore()
        talk_ref = db.collection("talk_history").document(existing_talk_id)
        existing_snap = talk_ref.get()
        if existing_snap.exists:
            patch = _build_patch_from_match(match_request, existing_snap.to_dict() or {})
            if patch:
                talk_ref.update(patch)
            return jsonify(success=True, talk_id=existing_talk_id)
        # stale talk_id in RTDB → continue to create/resolve a valid talk_history

    initiator = match_request.get("initiator")
    receiver = match_request.get("receiver")
    if not initiator or not receiver:
        return jsonify(success=False, message="invalid match_request"), 400

    initiator_selection = match_request.get("initiator_selection")
    receiver_selection = match_request.get("receiver_selection")

    call_started = match_request.get("call_started_at")
    call_ended = match_request.get("ended_at")
    duration = 0
    if call_started and call_ended:
        duration = int((call_ended - call_started) / 1000)

    talk_data = {
        "match_request_id": request_id,
        "participants": {"user_a": initiator, "user_b": receiver},
        "round": match_request.get("round", 1),
        "topic": match_request.get("topic", "food"),
        "question": match_request.get("question")
        or (
            "choose a topic and discuss it freely"
            if match_request.get("topic") == "life"
            else ""
        ),
        "options": match_request.get("options") or [],
        "selections": {
            initiator: initiator_selection,
            receiver: receiver_selection,
        },
        "completed": True,
        "timestamp": call_ended or call_started or int(time.time() * 1000),
        "duration": duration,
        "call_started_at": call_started,
        "call_ended_at": call_ended,
        "recording_files": match_request.get("recording_file_list") or [],
        "recording_uploading_status": match_request.get("recording_uploading_status"),
        "uid_mapping": match_request.get("uid_mapping") or {},
        "analysis": None,
        "created_at": int(time.time() * 1000),
    }

    db = get_firestore()
    # Idempotency: if talk_history already created for this request, reuse it.
    existing = (
        db.collection("talk_history")
        .where("match_request_id", "==", request_id)
        .limit(1)
        .stream()
    )
    existing_doc = next(existing, None)
    if existing_doc:
        talk_id = existing_doc.id
        patch = _build_patch_from_match(match_request, existing_doc.to_dict() or {})
        if patch:
            db.collection("talk_history").document(talk_id).update(patch)
        match_request_ref.update({"talk_id": talk_id})
        return jsonify(success=True, talk_id=talk_id)

    # Use request_id as deterministic talk_id to avoid duplicates on race.
    talk_ref = db.collection("talk_history").document(request_id)
    try:
        talk_ref.create(talk_data)
    except AlreadyExists:
        existing = talk_ref.get()
        if existing.exists:
            patch = _build_patch_from_match(match_request, existing.to_dict() or {})
            if patch:
                talk_ref.update(patch)
            match_request_ref.update({"talk_id": request_id})
            return jsonify(success=True, talk_id=request_id)
        raise

    match_request_ref.update({"talk_id": request_id})
    return jsonify(success=True, talk_id=request_id, talk=talk_data)


# =========================
# Get Talk History
# =========================
@talks_bp.route("/history/<talk_id>", methods=["GET"])
def get_history(talk_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    db = get_firestore()
    snap = db.collection("talk_history").document(talk_id).get()
    if not snap.exists:
        return jsonify(success=False, message="talk_history not found"), 404

    return jsonify(success=True, talk=snap.to_dict())


# =========================
# History List
# =========================
@talks_bp.route("/history-list", methods=["GET"])
def history_list():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    db = get_firestore()
    talks_ref = db.collection("talk_history")

    query1 = (
        talks_ref.where("participants.user_a", "==", user_id).stream()
    )
    query2 = (
        talks_ref.where("participants.user_b", "==", user_id).stream()
    )

    partner_map = {}

    def add_talk(talk):
        participants = talk.get("participants") or {}
        partner_id = (
            participants.get("user_b")
            if participants.get("user_a") == user_id
            else participants.get("user_a")
        )
        if not partner_id:
            return

        entry = partner_map.setdefault(
            partner_id,
            {
                "partner_id": partner_id,
                "talks_by_round": {},
                "last_timestamp": 0,
                "had_no": False,
            },
        )

        round_num = talk.get("round") or 1
        ts = talk.get("timestamp") or 0
        score = None
        if isinstance(talk.get("analysis"), dict):
            raw = talk.get("analysis", {}).get("chemistry_score")
            if isinstance(raw, (int, float)):
                score = round(raw)

        existing = entry["talks_by_round"].get(round_num)
        if not existing or ts > existing.get("ts", 0):
            entry["talks_by_round"][round_num] = {
                "topic": talk.get("topic"),
                "score": score,
                "ts": ts,
            }

        entry["last_timestamp"] = max(entry["last_timestamp"], ts)

        go_no_go = talk.get("go_no_go") or {}
        if isinstance(go_no_go, dict) and any(v is False for v in go_no_go.values()):
            entry["had_no"] = True
        elif talk.get("initiator_response") == "no" or talk.get("receiver_response") == "no":
            entry["had_no"] = True

    for doc in query1:
        add_talk(doc.to_dict() or {})
    for doc in query2:
        add_talk(doc.to_dict() or {})

    partner_ids = list(partner_map.keys())
    if not partner_ids:
        return jsonify(success=True, items=[])

    # current user's block list
    me = db.collection("users").document(user_id).get().to_dict() or {}
    my_blocked = set(me.get("blocked_users") or [])

    items = []
    for pid in partner_ids:
        user_doc = db.collection("users").document(pid).get()
        if not user_doc.exists:
            items.append(
                {
                    "partner_id": pid,
                    "first_name": "(deleted)",
                    "last_name": "",
                    "talks_by_round": partner_map[pid]["talks_by_round"],
                    "last_timestamp": partner_map[pid]["last_timestamp"],
                    "had_no": partner_map[pid]["had_no"],
                    "blocked": pid in my_blocked,
                    "deleted": True,
                }
            )
            continue

        user_data = user_doc.to_dict() or {}
        other_blocked = set(user_data.get("blocked_users") or [])
        is_blocked = pid in my_blocked or user_id in other_blocked

        items.append(
            {
                "partner_id": pid,
                "first_name": user_data.get("first_name") or user_data.get("firstName"),
                "last_name": user_data.get("last_name") or user_data.get("lastName"),
                "talks_by_round": partner_map[pid]["talks_by_round"],
                "last_timestamp": partner_map[pid]["last_timestamp"],
                "had_no": partner_map[pid]["had_no"],
                "blocked": is_blocked,
                "deleted": False,
            }
        )

    items.sort(key=lambda x: x.get("last_timestamp", 0), reverse=True)
    return jsonify(success=True, items=items)


# =========================
# History Detail
# =========================
@talks_bp.route("/history-detail/<partner_id>", methods=["GET"])
def history_detail(partner_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    db = get_firestore()
    partner_doc = db.collection("users").document(partner_id).get()
    partner_exists = partner_doc.exists
    partner = partner_doc.to_dict() or {}
    deleted = not partner_exists
    if deleted:
        partner = {
            "first_name": "(deleted)",
            "last_name": "",
            "phone": None,
        }

    # block check
    me = db.collection("users").document(user_id).get().to_dict() or {}
    my_blocked = set(me.get("blocked_users") or [])
    other_blocked = set(partner.get("blocked_users") or []) if partner_exists else set()
    is_blocked = partner_id in my_blocked or user_id in other_blocked

    talks_ref = db.collection("talk_history")
    query1 = (
        talks_ref.where("participants.user_a", "==", user_id)
        .where("participants.user_b", "==", partner_id)
        .stream()
    )
    query2 = (
        talks_ref.where("participants.user_a", "==", partner_id)
        .where("participants.user_b", "==", user_id)
        .stream()
    )

    rounds = {}
    had_no = False
    max_round = 0

    def add_detail(talk):
        nonlocal had_no, max_round
        round_num = talk.get("round") or 1
        max_round = max(max_round, round_num)
        ts = talk.get("timestamp") or 0

        score = None
        if isinstance(talk.get("analysis"), dict):
            raw = talk.get("analysis", {}).get("chemistry_score")
            if isinstance(raw, (int, float)):
                score = round(raw)

        existing = rounds.get(round_num)
        if existing and existing.get("timestamp", 0) >= ts:
            return

        rounds[round_num] = {
            "topic": talk.get("topic"),
            "score": score,
            "selections": talk.get("selections") or {},
            "options": talk.get("options") or [],
            "conversation": talk.get("conversation") or [],
            "uid_mapping": talk.get("uid_mapping") or {},
            "timestamp": ts,
        }

        go_no_go = talk.get("go_no_go") or {}
        if isinstance(go_no_go, dict) and any(v is False for v in go_no_go.values()):
            had_no = True
        elif talk.get("initiator_response") == "no" or talk.get("receiver_response") == "no":
            had_no = True

    for doc in query1:
        add_detail(doc.to_dict() or {})
    for doc in query2:
        add_detail(doc.to_dict() or {})

    return jsonify(
        success=True,
        blocked=is_blocked,
        partner={
            "id": partner_id,
            "first_name": partner.get("first_name") or partner.get("firstName"),
            "last_name": partner.get("last_name") or partner.get("lastName"),
            "phone": partner.get("phone"),
        },
        deleted=deleted,
        rounds=rounds,
        had_no=had_no,
        max_round=max_round,
    )


# =========================
# Save Response (go/no)
# =========================
@talks_bp.route("/respond", methods=["POST"])
def save_response():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    talk_id = data.get("talk_id")
    choice = data.get("choice")
    if not talk_id or choice not in ["go", "no"]:
        return jsonify(success=False, message="talk_id and choice required"), 400

    db = get_firestore()
    talk_ref = db.collection("talk_history").document(talk_id)
    snap = talk_ref.get()
    if not snap.exists:
        return jsonify(success=False, message="talk_history not found"), 404

    talk = snap.to_dict() or {}
    participants = talk.get("participants") or {}
    is_initiator = participants.get("user_a") == user_id
    is_receiver = participants.get("user_b") == user_id
    if not (is_initiator or is_receiver):
        return jsonify(success=False, message="not a participant"), 403

    go_no_go = talk.get("go_no_go") or {}
    merged = {**go_no_go, user_id: (choice == "go")}

    # Use update() so dotted field path is treated as nested map, not a literal key.
    talk_ref.update({f"go_no_go.{user_id}": choice == "go"})

    # Store go/no in RTDB for realtime sync
    try:
        rtdb = get_rtdb()
        if rtdb:
            request_id = talk.get("match_request_id") or talk_id
            match_ref = rtdb.child("match_requests").child(request_id)
            match_ref.child("go_no_go").child(user_id).set(choice == "go")
            match_ref.child("go_no_go_updated_at").set(int(time.time() * 1000))

            # If both participants responded, clean up match_request
            match_data = match_ref.get() or {}
            initiator = match_data.get("initiator")
            receiver = match_data.get("receiver")
            go_no_go = match_data.get("go_no_go") or {}
            if (
                initiator
                and receiver
                and initiator in go_no_go
                and receiver in go_no_go
            ):
                match_ref.delete()
    except Exception:
        pass

    # Update user stats (only if this is the first response from the user)
    try:
        if user_id not in go_no_go:
            update_user_stats(user_id, is_go=(choice == "go"))
    except Exception:
        pass

    # Update user embedding if analysis already exists
    try:
        analysis = talk.get("analysis") or {}
        pair_embedding = analysis.get("pair_embedding")
        embedding_updated = talk.get("embedding_updated") or {}
        if pair_embedding and not embedding_updated.get(user_id):
            updated = update_user_embedding(user_id, pair_embedding, go=(choice == "go"))
            if updated:
                talk_ref.update({f"embedding_updated.{user_id}": True})
    except Exception:
        pass

    if choice == "no":
        partner_id = participants.get("user_b") if is_initiator else participants.get("user_a")
        if partner_id:
            db.collection("users").document(user_id).set(
                {
                    "blocked_users": firestore.ArrayUnion([partner_id]),
                    "blocked_updated_at": datetime.utcnow().isoformat(),
                },
                merge=True,
            )

    return jsonify(success=True)


# =========================
# Realtime Go/No status (RTDB)
# =========================
@talks_bp.route("/go-no-go/<request_id>", methods=["GET"])
def get_go_no_go(request_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    rtdb = get_rtdb()
    if not rtdb:
        return jsonify(success=False, message="rtdb not configured"), 500

    match_request_ref = rtdb.child("match_requests").child(request_id)
    match_request = match_request_ref.get()
    if not match_request:
        return jsonify(success=False, message="match_request not found"), 404

    initiator = match_request.get("initiator")
    receiver = match_request.get("receiver")
    if user_id not in [initiator, receiver]:
        return jsonify(success=False, message="not a participant"), 403

    go_no_go = match_request.get("go_no_go") or {}
    return jsonify(
        success=True,
        go_no_go=go_no_go,
        participants={"user_a": initiator, "user_b": receiver},
        updated_at=match_request.get("go_no_go_updated_at"),
    )


# =========================
# 🔥 Helper Functions
# =========================

def count_completed_talks(db, user_id, partner_id):
    """
    두 사용자 간 완료된 대화 수 계산 (top-level talk_history)
    """
    talks_ref = db.collection("talk_history")
    
    # Case 1: user_a = user_id, user_b = partner_id
    query1 = (
        talks_ref
        .where("participants.user_a", "==", user_id)
        .where("participants.user_b", "==", partner_id)
        .where("completed", "==", True)
        .stream()
    )
    
    # Case 2: user_a = partner_id, user_b = user_id
    query2 = (
        talks_ref
        .where("participants.user_a", "==", partner_id)
        .where("participants.user_b", "==", user_id)
        .where("completed", "==", True)
        .stream()
    )
    
    # 두 쿼리 결과 합치기
    count = len(list(query1)) + len(list(query2))
    
    return count


def get_used_questions(db, user_id, partner_id, topic):
    """
    이 파트너와 이미 받은 질문들 (top-level talk_history)
    """
    talks_ref = db.collection("talk_history")
    
    # Case 1
    query1 = (
        talks_ref
        .where("participants.user_a", "==", user_id)
        .where("participants.user_b", "==", partner_id)
        .where("topic", "==", topic)
        .where("completed", "==", True)
        .stream()
    )
    
    # Case 2
    query2 = (
        talks_ref
        .where("participants.user_a", "==", partner_id)
        .where("participants.user_b", "==", user_id)
        .where("topic", "==", topic)
        .where("completed", "==", True)
        .stream()
    )
    
    questions = set()
    for doc in query1:
        q = doc.to_dict().get("question")
        if q:
            questions.add(q)
    for doc in query2:
        q = doc.to_dict().get("question")
        if q:
            questions.add(q)
    
    return questions


def select_new_question(topic, used_questions):
    """
    새로운 질문 선택 (이미 받은 질문 제외)
    """
    all_questions = QUESTIONS.get(topic, [])
    
    # 사용 안 한 질문
    unused = [q for q in all_questions if q not in used_questions]
    
    if unused:
        return random.choice(unused)
    
    # 다 사용했으면 아무거나
    return random.choice(all_questions)


def select_random_options(topic):
    """
    랜덤 옵션 선택 (실제 파일명 사용)
    """
    categories = IMAGE_CATEGORIES.get(topic)
    if not categories:
        return []
    
    # 3개 카테고리 랜덤 선택
    selected_categories = random.sample(categories, 3)
    
    options = []
    for category in selected_categories:
        # 실제 파일 목록
        files = get_image_files(topic, category)
        
        if not files:
            print(f"⚠️ No files: {topic}/{category}")
            continue
        
        # 랜덤 파일 선택
        random_file = random.choice(files)
        
        options.append({
            "category": category,
            "fileName": random_file
        })
    
    return options


# =========================
# 🔥 Get Talk History
# =========================
@talks_bp.route("/history", methods=["GET"])
def get_talk_history():
    """
    사용자의 대화 기록 조회
    
    Query params:
    - partner_id (optional): 특정 파트너와의 대화만
    - limit (optional): 최대 개수
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    partner_id = request.args.get("partner_id")
    limit = int(request.args.get("limit", 50))

    try:
        db = get_firestore()
        talks_ref = db.collection("talk_history")
        
        # user_a 또는 user_b인 대화 모두 가져오기
        query1 = talks_ref.where("participants.user_a", "==", user_id).stream()
        query2 = talks_ref.where("participants.user_b", "==", user_id).stream()
        
        talks = []
        for doc in query1:
            talk = doc.to_dict()
            talk["id"] = doc.id
            talks.append(talk)
        for doc in query2:
            talk = doc.to_dict()
            talk["id"] = doc.id
            talks.append(talk)
        
        # partner_id 필터링
        if partner_id:
            talks = [
                t for t in talks
                if t["participants"]["user_a"] == partner_id or t["participants"]["user_b"] == partner_id
            ]
        
        # 시간순 정렬 (최신순)
        talks.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # 제한
        talks = talks[:limit]
        
        print(f"📜 Talk history: {user_id} → {len(talks)}개")
        
        return jsonify(success=True, talks=talks)

    except Exception as e:
        print(f"❌ get-talk-history 실패: {e}")
        return jsonify(success=False, message=str(e)), 500
