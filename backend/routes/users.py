"""
users.py - 사용자 관련 엔드포인트

- playlist: 플레이리스트 저장
- location: 위치 업데이트
- profile: 프로필 조회/수정
- list: 주변 사용자 목록
"""

from datetime import datetime
from flask import Blueprint, jsonify, session
from firebase_admin import firestore
from firebase_admin import auth as fb_auth
from backend.services.firestore import get_firestore
from backend.utils.request import get_json
import math

users_bp = Blueprint("users", __name__, url_prefix="/api/users")

# DEBUG: set True to return all users without filters
BYPASS_USER_FILTERS = False

# =========================
# Playlist
# =========================
@users_bp.route("/playlist", methods=["POST"])
def save_playlist():
    """플레이리스트 저장"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    tracks = data.get("tracks", [])

    db = get_firestore()

    db.collection("users").document(user_id).set({
        "playlist": tracks,
        "playlist_updated_at": datetime.utcnow().isoformat()
    }, merge=True)

    return jsonify(success=True)


# =========================
# Location
# =========================
@users_bp.route("/update-location", methods=["POST"])
def update_location():
    """위치 업데이트"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    db = get_firestore()

    db.collection("users").document(user_id).set({
        "location": {
            "lat": data.get("lat"),
            "lng": data.get("lng")
        }
    }, merge=True)

    return jsonify(success=True)


# =========================
# 거리 계산
# =========================
def distance_km(lat1, lng1, lat2, lng2):
    """두 좌표 간 거리 계산 (km)"""
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)

    a = (
        math.sin(d_lat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(d_lng/2)**2
    )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# =========================
# 성적 지향 매칭
# =========================
def matches_orientation(orientation, target_gender):
    """
    성적 지향과 상대 성별 매칭 확인
    
    orientation: 내 성적 지향 (예: "men", "women", "all types of genders")
    target_gender: 상대방의 성별 (예: "man", "woman", "non-binary")
    """
    if not orientation:
        return True  # 기본값: 모두 허용
    
    orientation = orientation.lower()
    target_gender = target_gender.lower()
    
    # "all types of genders" → 모두 허용
    if "all types" in orientation:
        return True
    
    # "men" → man만
    if orientation == "men":
        return target_gender == "man"
    
    # "women" → woman만
    if orientation == "women":
        return target_gender == "woman"
    
    # "men and women" → man 또는 woman
    if orientation == "men and women":
        return target_gender in ["man", "woman"]
    
    # "men and non-binary people"
    if "men and non-binary" in orientation:
        return target_gender in ["man", "non-binary"]
    
    # "women and non-binary people"
    if "women and non-binary" in orientation:
        return target_gender in ["woman", "non-binary"]
    
    return False


def partners_completed_three_rounds(db, user_id):
    """
    나와 3라운드까지 대화를 완료한 파트너 ID 집합 반환
    """
    talks_ref = db.collection("talk_history")
    partner_rounds = {}
    required = {1, 2, 3}

    queries = [
        talks_ref.where("participants.user_a", "==", user_id).stream(),
        talks_ref.where("participants.user_b", "==", user_id).stream(),
    ]

    for query in queries:
        for doc in query:
            talk = doc.to_dict() or {}
            if talk.get("completed") is not True:
                continue

            participants = talk.get("participants") or {}
            if participants.get("user_a") == user_id:
                partner_id = participants.get("user_b")
            else:
                partner_id = participants.get("user_a")

            if not partner_id:
                continue

            round_raw = talk.get("round") or 1
            try:
                round_num = int(round_raw)
            except (TypeError, ValueError):
                round_num = 1
            round_num = max(1, min(round_num, 3))

            partner_rounds.setdefault(partner_id, set()).add(round_num)

    return {
        pid for pid, rounds in partner_rounds.items()
        if required.issubset(rounds)
    }


# =========================
# Nearby Users List
# =========================
@users_bp.route("/list")
def list_users():
    """
    주변 사용자 목록 (필터링 적용)
    
    필터:
    - 거리 10km 이내
    - 성적 지향 일치 (양방향)
    - 나이 선호 일치 (양방향)
    - 이미 3라운드를 완료한 파트너 제외
    """
    uid = session.get("user_id")
    if not uid:
        return jsonify(success=False), 401

    db = get_firestore()
    if BYPASS_USER_FILTERS:
        users = []
        for doc in db.collection("users").stream():
            u = doc.to_dict()
            if "id" not in u:
                u["id"] = doc.id
            if u["id"] == uid:
                continue
            users.append(u)
        return jsonify(success=True, users=users)

    me = db.collection("users").document(uid).get().to_dict()

    print(f"\n=== USER LIST DEBUG ===")
    print(f"My ID: {uid}")
    print(f"My profile: {me}")

    my_loc = me.get("location")
    my_gender = me.get("gender")
    my_age = me.get("age")
    my_sexual_orientation = me.get("sexual_orientation")
    my_age_pref = me.get("age_preference", {})
    my_blocked = set(me.get("blocked_users") or [])

    print(f"My location: {my_loc}")
    print(f"My gender: {my_gender}, age: {my_age}")
    print(f"My preferences: orientation={my_sexual_orientation}, age_range={my_age_pref}")

    if not my_loc or my_loc.get("lat") is None or my_loc.get("lng") is None:
        return jsonify(success=False, message="missing my location"), 400

    if not my_gender or my_age is None:
        return jsonify(success=False, message="missing my profile"), 400

    completed_partners = partners_completed_three_rounds(db, uid)

    users = []
    total_count = 0
    filtered_stats = {
        "same_user": 0,
        "blocked": 0,
        "no_onboarding": 0,
        "no_location": 0,
        "too_far": 0,
        "missing_gender_age": 0,
        "orientation_mismatch": 0,
        "age_mismatch": 0,
        "reverse_orientation": 0,
        "reverse_age": 0,
        "completed_all_rounds": 0,
        "passed": 0
    }

    for doc in db.collection("users").stream():
        u = doc.to_dict()
        if "id" not in u:
            u["id"] = doc.id
        total_count += 1

        # 본인 제외
        if u["id"] == uid:
            filtered_stats["same_user"] += 1
            continue

        # 차단 확인 (양방향)
        other_blocked = set(u.get("blocked_users") or [])
        if u.get("id") in my_blocked or uid in other_blocked:
            filtered_stats["blocked"] += 1
            continue

        # 3라운드까지 완료한 파트너 제외
        if u["id"] in completed_partners:
            filtered_stats["completed_all_rounds"] += 1
            continue

        # 온보딩 완료 확인
        if not u.get("onboarding_completed"):
            filtered_stats["no_onboarding"] += 1
            continue

        # 위치 확인
        loc = u.get("location")
        if not loc:
            filtered_stats["no_location"] += 1
            continue

        # 거리 체크 (10km)
        d = distance_km(
            my_loc["lat"], my_loc["lng"],
            loc["lat"], loc["lng"]
        )
        if d > 10:
            filtered_stats["too_far"] += 1
            continue

        other_gender = u.get("gender")
        other_age = u.get("age")
        other_sexual_orientation = u.get("sexual_orientation")
        other_age_pref = u.get("age_preference", {})

        if not other_gender or not other_age:
            filtered_stats["missing_gender_age"] += 1
            continue

        # 내가 상대를 선호하는지
        if not matches_orientation(my_sexual_orientation, other_gender):
            filtered_stats["orientation_mismatch"] += 1
            continue

        if my_age_pref:
            if not (my_age_pref.get("min", 0) <= other_age <= my_age_pref.get("max", 100)):
                filtered_stats["age_mismatch"] += 1
                continue

        # 상대가 나를 선호하는지 (양방향 확인)
        if not matches_orientation(other_sexual_orientation, my_gender):
            filtered_stats["reverse_orientation"] += 1
            continue

        if other_age_pref:
            if not (other_age_pref.get("min", 0) <= my_age <= other_age_pref.get("max", 100)):
                filtered_stats["reverse_age"] += 1
                continue

        # 모든 필터 통과
        filtered_stats["passed"] += 1
        users.append(u)

    print(f"\nTotal users in DB: {total_count}")
    print(f"Filter results:")
    for key, value in filtered_stats.items():
        print(f"  {key}: {value}")
    print(f"Final result: {len(users)} users")
    print("======================\n")

    return jsonify(success=True, users=users)


# =========================
# Get Profile
# =========================
@users_bp.route("/profile", methods=["GET"])
def get_profile():
    """프로필 조회"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    db = get_firestore()
    user = db.collection("users").document(user_id).get().to_dict()
    
    if not user:
        return jsonify(success=False, message="user not found"), 404
    
    return jsonify(success=True, user=user)


# =========================
# Update Profile
# =========================
@users_bp.route("/profile", methods=["POST"])
def update_profile():
    """프로필 수정"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    db = get_firestore()

    def normalize_lower(value):
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned if cleaned else None
        return None

    # 업데이트할 필드만 전송
    update_data = {}

    existing = db.collection("users").document(user_id).get().to_dict() or {}
    incoming_profile = data.get("onboarding_profile")
    if not isinstance(incoming_profile, dict):
        incoming_profile = None

    def has_incoming(key):
        return key in data or (incoming_profile is not None and key in incoming_profile)

    def get_incoming(key):
        if key in data:
            return data.get(key)
        if incoming_profile is not None and key in incoming_profile:
            return incoming_profile.get(key)
        return None

    if "bio" in data:
        update_data["bio"] = data["bio"]

    if "first_name" in data:
        update_data["first_name"] = data["first_name"]

    if "last_name" in data:
        update_data["last_name"] = data["last_name"]

    if "age" in data:
        update_data["age"] = data["age"]

    if has_incoming("age_preference"):
        update_data["age_preference"] = get_incoming("age_preference")

    if has_incoming("drink"):
        update_data["drink"] = get_incoming("drink")

    if has_incoming("smoke"):
        update_data["smoke"] = get_incoming("smoke")

    if has_incoming("marijuana"):
        update_data["marijuana"] = get_incoming("marijuana")

    incoming_gender = normalize_lower(get_incoming("gender"))
    incoming_orientation = normalize_lower(get_incoming("sexual_orientation"))
    incoming_gender_detail = normalize_lower(get_incoming("gender_detail"))

    if has_incoming("gender"):
        if not incoming_gender:
            return jsonify(success=False, message="gender is required"), 400
        update_data["gender"] = incoming_gender

    if has_incoming("sexual_orientation"):
        if not incoming_orientation:
            return jsonify(success=False, message="sexual_orientation is required"), 400
        update_data["sexual_orientation"] = incoming_orientation

    if has_incoming("gender_detail"):
        update_data["gender_detail"] = incoming_gender_detail

    final_gender = update_data.get("gender") or normalize_lower(existing.get("gender"))
    final_orientation = update_data.get("sexual_orientation") or normalize_lower(
        existing.get("sexual_orientation")
    )
    final_gender_detail = (
        update_data.get("gender_detail")
        if "gender_detail" in update_data
        else normalize_lower(existing.get("gender_detail"))
    )

    if not final_gender or not final_orientation:
        return jsonify(success=False, message="gender and sexual_orientation are required"), 400

    update_data["gender"] = final_gender
    update_data["sexual_orientation"] = final_orientation
    if "gender_detail" in update_data:
        update_data["gender_detail"] = final_gender_detail
    elif final_gender_detail:
        update_data["gender_detail"] = final_gender_detail

    db.collection("users").document(user_id).set(update_data, merge=True)
    
    return jsonify(success=True)


# =========================
# Delete Account
# =========================
@users_bp.route("/delete", methods=["POST"])
def delete_account():
    """계정 삭제 (Firestore users 문서 삭제 + 세션 정리)"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    db = get_firestore()
    firebase_uid = session.get("firebase_uid")
    if not firebase_uid:
        try:
            existing = db.collection("users").document(user_id).get().to_dict() or {}
            firebase_uid = existing.get("firebase_uid")
        except Exception:
            firebase_uid = None

    if firebase_uid:
        try:
            fb_auth.delete_user(firebase_uid)
        except fb_auth.UserNotFoundError:
            pass
        except Exception as e:
            print(f"❌ Failed to delete auth user {firebase_uid}: {e}")
            return jsonify(success=False, message="auth delete failed"), 500

    try:
        db.collection("users").document(user_id).delete()
    except Exception as e:
        print(f"❌ Failed to delete user {user_id}: {e}")
        return jsonify(success=False, message="delete failed"), 500

    session.clear()
    return jsonify(success=True)


# =========================
# Block User
# =========================
@users_bp.route("/block", methods=["POST"])
def block_user():
    """유저 차단은 talk-result에서 no 선택 시에만 가능합니다."""
    return (
        jsonify(
            success=False,
            message="blocking is only available via talk-result 'no' response",
        ),
        403,
    )
