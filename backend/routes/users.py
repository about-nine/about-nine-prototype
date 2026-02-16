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
from backend.services.firestore import get_firestore
from backend.utils.request import get_json
import math

users_bp = Blueprint("users", __name__, url_prefix="/api/users")

# DEBUG: set True to return all users without filters
BYPASS_USER_FILTERS = True

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

    users = []
    total_count = 0
    filtered_stats = {
        "same_user": 0,
        "no_onboarding": 0,
        "no_location": 0,
        "too_far": 0,
        "orientation_mismatch": 0,
        "age_mismatch": 0,
        "reverse_orientation": 0,
        "reverse_age": 0,
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
            filtered_stats["same_user"] += 1
            continue

        # 온보딩 완료 확인
        if not u.get("onboarding_completed"):
            filtered_stats["no_onboarding"] += 1
            continue

        # 위치 확인
        loc = u.get("location")
        if not loc or not my_loc:
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
            filtered_stats["no_location"] += 1
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
    
    # 업데이트할 필드만 전송
    update_data = {}
    
    if "onboarding_profile" in data:
        update_data["onboarding_profile"] = data["onboarding_profile"]
    
    if "bio" in data:
        update_data["bio"] = data["bio"]
    
    if "sexual_orientation" in data:
        update_data["sexual_orientation"] = data["sexual_orientation"]
    
    if "age_preference" in data:
        update_data["age_preference"] = data["age_preference"]
    
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
