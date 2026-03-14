# routes/users.py - user profile, location, playlist, and account endpoints
"""
users.py - 사용자 관련 엔드포인트

- playlist: 플레이리스트 저장
- location: 위치 업데이트
- profile: 프로필 조회/수정
"""

from datetime import datetime
from flask import Blueprint, jsonify, session
from firebase_admin import auth as fb_auth
from backend.services.firestore import get_firestore
from backend.services.rtdb import get_rtdb
from backend.utils.age_policy import AGE_MAX, AGE_MIN, normalize_age_preference, parse_age
from backend.utils.request import get_json

users_bp = Blueprint("users", __name__, url_prefix="/api/users")

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
        },
        "location_filter_enabled": True,
    }, merge=True)

    return jsonify(success=True)


@users_bp.route("/location-filter", methods=["POST"])
def update_location_filter():
    """위치 필터 사용 여부 업데이트"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify(success=False, message="enabled must be boolean"), 400

    db = get_firestore()
    db.collection("users").document(user_id).set(
        {"location_filter_enabled": enabled},
        merge=True,
    )
    return jsonify(success=True)


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

    def normalize_bool(value):
        if isinstance(value, bool):
            return value
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
        parsed_age = parse_age(data.get("age"))
        if parsed_age is None:
            return jsonify(success=False, message=f"age must be between {AGE_MIN} and {AGE_MAX}"), 400
        update_data["age"] = parsed_age

    if has_incoming("age_preference"):
        age_preference = normalize_age_preference(get_incoming("age_preference"))
        if age_preference is None:
            return (
                jsonify(
                    success=False,
                    message=f"age_preference must be between {AGE_MIN} and {AGE_MAX}",
                ),
                400,
            )
        update_data["age_preference"] = age_preference

    if has_incoming("drink"):
        update_data["drink"] = get_incoming("drink")

    if has_incoming("smoke"):
        update_data["smoke"] = get_incoming("smoke")

    if has_incoming("marijuana"):
        update_data["marijuana"] = get_incoming("marijuana")

    if has_incoming("location_filter_enabled"):
        location_filter_enabled = normalize_bool(get_incoming("location_filter_enabled"))
        if location_filter_enabled is None:
            return jsonify(success=False, message="location_filter_enabled must be boolean"), 400
        update_data["location_filter_enabled"] = location_filter_enabled

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
        rtdb = get_rtdb()
        if rtdb:
            rtdb.child("presence").child(user_id).delete()
    except Exception as e:
        print(f"❌ Failed to delete presence for {user_id}: {e}")

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
