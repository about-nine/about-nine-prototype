from datetime import datetime
from flask import Blueprint, jsonify, session

from backend.services.firestore import get_firestore
from backend.utils.request import get_json

onboarding_bp = Blueprint("onboarding", __name__, url_prefix="/api/onboarding")

@onboarding_bp.route("/save", methods=["POST"])
def save_onboarding():

    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="not logged in"), 401

    data, err, code = get_json()
    if err:
        return err, code

    profile = data.get("profile")
    first_name = data.get("firstName")
    last_name = data.get("lastName")
    age = data.get("age")
    birthdate = data.get("birthdate")
    phone = data.get("phone")
    bio = data.get("bio")

    if profile is None:
        return jsonify(success=False, message="profile is required"), 400

    db = get_firestore()

    # ✅ 필터링에 사용될 필드들을 루트 레벨에 저장
    update_data = {
        "onboarding_profile": profile,
        "onboarding_completed": True,
        "onboarding_updated_at": datetime.utcnow().isoformat(),
        # 필터링용 루트 필드
        "gender": profile.get("gender"),
        "gender_detail": profile.get("gender_detail"),
        "sexual_orientation": profile.get("sexual_orientation"),
        "age_preference": profile.get("age_preference"),
    }

    # 선택적 필드
    if first_name:
        update_data["first_name"] = first_name
    if last_name:
        update_data["last_name"] = last_name
    if age is not None:
        update_data["age"] = age
    if birthdate:
        update_data["birthdate"] = birthdate
    if phone:
        update_data["phone"] = phone
        
    if bio and bio.strip():
        update_data["bio"] = bio.strip()

    db.collection("users").document(user_id).set(update_data, merge=True)

    return jsonify(success=True)
