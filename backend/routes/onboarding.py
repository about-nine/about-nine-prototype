from datetime import datetime
from flask import Blueprint, jsonify, session

from backend.services.firestore import get_firestore
from backend.services.user_profile_service import (
    default_embedding_payload,
    embedding_payload_from_text,
    transcripts_to_text,
)
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
    transcripts = data.get("transcripts")

    if profile is None:
        return jsonify(success=False, message="profile is required"), 400

    def normalize_lower(value):
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned if cleaned else None
        return None

    gender = normalize_lower(profile.get("gender"))
    sexual_orientation = normalize_lower(profile.get("sexual_orientation"))
    gender_detail = normalize_lower(profile.get("gender_detail"))

    if not gender or not sexual_orientation:
        return (
            jsonify(success=False, message="gender and sexual_orientation are required"),
            400,
        )

    db = get_firestore()

    # ✅ 루트 필드에 직접 저장 (onboarding_profile 제거)
    update_data = {
        "onboarding_completed": True,
        "onboarding_updated_at": datetime.utcnow().isoformat(),
        "gender": gender,
        "gender_detail": gender_detail,
        "sexual_orientation": sexual_orientation,
        "age_preference": profile.get("age_preference"),
        "drink": profile.get("drink"),
        "smoke": profile.get("smoke"),
        "marijuana": profile.get("marijuana"),
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

    # Embedding: voice onboarding uses transcripts; chat onboarding gets default embedding.
    embedding_payload = None
    if isinstance(transcripts, list) and transcripts:
        text = transcripts_to_text(transcripts)
        embedding_payload = embedding_payload_from_text(text, "onboarding_voice")
        if not embedding_payload:
            embedding_payload = default_embedding_payload("onboarding_voice_fallback")
    else:
        embedding_payload = default_embedding_payload("onboarding_chat")

    update_data["embedding"] = embedding_payload

    db.collection("users").document(user_id).set(update_data, merge=True)

    return jsonify(success=True)
