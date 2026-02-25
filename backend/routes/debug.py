# routes/debug.py - debug-only endpoints
from flask import Blueprint, session, jsonify
from backend.services.firestore import get_firestore

debug_bp = Blueprint("debug", __name__, url_prefix="/api/debug")

@debug_bp.route("/me")
def debug_me():
    uid = session.get("user_id")
    if not uid:
        return jsonify(error="not logged in"), 401
    
    db = get_firestore()
    me = db.collection("users").document(uid).get()
    
    if not me.exists:
        return jsonify(error="user not found"), 404
    
    return jsonify(me.to_dict())

@debug_bp.route("/all-users")
def debug_all_users():
    db = get_firestore()
    all_users = []
    
    for doc in db.collection("users").stream():
        u = doc.to_dict()
        all_users.append({
            "id": u.get("id"),
            "first_name": u.get("first_name"),
            "last_name": u.get("last_name"),
            "gender": u.get("gender"),
            "age": u.get("age"),
            "location": u.get("location"),
            "onboarding_completed": u.get("onboarding_completed"),
            "sexual_orientation": u.get("sexual_orientation"),
            "age_preference": u.get("age_preference")
        })
    
    return jsonify(users=all_users, count=len(all_users))
