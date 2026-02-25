# routes/auth.py - auth, invite verification, and logout endpoints
import secrets
from datetime import datetime
from flask import Blueprint, session, jsonify
from backend.utils.request import get_json
from backend.services.firestore import get_firestore
from firebase_admin import auth as fb_auth

auth_bp = Blueprint("auth", __name__, url_prefix="/api")

INVITE_CODES = {"9191", "ABOUTNINE"}


def is_valid_invite(code):
    if not code:
        return False
    try:
        normalized = str(code).strip()
    except Exception:
        return False
    return normalized in INVITE_CODES


# =========================
# Invite
# =========================
@auth_bp.route("/verify-invite", methods=["POST"])
def verify_invite():
    data, err, code = get_json()
    if err:
        return err, code

    if is_valid_invite(data.get("code")):
        return jsonify(success=True)

    return jsonify(success=False), 400


# =========================
# Firebase Login
# =========================
@auth_bp.route("/auth/firebase-login", methods=["POST"])
def firebase_login():

    data, err, code = get_json()
    if err:
        return err, code

    id_token = data.get("idToken")
    if not id_token:
        return jsonify(success=False, message="missing idToken"), 400

    try:
        decoded = fb_auth.verify_id_token(id_token)
    except Exception as e:
        print(f"❌ Token verification failed: {type(e).__name__}: {e}")
        return jsonify(success=False, message="invalid token"), 401

    firebase_uid = decoded["uid"]
    db = get_firestore()

    # user 조회
    query = (
        db.collection("users")
        .where("firebase_uid", "==", firebase_uid)
        .limit(1)
        .get()
    )

    doc = query[0] if query else None

    if doc:
        # 기존 유저 → invite 불필요
        user_data = doc.to_dict()
        user_id = user_data["id"]
        is_existing_user = user_data.get("onboarding_completed", False)
    else:
        # 신규 유저 → invite 필수 (cookie/session 없이 body로만 검증)
        invite_code = data.get("invite_code") or data.get("inviteCode")
        if not is_valid_invite(invite_code):
            return jsonify(success=False, message="invite required"), 403

        user_id = secrets.token_urlsafe(16)
        db.collection("users").document(user_id).set({
            "id": user_id,
            "firebase_uid": firebase_uid,
            "created_at": datetime.utcnow().isoformat(),
            "playlist": [],
            "location": None,
            "onboarding_completed": False,
        })
        is_existing_user = False
        print(f"🆕 New user created: {user_id}")

    # session
    session["user_id"] = user_id
    session["firebase_uid"] = firebase_uid
    session["phone_verified"] = True
    session.permanent = True

    # invite_verified는 더 이상 사용하지 않음

    return jsonify(
        success=True,
        user_id=user_id,
        is_existing_user=is_existing_user
    )

@auth_bp.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify(success=True)
