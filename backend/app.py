from flask import Flask, send_from_directory, jsonify, request, session
from flask_cors import CORS
from pathlib import Path
import os

from backend.config import SECRET_KEY, CORS_ORIGINS, DEBUG
from backend.services.firestore import get_firestore
from firebase_admin import auth as fb_auth

# =========================
# App init
# =========================

app = Flask(__name__)
app.secret_key = SECRET_KEY

CORS(app, supports_credentials=True, origins=CORS_ORIGINS)

if DEBUG:
    app.config.update(
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=False,
    )
else:
    app.config.update(
        SESSION_COOKIE_SAMESITE="None",
        SESSION_COOKIE_SECURE=True,
    )


# =========================
# ✅ Token-based session hydration (for cross-site cookie issues)
# =========================

@app.before_request
def hydrate_session_from_token():
    if session.get("user_id"):
        return

    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.headers.get("X-Id-Token") or request.headers.get("X-Firebase-Token")
    if not token:
        return

    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception:
        return

    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        return

    db = get_firestore()
    query = (
        db.collection("users")
        .where("firebase_uid", "==", firebase_uid)
        .limit(1)
        .get()
    )
    doc = query[0] if query else None
    if not doc:
        return

    user_data = doc.to_dict() or {}
    user_id = user_data.get("id") or doc.id
    if not user_id:
        return

    session["user_id"] = user_id
    session["firebase_uid"] = firebase_uid
    session["phone_verified"] = True


# =========================
# ✅ API Blueprints (먼저 등록)
# =========================

from backend.routes.auth import auth_bp
from backend.routes.users import users_bp
from backend.routes.onboarding import onboarding_bp
from backend.routes.agora import agora_bp
from backend.routes.talks import talks_bp
from backend.routes.match import match_bp, analysis_bp
from backend.routes.debug import debug_bp
from backend.routes.spotify import spotify_bp, spotify_auth_bp
from backend.routes.voice import voice_bp

app.register_blueprint(auth_bp)
app.register_blueprint(users_bp)
app.register_blueprint(onboarding_bp)
app.register_blueprint(agora_bp)
app.register_blueprint(talks_bp)
app.register_blueprint(match_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(spotify_bp)
app.register_blueprint(spotify_auth_bp)
app.register_blueprint(voice_bp)

if DEBUG:
    app.register_blueprint(debug_bp)

# =========================
# ✅ Firebase 즉시 초기화
# =========================
get_firestore()

# =========================
# ✅ Analysis Worker (백그라운드 스레드)
# =========================
import threading
from backend.scripts.analysis_worker import run_worker

def _start_worker():
    t = threading.Thread(target=run_worker, daemon=True, name="analysis-worker")
    t.start()
    print("✅ Analysis worker started")

_start_worker()

# =========================
# Health
# =========================

@app.route("/api/health")
def health():
    return jsonify(status="ok")


# =========================
# ✅ Frontend serving (API 절대 건드리지 않음)
# =========================

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


# 정적 파일 전용
@app.route("/<path:filename>")
def static_files(filename):

    # 🔥 가장 중요: API는 여기 오면 안 됨
    if filename.startswith("api/"):
        return jsonify(success=False, message="Not found"), 404

    file_path = FRONTEND_DIR / filename

    if file_path.exists():
        return send_from_directory(FRONTEND_DIR, filename)

    return jsonify(success=False, message="Not found"), 404


# 루트 → index.html
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

# =========================
# Run
# =========================

if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    # Prefer Render's $PORT; use 5001 locally, 10000 as prod-like fallback.
    if port_env:
        port = int(port_env)
    else:
        port = 5001 if DEBUG else 10000
    app.run(host="0.0.0.0", port=port, debug=DEBUG)
