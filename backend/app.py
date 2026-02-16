from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from pathlib import Path
import os

from backend.config import SECRET_KEY, CORS_ORIGINS, DEBUG
from backend.services.firestore import get_firestore

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
# ✅ API Blueprints (먼저 등록)
# =========================

from backend.routes.auth import auth_bp
from backend.routes.users import users_bp
from backend.routes.music import music_bp
from backend.routes.onboarding import onboarding_bp
from backend.routes.agora import agora_bp
from backend.routes.talks import talks_bp
from backend.routes.match import match_bp
from backend.routes.debug import debug_bp
from backend.routes.spotify import spotify_bp, spotify_auth_bp

app.register_blueprint(auth_bp)
app.register_blueprint(users_bp)
app.register_blueprint(music_bp)
app.register_blueprint(onboarding_bp)
app.register_blueprint(agora_bp)
app.register_blueprint(talks_bp)
app.register_blueprint(match_bp)
app.register_blueprint(spotify_bp)
app.register_blueprint(spotify_auth_bp)

if DEBUG:
    app.register_blueprint(debug_bp)

# =========================
# ✅ Firebase 즉시 초기화
# =========================
get_firestore()

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
