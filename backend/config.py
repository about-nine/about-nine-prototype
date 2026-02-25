import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# =================================================
# Environment
# =================================================

ENV = os.getenv("FLASK_ENV", "development")
IS_PROD = ENV == "production"
DEBUG = not IS_PROD


# =================================================
# Core
# =================================================

SECRET_KEY = os.getenv("SECRET_KEY")

if IS_PROD and not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in production")


# =================================================
# CORS
# =================================================

raw_cors_origins = os.getenv("CORS_ORIGINS")

if raw_cors_origins:
    # Render에서 환경변수로 직접 지정 가능
    CORS_ORIGINS = [
        origin.strip()
        for origin in raw_cors_origins.split(",")
        if origin.strip()
    ]
else:
    if IS_PROD:
        # 프로덕션: 배포 도메인만 허용 (보안)
        CORS_ORIGINS = [
            "https://about-nine.onrender.com",
            "https://about-nine-prototype-46a2c.firebaseapp.com",
            "https://about-nine-prototype-46a2c.web.app",
        ]
    else:
        # 개발: 로컬 전체 허용
        CORS_ORIGINS = [
            "http://localhost:5001",
            "http://127.0.0.1:5001",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]


# =================================================
# Firebase
# =================================================

FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
