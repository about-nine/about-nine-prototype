# services/firestore.py - Firebase app and Firestore client initialization
import json
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from backend.config import (
    FIREBASE_SERVICE_ACCOUNT_JSON,
    FIREBASE_SERVICE_ACCOUNT_PATH,
    FIREBASE_DB_URL,
    FIREBASE_STORAGE_BUCKET,
)

_app = None
_db = None


def get_firestore():
    global _app, _db

    if _db:
        return _db

    cred = None

    # 1️⃣ JSON env (권장)
    if FIREBASE_SERVICE_ACCOUNT_JSON:
        cred = credentials.Certificate(
            json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        )

    # 2️⃣ file path
    elif FIREBASE_SERVICE_ACCOUNT_PATH:
        path = Path(FIREBASE_SERVICE_ACCOUNT_PATH)
        if not path.exists():
            raise RuntimeError(
                f"Firebase service account file not found: {path}"
            )
        cred = credentials.Certificate(str(path))

    else:
        # Fallback to Application Default Credentials (GCP environment)
        cred = credentials.ApplicationDefault()

    try:
        firebase_admin.get_app()
    except ValueError:
        options = {}
        if FIREBASE_DB_URL:
            options['databaseURL'] = FIREBASE_DB_URL
        if FIREBASE_STORAGE_BUCKET:
            options['storageBucket'] = FIREBASE_STORAGE_BUCKET
        
        _app = initialize_app(cred, options)

    print("✅ Firestore initialized")
    if FIREBASE_DB_URL:
        print(f"✅ RTDB URL: {FIREBASE_DB_URL}")

    _db = firestore.client()
    return _db
