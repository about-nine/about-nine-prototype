# services/rtdb.py - Firebase Realtime Database client
import firebase_admin
from firebase_admin import db
from backend.config import FIREBASE_DB_URL

_rtdb = None


def get_rtdb():
    global _rtdb

    if not FIREBASE_DB_URL:
        return None

    if _rtdb:
        return _rtdb

    try:
        firebase_admin.get_app()
    except ValueError:
        raise RuntimeError(
            "Firebase app not initialized. Initialize Firestore first."
        )

    print("✅ RTDB connected:", FIREBASE_DB_URL)

    _rtdb = db.reference("/", url=FIREBASE_DB_URL)
    return _rtdb
