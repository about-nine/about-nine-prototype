#!/usr/bin/env python3
"""Create or update visible test users for App Review.

Usage:
  python3 backend/scripts/test_user.py --phone-woman "+1XXXXXXXXXX" --phone-man "+1YYYYYYYYYY"
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.firestore import get_firestore  # noqa: E402
from backend.services.rtdb import get_rtdb  # noqa: E402
from backend.services.user_profile_service import default_embedding_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update a test user that appears for location-filter-off users.",
    )
    parser.add_argument("--phone-woman", required=True, help="Woman test user phone number")
    parser.add_argument("--phone-man", required=True, help="Man test user phone number")
    parser.add_argument("--first-name", default="Test", help="First name")
    parser.add_argument("--last-name", default="User", help="Last name")
    parser.add_argument("--age", type=int, default=28, help="Age (20-60)")
    parser.add_argument("--birthdate", default="1998-01-01", help="Birthdate YYYY-MM-DD")
    return parser.parse_args()


def find_user_id_by_phone(db, phone: str) -> str | None:
    docs = db.collection("users").where("phone", "==", phone).limit(1).get()
    if not docs:
        return None
    return docs[0].id


def upsert_test_user(
    db,
    user_id: str,
    phone: str,
    first_name: str,
    last_name: str,
    age: int,
    birthdate: str,
    gender: str = "woman",
) -> None:
    now_iso = datetime.utcnow().isoformat()
    payload = {
        "id": user_id,
        "created_at": now_iso,
        "onboarding_completed": True,
        "onboarding_updated_at": now_iso,
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "age": age,
        "birthdate": birthdate,
        "gender": gender,
        "gender_detail": "cis woman" if gender == "woman" else "cis man",
        "sexual_orientation": "all types of genders",
        "age_preference": {"min": 20, "max": 60},
        "drink": "no",
        "smoke": "no",
        "marijuana": "no",
        "bio": "for app review testing account",
        "playlist": [
            {"name": "Test Conversation", "artist": "About Nine"},
            {"name": "App Review Track", "artist": "About Nine"},
        ],
        # Key requirement for your current review strategy:
        # location permission denied user -> location_filter_enabled=false pool.
        "location_filter_enabled": False,
        "location": None,
        "is_test_user": True,
        "test_user_gender": gender,
        "embedding": default_embedding_payload("test_user_script"),
        "talk_profile": {
            "avg_turn_length": 0.0,
            "speech_pace": 0.0,
            "emotional_expression": 0.0,
            "vocabulary_diversity": 0.0,
            "talk_count": 0,
        },
    }
    db.collection("users").document(user_id).set(payload, merge=True)


def set_presence_online(user_id: str) -> bool:
    rtdb = get_rtdb()
    if not rtdb:
        return False
    rtdb.child("presence").child(user_id).set(
        {
            "online": True,
            "updated_at": int(time.time() * 1000),
        }
    )
    return True


def main() -> int:
    args = parse_args()
    db = get_firestore()

    targets = [
        ("woman", args.phone_woman, f"{args.first_name} Woman"),
        ("man", args.phone_man, f"{args.first_name} Man"),
    ]

    print("✅ Test user setup started")
    for gender, phone, first_name in targets:
        user_id = find_user_id_by_phone(db, phone)
        created = False
        if not user_id:
            user_id = secrets.token_urlsafe(16)
            created = True

        upsert_test_user(
            db=db,
            user_id=user_id,
            phone=phone,
            first_name=first_name,
            last_name=args.last_name,
            age=args.age,
            birthdate=args.birthdate,
            gender=gender,
        )
        presence_ok = set_presence_online(user_id)

        print(f"\n- {gender} test user")
        print(f"  user_id: {user_id}")
        print(f"  phone: {phone}")
        print(f"  created: {created}")
        print("  location_filter_enabled: false")
        print(f"  presence_online: {presence_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
