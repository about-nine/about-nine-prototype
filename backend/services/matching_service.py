# services/matching_service.py - matching filters and recommendation scoring
from typing import Dict, List, Tuple
import math
import random
import os

import numpy as np

from backend.services.firestore import get_firestore
from backend.services.pair_feature_model import (
    PairFeatureModel,
    _embedding_interaction,
    _talk_profile_features,
)
from backend.services.chemistry_model import ChemistryModel

DEFAULT_DISTANCE_KM = 10
BYPASS_USER_FILTERS = False

_pair_feature_model = PairFeatureModel()
_chemistry_model = ChemistryModel()

def _init_models():
    bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
    if bucket:
        _pair_feature_model.load(
            f"gs://{bucket}/models/pair_feature/latest.pkl"
        )
        _chemistry_model.load(
            f"gs://{bucket}/models/chemistry/latest.pkl"
        )

def distance_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def matches_orientation(orientation, target_gender) -> bool:
    if not target_gender:
        return False

    if not orientation:
        return True

    orientation = str(orientation).lower()
    target_gender = str(target_gender).lower()

    if orientation in ["everyone", "all", "all genders", "anyone", "all types of genders"]:
        return target_gender in ["man", "woman", "non-binary"]

    if "men and women" in orientation:
        return target_gender in ["man", "woman"]

    if orientation == "men" or "men only" in orientation:
        return target_gender == "man"

    if orientation == "women" or "women only" in orientation:
        return target_gender == "woman"

    if orientation == "non-binary" or "non-binary only" in orientation:
        return target_gender == "non-binary"

    if "men and non-binary" in orientation:
        return target_gender in ["man", "non-binary"]

    if "women and non-binary" in orientation:
        return target_gender in ["woman", "non-binary"]

    return False


def partners_completed_three_rounds(db, user_id: str):
    if not user_id:
        return set()

    talks_ref = db.collection("talk_history")
    partner_rounds = {}
    required = {1, 2, 3}

    queries = [
        talks_ref.where("participants.user_a", "==", user_id).stream(),
        talks_ref.where("participants.user_b", "==", user_id).stream(),
    ]

    for query in queries:
        for doc in query:
            talk = doc.to_dict() or {}
            if talk.get("completed") is not True:
                continue

            participants = talk.get("participants") or {}
            if participants.get("user_a") == user_id:
                partner_id = participants.get("user_b")
            else:
                partner_id = participants.get("user_a")

            if not partner_id:
                continue

            round_raw = talk.get("round") or 1
            try:
                round_num = int(round_raw)
            except (TypeError, ValueError):
                round_num = 1
            round_num = max(1, min(round_num, 3))

            partner_rounds.setdefault(partner_id, set()).add(round_num)

    return {
        pid for pid, rounds in partner_rounds.items()
        if required.issubset(rounds)
    }

_models_initialized = False

def _ensure_models():
    global _models_initialized
    if not _models_initialized:
        _init_models()
        _models_initialized = True

def recommend_for_user(uid: str, top_k: int = 5) -> List[Tuple[str, float]]:
    _ensure_models()
    if not uid:
        return []

    db = get_firestore()
    users: Dict[str, Dict] = {
        d.id: (d.to_dict() or {}) for d in db.collection("users").stream()
    }

    me = users.get(uid) or {}
    my_embedding = me.get("embedding") or {}
    my_vec = my_embedding.get("vector")
    my_default = my_embedding.get("is_default") is True

    my_blocked = set(me.get("blocked_users") or [])
    my_loc = me.get("location") or {}
    my_gender = me.get("gender")
    my_age = me.get("age")
    my_orientation = me.get("sexual_orientation")
    my_age_pref = me.get("age_preference", {})

    completed_partners = partners_completed_three_rounds(db, uid)

    scores: List[Tuple[str, float]] = []
    candidates: List[str] = []

    for other_id, user in users.items():
        if other_id == uid:
            continue

        if other_id in completed_partners:
            continue

        other_blocked = set(user.get("blocked_users") or [])
        if other_id in my_blocked or uid in other_blocked:
            continue

        if not user.get("onboarding_completed"):
            continue

        other_loc = user.get("location") or {}
        if not my_loc or not other_loc:
            continue

        try:
            d = distance_km(
                float(my_loc.get("lat")),
                float(my_loc.get("lng")),
                float(other_loc.get("lat")),
                float(other_loc.get("lng")),
            )
        except Exception:
            continue
        if d > DEFAULT_DISTANCE_KM:
            continue

        other_gender = user.get("gender")
        other_age = user.get("age")
        other_orientation = user.get("sexual_orientation")
        other_age_pref = user.get("age_preference", {})

        if not other_gender or other_age is None:
            continue

        if not matches_orientation(my_orientation, other_gender):
            continue

        if my_age_pref:
            if not (my_age_pref.get("min", 0) <= other_age <= my_age_pref.get("max", 100)):
                continue

        if not matches_orientation(other_orientation, my_gender):
            continue

        if other_age_pref:
            if not (other_age_pref.get("min", 0) <= my_age <= other_age_pref.get("max", 100)):
                continue

        candidates.append(other_id)
        if my_default or not my_vec:
            continue

        other_vec = (user.get("embedding") or {}).get("vector")
        if not other_vec:
            continue

        emb_interaction  = _embedding_interaction(my_vec, other_vec)
        talk_features    = _talk_profile_features(
            me.get("talk_profile") or {},
            user.get("talk_profile") or {},
        )
        predicted        = _pair_feature_model.predict(emb_interaction, talk_features)
        s                = _chemistry_model.predict(predicted) / 100.0
        scores.append((other_id, s))

    def _random_fallback():
        if not candidates:
            return []
        random.shuffle(candidates)
        return [(cid, 0.0) for cid in candidates[: min(top_k, len(candidates))]]

    if my_default or not my_vec:
        return _random_fallback()

    scores.sort(key=lambda x: -x[1])
    if not scores:
        return _random_fallback()
    return scores[:top_k]


def filter_users_for_list(user_id, bypass_filters=False):
    """
    주변 사용자 목록 (필터링 적용)

    필터:
    - 거리 10km 이내
    - 성적 지향 일치 (양방향)
    - 나이 선호 일치 (양방향)
    - 이미 3라운드를 완료한 파트너 제외
    """
    db = get_firestore()

    if bypass_filters:
        users = []
        for doc in db.collection("users").stream():
            u = doc.to_dict()
            if "id" not in u:
                u["id"] = doc.id
            if u["id"] == user_id:
                continue
            users.append(u)
        return users, {"bypass": True}, None

    me = db.collection("users").document(user_id).get().to_dict() or {}

    my_loc = me.get("location")
    my_gender = me.get("gender")
    my_age = me.get("age")
    my_sexual_orientation = me.get("sexual_orientation")
    my_age_pref = me.get("age_preference", {})
    my_blocked = set(me.get("blocked_users") or [])

    if not my_loc or my_loc.get("lat") is None or my_loc.get("lng") is None:
        return None, None, (400, "missing my location")

    if not my_gender or my_age is None:
        return None, None, (400, "missing my profile")

    completed_partners = partners_completed_three_rounds(db, user_id)

    users = []
    total_count = 0
    filtered_stats = {
        "same_user": 0,
        "blocked": 0,
        "no_onboarding": 0,
        "no_location": 0,
        "too_far": 0,
        "missing_gender_age": 0,
        "orientation_mismatch": 0,
        "age_mismatch": 0,
        "reverse_orientation": 0,
        "reverse_age": 0,
        "completed_all_rounds": 0,
        "passed": 0,
    }

    for doc in db.collection("users").stream():
        u = doc.to_dict()
        if "id" not in u:
            u["id"] = doc.id
        total_count += 1

        # 본인 제외
        if u["id"] == user_id:
            filtered_stats["same_user"] += 1
            continue

        # 차단 확인 (양방향)
        other_blocked = set(u.get("blocked_users") or [])
        if u.get("id") in my_blocked or user_id in other_blocked:
            filtered_stats["blocked"] += 1
            continue

        # 3라운드까지 완료한 파트너 제외
        if u["id"] in completed_partners:
            filtered_stats["completed_all_rounds"] += 1
            continue

        # 온보딩 완료 확인
        if not u.get("onboarding_completed"):
            filtered_stats["no_onboarding"] += 1
            continue

        # 위치 확인
        loc = u.get("location")
        if not loc:
            filtered_stats["no_location"] += 1
            continue

        # 거리 체크 (10km)
        d = distance_km(
            my_loc["lat"], my_loc["lng"],
            loc["lat"], loc["lng"]
        )
        if d > DEFAULT_DISTANCE_KM:
            filtered_stats["too_far"] += 1
            continue

        other_gender = u.get("gender")
        other_age = u.get("age")
        other_sexual_orientation = u.get("sexual_orientation")
        other_age_pref = u.get("age_preference", {})

        if not other_gender or not other_age:
            filtered_stats["missing_gender_age"] += 1
            continue

        # 내가 상대를 선호하는지
        if not matches_orientation(my_sexual_orientation, other_gender):
            filtered_stats["orientation_mismatch"] += 1
            continue

        if my_age_pref:
            if not (my_age_pref.get("min", 0) <= other_age <= my_age_pref.get("max", 100)):
                filtered_stats["age_mismatch"] += 1
                continue

        # 상대가 나를 선호하는지 (양방향 확인)
        if not matches_orientation(other_sexual_orientation, my_gender):
            filtered_stats["reverse_orientation"] += 1
            continue

        if other_age_pref:
            if not (other_age_pref.get("min", 0) <= my_age <= other_age_pref.get("max", 100)):
                filtered_stats["reverse_age"] += 1
                continue

        # 모든 필터 통과
        filtered_stats["passed"] += 1
        users.append(u)

    debug = {
        "total_count": total_count,
        "filtered_stats": filtered_stats,
        "me": me,
    }

    return users, debug, None