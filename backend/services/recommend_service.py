from typing import Dict, List, Tuple
import math
import random

import numpy as np

from backend.services.firestore import get_firestore


def _get_db():
    return get_firestore()


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

def _distance_km(lat1, lng1, lat2, lng2) -> float:
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


def _matches_orientation(orientation, target_gender) -> bool:
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

    if "non-binary" == orientation or "non-binary only" in orientation:
        return target_gender == "non-binary"

    if "men and non-binary" in orientation:
        return target_gender in ["man", "non-binary"]

    if "women and non-binary" in orientation:
        return target_gender in ["woman", "non-binary"]

    return False


def _partners_completed_three_rounds(db, user_id: str):
    """
    Return partner IDs that have completed all 3 rounds with the user.
    """
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


def recommend_for_user(uid: str, top_k: int = 5) -> List[Tuple[str, float]]:
    if not uid:
        return []

    db = _get_db()
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

    completed_partners = _partners_completed_three_rounds(db, uid)

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
            d = _distance_km(
                float(my_loc.get("lat")),
                float(my_loc.get("lng")),
                float(other_loc.get("lat")),
                float(other_loc.get("lng")),
            )
        except Exception:
            continue
        if d > 10:
            continue

        other_gender = user.get("gender")
        other_age = user.get("age")
        other_orientation = user.get("sexual_orientation")
        other_age_pref = user.get("age_preference", {})

        if not other_gender or other_age is None:
            continue

        if not _matches_orientation(my_orientation, other_gender):
            continue

        if my_age_pref:
            if not (my_age_pref.get("min", 0) <= other_age <= my_age_pref.get("max", 100)):
                continue

        if not _matches_orientation(other_orientation, my_gender):
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

        s = _cosine(my_vec, other_vec)
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
