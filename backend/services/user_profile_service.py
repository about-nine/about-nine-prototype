# services/user_profile_service.py - user embedding and stats updates
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from firebase_admin import firestore

from backend.services.embedding_service import EmbeddingService, normalize_vector
from backend.services.firestore import get_firestore

def _get_db():
    return get_firestore()


def _now_ms() -> int:
    return int(time.time() * 1000)


DEFAULT_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def default_embedding_payload(source: str = "default") -> Dict[str, Any]:
    return {
        "is_default": True,
        "dim": DEFAULT_EMBEDDING_DIM,
        "updated_at": _now_ms(),
        "source": source,
    }


def transcripts_to_text(transcripts: Iterable[Dict[str, Any]]) -> str:
    if not transcripts:
        return ""
    lines: List[str] = []
    for item in transcripts:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        said = (item.get("said") or "").strip()
        if not said:
            continue
        if question:
            lines.append(f"Q: {question}\nA: {said}")
        else:
            lines.append(said)
    return "\n".join(lines).strip()


def embedding_payload_from_text(text: str, source: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    vec = EmbeddingService().encode_text(text)
    if not vec:
        return None
    return {
        "vector": vec,
        "dim": len(vec),
        "updated_at": _now_ms(),
        "is_default": False,
        "source": source,
    }


def _as_vector(values: Iterable) -> Optional[List[float]]:
    if not values:
        return None
    try:
        arr = np.array(list(values), dtype=float)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr.tolist()


def update_user_embedding(uid: str, pair_embedding, go: Optional[bool], alpha: float = 0.05) -> bool:
    if not uid:
        return False

    pair_vec = _as_vector(pair_embedding)
    if not pair_vec:
        return False

    db = _get_db()
    ref = db.collection("users").document(uid)
    snap = ref.get()
    data = snap.to_dict() or {}

    old_vec = (data.get("embedding") or {}).get("vector")
    old_vec = _as_vector(old_vec)

    pair_arr = np.array(pair_vec, dtype=float)

    if old_vec is None or len(old_vec) != len(pair_vec):
        old_arr = np.zeros_like(pair_arr)
    else:
        old_arr = np.array(old_vec, dtype=float)

    diff = pair_arr - old_arr  # 설계의 핵심: 현재와 pair 사이 방향
    
    if go is None:
        return True
    elif go:
        new_arr = old_arr + alpha * diff           # Go: 0.05 × (pair - old)
    else:
        new_arr = old_arr - alpha * diff            # No: -0.05 × (pair - old)

    new_vec = normalize_vector(new_arr.tolist())

    ref.set(
        {
            "embedding": {
                "vector": new_vec,
                "dim": len(new_vec),
                "updated_at": _now_ms(),
                "is_default": False,
                "source": "pair_embedding",
            }
        },
        merge=True,
    )
    return True


def update_user_stats(uid: str, is_go: bool) -> bool:
    if not uid:
        return False

    db = _get_db()
    ref = db.collection("users").document(uid)
    txn = db.transaction()

    @firestore.transactional
    def _update(transaction):
        snap = ref.get(transaction=transaction)
        data = snap.to_dict() or {}
        stats = data.get("stats") or {}
        talk_count = int(stats.get("talk_count") or 0)
        go_count = int(stats.get("go_count") or 0)

        talk_count += 1
        if is_go:
            go_count += 1

        go_rate = (go_count / talk_count) if talk_count else 0.0

        transaction.set(
            ref,
            {
                "stats": {
                    "talk_count": talk_count,
                    "go_count": go_count,
                    "go_rate": go_rate,
                }
            },
            merge=True,
        )

    _update(txn)
    return True
