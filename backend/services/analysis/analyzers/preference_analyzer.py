"""
Preference Sync Analyzer (preference_sync)
===========================================
두 화자의 취향 일치도를 측정.

측정 원리:
  1. 취향 표현 추출: regex 패턴으로 "I love/like/enjoy..." 등 감지
  2. 명시적 동의 감지: "me too", "same here", "I agree" 등 카운트
  3. 의미적 유사도: 추출된 취향 발화를 OpenAI 임베딩으로 벡터화 후
     화자 간 코사인 유사도 매칭
  4. 카테고리별 일치도: Food, Music, Travel 등 카테고리별 취향 분포 비교
  5. 최종 점수 = 0.4×semantic + 0.3×agreement + 0.3×category_avg → 100점

의존성: openai (embedding), numpy
Fallback: 키워드 매칭 (OpenAI 없을 때)
"""

import os
import re
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Preference patterns
# ============================================================================

PREFERENCE_INDICATORS = {
    "positive": [
        r"i (?:really |absolutely |just )?(?:love|like|enjoy|prefer|adore)",
        r"(?:my |that's my )?favorite",
        r"i'm (?:a fan of|into|fond of)",
        r"i (?:always|usually|often) (?:eat|drink|watch|listen|go)",
        r"nothing beats",
        r"the best (?:thing|part)",
        r"(?:that's |it's |so )(?:great|awesome|amazing|the best)",
        r"i (?:go|do|have) (?:that |this |it )?(?:all the time|every ?(?:day|week))",
        r"(?:oh )?(?:yeah|yes),? i (?:know|love) (?:that|this)",
        r"(?:have you (?:tried|been to)|you should try)",
    ],
    "negative": [
        r"i (?:don't|do not|never) (?:like|enjoy|eat|watch)",
        r"i (?:hate|dislike|can't stand)",
        r"not (?:a fan of|into|my thing)",
    ],
}

AGREEMENT_PATTERNS = [
    r"me too",
    r"same (?:here)?",
    r"i (?:also|too) (?:love|like|enjoy)",
    r"(?:oh )?i love that too",
    r"exactly",
    r"i (?:totally )?agree",
    r"(?:that's )?so true",
    r"i feel the same",
    r"(?:no way,? )?me too",
]

CATEGORY_KEYWORDS = {
    "Food & Cuisine": [
        "food", "eat", "sushi", "restaurant", "cook", "taste",
        "drink", "menu", "meal", "cuisine", "dish", "flavor",
    ],
    "Music & Entertainment": [
        "music", "song", "movie", "show", "watch", "listen", "concert",
    ],
    "Travel & Places": [
        "travel", "trip", "visit", "country", "city", "vacation",
    ],
    "Lifestyle & Habits": [
        "morning", "everyday", "always", "routine", "habit",
    ],
    "Hobbies & Activities": [
        "hobby", "sport", "game", "exercise", "read", "write",
    ],
    "Values & Beliefs": [
        "believe", "think", "feel", "value", "important", "meaningful",
    ],
    "People & Relationships": [
        "friend", "family", "together", "people", "relationship",
    ],
}


# ============================================================================
# Helpers
# ============================================================================

@dataclass
class PrefItem:
    text: str
    category: str
    speaker: str
    sentiment: str = "positive"


def _categorize(text: str) -> str:
    text_lower = text.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            return cat
    return "General"


def _extract_preferences(utterances: List[str], speakers: List[str]) -> List[PrefItem]:
    prefs = []
    for utt, sp in zip(utterances, speakers):
        utt_lower = utt.lower()
        matched = False

        for pattern in PREFERENCE_INDICATORS["positive"]:
            if re.search(pattern, utt_lower):
                prefs.append(PrefItem(text=utt, category=_categorize(utt), speaker=sp, sentiment="positive"))
                matched = True
                break

        if not matched:
            for pattern in PREFERENCE_INDICATORS["negative"]:
                if re.search(pattern, utt_lower):
                    prefs.append(PrefItem(text=utt, category=_categorize(utt), speaker=sp, sentiment="negative"))
                    break
    return prefs


def _count_agreements(utterances: List[str]) -> int:
    count = 0
    for utt in utterances:
        utt_lower = utt.lower()
        for pattern in AGREEMENT_PATTERNS:
            if re.search(pattern, utt_lower):
                count += 1
                break
    return count


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 1e-9 else 0.0


def _openai_embed(texts: List[str]) -> List[np.ndarray]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _semantic_similarity_openai(
    prefs_a: List[PrefItem], prefs_b: List[PrefItem]
) -> float:
    if not prefs_a or not prefs_b:
        return 0.0

    texts_a = [p.text for p in prefs_a]
    texts_b = [p.text for p in prefs_b]
    emb_a = _openai_embed(texts_a)
    emb_b = _openai_embed(texts_b)

    total_sim = 0.0
    matched = 0
    used_b = set()

    for ea in emb_a:
        best_sim = 0.0
        best_j = -1
        for j, eb in enumerate(emb_b):
            if j in used_b:
                continue
            sim = _cosine_sim(ea, eb)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0 and best_sim > 0.3:
            total_sim += best_sim
            matched += 1
            used_b.add(best_j)

    return total_sim / max(matched, 1)


def _semantic_similarity_fallback(
    prefs_a: List[PrefItem], prefs_b: List[PrefItem]
) -> float:
    if not prefs_a or not prefs_b:
        return 0.0
    matched = 0
    for pa in prefs_a:
        words_a = set(pa.text.lower().split())
        for pb in prefs_b:
            words_b = set(pb.text.lower().split())
            overlap = words_a & words_b
            union = words_a | words_b
            if len(overlap) >= 2 or (len(union) > 0 and len(overlap) / len(union) > 0.2):
                matched += 1
                break
    return matched / max(len(prefs_a), len(prefs_b))


def _category_sync(prefs_a: List[PrefItem], prefs_b: List[PrefItem]) -> Dict[str, float]:
    all_cats = list(CATEGORY_KEYWORDS.keys()) + ["General"]
    sync = {}
    for cat in all_cats:
        ca = [p for p in prefs_a if p.category == cat]
        cb = [p for p in prefs_b if p.category == cat]
        if ca and cb:
            sync[cat] = min(len(ca), len(cb)) / max(len(ca), len(cb))
        elif ca or cb:
            sync[cat] = 0.2
    return sync


# ============================================================================
# Public interface
# ============================================================================


def _normalize_conversation(conversation_obj):
    if isinstance(conversation_obj, dict) and "conversation" in conversation_obj:
        return conversation_obj
    conv = getattr(conversation_obj, "conversation", None)
    if conv is None:
        return {"conversation": []}
    normalized = []
    for u in conv:
        if isinstance(u, dict):
            normalized.append(u)
        else:
            normalized.append({
                "speaker": getattr(u, "speaker", None),
                "start": getattr(u, "start", None),
                "end": getattr(u, "end", None),
                "text": getattr(u, "text", ""),
            })
    return {"conversation": normalized}


def analyze(data: Dict) -> Dict[str, Any]:
    conv = data["conversation"]
    if not conv:
        return {"score": 0, "method": "empty"}

    utterances = [u["text"] for u in conv if u.get("text")]
    speakers = [u["speaker"] for u in conv if u.get("text")]
    unique_speakers = list(dict.fromkeys(speakers))

    if len(unique_speakers) < 2:
        return {"score": 0, "error": "need_2_speakers"}

    sp_a, sp_b = unique_speakers[0], unique_speakers[1]

    # 1. Extract preferences
    all_prefs = _extract_preferences(utterances, speakers)
    prefs_a = [p for p in all_prefs if p.speaker == sp_a]
    prefs_b = [p for p in all_prefs if p.speaker == sp_b]
    
    # prefs가 하나도 없으면 기본값
    if not prefs_a and not prefs_b:
        return {"score": 50, "error": "no_preferences_detected", "method": "default"}

    # 2. Explicit agreements
    agreement_count = _count_agreements(utterances)
    agreement_score = min(agreement_count / max(len(utterances) * 0.2, 1), 1.0)

    # 3. Semantic similarity
    use_openai = HAS_OPENAI and os.getenv("OPENAI_API_KEY")
    method = "openai_embedding"
    try:
        if use_openai and prefs_a and prefs_b:
            semantic_sim = _semantic_similarity_openai(prefs_a, prefs_b)
        else:
            semantic_sim = _semantic_similarity_fallback(prefs_a, prefs_b)
            method = "keyword_fallback"
    except Exception:
        semantic_sim = _semantic_similarity_fallback(prefs_a, prefs_b)
        method = "keyword_fallback"

    # 4. Category sync
    cat_sync = _category_sync(prefs_a, prefs_b)
    cat_avg = float(np.mean(list(cat_sync.values()))) if cat_sync else 0.0

    # 5. Final score
    final = 0.40 * semantic_sim + 0.30 * agreement_score + 0.30 * cat_avg
    score_100 = int(round(final * 100))

    return {
        "score": score_100,
        "semantic_similarity": round(semantic_sim, 4),
        "explicit_agreement_score": round(agreement_score, 4),
        "explicit_agreement_count": agreement_count,
        "category_sync": {k: round(v, 4) for k, v in cat_sync.items()},
        "speaker_a_prefs": len(prefs_a),
        "speaker_b_prefs": len(prefs_b),
        "method": method,
    }


class PreferenceAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        raw = analyze(data)
        return {
            "scores": {
                "preference_sync": float(raw.get("score", 0)),
            },
            "raw": raw,
        }