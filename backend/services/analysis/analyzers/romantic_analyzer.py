"""
Romantic Intent Analyzer (romantic_intent)
==========================================
대화에서 로맨틱 의도를 감지하고 강도를 측정.

측정 원리:
  1. 각 발화를 OpenAI Embedding API로 벡터화
  2. 8개 로맨틱 카테고리(Longing, Adoration, Affection, Care,
     Intimacy, Physical Response, Future Together, Exclusivity)의
     seed example 벡터와 코사인 유사도 비교
  3. threshold(0.4) 이상이면 로맨틱 발화로 분류
  4. intensity × coverage 기반으로 최종 0~100점 환산

의존성: openai (embedding), numpy
Fallback: 키워드 매칭 (sentence-transformers/OpenAI 모두 없을 때)
"""

import os
import re
import json
import numpy as np
from typing import Dict, Any, List

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Romantic Intent Categories (Seed Examples)
# ============================================================================

ROMANTIC_SEED_EXAMPLES = {
    "Longing": {
        "examples": [
            "I miss you so much",
            "I can't wait to see you",
            "I wish you were here",
            "보고 싶어",
            "빨리 와",
        ],
        "intensity": 0.85,
        "directness": 0.8,
    },
    "Adoration": {
        "examples": [
            "You're so beautiful",
            "You're amazing",
            "I think you're incredible",
            "너 정말 예뻐",
            "너 진짜 멋있다",
        ],
        "intensity": 0.8,
        "directness": 0.9,
    },
    "Affection": {
        "examples": [
            "I love you",
            "I really like you",
            "You mean everything to me",
            "사랑해",
            "좋아해",
        ],
        "intensity": 1.0,
        "directness": 1.0,
    },
    "Care": {
        "examples": [
            "Have you eaten?",
            "Take care of yourself",
            "I'm worried about you",
            "밥 먹었어?",
            "걱정돼",
        ],
        "intensity": 0.6,
        "directness": 0.5,
    },
    "Intimacy": {
        "examples": [
            "My love",
            "Sweetheart",
            "Honey",
            "자기야",
            "내 사랑",
        ],
        "intensity": 0.8,
        "directness": 0.9,
    },
    "Physical Response": {
        "examples": [
            "My heart is racing",
            "I feel butterflies",
            "You make me nervous",
            "심장이 두근거려",
            "설레",
        ],
        "intensity": 0.85,
        "directness": 0.7,
    },
    "Future Together": {
        "examples": [
            "Let's travel together someday",
            "We should do this again",
            "I want to spend more time with you",
            "같이 여행 가자",
            "다음에 또 만나자",
        ],
        "intensity": 0.75,
        "directness": 0.7,
    },
    "Exclusivity": {
        "examples": [
            "You're the only one for me",
            "You're special to me",
            "I've never felt this way before",
            "너만 이래",
            "특별해",
        ],
        "intensity": 0.95,
        "directness": 0.85,
    },
}

MUTUALITY_PATTERNS = [
    r"\b(?:we|us|our|together|both)\b",
    r"each other",
    r"let's",
    r"우리",
    r"같이",
]


# ============================================================================
# Embedding helpers
# ============================================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-9:
        return 0.0
    return float(dot / norm)


def _openai_embed(texts: List[str]) -> List[np.ndarray]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _check_mutuality(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for p in MUTUALITY_PATTERNS if re.search(p, text_lower))
    return min(count * 0.4, 1.0)


# ============================================================================
# Embedding-based classifier
# ============================================================================

_seed_cache: Dict[str, Any] = {}


def _get_seed_embeddings() -> Dict[str, Any]:
    """Seed examples를 한 번만 임베딩하고 캐싱"""
    if _seed_cache:
        return _seed_cache

    all_texts = []
    index_map = []  # (category, idx_in_category)
    for cat, data in ROMANTIC_SEED_EXAMPLES.items():
        for i, ex in enumerate(data["examples"]):
            all_texts.append(ex)
            index_map.append((cat, i))

    embeddings = _openai_embed(all_texts)

    for (cat, _), emb in zip(index_map, embeddings):
        if cat not in _seed_cache:
            _seed_cache[cat] = {
                "embeddings": [],
                "intensity": ROMANTIC_SEED_EXAMPLES[cat]["intensity"],
                "directness": ROMANTIC_SEED_EXAMPLES[cat]["directness"],
            }
        _seed_cache[cat]["embeddings"].append(emb)

    return _seed_cache


def _classify_with_embeddings(utterances: List[str], speakers: List[str]) -> Dict[str, Any]:
    seeds = _get_seed_embeddings()
    utt_embeddings = _openai_embed(utterances)

    results = []
    for emb, text, speaker in zip(utt_embeddings, utterances, speakers):
        best_cat = "Neutral"
        best_score = 0.0

        for cat, data in seeds.items():
            sims = [_cosine_similarity(emb, s) for s in data["embeddings"]]
            score = 0.7 * max(sims) + 0.3 * np.mean(sims)
            if score > best_score:
                best_score = score
                best_cat = cat

        is_romantic = best_score >= 0.4

        results.append({
            "speaker": speaker,
            "category": best_cat if is_romantic else "Neutral",
            "score": round(best_score, 4),
            "intensity": round(seeds[best_cat]["intensity"] * best_score, 4) if is_romantic else 0.0,
            "directness": round(seeds[best_cat]["directness"] * best_score, 4) if is_romantic else 0.0,
            "mutuality": _check_mutuality(text),
            "is_romantic": is_romantic,
        })

    return _aggregate(results)


# ============================================================================
# Keyword fallback
# ============================================================================

_KEYWORD_MAP = {
    "Affection": ["love", "like you", "adore", "사랑", "좋아해"],
    "Longing": ["miss", "wait", "보고 싶", "기다"],
    "Care": ["eaten", "sleep well", "worried", "밥 먹", "걱정"],
    "Future Together": ["together", "again", "next time", "같이", "다음에"],
    "Adoration": ["beautiful", "amazing", "gorgeous", "예쁘", "멋있"],
    "Physical Response": ["heart", "butterflies", "nervous", "두근", "설레"],
    "Exclusivity": ["only one", "special", "never felt", "너만", "특별"],
    "Intimacy": ["honey", "sweetheart", "darling", "자기", "여보"],
}


def _classify_with_keywords(utterances: List[str], speakers: List[str]) -> Dict[str, Any]:
    results = []
    for text, speaker in zip(utterances, speakers):
        text_lower = text.lower()
        best_cat = "Neutral"
        is_romantic = False

        for cat, keywords in _KEYWORD_MAP.items():
            if any(kw in text_lower for kw in keywords):
                best_cat = cat
                is_romantic = True
                break

        intensity = ROMANTIC_SEED_EXAMPLES[best_cat]["intensity"] * 0.5 if is_romantic else 0.0

        results.append({
            "speaker": speaker,
            "category": best_cat,
            "score": 0.5 if is_romantic else 0.0,
            "intensity": intensity,
            "directness": ROMANTIC_SEED_EXAMPLES[best_cat]["directness"] * 0.5 if is_romantic else 0.0,
            "mutuality": _check_mutuality(text),
            "is_romantic": is_romantic,
        })

    return _aggregate(results)


def _aggregate(results: List[Dict]) -> Dict[str, Any]:
    romantic = [r for r in results if r["is_romantic"]]
    if romantic:
        avg_intensity = np.mean([r["intensity"] for r in romantic])
        avg_directness = np.mean([r["directness"] for r in romantic])
    else:
        avg_intensity = avg_directness = 0.0

    avg_mutuality = np.mean([r["mutuality"] for r in results]) if results else 0.0
    coverage = len(romantic) / len(results) if results else 0.0

    cat_counts = {}
    for r in romantic:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    overall = avg_intensity * (0.5 + 0.5 * min(coverage * 2, 1.0))
    score_100 = int(round(overall * 100))

    return {
        "score": score_100,
        "intensity": round(float(avg_intensity), 4),
        "directness": round(float(avg_directness), 4),
        "mutuality": round(float(avg_mutuality), 4),
        "coverage": round(coverage, 4),
        "category_distribution": cat_counts,
    }


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
    conversation = data["conversation"]
    if not conversation:
        return {"score": 0, "method": "empty"}

    utterances = [u["text"] for u in conversation if u.get("text")]
    speakers = [u["speaker"] for u in conversation if u.get("text")]

    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            result = _classify_with_embeddings(utterances, speakers)
            result["method"] = "openai_embedding"
            return result
        except Exception as e:
            result = _classify_with_keywords(utterances, speakers)
            result["method"] = "keyword_fallback"
            result["openai_error"] = str(e)
            return result

    result = _classify_with_keywords(utterances, speakers)
    result["method"] = "keyword_fallback"
    return result


class RomanticAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        raw = analyze(data)
        return {
            "scores": {
                "romantic_intent": float(raw.get("score", 0)),
            },
            "raw": raw,
        }