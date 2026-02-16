"""
Language Style Matching Analyzer (lsm)
=======================================
LIWC 기반 기능어(function word) 사용 패턴 유사도 측정.

측정 원리:
  1. 두 화자의 발화를 각각 토큰화
  2. 9개 기능어 카테고리(대명사, 관사, 전치사, 조동사, 부사, 접속사,
     부정어, 비인칭대명사, 한정사)별 사용 비율 계산
  3. 카테고리별 LSM = 1 - |rate_a - rate_b| / (rate_a + rate_b + ε)
  4. 전체 평균 → 100점 만점 환산

의미: 기능어는 무의식적으로 사용하므로, 두 사람이 비슷한
기능어 패턴을 보이면 심리적 동조(accommodation)가 일어나고 있다는 신호.

의존성: 없음 (pure Python)
"""

import re
from collections import Counter
from typing import Dict, Any, List, Set


# ============================================================================
# LIWC-level Function Word Lexicon
# ============================================================================

LEXICON: Dict[str, Set[str]] = {
    "ppron": {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
    },
    "ipron": {
        "it", "its", "itself",
        "this", "that", "these", "those",
        "anything", "something", "nothing", "everything",
    },
    "article": {"a", "an", "the"},
    "prep": {
        "in", "on", "at", "with", "by", "for", "to", "from",
        "about", "over", "under", "between", "into", "through",
        "during", "before", "after", "above", "below",
    },
    "auxverb": {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did",
        "have", "has", "had",
        "will", "would", "shall", "should",
        "can", "could", "may", "might", "must",
    },
    "adverb": {
        "very", "really", "just", "so", "too", "quite",
        "rather", "almost", "already", "still", "often",
        "never", "always", "sometimes",
    },
    "conj": {
        "and", "or", "but", "because", "so", "although",
        "though", "while", "whereas", "if", "unless",
    },
    "negate": {
        "no", "not", "never", "none", "nobody",
        "nothing", "nowhere", "cannot", "can't", "won't",
        "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    },
    "quant": {
        "all", "some", "many", "much", "few", "several",
        "most", "more", "less", "least", "enough",
    },
}

LSM_CATEGORIES = list(LEXICON.keys())
EPS = 1e-5


# ============================================================================
# Core functions
# ============================================================================

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def _category_rates(texts: List[str]) -> Dict[str, float]:
    tokens: List[str] = []
    for t in texts:
        tokens.extend(_tokenize(t))

    total = len(tokens)
    if total == 0:
        return {cat: 0.0 for cat in LSM_CATEGORIES}

    counts = Counter(tokens)
    rates = {}
    for cat, vocab in LEXICON.items():
        hit = sum(counts.get(w, 0) for w in vocab)
        rates[cat] = hit / total
    return rates


def _lsm_category(rate_a: float, rate_b: float) -> float:
    return 1.0 - abs(rate_a - rate_b) / (rate_a + rate_b + EPS)


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

    speakers_texts: Dict[str, List[str]] = {}
    for u in conv:
        sp = u.get("speaker")
        txt = u.get("text", "")
        if sp and txt:
            speakers_texts.setdefault(sp, []).append(txt)

    speaker_ids = list(speakers_texts.keys())
    if len(speaker_ids) < 2:
        return {"score": 0, "error": "need_2_speakers"}

    rates_a = _category_rates(speakers_texts[speaker_ids[0]])
    rates_b = _category_rates(speakers_texts[speaker_ids[1]])

    category_scores = {}
    for cat in LSM_CATEGORIES:
        category_scores[cat] = round(_lsm_category(rates_a[cat], rates_b[cat]), 4)

    total = sum(category_scores.values()) / len(category_scores)
    score_100 = int(round(total * 100))

    return {
        "score": score_100,
        "lsm_raw": round(total, 4),
        "category_scores": category_scores,
        "speaker_a_rates": {k: round(v, 4) for k, v in rates_a.items()},
        "speaker_b_rates": {k: round(v, 4) for k, v in rates_b.items()},
    }


class LSMAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        raw = analyze(data)
        return {
            "scores": {
                "lsm": float(raw.get("score", 0)),
            },
            "raw": raw,
        }