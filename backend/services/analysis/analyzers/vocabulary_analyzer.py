"""
Vocabulary Diversity Analyzer (vocabulary_diversity)
=====================================================
발화 내 어휘 다양성을 type-token ratio(TTR) 기반으로 측정.

측정 원리:
  TTR = unique tokens / total tokens

  단, 순수 TTR은 발화 길이에 반비례하는 문제가 있음.
  → MATTR(Moving Average TTR) 사용: 고정 윈도우(50토큰)로 슬라이딩 평균
  → 발화 길이가 달라도 비교 가능

개인 피처 (personal):
  speaker별 MATTR → avg_turn_length처럼 EMA로 누적

페어 피처 (pair score):
  두 화자의 vocabulary_diversity 유사도
  → 1 - |ttr_a - ttr_b| 방식

다국어 토크나이저:
  한국어: 어절(공백) 분리 + 조사 제거
  영어/기타: 알파벳 단어 단위 분리 + 소문자화

STT 이슈:
  문장 부호 의존 없음 → 영향 없음

의존성: 없음 (pure Python)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from analyzers.language_utils import detect_conversation_language

# MATTR 윈도우 크기
MATTR_WINDOW = 50

# 한국어 조사/어미 제거용 접미사 (길이 내림차순)
_KO_JOSA = sorted([
    "에서부터", "으로부터", "로부터", "에게서", "한테서",
    "이라도", "이든지", "든지", "으로서", "로서", "으로써", "로써",
    "에게", "한테", "께서", "에서", "으로", "처럼", "보다", "까지",
    "마저", "조차", "뿐", "이다", "예요", "이에요",
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로", "와", "과", "랑",
], key=len, reverse=True)

# 불용어 (너무 짧거나 의미 없는 토큰)
_EN_STOPWORDS = {
    "i", "you", "he", "she", "we", "they", "it",
    "a", "an", "the", "and", "or", "but", "so",
    "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for",
}


# ============================================================================
# Tokenizers
# ============================================================================

def _ko_stem(token: str) -> str:
    """한국어 어절에서 조사/어미 제거 후 어간 반환."""
    for josa in _KO_JOSA:
        if token.endswith(josa) and len(token) > len(josa) + 1:
            return token[: -len(josa)]
    return token


def _tokenize_ko(text: str) -> List[str]:
    """
    한국어 토크나이저.
    공백 기준 어절 분리 → 특수문자 제거 → 조사 제거 → 2글자 미만 제외
    """
    tokens = []
    for raw in text.split():
        cleaned = re.sub(r"[^가-힣a-zA-Z0-9]", "", raw)
        if not cleaned:
            continue
        stemmed = _ko_stem(cleaned)
        if len(stemmed) >= 2:
            tokens.append(stemmed)
    return tokens


def _tokenize_en(text: str) -> List[str]:
    """
    영어/라틴계 토크나이저.
    알파벳 단어 추출 → 소문자화 → 불용어 제거 → 2글자 미만 제외
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= 2 and w not in _EN_STOPWORDS]


def _tokenize(text: str, lang: str) -> List[str]:
    """언어에 따라 적절한 토크나이저 선택."""
    if lang == "ko":
        return _tokenize_ko(text)
    return _tokenize_en(text)


# ============================================================================
# MATTR (Moving Average Type-Token Ratio)
# ============================================================================

def _mattr(tokens: List[str], window: int = MATTR_WINDOW) -> Optional[float]:
    """
    MATTR 계산.
    토큰이 window보다 적으면 단순 TTR로 fallback.
    토큰이 10개 미만이면 None (신뢰 불가).
    """
    n = len(tokens)
    if n < 10:
        return None
    if n <= window:
        # 단순 TTR
        return len(set(tokens)) / n

    # 슬라이딩 윈도우 평균
    ttrs = []
    for i in range(n - window + 1):
        window_tokens = tokens[i: i + window]
        ttrs.append(len(set(window_tokens)) / window)
    return sum(ttrs) / len(ttrs)


# ============================================================================
# Per-speaker analysis
# ============================================================================

def _analyze_speaker(texts: List[str], lang: str) -> Dict[str, Any]:
    """
    단일 화자의 발화 리스트로 vocabulary_diversity 계산.

    Returns:
        diversity : MATTR 기반 0~100 점수 (None이면 샘플 부족)
        token_count : 총 토큰 수
        unique_count : 고유 토큰 수
        raw_ttr : 단순 TTR (참고용)
    """
    all_tokens: List[str] = []
    for text in texts:
        all_tokens.extend(_tokenize(text, lang))

    if len(all_tokens) < 10:
        return {
            "diversity": None,
            "token_count": len(all_tokens),
            "unique_count": len(set(all_tokens)),
            "raw_ttr": None,
            "note": "insufficient_tokens",
        }

    mattr_val = _mattr(all_tokens)
    raw_ttr = len(set(all_tokens)) / len(all_tokens)

    # 0~100 변환: MATTR은 보통 0.4~0.9 범위
    # 0.4 → 0점, 0.9 → 100점으로 선형 스케일
    if mattr_val is not None:
        scaled = (mattr_val - 0.4) / 0.5  # 0~1
        diversity = round(max(0.0, min(1.0, scaled)) * 100, 2)
    else:
        diversity = None

    return {
        "diversity": diversity,
        "token_count": len(all_tokens),
        "unique_count": len(set(all_tokens)),
        "raw_ttr": round(raw_ttr, 4),
        "mattr": round(mattr_val, 4) if mattr_val is not None else None,
    }


# ============================================================================
# Main analyze
# ============================================================================

def _normalize_conversation(conversation_obj) -> Dict[str, Any]:
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


def analyze(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    conv = conversation_data.get("conversation", [])
    if not conv:
        return {"score": None, "error": "empty"}

    # 언어 감지 (대화 전체 기준)
    lang_stats = detect_conversation_language(conv)
    lang = lang_stats.get("label", "en")
    if lang in ("mixed", "unknown"):
        # 혼합/불명이면 영어 토크나이저 사용 (latin 처리 가능)
        lang = "en"

    # speaker별 발화 분리
    speakers_texts: Dict[str, List[str]] = {}
    for u in conv:
        sp = u.get("speaker")
        text = (u.get("text") or "").strip()
        if sp and text:
            speakers_texts.setdefault(sp, []).append(text)

    unique_speakers = list(speakers_texts.keys())
    if len(unique_speakers) < 2:
        return {"score": None, "error": "need_2_speakers"}

    speaker_a, speaker_b = unique_speakers[0], unique_speakers[1]

    result_a = _analyze_speaker(speakers_texts[speaker_a], lang)
    result_b = _analyze_speaker(speakers_texts[speaker_b], lang)

    return {
        "personal": {
            speaker_a: {"vocabulary_diversity": result_a["diversity"]},
            speaker_b: {"vocabulary_diversity": result_b["diversity"]},
        },
        "detail": {
            speaker_a: result_a,
            speaker_b: result_b,
        },
        "language": lang,
        "language_stats": lang_stats,
    }


# ============================================================================
# Class interface (다른 analyzer와 동일한 구조)
# ============================================================================

class VocabularyAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        raw = analyze(data)
        return {
            "scores": {},           # pair feature 없음
            "personal": raw.get("personal", {}),
            "raw": raw,
        }