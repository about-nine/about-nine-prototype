# services/analysis/analyzers/language_utils.py - language feature helpers
"""Utility helpers for lightweight language detection used by analyzers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_SPANISH_HINT_RE = re.compile(r"\b(el|la|los|las|de|que|y|en|por|para|muy)\b", re.IGNORECASE)
_FRENCH_HINT_RE = re.compile(r"\b(le|la|les|de|des|que|et|dans|pour|tr[eè]s)\b", re.IGNORECASE)
_GERMAN_HINT_RE = re.compile(r"\b(der|die|das|und|ich|nicht|mit|sehr)\b", re.IGNORECASE)
_PORTUGUESE_HINT_RE = re.compile(r"\b(o|a|os|as|de|que|e|em|para|muito)\b", re.IGNORECASE)


def _iter_texts(conversation: Iterable[Any]):
    for utterance in conversation or []:
        if isinstance(utterance, dict):
            text = utterance.get("text", "")
        else:
            text = getattr(utterance, "text", "")
        if text:
            yield str(text)


def _count_chars(text: str) -> Dict[str, int]:
    hangul = len(_HANGUL_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    return {"hangul": hangul, "latin": latin}


def detect_text_language(text: str, *, min_chars: int = 4) -> Dict[str, Any]:
    if not text:
        return {"label": "unknown", "confidence": 0.0, "counts": {}}

    counts = {
        "hangul": len(_HANGUL_RE.findall(text)),
        "latin": len(_LATIN_RE.findall(text)),
        "ja_kana": len(_HIRAGANA_KATAKANA_RE.findall(text)),
        "cjk": len(_CJK_RE.findall(text)),
        "arabic": len(_ARABIC_RE.findall(text)),
        "cyrillic": len(_CYRILLIC_RE.findall(text)),
        "devanagari": len(_DEVANAGARI_RE.findall(text)),
    }
    total = sum(counts.values())
    if total < max(1, min_chars):
        return {"label": "unknown", "confidence": 0.0, "counts": counts}

    max_bucket = max(counts, key=counts.get)
    confidence = counts[max_bucket] / total if total else 0.0

    if max_bucket == "hangul":
        label = "ko"
    elif max_bucket == "ja_kana":
        label = "ja"
    elif max_bucket == "arabic":
        label = "ar"
    elif max_bucket == "cyrillic":
        label = "ru"
    elif max_bucket == "devanagari":
        label = "hi"
    elif max_bucket == "cjk":
        label = "zh"
    elif max_bucket == "latin":
        lower = text.lower()
        if _SPANISH_HINT_RE.search(lower):
            label = "es"
        elif _FRENCH_HINT_RE.search(lower):
            label = "fr"
        elif _GERMAN_HINT_RE.search(lower):
            label = "de"
        elif _PORTUGUESE_HINT_RE.search(lower):
            label = "pt"
        else:
            label = "en"
    else:
        label = "unknown"

    return {"label": label, "confidence": round(confidence, 4), "counts": counts}


def detect_conversation_language(
    conversation: Iterable[Any],
    *,
    ko_threshold: float = 0.55,
    en_threshold: float = 0.35,
    min_chars: int = 40,
) -> Dict[str, Any]:
    """Return coarse language stats for the provided conversation iterable.

    The returned dictionary contains:
      - label: 'ko', 'en', 'mixed', or 'unknown'
      - hangul_ratio: fraction of Hangul characters among counted letters
      - latin_ratio: complementary latin fraction
      - hangul_chars / latin_chars / total_chars: absolute counts
    """

    hangul_total = 0
    latin_total = 0

    for text in _iter_texts(conversation):
        counts = _count_chars(text)
        hangul_total += counts["hangul"]
        latin_total += counts["latin"]

    total = hangul_total + latin_total
    ratio = (hangul_total / total) if total else 0.0

    if total < max(1, min_chars):
        label = "unknown"
    elif ratio >= ko_threshold:
        label = "ko"
    elif ratio <= en_threshold:
        label = "en"
    else:
        label = "mixed"

    return {
        "label": label,
        "hangul_ratio": ratio,
        "latin_ratio": 1.0 - ratio if total else 0.0,
        "hangul_chars": hangul_total,
        "latin_chars": latin_total,
        "total_chars": total,
    }


def detect_utterance_languages(conversation: Iterable[Any]) -> Dict[str, Any]:
    utterances: List[Dict[str, Any]] = []
    lang_counter: Counter = Counter()
    confidence_sum = 0.0

    for idx, utterance in enumerate(conversation or []):
        if isinstance(utterance, dict):
            text = str(utterance.get("text", "") or "")
            speaker = utterance.get("speaker")
        else:
            text = str(getattr(utterance, "text", "") or "")
            speaker = getattr(utterance, "speaker", None)
        info = detect_text_language(text)
        label = info["label"]
        if label != "unknown":
            lang_counter[label] += 1
        confidence_sum += float(info.get("confidence", 0.0))
        utterances.append(
            {
                "index": idx,
                "speaker": speaker,
                "text": text,
                "language": label,
                "confidence": info.get("confidence", 0.0),
            }
        )

    total = len(utterances)
    dominant = "unknown"
    if lang_counter:
        dominant = lang_counter.most_common(1)[0][0]

    return {
        "dominant_language": dominant,
        "language_distribution": dict(lang_counter),
        "avg_confidence": round(confidence_sum / total, 4) if total else 0.0,
        "utterances": utterances,
    }
