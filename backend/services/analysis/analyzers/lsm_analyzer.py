"""
Language Style Matching Analyzer (lsm)
=======================================
Single-file multilingual LSM with language adapters.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from .language_utils import detect_conversation_language, detect_utterance_languages


LEXICON: Dict[str, Set[str]] = {
    "ppron": {
        "i", "me", "my", "you", "your", "he", "she", "we", "us", "they", "them",
        "yo", "tu", "tú", "usted", "el", "él", "ella", "nosotros", "nosotras", "ellos", "ellas",
        "je", "tu", "vous", "il", "elle", "nous", "ils", "elles",
        "ich", "du", "sie", "er", "wir", "ihr", "ihnen",
        "eu", "tu", "voce", "você", "ele", "ela", "nos", "nós", "eles", "elas",
    },
    "ipron": {
        "it", "this", "that", "these", "those", "anything", "something", "nothing", "everything",
        "esto", "eso", "aquello", "algo", "nada", "todo",
        "ce", "cela", "quelque", "rien", "tout",
        "das", "dies", "jenes", "etwas", "nichts", "alles",
        "isso", "isto", "aquilo", "algo", "nada", "tudo",
    },
    "article": {
        "a", "an", "the", "el", "la", "los", "las", "un", "una", "unos", "unas",
        "le", "les", "une", "des", "der", "die", "das", "ein", "eine", "o", "os", "as", "um", "uma",
    },
    "prep": {
        "in", "on", "at", "with", "by", "for", "to", "from", "about", "between", "through",
        "en", "con", "por", "para", "de", "desde", "entre", "sobre",
        "dans", "avec", "pour", "depuis", "sur",
        "mit", "fur", "für", "von", "zu", "aus", "uber", "über", "zwischen",
        "em", "com", "para", "desde", "sobre",
    },
    "auxverb": {
        "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "may", "might", "must",
        "ser", "estar", "haber", "soy", "es", "son", "estoy", "está",
        "etre", "être", "avoir", "suis", "est", "sommes", "êtes", "ont",
        "sein", "haben", "bin", "bist", "ist", "sind", "war", "waren", "habe", "hat",
        "ser", "estar", "ter", "sou", "é", "são", "estou", "está", "tenho", "tem",
    },
    "adverb": {
        "very", "really", "just", "so", "too", "quite", "almost", "already", "still", "often", "never", "always", "sometimes",
        "muy", "realmente", "solo", "ya", "siempre", "nunca", "todavía", "a", "veces",
        "très", "vraiment", "déjà", "toujours", "jamais", "parfois",
        "sehr", "wirklich", "nur", "schon", "immer", "nie", "manchmal",
        "muito", "realmente", "só", "já", "sempre", "nunca", "às", "vezes",
    },
    "conj": {
        "and", "or", "but", "because", "so", "although", "though", "while", "if", "unless",
        "y", "o", "pero", "porque", "aunque", "si", "mientras",
        "et", "ou", "mais", "parce", "si", "alors", "pendant",
        "und", "oder", "aber", "weil", "wenn", "obwohl", "während",
        "e", "ou", "mas", "porque", "se", "embora", "enquanto",
    },
    "negate": {
        "no", "not", "never", "none", "nobody", "nothing", "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't",
        "no", "nunca", "nadie", "nada", "jamás",
        "ne", "pas", "jamais", "rien", "personne",
        "nicht", "kein", "keine", "niemals", "nie",
        "não", "nunca", "ninguém", "nada",
    },
    "quant": {
        "all", "some", "many", "much", "few", "several", "most", "more", "less", "least", "enough",
        "todo", "todos", "algunos", "muchos", "pocos", "más", "menos", "bastante",
        "tout", "tous", "quelques", "beaucoup", "peu", "plus", "moins", "assez",
        "alle", "einige", "viele", "wenige", "mehr", "weniger", "genug",
        "todo", "todos", "alguns", "muitos", "poucos", "mais", "menos", "bastante",
    },
}

LSM_CATEGORIES = list(LEXICON.keys())
EPS = 1e-5

# Korean adapter signals (single-file replacement for previous KO module).
KO_PPRON = {
    "나", "내", "저", "제", "우리", "저희", "너", "네", "니", "당신", "그", "그녀", "그들",
    "자기", "자신", "본인", "우리가", "너는", "저도", "나도",
}
KO_IPRON = {
    "이것", "그것", "저것", "이거", "그거", "저거", "여기", "거기", "저기", "이곳", "그곳",
    "뭐", "무엇", "누구", "어디", "아무것", "아무나", "모든것", "모든게",
}
KO_ADVERB = {
    "정말", "진짜", "너무", "아주", "매우", "항상", "자주", "가끔", "이미", "아직", "바로",
    "또", "또한", "다시", "함께", "같이", "그냥", "절대", "전혀", "분명히",
}
KO_QUANT = {"많이", "조금", "모두", "전부", "대부분", "여러", "몇", "더", "덜", "훨씬", "항상", "매번"}
KO_CONJ = {"그리고", "근데", "그런데", "하지만", "그러나", "그래서", "그러면", "또", "또한"}
KO_NEGATE = {"안", "못", "아니", "없", "않"}
KO_PREP_ENDINGS = ["에서", "에게", "한테", "으로", "로", "와", "과", "랑", "처럼", "보다", "에", "께"]
KO_CASE_ENDINGS = ["이", "가", "을", "를", "의"]
KO_AUX_ENDINGS = ["은", "는", "도", "만", "까지", "마저", "조차", "뿐"]
KO_CONN_ENDINGS = ["고", "지만", "는데", "면서", "면", "니까", "아서", "어서", "라도", "든지", "려고", "도록"]
KO_AUX_VERB_TOKENS = {"하다", "되다", "있다", "없다", "아니다", "같다", "싶다"}
KO_EOMI_ENDINGS = ["어요", "아요", "해요", "합니다", "다", "지", "네", "군요", "죠", "자"]
KO_PREENDING_MARKERS = ["었", "았", "겠", "시"]
KO_NEGATE_PATTERNS = [r"지\s*않", r"지\s*못", r"안\S+", r"못\S+", r"없(?:다|어|네)"]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text.lower(), flags=re.UNICODE)


def _lsm_category(rate_a: float, rate_b: float, min_rate: float = 0.004) -> Optional[float]:
    if rate_a + rate_b < min_rate:
        return None
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
            normalized.append(
                {
                    "speaker": getattr(u, "speaker", None),
                    "start": getattr(u, "start", None),
                    "end": getattr(u, "end", None),
                    "text": getattr(u, "text", ""),
                }
            )
    return {"conversation": normalized}


def _build_speakers_texts(conv: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    speakers_texts: Dict[str, List[str]] = {}
    for u in conv:
        sp = u.get("speaker")
        txt = u.get("text", "")
        if sp and txt:
            speakers_texts.setdefault(sp, []).append(txt)
    return speakers_texts


def _base_category_rates(texts: List[str]) -> Dict[str, float]:
    tokens: List[str] = []
    for text in texts:
        tokens.extend(_tokenize(text))

    total = len(tokens)
    if total == 0:
        return {cat: 0.0 for cat in LSM_CATEGORIES}

    counts = Counter(tokens)
    rates: Dict[str, float] = {}
    for cat, vocab in LEXICON.items():
        hit = sum(counts.get(w, 0) for w in vocab)
        rates[cat] = hit / total
    return rates


def _ko_adapter_counts(texts: List[str]) -> Tuple[Dict[str, int], int]:
    counts: Dict[str, int] = {cat: 0 for cat in LSM_CATEGORIES}
    token_count = 0

    for text in texts:
        for raw in text.split():
            eojeol = re.sub(r"[^\w가-힣]", "", raw)
            if not eojeol:
                continue
            token_count += 1

            if eojeol in KO_PPRON:
                counts["ppron"] += 1
            if eojeol in KO_IPRON:
                counts["ipron"] += 1
            if eojeol in KO_ADVERB:
                counts["adverb"] += 1
            if eojeol in KO_QUANT:
                counts["quant"] += 1
            if eojeol in KO_CONJ:
                counts["conj"] += 1
            if eojeol in KO_AUX_VERB_TOKENS:
                counts["auxverb"] += 1
            if eojeol in KO_NEGATE or any(key in eojeol for key in KO_NEGATE):
                counts["negate"] += 1
            elif any(re.search(pat, eojeol) for pat in KO_NEGATE_PATTERNS):
                counts["negate"] += 1

            if any(eojeol.endswith(j) and len(eojeol) > len(j) + 1 for j in KO_PREP_ENDINGS):
                counts["prep"] += 1
            if any(eojeol.endswith(j) and len(eojeol) > len(j) + 1 for j in KO_CASE_ENDINGS):
                counts["ppron"] += 1
            if any(eojeol.endswith(j) and len(eojeol) > len(j) + 1 for j in KO_AUX_ENDINGS):
                counts["quant"] += 1
            if any(eojeol.endswith(j) and len(eojeol) > len(j) + 1 for j in KO_CONN_ENDINGS):
                counts["conj"] += 1
            if any(eojeol.endswith(j) and len(eojeol) > len(j) + 1 for j in KO_EOMI_ENDINGS):
                counts["auxverb"] += 1
            if any(marker in eojeol for marker in KO_PREENDING_MARKERS):
                counts["auxverb"] += 1

    return counts, token_count


def _ko_adapter_rates(texts: List[str]) -> Dict[str, float]:
    counts, total = _ko_adapter_counts(texts)
    if total == 0:
        return {cat: 0.0 for cat in LSM_CATEGORIES}
    return {cat: counts[cat] / total for cat in LSM_CATEGORIES}


def _blend_rates(base_rates: Dict[str, float], ko_rates: Dict[str, float], ko_weight: float) -> Dict[str, float]:
    if ko_weight <= 0:
        return base_rates
    out: Dict[str, float] = {}
    for cat in LSM_CATEGORIES:
        out[cat] = base_rates.get(cat, 0.0) * (1.0 - ko_weight) + ko_rates.get(cat, 0.0) * ko_weight
    return out


def _style_fallback_score(texts_a: List[str], texts_b: List[str]) -> float:
    def stats(texts: List[str]) -> Dict[str, float]:
        joined = " ".join(texts)
        tokens = _tokenize(joined)
        token_count = max(len(tokens), 1)
        chars = max(len(joined), 1)
        return {
            "question": joined.count("?") / max(len(texts), 1),
            "exclaim": joined.count("!") / max(len(texts), 1),
            "comma": joined.count(",") / chars,
            "avg_token": sum(len(t) for t in tokens) / token_count,
        }

    a = stats(texts_a)
    b = stats(texts_b)

    sims = []
    for key in ("question", "exclaim", "comma", "avg_token"):
        va, vb = a[key], b[key]
        denom = max(va, vb, 1e-9)
        sims.append(max(0.0, 1.0 - abs(va - vb) / denom))
    return sum(sims) / len(sims)


def _speaker_ko_ratio(texts: List[str]) -> float:
    if not texts:
        return 0.0
    ko_chars = sum(len(re.findall(r"[\uAC00-\uD7A3]", t)) for t in texts)
    latin_chars = sum(len(re.findall(r"[A-Za-z]", t)) for t in texts)
    total = ko_chars + latin_chars
    return (ko_chars / total) if total else 0.0


def _compute_lsm(rates_a: Dict[str, float], rates_b: Dict[str, float]) -> Tuple[float, Dict[str, Optional[float]]]:
    category_scores: Dict[str, Optional[float]] = {}
    for cat in LSM_CATEGORIES:
        result = _lsm_category(rates_a[cat], rates_b[cat])
        category_scores[cat] = round(result, 4) if result is not None else None

    valid_scores = [s for s in category_scores.values() if s is not None]
    total = sum(valid_scores) / len(valid_scores) if valid_scores else -1.0
    return total, category_scores


def _analyze_texts(speakers_texts: Dict[str, List[str]]) -> Dict[str, Any]:
    speaker_ids = list(speakers_texts.keys())
    if len(speaker_ids) < 2:
        return {"score": 0, "error": "need_2_speakers"}

    texts_a = speakers_texts[speaker_ids[0]]
    texts_b = speakers_texts[speaker_ids[1]]

    base_a = _base_category_rates(texts_a)
    base_b = _base_category_rates(texts_b)

    ko_w_a = _speaker_ko_ratio(texts_a)
    ko_w_b = _speaker_ko_ratio(texts_b)
    ko_weight = (ko_w_a + ko_w_b) / 2.0

    ko_a = _ko_adapter_rates(texts_a) if ko_w_a >= 0.2 else {cat: 0.0 for cat in LSM_CATEGORIES}
    ko_b = _ko_adapter_rates(texts_b) if ko_w_b >= 0.2 else {cat: 0.0 for cat in LSM_CATEGORIES}

    blended_a = _blend_rates(base_a, ko_a, min(ko_w_a, 0.7))
    blended_b = _blend_rates(base_b, ko_b, min(ko_w_b, 0.7))

    lsm_total, category_scores = _compute_lsm(blended_a, blended_b)

    if lsm_total >= 0:
        method = "multilingual_lexicon_with_adapters"
        total = lsm_total
    else:
        total = _style_fallback_score(texts_a, texts_b)
        method = "style_fallback"

    valid_categories = sum(1 for v in category_scores.values() if v is not None)
    coverage = valid_categories / max(len(LSM_CATEGORIES), 1)
    token_signal = min((sum(len(_tokenize(t)) for t in texts_a + texts_b) / 80.0), 1.0)
    confidence = max(
        0.0,
        min(
            1.0,
            0.35
            + 0.30 * (1.0 if lsm_total >= 0 else 0.0)
            + 0.20 * coverage
            + 0.10 * ko_weight
            + 0.05 * token_signal,
        ),
    )

    return {
        "score": int(round(total * 100)),
        "lsm_raw": round(total, 4),
        "category_scores": category_scores,
        "speaker_a_rates": {k: round(v, 4) for k, v in blended_a.items()},
        "speaker_b_rates": {k: round(v, 4) for k, v in blended_b.items()},
        "adapter_weights": {
            "speaker_a_ko": round(ko_w_a, 4),
            "speaker_b_ko": round(ko_w_b, 4),
        },
        "confidence": round(confidence, 4),
        "method": method,
    }


def analyze(data: Dict) -> Dict[str, Any]:
    conv = data.get("conversation", [])
    speakers_texts = _build_speakers_texts(conv)
    return _analyze_texts(speakers_texts)


class LSMAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        conv = data.get("conversation", [])
        language_stats = detect_conversation_language(conv)
        utterance_lang = detect_utterance_languages(conv)

        raw = analyze(data)
        raw["language_detected"] = {
            "conversation": language_stats,
            "utterance_level": utterance_lang,
        }
        return {
            "scores": {"lsm": float(raw.get("score", 0))},
            "raw": raw,
        }
