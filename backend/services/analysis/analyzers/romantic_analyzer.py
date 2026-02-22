"""Global Romantic Intent Analyzer (2-party conversation)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .language_utils import detect_conversation_language, detect_utterance_languages

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

EMBEDDING_THRESHOLD_BY_LANG: Dict[str, float] = {
    "ko": 0.35,
    "ja": 0.36,
    "zh": 0.36,
    "en": 0.40,
    "es": 0.38,
    "fr": 0.38,
    "de": 0.39,
    "pt": 0.38,
    "default": 0.40,
}

ROMANTIC_DECISION_THRESHOLD_BY_LANG: Dict[str, float] = {
    "ko": 0.42,
    "ja": 0.44,
    "zh": 0.44,
    "default": 0.45,
}

ROMANTIC_CATEGORIES = [
    "Longing",
    "Adoration",
    "Affection",
    "Care",
    "Intimacy",
    "Physical Response",
    "Future Together",
    "Exclusivity",
    "Curiosity",
    "Playfulness",
    "Availability",
]

ROMANTIC_SEED_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Affection": {
        "examples": [
            "I love you", "I really like you", "좋아해", "사랑해",
            "te quiero", "te amo", "je t'aime", "ich liebe dich", "eu te amo",
        ],
        "intensity": 1.0,
        "directness": 1.0,
    },
    "Longing": {
        "examples": [
            "I miss you", "can't wait to see you", "보고 싶어", "자꾸 생각나",
            "te extrano", "te extraño", "tu me manques", "ich vermisse dich", "sinto sua falta",
        ],
        "intensity": 0.85,
        "directness": 0.8,
    },
    "Care": {
        "examples": [
            "take care", "get home safe", "밥 먹었어", "건강 챙겨",
            "cuidate", "cuídate", "prends soin de toi", "pass auf dich auf", "se cuida",
        ],
        "intensity": 0.65,
        "directness": 0.55,
    },
    "Future Together": {
        "examples": [
            "let's meet again", "let's travel together", "같이 가자", "다음에 또 보자",
            "vamos juntos", "nos vemos otra vez", "on se voit encore", "lass uns zusammen", "vamos sair de novo",
        ],
        "intensity": 0.75,
        "directness": 0.65,
    },
    "Adoration": {
        "examples": [
            "you are beautiful", "you are amazing", "너 정말 예뻐", "멋있다",
            "eres preciosa", "eres increible", "tu es magnifique", "du bist wunderschon", "você é linda",
        ],
        "intensity": 0.8,
        "directness": 0.85,
    },
}

INTENSITY_LANGUAGE_OVERRIDES: Dict[str, Dict[str, float]] = {
    "ko": {
        # In Korean conversation, "care" expressions are stronger romantic signal.
        "Care": 0.72,
        "Intimacy": 0.85,
    }
}

MUTUALITY_PATTERNS = [
    r"\b(?:we|us|our|together|both|let's)\b",
    r"우리|같이|함께|서로",
    r"\b(?:nosotros|juntos|juntas|ambos)\b",
    r"\b(?:nous|ensemble|tous les deux)\b",
    r"\b(?:wir|zusammen|beide)\b",
    r"\b(?:nos|juntos|juntas|ambos)\b",
]

# Language adapters: high precision regex by language.
LANGUAGE_ADAPTERS: Dict[str, Dict[str, List[str]]] = {
    "ko": {
        "Affection": [r"좋아(?:해|합니다|해요|하는)", r"사랑(?:해|합니다|해요)", r"너라서\s*좋"],
        "Longing": [r"보고\s*싶", r"생각나", r"기다리"],
        "Care": [r"밥\s*먹", r"걱정", r"건강\s*챙", r"잘\s*자", r"잘\s*들어가", r"피곤하겠다", r"아프지\s*마"],
        "Future Together": [r"같이\s*(?:가자|보자|먹자|하자)", r"다음에\s*또", r"언제\s*시간"],
        "Adoration": [r"예쁘", r"멋있", r"귀엽", r"내\s*스타일"],
        "Intimacy": [r"자기야|자기\b|내\s*사랑|여보|우리\s*자기|아가"],
        "Physical Response": [r"설레|두근|심장\s*뛰|떨려|긴장돼|얼굴\s*빨개"],
        "Exclusivity": [r"너만|너밖에|특별해|처음이야|다른\s*사람.*아니"],
        "Curiosity": [r"더\s*알고\s*싶|궁금해|평소에\s*뭐\s*해|취미가\s*뭐야|어떤\s*사람"],
        "Playfulness": [r"ㅋㅋ+|ㅎㅎ+|장난이야|놀리는\s*거야|웃겨|귀엽게\s*구네"],
        "Availability": [r"연락해|카톡해|문자해|언제\s*봐|주말에\s*뭐\s*해|시간\s*돼"],
    },
    "en": {
        "Affection": [r"\bi\s+love\s+you\b", r"\bi\s+really\s+like\s+you\b", r"\bfallen\s+for\s+you\b"],
        "Longing": [r"\bmiss\s+you\b", r"\bcan'?t\s+wait\s+to\s+see\s+you\b", r"\bthinking\s+about\s+you\b"],
        "Care": [r"\btake\s+care\b", r"\bget\s+home\s+safe\b", r"\bworried\s+about\s+you\b"],
        "Future Together": [r"\blet'?s\b", r"\btogether\b", r"\bmeet\s+again\b", r"\bare\s+you\s+free\b"],
        "Adoration": [r"\bbeautiful\b", r"\bamazing\b", r"\bgorgeous\b", r"\battractive\b"],
        "Intimacy": [r"\b(?:sweetheart|honey|babe|darling|my love)\b"],
        "Physical Response": [r"\bbutterflies\b|\bheart\s+(?:racing|pounding)\b|\bnervous\s+around\s+you\b"],
        "Exclusivity": [r"\bonly\s+one\b|\bno\s+one\s+else\b|\bspecial\s+to\s+me\b"],
        "Curiosity": [r"\btell\s+me\s+more\b|\bwhat\s+do\s+you\s+do\s+for\s+fun\b|\bi'?d\s+love\s+to\s+know\s+more\b"],
        "Playfulness": [r"\byou'?re\s+so\s+funny\b|\bmaking\s+me\s+laugh\b|\bsuch\s+a\s+tease\b"],
        "Availability": [r"\bwhen\s+are\s+you\s+free\b|\blet\s+me\s+know\b|\bcontinue\s+this\s+conversation\b"],
    },
    "es": {
        "Affection": [r"te\s+quiero", r"te\s+amo", r"me\s+gustas"],
        "Longing": [r"te\s+extran", r"te\s+echo\s+de\s+menos"],
        "Care": [r"cu[ií]date", r"me\s+preocupa", r"descansa"],
        "Future Together": [r"vamos\s+juntos", r"nos\s+vemos\s+otra\s+vez", r"cuando\s+puedes"],
        "Adoration": [r"eres\s+hermos", r"eres\s+precios", r"incre[ií]ble"],
    },
    "fr": {
        "Affection": [r"je\s+t'aime", r"je\s+t'adore"],
        "Longing": [r"tu\s+me\s+manques"],
        "Care": [r"prends\s+soin", r"repose-toi"],
        "Future Together": [r"on\s+se\s+voit", r"ensemble", r"quand\s+tu\s+es\s+libre"],
        "Adoration": [r"tu\s+es\s+magnifique", r"tu\s+es\s+belle"],
    },
    "de": {
        "Affection": [r"ich\s+liebe\s+dich", r"ich\s+mag\s+dich"],
        "Longing": [r"ich\s+vermisse\s+dich"],
        "Care": [r"pass\s+auf\s+dich\s+auf", r"ruh\s+dich\s+aus"],
        "Future Together": [r"lass\s+uns\s+zusammen", r"sehen\s+wir\s+uns\s+wieder"],
        "Adoration": [r"du\s+bist\s+wundersch", r"du\s+bist\s+toll"],
    },
    "pt": {
        "Affection": [r"eu\s+te\s+amo", r"gosto\s+de\s+você", r"gosto\s+de\s+voce"],
        "Longing": [r"sinto\s+sua\s+falta"],
        "Care": [r"se\s+cuida", r"descansa"],
        "Future Together": [r"vamos\s+juntos", r"a\s+gente\s+se\s+vê\s+de\s+novo"],
        "Adoration": [r"você\s+é\s+linda", r"você\s+é\s+incrível", r"voce\s+e\s+linda"],
    },
}

KEYWORD_MAP: Dict[str, List[str]] = {
    "Affection": ["love you", "like you", "좋아해", "사랑해", "te quiero", "te amo", "je t'aime", "ich liebe dich", "eu te amo"],
    "Longing": ["miss you", "보고 싶", "생각나", "te extraño", "te extrano", "tu me manques", "vermisse dich", "sinto sua falta"],
    "Care": ["take care", "worry", "밥 먹", "걱정", "cuidate", "cuídate", "prends soin", "pass auf", "se cuida"],
    "Future Together": ["let's", "together", "같이", "또 보자", "vamos", "ensemble", "zusammen", "de novo"],
    "Adoration": ["beautiful", "amazing", "예뻐", "멋있", "preciosa", "magnifique", "wundersch", "linda"],
    "Intimacy": ["sweetheart", "honey", "babe", "자기야", "내 사랑", "여보"],
    "Physical Response": ["butterflies", "heart racing", "설레", "두근", "떨려"],
    "Exclusivity": ["only one", "너만", "너밖에", "special to me"],
    "Curiosity": ["tell me more", "궁금해", "더 알고 싶", "what do you do for fun"],
    "Playfulness": ["funny", "tease", "ㅋㅋ", "장난"],
    "Availability": ["when are you free", "언제 시간", "연락해", "다음에 또 봐"],
}


_seed_cache: Dict[str, List[np.ndarray]] = {}


def _normalize_conversation(conversation_obj) -> Dict:
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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _openai_embed(texts: List[str]) -> List[np.ndarray]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=float(os.getenv("OPENAI_TIMEOUT", "30")),
        max_retries=1,
    )
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _check_mutuality(text: str) -> float:
    lowered = text.lower()
    return min(sum(1 for p in MUTUALITY_PATTERNS if re.search(p, lowered)) * 0.3, 1.0)


def _embedding_threshold(lang: str) -> float:
    return EMBEDDING_THRESHOLD_BY_LANG.get(lang, EMBEDDING_THRESHOLD_BY_LANG["default"])


def _decision_threshold(lang: str) -> float:
    return ROMANTIC_DECISION_THRESHOLD_BY_LANG.get(lang, ROMANTIC_DECISION_THRESHOLD_BY_LANG["default"])


def _get_intensity(cat: str, lang: str = "unknown") -> float:
    lang_overrides = INTENSITY_LANGUAGE_OVERRIDES.get(lang, {})
    if cat in lang_overrides:
        return lang_overrides[cat]
    return ROMANTIC_SEED_EXAMPLES.get(cat, {}).get("intensity", 0.5)


def _get_directness(cat: str) -> float:
    return ROMANTIC_SEED_EXAMPLES.get(cat, {}).get("directness", 0.5)


def _get_seed_embeddings() -> Dict[str, List[np.ndarray]]:
    if _seed_cache:
        return _seed_cache

    texts: List[str] = []
    owners: List[str] = []
    for cat, data in ROMANTIC_SEED_EXAMPLES.items():
        for ex in data.get("examples", []):
            texts.append(ex)
            owners.append(cat)

    embeddings = _openai_embed(texts)
    for cat, emb in zip(owners, embeddings):
        _seed_cache.setdefault(cat, []).append(emb)
    return _seed_cache


def _classify_utterance_embedding(emb: np.ndarray, seeds: Dict[str, List[np.ndarray]]) -> Tuple[str, float]:
    best_cat = "Neutral"
    best_score = 0.0
    for cat, emb_list in seeds.items():
        sims = [_cosine_similarity(emb, s) for s in emb_list]
        if not sims:
            continue
        score = 0.7 * max(sims) + 0.3 * float(np.mean(sims))
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat, best_score


def _adapter_classify(text: str, lang: str) -> Tuple[str, float, str]:
    adapters = LANGUAGE_ADAPTERS.get(lang, {})
    for cat, patterns in adapters.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return cat, 0.78, "language_adapter"

    # Backoff to English adapter when latin script language is uncertain.
    if lang not in adapters and lang not in ("ko", "ja", "zh", "ar", "ru", "hi"):
        for cat, patterns in LANGUAGE_ADAPTERS.get("en", {}).items():
            for pat in patterns:
                if re.search(pat, text, flags=re.IGNORECASE):
                    return cat, 0.68, "language_adapter_backoff"

    return "Neutral", 0.0, "language_adapter"


def _keyword_classify(text: str) -> Tuple[str, float]:
    lowered = text.lower()
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw.isascii():
                if kw in lowered:
                    return cat, 0.55
            else:
                if kw in text:
                    return cat, 0.55
    return "Neutral", 0.0


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    romantic = [r for r in rows if r["is_romantic"]]
    avg_intensity = float(np.mean([r["intensity"] for r in romantic])) if romantic else 0.0
    avg_directness = float(np.mean([r["directness"] for r in romantic])) if romantic else 0.0
    avg_mutuality = float(np.mean([r["mutuality"] for r in rows])) if rows else 0.0
    avg_conf = float(np.mean([r["confidence"] for r in rows])) if rows else 0.0
    coverage = len(romantic) / len(rows) if rows else 0.0

    cat_counts: Dict[str, int] = {}
    for r in romantic:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    overall = avg_intensity * (0.5 + 0.5 * min(coverage * 2, 1.0))
    return {
        "score": int(round(overall * 100)),
        "intensity": round(avg_intensity, 4),
        "directness": round(avg_directness, 4),
        "mutuality": round(avg_mutuality, 4),
        "coverage": round(coverage, 4),
        "confidence": round(avg_conf, 4),
        "category_distribution": cat_counts,
        "evidence_spans": [
            {
                "speaker": r["speaker"],
                "language": r["language"],
                "category": r["category"],
                "method": r["method"],
                "confidence": round(r["confidence"], 4),
                "evidence": r["evidence"],
            }
            for r in romantic
        ],
    }


def analyze_global(data: Dict[str, Any]) -> Dict[str, Any]:
    conversation = data.get("conversation", [])
    if not conversation:
        return {"score": 0, "method": "empty"}

    utterances = [u.get("text", "") for u in conversation if u.get("text")]
    speakers = [u.get("speaker") for u in conversation if u.get("text")]
    if len(set(filter(None, speakers))) < 2:
        return {"score": 0, "error": "need_2_speakers"}

    # utterance-level language for adapter routing.
    lang_meta = detect_utterance_languages([{"speaker": s, "text": t} for s, t in zip(speakers, utterances)])
    langs = [u.get("language", "unknown") for u in lang_meta.get("utterances", [])]

    emb_rows: List[Tuple[str, float]] = []
    emb_method = "embedding_unavailable"
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            seeds = _get_seed_embeddings()
            utt_embs = _openai_embed(utterances)
            emb_rows = [_classify_utterance_embedding(e, seeds) for e in utt_embs]
            emb_method = "openai_embedding_multilingual"
        except Exception as exc:
            emb_rows = []
            emb_method = f"embedding_failed:{exc}"

    rows: List[Dict[str, Any]] = []
    for idx, (text, speaker) in enumerate(zip(utterances, speakers)):
        lang = langs[idx] if idx < len(langs) else "unknown"
        a_cat, a_conf, a_method = _adapter_classify(text, lang)
        k_cat, k_conf = _keyword_classify(text)

        best_cat = "Neutral"
        best_conf = 0.0
        method = "keyword_fallback"

        if emb_rows:
            e_cat, e_conf = emb_rows[idx]
            e_valid = (e_cat in ROMANTIC_CATEGORIES) and (e_conf >= _embedding_threshold(lang))
            a_valid = (a_cat in ROMANTIC_CATEGORIES) and (a_conf > 0)

            if a_valid and e_valid and a_cat == e_cat:
                best_cat = a_cat
                best_conf = min(1.0, max(e_conf, a_conf) + 0.08)
                method = "adapter+embedding_agree"
            elif a_valid and (not e_valid or a_conf >= e_conf + 0.08):
                best_cat = a_cat
                best_conf = a_conf
                method = a_method
            elif e_valid:
                best_cat = e_cat
                best_conf = e_conf
                method = emb_method
            elif k_cat in ROMANTIC_CATEGORIES:
                best_cat = k_cat
                best_conf = k_conf
                method = "keyword_fallback"
        else:
            if a_cat in ROMANTIC_CATEGORIES:
                best_cat = a_cat
                best_conf = a_conf
                method = a_method
            elif k_cat in ROMANTIC_CATEGORIES:
                best_cat = k_cat
                best_conf = k_conf
                method = "keyword_fallback"

        is_romantic = best_cat in ROMANTIC_CATEGORIES and best_conf >= _decision_threshold(lang)
        intensity = _get_intensity(best_cat, lang=lang) * best_conf if is_romantic else 0.0
        directness = _get_directness(best_cat) * best_conf if is_romantic else 0.0

        rows.append(
            {
                "speaker": speaker,
                "language": lang,
                "category": best_cat if is_romantic else "Neutral",
                "confidence": best_conf,
                "intensity": round(intensity, 4),
                "directness": round(directness, 4),
                "mutuality": _check_mutuality(text),
                "is_romantic": is_romantic,
                "method": method,
                "evidence": text,
            }
        )

    result = _aggregate(rows)
    result["method"] = "hybrid_adapter_embedding"
    result["embedding_status"] = emb_method
    return result


class RomanticAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        conv = data.get("conversation", [])

        lang_stats = detect_conversation_language(conv)
        utterance_stats = detect_utterance_languages(conv)

        raw = analyze_global(data)
        raw["language_detected"] = {
            "conversation": lang_stats,
            "utterance_level": utterance_stats,
        }
        return {
            "scores": {"romantic_intent": float(raw.get("score", 0))},
            "raw": raw,
        }
