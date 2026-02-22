"""
Global Preference Sync Analyzer (2-party conversation)
=======================================================
A multilingual preference analyzer that uses an embedding-first pipeline
with regex fallback for robust operation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .language_utils import detect_conversation_language, detect_utterance_languages
from .analyzer_core import (
    BaseSyncModule,
    ConversationSyncCore,
    SyncItem,
    normalize_conversation,
)

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


INTENT_LABELS = (
    "preference_positive",
    "preference_negative",
    "agreement",
    "disagreement",
    "neutral",
)

# Multilingual prototypes used for embedding intent classification.
INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "preference_positive": [
        "I really like this",
        "This is my favorite",
        "I enjoy this often",
        "저는 이걸 정말 좋아해요",
        "이게 제 취향이에요",
        "Me encanta esto",
        "Esto es mi favorito",
        "J'aime beaucoup ca",
    ],
    "preference_negative": [
        "I do not like this",
        "I hate this",
        "This is not my thing",
        "저는 이거 별로예요",
        "이건 싫어요",
        "No me gusta esto",
        "Je n'aime pas ca",
    ],
    "agreement": [
        "I agree with you",
        "Me too",
        "Same here",
        "저도요",
        "나도 그래",
        "Estoy de acuerdo",
        "Yo tambien",
        "Moi aussi",
    ],
    "disagreement": [
        "I disagree",
        "Not really",
        "I don't think so",
        "저는 좀 달라요",
        "저는 동의하지 않아요",
        "No estoy de acuerdo",
        "Pas vraiment",
    ],
    "neutral": [
        "Hello how are you",
        "Let's continue",
        "Okay",
        "안녕하세요",
        "계속 이야기해요",
    ],
}

FALLBACK_INTENT_PATTERNS: Dict[str, List[str]] = {
    "preference_positive": [
        r"\bi\s+(really\s+|just\s+)?(love|like|enjoy|prefer|adore)\b",
        r"\bmy\s+favorite\b",
        r"\b(i\s+am|i'm)\s+(a\s+fan\s+of|into)\b",
        r"좋아해|좋아요|좋아하|즐겨|취향|선호|맛있|재미있|최고|강추",
        r"(?:매일|자주|항상)\s*(?:먹|마시|보|듣|하|가)",
        r"(?:운동|독서|요리|산책|게임|여행)(?:을|를)?\s*즐겨",
        r"me\s+encanta|me\s+gusta|mi\s+favorito",
        r"j'?adore|j'aime\s+beaucoup|mon\s+pr[ée]f[ée]r[ée]",
        r"ich\s+(?:mag|liebe)\s+das|mein\s+favorit",
        r"eu\s+(?:gosto|adoro)|meu\s+favorito",
        r"j'aime|prefere",
    ],
    "preference_negative": [
        r"\bi\s+(do\s+not|don't|never)\s+(like|enjoy|prefer)\b",
        r"\b(i\s+)?(hate|dislike|can't\s+stand)\b",
        r"싫어|별로|안\s*좋아|못\s*먹|귀찮|힘들어",
        r"잘\s*안\s*(?:먹|마시|보|듣|하)|안\s*(?:맞아|맞는)",
        r"no\s+me\s+gusta|odio",
        r"je\s+n'aime\s+pas|je\s+d[ée]teste",
        r"ich\s+mag\s+.*nicht|ich\s+hasse",
        r"n[aã]o\s+gosto|odeio",
        r"je\s+n'aime\s+pas",
    ],
    "agreement": [
        r"\bme\s+too\b|\bsame\s+here\b|\bi\s+agree\b|\bexactly\b",
        r"저도요|나도|맞아요|맞아|그러게요|동감|공감|저도\s*그래요|나도\s*그래",
        r"완전\s*(?:공감|동의|맞아)",
        r"yo\s+tambien|estoy\s+de\s+acuerdo",
        r"moi\s+aussi|je\s+suis\s+d'accord",
        r"ich\s+stimme\s+zu|ich\s+auch",
        r"eu\s+tamb[eé]m|concordo",
        r"moi\s+aussi|d'accord",
    ],
    "disagreement": [
        r"\bi\s+disagree\b|\bnot\s+really\b|\bi\s+don't\s+think\s+so\b",
        r"동의\s*안|아닌데|다르|글쎄",
        r"no\s+estoy\s+de\s+acuerdo|para\s+nada",
        r"je\s+ne\s+suis\s+pas\s+d'accord",
        r"ich\s+bin\s+nicht\s+einverstanden",
        r"n[aã]o\s+concordo",
        r"pas\s+vraiment",
    ],
}

PREFERENCE_CATEGORIES: Dict[str, List[str]] = {
    "Food": [
        "food", "eat", "drink", "cuisine", "dish", "restaurant", "cook",
        "음식", "먹", "요리", "맛집", "카페", "커피", "식당", "레스토랑", "디저트", "빵", "맥주", "와인",
        "comida", "cocina", "restaurante", "cafe", "cafeteria", "postre",
        "nourriture", "restaurant", "cuisine",
        "essen", "restaurant", "küche", "kuche",
        "comida", "restaurante", "cozinha",
    ],
    "Music & Entertainment": [
        "music", "song", "movie", "show", "concert", "game",
        "음악", "노래", "영화", "드라마", "공연", "게임",
        "musica", "pelicula", "serie", "musique", "film",
    ],
    "Travel": [
        "travel", "trip", "city", "country", "vacation", "flight", "hotel",
        "여행", "도시", "해외", "국내", "호텔", "숙소",
        "viaje", "ciudad", "pais", "vacaciones", "voyage",
    ],
    "Lifestyle": [
        "routine", "habit", "morning", "weekend", "daily", "sleep",
        "루틴", "습관", "일상", "아침", "주말", "수면",
        "rutina", "habito", "quotidien",
    ],
    "Hobbies": [
        "hobby", "exercise", "reading", "writing", "sports", "drawing",
        "취미", "운동", "독서", "글쓰기", "등산", "요가", "수영", "달리기", "게임", "사진",
        "pasatiempo", "deporte", "lecture", "aficion",
        "loisir", "sport", "lecture",
        "hobby", "sport", "lesen",
        "hobby", "esporte", "leitura",
    ],
    "Values": [
        "value", "belief", "important", "meaning", "future", "goal",
        "가치", "신념", "중요", "의미", "미래", "목표",
        "valor", "creencia", "objectif",
    ],
    "Relationships": [
        "friend", "family", "partner", "people", "together", "relationship",
        "친구", "가족", "연인", "사람", "함께", "관계",
        "amigo", "familia", "relation",
    ],
}

_SIMPLE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "i", "you", "it", "we", "they", "this", "that", "저", "나", "이", "그", "것",
    "de", "la", "el", "y", "que", "et", "le", "les",
}

_KO_JOSA_SUFFIXES = sorted(
    [
        "에서부터", "으로부터", "로부터", "에게서", "한테서", "이라든가", "라든가",
        "이라도", "이든지", "든지", "으로서", "로서", "으로써", "로써",
        "에게", "한테", "께서", "에서", "으로", "처럼", "보다", "까지",
        "마저", "조차", "뿐", "에게", "한테", "이다", "예요", "이에요",
        "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로", "와", "과", "랑",
    ],
    key=len,
    reverse=True,
)


@dataclass
class IntentResult:
    label: str
    confidence: float
    method: str
    evidence: str


def _openai_timeout() -> float:
    raw = os.getenv("OPENAI_TIMEOUT", "30")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 30.0


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _weight_env(defaults: Tuple[float, float, float]) -> Tuple[float, float, float]:
    raw = os.getenv("PREFERENCE_WEIGHTS", "")
    if not raw:
        return defaults
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        return defaults
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        return defaults
    if sum(vals) <= 0:
        return defaults
    return vals


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 1e-9 else 0.0


def _normalize_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s\uAC00-\uD7A3\u00C0-\u024F]", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) >= 2 and t not in _SIMPLE_STOPWORDS]
    return tokens


def _ko_stem(token: str) -> str:
    out = token
    for suffix in _KO_JOSA_SUFFIXES:
        if out.endswith(suffix) and len(out) > len(suffix) + 1:
            out = out[: -len(suffix)]
            break
    return out


def _token_jaccard(a: str, b: str) -> float:
    ta_raw = _normalize_tokens(a)
    tb_raw = _normalize_tokens(b)
    ta = set(_ko_stem(t) for t in ta_raw)
    tb = set(_ko_stem(t) for t in tb_raw)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _alias_found(text: str, alias: str) -> bool:
    if re.fullmatch(r"[a-z0-9\s&]+", alias):
        return re.search(r"\b" + re.escape(alias) + r"\b", text, flags=re.IGNORECASE) is not None
    return alias in text


class GlobalPreferenceModule(BaseSyncModule):
    method_label = "global_multilingual"

    def __init__(self):
        semantic_w, agreement_w, category_w = _weight_env((0.40, 0.30, 0.30))
        self.semantic_weight = semantic_w
        self.agreement_weight = agreement_w
        self.category_weight = category_w

        self.intent_min_conf = _float_env("PREFERENCE_INTENT_MIN_CONF", 0.32)
        self.semantic_match_threshold = _float_env("PREFERENCE_SEMANTIC_MATCH_THRESHOLD", 0.33)
        self.agreement_norm_factor = _float_env("PREFERENCE_AGREEMENT_NORM", 0.22)

        self._client = None
        self._prototype_embeds: Dict[str, List[np.ndarray]] = {}
        self._last_intents: List[IntentResult] = []
        self._last_meta: Dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=_openai_timeout(),
                max_retries=1,
            )
        return self._client

    def _openai_embed(self, texts: List[str]) -> List[np.ndarray]:
        client = self._get_client()
        resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
        return [np.array(item.embedding) for item in resp.data]

    def _ensure_prototype_embeddings(self):
        if self._prototype_embeds:
            return
        flat_texts: List[str] = []
        ranges: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for label in INTENT_LABELS:
            samples = INTENT_PROTOTYPES.get(label, [])
            start = cursor
            flat_texts.extend(samples)
            cursor += len(samples)
            ranges[label] = (start, cursor)
        embeddings = self._openai_embed(flat_texts)
        for label, (start, end) in ranges.items():
            self._prototype_embeds[label] = embeddings[start:end]

    def _intent_from_regex(self, text: str) -> IntentResult:
        for label in ("preference_positive", "preference_negative", "agreement", "disagreement"):
            for pattern in FALLBACK_INTENT_PATTERNS.get(label, []):
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    evidence = match.group(0).strip() or text
                    return IntentResult(label=label, confidence=0.65, method="regex_fallback", evidence=evidence)
        return IntentResult(label="neutral", confidence=0.5, method="regex_fallback", evidence="")

    def _intent_from_embedding(self, text: str) -> IntentResult:
        text_vec = self._openai_embed([text])[0]
        best_label = "neutral"
        best_score = -1.0

        for label in INTENT_LABELS:
            label_vecs = self._prototype_embeds.get(label, [])
            if not label_vecs:
                continue
            sims = sorted((_cosine_sim(text_vec, p) for p in label_vecs), reverse=True)
            if not sims:
                continue
            score = float(np.mean(sims[:2])) if len(sims) > 1 else sims[0]
            if score > best_score:
                best_score = score
                best_label = label

        if best_score < self.intent_min_conf:
            return IntentResult(label="neutral", confidence=max(best_score, 0.0), method="embedding", evidence=text)
        return IntentResult(label=best_label, confidence=max(best_score, 0.0), method="embedding", evidence=text)

    def _classify_intents(self, utterances: List[str]) -> List[IntentResult]:
        use_openai = HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
        results: List[IntentResult] = []

        if use_openai:
            try:
                self._ensure_prototype_embeddings()
                for utt in utterances:
                    emb_result = self._intent_from_embedding(utt)
                    if emb_result.label == "neutral":
                        regex_result = self._intent_from_regex(utt)
                        # Keep regex decision only when it finds a stronger actionable label.
                        if regex_result.label != "neutral":
                            regex_result.confidence = max(regex_result.confidence, emb_result.confidence)
                            results.append(regex_result)
                            continue
                    results.append(emb_result)
                return results
            except Exception:
                pass

        for utt in utterances:
            results.append(self._intent_from_regex(utt))
        return results

    def _categorize(self, text: str) -> str:
        lowered = text.lower()
        for category, aliases in PREFERENCE_CATEGORIES.items():
            for alias in aliases:
                if _alias_found(lowered if alias.isascii() else text, alias.lower() if alias.isascii() else alias):
                    return category
        return "General"

    def extract_items(self, utterances: List[str], speakers: List[str]) -> List[SyncItem]:
        self._last_intents = self._classify_intents(utterances)
        prefs: List[SyncItem] = []

        fallback_count = 0
        confidence_values: List[float] = []
        evidence_spans: List[Dict[str, Any]] = []

        for idx, (utt, sp, intent) in enumerate(zip(utterances, speakers, self._last_intents)):
            confidence_values.append(intent.confidence)
            if intent.method == "regex_fallback":
                fallback_count += 1

            if intent.label in ("preference_positive", "preference_negative"):
                sentiment = "positive" if intent.label == "preference_positive" else "negative"
                prefs.append(
                    SyncItem(
                        text=utt,
                        category=self._categorize(utt),
                        speaker=sp,
                        sentiment=sentiment,
                    )
                )
                evidence_spans.append(
                    {
                        "utterance_index": idx,
                        "speaker": sp,
                        "label": intent.label,
                        "evidence": intent.evidence or utt,
                        "method": intent.method,
                        "confidence": round(intent.confidence, 4),
                    }
                )

        avg_intent_conf = float(np.mean(confidence_values)) if confidence_values else 0.0
        self._last_meta = {
            "confidence": {
                "intent_avg": round(avg_intent_conf, 4),
            },
            "used_fallback": fallback_count > 0,
            "fallback_reasons": ["intent_embedding_unavailable_or_low_confidence"] if fallback_count else [],
            "evidence_spans": evidence_spans,
        }
        return prefs

    def count_agreements(self, utterances: List[str], speakers: List[str]) -> Tuple[float, int]:
        if len(self._last_intents) != len(utterances):
            self._last_intents = self._classify_intents(utterances)

        weighted = 0.0
        agreement_count = 0
        disagreement_count = 0

        for i, (intent, speaker) in enumerate(zip(self._last_intents, speakers)):
            changed = i > 0 and speakers[i - 1] != speaker
            if intent.label == "agreement":
                weighted += (0.8 + 0.5 * intent.confidence) + (0.25 if changed else 0.0)
                agreement_count += 1
            elif intent.label == "disagreement":
                weighted -= 0.4 + 0.35 * intent.confidence
                disagreement_count += 1

        denom = max(len(utterances) * self.agreement_norm_factor, 1.0)
        agreement_score = max(0.0, min(weighted / denom, 1.0))

        confidence_obj = dict(self._last_meta.get("confidence", {}))
        confidence_obj["agreement_signal"] = round(agreement_score, 4)
        self._last_meta["confidence"] = confidence_obj
        self._last_meta["agreement_detail"] = {
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "weighted_signal": round(weighted, 4),
        }
        return agreement_score, agreement_count

    def semantic_similarity(self, prefs_a: List[SyncItem], prefs_b: List[SyncItem]) -> Tuple[float, str]:
        if not prefs_a or not prefs_b:
            return 0.0, "no_preferences"

        use_openai = HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
        if use_openai:
            try:
                emb_a = self._openai_embed([p.text for p in prefs_a])
                emb_b = self._openai_embed([p.text for p in prefs_b])

                total_sim = 0.0
                matched = 0
                used_b = set()

                for i, pa in enumerate(prefs_a):
                    best_sim = -1.0
                    best_j = -1
                    for j, pb in enumerate(prefs_b):
                        if j in used_b or pa.sentiment != pb.sentiment:
                            continue
                        sim = _cosine_sim(emb_a[i], emb_b[j])
                        if sim > best_sim:
                            best_sim = sim
                            best_j = j
                    if best_j >= 0 and best_sim >= self.semantic_match_threshold:
                        total_sim += best_sim
                        matched += 1
                        used_b.add(best_j)

                if matched == 0:
                    sem = 0.0
                else:
                    avg_sim = total_sim / matched
                    coverage = matched / max(len(prefs_a), len(prefs_b))
                    sem = avg_sim * coverage

                confidence_obj = dict(self._last_meta.get("confidence", {}))
                confidence_obj["semantic_coverage"] = round(matched / max(len(prefs_a), len(prefs_b)), 4)
                self._last_meta["confidence"] = confidence_obj
                return sem, "multilingual_embedding"
            except Exception:
                pass

        matched = 0
        used_b = set()
        for pa in prefs_a:
            best_sim = 0.0
            best_j = -1
            for j, pb in enumerate(prefs_b):
                if j in used_b or pa.sentiment != pb.sentiment:
                    continue
                sim = _token_jaccard(pa.text, pb.text)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j >= 0 and best_sim >= 0.2:
                matched += 1
                used_b.add(best_j)

        score = matched / max(len(prefs_a), len(prefs_b))
        return score, "token_fallback"

    def category_sync(self, prefs_a: List[SyncItem], prefs_b: List[SyncItem]) -> Dict[str, float]:
        sync: Dict[str, float] = {}
        categories = set(p.category for p in prefs_a) | set(p.category for p in prefs_b)

        for category in categories:
            a_items = [p for p in prefs_a if p.category == category]
            b_items = [p for p in prefs_b if p.category == category]

            if not a_items or not b_items:
                sync[category] = 0.0
                continue

            volume_ratio = min(len(a_items), len(b_items)) / max(len(a_items), len(b_items))

            a_pos_ratio = sum(1 for p in a_items if p.sentiment == "positive") / len(a_items)
            b_pos_ratio = sum(1 for p in b_items if p.sentiment == "positive") / len(b_items)
            sentiment_align = 1.0 - abs(a_pos_ratio - b_pos_ratio)

            sync[category] = volume_ratio * sentiment_align

        return sync

    def category_detail(
        self,
        prefs_a: List[SyncItem],
        prefs_b: List[SyncItem],
        cat_sync: Dict[str, float],
    ) -> Dict[str, Any]:
        detail: Dict[str, Any] = {}
        categories = set(p.category for p in prefs_a) | set(p.category for p in prefs_b)
        for category in categories:
            detail[category] = {
                "speaker_a": [p.text for p in prefs_a if p.category == category],
                "speaker_b": [p.text for p in prefs_b if p.category == category],
                "sync": round(cat_sync.get(category, 0.0), 4),
            }
        return detail

    def analysis_metadata(
        self,
        utterances: List[str],
        speakers: List[str],
        prefs_a: List[SyncItem],
        prefs_b: List[SyncItem],
        cat_sync: Dict[str, float],
    ) -> Dict[str, Any]:
        meta = dict(self._last_meta)
        confidence_obj = dict(meta.get("confidence", {}))

        pref_balance = min(len(prefs_a), len(prefs_b)) / max(len(prefs_a), len(prefs_b), 1)
        confidence_obj["preference_balance"] = round(pref_balance, 4)

        language_counts = detect_utterance_languages(
            [{"speaker": s, "text": t} for s, t in zip(speakers, utterances)]
        )
        confidence_obj["language_signal"] = round(language_counts.get("avg_confidence", 0.0), 4)

        meta["confidence"] = confidence_obj
        return meta


class PreferenceAnalyzer:
    def __init__(self):
        self._core = ConversationSyncCore(GlobalPreferenceModule())

    def score(self, conversation_obj) -> Dict[str, Any]:
        data = normalize_conversation(conversation_obj)
        conversation = data.get("conversation", [])

        conv_lang = detect_conversation_language(conversation)
        utt_lang = detect_utterance_languages(conversation)

        raw = self._core.analyze(data)
        raw["language_detected"] = {
            "conversation": conv_lang,
            "utterance_level": utt_lang,
        }

        return {
            "scores": {"preference_sync": float(raw.get("score", 0))},
            "raw": raw,
        }
