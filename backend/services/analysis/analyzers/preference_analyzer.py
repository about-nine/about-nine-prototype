# services/analysis/analyzers/preference_analyzer.py - preference alignment analyzer
"""
Preference Sync Analyzer (preference_sync)
===========================================
두 화자의 취향 일치도를 측정하는 임베딩-우선 분석기.

측정 원리:
  1. 의도 분류: OpenAI 임베딩 + 프로토타입 비교
     - fallback: 정규식 패턴(preference/agree/disagree)
  2. 취향 추출: preference_positive/negative 의도만 취향 항목으로 수집
  3. 명시적 동의 점수: agreement/disagreement + 화자 교체 가중치
  4. 의미 유사도: 취향 발화를 임베딩 코사인 매칭
     - fallback: 조사 제거 기반 토큰 자카드(한국어)
  5. 카테고리 동기화: 카테고리 분포 + 긍정/부정 비율 정렬
  6. 최종 점수 = w_semantic + w_agreement + w_category (기본 0.4/0.3/0.3)

의존성: numpy, openai(옵션)
환경 변수: PREFERENCE_WEIGHTS, PREFERENCE_INTENT_MIN_CONF,
          PREFERENCE_SEMANTIC_MATCH_THRESHOLD, PREFERENCE_AGREEMENT_NORM
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.services.analysis.analyzers.language_utils import detect_conversation_language, detect_utterance_languages
from backend.services.analysis.analyzers.analyzer_core import (
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
        "I absolutely love this",
        "That's my favorite",
        "I always watch this",
        "저는 이걸 정말 좋아해요",
        "이게 제 취향이에요",
        "저 이거 진짜 좋아해요",
        "매일 먹는 편이에요",
        "취미가 이거예요",
        "Me encanta esto",
        "Esto es mi favorito",
        "J'aime beaucoup ca",
    ],
    "preference_negative": [
        "I do not like this",
        "I hate this",
        "This is not my thing",
        "I can't stand this",
        "Not really my thing",
        "저는 이거 별로예요",
        "이건 싫어요",
        "안 좋아해요",
        "못 먹어요",
        "No me gusta esto",
        "Je n'aime pas ca",
    ],
    "agreement": [
        "I agree with you",
        "Me too",
        "Same here",
        "I love that too",
        "That's so true",
        "I feel the same",
        "저도요",
        "나도 그래",
        "저도 완전 공감해요",
        "Estoy de acuerdo",
        "Yo tambien",
        "Moi aussi",
    ],
    "disagreement": [
        "I disagree",
        "Not really",
        "I don't think so",
        "No way, not really",
        "저는 좀 달라요",
        "저는 동의하지 않아요",
        "그건 좀 아닌데요",
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
        r"\bi\s+(really\s+|absolutely\s+|just\s+)?(love|like|enjoy|prefer|adore)\b",
        r"\b(?:my|that's my)\s+favorite\b",
        r"\b(i\s+am|i'm)\s+(a\s+fan\s+of|into|fond\s+of)\b",
        r"\bi\s+(?:always|usually|often)\s+(?:eat|drink|watch|listen|go)\b",
        r"nothing\s+beats",
        r"the\s+best\s+(?:thing|part)",
        r"(?:that's|it's|so)\s+(?:great|awesome|amazing|the\s+best)",
        r"\bi\s+(?:go|do|have)\s+(?:that|this|it)?\s*(?:all\s+the\s+time|every\s?(?:day|week))",
        r"(?:oh\s+)?(?:yeah|yes),?\s*i\s+(?:know|love)\s+(?:that|this)",
        r"(?:have\s+you\s+(?:tried|been\s+to)|you\s+should\s+try)",
        r"좋아해|좋아요|좋아하|즐겨|취향|선호|맛있|재미있|최고|강추",
        r"(?:정말|진짜|엄청|너무|완전|되게)?\s*좋아(?:해|하는|했|하고|합니다|해요|하거든|한다|했어|해서|할)",
        r"제일\s*좋아",
        r"가장\s*좋아",
        r"좋아(?:요|하죠|하지)",
        r"즐겨\s*(?:먹|마시|봐|보|듣|하|읽|가|찾|즐기)",
        r"즐기(?:는|고|며|어|ㄴ다|고\s*있|는\s*편)",
        r"즐겨(?:요)?$",
        r"(?:운동|독서|요리|산책|게임|여행|등산|낚시|영화|음악|드라마)(?:을|를)?\s*즐겨",
        r"취미(?:예요|입니다|야|이에요|가|로)",
        r"취미(?:가|로)\s*(?:있어|삼아)",
        r"(?:매일|자주|항상)\s*(?:먹|마시|보|듣|하|가)",
        r"매일\s*(?:먹|마시|봐|보|들어|해|가|마셔|읽어|써)",
        r"자주\s*(?:먹|마시|봐|보|들어|해|가|읽|가요|해요)",
        r"항상\s*(?:먹|마시|봐|보|들어|해|가|즐겨)",
        r"아침마다|저녁마다|주말마다",
        r"(?:매주|매달|매년)\s*(?:가|해|봐|먹)",
        r"(?:운동|독서|요리|산책|게임|여행)(?:을|를)?\s*즐겨",
        r"맛있(?:어|었|다|는|더라|어요|네요|었어요)",
        r"재미있(?:어|었|다|는|더라|어요|네요)",
        r"좋더라(?:고요?)?",
        r"최고(?:예요|다|야|네요|였어)",
        r"대박이(?:에요|야|다)",
        r"짱이(?:에요|야)",
        r"선호(?:해요|합니다|하는|했어)",
        r"추천해(?:요|드려|드릴)",
        r"강추",
        r"꼭\s*(?:먹어봐|가봐|해봐)",
        r"한번\s*(?:먹어봐|가봐|해봐)",
        r"좋아하는\s*(?:편|것|거)",
        r"빠져\s*(?:있어|있는|있었|있었어)",
        r"푹\s*빠져",
        r"열심히\s*(?:하고|다니고|보고)",
        r"(?:한|두)\s*번도\s*빠짐없이",
        r"me\s+encanta|me\s+gusta|mi\s+favorito",
        r"j'?adore|j'aime\s+beaucoup|mon\s+pr[ée]f[ée]r[ée]",
        r"ich\s+(?:mag|liebe)\s+das|mein\s+favorit",
        r"eu\s+(?:gosto|adoro)|meu\s+favorito",
        r"j'aime|prefere",
    ],
    "preference_negative": [
        r"\bi\s+(do\s+not|don't|never)\s+(like|enjoy|prefer|eat|watch)\b",
        r"\b(i\s+)?(hate|dislike|can't\s+stand)\b",
        r"not\s+(?:a\s+fan\s+of|into|my\s+thing)",
        r"싫어|별로|안\s*좋아|못\s*먹|귀찮|힘들어",
        r"싫어(?:요|해|하는|했|합니다)?",
        r"싫(?:은|다|고)",
        r"별로(?:야|예요|더라고요?|에요|인\s*것\s*같아|안\s*좋아)?",
        r"그닥\s*(?:좋지|안)",
        r"별로\s*안\s*좋아",
        r"못\s*(?:먹|마시|봐|들어|해|가|읽)",
        r"잘\s*안\s*(?:먹|마시|보|듣|하)|안\s*(?:맞아|맞는)",
        r"안\s*좋아(?:해|하는|했|합니다|해요)?",
        r"안\s*(?:먹|마시|봐|들어|해)",
        r"귀찮(?:아|은|은데|아서|아요)",
        r"어렵(?:고|어|다|더라)",
        r"힘들어서\s*(?:못|안)",
        r"못\s*(?:먹겠|마시겠|보겠|참겠)",
        r"안\s*(?:맞아|맞는)",
        r"no\s+me\s+gusta|odio",
        r"je\s+n'aime\s+pas|je\s+d[ée]teste",
        r"ich\s+mag\s+.*nicht|ich\s+hasse",
        r"n[aã]o\s+gosto|odeio",
        r"je\s+n'aime\s+pas",
    ],
    "agreement": [
        r"\bme\s+too\b|\bsame\s+here\b|\bi\s+agree\b|\bexactly\b",
        r"\bi\s+(?:also|too)\s+(?:love|like|enjoy)\b",
        r"(?:oh\s+)?i\s+love\s+that\s+too",
        r"(?:that's\s+)?so\s+true",
        r"i\s+feel\s+the\s+same",
        r"(?:no\s+way,?\s+)?me\s+too",
        r"저도요|나도|맞아요|맞아|그러게요|동감|공감|저도\s*그래요|나도\s*그래",
        r"저도\s*(?:정말|진짜|완전|너무)?\s*(?:좋아해요|좋아해|좋아하거든|즐겨요)",
        r"나도\s*(?:정말|진짜|완전|너무)?\s*(?:좋아해|좋아하거든|즐겨)",
        r"저도\s*(?:그래요|마찬가지예요|똑같아요|그렇게\s*생각해요)",
        r"나도\s*(?:그래|마찬가지야|똑같아|그렇게\s*생각해)",
        r"완전\s*(?:공감|동의|맞아)",
        r"(?:오|아|어|와)\s*저도",
        r"저도요",
        r"나도요?!?",
        r"(?:진짜요?\s*)?공감이에요",
        r"동감이에요",
        r"맞아요",
        r"그러게요",
        r"맞아(?:요)?",
        r"그러게(?:요)?",
        r"그렇죠",
        r"그러네요",
        r"(?:완전|진짜)\s*맞아",
        r"저도\s*(?:요|)",
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
        "sushi", "taste", "menu", "meal", "flavor",
        "음식", "먹", "요리", "맛집", "카페", "커피", "식당", "레스토랑", "디저트", "빵", "맥주", "와인",
        "식사", "맛", "반찬", "국", "밥", "면", "고기", "해산물", "채소", "과일",
        "차", "술", "소주", "막걸리", "초밥", "라면", "파스타", "피자", "치킨", "삼겹살", "곱창",
        "한식", "일식", "양식", "중식", "분식",
        "comida", "cocina", "restaurante", "cafe", "cafeteria", "postre",
        "nourriture", "restaurant", "cuisine",
        "essen", "restaurant", "küche", "kuche",
        "comida", "restaurante", "cozinha",
    ],
    "Music & Entertainment": [
        "music", "song", "movie", "show", "concert", "game", "watch", "listen",
        "음악", "노래", "영화", "드라마", "공연", "게임", "뮤지컬", "콘서트",
        "아이돌", "밴드", "힙합", "재즈", "클래식", "팝", "인디",
        "넷플릭스", "유튜브", "스트리밍", "OST", "플레이리스트",
        "웹툰", "만화", "애니", "책", "소설", "웹소설",
        "musica", "pelicula", "serie", "musique", "film",
    ],
    "Travel": [
        "travel", "trip", "visit", "city", "country", "vacation", "flight", "hotel",
        "여행", "도시", "나라", "해외", "국내", "관광", "캠핑",
        "바다", "산", "제주", "부산", "강원", "경주", "전주",
        "강", "호수", "숙소", "호텔", "펜션", "글램핑",
        "배낭여행", "패키지여행", "자유여행",
        "viaje", "ciudad", "pais", "vacaciones", "voyage",
    ],
    "Lifestyle": [
        "routine", "habit", "morning", "weekend", "daily", "sleep", "everyday", "always",
        "루틴", "습관", "일상", "아침", "주말", "수면",
        "아침마다", "아침에 일어", "아침 루틴", "매일 아침",
        "산책하", "명상", "수면 패턴", "저녁 루틴", "주말 루틴", "생활 패턴",
        "rutina", "habito", "quotidien",
    ],
    "Hobbies": [
        "hobby", "exercise", "reading", "writing", "sports", "drawing",
        "sport", "game", "read", "write",
        "취미", "운동", "독서", "글쓰기", "등산", "요가", "수영", "달리기", "게임", "사진",
        "그림", "영상", "자전거", "낚시", "골프", "테니스", "배드민턴", "볼링", "클라이밍",
        "뜨개질", "공예", "악기", "기타 치", "피아노 치", "동호회", "동아리",
        "pasatiempo", "deporte", "lecture", "aficion",
        "loisir", "sport", "lecture",
        "hobby", "sport", "lesen",
        "hobby", "esporte", "leitura",
    ],
    "Values": [
        "value", "belief", "important", "meaning", "meaningful", "future", "goal",
        "believe", "think", "feel",
        "가치", "신념", "중요", "의미", "미래", "목표",
        "생각해", "믿어", "중요한", "의미 있", "인생", "행복", "꿈", "계획", "철학",
        "살아가", "살면서",
        "valor", "creencia", "objectif",
    ],
    "Relationships": [
        "friend", "family", "partner", "people", "together", "relationship",
        "친구", "가족", "연인", "사람", "사람들", "함께", "같이", "관계",
        "부모님", "형제", "자매", "반려동물", "반려견", "반려묘",
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
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
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

    def semantic_similarity(self, items_a: List[SyncItem], items_b: List[SyncItem]) -> Tuple[float, str]:
        if not items_a or not items_b:
            return 0.0, "no_preferences"

        use_openai = HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
        if use_openai:
            try:
                emb_a = self._openai_embed([p.text for p in items_a])
                emb_b = self._openai_embed([p.text for p in items_b])

                total_sim = 0.0
                matched = 0
                used_b = set()

                for i, pa in enumerate(items_a):
                    best_sim = -1.0
                    best_j = -1
                    for j, pb in enumerate(items_b):
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
                    coverage = matched / max(len(items_a), len(items_b))
                    sem = avg_sim * coverage

                confidence_obj = dict(self._last_meta.get("confidence", {}))
                confidence_obj["semantic_coverage"] = round(matched / max(len(items_a), len(items_b)), 4)
                self._last_meta["confidence"] = confidence_obj
                return sem, "multilingual_embedding"
            except Exception:
                pass

        matched = 0
        used_b = set()
        for pa in items_a:
            best_sim = 0.0
            best_j = -1
            for j, pb in enumerate(items_b):
                if j in used_b or pa.sentiment != pb.sentiment:
                    continue
                sim = _token_jaccard(pa.text, pb.text)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j >= 0 and best_sim >= 0.2:
                matched += 1
                used_b.add(best_j)

        score = matched / max(len(items_a), len(items_b))
        return score, "token_fallback"

    def category_sync(self, items_a: List[SyncItem], items_b: List[SyncItem]) -> Dict[str, float]:
        sync: Dict[str, float] = {}
        categories = set(p.category for p in items_a) | set(p.category for p in items_b)

        for category in categories:
            a_items = [p for p in items_a if p.category == category]
            b_items = [p for p in items_b if p.category == category]

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
        items_a: List[SyncItem],
        items_b: List[SyncItem],
        cat_sync: Dict[str, float],
    ) -> Dict[str, Any]:
        detail: Dict[str, Any] = {}
        categories = set(p.category for p in items_a) | set(p.category for p in items_b)
        for category in categories:
            detail[category] = {
                "speaker_a": [p.text for p in items_a if p.category == category],
                "speaker_b": [p.text for p in items_b if p.category == category],
                "sync": round(cat_sync.get(category, 0.0), 4),
            }
        return detail

    def analysis_metadata(
        self,
        utterances: List[str],
        speakers: List[str],
        items_a: List[SyncItem],
        items_b: List[SyncItem],
        cat_sync: Dict[str, float],
    ) -> Dict[str, Any]:
        meta = dict(self._last_meta)
        confidence_obj = dict(meta.get("confidence", {}))

        pref_balance = min(len(items_a), len(items_b)) / max(len(items_a), len(items_b), 1)
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
