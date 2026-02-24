"""
Discourse Quality Analyzer (topic_continuity)
==============================================
OpenAI API로 대화의 담화 품질을 평가.

측정 원리:
  1. 대화 텍스트를 gpt-4o-mini에 전달
  2. topic_continuity / logical_flow / collaborative_building 3개 지표 산출
  3. 3개 지표 평균을 최종 점수로 반환

의존성: openai(옵션)
Fallback: API 실패 시 인접 발화 간 토큰 겹침(Jaccard) 기반 휴리스틱
"""

import json
import os
import re
from typing import Dict, Any

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Core
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
            normalized.append(
                {
                    "speaker": getattr(u, "speaker", None),
                    "start": getattr(u, "start", None),
                    "end": getattr(u, "end", None),
                    "text": getattr(u, "text", ""),
                }
            )
    return {"conversation": normalized}


def _format_conversation(data: Dict) -> str:
    lines = []
    for u in data.get("conversation", []):
        speaker = u.get("speaker", "unknown")
        text = u.get("text", "")
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _parse_json_response(response: str) -> Dict:
    match = re.search(r"\{[^{}]*\}", response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"topic_continuity": None, "logical_flow": None, "collaborative_building": None, "parse_error": True}


def _openai_timeout() -> float:
    raw = os.getenv("OPENAI_TIMEOUT", "30")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 30.0


def _analyze_with_openai(data: Dict) -> Dict[str, Any]:
    """OpenAI API로 담화 품질 평가 (3개 지표 평균)"""
    formatted = _format_conversation(data)

    prompt = f"""Analyze the following 2-speaker conversation. It may be in any language.
Use the original utterance language as-is. Do not translate.
Do not infer missing context beyond what is explicitly said.

Conversation:
{formatted}

Evaluate the following items on a continuous 0-100 scale (any integer value is valid):

topic_continuity: How naturally does the topic flow throughout the conversation?
   Higher scores indicate smoother, more coherent topic flow.

logical_flow: How well do causal relationships and logical connections appear in the conversation?
   Higher scores indicate clearer cause-and-effect reasoning and logical progression between utterances.

collaborative_building: To what extent do speakers build upon and develop each other's ideas?
   Higher scores indicate that speakers actively extend, elaborate, or deepen what the other person said.

Return ONLY one JSON object with integer values in [0, 100]:
{{"topic_continuity": X, "logical_flow": X, "collaborative_building": X}}"""

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=_openai_timeout(),
        max_retries=1,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    scores = _parse_json_response(text)

    tc = scores.get("topic_continuity")
    lf = scores.get("logical_flow")
    cb = scores.get("collaborative_building")

    valid = [s for s in [tc, lf, cb] if isinstance(s, (int, float))]
    if not valid:
        return {"score": 0, "parse_error": True, "method": "openai"}

    combined = int(round(sum(valid) / len(valid)))
    return {
        "score": combined,
        "topic_continuity": int(round(tc)) if isinstance(tc, (int, float)) else None,
        "logical_flow": int(round(lf)) if isinstance(lf, (int, float)) else None,
        "collaborative_building": int(round(cb)) if isinstance(cb, (int, float)) else None,
        "method": "openai",
    }


def _analyze_fallback(data: Dict) -> Dict[str, Any]:
    texts = [u["text"] for u in data["conversation"] if u.get("text")]
    if len(texts) < 2:
        return {"score": 0, "method": "fallback"}
    
    def _tokenize(text: str):
        # Unicode letter tokenization for multilingual fallback.
        return set(re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE))

    sims = []
    for i in range(1, len(texts)):
        words_prev = _tokenize(texts[i-1])
        words_curr = _tokenize(texts[i])
        union = words_prev | words_curr
        if union:
            sims.append(len(words_prev & words_curr) / len(union))
    
    avg_sim = sum(sims) / len(sims) if sims else 0
    score = min(100, int(avg_sim * 150))  # 스케일 조정
    return {"score": score, "method": "fallback"}


def analyze(data: Dict) -> Dict[str, Any]:
    speakers = list(dict.fromkeys(u["speaker"] for u in data["conversation"] if u.get("speaker")))
    if len(speakers) < 2:
        return {"score": 0, "error": "need_2_speakers"}

    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            return _analyze_with_openai(data)
        except Exception as e:
            return {**_analyze_fallback(data), "openai_error": str(e)}
    return _analyze_fallback(data)


class DiscourseAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        raw = analyze(data)
        return {
            "scores": {
                "topic_continuity": float(raw.get("score", 0)),
            },
            "raw": raw,
        }
