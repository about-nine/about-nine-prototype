"""
Discourse Quality Analyzer (topic_continuity)
==============================================
OpenAI API로 대화의 주제 흐름 자연스러움을 평가.

측정 원리:
  1. 대화 텍스트를 GPT-4o-mini에 전달
  2. LLM이 주제 연속성을 0~10 스케일로 평가
     - 10: 주제가 자연스럽게 이어짐
     -  5: 주제가 약간 바뀌지만 연결됨
     -  0: 주제가 급격히 바뀜
  3. ×10 하여 100점 만점 환산

의존성: openai
Fallback: API 실패 시 평균 단어 수 기반 휴리스틱
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
    for u in data["conversation"]:
        lines.append(f"{u['speaker']}: {u['text']}")
    return "\n".join(lines)


def _parse_json_response(response: str) -> Dict:
    match = re.search(r"\{[^{}]*\}", response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"topic_continuity": None, "parse_error": True}


def _openai_timeout() -> float:
    raw = os.getenv("OPENAI_TIMEOUT", "30")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 30.0


def _analyze_with_openai(data: Dict) -> Dict[str, Any]:
    """OpenAI API로 주제 연속성 평가"""
    formatted = _format_conversation(data)

    prompt = f"""Analyze the following conversation and evaluate it:

Conversation:
{formatted}

Evaluate the following item on a 0-10 scale:

topic_continuity: Does the topic flow naturally?
   - 10: Perfectly connected, same topic maintained
   - 5: Topic shifts slightly but remains connected
   - 0: Topic changes abruptly throughout

    Respond ONLY in JSON format:
{{"topic_continuity": X}}"""

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=_openai_timeout(),
        max_retries=1,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=128,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    scores = _parse_json_response(text)

    raw_score = scores.get("topic_continuity")
    if raw_score is not None and isinstance(raw_score, (int, float)):
        return {"score": int(round(raw_score * 10)), "raw_0_10": raw_score, "method": "openai"}
    return {"score": 0, "parse_error": True, "method": "openai"}


def _analyze_fallback(data: Dict) -> Dict[str, Any]:
    texts = [u["text"] for u in data["conversation"] if u.get("text")]
    if len(texts) < 2:
        return {"score": 50, "method": "fallback"}
    
    sims = []
    for i in range(1, len(texts)):
        words_prev = set(texts[i-1].lower().split())
        words_curr = set(texts[i].lower().split())
        union = words_prev | words_curr
        if union:
            sims.append(len(words_prev & words_curr) / len(union))
    
    avg_sim = sum(sims) / len(sims) if sims else 0
    score = min(100, int(avg_sim * 150))  # 스케일 조정
    return {"score": score, "method": "fallback"}


def analyze(data: Dict) -> Dict[str, Any]:
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
