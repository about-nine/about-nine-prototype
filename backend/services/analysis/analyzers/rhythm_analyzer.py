"""
Turn-Taking Analyzer (rhythm_synchrony)
=========================================
대화 타이밍 데이터에서 의미 있는 3가지 지표를 추출:

1. Balance (40%) — 발화 시간 균형
   두 화자의 총 발화 시간 비율. 50:50에 가까울수록 높은 점수.
   한쪽만 80% 말하면 낮은 점수.

2. Silence (30%) — 침묵 패턴
   긴 침묵(>5초)의 비율. 어색한 긴 침묵이 많으면 낮은 점수.
   자연스러운 짧은 pause는 감점 없음.

3. Engagement (30%) — 참여도 균형
   응답 길이의 균형. 한쪽이 길게 말하고 상대가 "어"로만 응답하면 낮은 점수.
   서로 비슷한 길이로 응답하면 높은 점수.

최종: turn_taking = 0.4×balance + 0.3×silence + 0.3×engagement → 0~100

의존성: numpy
"""

import numpy as np
from typing import Dict, Any, List, Tuple


# ============================================================================
# Helpers
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


def _get_speakers(conversation: List[Dict]) -> Tuple[str, str]:
    """첫 두 화자 추출"""
    speakers = list(dict.fromkeys(u["speaker"] for u in conversation if u.get("speaker")))
    if len(speakers) < 2:
        return (speakers[0] if speakers else "A"), "B"
    return speakers[0], speakers[1]


# ============================================================================
# 1. Balance — 발화 시간 균형
# ============================================================================

def _calc_balance(conversation: List[Dict], speaker_a: str, speaker_b: str) -> Dict:
    """
    각 화자의 총 발화 시간 비율 → 균형 점수

    50:50 → 100점
    80:20 → 낮은 점수
    100:0 → 0점
    """
    time_a = 0.0
    time_b = 0.0

    for u in conversation:
        if u.get("start") is None or u.get("end") is None:
            continue
        dur = max(0, u["end"] - u["start"])
        if u["speaker"] == speaker_a:
            time_a += dur
        elif u["speaker"] == speaker_b:
            time_b += dur

    total = time_a + time_b
    if total < 1.0:  # 거의 대화 없음
        return {"score": 0, "time_a": 0, "time_b": 0, "ratio": 0, "error": "no_speech"}

    # 비율: 0.5 = 완벽한 균형, 0 또는 1 = 한쪽만
    ratio = time_a / total  # 0~1
    # 0.5에서 얼마나 벗어났는지 → 0~0.5 범위
    deviation = abs(ratio - 0.5)
    # 0~100 변환: deviation 0 → 100, deviation 0.5 → 0
    score = max(0, (1.0 - (deviation * 1.5) ** 1.5)) * 100

    return {
        "score": round(score, 1),
        "time_a": round(time_a, 1),
        "time_b": round(time_b, 1),
        "ratio_a": round(ratio, 3),
    }


# ============================================================================
# 2. Silence — 침묵 패턴
# ============================================================================

def _calc_silence(conversation: List[Dict]) -> Dict:
    """
    대화 사이 침묵 분석

    긴 침묵(>5초): 어색함 신호 → 많으면 감점
    짧은 pause(<2초): 자연스러움 → 감점 없음
    중간(2~5초): 약간 감점
    """
    if len(conversation) < 2:
        return {"score": 50, "gaps": [], "error": "not_enough_utterances"}

    gaps = []

    # 시간순 정렬된 발화에서 gap 추출
    sorted_conv = sorted(conversation, key=lambda x: x.get("start", 0))

    for i in range(1, len(sorted_conv)):
        prev_end = sorted_conv[i - 1].get("end")
        cur_start = sorted_conv[i].get("start")
        if prev_end is None or cur_start is None:
            continue

        # 같은 화자 연속이면 건너뜀 (화자 전환 gap만 측정)
        if sorted_conv[i]["speaker"] == sorted_conv[i - 1]["speaker"]:
            continue

        gap = cur_start - prev_end
        if gap > 0.1:  # 100ms 이상만
            gaps.append(gap)

    if not gaps:
        return {"score": 70, "total_gaps": 0, "long_silence_count": 0}

    long_silences = [g for g in gaps if g > 5.0]      # 5초 초과
    medium_silences = [g for g in gaps if 2.0 < g <= 5.0]  # 2~5초

    total_gaps = len(gaps)
    long_ratio = len(long_silences) / total_gaps if total_gaps > 0 else 0
    medium_ratio = len(medium_silences) / total_gaps if total_gaps > 0 else 0

    # 점수: 긴 침묵 비율에 따라 감점
    # long_ratio 0 → 100, long_ratio 0.3+ → 낮은 점수
    score = 100 * (1.0 - long_ratio * 2.0 - medium_ratio * 0.5)
    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "total_gaps": total_gaps,
        "long_silence_count": len(long_silences),
        "medium_silence_count": len(medium_silences),
        "avg_gap_sec": round(float(np.mean(gaps)), 2) if gaps else 0,
        "max_gap_sec": round(max(gaps), 2) if gaps else 0,
    }


# ============================================================================
# 3. Engagement — 참여도 균형
# ============================================================================

def _calc_engagement(conversation: List[Dict], speaker_a: str, speaker_b: str) -> Dict:
    """
    응답 길이(글자 수) 균형 측정

    한쪽이 길게 말하고 상대가 "어"만 하면 → 낮은 점수
    서로 비슷한 길이로 주고받으면 → 높은 점수
    """
    lens_a = []
    lens_b = []

    for u in conversation:
        text = u.get("text", "").strip()
        if not text:
            continue
        text_len = len(text)
        if u["speaker"] == speaker_a:
            lens_a.append(text_len)
        elif u["speaker"] == speaker_b:
            lens_b.append(text_len)

    if not lens_a or not lens_b:
        return {"score": 0, "error": "single_speaker"}

    avg_a = np.mean(lens_a)
    avg_b = np.mean(lens_b)

    # 평균 응답 길이 비율
    max_avg = max(avg_a, avg_b)
    min_avg = min(avg_a, avg_b)

    if max_avg < 1:
        return {"score": 50, "avg_len_a": 0, "avg_len_b": 0}

    # 비율: 1.0 = 동일, 0 = 극단적 차이
    length_ratio = min_avg / max_avg

    # 발화 횟수 균형도 반영
    count_a = len(lens_a)
    count_b = len(lens_b)
    max_count = max(count_a, count_b)
    min_count = min(count_a, count_b)
    count_ratio = min_count / max_count if max_count > 0 else 0

    # 가중 합산: 길이 균형 60% + 횟수 균형 40%
    score = (length_ratio * 0.6 + count_ratio * 0.4) * 100

    return {
        "score": round(score, 1),
        "avg_len_a": round(float(avg_a), 1),
        "avg_len_b": round(float(avg_b), 1),
        "count_a": count_a,
        "count_b": count_b,
        "length_ratio": round(length_ratio, 3),
        "count_ratio": round(count_ratio, 3),
    }


# ============================================================================
# Main analyzer
# ============================================================================

W_BALANCE = 0.4
W_SILENCE = 0.3
W_ENGAGEMENT = 0.3


def analyze(conversation_data: Dict) -> Dict[str, Any]:
    conv = conversation_data.get("conversation", [])

    if len(conv) < 3:
        return {
            "score": 0,
            "error": "not_enough_utterances",
            "balance": {},
            "silence": {},
            "engagement": {},
        }

    speaker_a, speaker_b = _get_speakers(conv)

    balance = _calc_balance(conv, speaker_a, speaker_b)
    silence = _calc_silence(conv)
    engagement = _calc_engagement(conv, speaker_a, speaker_b)

    # 가중 합산
    score = (
        W_BALANCE * balance["score"]
        + W_SILENCE * silence["score"]
        + W_ENGAGEMENT * engagement["score"]
    )
    score = max(0, min(100, round(score, 1)))

    return {
        "score": score,
        "balance": balance,
        "silence": silence,
        "engagement": engagement,
        "weights": {
            "balance": W_BALANCE,
            "silence": W_SILENCE,
            "engagement": W_ENGAGEMENT,
        },
    }


class RhythmAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        raw = analyze(data)
        return {
            "scores": {
                "rhythm_synchrony": float(raw.get("score", 0)),
            },
            "raw": raw,
        }