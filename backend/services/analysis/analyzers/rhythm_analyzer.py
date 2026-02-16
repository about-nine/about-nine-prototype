"""
Turn-Taking Rhythm Analyzer (rhythm_synchrony)
================================================
두 화자의 응답 시간 패턴을 DTW로 비교하여 리듬 동기화 측정.
응답이 빠르고 일정할수록, 그리고 두 사람의 패턴이 유사할수록 높은 점수.

측정 원리:
  1. 연속 발화 사이의 응답 시간(ms) 추출 (화자별 분리)
  2. 두 화자의 응답 시간 시퀀스를 DTW distance로 비교
  3. distance가 작을수록 리듬이 맞음 → 1/(1 + d/100)으로 0~1 변환 → 100점 만점

의존성: dtaidistance (선택) / 없으면 pure Python DTW fallback
"""

import numpy as np
from typing import Dict, Any, List, Tuple

# dtaidistance는 C 빌드 문제가 있을 수 있어 fallback 제공
try:
    from dtaidistance import dtw as dtw_lib
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False


# ============================================================================
# Pure Python DTW fallback
# ============================================================================

def _pure_python_dtw(a: List[float], b: List[float]) -> float:
    n, m = len(a), len(b)
    matrix = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    matrix[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            matrix[i][j] = cost + min(
                matrix[i - 1][j],
                matrix[i][j - 1],
                matrix[i - 1][j - 1],
            )
    return matrix[n][m]


def dtw_distance(a, b) -> float:
    if HAS_DTAIDISTANCE:
        return float(dtw_lib.distance(a, b))
    return _pure_python_dtw(list(a), list(b))


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
            normalized.append({
                "speaker": getattr(u, "speaker", None),
                "start": getattr(u, "start", None),
                "end": getattr(u, "end", None),
                "text": getattr(u, "text", ""),
            })
    return {"conversation": normalized}


def extract_response_patterns(
    conversation: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """화자별 응답 시간(ms) 시퀀스 추출"""
    speakers = list(dict.fromkeys(u["speaker"] for u in conversation))
    if len(speakers) < 2:
        return np.array([]), np.array([])

    speaker_a, speaker_b = speakers[0], speakers[1]
    times_a: List[float] = []
    times_b: List[float] = []

    for i in range(1, len(conversation)):
        prev = conversation[i - 1]
        cur = conversation[i]
        if prev["end"] is None or cur["start"] is None:
            continue
        rt = max(0.0, (cur["start"] - prev["end"]) * 1000)  # ms

        if cur["speaker"] == speaker_a:
            times_a.append(rt)
        else:
            times_b.append(rt)

    return np.array(times_a), np.array(times_b)


def analyze(conversation_data: Dict) -> Dict[str, Any]:
    conv = conversation_data["conversation"]
    a_times, b_times = extract_response_patterns(conv)

    if len(a_times) < 2 or len(b_times) < 2:
        return {
            "score": 0,
            "dtw_distance": 0,
            "user_a_response_times_ms": a_times.tolist(),
            "user_b_response_times_ms": b_times.tolist(),
            "error": "not_enough_turns",
        }

    dist = dtw_distance(a_times, b_times)
    synchrony = 1.0 / (1.0 + dist / 100.0)
    score = int(round(synchrony * 100))

    return {
        "score": score,
        "synchrony_raw": round(synchrony, 4),
        "dtw_distance": round(dist, 2),
        "user_a_response_times_ms": [round(t, 2) for t in a_times.tolist()],
        "user_b_response_times_ms": [round(t, 2) for t in b_times.tolist()],
        "user_a_avg_ms": round(float(np.mean(a_times)), 2),
        "user_b_avg_ms": round(float(np.mean(b_times)), 2),
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