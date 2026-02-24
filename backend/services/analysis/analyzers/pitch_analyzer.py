"""
Pitch Similarity Analyzer (voice_pitch)
========================================
두 화자의 음 높이(F0) 특성이 얼마나 비슷한지 측정.

측정 원리 (개별 WAV 파일이 있을 때):
  1. 화자별 wav에서 librosa.pyin으로 F0 추출
  2. VAD(음성 활동 감지)로 무음 구간 제거
  3. 각 화자의 발화 구간 F0 통계(중앙값, 표준편차) 계산
  4. 중앙 피치 유사도: 반음(semitone) 거리 기반 — 1옥타브 차이면 0점
  5. 피치 변동성 유사도: std 비율
  6. 최종 = 0.7 × 중앙피치 + 0.3 × 변동성 → 0~100점

측정 원리 (단일 혼합 오디오일 때):
  1. 전체 오디오에서 F0 추출 + VAD
  2. 발화 세그먼트 단위로 median F0 계산
  3. KMeans(k=2)로 화자 클러스터링 (피치 높낮이 기반)
  4. 두 클러스터의 F0 분포로 동일하게 유사도 계산

환경 변수:
  - PITCH_FMIN, PITCH_FMAX, PITCH_VAD_DB

의존성: librosa, numpy (선택: sklearn)
Fallback: librosa 없으면 0점 반환, 혼합 오디오는 sklearn 없으면 실패 처리
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# Audio utilities
# ============================================================================

def _rms_envelope(y, frame_length, hop_length):
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]


def _vad_mask(rms, thresh_db=-35):
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return rms_db > thresh_db


def _extract_f0(path, sr=16000, hop_ms=10, fmin=None, fmax=None, vad_db=None, max_duration=3600):
    if fmin is None:
        fmin = float(os.getenv("PITCH_FMIN", "65"))
    if fmax is None:
        fmax = float(os.getenv("PITCH_FMAX", "500"))
    if vad_db is None:
        vad_db = float(os.getenv("PITCH_VAD_DB", "-35"))
    info = librosa.get_duration(path=path)
    duration = min(info, max_duration)
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)

    hop_length = int(sr * hop_ms / 1000)
    frame_length = hop_length * 4
    chunk_samples = sr * 30  # 30초 청크

    all_f0, all_vad = [], []

    for start in range(0, len(y), chunk_samples):
        chunk = y[start:start + chunk_samples]
        rms = _rms_envelope(chunk, frame_length, hop_length)
        vad = _vad_mask(rms, thresh_db=vad_db)
        f0, _, _ = librosa.pyin(
            chunk, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length,
        )
        all_f0.append(f0)
        all_vad.append(vad)
        del chunk

    del y
    import gc; gc.collect()

    return {
        "f0": np.concatenate(all_f0),
        "vad": np.concatenate(all_vad),
        "hop_length": hop_length,
        "sr": sr,
    }


# ============================================================================
# Pitch similarity scoring
# ============================================================================

def _pitch_similarity_score(
    f0_a: np.ndarray, voiced_a: np.ndarray,
    f0_b: np.ndarray, voiced_b: np.ndarray,
) -> Dict[str, Any]:
    """
    두 화자의 F0 분포 유사도 측정.

    - 중앙 피치(Hz) 유사도: 반음 거리 기반 (1옥타브=12반음 차이 → 0점)
    - 피치 변동성(std) 유사도: std 비율
    """
    vals_a = f0_a[voiced_a & np.isfinite(f0_a)]
    vals_b = f0_b[voiced_b & np.isfinite(f0_b)]

    if len(vals_a) < 10 or len(vals_b) < 10:
        return {"score": 0, "error": "insufficient_voiced_frames"}

    med_a = float(np.median(vals_a))
    med_b = float(np.median(vals_b))
    std_a = float(np.std(vals_a))
    std_b = float(np.std(vals_b))

    # 1. 중앙 피치 유사도 — 반음 거리(log scale)
    semitone_diff = 12.0 * np.log2(max(med_a, med_b) / min(med_a, med_b))
    MAX_SEMITONES = 12.0  # 1옥타브 차이 → 0점
    median_sim = max(0.0, 1.0 - semitone_diff / MAX_SEMITONES)

    # 2. 피치 변동성 유사도 — std 비율
    max_std = max(std_a, std_b)
    std_sim = min(std_a, std_b) / max_std if max_std > 1.0 else 1.0

    score = 0.7 * median_sim + 0.3 * std_sim

    return {
        "score": int(round(score * 100)),
        "median_hz_a": round(med_a, 2),
        "median_hz_b": round(med_b, 2),
        "semitone_diff": round(float(semitone_diff), 2),
        "std_hz_a": round(std_a, 2),
        "std_hz_b": round(std_b, 2),
    }


# ============================================================================
# 개별 WAV 분석 (Agora individual recording)
# ============================================================================

def _analyze_separate_wavs(wav_paths: Dict[str, str], call_id: str) -> Dict[str, Any]:
    """화자별 개별 WAV 파일이 있을 때"""
    speakers = list(wav_paths.keys())
    if len(speakers) < 2:
        return {"score": 0, "error": "need_2_speakers", "method": "separate_wavs"}

    sp_a, sp_b = speakers[0], speakers[1]

    data_a = _extract_f0(wav_paths[sp_a])
    data_b = _extract_f0(wav_paths[sp_b])

    f0_a, vad_a = data_a["f0"], data_a["vad"]
    f0_b, vad_b = data_b["f0"], data_b["vad"]

    voiced_a = vad_a & np.isfinite(f0_a)
    voiced_b = vad_b & np.isfinite(f0_b)

    result = _pitch_similarity_score(f0_a, voiced_a, f0_b, voiced_b)

    del f0_a, f0_b, vad_a, vad_b, voiced_a, voiced_b
    import gc; gc.collect()

    return {**result, "method": "separate_wavs"}


# ============================================================================
# 단일 혼합 오디오 분석 (KMeans clustering)
# ============================================================================

def _segments_from_mask(mask, min_len=10):
    segs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segs.append((start, len(mask)))
    return segs


def _analyze_mixed_audio(audio_path: str, call_id: str) -> Dict[str, Any]:
    """단일 오디오에서 KMeans로 화자 분리 후 F0 유사도 분석"""
    if not HAS_SKLEARN:
        return {"score": 0, "error": "sklearn_not_available", "method": "mixed_audio"}

    data = _extract_f0(audio_path)
    f0, vad = data["f0"], data["vad"]

    voiced = vad & np.isfinite(f0)
    segs = _segments_from_mask(voiced, min_len=12)

    seg_stats = []
    for s, e in segs:
        vals = f0[s:e]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        seg_stats.append({"start": s, "end": e, "median_f0": float(np.median(vals))})

    if len(seg_stats) < 2:
        return {"score": 0, "error": "too_few_segments", "method": "mixed_audio"}

    X = np.array([np.log(s["median_f0"] + 1e-9) for s in seg_stats]).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    c0, c1 = np.mean(X[labels == 0]), np.mean(X[labels == 1])
    low_label = 0 if c0 < c1 else 1
    cluster_gap = abs(float(c0 - c1))

    n_frames = len(f0)
    mask_low = np.zeros(n_frames, dtype=bool)
    mask_high = np.zeros(n_frames, dtype=bool)

    for i, s in enumerate(seg_stats):
        target = mask_low if labels[i] == low_label else mask_high
        target[s["start"]:s["end"]] = True

    result = _pitch_similarity_score(f0, mask_low, f0, mask_high)
    if cluster_gap < 0.08:
        result["warning"] = "weak_cluster_separation"
    result["cluster_gap_log_hz"] = round(cluster_gap, 4)

    return {**result, "method": "mixed_audio_kmeans"}


# ============================================================================
# Public interface
# ============================================================================

class PitchAnalyzer:
    def score(
        self,
        wav_paths_by_speaker: Optional[Dict[str, str]] = None,
        wav_paths: Optional[List[str]] = None,
        call_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not HAS_LIBROSA:
            return {
                "scores": {"voice_pitch": 0.0},
                "raw": {"score": 0, "error": "librosa_not_installed", "method": "default"},
            }

        # Case 1: 화자별 개별 WAV (최적)
        if isinstance(wav_paths_by_speaker, dict) and len(wav_paths_by_speaker) >= 2:
            valid = {k: v for k, v in wav_paths_by_speaker.items() if v and os.path.exists(v)}
            if len(valid) >= 2:
                try:
                    raw = _analyze_separate_wavs(valid, call_id or "unknown")
                    return {"scores": {"voice_pitch": float(raw["score"])}, "raw": raw}
                except Exception as e:
                    return {
                        "scores": {"voice_pitch": 0.0},
                        "raw": {"score": 0, "error": str(e), "method": "separate_wavs_failed"},
                    }

        # Case 2: WAV 파일 리스트 (2개면 개별, 1개면 혼합)
        if isinstance(wav_paths, list):
            valid_paths = [p for p in wav_paths if p and os.path.exists(p)]

            if len(valid_paths) >= 2:
                speaker_map = {f"speaker_{i}": p for i, p in enumerate(valid_paths[:2])}
                try:
                    raw = _analyze_separate_wavs(speaker_map, call_id or "unknown")
                    return {"scores": {"voice_pitch": float(raw["score"])}, "raw": raw}
                except Exception as e:
                    return {
                        "scores": {"voice_pitch": 0.0},
                        "raw": {"score": 0, "error": str(e), "method": "separate_wavs_failed_list"},
                    }

            if len(valid_paths) == 1:
                try:
                    raw = _analyze_mixed_audio(valid_paths[0], call_id or "unknown")
                    return {"scores": {"voice_pitch": float(raw["score"])}, "raw": raw}
                except Exception as e:
                    return {
                        "scores": {"voice_pitch": 0.0},
                        "raw": {"score": 0, "error": str(e), "method": "mixed_audio_failed"},
                    }

        return {
            "scores": {"voice_pitch": 0.0},
            "raw": {"score": 0, "error": "no_valid_audio", "method": "default"},
        }
