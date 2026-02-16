"""
Pitch Synchrony Analyzer (voice_pitch)
=======================================
두 화자의 피치(F0) 패턴이 얼마나 동조하는지 측정.

측정 원리 (개별 WAV 파일이 있을 때):
  1. 화자별 wav에서 librosa.pyin으로 F0 추출
  2. VAD(음성 활동 감지)로 무음 구간 제거
  3. F0를 보간 + robust z-score 정규화
  4. 두 화자의 정규화된 F0 시퀀스 간 weighted cross-correlation 계산
  5. 최적 lag에서의 상관계수 → (corr+1)/2 × 100으로 0~100점 환산

측정 원리 (단일 혼합 오디오일 때):
  1. 전체 오디오에서 F0 추출 + VAD
  2. 발화 세그먼트 단위로 median F0 계산
  3. KMeans(k=2)로 화자 클러스터링 (피치 높낮이 기반)
  4. 이후 동일하게 cross-correlation

의존성: librosa, numpy, scipy (선택: sklearn)
Fallback: librosa 없으면 기본값 50점 반환
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.signal import correlate

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


def _interpolate_nans(x):
    x = x.astype(float)
    idx = np.arange(len(x))
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.zeros_like(x)
    return np.interp(idx, idx[m], x[m])


def _robust_z(x, mask=None):
    x = x.astype(float)
    if mask is None:
        mask = np.isfinite(x)
    vals = x[mask]
    if len(vals) < 10:
        return np.zeros_like(x)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-9
    z = (x - med) / (1.4826 * mad)
    z[~np.isfinite(z)] = 0.0
    return np.clip(z, -5, 5)


def _extract_f0(path, sr=16000, hop_ms=10, fmin=65, fmax=400, vad_db=-35):
    """WAV 파일에서 F0 + VAD 마스크 추출"""
    y, file_sr = librosa.load(path, sr=None, mono=True)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    hop_length = int(sr * hop_ms / 1000)
    frame_length = hop_length * 4

    rms = _rms_envelope(y, frame_length, hop_length)
    vad = _vad_mask(rms, thresh_db=vad_db)

    f0, _, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )

    f0_clean = f0.copy()
    f0_clean[~vad] = np.nan

    return {
        "f0": f0_clean,
        "vad": vad,
        "hop_length": hop_length,
        "sr": sr,
    }


def _best_lag_sync(z_a, z_b, mask_a, mask_b, max_lag_frames=200):
    L = min(len(z_a), len(z_b))
    a = z_a[:L] * mask_a[:L]
    b = z_b[:L] * mask_b[:L]
    
    corr = correlate(a, b, mode='full')
    # normalize
    norm = np.sqrt(correlate(a*a, np.ones_like(b)) * correlate(np.ones_like(a), b*b))
    norm[norm < 1e-8] = 1e-8
    corr_norm = corr / norm
    
    center = L - 1
    valid = corr_norm[center - max_lag_frames:center + max_lag_frames + 1]
    best_idx = np.argmax(valid)
    best_lag = best_idx - max_lag_frames
    best_corr = float(valid[best_idx])
    
    return best_corr, best_lag


def _corr_to_score(corr: float) -> int:
    return int(round(np.clip((corr + 1) / 2, 0, 1) * 100))


# ============================================================================
# 개별 WAV 분석 (Agora individual recording)
# ============================================================================


def _analyze_separate_wavs(wav_paths: Dict[str, str], call_id: str) -> Dict[str, Any]:
    """화자별 개별 WAV 파일이 있을 때"""
    speakers = list(wav_paths.keys())
    if len(speakers) < 2:
        return {"score": 50, "error": "need_2_speakers", "method": "separate_wavs"}

    sp_a, sp_b = speakers[0], speakers[1]

    data_a = _extract_f0(wav_paths[sp_a])
    data_b = _extract_f0(wav_paths[sp_b])

    f0_a, vad_a = data_a["f0"], data_a["vad"]
    f0_b, vad_b = data_b["f0"], data_b["vad"]
    hop_sec = data_a["hop_length"] / data_a["sr"]

    # 정규화
    voiced_a = vad_a & np.isfinite(f0_a)
    voiced_b = vad_b & np.isfinite(f0_b)

    f0_a_i = _interpolate_nans(f0_a)
    f0_b_i = _interpolate_nans(f0_b)
    z_a = _robust_z(f0_a_i, mask=voiced_a)
    z_b = _robust_z(f0_b_i, mask=voiced_b)

    max_lag = int(2.0 / hop_sec)  # 2초
    corr, lag = _best_lag_sync(z_a, z_b, voiced_a, voiced_b, max_lag_frames=max_lag)
    score = _corr_to_score(corr)

    # Median F0 per speaker
    f0_a_voiced = f0_a[np.isfinite(f0_a)]
    f0_b_voiced = f0_b[np.isfinite(f0_b)]

    return {
        "score": score,
        "best_corr": round(corr, 4),
        "best_lag_frames": lag,
        "best_lag_seconds": round(lag * hop_sec, 4),
        "speaker_a_median_hz": round(float(np.median(f0_a_voiced)), 2) if len(f0_a_voiced) > 0 else 0,
        "speaker_b_median_hz": round(float(np.median(f0_b_voiced)), 2) if len(f0_b_voiced) > 0 else 0,
        "method": "separate_wavs",
    }


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
    """단일 오디오에서 KMeans로 화자 분리 후 분석"""
    if not HAS_SKLEARN:
        return {"score": 50, "error": "sklearn_not_available", "method": "mixed_audio"}

    data = _extract_f0(audio_path)
    f0, vad = data["f0"], data["vad"]
    hop_sec = data["hop_length"] / data["sr"]

    voiced = vad & np.isfinite(f0)
    segs = _segments_from_mask(voiced, min_len=12)

    # Segment-level median F0
    seg_stats = []
    for s, e in segs:
        vals = f0[s:e]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        seg_stats.append({"start": s, "end": e, "median_f0": float(np.median(vals))})

    if len(seg_stats) < 2:
        return {"score": 50, "error": "too_few_segments", "method": "mixed_audio"}

    X = np.array([np.log(s["median_f0"] + 1e-9) for s in seg_stats]).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    c0, c1 = np.mean(X[labels == 0]), np.mean(X[labels == 1])
    low_label = 0 if c0 < c1 else 1

    n_frames = len(f0)
    mask_low = np.zeros(n_frames, dtype=bool)
    mask_high = np.zeros(n_frames, dtype=bool)

    for i, s in enumerate(seg_stats):
        target = mask_low if labels[i] == low_label else mask_high
        target[s["start"]:s["end"]] = True

    f0_low = f0.copy()
    f0_high = f0.copy()
    f0_low[~mask_low] = np.nan
    f0_high[~mask_high] = np.nan

    z_low = _robust_z(_interpolate_nans(f0_low), mask=mask_low)
    z_high = _robust_z(_interpolate_nans(f0_high), mask=mask_high)

    max_lag = int(2.0 / hop_sec)
    corr, lag = _best_lag_sync(z_low, z_high, mask_low, mask_high, max_lag_frames=max_lag)
    score = _corr_to_score(corr)

    return {
        "score": score,
        "best_corr": round(corr, 4),
        "best_lag_seconds": round(lag * hop_sec, 4),
        "method": "mixed_audio_kmeans",
    }


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
                "scores": {"voice_pitch": 50.0},
                "raw": {"score": 50, "error": "librosa_not_installed", "method": "default"},
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
                        "scores": {"voice_pitch": 50.0},
                        "raw": {"score": 50, "error": str(e), "method": "separate_wavs_failed"},
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
                    pass

            if len(valid_paths) == 1:
                try:
                    raw = _analyze_mixed_audio(valid_paths[0], call_id or "unknown")
                    return {"scores": {"voice_pitch": float(raw["score"])}, "raw": raw}
                except Exception as e:
                    pass

        return {
            "scores": {"voice_pitch": 50.0},
            "raw": {"score": 50, "error": "no_valid_audio", "method": "default"},
        }