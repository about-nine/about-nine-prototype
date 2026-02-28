# services/analysis/loaders/conversation_builder.py - STT and conversation assembly
"""
Conversation Builder (OpenAI Whisper API + VAD preprocessing)
==============================================================
Agora 개별 녹음 WAV → OpenAI Whisper API STT → 통합 대화 전사본

핵심 전처리:
  1. VAD (Voice Activity Detection): 무음 구간 제거 → Whisper 환각 방지
  2. 타임스탬프 복원: VAD로 잘라낸 구간의 원래 시간 복원
  3. 중복 제거: Whisper 반복 생성 버그 필터링
  4. 환각 필터: 짧은 무의미 세그먼트 제거

의존성: openai, numpy, soundfile
"""

import os
import wave
import struct
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


WHISPER_MAX_BYTES = _int_env("WHISPER_MAX_BYTES", 24 * 1024 * 1024)  # ~24MB default
WHISPER_CHUNK_SECONDS = max(60, _int_env("WHISPER_CHUNK_SECONDS", 600))  # chunk at 10 min


# ============================================================================
# VAD (Voice Activity Detection)
# ============================================================================

def _read_wav_np(wav_path: str) -> Tuple[np.ndarray, int]:
    """WAV → numpy array"""
    if HAS_SOUNDFILE:
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]  # mono
        return data, sr

    # fallback: wave module
    w = wave.open(wav_path, "r")
    sr = w.getframerate()
    n = w.getnframes()
    ch = w.getnchannels()
    raw = w.readframes(n)
    w.close()
    samples = struct.unpack(f"<{n * ch}h", raw)
    data = np.array(samples, dtype=np.float32) / 32768.0
    if ch > 1:
        data = data[::ch]
    return data, sr


def _write_wav_np(data: np.ndarray, sr: int, path: str):
    """numpy array → WAV"""
    if HAS_SOUNDFILE:
        sf.write(path, data, sr)
        return

    # fallback
    samples = (data * 32767).astype(np.int16)
    w = wave.open(path, "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(samples.tobytes())
    w.close()


def _compute_rms_frames(data, sr, frame_ms=30):
    frame_size = int(sr * frame_ms / 1000)
    n_frames = len(data) // frame_size
    frames = data[:n_frames * frame_size].reshape(n_frames, frame_size)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def _vad_segments(
    data: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    energy_threshold: float = 0.01,
    min_speech_ms: int = 300,
    padding_ms: int = 200,
) -> List[Tuple[float, float]]:
    """
    에너지 기반 VAD → 음성 구간 [(start_sec, end_sec), ...] 반환

    Parameters:
        energy_threshold: RMS 에너지 임계값 (0.01 = 무음 대비 충분히 큰 소리)
        min_speech_ms: 최소 음성 구간 길이 (짧은 노이즈 무시)
        padding_ms: 음성 구간 앞뒤 패딩 (자연스러운 잘림 방지)
    """
    rms = _compute_rms_frames(data, sr, frame_ms)

    # 적응형 임계값: 상위 10% 에너지의 10%를 threshold로
    if len(rms) > 0:
        sorted_rms = np.sort(rms)
        top_10 = sorted_rms[int(len(sorted_rms) * 0.9):]
        if len(top_10) > 0 and np.mean(top_10) > 0:
            adaptive_threshold = np.mean(top_10) * 0.05
            energy_threshold = max(energy_threshold, adaptive_threshold)

    # 프레임별 음성 여부
    is_speech = rms > energy_threshold

    # 연속 음성 구간 찾기
    segments = []
    in_speech = False
    start_frame = 0

    for i, v in enumerate(is_speech):
        if v and not in_speech:
            start_frame = i
            in_speech = True
        elif not v and in_speech:
            duration_ms = (i - start_frame) * frame_ms
            if duration_ms >= min_speech_ms:
                start_sec = max(0, start_frame * frame_ms / 1000 - padding_ms / 1000)
                end_sec = min(len(data) / sr, i * frame_ms / 1000 + padding_ms / 1000)
                segments.append((start_sec, end_sec))
            in_speech = False

    # 마지막 구간 처리
    if in_speech:
        duration_ms = (len(is_speech) - start_frame) * frame_ms
        if duration_ms >= min_speech_ms:
            start_sec = max(0, start_frame * frame_ms / 1000 - padding_ms / 1000)
            end_sec = len(data) / sr
            segments.append((start_sec, end_sec))

    # 인접 구간 병합 (gap < 500ms)
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < 0.5:  # 500ms 이내면 병합
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def _extract_speech_chunks(
    wav_path: str,
) -> Tuple[Optional[str], List[Tuple[float, float]]]:
    """
    WAV에서 음성 구간만 추출하여 새 WAV 생성

    Returns:
        (trimmed_wav_path, original_time_segments)
        - trimmed_wav_path: 음성 구간만 합친 WAV
        - original_time_segments: [(orig_start, orig_end), ...] 원본 시간 매핑
    """
    data, sr = _read_wav_np(wav_path)

    if len(data) == 0:
        return None, []

    segments = _vad_segments(data, sr)

    if not segments:
        return None, []

    # 음성 구간만 합치기
    chunks = []
    for start_sec, end_sec in segments:
        s = int(start_sec * sr)
        e = int(end_sec * sr)
        chunks.append(data[s:e])

    if not chunks:
        return None, []

    combined = np.concatenate(chunks)

    # 임시 파일에 저장
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _write_wav_np(combined, sr, tmp.name)

    return tmp.name, segments


# ============================================================================
# Timestamp restoration
# ============================================================================

def _restore_timestamps(
    segments: List[Dict],
    vad_regions: List[Tuple[float, float]],
) -> List[Dict]:
    """
    VAD로 잘라낸 음성의 Whisper 타임스탬프를 원본 시간으로 복원

    Whisper는 합쳐진 음성(VAD output)에 대해 0초부터 타임스탬프를 매기므로,
    원본 WAV에서의 실제 시간으로 변환해야 함.

    예: VAD regions = [(2.0, 5.0), (10.0, 15.0)]
        Whisper segment start=3.5 → 합쳐진 음성에서 3.5초
        → 첫 region이 3초, 3.0 < 3.5이므로 두 번째 region에 해당
        → 10.0 + (3.5 - 3.0) = 10.5초가 원본 시간
    """
    if not vad_regions:
        return segments

    # 각 region의 합쳐진 음성에서의 시작 offset 계산
    region_offsets = []  # (combined_start, combined_end, original_start, original_end)
    combined_pos = 0.0
    for orig_start, orig_end in vad_regions:
        duration = orig_end - orig_start
        region_offsets.append((combined_pos, combined_pos + duration, orig_start, orig_end))
        combined_pos += duration

    def _map_time(combined_time: float) -> float:
        for c_start, c_end, o_start, o_end in region_offsets:
            if c_start <= combined_time <= c_end + 0.01:  # small tolerance
                offset = combined_time - c_start
                return o_start + offset
        # 범위 밖이면 마지막 region 기준
        if region_offsets:
            _, c_end, _, o_end = region_offsets[-1]
            return o_end
        return combined_time

    restored = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["start"] = round(_map_time(seg["start"]), 2)
        new_seg["end"] = round(_map_time(seg["end"]), 2)
        restored.append(new_seg)

    return restored


# ============================================================================
# Post-processing filters
# ============================================================================

def _deduplicate_segments(segments: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """
    Whisper 반복 생성 버그 필터링
    연속으로 동일/유사한 텍스트가 나오면 첫 번째만 유지
    """
    if not segments:
        return []

    def _text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        intersection = a_set & b_set
        union = a_set | b_set
        return len(intersection) / len(union)

    filtered = [segments[0]]
    for seg in segments[1:]:
        prev_text = filtered[-1]["text"]
        curr_text = seg["text"]

        # 완전 동일
        if curr_text.strip() == prev_text.strip():
            continue

        # 높은 유사도 (반복 생성)
        if _text_similarity(prev_text, curr_text) >= similarity_threshold:
            # 더 긴 쪽을 유지
            if len(curr_text) > len(prev_text):
                filtered[-1] = seg
            continue

        filtered.append(seg)

    return filtered


def _filter_hallucinations(segments: List[Dict], min_text_len: int = 2) -> List[Dict]:
    """
    Whisper 환각 필터링:
    - 너무 짧은 세그먼트 (1글자)
    - 알려진 환각 패턴
    """
    HALLUCINATION_PATTERNS = {
        # English
        "listening to music", "music", "laughs", "laughter", "applause",
        "thanks for watching", "thank you for watching", "please subscribe",
        "like and subscribe", "silence", "inaudible",
        # Korean
        "음악", "웃음", "박수", "침묵", "알아들을 수 없음", "자막 제공",
        "구독과 좋아요", "시청해 주셔서 감사합니다", "영상 시청",
    }

    filtered = []
    for seg in segments:
        text = seg["text"].strip()

        # 너무 짧음
        if len(text) < min_text_len:
            continue

        # 알려진 환각 패턴
        if text.lower().strip(".,!? ") in HALLUCINATION_PATTERNS:
            continue

        # 숫자만 있는 세그먼트
        if text.replace(" ", "").replace(".", "").isdigit():
            continue

        filtered.append(seg)

    return filtered


# ============================================================================
# OpenAI Whisper API
# ============================================================================

_client = None
WHISPER_PROMPT = (
    "This audio may contain conversation in any language. "
    "Transcribe speech as spoken, preserving the original language and script."
)


def _openai_timeout() -> float:
    raw = os.getenv("WHISPER_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "120"))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 120.0


def _get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_openai_timeout(),
            max_retries=1,
        )
    return _client


def _call_whisper_api(path: str):
    client = _get_openai_client()
    with open(path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            prompt=WHISPER_PROMPT,
        )


def _parse_whisper_segments(result, speaker_id: str, time_offset: float = 0.0) -> List[Dict]:
    parsed: List[Dict] = []
    if not result:
        return parsed
    segments_iter = getattr(result, "segments", []) or []
    for seg in segments_iter:
        if isinstance(seg, dict):
            text = seg.get("text", "").strip()
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
        else:
            text = getattr(seg, "text", "").strip()
            start = float(getattr(seg, "start", 0))
            end = float(getattr(seg, "end", 0))
        if not text:
            continue
        parsed.append(
            {
                "speaker": speaker_id,
                "start": start + time_offset,
                "end": end + time_offset,
                "text": text,
            }
        )
    return parsed


def _transcribe_with_chunking(wav_path: str, speaker_id: str) -> Tuple[List[Dict], Optional[str]]:
    """Split long audio into manageable pieces (<25MB) for Whisper."""
    try:
        data, sr = _read_wav_np(wav_path)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ STT chunk read failed ({wav_path}): {exc}")
        return [], "chunk_read_failed"

    if len(data) == 0:
        return [], "empty_audio"

    chunk_samples = max(int(sr * WHISPER_CHUNK_SECONDS), sr * 60)
    segments: List[Dict] = []
    offset = 0.0

    for start in range(0, len(data), chunk_samples):
        chunk = data[start:start + chunk_samples]
        if len(chunk) == 0:
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            _write_wav_np(chunk, sr, tmp.name)
            result = _call_whisper_api(tmp.name)
            segments.extend(_parse_whisper_segments(result, speaker_id, offset))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Whisper chunk failed ({wav_path}, offset={offset:.2f}s): {exc}")
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return segments, "chunk_transcription_failed"
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        offset += len(chunk) / sr

    return segments, None


def _transcribe_wav(wav_path: str, speaker_id: str) -> Tuple[List[Dict], Optional[str]]:
    """단일 WAV → segments (with VAD preprocessing)"""
    if not os.path.exists(wav_path):
        return [], "file_missing"
    try:
        if os.path.getsize(wav_path) < 1024:
            return [], "file_too_small"
    except OSError:
        return [], "file_unreadable"

    if not HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return [], "openai_not_configured"

    # 1) VAD로 음성 구간만 추출
    trimmed_path, vad_regions = _extract_speech_chunks(wav_path)
    if not trimmed_path or not vad_regions:
        trimmed_path = wav_path
        vad_regions = []

    path_for_whisper = trimmed_path
    raw_segments: List[Dict] = []
    error_reason: Optional[str] = None

    try:
        needs_chunking = os.path.getsize(path_for_whisper) > WHISPER_MAX_BYTES
    except OSError:
        needs_chunking = False

    try:
        if needs_chunking:
            raw_segments, error_reason = _transcribe_with_chunking(path_for_whisper, speaker_id)
        else:
            result = _call_whisper_api(path_for_whisper)
            raw_segments = _parse_whisper_segments(result, speaker_id)
            if not raw_segments:
                error_reason = "no_segments"
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ STT 실패 ({wav_path}): {e}")
        error_reason = "stt_failed"
        raw_segments = []
    finally:
        # 임시 파일 정리
        if trimmed_path != wav_path and os.path.exists(trimmed_path):
            try:
                os.unlink(trimmed_path)
            except OSError:
                pass

    if not raw_segments:
        return [], error_reason or "no_segments"

    # 4) 타임스탬프 복원 (VAD 사용했을 때)
    if vad_regions:
        raw_segments = _restore_timestamps(raw_segments, vad_regions)

    # 5) 후처리: 환각 필터 + 중복 제거
    raw_segments = _filter_hallucinations(raw_segments)
    raw_segments = _deduplicate_segments(raw_segments)

    return raw_segments, None


# ============================================================================
# Multi-speaker merge
# ============================================================================


def build_conversation(
    call_id: str,
    speaker_audio_map: Dict[str, str],
) -> Dict:
    """여러 화자의 WAV → 시간순 정렬된 conversation"""
    all_segments: List[Dict] = []
    warnings: Dict[str, Dict[str, str]] = {}

    for speaker_id, wav_path in speaker_audio_map.items():
        segments, error = _transcribe_wav(wav_path, speaker_id)
        if error:
            warnings.setdefault("transcription", {})[speaker_id] = error
        all_segments.extend(segments)

    # 시간순 정렬
    all_segments.sort(key=lambda x: x["start"])

    result = {
        "call_id": call_id,
        "conversation": all_segments,
    }
    if warnings:
        result["warnings"] = warnings
    return result


# ============================================================================
# Class interface (analysis_service.py 호환)
# ============================================================================


class ConversationBuilder:
    def build(
        self,
        call_id: str,
        wav_items: List[Dict],
        uid_mapping: Optional[Dict] = None,
        participants: Optional[List[str]] = None,
    ) -> Dict:
        speaker_audio_map: Dict[str, str] = {}
        speaker_wavs: Dict[str, str] = {}

        for idx, item in enumerate(wav_items or []):
            if not isinstance(item, dict):
                continue
            wav_path = item.get("wav_path")
            if not wav_path:
                continue

            speaker_id = item.get("speaker_hint")
            uid = item.get("uid")

            if not speaker_id and uid_mapping and uid is not None:
                speaker_id = uid_mapping.get(str(uid)) or uid_mapping.get(uid)

            if not speaker_id and uid is not None:
                speaker_id = f"uid_{uid}"

            if not speaker_id and participants and idx < len(participants):
                speaker_id = str(participants[idx])

            if not speaker_id:
                speaker_id = f"speaker_{idx + 1}"

            speaker_audio_map[speaker_id] = wav_path
            speaker_wavs[speaker_id] = wav_path

        conv = build_conversation(call_id=call_id, speaker_audio_map=speaker_audio_map)
        conv["speaker_wavs"] = speaker_wavs
        return conv
