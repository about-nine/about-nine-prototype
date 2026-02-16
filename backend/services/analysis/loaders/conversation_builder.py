# backend/services/analysis/loaders/conversation_builder.py

from typing import List, Dict
import os
import openai

# -------------------------
# OpenAI Whisper API client
# -------------------------
_client = None

def get_openai_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# -------------------------
# Single audio → segments
# -------------------------
def audio_to_segments(
    wav_path: str,
    speaker_id: str,
) -> List[Dict]:
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
        return []

    client = get_openai_client()

    try:
        with open(wav_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language="ko"
            )
    except Exception as e:
        print(f"⚠️ STT 실패 ({wav_path}): {e}")
        return []

    segments = []
    for seg in (result.segments or []):
        text = seg.get("text", "").strip() if isinstance(seg, dict) else getattr(seg, "text", "").strip()
        if not text:
            continue

        start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)
        end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)

        segments.append({
            "speaker": speaker_id,
            "start": float(start),
            "end": float(end),
            "text": text
        })

    return segments


# -------------------------
# Multiple speakers merge
# -------------------------
def build_conversation(
    call_id: str,
    speaker_audio_map: Dict[str, str],
) -> Dict:
    all_segments: List[Dict] = []

    for speaker_id, wav_path in speaker_audio_map.items():
        segments = audio_to_segments(wav_path, speaker_id)
        all_segments.extend(segments)

    all_segments.sort(key=lambda x: x["start"])

    return {
        "call_id": call_id,
        "conversation": all_segments
    }


class ConversationBuilder:
    def build(
        self,
        call_id: str,
        wav_items: List[Dict],
        uid_mapping: Dict | None = None,
        participants: List[str] | None = None,
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