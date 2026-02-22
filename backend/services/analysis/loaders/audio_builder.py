import subprocess
import shutil
import glob
import os


def _build_wav_path(m3u8_path: str, uid):
    base, _ = os.path.splitext(m3u8_path)
    if uid:
        return f"{base}_{uid}.wav"
    return f"{base}.wav"


def find_m3u8(directory: str):
    files = glob.glob(os.path.join(directory, "*.m3u8"))
    if not files:
        raise RuntimeError("m3u8 not found")
    return files[0]


def m3u8_to_wav(m3u8_path: str, wav_path: str):
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not found in PATH")
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", m3u8_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ], check=True)


def _safe_convert_m3u8(local_path: str, wav_path: str, uid):
    """Convert a single m3u8, swallowing per-file errors so the pipeline keeps going."""
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        print(f"⚠️ [audio] Skipping empty or missing m3u8: {local_path}")
        return None
    try:
        m3u8_to_wav(local_path, wav_path)
        return wav_path
    except FileNotFoundError:
        # Bubble up so workers can alert ops (ffmpeg missing)
        raise
    except subprocess.CalledProcessError as e:
        print(f"⚠️ [audio] ffmpeg failed for uid={uid}: {local_path} ({e})")
    except Exception as e:  # noqa: BLE001 – best effort logging only
        print(f"⚠️ [audio] Unexpected ffmpeg error for {local_path}: {e}")
    return None


def build_wav_from_directory(directory: str):
    m3u8 = find_m3u8(directory)
    wav = os.path.join(directory, "audio.wav")
    m3u8_to_wav(m3u8, wav)
    return wav


class AudioBuilder:
    def to_wav(self, downloaded, talk_id: str):
        """
        downloaded: [{"uid":..., "storage_path":..., "local_path":...}, ...]
        returns: [{"uid":..., "wav_path":..., "speaker_hint":...}, ...]
        """
        wav_items = []

        for item in downloaded:
            if not isinstance(item, dict):
                continue
            local_path = item.get("local_path")
            if not local_path:
                continue
            if local_path.endswith(".wav"):
                uid = item.get("uid")
                wav_items.append(
                    {
                        "uid": uid,
                        "wav_path": local_path,
                        "speaker_hint": f"uid_{uid}" if uid else None,
                    }
                )

        for item in downloaded:
            if not isinstance(item, dict):
                continue
            local_path = item.get("local_path")
            if not local_path or not local_path.endswith(".m3u8"):
                continue
            uid = item.get("uid")
            wav_path = _build_wav_path(local_path, uid)
            converted = _safe_convert_m3u8(local_path, wav_path, uid)
            if not converted:
                continue
            wav_items.append(
                {
                    "uid": uid,
                    "wav_path": converted,
                    "speaker_hint": f"uid_{uid}" if uid else None,
                    "_temp_wav": True,  # ✅ 임시 생성 표시
                }
            )

        return wav_items
