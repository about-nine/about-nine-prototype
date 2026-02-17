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
            m3u8_to_wav(local_path, wav_path)
            wav_items.append(
                {
                    "uid": uid,
                    "wav_path": wav_path,
                    "speaker_hint": f"uid_{uid}" if uid else None,
                    "_temp_wav": True,  # ✅ 임시 생성 표시
                }
            )

        return wav_items
