# services/analysis/loaders/storage_loader.py - storage download utilities
import os
import tempfile
from firebase_admin import storage
from backend.config import FIREBASE_STORAGE_BUCKET
from backend.services.firestore import get_firestore

import re

_UID_RE = re.compile(r"__uid_s_(\d+)__uid_e_")

DOWNLOAD_TIMEOUT = int(os.getenv("STORAGE_DOWNLOAD_TIMEOUT", "60"))

def _extract_uid_from_filename(storage_path: str):
    """파일명에서 Agora uid 추출."""
    m = _UID_RE.search(storage_path)
    return m.group(1) if m else None

_bucket = None

def get_bucket():
    global _bucket
    if _bucket:
        return _bucket
    if not FIREBASE_STORAGE_BUCKET:
        raise RuntimeError(
            "FIREBASE_STORAGE_BUCKET is not set. "
            "Set it to your Firebase Storage bucket name "
            "(e.g. 'your-project-id.appspot.com')."
        )
    get_firestore()
    _bucket = storage.bucket()
    return _bucket


def download_prefix(prefix: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)

    bucket = get_bucket()
    blobs = bucket.list_blobs(prefix=prefix)

    local_files = []
    for blob in blobs:
        filename = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, filename)
        blob.download_to_filename(local_path, timeout=DOWNLOAD_TIMEOUT)
        local_files.append(local_path)

    return local_files


class StorageLoader:
    def __init__(self, local_root: str | None = None): 
        self.local_root = local_root or os.path.join(
            tempfile.gettempdir(), "about_nine_recordings"
        )
        
    def download_recordings(self, recording_files, talk_id: str):
        bucket = get_bucket()
        local_dir = os.path.join(self.local_root, str(talk_id))
        os.makedirs(local_dir, exist_ok=True)

        downloaded = []
        downloaded_paths = set()

        def _download_blob(storage_path: str) -> str:
            if storage_path in downloaded_paths:
                return os.path.join(local_dir, os.path.basename(storage_path))
            local_path = os.path.join(local_dir, os.path.basename(storage_path))
            blob = bucket.blob(storage_path)
            print(f"⬇️ [{talk_id}] Downloading: {os.path.basename(storage_path)}")
            blob.download_to_filename(local_path, timeout=DOWNLOAD_TIMEOUT)
            print(f"✅ [{talk_id}] Downloaded: {os.path.basename(storage_path)}")
            downloaded_paths.add(storage_path)
            return local_path

        for item in recording_files:
            if not isinstance(item, dict):
                continue
            storage_path = (
                item.get("fileName") or item.get("storage_path")
                or item.get("path") or item.get("filename")
            )
            if not storage_path:
                continue

            uid = item.get("uid") or _extract_uid_from_filename(storage_path)

            local_path = _download_blob(storage_path)
            downloaded.append({
                "uid": uid,
                "storage_path": storage_path,
                "local_path": local_path,
            })

            if storage_path.endswith(".m3u8"):
                ts_prefix = storage_path[:-5]
                ts_blobs = [b for b in bucket.list_blobs(prefix=ts_prefix) if b.name.endswith(".ts")]
                if not ts_blobs:
                    print(f"⚠️ [{talk_id}] No .ts files found for {os.path.basename(storage_path)} — recording may not be uploaded yet")
                for blob in ts_blobs:
                    _download_blob(blob.name)

        return downloaded