# services/analysis/loaders/storage_loader.py - storage download utilities
import os
import tempfile
from firebase_admin import storage
from backend.config import FIREBASE_STORAGE_BUCKET
from backend.services.firestore import get_firestore

import re

_UID_RE = re.compile(r"__uid_s_(\d+)__uid_e_")

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
    # Ensure Firebase app is initialized before accessing storage
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
        blob.download_to_filename(local_path)
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
            # flat 구조: basename만 사용 (디렉토리 중첩 방지)
            local_path = os.path.join(local_dir, os.path.basename(storage_path))
            blob = bucket.blob(storage_path)
            blob.download_to_filename(local_path)
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

            # uid: item에 있으면 사용, 없으면 파일명에서 파싱
            uid = item.get("uid") or _extract_uid_from_filename(storage_path)

            local_path = _download_blob(storage_path)
            downloaded.append({
                "uid": uid,
                "storage_path": storage_path,
                "local_path": local_path,
            })

            # .m3u8면 같은 uid prefix의 .ts 파일만 다운로드
            if storage_path.endswith(".m3u8"):
                # 이 m3u8의 prefix = 파일명에서 .m3u8 제거한 부분
                # "recordings/1cc4dce9..._uid_s_1317776729__uid_e_audio"
                ts_prefix = storage_path[:-5]  # .m3u8 제거
                for blob in bucket.list_blobs(prefix=ts_prefix):
                    if blob.name.endswith(".ts"):
                        _download_blob(blob.name)

        return downloaded