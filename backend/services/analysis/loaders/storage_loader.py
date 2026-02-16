import os
import tempfile
from firebase_admin import storage
from backend.config import FIREBASE_STORAGE_BUCKET
from backend.services.firestore import get_firestore

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

        def _download_blob(storage_path: str):
            if storage_path in downloaded_paths:
                # 이미 다운로드됐어도 로컬 경로는 반환
                local_path = os.path.join(local_dir, storage_path)
                return local_path
            local_path = os.path.join(local_dir, storage_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob = bucket.blob(storage_path)
            blob.download_to_filename(local_path)
            downloaded_paths.add(storage_path)
            return local_path

        for item in recording_files:
            if not isinstance(item, dict):
                continue
            storage_path = (
                item.get("fileName")
                or item.get("storage_path")
                or item.get("path")
                or item.get("filename")
            )
            if not storage_path:
                continue

            local_path = _download_blob(storage_path)
            # 항상 downloaded 리스트에 추가 (중복 다운로드만 방지, 등록은 항상)
            downloaded.append(
                {
                    "uid": item.get("uid"),
                    "storage_path": storage_path,
                    "local_path": local_path,
                }
            )

            if storage_path.endswith(".m3u8"):
                prefix = os.path.dirname(storage_path)
                if prefix:
                    for blob in bucket.list_blobs(prefix=prefix):
                        _download_blob(blob.name)
                else:
                    base_prefix = storage_path[:-5]
                    for blob in bucket.list_blobs(prefix=base_prefix):
                        _download_blob(blob.name)

        return downloaded