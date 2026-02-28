# services/embedding_service.py - OpenAI embedding helper
import os
from typing import List, Optional

import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class EmbeddingService:
    def __init__(self, model_name= None):
        self._model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._client: Optional[OpenAI] = None
        
    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=self._openai_timeout(),
                max_retries=1,
            )
        return self._client

    def _openai_timeout(self) -> float:
        raw = os.getenv("OPENAI_TIMEOUT", "30")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 30.0

    def encode_text(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        if not HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=self._openai_timeout(),
                max_retries=1,
            )
            resp = client.embeddings.create(
                model=self._model_name,
                input=text[:8000],  # API 토큰 제한 대비
            )
            vec = resp.data[0].embedding
            return normalize_vector(vec)
        except Exception:
            return None


def normalize_vector(vec: List[float]) -> List[float]:
    if not vec:
        return vec
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()
