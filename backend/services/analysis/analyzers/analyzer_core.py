# services/analysis/analyzers/analyzer_core.py - analyzer base utilities
"""Shared analyzer core for 2-party conversation analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class SyncItem:
    text: str
    category: str
    speaker: str
    sentiment: str = "positive"


def normalize_conversation(conversation_obj) -> Dict[str, Any]:
    if isinstance(conversation_obj, dict) and "conversation" in conversation_obj:
        return conversation_obj
    conv = getattr(conversation_obj, "conversation", None)
    if conv is None:
        return {"conversation": []}
    normalized = []
    for u in conv:
        if isinstance(u, dict):
            normalized.append(u)
        else:
            normalized.append(
                {
                    "speaker": getattr(u, "speaker", None),
                    "start": getattr(u, "start", None),
                    "end": getattr(u, "end", None),
                    "text": getattr(u, "text", ""),
                }
            )
    return {"conversation": normalized}


class BaseSyncModule(ABC):
    """Generic hooks for similarity/sync analyzers."""

    method_label: str = "default"
    semantic_weight: float = 0.40
    agreement_weight: float = 0.30
    category_weight: float = 0.30

    def get_weights(self) -> Tuple[float, float, float]:
        total = self.semantic_weight + self.agreement_weight + self.category_weight
        if total <= 0:
            return (0.40, 0.30, 0.30)
        return (
            self.semantic_weight / total,
            self.agreement_weight / total,
            self.category_weight / total,
        )

    @abstractmethod
    def extract_items(self, utterances: List[str], speakers: List[str]) -> List[SyncItem]:
        ...

    @abstractmethod
    def count_agreements(self, utterances: List[str], speakers: List[str]) -> Tuple[float, int]:
        ...

    @abstractmethod
    def semantic_similarity(self, items_a: List[SyncItem], items_b: List[SyncItem]) -> Tuple[float, str]:
        ...

    @abstractmethod
    def category_sync(self, items_a: List[SyncItem], items_b: List[SyncItem]) -> Dict[str, float]:
        ...

    def category_detail(
        self,
        items_a: List[SyncItem],
        items_b: List[SyncItem],
        cat_sync: Dict[str, float],
    ) -> Dict[str, Any]:
        return {}

    def analysis_metadata(
        self,
        utterances: List[str],
        speakers: List[str],
        items_a: List[SyncItem],
        items_b: List[SyncItem],
        cat_sync: Dict[str, float],
    ) -> Dict[str, Any]:
        return {}


class ConversationSyncCore:
    """Runs shared scoring pipeline using a module implementation."""

    def __init__(self, module: BaseSyncModule):
        self.module = module

    def analyze(self, conversation_obj) -> Dict[str, Any]:
        data = normalize_conversation(conversation_obj)
        conv = data.get("conversation", [])
        if not conv:
            return {"score": 0, "method": self.module.method_label, "error": "empty"}

        utterances = [u.get("text", "") for u in conv if u.get("text")]
        speakers = [u.get("speaker") for u in conv if u.get("text")]
        unique_speakers = list(dict.fromkeys(speakers))
        if len(unique_speakers) < 2:
            return {"score": 0, "error": "need_2_speakers", "method": self.module.method_label}

        speaker_a, speaker_b = unique_speakers[0], unique_speakers[1]

        all_items = self.module.extract_items(utterances, speakers)
        items_a = [p for p in all_items if p.speaker == speaker_a]
        items_b = [p for p in all_items if p.speaker == speaker_b]
        if not items_a and not items_b:
            return {
                "score": 0,
                "error": "no_items_detected",
                "method": self.module.method_label,
            }

        agreement_score, agreement_count = self.module.count_agreements(utterances, speakers)
        semantic_sim, semantic_method = self.module.semantic_similarity(items_a, items_b)
        cat_sync = self.module.category_sync(items_a, items_b)
        cat_avg = sum(cat_sync.values()) / len(cat_sync) if cat_sync else 0.0
        cat_detail = self.module.category_detail(items_a, items_b, cat_sync)
        semantic_w, agreement_w, category_w = self.module.get_weights()

        final = semantic_w * semantic_sim + agreement_w * agreement_score + category_w * cat_avg
        score_100 = int(round(final * 100))

        payload: Dict[str, Any] = {
            "score": score_100,
            "semantic_similarity": round(semantic_sim, 4),
            "explicit_agreement_score": round(agreement_score, 4),
            "explicit_agreement_count": agreement_count,
            "category_sync": {k: round(v, 4) for k, v in cat_sync.items()},
            "speaker_a_items": len(items_a),
            "speaker_b_items": len(items_b),
            "method": semantic_method,
            "scoring_weights": {
                "semantic": round(semantic_w, 4),
                "agreement": round(agreement_w, 4),
                "category": round(category_w, 4),
            },
        }
        if cat_detail:
            payload["category_detail"] = cat_detail

        payload.update(
            self.module.analysis_metadata(
                utterances=utterances,
                speakers=speakers,
                items_a=items_a,
                items_b=items_b,
                cat_sync=cat_sync,
            )
        )
        return payload
