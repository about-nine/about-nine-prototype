# services/analysis/models/schema.py - analysis data schemas
from dataclasses import dataclass
from typing import List


@dataclass
class ConversationTurn:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Conversation:
    call_id: str
    conversation: List[ConversationTurn]
