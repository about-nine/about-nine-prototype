from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.services.firestore import get_firestore
from firebase_admin import firestore
from backend.services.chemistry_model import ChemistryModel
from backend.services.embedding_service import EmbeddingService
from backend.services.user_profile_service import update_user_embedding

from backend.services.analysis.models.schema import (
    Conversation,
    ConversationTurn,
)

from backend.services.analysis.loaders.storage_loader import StorageLoader
from backend.services.analysis.loaders.audio_builder import AudioBuilder
from backend.services.analysis.loaders.conversation_builder import ConversationBuilder

from backend.services.analysis.analyzers.rhythm_analyzer import RhythmAnalyzer
from backend.services.analysis.analyzers.discourse_analyzer import DiscourseAnalyzer
from backend.services.analysis.analyzers.romantic_analyzer import RomanticAnalyzer
from backend.services.analysis.analyzers.lsm_analyzer import LSMAnalyzer
from backend.services.analysis.analyzers.preference_analyzer import PreferenceAnalyzer
from backend.services.analysis.analyzers.pitch_analyzer import PitchAnalyzer


def _get_db():
    return get_firestore()


# -----------------------------
# Helpers
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _normalize_recording_files(talk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    talk_history.recording_files (from RTDB match_requests.recording_file_list)
    is expected to be a list of dicts.
    Typical Agora Cloud Recording fileList item contains:
      - fileName (storage path)
      - uid
      - trackType / mixedAllAudio etc. (varies)
    We accept anything that has a usable path key.
    """
    files = talk.get("recording_files") or talk.get("recording_file_list") or []
    if not isinstance(files, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        # common keys seen across implementations
        path = f.get("fileName") or f.get("filename") or f.get("path") or f.get("storage_path")
        if not path:
            continue
        normalized.append(
            {
                "fileName": path,
                "uid": f.get("uid"),
                "trackType": f.get("trackType"),
                "isAudio": f.get("isAudio"),
                "isVideo": f.get("isVideo"),
                "mixedAllAudio": f.get("mixedAllAudio"),
                **{k: v for k, v in f.items() if k not in {"fileName", "filename", "path", "storage_path"}},
            }
        )
    return normalized


def _participants_from_talk(talk: Dict[str, Any]) -> List[str]:
    """
    talk_history.participants can be:
      - {"user_a": "...", "user_b": "..."}
      - ["uid1", "uid2"]
    """
    p = talk.get("participants")
    if isinstance(p, dict):
        vals = [p.get("user_a"), p.get("user_b")]
        return [v for v in vals if isinstance(v, str) and v]
    if isinstance(p, list):
        return [v for v in p if isinstance(v, str) and v]
    return []


def _conversation_from_talk(talk: Dict[str, Any]) -> Optional[Conversation]:
    conv = talk.get("conversation")
    if not conv:
        return None
    if isinstance(conv, dict) and "conversation" in conv:
        conv = conv["conversation"]
    if not isinstance(conv, list):
        return None

    turns: List[ConversationTurn] = []
    for u in conv:
        if not isinstance(u, dict):
            continue
        speaker = u.get("speaker")
        start = u.get("start")
        end = u.get("end")
        text = u.get("text", "")
        if not speaker or start is None or end is None:
            continue
        turns.append(
            ConversationTurn(
                speaker=str(speaker),
                start=float(start),
                end=float(end),
                text=str(text or ""),
            )
        )

    if not turns:
        return None

    return Conversation(call_id=str(talk.get("id") or talk.get("talk_id") or "unknown"), conversation=turns)


def _conversation_text(conversation_obj: Any, max_chars: int = 8000) -> str:
    if not conversation_obj:
        return ""
    turns = []
    if isinstance(conversation_obj, Conversation):
        turns = conversation_obj.conversation
    elif isinstance(conversation_obj, dict):
        turns = conversation_obj.get("conversation") or []

    chunks: List[str] = []
    for t in turns:
        if isinstance(t, ConversationTurn):
            speaker = t.speaker
            text = t.text
        elif isinstance(t, dict):
            speaker = t.get("speaker")
            text = t.get("text")
        else:
            continue
        if not text:
            continue
        if speaker:
            chunks.append(f"{speaker}: {text}")
        else:
            chunks.append(str(text))

    full_text = "\n".join(chunks).strip()
    if len(full_text) > max_chars:
        return full_text[:max_chars]
    return full_text


def _go_no_go_from_talk(talk: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    stored = talk.get("go_no_go")
    if isinstance(stored, dict) and stored:
        return {k: (True if v is True else False if v is False else None) for k, v in stored.items()}

    participants = talk.get("participants") or {}
    if not isinstance(participants, dict):
        return {}

    initiator = participants.get("user_a")
    receiver = participants.get("user_b")
    a = talk.get("initiator_response")
    b = talk.get("receiver_response")

    result: Dict[str, Optional[bool]] = {}
    if initiator:
        result[initiator] = True if a == "go" else False if a == "no" else None
    if receiver:
        result[receiver] = True if b == "go" else False if b == "no" else None
    return result


# -----------------------------
# Main pipeline
# -----------------------------
class AnalysisService:
    """
    Orchestrates:
      - Load talk_history
      - Build conversation if missing (from Firebase Storage recordings)
      - Run analyzers (5 text-based + 1 pitch-based)
      - Combine into chemistry score (ChemistryModel)
      - Persist analysis + update user profiles
    """

    def __init__(
        self,
        chemistry_model_path: str = None,
    ):
        self.storage_loader = StorageLoader()
        self.audio_builder = AudioBuilder()
        self.conversation_builder = ConversationBuilder()
        self.model = ChemistryModel()
        model_path = chemistry_model_path or os.getenv("CHEMISTRY_MODEL_PATH")
        if not model_path:
            bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
            if bucket:
                model_path = f"gs://{bucket}/models/chemistry/latest.pkl"
            else:
                model_path = "chemistry_model.pkl"
        # load() will fall back to baseline if missing/unreachable
        self.model.load(model_path)
        self.embedding = EmbeddingService()

        self.rhythm = RhythmAnalyzer()
        self.discourse = DiscourseAnalyzer()
        self.romantic = RomanticAnalyzer()
        self.lsm = LSMAnalyzer()
        self.preference = PreferenceAnalyzer()
        self.pitch = PitchAnalyzer()

    def _cleanup_local_recordings(self, talk_id: str) -> None:
        if not talk_id:
            return
        if os.getenv("ANALYSIS_KEEP_TEMP", "").lower() in {"1", "true", "yes"}:
            return
        local_root = getattr(self.storage_loader, "local_root", None)
        if not local_root:
            return
        target_dir = os.path.join(local_root, str(talk_id))
        if not os.path.exists(target_dir):
            return
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
        except Exception:
            pass

    def analyze_talk_pipeline(self, talk_id: str, force: bool = False) -> Dict[str, Any]:
        db = _get_db()
        talk_ref = db.collection("talk_history").document(talk_id)
        conversation_list = None
        speaker_wavs = None

        def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
            self._cleanup_local_recordings(talk_id)
            return result
        try:
            snap = talk_ref.get()
            if not snap.exists:
                return _finish({"success": False, "message": "talk_history not found", "talk_id": talk_id})

            talk = snap.to_dict() or {}
            existing_analysis = talk.get("analysis") or {}
            if isinstance(existing_analysis, dict) and existing_analysis.get("chemistry_score") is not None:
                return _finish(
                    {"success": True, "talk_id": talk_id, "analysis": existing_analysis, "status": "complete"}
                )

            if talk.get("analysis_status") == "running" and not force:
                started = int(talk.get("analysis_started_at") or 0)
                if started and _now_ms() - started < 180_000:
                    return _finish({"success": True, "talk_id": talk_id, "status": "running"})

            try:
                talk_ref.update({"analysis_status": "running", "analysis_started_at": _now_ms()})
            except Exception:
                pass
            participants = _participants_from_talk(talk)

            # 1) Ensure conversation exists (build if missing)
            conversation_obj = _conversation_from_talk(talk)

            # Audio inputs for pitch analyzer: prefer already-built wav paths if saved.
            # Otherwise build from Storage recordings.
            wav_paths_by_speaker: Dict[str, str] = talk.get("wav_paths_by_speaker") or {}
            wav_paths_all: List[str] = talk.get("wav_paths") or talk.get("audio_paths") or []

            if conversation_obj is None:
                recording_files = _normalize_recording_files(talk)
                if not recording_files:
                    try:
                        talk_ref.update(
                            {
                                "analysis_error": "no conversation and no recording_files",
                                "analysis_failed_at": _now_ms(),
                                "analysis_status": "failed",
                            }
                        )
                    except Exception:
                        pass
                    return _finish(
                        {
                            "success": False,
                            "message": "no conversation and no recording_files",
                            "talk_id": talk_id,
                        }
                    )

                # (a) Download from Firebase Storage to local temp paths
                # storage_loader should return local file paths + metadata
                # expected item: {"uid": <agora_uid or None>, "storage_path": "...", "local_path": "..."}
                downloaded = self.storage_loader.download_recordings(recording_files, talk_id=talk_id)

                # (b) Convert to wav (or extract wav) for STT & pitch
                # expected return: [{"uid":..., "wav_path":..., "speaker_hint":...}, ...]
                wav_items = self.audio_builder.to_wav(downloaded, talk_id=talk_id)

                # build mapping for pitch analyzer
                wav_paths_all = [x["wav_path"] for x in wav_items if x.get("wav_path")]
                wav_paths_by_speaker = {}  # will be set after speaker mapping

                # (c) Build conversation (STT -> segments -> unified conversation)
                # conversation_builder should:
                #  - transcribe per wav
                #  - label speaker using uid_mapping if present, else best-effort
                uid_mapping = talk.get("uid_mapping") or _safe_get(talk, "meta", "uid_mapping", default={}) or {}
                conv_dict = self.conversation_builder.build(
                    call_id=talk_id,
                    wav_items=wav_items,
                    uid_mapping=uid_mapping,
                    participants=participants,
                )
                # conv_dict expected: {"call_id":..., "conversation":[{speaker,start,end,text},...], "speaker_wavs":{speaker:wav_path}}
                conversation_list = conv_dict.get("conversation") or []
                speaker_wavs = conv_dict.get("speaker_wavs") or {}

                # persist built conversation for caching
                talk_ref.update(
                    {
                        "conversation": conversation_list,
                        "wav_paths": wav_paths_all,
                        "wav_paths_by_speaker": speaker_wavs,
                        "analysis_built_at": _now_ms(),
                    }
                )
        except Exception as e:
            import traceback
            err_msg = f"{type(e).__name__}: {e}"
            err_trace = traceback.format_exc()
            try:
                talk_ref.update(
                    {
                        "analysis_error": err_msg,
                        "analysis_trace": err_trace,
                        "analysis_failed_at": _now_ms(),
                        "analysis_status": "failed",
                    }
                )
            except Exception:
                pass
            return _finish(
                {
                    "success": False,
                    "message": "analysis failed",
                    "error": err_msg,
                    "talk_id": talk_id,
                }
            )

        if conversation_obj is None and conversation_list is not None:
            # reload object for analyzers
            talk["conversation"] = conversation_list
            talk["wav_paths"] = wav_paths_all
            talk["wav_paths_by_speaker"] = speaker_wavs or {}
            conversation_obj = _conversation_from_talk(talk)
            wav_paths_by_speaker = speaker_wavs or {}

        # If conversation couldn't be built, proceed with empty conversation
        if conversation_obj is None:
            conversation_obj = {"conversation": []}
            try:
                talk_ref.update({"analysis_warning": "conversation_empty", "analysis_built_at": _now_ms()})
            except Exception:
                pass
        else:
            # Clear stale warning if conversation is now available
            try:
                if talk.get("analysis_warning") == "conversation_empty":
                    talk_ref.update({"analysis_warning": firestore.DELETE_FIELD})
            except Exception:
                pass

        # 2) Run analyzers
        try:
            rhythm_out = self.rhythm.score(conversation_obj)
            discourse_out = self.discourse.score(conversation_obj)
            romantic_out = self.romantic.score(conversation_obj)
            lsm_out = self.lsm.score(conversation_obj)
            pref_out = self.preference.score(conversation_obj)

            # Pitch analyzer can accept:
            #  - per-speaker wavs (best)
            #  - or list of wav paths (fallback)
            disable_pitch = os.getenv("ANALYSIS_DISABLE_PITCH", "").lower() in {"1", "true", "yes"}
            if disable_pitch:
                pitch_out = {
                    "scores": {"voice_pitch": 50.0},
                    "raw": {"score": 50, "error": "disabled", "method": "disabled"},
                }
            else:
                pitch_out = self.pitch.score(
                    wav_paths_by_speaker=wav_paths_by_speaker if isinstance(wav_paths_by_speaker, dict) else {},
                    wav_paths=wav_paths_all if isinstance(wav_paths_all, list) else [],
                    call_id=talk_id,
                )
        except Exception as e:
            try:
                talk_ref.update(
                    {"analysis_error": str(e), "analysis_failed_at": _now_ms(), "analysis_status": "failed"}
                )
            except Exception:
                pass
            return _finish(
                {"success": False, "message": "analyzer failed", "error": str(e), "talk_id": talk_id}
            )

        feats: Dict[str, float] = {
            # keep keys stable (your earlier convention)
            "turn_taking": float(rhythm_out["scores"].get("rhythm_synchrony", 0)),
            "flow_continuity": float(discourse_out["scores"].get("topic_continuity", 0)),
            "romantic_intent": float(romantic_out["scores"].get("romantic_intent", 0)),
            "language_style_ma": float(lsm_out["scores"].get("lsm", 0)),
            "preference_sync": float(pref_out["scores"].get("preference_sync", 0)),
            "voice_pitch": float(pitch_out["scores"].get("voice_pitch", pitch_out["scores"].get("voice pitch", 0))),
        }

        # 3) Chemistry score (model can combine + optionally update weights elsewhere)
        try:
            chemistry_score = float(self.model.predict(feats))
        except Exception as e:
            try:
                talk_ref.update(
                    {
                        "analysis_error": f"chemistry_model: {e}",
                        "analysis_failed_at": _now_ms(),
                        "analysis_status": "failed",
                    }
                )
            except Exception:
                pass
            return _finish(
                {"success": False, "message": "chemistry model failed", "error": str(e), "talk_id": talk_id}
            )

        analysis: Dict[str, Any] = {
            "features": feats,
            "chemistry_score": chemistry_score,
            "details": {
                "turn_taking": rhythm_out,
                "flow_continuity": discourse_out,
                "romantic_intent": romantic_out,
                "language_style_ma": lsm_out,
                "preference_sync": pref_out,
                "voice_pitch": pitch_out,
            },
            "model_version": self.model.version(),
            "version": self.model.version(),
            "analyzed_at": _now_ms(),
        }

        # 3.5) Conversation embedding
        pair_embedding = None
        try:
            full_text = _conversation_text(conversation_obj)
            pair_embedding = self.embedding.encode_text(full_text)
        except Exception:
            pair_embedding = None
        if pair_embedding:
            analysis["pair_embedding"] = pair_embedding

        go_no_go = _go_no_go_from_talk(talk)

        # 4) Persist analysis
        talk_ref.update(
            {
                "analysis": analysis,
                "analysis_status": "complete",
                "analysis_completed_at": _now_ms(),
            }
        )

        # 5) Update user embeddings (for recommendation)
        # Only update for users with explicit go/no labels.
        if pair_embedding:
            updated_map = (talk.get("embedding_updated") or {}) if isinstance(talk, dict) else {}
            for uid, go in go_no_go.items():
                if go is None:
                    continue
                if isinstance(updated_map, dict) and updated_map.get(uid) is True:
                    continue
                try:
                    updated = update_user_embedding(uid, pair_embedding, go=go)
                    if updated:
                        talk_ref.update({f"embedding_updated.{uid}": True})
                except Exception:
                    # don't fail the whole pipeline on profile update
                    pass

        return _finish({"success": True, "talk_id": talk_id, "analysis": analysis})


# Convenience function to match your existing import style
_service = AnalysisService()


def analyze_talk_pipeline(talk_id: str, force: bool = False) -> Dict[str, Any]:
    return _service.analyze_talk_pipeline(talk_id, force=force)
