# services/analysis_service.py - talk analysis pipeline orchestration
from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.services.firestore import get_firestore
from backend.services.rtdb import get_rtdb
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


def _apply_patch_to_talk(talk: Dict[str, Any], patch: Dict[str, Any]) -> None:
    if not patch:
        return
    for key, value in patch.items():
        if "." in key:
            root, sub = key.split(".", 1)
            cur = talk.get(root)
            if not isinstance(cur, dict):
                cur = {}
                talk[root] = cur
            cur[sub] = value
        else:
            talk[key] = value


def _build_patch_from_match_request(
    match_request: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}

    call_started = match_request.get("call_started_at")
    call_ended = match_request.get("ended_at")
    existing_ts = existing.get("timestamp") or 0

    if call_started:
        patch["call_started_at"] = call_started
    if call_ended:
        patch["call_ended_at"] = call_ended
        if call_started:
            patch["duration"] = int((call_ended - call_started) / 1000)
        if not existing_ts or call_ended > existing_ts:
            patch["timestamp"] = call_ended
    elif call_started and not existing_ts:
        patch["timestamp"] = call_started

    files = match_request.get("recording_file_list") or []
    if isinstance(files, list) and files:
        existing_files = existing.get("recording_files") or []
        if not isinstance(existing_files, list) or len(files) >= len(existing_files):
            patch["recording_files"] = files

    status = match_request.get("recording_uploading_status")
    if status:
        patch["recording_uploading_status"] = status

    uid_map = match_request.get("uid_mapping") or {}
    if isinstance(uid_map, dict) and uid_map:
        existing_map = existing.get("uid_mapping") or {}
        if not isinstance(existing_map, dict):
            existing_map = {}
        merged = {**existing_map, **uid_map}
        patch["uid_mapping"] = merged

    initiator = match_request.get("initiator")
    receiver = match_request.get("receiver")
    if initiator:
        initiator_selection = match_request.get("initiator_selection")
        if initiator_selection is not None:
            patch[f"selections.{initiator}"] = initiator_selection
    if receiver:
        receiver_selection = match_request.get("receiver_selection")
        if receiver_selection is not None:
            patch[f"selections.{receiver}"] = receiver_selection

    return patch


def _refresh_talk_from_rtdb(
    talk_id: str,
    talk: Dict[str, Any],
    talk_ref,
) -> Dict[str, Any]:
    match_request_id = talk.get("match_request_id") or talk_id
    if not match_request_id:
        return talk
    rtdb = get_rtdb()
    if not rtdb:
        return talk
    try:
        match_request = rtdb.child("match_requests").child(match_request_id).get()
    except Exception:
        return talk
    if not match_request:
        return talk

    patch = _build_patch_from_match_request(match_request, talk)
    if patch:
        try:
            talk_ref.update(patch)
        except Exception:
            pass
        _apply_patch_to_talk(talk, patch)
    return talk


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
      - Run analyzers (5 text-based)
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
            print(f"🗑️ [{talk_id}] Cleaned up temp recordings")  # ✅ 로그 추가
        except Exception:
            pass

    def analyze_talk_pipeline(self, talk_id: str, force: bool = False) -> Dict[str, Any]:
        db = _get_db()
        talk_ref = db.collection("talk_history").document(talk_id)
        conversation_list = None

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

            if conversation_obj is None:
                recording_files = _normalize_recording_files(talk)
                if not recording_files:
                    talk = _refresh_talk_from_rtdb(talk_id, talk, talk_ref)
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

                # (b) Convert to wav (or extract wav) for STT
                # expected return: [{"uid":..., "wav_path":..., "speaker_hint":...}, ...]
                wav_items = self.audio_builder.to_wav(downloaded, talk_id=talk_id)

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
                # conv_dict expected: {"call_id":..., "conversation":[{speaker,start,end,text},...]}
                conversation_list = conv_dict.get("conversation") or []
                conv_warnings = conv_dict.get("warnings") or {}
                transcription_warn = conv_warnings.get("transcription")
                if transcription_warn:
                    try:
                        talk_ref.update(
                            {
                                "analysis_warning": "transcription_failed",
                                "analysis_transcription_errors": transcription_warn,
                            }
                        )
                    except Exception:
                        pass
                    talk["analysis_warning"] = "transcription_failed"
                    talk["analysis_transcription_errors"] = transcription_warn

                # persist built conversation for caching
                talk_ref.update(
                    {
                        "conversation": conversation_list,
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
            conversation_obj = _conversation_from_talk(talk)

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
            print(f"🔬 [{talk_id}] Starting analyzers...")
            
            print(f"  → rhythm_analyzer...")
            rhythm_out = self.rhythm.score(conversation_obj)
            print(f"  ✓ rhythm done")
            
            print(f"  → discourse_analyzer...")
            discourse_out = self.discourse.score(conversation_obj)
            print(f"  ✓ discourse done")
            
            print(f"  → romantic_analyzer...")
            romantic_out = self.romantic.score(conversation_obj)
            print(f"  ✓ romantic done")
            
            print(f"  → lsm_analyzer...")
            lsm_out = self.lsm.score(conversation_obj)
            print(f"  ✓ lsm done")
            
            print(f"  → preference_analyzer...")
            pref_out = self.preference.score(conversation_obj)
            print(f"  ✓ preference done")

            print(f"✅ [{talk_id}] All analyzers complete")
            
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            print(f"❌ [{talk_id}] Analyzer failed: {e}")
            print(err_trace)
            try:
                talk_ref.update(
                    {
                        "analysis_error": f"analyzer: {str(e)}",
                        "analysis_trace": err_trace,
                        "analysis_failed_at": _now_ms(),
                        "analysis_status": "failed",
                    }
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

        # 🔥 Convert numpy types to Python native types for Firestore
        def _sanitize_for_firestore(obj):
            """Recursively convert numpy types to Python native types"""
            try:
                import numpy as np
                
                if isinstance(obj, dict):
                    return {k: _sanitize_for_firestore(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_sanitize_for_firestore(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            except ImportError:
                # numpy not installed, return as-is
                return obj
        
        analysis_sanitized = _sanitize_for_firestore(analysis)

        # 4) Persist analysis
        talk_ref.update(
            {
                "analysis": analysis_sanitized,
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

        return _finish({"success": True, "talk_id": talk_id, "analysis": analysis_sanitized})


# Convenience function to match your existing import style
_service = AnalysisService()


def analyze_talk_pipeline(talk_id: str, force: bool = False) -> Dict[str, Any]:
    return _service.analyze_talk_pipeline(talk_id, force=force)
