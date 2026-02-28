# services/pair_feature_model.py - Model 1: pair feature predictor
"""
PairFeatureModel (Model 1)
==========================
두 사람이 만나면 어떤 대화가 펼쳐질지 예측.

입력 (12 features):
  embedding interaction (4):
    cosine, l2, mean_abs_diff, std_abs_diff

  A/B talk_profile (8):
    a_avg_turn_length, a_speech_pace,
    a_emotional_expression, a_vocabulary_diversity
    b_avg_turn_length, b_speech_pace,
    b_emotional_expression, b_vocabulary_diversity

출력 (5 predicted pair features):
  lsm, flow_continuity, turn_balance,
  preference_sync, romantic_sync

학습 전 (fallback):
  - embedding cosine + talk_profile 차이 기반 휴리스틱
  - 각 pair feature를 별도 규칙으로 근사

학습 후:
  - XGBoost 5개 독립 회귀 모델
  - 피처별로 별도 학습 (multi-output 아님)

대칭성:
  학습 시 (A,B)와 (B,A) 양방향 augmentation 필요
  → train_pair_feature_model.py 에서 처리
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from firebase_admin import storage
    from backend.services.firestore import get_firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False


# ── 피처 정의 ─────────────────────────────────────────────────────

EMB_FEATURES = ["cosine", "l2", "mean_abs_diff", "std_abs_diff"]

PROFILE_FIELDS = [
    "avg_turn_length",
    "speech_pace",
    "emotional_expression",
    "vocabulary_diversity",
]

PROFILE_FEATURES = (
    [f"a_{f}" for f in PROFILE_FIELDS] +
    [f"b_{f}" for f in PROFILE_FIELDS]
)

INPUT_FEATURES = EMB_FEATURES + PROFILE_FEATURES  # 12개

PAIR_FEATURES = [
    "lsm",
    "flow_continuity",
    "turn_balance",
    "preference_sync",
    "romantic_sync",
]


# ── 기본 XGBoost 파라미터 ─────────────────────────────────────────

def _make_regressor() -> "XGBRegressor":
    return XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric="rmse",
    )


# ── Fallback 휴리스틱 ─────────────────────────────────────────────
#
# 각 pair feature를 입력 피처로 근사:
#
# lsm            → talk_profile 스타일 유사도 주도
#                  (언어 스타일은 말하는 방식이 비슷할수록 높음)
#
# flow_continuity → embedding cosine 주도
#                  (공통 관심사/가치관이 있을수록 대화 흐름 좋음)
#
# turn_balance   → talk_profile 발화량 균형 주도
#                  (avg_turn_length가 비슷할수록 균형 좋음)
#
# preference_sync → embedding cosine 주도
#                  (취향은 semantic 공간에서 가까울수록 일치)
#
# romantic_sync  → emotional_expression 둘 다 주도
#                  (둘 다 감정 표현이 높아야 교환 가능)

def _profile_similarity(a_val: float, b_val: float) -> float:
    """두 값의 유사도. 0~1."""
    if a_val <= 0 and b_val <= 0:
        return 0.5  # 둘 다 없으면 중립
    if a_val <= 0 or b_val <= 0:
        return 0.3  # 한쪽만 없으면 낮음
    max_val = max(a_val, b_val)
    diff_ratio = abs(a_val - b_val) / max_val
    return float(np.clip(1.0 - diff_ratio, 0.0, 1.0))


def _fallback_predict(features: Dict[str, float]) -> Dict[str, float]:
    cosine          = features.get("cosine", 0.0)
    mean_abs_diff   = features.get("mean_abs_diff", 0.0)

    a_turn  = features.get("a_avg_turn_length", 0.0)
    b_turn  = features.get("b_avg_turn_length", 0.0)
    a_pace  = features.get("a_speech_pace", 0.0)
    b_pace  = features.get("b_speech_pace", 0.0)
    a_emo   = features.get("a_emotional_expression", 0.0)
    b_emo   = features.get("b_emotional_expression", 0.0)
    a_vocab = features.get("a_vocabulary_diversity", 0.0)
    b_vocab = features.get("b_vocabulary_diversity", 0.0)

    # lsm: 말하는 속도 + 발화량 유사도 (언어 스타일 동기화)
    turn_sim  = _profile_similarity(a_turn, b_turn)
    pace_sim  = _profile_similarity(a_pace, b_pace)
    vocab_sim = _profile_similarity(a_vocab, b_vocab)
    lsm = 0.4 * turn_sim + 0.3 * pace_sim + 0.3 * vocab_sim

    # flow_continuity: embedding cosine 주도
    # embedding이 가까울수록 공통 화제가 많아 흐름이 좋음
    flow_continuity = 0.75 * cosine + 0.25 * (1.0 - min(mean_abs_diff, 1.0))

    # turn_balance: 발화량(avg_turn_length) 균형
    turn_balance = _profile_similarity(a_turn, b_turn)

    # preference_sync: embedding cosine 주도
    preference_sync = 0.8 * cosine + 0.2 * _profile_similarity(a_vocab, b_vocab)

    # romantic_sync: 둘 다 emotional_expression이 높아야
    # 한쪽이 0이면 교환이 안 일어남 → min 기반
    emo_both = min(a_emo, b_emo) / 100.0 if max(a_emo, b_emo) > 0 else 0.0
    emo_sim  = _profile_similarity(a_emo, b_emo)
    romantic_sync = 0.6 * emo_both + 0.4 * emo_sim

    # 0~100 스케일로 변환
    return {
        "lsm":              round(float(np.clip(lsm, 0, 1)) * 100, 2),
        "flow_continuity":  round(float(np.clip(flow_continuity, 0, 1)) * 100, 2),
        "turn_balance":     round(float(np.clip(turn_balance, 0, 1)) * 100, 2),
        "preference_sync":  round(float(np.clip(preference_sync, 0, 1)) * 100, 2),
        "romantic_sync":    round(float(np.clip(romantic_sync, 0, 1)) * 100, 2),
    }


# ── 메인 클래스 ───────────────────────────────────────────────────

class PairFeatureModel:
    """
    Model 1: embedding + talk_profile → predicted pair features

    사용법:
        model = PairFeatureModel()
        model.load("gs://bucket/models/pair_feature/latest.pkl")

        predicted = model.predict(emb_interaction, talk_features)
        # → {"lsm": 72.0, "flow_continuity": 65.0, ...}
    """

    def __init__(self):
        # pair feature별 독립 회귀 모델
        self._models: Dict[str, Any] = {}
        self._loaded = False
        self._version = "baseline_heuristic"

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        """
        rows: List of {
            "cosine": ..., "l2": ..., "mean_abs_diff": ..., "std_abs_diff": ...,
            "a_avg_turn_length": ..., ..., "b_vocabulary_diversity": ...,
            "lsm": ..., "flow_continuity": ..., ...  ← 정답
        }

        대칭성 보장: 학습 전 (A,B)/(B,A) augmentation 필요
        train_pair_feature_model.py 에서 처리
        """
        if not HAS_XGB:
            raise ImportError("xgboost is required for training")

        X = np.array(
            [[row.get(f, 0.0) for f in INPUT_FEATURES] for row in rows],
            dtype=float,
        )

        for target in PAIR_FEATURES:
            y = np.array([row.get(target, 0.0) for row in rows], dtype=float)
            reg = _make_regressor()
            reg.fit(X, y)
            self._models[target] = reg

        self._loaded = True

    def predict(
        self,
        emb_interaction: Dict[str, float],
        talk_features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        emb_interaction: _embedding_interaction() 출력
        talk_features:   _talk_profile_features() 출력
        """
        features = {**emb_interaction, **talk_features}

        if not self._loaded or not self._models:
            return _fallback_predict(features)

        x = np.array(
            [[features.get(f, 0.0) for f in INPUT_FEATURES]],
            dtype=float,
        )

        result: Dict[str, float] = {}
        for target in PAIR_FEATURES:
            reg = self._models.get(target)
            if reg is None:
                # 해당 피처 모델만 없으면 fallback 값 사용
                fallback = _fallback_predict(features)
                result[target] = fallback[target]
            else:
                val = float(reg.predict(x)[0])
                result[target] = round(float(np.clip(val, 0.0, 100.0)), 2)

        return result

    def save(self, path: str) -> None:
        payload = (self._models, self._version)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        try:
            local_path = path
            if isinstance(path, str) and path.startswith("gs://"):
                local_path = self._download_from_gcs(path)

            with open(local_path, "rb") as f:
                payload = pickle.load(f)

            if isinstance(payload, tuple) and len(payload) == 2:
                self._models, self._version = payload
            else:
                self._models = payload
                self._version = "legacy"

            self._loaded = bool(self._models)

        except FileNotFoundError:
            print(f"⚠️ pair_feature model not found at {path}; using heuristic fallback")
            self._loaded = False
            self._version = "baseline_heuristic"
        except Exception as e:
            print(f"⚠️ pair_feature model load failed ({e}); using heuristic fallback")
            self._loaded = False
            self._version = "baseline_heuristic"

    def version(self) -> str:
        return self._version

    def _download_from_gcs(self, gs_path: str) -> str:
        if not HAS_FIREBASE:
            raise ImportError("firebase_admin is required for GCS download")

        parts = gs_path.replace("gs://", "", 1).split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        if not bucket_name or not blob_path:
            raise FileNotFoundError(f"Invalid GCS path: {gs_path}")

        get_firestore()
        bucket = storage.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        local_dir = Path("/tmp/pair_feature_models")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / os.path.basename(blob_path)

        blob.download_to_filename(str(local_path))
        return str(local_path)