# training/train_pair_feature_model.py - Model 1: pair feature predictor training
"""
Model 1 학습 스크립트
=====================
talk_history에서 데이터를 수집해 PairFeatureModel을 학습.

입력 피처 (12개):
  embedding interaction (4) + A/B talk_profile (8)

정답 레이블 (5개):
  실측 pair features (lsm, flow_continuity, turn_balance,
                      preference_sync, romantic_sync)

대칭성 augmentation:
  (A, B) 페어와 (B, A) 페어를 모두 학습 데이터로 사용.
  → A→B 예측값과 B→A 예측값이 동일하게 나오도록.

최소 데이터: 50건
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from firebase_admin import storage
from sklearn.metrics import mean_absolute_error, r2_score

from backend.services.firestore import get_firestore
from backend.services.pair_feature_model import (
    PairFeatureModel,
    INPUT_FEATURES,
    PAIR_FEATURES,
    EMB_FEATURES,
    PROFILE_FIELDS,
    _embedding_interaction,
    _talk_profile_features,
)


def now_ms() -> int:
    return int(time.time() * 1000)


# ── 데이터 수집 ───────────────────────────────────────────────────

def _get_user_snapshot(db, uid: str) -> Dict[str, Any]:
    """분석 시점의 embedding/talk_profile 스냅샷 반환."""
    snap = db.collection("users").document(uid).get()
    if not snap.exists:
        return {}
    return snap.to_dict() or {}


def _extract_embedding_vec(user_data: Dict) -> Optional[List[float]]:
    emb = user_data.get("embedding") or {}
    if isinstance(emb, dict):
        return emb.get("vector")
    return None


def _extract_talk_profile(user_data: Dict) -> Dict[str, float]:
    profile = user_data.get("talk_profile") or {}
    return {f: float(profile.get(f) or 0.0) for f in PROFILE_FIELDS}


def _build_input_row(
    vec_a: List[float],
    vec_b: List[float],
    profile_a: Dict[str, float],
    profile_b: Dict[str, float],
) -> Dict[str, float]:
    """12개 입력 피처 딕셔너리 구성."""
    emb = _embedding_interaction(vec_a, vec_b)
    talk = _talk_profile_features(profile_a, profile_b)
    return {**emb, **talk}


def _extract_pair_features(analysis: Dict) -> Optional[Dict[str, float]]:
    """talk_history.analysis.pair_features 추출."""
    pair = analysis.get("pair_features") or {}
    if not pair:
        return None
    result = {}
    for f in PAIR_FEATURES:
        val = pair.get(f)
        if val is None:
            return None  # 하나라도 없으면 이 행 제외
        result[f] = float(val)
    return result


def build_dataset(talks: List[Dict], db) -> List[Dict[str, Any]]:
    """
    talk_history 목록에서 학습 데이터 구성.

    각 talk당 두 행 생성 (A→B, B→A 대칭 augmentation).
    """
    rows = []
    skipped = {"no_analysis": 0, "no_pair_features": 0,
               "no_participants": 0, "no_embedding": 0,
               "no_talk_profile": 0}

    for talk in talks:
        analysis = talk.get("analysis") or {}
        if not analysis:
            skipped["no_analysis"] += 1
            continue

        pair_features = _extract_pair_features(analysis)
        if pair_features is None:
            skipped["no_pair_features"] += 1
            continue

        # 분석 시점 스냅샷: analysis에 저장된 embedding 사용
        # (현재 users.embedding은 피드백으로 업데이트됐을 수 있음)
        participants_raw = talk.get("participants") or {}
        if isinstance(participants_raw, dict):
            uid_a = participants_raw.get("user_a")
            uid_b = participants_raw.get("user_b")
        elif isinstance(participants_raw, list) and len(participants_raw) >= 2:
            uid_a, uid_b = participants_raw[0], participants_raw[1]
        else:
            skipped["no_participants"] += 1
            continue

        if not uid_a or not uid_b:
            skipped["no_participants"] += 1
            continue

        # personal 피처는 분석 시점 값 사용 (analysis.personal)
        personal = analysis.get("personal") or {}
        profile_a_stored = personal.get(uid_a) or {}
        profile_b_stored = personal.get(uid_b) or {}

        # embedding은 talk_history에 저장된 스냅샷 우선
        # 없으면 현재 users 컬렉션에서 가져옴
        vec_a = None
        vec_b = None

        # talk_history.analysis에 embedding 스냅샷이 있으면 사용
        # (없으면 현재 users에서 가져옴 — 약간의 노이즈 있지만 허용)
        user_a_data = _get_user_snapshot(db, uid_a)
        user_b_data = _get_user_snapshot(db, uid_b)

        vec_a = _extract_embedding_vec(user_a_data)
        vec_b = _extract_embedding_vec(user_b_data)

        if not vec_a or not vec_b:
            skipped["no_embedding"] += 1
            continue

        # talk_profile: analysis.personal 우선, 없으면 users 컬렉션
        profile_a = {
            f: float(profile_a_stored.get(f) or
                     _extract_talk_profile(user_a_data).get(f) or 0.0)
            for f in PROFILE_FIELDS
        }
        profile_b = {
            f: float(profile_b_stored.get(f) or
                     _extract_talk_profile(user_b_data).get(f) or 0.0)
            for f in PROFILE_FIELDS
        }

        has_profile_a = any(v > 0 for v in profile_a.values())
        has_profile_b = any(v > 0 for v in profile_b.values())
        if not has_profile_a or not has_profile_b:
            skipped["no_talk_profile"] += 1
            continue

        # ── 대칭 augmentation: A→B 와 B→A 두 행 모두 추가 ──
        for (va, vb, pa, pb) in [
            (vec_a, vec_b, profile_a, profile_b),  # A→B
            (vec_b, vec_a, profile_b, profile_a),  # B→A
        ]:
            input_row = _build_input_row(va, vb, pa, pb)
            row = {**input_row, **pair_features}
            rows.append(row)

    print(f"Skipped — {skipped}")
    return rows


# ── 평가 ─────────────────────────────────────────────────────────

def evaluate(model: PairFeatureModel, rows: List[Dict]) -> Dict[str, Any]:
    """피처별 MAE, R² 계산."""
    metrics = {}
    for target in PAIR_FEATURES:
        y_true = [row[target] for row in rows]
        y_pred = [
            model.predict(
                {k: row[k] for k in EMB_FEATURES},
                {k: row[k] for k in (
                    [f"a_{f}" for f in PROFILE_FIELDS] +
                    [f"b_{f}" for f in PROFILE_FIELDS]
                )},
            )[target]
            for row in rows
        ]
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        metrics[target] = {"mae": round(mae, 4), "r2": round(r2, 4)}
    return metrics


# ── 메인 ─────────────────────────────────────────────────────────

MIN_SAMPLES = 50


def main():
    db = get_firestore()

    print("📥 Loading talk_history...")
    talks = []
    for doc in db.collection("talk_history").stream():
        talk = doc.to_dict() or {}
        talk["id"] = doc.id
        talks.append(talk)
    print(f"  총 {len(talks)}건 로드")

    print("🔧 Building dataset...")
    rows = build_dataset(talks, db)
    print(f"  유효 샘플: {len(rows)}행 (augmentation 포함)")

    if len(rows) < MIN_SAMPLES:
        print(f"❌ 샘플 부족: {len(rows)} < {MIN_SAMPLES}. 학습 중단.")
        return

    # pair feature 분포 확인
    print("\nPair feature distributions:")
    for f in PAIR_FEATURES:
        vals = [r[f] for r in rows]
        print(f"  {f:18s}: mean={np.mean(vals):.1f}  std={np.std(vals):.1f}  "
              f"min={np.min(vals):.1f}  max={np.max(vals):.1f}")

    print("\n🏋️ Training Model 1...")
    model = PairFeatureModel()
    model.fit(rows)
    print("  학습 완료")

    # 평가 (train set — 데이터 적을 때는 별도 val set 없음)
    print("\n📊 Evaluation (train set):")
    metrics = evaluate(model, rows)
    for target, m in metrics.items():
        print(f"  {target:18s}: MAE={m['mae']:.2f}  R²={m['r2']:.4f}")

    # feature importance (XGBoost)
    print("\nFeature importances (per target):")
    for target in PAIR_FEATURES:
        reg = model._models.get(target)
        if reg is None:
            continue
        importances = reg.feature_importances_
        top = sorted(
            zip(INPUT_FEATURES, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:4]
        top_str = "  ".join(f"{f}:{v:.3f}" for f, v in top)
        print(f"  {target:18s}: {top_str}")

    # 버전 및 저장
    version = time.strftime("%Y%m%d%H%M%S")
    model.set_version(version) if hasattr(model, "set_version") else setattr(model, "_version", version)

    model_path = os.getenv("PAIR_FEATURE_MODEL_PATH", "/tmp/pair_feature_model.pkl")
    model.save(model_path)
    print(f"\n💾 Saved to {model_path}")

    # GCS 업로드
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not bucket_name:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET is required to upload model")

    bucket = storage.bucket(bucket_name)
    base_path = f"models/pair_feature/pair_feature_model_{version}.pkl"
    latest_path = "models/pair_feature/latest.pkl"

    bucket.blob(base_path).upload_from_filename(model_path)
    bucket.blob(latest_path).upload_from_filename(model_path)

    # model_versions 기록
    db.collection("model_versions").add({
        "model_type":   "pair_feature",
        "version":      version,
        "trained_at":   now_ms(),
        "sample_count": len(rows),
        "raw_talk_count": len(rows) // 2,  # augmentation 전 원본 수
        "metrics":      metrics,
        "model_path":   f"gs://{bucket_name}/{base_path}",
        "latest_path":  f"gs://{bucket_name}/{latest_path}",
    })

    print(f"✅ Model 1 v{version} uploaded")
    print(f"   samples={len(rows)} (원본 {len(rows)//2}건 × 2 augmentation)")
    for target, m in metrics.items():
        print(f"   {target}: MAE={m['mae']:.2f}  R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()