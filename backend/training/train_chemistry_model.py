import os
import time
from typing import Dict, List

from firebase_admin import firestore, storage
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)

from backend.services.chemistry_model import ChemistryModel, FEATURES
from backend.services.firestore import get_firestore


def now_ms() -> int:
    return int(time.time() * 1000)


def feature_row(feats: Dict[str, float]) -> Dict[str, float]:
    return {
        "turn":       feats.get("turn",       feats.get("turn_taking", 0)),
        "flow":       feats.get("flow",       feats.get("flow_continuity", 0)),
        "romantic":   feats.get("romantic",   feats.get("romantic_intent", 0)),
        "lsm":        feats.get("lsm",        feats.get("language_style_ma", 0)),
        "preference": feats.get("preference", feats.get("preference_sync", 0)),
    }


def _label_from_go_no_go(go_no_go: Dict) -> int | None:
    if not isinstance(go_no_go, dict) or not go_no_go:
        return None
    values = [v for v in go_no_go.values() if isinstance(v, bool)]
    if len(values) < 2:
        return None

    # 둘 다 Go → 1 (케미 있음)
    # 한쪽이라도 No → 0 (케미 없음)
    # 둘 다 응답한 페어만 사용
    return 1 if all(values) else 0


def build_dataset(talks: List[Dict]) -> List[Dict]:
    rows = []
    for t in talks:
        label = _label_from_go_no_go(t.get("go_no_go"))
        if label is None:
            label = t.get("label")
        if label is None:
            continue
        feats = ((t.get("analysis") or {}).get("features") or {})
        if not feats:
            continue
        row = feature_row(feats)
        row["label"] = label
        rows.append(row)
    return rows


def main():
    db = get_firestore()
    talks = []
    for doc in db.collection("talk_history").stream():
        talk = doc.to_dict() or {}
        talk["id"] = doc.id
        talks.append(talk)

    rows = build_dataset(talks)
    if not rows:
        print("No labeled data found.")
        return

    # 클래스 분포 확인
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"Class distribution — positive(Go/Go): {n_pos}, negative: {n_neg}")

    if n_pos == 0:
        print("No positive samples (Go/Go). Cannot train.")
        return

    # 클래스 불균형 보정
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = ChemistryModel()
    model.model.set_params(scale_pos_weight=scale_pos_weight)
    model.fit(rows)

    # 평가 (분류 모델에 맞는 지표로 변경)
    X = [{k: row.get(k, 0) for k in FEATURES} for row in rows]
    y_true = [row["label"] for row in rows]
    y_pred_proba = [model.predict(x) / 100 for x in X]   # 0~100 → 0~1
    y_pred_label = [1 if p >= 0.5 else 0 for p in y_pred_proba]

    accuracy = accuracy_score(y_true, y_pred_label)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.0  # 클래스가 하나뿐일 때

    print(f"\n{classification_report(y_true, y_pred_label)}")
    print(f"accuracy={accuracy:.4f}  auc={auc:.4f}")

    # feature importances 출력
    print("\nFeature importances:")
    for feat, weight in sorted(
        model.weights().items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(weight * 40)
        print(f"  {feat:12s}: {weight:.4f}  {bar}")

    version = time.strftime("%Y%m%d%H%M%S")
    model.set_version(version)

    model_path = os.getenv("CHEMISTRY_MODEL_PATH", "/tmp/chemistry_model.pkl")
    model.save(model_path)

    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not bucket_name:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET is required to upload model")

    bucket = storage.bucket(bucket_name)
    base_path = f"models/chemistry/chemistry_model_{version}.pkl"
    latest_path = "models/chemistry/latest.pkl"

    blob = bucket.blob(base_path)
    blob.upload_from_filename(model_path)

    latest_blob = bucket.blob(latest_path)
    latest_blob.upload_from_filename(model_path)

    db.collection("model_versions").add({
        "version":            version,
        "trained_at":         now_ms(),
        "sample_count":       len(rows),
        "class_distribution": {"positive": n_pos, "negative": n_neg},
        "scale_pos_weight":   scale_pos_weight,
        "metrics": {
            "accuracy": accuracy,
            "auc":      auc,
        },
        "feature_importances": model.weights(),
        "model_path":   f"gs://{bucket_name}/{base_path}",
        "latest_path":  f"gs://{bucket_name}/{latest_path}",
    })

    print(f"\nSaved model {version} to {model_path}")
    print(f"samples={len(rows)}  accuracy={accuracy:.4f}  auc={auc:.4f}")


if __name__ == "__main__":
    main()
