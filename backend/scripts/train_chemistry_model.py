import os
import time
from typing import Dict, List

from firebase_admin import firestore, storage
from sklearn.metrics import mean_squared_error, r2_score

from backend.services.chemistry_model import ChemistryModel, FEATURES
from backend.services.firestore import get_firestore


def now_ms() -> int:
    return int(time.time() * 1000)


def feature_row(feats: Dict[str, float]) -> Dict[str, float]:
    return {
        "turn": feats.get("turn", feats.get("turn_taking", 0)),
        "flow": feats.get("flow", feats.get("flow_continuity", 0)),
        "romantic": feats.get("romantic", feats.get("romantic_intent", 0)),
        "lsm": feats.get("lsm", feats.get("language_style_ma", 0)),
        "preference": feats.get("preference", feats.get("preference_sync", 0)),
        "pitch": feats.get("pitch", feats.get("voice_pitch", feats.get("voice pitch", 0))),
    }


def _label_from_go_no_go(go_no_go: Dict) -> int | None:
    if not isinstance(go_no_go, dict) or not go_no_go:
        return None
    values = [v for v in go_no_go.values() if isinstance(v, bool)]
    if not values:
        return None
    # 한 명만 응답해도 사용 (데이터 버리지 않음)
    if all(values):
        return 100       # 양쪽 go
    if all(v is False for v in values):
        return 0          # 양쪽 no
    return 50             # 한쪽만 go


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

    model = ChemistryModel()
    model.fit(rows)

    X = [[row.get(k, 0) for k in FEATURES] for row in rows]
    y = [row.get("label", 0) for row in rows]
    preds = [model.predict({k: row.get(k, 0) for k in FEATURES}) for row in rows]

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

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

    db.collection("model_versions").add(
        {
            "version": version,
            "trained_at": now_ms(),
            "sample_count": len(rows),
            "metrics": {"mse": mse, "r2": r2},
            "model_path": f"gs://{bucket_name}/{base_path}",
            "latest_path": f"gs://{bucket_name}/{latest_path}",
        }
    )

    print(f"Saved model {version} to {model_path}")
    print(f"samples={len(rows)} mse={mse:.4f} r2={r2:.4f}")


if __name__ == "__main__":
    main()
