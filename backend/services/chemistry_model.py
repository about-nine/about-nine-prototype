# services/chemistry_model.py - chemistry model loading and scoring
import os
import numpy as np
import pickle
from pathlib import Path
from firebase_admin import storage

from backend.services.firestore import get_firestore
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = ["lsm", "flow_continuity", "turn_balance", "preference_sync", "romantic_sync"]

# 데이터 없을 때 사용하는 수동 가중치
# 직관적 중요도 기반 (romantic/preference가 가장 직접적인 신호)
FALLBACK_WEIGHTS = {
    "romantic_sync":   0.30,
    "preference_sync": 0.25,
    "lsm":             0.20,
    "flow_continuity": 0.15,
    "turn_balance":    0.10,
}

class ChemistryModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=50,       # 100 → 50 (데이터 적을 때 과적합 방지)
            max_depth=2,           # 3 → 2 (피처 5개에 depth 3은 과함)
            learning_rate=0.05,    # 0.1 → 0.05
            subsample=0.8,         # 추가: 행 샘플링
            colsample_bytree=1.0,  # 피처가 5개라 전체 사용
            scale_pos_weight=4,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self._loaded = False
        self._version = "baseline"
        self._weights = FALLBACK_WEIGHTS.copy()  # feature_importances로 업데이트됨

    def fit(self, df):
        if isinstance(df, list):
            X = []
            y = []
            for row in df:
                X.append([row.get(k, 0) for k in FEATURES])
                y.append(row.get("label", 0))
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=float)
        else:
            X = df[FEATURES].values
            y = df["label"].values

        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

        # 학습 후 feature_importances → _weights 업데이트
        # 이후 fallback에서도 실제 데이터 기반 가중치 사용
        importances = self.model.feature_importances_
        total = importances.sum()
        if total > 0:
            self._weights = {
                f: float(importances[i] / total)
                for i, f in enumerate(FEATURES)
            }

    def predict(self, feats: dict) -> float:
        """
        입력: pair_features dict
        {"lsm": 72.0, "flow_continuity": 65.0, "turn_balance": 58.0,
        "preference_sync": 80.0, "romantic_sync": 45.0}
        출력: 0~100 chemistry score
        """
        values = [float(feats.get(k, 0)) for k in FEATURES]
        x = np.array([values], dtype=float)

        if not self._loaded:
            # fallback: 가중 평균
            weighted = sum(
                self._weights.get(f, 0) * v
                for f, v in zip(FEATURES, values)
            )
            return float(np.clip(weighted, 0, 100))

        # 모델 로드된 경우: Go/Go 확률 → 0~100
        xs = self.scaler.transform(x)
        return float(self.model.predict_proba(xs)[0, 1] * 100)

    def save(self, path):
        # weights도 함께 저장해서 load 시 fallback에 반영
        pickle.dump(
            (self.scaler, self.model, self._version, self._weights),
            open(path, "wb")
        )

    def load(self, path):
        try:
            local_path = path
            if isinstance(path, str) and path.startswith("gs://"):
                local_path = self._download_from_gcs(path)

            payload = pickle.load(open(local_path, "rb"))

            if isinstance(payload, tuple) and len(payload) == 4:
                # 현재 포맷 (scaler, model, version, weights)
                self.scaler, self.model, self._version, self._weights = payload
            elif isinstance(payload, tuple) and len(payload) == 3:
                # 이전 포맷 (scaler, model, version) — weights는 기본값 유지
                self.scaler, self.model, self._version = payload
            else:
                self.scaler, self.model = payload
                self._version = "legacy"

            self._loaded = True

        except FileNotFoundError:
            print(f"⚠️ chemistry model not found at {path}; using fallback scoring")
            self._loaded = False
            self._version = "baseline"
        except Exception as e:
            print(f"⚠️ chemistry model load failed ({e}); using fallback scoring")
            self._loaded = False
            self._version = "baseline"

    def set_version(self, version: str):
        self._version = version

    def version(self):
        return self._version

    def weights(self):
        return self._weights.copy()

    def _download_from_gcs(self, gs_path: str) -> str:
        if not gs_path.startswith("gs://"):
            return gs_path

        get_firestore()

        parts = gs_path.replace("gs://", "", 1).split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        if not bucket_name or not blob_path:
            raise FileNotFoundError(f"Invalid GCS path: {gs_path}")

        bucket = storage.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        local_dir = Path("/tmp/chemistry_models")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / os.path.basename(blob_path)

        blob.download_to_filename(str(local_path))
        return str(local_path)
