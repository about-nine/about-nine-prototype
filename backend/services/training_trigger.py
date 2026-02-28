# services/training_trigger.py - 학습 자동 트리거
import os
from typing import Optional

PAIR_FIRST_TRAIN_THRESHOLD = 50  # Model 1 트리거 (회귀, 복잡)
CHEMISTRY_FIRST_TRAIN_THRESHOLD = 30       # Model 2 트리거 (분류, 단순)
RETRAIN_INTERVAL = 20

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
REGION = os.getenv("CLOUD_RUN_REGION", "asia-northeast3")
CHEMISTRY_JOB = os.getenv("CHEMISTRY_TRAIN_JOB", "train-chemistry-model")
PAIR_FEATURE_JOB = os.getenv("PAIR_FEATURE_TRAIN_JOB", "train-pair-feature-model")


def _get_last_trained_count(db, model_type: str) -> int:
    """model_versions에서 마지막 학습 시점의 sample_count 반환."""
    try:
        docs = (
            db.collection("model_versions")
            .where("model_type", "==", model_type)
            .order_by("trained_at", direction="DESCENDING")
            .limit(1)
            .stream()
        )
        for doc in docs:
            data = doc.to_dict() or {}
            # chemistry는 raw count, pair_feature는 augmentation 전 원본 수
            return int(data.get("raw_talk_count") or data.get("sample_count") or 0)
    except Exception as e:
        print(f"⚠️ [trigger] Failed to get last trained count for {model_type}: {e}")
    return 0


def _count_labeled_talks(db) -> int:
    """label이 있는 talk_history 수."""
    try:
        result = (
            db.collection("talk_history")
            .where("label", "in", [0, 1])
            .count()
            .get()
        )
        return result[0][0].value
    except Exception as e:
        print(f"⚠️ [trigger] Failed to count labeled talks: {e}")
        return 0


def _trigger_cloud_run_job(job_name: str) -> bool:
    """Cloud Run Job 실행."""
    if not PROJECT_ID:
        print(f"⚠️ [trigger] GOOGLE_CLOUD_PROJECT not set, skipping {job_name}")
        return False
    try:
        from google.cloud import run_v2
        client = run_v2.JobsClient()
        job_path = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{job_name}"
        client.run_job(name=job_path)
        print(f"✅ [trigger] Triggered Cloud Run Job: {job_name}")
        return True
    except Exception as e:
        print(f"❌ [trigger] Failed to trigger {job_name}: {e}")
        return False


def should_train(labeled_count: int, last_trained_count: int, threshold: int) -> bool:
    if last_trained_count == 0:
        return labeled_count >= threshold
    return labeled_count >= last_trained_count + RETRAIN_INTERVAL


def check_and_trigger_training(db) -> None:
    """분석 완료 후 호출. 조건 충족 시 학습 Job 트리거."""
    labeled_count = _count_labeled_talks(db)
    print(f"📊 [trigger] Labeled talks: {labeled_count}")

    # Chemistry
    last_chemistry = _get_last_trained_count(db, "chemistry")
    if should_train(labeled_count, last_chemistry, CHEMISTRY_FIRST_TRAIN_THRESHOLD):
        _trigger_cloud_run_job(CHEMISTRY_JOB)

    # Pair Feature
    last_pair = _get_last_trained_count(db, "pair_feature")
    if should_train(labeled_count, last_pair, PAIR_FIRST_TRAIN_THRESHOLD):
        _trigger_cloud_run_job(PAIR_FEATURE_JOB)