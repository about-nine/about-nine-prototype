#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-about-nine-prototype-46a2c}"
REGION="${REGION:-asia-northeast3}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-train-chemistry@${PROJECT_ID}.iam.gserviceaccount.com}"
BUCKET="${FIREBASE_STORAGE_BUCKET:-about-nine-prototype-46a2c.firebasestorage.app}"

# ==========================================
# 공통 함수: 이미지 빌드 + Cloud Run Job 배포
# ==========================================
deploy_train_job() {
  local job_name="$1"
  local train_script="$2"
  local image="gcr.io/${PROJECT_ID}/${job_name}:latest"

  echo ""
  echo "========================================"
  echo "  ${job_name}"
  echo "  script: ${train_script}"
  echo "========================================"

  echo "1️⃣  Building image: ${image}"
  gcloud builds submit \
    --config backend/cloudbuild_train_job.yaml \
    --substitutions _IMAGE="${image}",_TRAIN_SCRIPT="${train_script}" \
    .

  echo "2️⃣  Deploying Cloud Run Job: ${job_name}"
  gcloud run jobs deploy "${job_name}" \
    --image "${image}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --service-account "${SERVICE_ACCOUNT}" \
    --set-env-vars "FIREBASE_STORAGE_BUCKET=${BUCKET}" \
    --max-retries 1 \
    --task-timeout 1800 \
    --memory 1Gi

  echo "✅ ${job_name} deployed"
}

# ==========================================
# Model 1: Pair Feature Predictor
#   입력: embedding interaction(4) + talk_profile(8)
#   출력: pair features 5개 예측
#   트리거: 이벤트 기반 (training_trigger.py)
# ==========================================
deploy_train_job \
  "train-pair-feature-model" \
  "backend/training/train_pair_feature_model.py"

# ==========================================
# Model 2: Chemistry Predictor
#   입력: pair features 5개
#   출력: P(go/go) → chemistry score
#   트리거: 이벤트 기반 (training_trigger.py)
# ==========================================
deploy_train_job \
  "train-chemistry-model" \
  "backend/training/train_chemistry_model.py"

echo ""
echo "✅ All training jobs deployed"
echo ""
echo "수동 실행:"
echo "  gcloud run jobs execute train-pair-feature-model --region ${REGION} --project ${PROJECT_ID}"
echo "  gcloud run jobs execute train-chemistry-model    --region ${REGION} --project ${PROJECT_ID}"