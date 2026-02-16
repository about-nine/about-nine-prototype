#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-about-nine-prototype-46a2c}"
REGION="${REGION:-asia-northeast3}"
JOB_NAME="${JOB_NAME:-train-chemistry-model}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-train-chemistry@${PROJECT_ID}.iam.gserviceaccount.com}"
IMAGE="${IMAGE:-gcr.io/${PROJECT_ID}/${JOB_NAME}:latest}"
SCHEDULE="${SCHEDULE:-0 12 * * *}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
BUCKET="${FIREBASE_STORAGE_BUCKET:-about-nine-prototype-46a2c.firebasestorage.app}"

# ==========================================
# Step 1: Docker 이미지 빌드 (Cloud Build)
# ==========================================
echo "1️⃣ Building image: ${IMAGE}"
gcloud builds submit --project "${PROJECT_ID}" \
  --config backend/cloudbuild_train.yaml \
  --substitutions _IMAGE="${IMAGE}" \
  .
# cloudbuild_train.yaml이 Dockerfile.train으로 빌드 → GCR에 push

# ==========================================
# Step 2: Cloud Run Job 생성/업데이트
# ==========================================
echo "2️⃣ Deploying Cloud Run Job: ${JOB_NAME}"
gcloud run jobs deploy "${JOB_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "FIREBASE_STORAGE_BUCKET=${BUCKET}" \
  --max-retries 1 \
  --task-timeout 1800 \
  --memory 1Gi

# ==========================================
# Step 3: Cloud Scheduler로 매일 자동 실행
# ==========================================
echo "3️⃣ Setting up daily schedule"
SCHEDULER_JOB="${JOB_NAME}-schedule"
SCHEDULER_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

# 먼저 생성 시도, 이미 있으면 업데이트
if gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
  --location "${REGION}" \
  --project "${PROJECT_ID}" &>/dev/null; then

  gcloud scheduler jobs update http "${SCHEDULER_JOB}" \
    --location "${REGION}" \
    --project "${PROJECT_ID}" \
    --schedule "${SCHEDULE}" \
    --time-zone "${TIMEZONE}" \
    --uri "${SCHEDULER_URI}" \
    --http-method POST \
    --oauth-service-account-email "${SERVICE_ACCOUNT}" \
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" \
    --quiet
else
  gcloud scheduler jobs create http "${SCHEDULER_JOB}" \
    --location "${REGION}" \
    --project "${PROJECT_ID}" \
    --schedule "${SCHEDULE}" \
    --time-zone "${TIMEZONE}" \
    --uri "${SCHEDULER_URI}" \
    --http-method POST \
    --oauth-service-account-email "${SERVICE_ACCOUNT}" \
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" \
    --quiet
fi

echo ""
echo "✅ Done!"
echo "   Job:       ${JOB_NAME}"
echo "   Schedule:  ${SCHEDULE} (${TIMEZONE})"
echo "   Model:     gs://${BUCKET}/models/chemistry/latest.pkl"
echo ""
echo "   수동 실행: gcloud run jobs execute ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID}"