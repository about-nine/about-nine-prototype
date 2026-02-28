#!/usr/bin/env bash
set -euo pipefail

echo "[startup] starting web + analysis worker"

worker_disabled="${ANALYSIS_WORKER_DISABLED:-}"
if [[ "${worker_disabled,,}" == "1" || "${worker_disabled,,}" == "true" || "${worker_disabled,,}" == "yes" ]]; then
  echo "[startup] analysis worker disabled via ANALYSIS_WORKER_DISABLED"
  worker_pid=""
else
  python -m backend.scripts.analysis_worker &
  worker_pid="$!"
  echo "[startup] analysis worker pid=${worker_pid}"
fi

python -m backend.app &
web_pid="$!"
echo "[startup] web pid=${web_pid}"

cleanup() {
  echo "[startup] shutting down..."
  if [[ -n "${worker_pid}" ]]; then
    kill "${worker_pid}" 2>/dev/null || true
  fi
  kill "${web_pid}" 2>/dev/null || true
}

trap cleanup SIGINT SIGTERM EXIT

wait "${web_pid}"
