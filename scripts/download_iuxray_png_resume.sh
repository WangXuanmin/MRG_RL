#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${RAW_DIR:-${ROOT_DIR}/resources/datasets/iu_xray}"
URL="https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
OUT="${RAW_DIR}/NLMCXR_png.tgz"

mkdir -p "${RAW_DIR}"

attempt=1
while true; do
  echo "Attempt ${attempt}: resuming ${OUT}"
  set +e
  curl -L -C - \
    --connect-timeout 30 \
    --speed-limit 50000 \
    --speed-time 60 \
    "${URL}" \
    -o "${OUT}"
  status=$?
  set -e
  if [ "${status}" -eq 0 ]; then
    echo "IU-Xray PNG download complete: ${OUT}"
    break
  fi
  echo "curl exited with status ${status}; retrying in 5 seconds"
  attempt=$((attempt + 1))
  sleep 5
done
