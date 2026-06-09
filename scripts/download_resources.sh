#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCE_DIR="${RESOURCE_DIR:-${ROOT_DIR}/resources}"
HF_HOME="${HF_HOME:-${RESOURCE_DIR}/hf_cache}"

mkdir -p "${RESOURCE_DIR}/models" "${RESOURCE_DIR}/datasets/iu_xray" "${RESOURCE_DIR}/external" "${HF_HOME}"

echo "[1/7] Cloning external scorer repositories"
if [ ! -d "${RESOURCE_DIR}/external/CheXbert/.git" ]; then
  git clone https://github.com/stanfordmlgroup/CheXbert.git "${RESOURCE_DIR}/external/CheXbert"
fi
if [ ! -d "${RESOURCE_DIR}/external/radgraph/.git" ]; then
  git clone https://github.com/Stanford-AIMI/radgraph.git "${RESOURCE_DIR}/external/radgraph"
fi

echo "[2/7] Downloading CheXbert checkpoint"
mkdir -p "${RESOURCE_DIR}/models/chexbert"
CHEXBERT_CKPT="${RESOURCE_DIR}/models/chexbert/chexbert.pth"
if [ ! -s "${CHEXBERT_CKPT}" ] || [ "$(wc -c < "${CHEXBERT_CKPT}")" -lt 1000000 ]; then
  rm -f "${CHEXBERT_CKPT}"
  python3 "${ROOT_DIR}/scripts/hf_download.py" StanfordAIMI/RRG_scorers \
    --filename chexbert.pth \
    --local_dir "${RESOURCE_DIR}/models/chexbert"
fi

echo "[3/7] Downloading RadGraph-XL scorer weights from Hugging Face"
python3 "${ROOT_DIR}/scripts/hf_download.py" StanfordAIMI/RRG_scorers \
  --filename modern-radgraph-xl.tar.gz \
  --local_dir "${RESOURCE_DIR}/models/radgraph"

echo "[4/7] Downloading DINOv2 vision model"
python3 "${ROOT_DIR}/scripts/hf_download.py" facebook/dinov2-base \
  --local_dir "${RESOURCE_DIR}/models/facebook_dinov2-base"

echo "[5/7] Downloading Qwen3-8B language model"
python3 "${ROOT_DIR}/scripts/hf_download.py" Qwen/Qwen3-8B \
  --local_dir "${RESOURCE_DIR}/models/Qwen3-8B"

echo "[6/7] Downloading IU-Xray/OpenI files"
pushd "${RESOURCE_DIR}/datasets/iu_xray" >/dev/null
if [ ! -s NLMCXR_png.tgz ]; then
  python3 "${ROOT_DIR}/scripts/range_download.py" \
    --url "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz" \
    --output NLMCXR_png.tgz \
    --workers 16 \
    --chunk_mb 1
fi
if [ ! -s NLMCXR_reports.tgz ]; then
  curl -L -C - "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz" -o NLMCXR_reports.tgz
fi
popd >/dev/null

echo "[7/7] Resource preparation complete"
echo "Resources are under: ${RESOURCE_DIR}"
echo "If any downloaded file is unexpectedly tiny, inspect it; Box/PhysioNet links may require browser/session access."
