#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-${REPO_ROOT}/data/lafan}"
OUT_DIR="${2:-${REPO_ROOT}/retargeting_data/d20_v2/lafan}"
OVERRIDE=0

if [[ "${3:-}" == "--override" ]]; then
  OVERRIDE=1
fi

SRC_DIR="$(realpath "${SRC_DIR}")"
OUT_DIR="$(realpath -m "${OUT_DIR}")"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source directory does not exist: ${SRC_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

mapfile -d '' BVH_FILES < <(find "${SRC_DIR}" -type f -name '*.bvh' -print0 | sort -z)
TOTAL="${#BVH_FILES[@]}"

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "No BVH files found in ${SRC_DIR}" >&2
  exit 1
fi

echo "Batch retargeting ${TOTAL} BVH files"
echo "  Source: ${SRC_DIR}"
echo "  Output: ${OUT_DIR}"
echo "  Robot:  d20_v2"

INDEX=0
for BVH_FILE in "${BVH_FILES[@]}"; do
  INDEX=$((INDEX + 1))
  REL_PATH="${BVH_FILE#"${SRC_DIR}/"}"
  SAVE_PATH="${OUT_DIR}/${REL_PATH%.bvh}.pkl"

  if [[ -f "${SAVE_PATH}" && "${OVERRIDE}" -eq 0 ]]; then
    echo "[${INDEX}/${TOTAL}] skip existing: ${SAVE_PATH}"
    continue
  fi

  mkdir -p "$(dirname "${SAVE_PATH}")"
  echo "[${INDEX}/${TOTAL}] retarget: ${REL_PATH}"

  python scripts/bvh_to_robot.py \
    --bvh_file "${BVH_FILE}" \
    --robot d20_v2 \
    --no_viewer \
    --save_path "${SAVE_PATH}"
done

echo "Done. Saved retargeted motions to ${OUT_DIR}"
