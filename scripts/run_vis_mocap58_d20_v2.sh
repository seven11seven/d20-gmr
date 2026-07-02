#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BVH_FILE="${1:-${REPO_ROOT}/data/mocap58/Xsens1/5.bvh}"

if [[ $# -gt 0 ]]; then
  shift
fi

cd "${REPO_ROOT}"

python scripts/vis_mocap58_bvh.py \
  --bvh_file "${BVH_FILE}" \
  --robot d20_v2 \
  "$@"
