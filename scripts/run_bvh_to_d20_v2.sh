#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BVH_FILE="${1:-${REPO_ROOT}/data/lafan/dance1_subject2.bvh}"

if [[ $# -gt 0 ]]; then
  shift
fi

cd "${REPO_ROOT}"

python scripts/bvh_to_robot.py \
  --bvh_file "${BVH_FILE}" \
  --robot d20_v2 \
  --debug_frame 10 \
  --rate_limit \
  "$@"
