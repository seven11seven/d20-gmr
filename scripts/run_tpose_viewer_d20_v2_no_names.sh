#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BVH_FILE="${1:-${REPO_ROOT}/data/lafan/dance1_subject2.bvh}"

cd "${REPO_ROOT}"

python scripts/tpose_viewer.py \
  --bvh_file "${BVH_FILE}" \
  --robot d20_v2 \
  --no_show_body_names
