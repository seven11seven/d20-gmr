#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BVH_FILE="${1:-${REPO_ROOT}/data/mocap58/Xsens2/7.bvh}"

if [[ $# -gt 0 ]]; then
  shift
fi

# Edit this block to tune the first-frame IK initial guess.
# Root quaternion uses MuJoCo/GMR order: W X Y Z.
INITIAL_ROOT_POS=(0.0 0.0 1.0)
INITIAL_ROOT_QUAT=(0.5 0.5 0.5 0.5)

# Joint values are in degrees. Add/remove entries as needed.
INITIAL_JOINTS=(
  "waist_yaw_joint=0.0"
  "waist_roll_joint=0.0"
  "waist_pitch_joint=0.0"

  "left_shoulder_pitch_joint=0.0"
  "left_shoulder_roll_joint=0.0"
  "left_shoulder_yaw_joint=0.0"
  "left_elbow_pitch_joint=0.0"
  "left_wrist_roll_joint=0.0"

  "right_shoulder_pitch_joint=0.0"
  "right_shoulder_roll_joint=0.0"
  "right_shoulder_yaw_joint=0.0"
  "right_elbow_pitch_joint=0.0"
  "right_wrist_roll_joint=0.0"

  "left_hip_pitch_joint=0.0"
  "left_hip_roll_joint=0.0"
  "left_hip_yaw_joint=90.0"
  "left_knee_pitch_joint=0.0"
  "left_ankle_pitch_joint=0.0"
  "left_ankle_roll_joint=0.0"

  "right_hip_pitch_joint=0.0"
  "right_hip_roll_joint=0.0"
  "right_hip_yaw_joint=0.0"
  "right_knee_pitch_joint=0.0"
  "right_ankle_pitch_joint=0.0"
  "right_ankle_roll_joint=0.0"
)

INITIAL_JOINT_ARGS=()
for JOINT_VALUE in "${INITIAL_JOINTS[@]}"; do
  INITIAL_JOINT_ARGS+=(--initial_joint "${JOINT_VALUE}")
done

cd "${REPO_ROOT}"

python scripts/bvh_to_robot.py \
  --bvh_file "${BVH_FILE}" \
  --format mocap58 \
  --robot d20_v2 \
  --initial_root_pos "${INITIAL_ROOT_POS[@]}" \
  --initial_root_quat "${INITIAL_ROOT_QUAT[@]}" \
  "${INITIAL_JOINT_ARGS[@]}" \
  --rate_limit \
  --debug_frame 4 \
  "$@"
