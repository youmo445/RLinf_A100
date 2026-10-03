#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=/data/zsq/RLinf_A100
CONFIG_NAME=wanmultiview_arxx5_newstructure_new_cam_packphone_0809_whole_task_stage_reward_grpo_pi05

cd "${RLINF_ROOT}"
source .venv/bin/activate

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/data/zsq/RoboTwin}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-ALOHA}

echo "Launching full-start PackPhone WM-RL with ordered single-frame rewards 0--4."
echo "Config: ${CONFIG_NAME}"
exec bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" "${ROBOT_PLATFORM}"
