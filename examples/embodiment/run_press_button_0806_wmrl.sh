#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/data/zsq/RLinf_A100}"
CONFIG_NAME="wanmultiview_arxx5_newstructure_new_cam_press_button_0806_grpo_pi05"
PYTHON_BIN="${PYTHON_BIN:-${RLINF_ROOT}/.venv/bin/python}"
DIFFSYNTH_PATH="${DIFFSYNTH_PATH:-/data/zsq/diffsynth-studio-wujie-ppu}"

: "${WM_CKPT:?Set WM_CKPT to the transferred new-data WM .safetensors file}"
: "${VLA_CKPT:?Set VLA_CKPT to the transferred previous 6-task PyTorch checkpoint directory}"

if [[ ! -f "${WM_CKPT}" ]]; then
  echo "WM_CKPT is not a file: ${WM_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${VLA_CKPT}" ]]; then
  echo "VLA_CKPT is not a directory: ${VLA_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${VLA_CKPT}/model.safetensors" ]]; then
  echo "Missing ${VLA_CKPT}/model.safetensors" >&2
  exit 1
fi
if ! find "${VLA_CKPT}" -name norm_stats.json -print -quit | grep -q .; then
  echo "No norm_stats.json found under ${VLA_CKPT}" >&2
  exit 1
fi
if [[ -n "${RESUME_DIR:-}" && ! -d "${RESUME_DIR}" ]]; then
  echo "RESUME_DIR is not a directory: ${RESUME_DIR}" >&2
  exit 1
fi

export DIFFSYNTH_PATH
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export REPO_PATH="${RLINF_ROOT}"
export PYTHONPATH="${DIFFSYNTH_PATH}:${RLINF_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"

LOG_DIR="${LOG_DIR:-${RLINF_ROOT}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}}"
mkdir -p "${LOG_DIR}"

cmd=(
  "${PYTHON_BIN}"
  "${EMBODIED_PATH}/train_embodied_agent.py"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "env.train.vace_model_path=${WM_CKPT}"
  "env.eval.vace_model_path=${WM_CKPT}"
  "rollout.model.model_path=${VLA_CKPT}"
  "actor.model.model_path=${VLA_CKPT}"
)
if [[ -n "${RESUME_DIR:-}" ]]; then
  cmd+=("runner.resume_dir=${RESUME_DIR}")
fi
cmd+=("$@")

printf '[launch] %q ' "${cmd[@]}"
printf '\n'
printf '[log] %s\n' "${LOG_DIR}"
"${cmd[@]}" 2>&1 | tee "${LOG_DIR}/run_embodiment.log"
