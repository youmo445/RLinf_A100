#!/usr/bin/env bash
set -euo pipefail

# Parallel PI05+WAN-WM batch rollout using the initial/base policy weights
# from the YAML. This script intentionally does not pass --policy-full-weights.

RLINF_ROOT="${RLINF_ROOT:-/data/zsq/RLinf}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-wanmultiview_arxx5_grpo_pi05}"
CONFIG_DIR="${CONFIG_DIR:-${RLINF_ROOT}/examples/embodiment/config}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${RLINF_ROOT}/examples/embodiment/run_pi05_wm_batch_autoregressive_eval.py}"

GPUS="${GPUS:-2,3,4,5,6,7}"
EPISODE_START="${EPISODE_START:-0}"
EPISODE_END="${EPISODE_END:-auto}"
MAX_EPISODES="${MAX_EPISODES:-}"
NUM_POLICY_CHUNKS="${NUM_POLICY_CHUNKS:-}"
POLICY_MODE="${POLICY_MODE:-eval}"
FPS="${FPS:-15}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SAVE_RAW_VIDEO="${SAVE_RAW_VIDEO:-0}"
SAVE_ACTION_VIDEO="${SAVE_ACTION_VIDEO:-0}"
SAVE_ACTION_PLOTS="${SAVE_ACTION_PLOTS:-0}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${RLINF_ROOT}/examples/embodiment/pi05_wm_batch_rollouts_initial/${CONFIG_NAME}_${RUN_TAG}}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${ROLLOUT_SCRIPT}" ]]; then
  echo "Rollout script not found: ${ROLLOUT_SCRIPT}" >&2
  exit 1
fi

detect_dataset_len() {
  "${PYTHON_BIN}" - <<PY
import glob
import os
from hydra import compose, initialize_config_dir

config_dir = "${CONFIG_DIR}"
config_name = "${CONFIG_NAME}".removesuffix(".yaml")
os.environ.setdefault("EMBODIED_PATH", os.path.dirname(config_dir))
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name=config_name)
path = cfg.env.train.initial_image_path
print(len(sorted(glob.glob(os.path.join(path, "*.npy")))))
PY
}

DATASET_LEN="$(detect_dataset_len)"
if [[ "${EPISODE_END}" == "auto" ]]; then
  EPISODE_END="${DATASET_LEN}"
fi
if [[ -n "${MAX_EPISODES}" ]]; then
  requested_end=$((EPISODE_START + MAX_EPISODES))
  if (( requested_end < EPISODE_END )); then
    EPISODE_END="${requested_end}"
  fi
fi

TOTAL=$((EPISODE_END - EPISODE_START))
if (( TOTAL <= 0 )); then
  echo "No episodes selected: start=${EPISODE_START}, end=${EPISODE_END}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
SHARD_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "RLINF_ROOT=${RLINF_ROOT}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "DATASET_LEN=${DATASET_LEN}"
echo "EPISODE_START=${EPISODE_START}"
echo "EPISODE_END=${EPISODE_END}"
echo "TOTAL=${TOTAL}"
echo "GPUS=${GPUS}"
echo "SHARD_SIZE=${SHARD_SIZE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "Using YAML initial policy weights. No --policy-full-weights is passed."

pids=()
shard_dirs=()

for i in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$i]}"
  shard_start=$((EPISODE_START + i * SHARD_SIZE))
  shard_end=$((shard_start + SHARD_SIZE))
  if (( shard_start >= EPISODE_END )); then
    continue
  fi
  if (( shard_end > EPISODE_END )); then
    shard_end="${EPISODE_END}"
  fi

  shard_dir="${OUT_ROOT}/shard_${i}_gpu_${gpu}"
  mkdir -p "${shard_dir}"
  shard_dirs+=("${shard_dir}")

  cmd=(
    "${PYTHON_BIN}" "${ROLLOUT_SCRIPT}"
    --config-dir "${CONFIG_DIR}"
    --config-name "${CONFIG_NAME}"
    --episode-start "${shard_start}"
    --episode-end "${shard_end}"
    --output-dir "${shard_dir}"
    --policy-mode "${POLICY_MODE}"
    --fps "${FPS}"
  )

  if [[ -n "${NUM_POLICY_CHUNKS}" ]]; then
    cmd+=(--num-policy-chunks "${NUM_POLICY_CHUNKS}")
  fi
  if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
    cmd+=(--stop-on-success)
  fi
  if [[ "${SAVE_RAW_VIDEO}" == "1" ]]; then
    cmd+=(--save-raw-video)
  fi
  if [[ "${SAVE_ACTION_VIDEO}" == "1" ]]; then
    cmd+=(--save-action-video)
  fi
  if [[ "${SAVE_ACTION_PLOTS}" == "1" ]]; then
    cmd+=(--save-action-plots)
  fi

  log_file="${LOG_DIR}/shard_${i}_gpu_${gpu}_${shard_start}_${shard_end}.log"
  echo "Starting shard ${i}: gpu=${gpu}, episodes=[${shard_start}, ${shard_end}), log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON_BIN}" - <<PY
import csv
import glob
import json
from pathlib import Path

out_root = Path("${OUT_ROOT}")
rows = []
for csv_path in sorted(out_root.glob("shard_*/summary.csv")):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["shard"] = csv_path.parent.name
            rows.append(row)

rows.sort(key=lambda r: int(r["episode_index"]))
summary_csv = out_root / "summary_all.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["episode_index", "success", "max_reward", "num_generated_frames", "annotated_video", "task", "shard"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

successes = sum(str(row.get("success", "")).lower() in ("true", "1") for row in rows)
summary = {
    "num_episodes": len(rows),
    "num_success": successes,
    "success_rate": successes / len(rows) if rows else 0.0,
    "summary_csv": str(summary_csv),
    "shards": sorted(str(p) for p in out_root.glob("shard_*")),
}
summary_json = out_root / "summary_all.json"
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"combined summary csv: {summary_csv}")
print(f"combined summary json: {summary_json}")
print(f"success: {successes}/{len(rows)} = {summary['success_rate']:.4f}")
PY

echo "Logs: ${LOG_DIR}"
exit "${status}"
