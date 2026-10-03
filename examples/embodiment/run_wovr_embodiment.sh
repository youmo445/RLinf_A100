#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFSYNTH_PATH=${DIFFSYNTH_PATH:-/data/zsq/diffsynth-studio-wovr-ppu}

exec bash "${SCRIPT_DIR}/run_embodiment.sh" "$@"
