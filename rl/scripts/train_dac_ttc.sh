#!/usr/bin/env bash
# Reproduce the DAC/TTC profile from scratch2_* in the original EasyR1 tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_PROFILE=dac_ttc
export RL_STAGE=walk
exec "${SCRIPT_DIR}/train_rl.sh" "$@"
