#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_STAGE=walk
export RL_PROFILE=dac_ttc
export TOTAL_EPOCHS="${WALK_EPOCHS:-15}"
exec "${SCRIPT_DIR}/train_rl.sh" "$@"
