#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_STAGE=run
export RL_PROFILE=pdms
export TOTAL_EPOCHS="${RUN_EPOCHS:-15}"
exec "${SCRIPT_DIR}/train_rl.sh" "$@"
