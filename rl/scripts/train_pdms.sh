#!/usr/bin/env bash
# Reproduce the PDMS profile from scratch1_* in the original EasyR1 tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_PROFILE=pdms
export RL_STAGE=run
exec "${SCRIPT_DIR}/train_rl.sh" "$@"
