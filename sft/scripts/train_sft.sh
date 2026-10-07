#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to a directory containing Dataset.save_to_disk batches}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
CONFIG_FILE="${CONFIG_FILE:-configs/qwen2_5vl_3b_sft_config.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/zero2.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/open_r1/sft_qwen_2_5_ours_noCoT.py}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --config_file "${ACCELERATE_CONFIG}" \
  "${TRAIN_SCRIPT}" \
  --config "${CONFIG_FILE}" \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}"
