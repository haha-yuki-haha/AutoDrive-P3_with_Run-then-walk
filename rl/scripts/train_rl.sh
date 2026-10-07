#!/usr/bin/env bash
# Public EasyR1 launcher. Defaults mirror the two released AutoDrive commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH=".:third_party/nuplan-devkit:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5-VL SFT/checkpoint directory}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to the processed RL train dataset}"
VAL_DATA="${VAL_DATA:?Set VAL_DATA to the processed RL validation dataset}"
export NAVSIM_METRIC_CACHE="${NAVSIM_METRIC_CACHE:?Set NAVSIM_METRIC_CACHE to the NavSim metric-cache directory}"

RL_PROFILE="${RL_PROFILE:-pdms}"
case "${RL_PROFILE}" in
  pdms)
    RL_STAGE="${RL_STAGE:-run}"
    DEFAULT_REWARD="examples/reward_function/autodrive_navsim.py:compute_score"
    DEFAULT_NAME="scratch1_navsim_qwen2_5_vl_3b_lr1e-6_epoch15_format_pdms_Stage2_noCoT_command_epoch5_kl1e-3"
    ;;
  dac_ttc)
    RL_STAGE="${RL_STAGE:-walk}"
    DEFAULT_REWARD="examples/reward_function/autodrive_navsim.py:compute_score"
    DEFAULT_NAME="scratch2_navsim_qwen2_5_vl_3b_lr1e-6_epoch15_format_softerrrrr2_dac_ttc_Stage2_noCoT_command_epoch5_kl1e-3"
    ;;
  *) echo "RL_PROFILE must be pdms or dac_ttc (got ${RL_PROFILE})" >&2; exit 2 ;;
esac
case "${RL_STAGE}" in run|walk) ;; *) echo "RL_STAGE must be run or walk" >&2; exit 2 ;; esac

CONFIG_FILE="${CONFIG_FILE:-examples/AutoDrive/config.yaml}"
QUESTION_TEMPLATE="${QUESTION_TEMPLATE:-examples/AutoDrive/prompt_noCoT.txt}"
REWARD_FUNCTION="${REWARD_FUNCTION:-${DEFAULT_REWARD}}"
VIDEO_KEY="${VIDEO_KEY:-video}"
VIDEO_SHAPE="${VIDEO_SHAPE:-[4,3,168,672]}"
OUTPUT_NAME="${OUTPUT_NAME:-${DEFAULT_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${OUTPUT_NAME}}"
N_GPUS="${N_GPUS:-8}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# These values intentionally match both original EasyR1 launch scripts.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MIN_PIXELS="${MIN_PIXELS:-112896}"
MAX_PIXELS="${MAX_PIXELS:-451584}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-5632}"
LEARNING_RATE="${LEARNING_RATE:-0.000001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-32}"
ACTOR_UPDATE_MICROBATCH="${ACTOR_UPDATE_MICROBATCH:-8}"
ACTOR_EXPERIENCE_MICROBATCH="${ACTOR_EXPERIENCE_MICROBATCH:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_FREQ="${VAL_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-200}"
SAVE_LIMIT="${SAVE_LIMIT:-12}"
KL_COEF="${KL_COEF:-0.001}"
LOGGER="${LOGGER:-['console','swanlab']}"

OVERRIDES=(
  "config=${CONFIG_FILE}"
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${VAL_DATA}"
  "data.question_template=${QUESTION_TEMPLATE}"
  "data.video_key=${VIDEO_KEY}"
  "data.video_shape=${VIDEO_SHAPE}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.min_pixels=${MIN_PIXELS}"
  "data.max_pixels=${MAX_PIXELS}"
  "data.val_batch_size=${VAL_BATCH_SIZE}"
  "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
  "worker.rollout.n=${ROLLOUT_N}"
  "worker.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
  "worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  "worker.actor.model.model_path=${MODEL_PATH}"
  "worker.actor.optim.lr=${LEARNING_RATE}"
  "worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}"
  "worker.actor.micro_batch_size_per_device_for_update=${ACTOR_UPDATE_MICROBATCH}"
  "worker.actor.micro_batch_size_per_device_for_experience=${ACTOR_EXPERIENCE_MICROBATCH}"
  "worker.actor.offload.offload_params=true"
  "worker.actor.offload.offload_optimizer=true"
  "worker.reward.reward_function=${REWARD_FUNCTION}"
  "worker.reward.reward_function_kwargs.stage=${RL_STAGE}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.val_freq=${VAL_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.save_limit=${SAVE_LIMIT}"
  "trainer.experiment_name=${OUTPUT_NAME}"
  "trainer.logger=${LOGGER}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "algorithm.kl_coef=${KL_COEF}"
  "trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH}"
)

[[ -n "${LOAD_CHECKPOINT_PATH:-}" ]] && OVERRIDES+=("trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH}")
[[ -n "${MAX_STEPS:-}" ]] && OVERRIDES+=("trainer.max_steps=${MAX_STEPS}")

echo "[train_rl] profile=${RL_PROFILE} stage=${RL_STAGE} gpus=${N_GPUS} epochs=${TOTAL_EPOCHS}"
echo "[train_rl] model=${MODEL_PATH}"
echo "[train_rl] train=${TRAIN_DATA}"
echo "[train_rl] val=${VAL_DATA}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python3 -m verl.trainer.main "${OVERRIDES[@]}" "$@"
