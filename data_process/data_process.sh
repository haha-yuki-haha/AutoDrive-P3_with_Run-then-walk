#!/usr/bin/env bash
# Build the Qwen2.5-VL NAVSIM data used by the public SFT/RL release.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ITEM="all"
X="672"
Y="168"
WORKERS="${DATA_PROCESS_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: bash data_process.sh [options]

Required environment variables:
  NAVSIM_ROOT       Raw NAVSIM root containing navsim_logs/ and sensor_blobs/
  DATA_ROOT         Directory where every generated artifact is written
  ANNOTATION_ROOT   Directory containing transfusion_outputs_{train,test}/
Optional:
  AUTODRIVE_P3_ROOT Directory containing replacement solution JSONs

Options:
  --item train|test|all   Split to build (default: all)
  --x WIDTH               Stitched image width (default: 672)
  --y HEIGHT              Stitched image height (default: 168)
  --workers N             Parallel workers (default: 8)
  --python PATH           Python executable (default: python)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --item) ITEM="${2:?missing value for --item}"; shift 2 ;;
    --x) X="${2:?missing value for --x}"; shift 2 ;;
    --y) Y="${2:?missing value for --y}"; shift 2 ;;
    --workers) WORKERS="${2:?missing value for --workers}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${ITEM}" in train|test|all) ;; *) echo "--item must be train, test, or all" >&2; exit 2 ;; esac
[[ "${X}" =~ ^[0-9]+$ && "${Y}" =~ ^[0-9]+$ && "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--x, --y, and --workers must be positive integers" >&2; exit 2;
}

: "${NAVSIM_ROOT:?Set NAVSIM_ROOT to the raw NAVSIM directory}"
: "${DATA_ROOT:?Set DATA_ROOT to the generated-data directory}"
: "${ANNOTATION_ROOT:?Set ANNOTATION_ROOT to the reasoning-trace directory}"
[[ -d "${NAVSIM_ROOT}/navsim_logs" ]] || { echo "Missing ${NAVSIM_ROOT}/navsim_logs" >&2; exit 1; }
[[ -d "${NAVSIM_ROOT}/sensor_blobs" ]] || { echo "Missing ${NAVSIM_ROOT}/sensor_blobs" >&2; exit 1; }
for split in train test; do
  [[ -d "${ANNOTATION_ROOT}/transfusion_outputs_${split}" ]] || {
    echo "Missing ${ANNOTATION_ROOT}/transfusion_outputs_${split}" >&2; exit 1;
  }
done
SOLUTION_ROOT="${AUTODRIVE_P3_ROOT:-${SCRIPT_DIR}/AutoDrive_P3}"
for split in train test; do
  [[ -f "${SOLUTION_ROOT}/1_RL_${split}_${X}_${Y}.json" ]] || {
    echo "Missing ${SOLUTION_ROOT}/1_RL_${split}_${X}_${Y}.json" >&2; exit 1;
  }
done
mkdir -p "${DATA_ROOT}"
export DATA_PROCESS_WORKERS="${WORKERS}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "[data_process] NAVSIM_ROOT=${NAVSIM_ROOT}"
echo "[data_process] DATA_ROOT=${DATA_ROOT}"
echo "[data_process] ANNOTATION_ROOT=${ANNOTATION_ROOT}"
echo "[data_process] item=${ITEM} resolution=${X}x${Y} workers=${WORKERS}"

# Scene filtering writes one pickle for each split. It is kept as one step
# because the NAVSIM filters are defined in the two YAML files.
"${PYTHON_BIN}" 0_get_useful_scenes.py

run_split() {
  local split="$1"
  [[ "${ITEM}" == all || "${ITEM}" == "${split}" ]] || return 0
  "${PYTHON_BIN}" 1_get_json.py --item "${split}"
  "${PYTHON_BIN}" 2_0_get_all_frames.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 2_1_get_all_videos.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 3_0_get_traj_heading.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 3_1_get_v.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 4_0_merge_json.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 4_1_process_json.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 4_2_merge_cot.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 4_3_add_solution.py --item "${split}" --x "${X}" --y "${Y}"
  "${PYTHON_BIN}" 4_4_build_hf_cache.py --item "${split}" --x "${X}" --y "${Y}"
}

run_split train
run_split test
echo "[data_process] complete: ${DATA_ROOT}/4_data_${X}_${Y}"
