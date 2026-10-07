# Inference and Evaluation

The scripts in this directory keep the original AutoDrive evaluation
interfaces. They do not download data or checkpoints and write output only to
the path given by the user.

## 1. Generate one concrete prediction JSON

Run a dependency-free parser smoke test first:

```bash
python inference/generate_json.py \
  --dry-run \
  --output-json outputs/final_results.json
```

The dry run uses one original annotation from
`data_process/AutoDrive_P3/1_RL_test_672_168.json`; it preserves the real
fenced JSON response, including all eight planning points. The output is a
one-row `final_results.json` with the same fields as the original
`eval_traj_val_gpus_vllm_1.py`: `id`, `input_text`, `model_output`,
`gt_perception`, `gt_prediction`, `gt_planning`, `ego_state`,
`driving_command`, and `historical_x_y_angle`.

To generate a prediction from a real Qwen2.5-VL model and one sample in a
Hugging Face dataset, provide both paths:

```bash
python inference/generate_json.py \
  --model-path ../models/Qwen2.5-VL-3B-Instruct-SFT \
  --dataset-dir ../data/RL_processed_dataset_test \
  --output-json outputs/final_results.json \
  --index 0
```

The dataset sample must contain raw uint8 TCHW `video` bytes and the metadata
used by the prompt. Raw bytes use `[4,3,168,672]` by default; override with
`--video-shape` when needed. The model path uses the same tokenizer,
processor, Ulysses patch, vLLM request, greedy sampling, and result fields as
the original validation script. Use `--skip-ulysses-patch` only when the
installed Transformers/vLLM stack does not require that patch. The original
two-way tensor parallel default is `--tensor-parallel-size 2`; set it to `1`
on a single-GPU machine.

## 2. Calculate PDMS

With a NavSim metric cache available:

```bash
python inference/calculate_pdms.py \
  --prediction-json outputs/final_results.json \
  --metric-cache ../data/navsim/metric_cache \
  --output-json outputs/autodrive_pdms.json
```

The PDMS helper accepts both this original result list and a compact
`frame_id`/`trajectory` object. It extracts the planning section from
`model_output` when needed and uses the final component of the original `id`
as the NavSim metric-cache token. The result records PDMS and the component
metrics returned by the official PDM scorer: NC, DAC, driving-direction
compliance, ego progress, TTC, and comfort. Use `--dry-run` to validate JSON
shape and produce a report without importing NavSim or requiring a cache.
