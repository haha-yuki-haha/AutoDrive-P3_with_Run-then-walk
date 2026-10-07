#!/usr/bin/env python3
"""Generate AutoDrive predictions with the original vLLM result schema.

The real inference path builds the Qwen2.5-VL chat prompt, sends a video
tensor to vLLM, and stores one row in the released ``final_results.json``
format. ``--dry-run`` uses one of the bundled AutoDrive-P3 annotations and
does not import GPU dependencies.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "rl"
BUILTIN_SOLUTION_JSON = ROOT / "data_process/AutoDrive_P3/1_RL_test_672_168.json"
DEFAULT_PROMPT = ROOT / "rl/examples/AutoDrive/prompt_noCoT.txt"


def _annotation_response(solution: Dict[str, str]) -> str:
    """Wrap the original fenced solution strings exactly as model output."""
    return "\n".join(
        f"<{section}>{solution[section]}</{section}>"
        for section in ("perception", "prediction", "planning")
    )


def _load_builtin_fixture() -> Tuple[Dict[str, Any], str]:
    data = json.loads(BUILTIN_SOLUTION_JSON.read_text(encoding="utf-8"))
    _, sample = next(iter(data.items()))
    return sample, _annotation_response(sample["solution"])


def _json_block(response: str, name: str) -> Any:
    match = re.search(rf"<{name}>(.*?)</{name}>", response, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Response does not contain <{name}>...</{name}>")
    text = match.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in <{name}>: {exc}") from exc


def parse_response(response: str) -> Tuple[List[Any], List[Any], List[List[float]]]:
    """Validate the structured response and return its eight planning points."""
    perception = _json_block(response, "perception")
    prediction = _json_block(response, "prediction")
    planning_raw = _json_block(response, "planning")
    if not isinstance(perception, list) or not isinstance(prediction, list):
        raise ValueError("Perception and prediction sections must be JSON lists")
    if not isinstance(planning_raw, list):
        raise ValueError("Planning section must be a JSON list")

    slots: Dict[str, Optional[List[float]]] = {f"{i * 0.5:.1f}s": None for i in range(1, 9)}
    for item in planning_raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        point = item.get("x_y_radian")
        if label in slots and isinstance(point, list) and len(point) == 3:
            try:
                slots[label] = [float(value) for value in point]
            except (TypeError, ValueError):
                continue
    planning = [slots[f"{i * 0.5:.1f}s"] or [0.0, 0.0, 0.0] for i in range(1, 9)]
    return perception, prediction, planning


def _sample_value(sample: Dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = sample.get(name)
        if value is not None:
            return str(value)
    return default


def build_prompt(template_path: Path, sample: Dict[str, Any]) -> str:
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "[VEHICLE_SPEED]": _sample_value(sample, "ego_state", "ego_dynamic_state", default="unknown"),
        "[Ego_Future_Action]": _sample_value(sample, "driving_command", default="unknown"),
        "[HISTORICAL_TRAJECTORY]": _sample_value(
            sample, "historical_x_y_angle", "historical_x_y_heading", "historical_trajectory", default="[]"
        ),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _solution_fields(sample: Dict[str, Any]) -> Dict[str, str]:
    solution = sample.get("solution")
    if isinstance(solution, dict):
        return {key: str(solution.get(key, "")) for key in ("perception", "prediction", "planning")}
    return {
        "perception": str(sample.get("solution_perception", "")),
        "prediction": str(sample.get("solution_prediction", "")),
        "planning": str(sample.get("solution_planning", "")),
    }


def make_result_row(sample: Dict[str, Any], prompt: str, model_output: str) -> Dict[str, Any]:
    """Create one row in the released final-results schema."""
    ground_truth = _solution_fields(sample)
    return {
        "id": str(sample.get("id", sample.get("frame_id", "unknown"))),
        "input_text": prompt,
        "model_output": model_output,
        "gt_perception": ground_truth["perception"],
        "gt_prediction": ground_truth["prediction"],
        "gt_planning": ground_truth["planning"],
        "ego_state": _sample_value(sample, "ego_state", "ego_dynamic_state"),
        "driving_command": _sample_value(sample, "driving_command"),
        "historical_x_y_angle": _sample_value(
            sample, "historical_x_y_angle", "historical_x_y_heading", "historical_trajectory"
        ),
    }


def _load_dataset_sample(dataset_dir: Path, index: int) -> Dict[str, Any]:
    try:
        from datasets import DatasetDict, concatenate_datasets, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install datasets to use --dataset-dir, or use --dry-run") from exc

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    paths = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if (dataset_dir / "dataset_info.json").exists() or (dataset_dir / "state.json").exists():
        paths = [dataset_dir]
    if not paths:
        raise FileNotFoundError(f"No Hugging Face dataset found under {dataset_dir}")

    datasets = []
    for path in paths:
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            datasets.extend(loaded.values())
        else:
            datasets.append(loaded)
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"--index {index} is outside dataset size {len(dataset)}")
    return dict(dataset[index])


def _video_tensor(sample: Dict[str, Any], video_shape: Sequence[int]):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Install numpy and torch for vLLM video generation") from exc
    video = sample.get("video")
    if not isinstance(video, (bytes, bytearray)):
        raise ValueError("Dataset sample must contain raw uint8 TCHW `video` bytes")
    raw = np.frombuffer(video, dtype=np.uint8)
    expected = 1
    for dimension in video_shape:
        expected *= dimension
    if raw.size != expected:
        raise ValueError(f"Video has {raw.size} bytes; expected {expected} for shape {tuple(video_shape)}")
    return torch.from_numpy(raw.reshape(tuple(video_shape)).copy()).to(torch.float32)


def generate_with_vllm(
    model_path: Path,
    sample: Dict[str, Any],
    prompt_template: Path,
    video_shape: Sequence[int],
    tensor_parallel_size: int,
    max_prompt_length: int,
    response_length: int,
    min_pixels: int,
    max_pixels: int,
    gpu_memory_utilization: float,
    skip_ulysses_patch: bool,
) -> Tuple[str, str]:
    """Run the same tokenizer/processor/vLLM path as the original script."""
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Install torch, transformers, and vllm for model generation") from exc

    if not skip_ulysses_patch:
        if str(RL_ROOT) not in sys.path:
            sys.path.insert(0, str(RL_ROOT))
        try:
            from verl.models.monkey_patch import apply_ulysses_patch
            apply_ulysses_patch("qwen2_5_vl")
        except Exception as exc:
            raise RuntimeError(
                "The Qwen Ulysses patch could not be applied. Use --skip-ulysses-patch "
                "only when your Transformers/vLLM stack does not need it."
            ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels
    )
    video_tensor = _video_tensor(sample, video_shape)
    prompt = build_prompt(prompt_template, sample)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": video_tensor, "fps": 2.0,
             "min_pixels": min_pixels, "max_pixels": max_pixels},
        ],
    }]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    raw_prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)

    llm_engine = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_prompt_length + response_length,
        max_num_batched_tokens=8192,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"video": 1},
    )
    sampling_params = SamplingParams(
        max_tokens=response_length,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    requests = [{
        "prompt_token_ids": raw_prompt_ids,
        "multi_modal_data": {"video": [video_tensor]},
    }]
    result = llm_engine.generate(requests, sampling_params=sampling_params, use_tqdm=False)[0]
    generated_ids = result.outputs[0].token_ids
    input_text = tokenizer.decode(raw_prompt_ids, skip_special_tokens=True)
    return tokenizer.decode(generated_ids, skip_special_tokens=True), input_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--video-shape", default="[4,3,168,672]")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--response-length", type=int, default=2048)
    parser.add_argument("--min-pixels", type=int, default=24 * 6 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=4 * 24 * 6 * 28 * 28)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--skip-ulysses-patch", action="store_true")
    parser.add_argument("--response", type=Path, help="Use a saved model response for schema testing")
    parser.add_argument("--dry-run", action="store_true", help="Use one bundled AutoDrive-P3 annotation")
    args = parser.parse_args()

    if args.response and args.dry_run:
        parser.error("choose only one of --response and --dry-run")
    if args.model_path and not args.dataset_dir:
        parser.error("--dataset-dir is required with --model-path")
    if args.dataset_dir and not args.model_path:
        parser.error("--model-path is required with --dataset-dir")

    if args.model_path:
        try:
            video_shape = tuple(int(value) for value in ast.literal_eval(args.video_shape))
        except (TypeError, ValueError, SyntaxError) as exc:
            parser.error(f"invalid --video-shape: {exc}")
        if len(video_shape) != 4 or any(value <= 0 for value in video_shape):
            parser.error("--video-shape must contain four positive integers")
        sample = _load_dataset_sample(args.dataset_dir, args.index)
        model_output, prompt = generate_with_vllm(
            args.model_path, sample, args.prompt_template, video_shape,
            args.tensor_parallel_size, args.max_prompt_length, args.response_length,
            args.min_pixels, args.max_pixels, args.gpu_memory_utilization,
            args.skip_ulysses_patch,
        )
    else:
        sample, builtin_output = _load_builtin_fixture()
        prompt = build_prompt(args.prompt_template, sample)
        model_output = args.response.read_text(encoding="utf-8") if args.response else builtin_output

    parse_response(model_output)
    row = make_result_row(sample, prompt, model_output)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote original-format prediction JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
