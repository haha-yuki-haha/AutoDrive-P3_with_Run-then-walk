#!/usr/bin/env python3
"""Calculate official NavSim PDMS for one generated prediction JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "rl"
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))


METRIC_FIELDS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
]


def load_prediction(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("A prediction list must contain exactly one row for this test")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("Prediction JSON must contain an object")
    token = value.get("frame_id", value.get("token"))
    if token is None and value.get("id") is not None:
        # The original final_results.json stores a scene/frame id. The NavSim
        # metric cache is keyed by its final frame-token component.
        token = str(value["id"]).rsplit("_", 1)[-1]
    if token is None:
        raise ValueError("Prediction JSON is missing frame_id/token/id")
    trajectory = value.get("trajectory", value.get("planning"))
    if trajectory is None and isinstance(value.get("model_output"), str):
        match = re.search(r"<planning>(.*?)</planning>", value["model_output"], re.DOTALL | re.IGNORECASE)
        if match:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", match.group(1).strip(), flags=re.IGNORECASE)
            try:
                planning = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in model_output planning section: {exc}") from exc
            slots = {f"{index * 0.5:.1f}s": None for index in range(1, 9)}
            for item in planning:
                if isinstance(item, dict) and item.get("label") in slots:
                    point = item.get("x_y_radian")
                    if isinstance(point, list) and len(point) == 3:
                        slots[item["label"]] = point
            trajectory = [slots[f"{index * 0.5:.1f}s"] or [0.0, 0.0, 0.0] for index in range(1, 9)]
    if not isinstance(trajectory, list) or len(trajectory) != 8:
        raise ValueError("Prediction trajectory/planning output must contain exactly 8 points")
    normalized: List[List[float]] = []
    for index, point in enumerate(trajectory):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"Trajectory point {index} must be [x, y, heading]")
        try:
            row = [float(item) for item in point]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Trajectory point {index} is not numeric") from exc
        if not all(math.isfinite(item) for item in row):
            raise ValueError(f"Trajectory point {index} contains NaN or infinity")
        normalized.append(row)
    return {"token": str(token), "trajectory": normalized}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-json", required=True, type=Path)
    parser.add_argument("--metric-cache", type=Path,
                        help="NavSim metric-cache directory (required unless --dry-run)")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate trajectory and write a no-score report")
    args = parser.parse_args()
    prediction = load_prediction(args.prediction_json)
    if args.dry_run:
        result: Dict[str, Any] = {
            "token": prediction["token"],
            "valid": True,
            "status": "dry_run",
            "score": None,
        }
    else:
        if not args.metric_cache:
            parser.error("--metric-cache is required unless --dry-run is set")
        os.environ["NAVSIM_METRIC_CACHE"] = str(args.metric_cache)
        try:
            from examples.reward_function.get_pdm import run_pdm_score
        except ImportError as exc:
            raise RuntimeError(
                "NavSim dependencies are unavailable. Install rl/requirements.txt or use --dry-run."
            ) from exc
        result = run_pdm_score({
            "token": prediction["token"],
            "traj": [point[:2] for point in prediction["trajectory"]],
            "heading": [point[2] for point in prediction["trajectory"]],
        })
        result = {key: (float(value) if hasattr(value, "item") else value)
                  for key, value in result.items()}
    output = {
        "schema_version": 1,
        "token": prediction["token"],
        "trajectory": prediction["trajectory"],
        "metrics": {key: result.get(key) for key in METRIC_FIELDS},
        "valid": bool(result.get("valid", True)),
        "status": result.get("status", "scored"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))
    print(f"Wrote PDMS report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
