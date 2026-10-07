import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import os
import json
import math
from typing import Any, Dict, List
from examples.reward_function.get_pdm import run_pdm_score

def parse_json(json_output):
    """
    parse output
    """
    try:
        lines = json_output.strip().splitlines()
        
        # remove Markdown JSON tag
        if lines[0].startswith("```json"):
            json_output = "\n".join(lines[1:])  # remove "```json"
        
        if "```" in json_output:
            json_output = json_output.split("```")[0]  # remove ```
        
        return json.loads(json_output)  # parse JSON
    except Exception:
        return None  # if fail, return None
    

def compute_iou(box1, box2):
    """Compute intersection-over-union for two ``[x1, y1, x2, y2]`` boxes."""
    # Unpack coordinates.
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Compute the intersection region.
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    # Compute intersection area.
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Compute union area.
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def compute_matching_cost(pred_boxes, gt_boxes):
    """Compute cost matrix for Hungarian matching."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    cost_matrix = np.zeros((n_pred, n_gt))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            cost_matrix[i, j] = 1 - compute_iou(pred_box, gt_box)
    return cost_matrix


def validate_box(box):
    """Return a valid box; malformed boxes map to a small dummy box."""
    if len(box) == 0:
        return (0, 0, 1, 1)
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        # raise ValueError(f"Invalid box coordinates: {box}")
        return (0, 0, 1, 1)  # Penalize malformed predictions through matching.
    return (x1, y1, x2, y2)

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match boxes with a strict one-to-one Hungarian assignment."""
    # Validate inputs.
    pred_boxes = [validate_box(b) for b in pred_boxes]
    gt_boxes = [validate_box(b) for b in gt_boxes]
    
    if not pred_boxes or not gt_boxes:
        return [], [], []

    # Build the cost matrix.
    cost_matrix = compute_matching_cost(pred_boxes, gt_boxes)
    
    # Add dummy nodes when the cost matrix is rectangular.
    n_pred, n_gt = cost_matrix.shape
    if n_pred > n_gt:
        # Add dummy columns for extra predictions.
        extended_matrix = np.hstack([cost_matrix, np.full((n_pred, n_pred - n_gt), 2.0)])
    else:
        extended_matrix = cost_matrix

    # Solve the assignment problem.
    row_ind, col_ind = linear_sum_assignment(extended_matrix)
    
    # Keep only valid assignments.
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        # Keep only assignments to original ground-truth boxes.
        if j < n_gt and (1 - cost_matrix[i, j]) >= iou_threshold:
            matched_pairs.append((i, j))
    
    # Build unmatched lists.
    all_preds = set(range(n_pred))
    matched_preds = set(p for p, _ in matched_pairs)
    unmatched_preds = list(all_preds - matched_preds)
    
    all_gts = set(range(n_gt))
    matched_gts = set(g for _, g in matched_pairs)
    unmatched_gts = list(all_gts - matched_gts)
    
    return matched_pairs, unmatched_preds, unmatched_gts


def parse_perception_answer(output_string):
    """Parse the perception section as a JSON list."""
    perception_ans = re.search(r'<perception>(.*?)</perception>', output_string, re.DOTALL)
    if not perception_ans:
        return [], [], False

    try:
        content = perception_ans.group(1).strip()
        data = parse_json(content)
        if not data:
            return [], [], False
        
        # The answer must be a list.
        if not isinstance(data, list):
            return [], [], False
            
        boxes = []
        labels = []
        flag = True
        
        for item in data:
            if isinstance(item, dict) and set(item.keys()) == {'bbox_2d', 'label'}:
                # Reject non-list boxes.
                if not isinstance(item['bbox_2d'], list):
                    flag = False
                    continue
                
                # An explicit empty list represents an empty scene.
                if len(item['bbox_2d']) == 0 and len(data) == 1:
                    return [], [], True
                
                # Every non-empty box has four coordinates.
                if len(item['bbox_2d']) != 4:
                    flag = False
                    continue
                
                # Normalize coordinates to integers.
                try:
                    box = [int(coord) for coord in item['bbox_2d']]
                    boxes.append(box)
                    labels.append(str(item.get('label', 'unknown')))
                except (ValueError, TypeError):
                    flag = False
            else:
                flag = False
        
        return boxes, labels, flag
    
    except json.JSONDecodeError:
        return [], [], False


def parse_prediction_answer(output_string):
    """Parse the prediction section as a JSON list."""
    prediction_ans = re.search(r'<prediction>(.*?)</prediction>', output_string, re.DOTALL)
    if not prediction_ans:
        return [], [], False

    try:
        content = prediction_ans.group(1).strip()
        data = parse_json(content)
        if not data:
            return [], [], False
        # The answer must be a list.
        if not isinstance(data, list):
            return [], [], False
            
        boxes = []
        states = []
        flag = True
        
        for item in data:
            if isinstance(item, dict) and set(item.keys()) == {'bbox_2d', 'future_position'}:
                # Reject non-list boxes.
                if not isinstance(item['bbox_2d'], list):
                    flag = False
                    continue
                                
                # An explicit empty list represents an empty scene.
                if len(item['bbox_2d']) == 0 and len(data) == 1:
                    return [], [], True
                
                # Every non-empty box has four coordinates.
                if len(item['bbox_2d']) != 4:
                    flag = False
                    continue
                
                # Normalize coordinates to integers.
                try:
                    box = [int(coord) for coord in item['bbox_2d']]
                    boxes.append(box)
                    states.append(str(item.get('future_position', 'unknown')))
                except (ValueError, TypeError):
                    flag = False
            else:
                flag = False
        
        return boxes, states, flag
    
    except json.JSONDecodeError:
        return [], [], False


def parse_planning_answer(output_string):
    """Parse eight planning waypoints with ``(x, y, heading)`` values."""
    default_traj = [[0.0, 0.0, 0.0] for _ in range(8)]
    planning_ans = re.search(r'<planning>(.*?)</planning>', output_string, re.DOTALL)
    if not planning_ans:
        return default_traj, False

    try:
        content = planning_ans.group(1).strip()
        parsed_data = parse_json(content)
        if not isinstance(parsed_data, list):
            return default_traj, False

        # Define the expected 0.5-second waypoint labels.
        time_slots = {
            "0.5s": None,
            "1.0s": None,
            "1.5s": None,
            "2.0s": None,
            "2.5s": None,
            "3.0s": None,
            "3.5s": None,
            "4.0s": None
        }

        for item in parsed_data:
            if not isinstance(item, dict):
                continue
            label = item.get("label", "")
            point = item.get("x_y_radian", [])
            if label not in time_slots:
                continue
            if (isinstance(point, list)
                and len(point) == 3
                and all(isinstance(c, (int, float)) for c in point)):
                if time_slots[label] is None:
                    time_slots[label] = [float(point[0]), float(point[1]), float(point[2])]

        pred_traj = []
        for label in ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "3.5s", "4.0s"]:
            if time_slots[label] is not None:
                pred_traj.append(time_slots[label])
            else:
                pred_traj.append([0.0, 0.0, 0.0])

        return pred_traj, True

    except Exception:
        return default_traj, False


def perception_reward(content, **kwargs):
    """Compute the perception reward for one response."""

    reward = 0.0
    pred_boxes, pred_labels, flag = parse_perception_answer(content)

    gt = kwargs['ground_truth']
    gt_perception = parse_json(gt["perception"])
    gt_boxes, gt_labels = [], []
    for item in gt_perception:
        if len(item['bbox_2d']) == 4:  # Ignore empty boxes.
            gt_boxes.append(item['bbox_2d'])
            gt_labels.append(item['label'])

    # 1. Check the answer format.
    if not flag:
        return 0.0

    # 2. Compare box counts.
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)       

    # Case 1: no ground-truth or predicted boxes.
    if num_gt == 0 and num_pred == 0:
        reward = 1.0

    # Case 2: false positives in an empty scene.
    elif num_gt == 0 and num_pred > 0:
        reward = 0.0

    # Case 3: false negatives.
    elif num_gt > 0 and num_pred == 0:
        reward = 0.0

    elif num_gt > 0 and num_pred > 0:
        matched_pairs, unmatched_preds, unmatched_gts = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
        tp = len(matched_pairs)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        
        # Compute mean IoU.
        avg_iou = sum(compute_iou(pred_boxes[i], gt_boxes[j]) for i,j in matched_pairs) / tp if tp > 0 else 0
        
        # Compute precision and recall.
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Combine localization quality with precision and recall.
        reward = avg_iou * (0.5 * precision + 0.5 * recall)
        
    return reward


def prediction_reward(content, **kwargs):
    """Compute the prediction reward for one response."""

    reward = 0.0
    
    pred_boxes, pred_states, flag = parse_prediction_answer(content)
    
    gt = kwargs['ground_truth']
    gt_prediction = parse_json(gt["prediction"])
    gt_boxes, gt_states = [], []
    for item in gt_prediction:
        if len(item['bbox_2d']) == 4:  # Ignore empty boxes.
            gt_boxes.append(item['bbox_2d'])
            gt_states.append(item['future_position'])
    
    if gt_boxes and gt_states:
        matched_pairs, unmatched_preds, unmatched_gts = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
        
        # Score the predicted future state.
        state_score = 0.0
        total_weight = 0.0
        for pred_idx, gt_idx in matched_pairs:
            iou = compute_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
            if pred_states[pred_idx].lower() == gt_states[gt_idx].lower():
                state_score += iou
            total_weight += iou
        
        state_acc = state_score / total_weight if total_weight > 0 else 0.0
        
        # Score detection quality.
        tp = len(matched_pairs)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        avg_iou = sum(compute_iou(pred_boxes[i], gt_boxes[j]) for i,j in matched_pairs) / tp if tp > 0 else 0
        detection_score = avg_iou * (0.5 * precision + 0.5 * recall)
        
        reward = state_acc * detection_score
        
    # An empty, well-formatted scene receives full credit.
    elif gt_boxes == [] and pred_boxes == [] and pred_states == [] and gt_states == [] and flag:
        reward = 1.0
    else:
        # Penalize false positives and malformed answers.
        reward = 0.0
        
    return reward


def endpoint_reward(pred_traj, gt_traj, delta=10.0, eta=0.2):
    """Return the paper's piecewise endpoint reward.

    The distance is the final-point 2D L1 distance.  The default ``delta`` and
    ``eta`` values match the Run-then-Walk implementation (10 and 0.2).
    """
    pred_endpoint = np.asarray(pred_traj[-1][:2], dtype=np.float32)
    gt_endpoint = np.asarray(gt_traj[-1][:2], dtype=np.float32)
    l1_dist = float(np.sum(np.abs(pred_endpoint - gt_endpoint)))
    max_steps = int(np.floor(1.0 / eta))
    decay_steps = int(np.floor((l1_dist - 2.0 * delta) / delta) + 1)
    decay_steps = min(max_steps, max(0, decay_steps))
    return max(0.0, 1.0 - eta * decay_steps)


def planning_reward(content, stage="run", **kwargs):
    """Compute stage-specific closed-loop planning rewards.

    ``run`` uses PDMS directly to discover progress-oriented modes. ``walk``
    replaces that objective with endpoint consistency plus NC/DAC safety,
    gated by NC, DAC, and TTC. The PDM metrics are returned in both modes for
    logging and validation.
    """
    stage = stage.lower()
    if stage not in {"run", "walk"}:
        raise ValueError(f"Unknown reward stage {stage!r}; expected 'run' or 'walk'.")
    pred_traj, flag = parse_planning_answer(content)
    frame_id = kwargs['ground_truth']["frame_id"]
    traj = [row[:2] for row in pred_traj]
    heading = [row[2] for row in pred_traj]

    args = {
        "traj": traj, 
        "token": frame_id,
        "heading": heading,
    }
    
    result = run_pdm_score(args)
    gt_traj_json = parse_json(kwargs['ground_truth']["planning"])
    if not isinstance(gt_traj_json, list):
        gt_traj_json = []
    gt_traj = []
    for item in gt_traj_json:
        gt_traj.append(item["x_y_radian"][:2])
    if not gt_traj:
        gt_traj = [[0.0, 0.0] for _ in range(8)]

    endpoint_r = endpoint_reward(traj, gt_traj)
    
    # Keep the open-loop trajectory error as a diagnostic metric.
    pred_array = np.array(traj)
    gt_array = np.array(gt_traj)  
    l2_loss = np.mean(np.sqrt(np.sum((pred_array - gt_array)**2, axis=-1)))        
    
    if stage == "run":
        # Run-GRPO uses the closed-loop PDMS score without an extra safety gate.
        planning_score = float(result["score"])
    else:
        nc = float(result["no_at_fault_collisions"])
        dac = float(result["drivable_area_compliance"])
        ttc = float(result["time_to_collision_within_bound"])
        safe = nc > 0.0 and dac > 0.0 and ttc > 0.0
        planning_score = (nc + dac + endpoint_r) if safe else 0.0
 
    
    # Return all metrics for logging.
    return {
        "l2_loss": l2_loss,
        "endpoint_reward": endpoint_r,
        "planning": planning_score,
        "PDMS": result['score'],
        "no_at_fault_collisions": result['no_at_fault_collisions'],
        "drivable_area_compliance": result['drivable_area_compliance'],
        "driving_direction_compliance": result['driving_direction_compliance'],
        "ego_progress": result['ego_progress'],
        "time_to_collision_within_bound": result['time_to_collision_within_bound'],
        "comfort": result['comfort'],
    }


def format_reward(content, **kwargs):
    """Check that all structured response sections are valid."""
    sections = [
        r'<perception>.*?</perception>',
        r'<prediction>.*?</prediction>',
        r'<planning>.*?</planning>'
    ]
    
    reward = 1.0
    
    # Check all required sections.
    for pattern in sections:
        if not re.search(pattern, content, re.DOTALL):
            reward = 0.0
            break
    
    # Check the parsed planning answer.
    if reward > 0:
        _, _, flag_perception = parse_perception_answer(content)
        _, _, flag_prediction = parse_prediction_answer(content)
        _, flag_planning = parse_planning_answer(content)
        if flag_perception and flag_prediction and flag_planning:
            reward = 1.0
        else:
            reward = 0.0
         
    return reward


# Register all reward functions.
reward_funcs_registry = {
    "perception": perception_reward,
    "prediction": prediction_reward,
    "planning": planning_reward,
    "format": format_reward,
}



def compute_score(
    reward_inputs: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
    stage: str = "run",
) -> List[Dict[str, float]]:
    """Compute the weighted reward for a batch of driving responses.

    ``stage`` is injected by EasyR1 through ``reward_function_kwargs`` and is
    deliberately the only reward schedule change between Run and Walk.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    
    if stage.lower() not in {"run", "walk"}:
        raise ValueError(f"Unknown reward stage {stage!r}; expected 'run' or 'walk'.")

    # Keep the structured auxiliary rewards fixed across both stages.
    default_weights = {"format": 0.1, "perception": 0.1, "prediction": 0.1, "planning": 0.7}
    weights = weights or default_weights
    
    scores = []
    for reward_input in reward_inputs:
        # Extract one response and its ground truth.
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        
        # Compute each module score.
        tmp_scores = {}
        for module, func in reward_funcs_registry.items():
            result = func(response, ground_truth=ground_truth, stage=stage)
            if module == "planning" and isinstance(result, dict):
                tmp_scores.update(result)  # Flatten planning metrics for logging.
            else:
                tmp_scores[module] = result

        # A malformed structured response receives zero total reward.
        if tmp_scores.get("format", 0.0) == 0.0:
            zero_scores = {k: 0.0 for k in tmp_scores}
            zero_scores["overall"] = 0.0
            scores.append(zero_scores)
            continue
        
        # Compute the weighted total.
        overall_score = 0.0
        for module, weight in weights.items():
            overall_score += tmp_scores.get(module, 0.0) * weight
        
        scores.append({
            "overall": overall_score,
            **tmp_scores
        })
    return scores
