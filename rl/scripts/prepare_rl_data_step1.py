import json
import os
import time
import argparse
from typing import Dict, List, Any
import random

import torch
import torchvision
from datasets import Dataset, Features, Value
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration defaults.
W = int(36 * 28 * 1)  # 1008
H = int(9 * 28 * 1)   # 252

input_json = ""
output_json_path = ""
ROOT_VIDEO_PATH = ""
BATCH_SIZE = 1000  # Number of items handled per batch.
MAX_WORKERS = 48   # Number of parallel workers.

def load_video_from_path(video_path: str) -> torch.Tensor:
    """Load the complete video and return a tensor."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    try:
        # Read the complete video.
        video, _, info = torchvision.io.read_video(
            video_path,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW"  # (T, C, H, W)
        )
        
        return video
        
    except Exception as e:
        print(f"Error loading video {video_path}: {str(e)}")
        return torch.zeros(0)

def process_single_item(item_id: str, item: Dict) -> Dict:
    """Process one annotation item in a worker process."""
    try:
        # Resolve the video path relative to the configured root.
        video_path = os.path.join(ROOT_VIDEO_PATH, item["video_path"])
        
        # Load the complete video.
        video_tensor = load_video_from_path(video_path)
        if video_tensor.numel() == 0:  # Skip videos that failed to load.
            return None
            
        # Build structured answer fields.
        key_objs = item.get("key_objs", [])

        if key_objs == []:
            perception_string = """```json\n[\n\t{"bbox_2d": [], "label": []}\n]\n```"""
        else:
            # Keep the source coordinate convention unchanged.
            strings = [f"""{{"bbox_2d": [{obj['bbox'][0]}, {obj['bbox'][1]}, {obj['bbox'][2]}, {obj['bbox'][3]}], "label": "{obj['gt_names']}"}}"""
                    for obj in key_objs]
            string = ",\n\t".join(strings)
            perception_string = f"```json\n[\n\t{string}\n]\n```"


        strings = [
            f"""{{"bbox_2d": {obj['bbox']}, "start": "{obj["prediction"]['from']}", "end": "{obj["prediction"]['to']}", "future_action": "{obj["prediction"]['future_status']}"}}"""
            for obj in key_objs
        ]
        
        if key_objs:
            joined_strings = ',\n\t'.join(strings)
            formatted_answer_predition = f"```json\n[\n\t{joined_strings}\n]\n```"
        else:
            formatted_answer_predition = "```json\n[\n\t{\"bbox_2d\": [], \"future_action\": []}\n]\n```"

        # Format the historical trajectory.
        historical_points = item["x_y_heading"]["history"]
        num_hist = len(historical_points)
        # History is sampled every 0.5 seconds, ending at -0.5 seconds.
        hist_time_labels = [-i * 0.5 for i in range(num_hist)]
        new_historical = []
        # Put the most recent point first.
        hist_time_labels = hist_time_labels[::-1]
        for i, point in enumerate(historical_points):
            # point: [x, y, heading]
            str_point = [point[0], point[1], point[2]]
            new_historical.append({
            "point_x_y_heading": str_point,
            "label": f"{hist_time_labels[i]}s"
            })
        hist_strings = [
            f"""{{"point_x_y_heading": {obj["point_x_y_heading"]}, "label": "{obj['label']}"}}"""
            for obj in new_historical
        ]
        hist_joined = ",\n\t".join(hist_strings)
        formatted_historical_points = f"""```json\n[\n\t{hist_joined}\n]\n```"""

        # Format the future trajectory.
        planning_points = item["x_y_heading"]["future"]
        num_plan = len(planning_points)
        # Future points are sampled every 0.5 seconds, starting at 0.5 seconds.
        plan_time_labels = [(i + 1) * 0.5 for i in range(num_plan)]
        new_planning = []
        for i, point in enumerate(planning_points):
            # point: [x, y, heading]
            str_point = [point[0], point[1], point[2]]
            new_planning.append({
                "point_x_y_heading": str_point,
                "label": f"{plan_time_labels[i]}s"
            })
        plan_strings = [
            f"""{{"point_x_y_heading": {obj["point_x_y_heading"]}, "label": "{obj['label']}"}}"""
            for obj in new_planning
        ]
        plan_joined = ",\n\t".join(plan_strings)
        formatted_new_planning = f"""```json\n[\n\t{plan_joined}\n]\n```"""

        # Other fields.
        ego_dynamic_state = item.get("ego_dynamic_state", [])
        driving_command = item.get("driving_command", [])

        # Convert ego dynamics to a readable string.
        if isinstance(ego_dynamic_state, list) and len(ego_dynamic_state) == 4:
            ego_dynamic_state_str = (
                f"vx: {ego_dynamic_state[0]:.2f} m/s, "
                f"vy: {ego_dynamic_state[1]:.2f} m/s, "
                f"ax: {ego_dynamic_state[2]:.2f} m/s², "
                f"ay: {ego_dynamic_state[3]:.2f} m/s²"
            )
        else:
            ego_dynamic_state_str = str(ego_dynamic_state)

        # Convert the one-hot driving command to text.
        # driving_cmd_str = ""
        # future_y = example["x_y_heading"]["future"][-1][1]
        # if future_y >= 2:
        #     driving_cmd_str = "turn left"
        # elif future_y <= -2:
        #     driving_cmd_str = "turn right"
        # else:
        #     driving_cmd_str = "go straight"
        driving_command = item["driving_command"][:3]
        if driving_command == [1, 0, 0]:
            driving_cmd_str = "turn left"
        elif driving_command == [0, 1, 0]:
            driving_cmd_str = "go straight"
        elif driving_command == [0, 0, 1]:  
            driving_cmd_str = "turn right"
        else:
            driving_cmd_str = str(item.get("driving_command", "unknown"))


        return {
            "id": item_id,
            "scene_name": item["scene_name"],
            "frame_id": item["frame_id"],
            "video_path": item["video_path"],  
            
            "driving_command": driving_cmd_str,
            "ego_dynamic_state": ego_dynamic_state_str,
            "historical_x_y_heading": formatted_historical_points,
            
            
            "solution": {
                "perception": perception_string,
                "prediction": formatted_answer_predition,
                "planning": formatted_new_planning,
            }
        }
        
    except Exception as e:
        print(f"Error processing item {item_id}: {str(e)}")
        return None

def process_json_data_parallel(json_path: str, max_workers: int = MAX_WORKERS) -> List[Dict]:
    """Process JSON data in parallel."""
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    processed_items = {}
    
    # Use a process pool for parallel processing.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks.
        futures = [
            executor.submit(process_single_item, item_id, item)
            for item_id, item in raw_data.items()
        ]
        
        # Collect successful results.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result is not None:
                processed_items[result["id"]] = result
    
    return processed_items




def main():
    global input_json, output_json_path, ROOT_VIDEO_PATH, MAX_WORKERS
    parser = argparse.ArgumentParser(description="Build the JSON intermediate used by RL preprocessing.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--num-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    input_json = args.input_json
    output_json_path = args.output_json
    ROOT_VIDEO_PATH = args.video_root
    MAX_WORKERS = args.num_workers

    # Process the input data.
    print("Loading and processing data in parallel...")
    processed_data = process_json_data_parallel(input_json, MAX_WORKERS)

    print(f"Processed {len(processed_data)} items.")
    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)   


if __name__ == "__main__":
    main()
