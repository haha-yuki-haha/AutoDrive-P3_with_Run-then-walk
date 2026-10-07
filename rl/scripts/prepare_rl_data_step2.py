import json
import os
import time
import argparse
from typing import Dict, List, Any
import random

import torch
import torchvision
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

input_json = ""
output_dir = ""
ROOT_VIDEO_PATH = ""
BATCH_SIZE = 5000  # Number of items handled per batch.
MAX_WORKERS = 96   # Number of parallel workers.

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
        # Extract optional reasoning and solution fields.
        think = item.get("think", {})
        solution = item.get("solution", {})
        
        return {
            "id": item_id,
            "video": video_tensor,  # Store the complete video tensor.
            
            "scene_name": item["scene_name"],
            "frame_id": item["frame_id"],
            "driving_command": item["driving_command"],
            "ego_state": item["ego_dynamic_state"],
            "historical_x_y_heading": item["historical_x_y_heading"],
            
            # Reasoning fields.
            "think_perception": think.get("perception", ""),
            "think_prediction": think.get("prediction", ""),
            "think_planning": think.get("planning", ""),
            
            # Ground-truth solution fields.
            "solution_perception": solution.get("perception", ""),
            "solution_prediction": solution.get("prediction", ""),
            "solution_planning": solution.get("planning", ""),
        }
        
    except Exception as e:
        print(f"Error processing item {item_id}: {str(e)}")
        return None

def load_json_in_batches(json_path: str, batch_size: int = BATCH_SIZE):
    """Read JSON data in batches."""
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    items = list(raw_data.items())
    for i in range(0, len(items), batch_size):
        yield {
            item_id: item
            for item_id, item in items[i:i + batch_size]
        }


def process_batch(batch: Dict[str, Any], max_workers: int = MAX_WORKERS) -> List[Dict]:
    """Process one batch in parallel."""
    processed_items = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_item, item_id, item)
            for item_id, item in batch.items()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch", leave=False):
            result = future.result()
            if result is not None:
                processed_items.append(result)
    return processed_items


def create_dataset(processed_data: List[Dict]) -> Dataset:
    features = Features({
        "id": Value("string"),
        "video": Value("binary"),  # Serialized PyTorch tensor bytes.
        
        "scene_name": Value("string"),
        "frame_id": Value("string"),
        "driving_command": Value("string"),
        "ego_state": Value("string"),
        "historical_x_y_heading": Value("string"),

        # Reasoning fields.
        "think_perception": Value("string"),
        "think_prediction": Value("string"),
        "think_planning": Value("string"),
        
        # Ground-truth solution fields.
        "solution_perception": Value("string"),
        "solution_prediction": Value("string"),
        "solution_planning": Value("string"),
    })
    for item in processed_data:
        item["video"] = item["video"].numpy().tobytes()
    dataset_dict = {
        key: [item[key] for item in processed_data]
        for key in features.keys()
    }
    return Dataset.from_dict(dataset_dict, features=features)


def save_batch(batch_data: List[Dict], output_dir: str, batch_idx: int):
    """Save one processed batch."""
    try:
        dataset = create_dataset(batch_data)
        batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
        dataset.save_to_disk(batch_dir)
        print(f"Batch {batch_idx} saved to {batch_dir}")
    except Exception as e:
        print(f"Error saving batch {batch_idx}: {str(e)}")

def main():    
    global input_json, output_dir, ROOT_VIDEO_PATH, BATCH_SIZE, MAX_WORKERS
    parser = argparse.ArgumentParser(description="Convert RL annotations and videos to Hugging Face datasets.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    input_json = args.input_json
    output_dir = args.output_dir
    ROOT_VIDEO_PATH = args.video_root
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.num_workers

    batch_size = BATCH_SIZE

    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)

    print("Processing data in batches...")
    for batch_idx, batch in enumerate(load_json_in_batches(input_json, batch_size)):
        print(f"Processing batch {batch_idx}...")
        processed_batch = process_batch(batch, MAX_WORKERS)
        save_batch(processed_batch, output_dir, batch_idx)
    print("All batches processed and saved.")

if __name__ == "__main__":
    main()
