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
input_json = ""
output_dir = ""
ROOT_VIDEO_PATH = ""
BATCH_SIZE = 1000
MAX_WORKERS = 32

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
            
        # Extract optional reasoning fields.
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

def process_json_data_parallel(json_path: str, max_workers: int = MAX_WORKERS) -> List[Dict]:
    """Process annotation JSON in parallel."""
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    processed_items = []
    
    # Use a process pool for video decoding.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all items.
        futures = [
            executor.submit(process_single_item, item_id, item)
            for item_id, item in raw_data.items()
        ]
        
        # Collect successful results.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result is not None:
                processed_items.append(result)
    
    return processed_items

def create_dataset(processed_data: List[Dict]) -> Dataset:
    """Create a Hugging Face dataset from processed items."""
    # Store videos as binary tensors in the dataset.
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
    
    # Convert tensors to a format supported by Hugging Face datasets.
    for item in processed_data:
        item["video"] = item["video"].numpy().tobytes()
    
    # Build the column-oriented dictionary.
    dataset_dict = {
        key: [item[key] for item in processed_data]
        for key in features.keys()
    }
    
    return Dataset.from_dict(dataset_dict, features=features)

def save_batches(processed_data: List[Dict], output_dir: str, batch_size: int = BATCH_SIZE):
    """Save the processed dataset in batches."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process one batch at a time to limit memory use.
    for i in tqdm(range(0, len(processed_data), batch_size), desc="Saving batches"):
        batch = processed_data[i:i+batch_size]
        batch_idx = i // batch_size
        
        try:
            # Create the dataset object.
            batch_dataset = create_dataset(batch)
            
            # Construct the batch path.
            batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
            
            # Write the batch to disk.
            batch_dataset.save_to_disk(batch_dir)
            
        except Exception as e:
            print(f"Error saving batch {batch_idx}: {str(e)}")
            continue

def main():
    global input_json, output_dir, ROOT_VIDEO_PATH, BATCH_SIZE, MAX_WORKERS
    parser = argparse.ArgumentParser(description="Convert NavSim SFT JSON and videos to Hugging Face datasets.")
    parser.add_argument("--input-json", required=True, help="Input JSON generated by the annotation pipeline.")
    parser.add_argument("--video-root", required=True, help="Directory containing the referenced videos.")
    parser.add_argument("--output-dir", required=True, help="Directory for Dataset.save_to_disk batches.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    input_json = args.input_json
    output_dir = args.output_dir
    ROOT_VIDEO_PATH = args.video_root
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.num_workers

    # Process the annotations.
    print("Loading and processing data in parallel...")
    processed_data = process_json_data_parallel(input_json, MAX_WORKERS)
    
    # Save in batches to avoid excessive memory use.
    print(f"Saving data in batches to {output_dir}...")
    save_batches(processed_data, output_dir, BATCH_SIZE)
    
    print("All batches saved successfully.")

if __name__ == "__main__":
    main()
