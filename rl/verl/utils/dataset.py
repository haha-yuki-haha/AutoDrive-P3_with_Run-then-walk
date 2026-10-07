# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import ast
from datasets import load_dataset, load_from_disk, concatenate_datasets
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[List[ImageObject], Tuple[List[ImageObject], List[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


def load_data(data_cache_dir):
    lm_datasets = []
    data_paths = os.listdir(data_cache_dir)
    for index, data_path in enumerate(data_paths):
        cache_path = os.path.join(data_cache_dir, data_path)
        processed_dataset = load_from_disk(cache_path)

        print(f'datasets-{cache_path} has been loaded from disk')
        if index == 0:
            lm_datasets = processed_dataset
        else:
            assert lm_datasets.features.type == processed_dataset.features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset])

    return lm_datasets



class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        question_template: Optional[str] = None,  # Optional prompt template.
        # video_shape: str = "[6, 3, 504, 896]",  # Video tensor shape (T, C, H, W).
        video_shape: List = [6, 3, 504, 896],  # Video tensor shape (T, C, H, W).
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_key = video_key
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.question_template = question_template  # Store the prompt template.
        self.video_shape = tuple(video_shape)  # Store the video shape.
        # self.video_shape = tuple(ast.literal_eval(video_shape))  # Parse string form if needed.

        # Load the dataset.
        self.dataset = load_data(data_path)
        
        with open(self.question_template, encoding="utf-8") as f:
            self.question_template = f.read()

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Build the video-specific message format.
        question = self.question_template.replace(
            "[VEHICLE_SPEED]", f"{example['ego_state']}"
        ).replace(
            "[Ego_Future_Action]", example["driving_command"]
        ).replace(
            "[HISTORICAL_TRAJECTORY]", example["historical_x_y_angle"]
        )

        # print(example.keys(), self.video_key)
        video_bytes = example[self.video_key]
        video_array = np.frombuffer(video_bytes, dtype=np.uint8)
        video_array = video_array.reshape(self.video_shape).copy()
        video_tensor = torch.from_numpy(video_array).to(torch.float32)

        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "video", 
                    "video": video_tensor,
                    "fps": self.video_fps,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            ]
        }]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        # Build the ground-truth solution dictionary.
        example["solution"] = {
            "scene_name": example.pop("scene_name"),
            "frame_id": example.pop("frame_id"),
            "perception": example.pop("solution_perception"),
            "prediction": example.pop("solution_prediction"),
            "planning": example.pop("solution_planning")
        }
        # print(example["solution"]["scene_name"], example["solution"]["frame_id"])
        
        # Process the video input.
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Read the video tensor from the message.
        video_tensor = messages[0]["content"][1]["video"]
        fps_inputs = [self.video_fps]

        model_inputs = self.processor(
            videos=[video_tensor], 
            text=[prompt], 
            fps=fps_inputs,  # Pass the video frame rates.
            add_special_tokens=False, 
            return_tensors="pt"
        )

        # prompt_ids, prompt_mask = model_inputs["input_ids"], model_inputs["attention_mask"]
        # pixel_values_videos = model_inputs["pixel_values_videos"]
        # video_grid_thw = model_inputs["video_grid_thw"]
        # print(prompt_ids.shape, prompt_mask.shape, pixel_values_videos.shape, video_grid_thw.shape)
        
        # Handle Qwen2VL position ids.
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            position_ids = get_rope_index(
                self.processor,
                input_ids=model_inputs["input_ids"][0],
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=model_inputs["attention_mask"][0],
            )
        else:
            position_ids = torch.clip(model_inputs["attention_mask"][0].cumsum(dim=0) - 1, min=0, max=None)
        
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        example["multi_modal_data"] = {"videos": [video_tensor]}  

        # Post-process model inputs.
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        
        # Attach the solution to the output example.
        example["ground_truth"] = example.pop("solution")
        
        return example
