"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import numpy as np
from tqdm import tqdm
import json
import random

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)


from transformers import Qwen2_5_VLForConditionalGeneration


W = 24*28
H = 6*28
min_pixels = 24*28*6*28
max_pixels = 4*24*28*6*28


@dataclass
class SFTScriptArguments(ScriptArguments):
    dataset_dir: str = field(
        default="",
        metadata={"help": "Path to the training data dir."},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})


processor = None
## noCoT ##
QUESTION_SFT = """
You are an expert driving assistant. As an expert driving assistant, analyze the 4-second driving video context and answer the perception, prediction and planning question in the final frame.
Output format is '<perception> </perception>\n<prediction> </prediction>\n<planning> </planning>'. 
Ego Future Action is [Ego_Future_Action]. Ego current state is [VEHICLE_SPEED], and the historical trajectory of the ego vehicle is [HISTORICAL_TRAJECTORY].
"""


def convert_example(example):
    messages = []
    
    video_tensor = "path"
    Question = QUESTION_SFT
    Question = QUESTION_SFT.replace("[VEHICLE_SPEED]", f"{example['ego_state']}")
    Question = Question.replace("[Ego_Future_Action]", example["driving_command"])
    Question = Question.replace("[HISTORICAL_TRAJECTORY]", example["historical_x_y_heading"])
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": Question},
            {
                "type": "video", "video": video_tensor, "fps": 2, 
                "min_pixels": min_pixels, "max_pixels": max_pixels,
            },
        ]
    })

    answer_text_perception = f"<perception>{example['solution_perception']}</perception>"
    answer_text_prediction = f"<prediction>{example['solution_prediction']}</prediction>"
    answer_text_planning = f"<planning>{example['solution_planning']}</planning>"

    messages.append({
        "role": "assistant",
        "content": answer_text_perception + "\n" + answer_text_prediction + "\n" + answer_text_planning,
    })
    
    # print(Question, messages)
    
    example["messages"] = messages
    return example


def collate_fn(examples):
    texts = [
        processor.apply_chat_template(convert_example(example)["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    # print(texts[0])
    # video_inputs = [x["video"] for x in examples]
    video_inputs = []
    fps_inputs = []
    image_inputs = None
    for x in examples:
        video_bytes = x["video"]
        video_array = np.frombuffer(video_bytes, dtype=np.uint8)  # Stored as fixed-shape (T, C, H, W) bytes.
        video_array = video_array.reshape(4, 3, H, W)    
        video_array = video_array.copy()
        video_inputs.append(torch.from_numpy(video_array).to(torch.float32))  # Convert to float32.
        fps_inputs.append(2)

    batch = processor(
        text=texts,
        images=image_inputs, 
        videos=video_inputs, 
        fps=fps_inputs, 
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    labels[labels == video_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    
    from datasets import concatenate_datasets
    def load_data(data_cache_dir):
        lm_datasets = []
        data_paths = os.listdir(data_cache_dir)
        for index, data_path in enumerate(data_paths):
            cache_path = os.path.join(data_cache_dir, data_path)
            processed_dataset = datasets.load_from_disk(cache_path)

            logger.info(f'datasets-{cache_path} has been loaded from disk')
            if index == 0:
                lm_datasets = processed_dataset
            else:
                assert lm_datasets.features.type == processed_dataset.features.type
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset])

        return lm_datasets
    
    dataset = load_data(script_args.dataset_dir)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_sliding_window=True,
    )
    # training_args.model_init_kwargs = model_kwargs

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, 
        # torch_dtype=torch.bfloat16,
        **model_kwargs
    )
        
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        # train_dataset=dataset["train"],
        # eval_dataset=dataset["eval"] if training_args.eval_strategy != "no" else None,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args)
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    # kwargs = {
    #     "dataset_name": script_args.dataset_name,
    #     "tags": ["Auto_Driven_GRPO"],
    # }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        
    #############
    # push to hub
    #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)
    #     processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
