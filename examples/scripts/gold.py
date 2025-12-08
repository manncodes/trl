#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""
GOLD (Generalized Online Logit Distillation) Training Script

This script demonstrates the base GOLD recipe for knowledge distillation:

┌─────────────────────────────────────────────────────────────────────────────┐
│                           GOLD TRAINING FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

1. ON-POLICY GENERATION (with probability λ):
   ┌──────────────────────────────────────┐
   │  Student (vLLM) generates sequences  │  ← Fast generation
   │  prompt → token_ids (no logits)      │
   └──────────────────────────────────────┘

2. LOG PROBABILITY COMPUTATION:
   ┌─────────────────────┐    ┌─────────────────────┐
   │  Student (HF)       │    │  Teacher (HF)       │
   │  forward pass       │    │  forward pass       │
   │  → student_logits   │    │  → teacher_logits   │
   │  (gradients ON)     │    │  (gradients OFF)    │
   └─────────────────────┘    └─────────────────────┘

3. LOSS COMPUTATION:
   ┌─────────────────────────────────────────────────┐
   │  Generalized JSD Loss                          │
   │  JSD = β·KL(teacher||mixture) +                │
   │        (1-β)·KL(student||mixture)              │
   │  where mixture = (1-β)·student + β·teacher     │
   └─────────────────────────────────────────────────┘

4. UPDATE:
   - Only student weights are updated
   - Sync student weights to vLLM (if using vLLM)

## Key Parameters

- `lmbda`: Probability of on-policy generation (0.5 = 50% generated, 50% dataset)
- `beta`: JSD interpolation (0=forward KL, 0.5=symmetric JSD, 1=reverse KL)
- `temperature`: Softmax temperature for distillation

## Usage

```bash
# Basic usage with local teacher
python examples/scripts/gold.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir ./gold-distilled-model

# With vLLM for fast generation
python examples/scripts/gold.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/Capybara \
    --use_vllm \
    --output_dir ./gold-distilled-model

# Multi-GPU with accelerate
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/gold.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir ./gold-distilled-model
```
"""

import logging
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl.experimental.gold import GOLDConfig, GOLDTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Arguments for the GOLD training script."""

    # Model arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Path to the student model"},
    )
    teacher_model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the teacher model"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading models"},
    )

    # Data arguments
    dataset_name: str = field(
        default="trl-lib/Capybara",
        metadata={"help": "HuggingFace dataset name"},
    )
    dataset_config: str | None = field(
        default=None,
        metadata={"help": "Dataset configuration name"},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (for debugging)"},
    )


def main():
    parser = HfArgumentParser((ScriptArguments, GOLDConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load dataset
    logger.info(f"Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        split="train",
    )

    if script_args.max_samples:
        dataset = dataset.select(range(min(script_args.max_samples, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)} examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student model
    logger.info(f"Loading student model: {script_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=script_args.trust_remote_code,
    )

    # Load teacher model
    logger.info(f"Loading teacher model: {script_args.teacher_model_name_or_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        script_args.teacher_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=script_args.trust_remote_code,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Initialize the GOLD trainer
    logger.info("Initializing GOLDTrainer...")
    trainer = GOLDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GOLD training...")
    logger.info(f"  - Lambda (on-policy prob): {training_args.lmbda}")
    logger.info(f"  - Beta (JSD interpolation): {training_args.beta}")
    logger.info(f"  - Temperature: {training_args.temperature}")
    logger.info(f"  - Using vLLM: {training_args.use_vllm}")

    trainer.train()

    # Save the model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
