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
GOLD Training with vLLM Teacher Example

This example demonstrates how to use the GOLD (Generalized Online Logit Distillation)
trainer with a teacher model running on a vLLM server.

This is useful when:
1. The teacher model is too large to fit on the same GPU as the student
2. You want to use vLLM's efficient inference for the teacher
3. You have a custom student architecture (e.g., Custom Split LLama)

## Setup

### Step 1: Start the vLLM server for the teacher model
```bash
# Start the teacher model on vLLM server (port 8002 by default for teacher)
trl vllm-serve --model meta-llama/Llama-3.3-70B-Instruct --port 8002 --tensor-parallel-size 4

# Or for your custom teacher:
# trl vllm-serve --model gpt-oss-120B --port 8002 --tensor-parallel-size 8 --trust-remote-code
```

### Step 2: Run this training script
```bash
python examples/scripts/gold_vllm_teacher_example.py \
    --model_name_or_path path/to/custom_split_llama \
    --teacher_tokenizer_name_or_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --output_dir ./gold-custom-split-llama
```

## Custom Split LLama Architecture

If you have a custom model architecture (e.g., Custom Split LLama with first 32 layers
of Llama 3.1 7B + 2 MLP + last 8 layers of Llama 3.3 80B), you can pass it directly
to the trainer. The model should be a HuggingFace model or compatible.

Example custom model loading:
```python
from your_custom_module import CustomSplitLlamaForCausalLM

model = CustomSplitLlamaForCausalLM.from_pretrained(
    "path/to/custom_split_llama",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```
"""

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl.experimental import GOLDConfig, GOLDTrainer


@dataclass
class ScriptArguments:
    """Arguments for the training script."""

    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "Path to the student model or model identifier from huggingface.co/models"},
    )
    teacher_tokenizer_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Path to the teacher tokenizer (required for vLLM teacher)"},
    )
    dataset_name: str = field(
        default="HuggingFaceH4/ultrachat_200k",
        metadata={"help": "Name of the dataset to use"},
    )
    dataset_config: str | None = field(
        default=None,
        metadata={"help": "Dataset configuration name"},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum number of samples to use from the dataset"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading the model"},
    )


def main():
    parser = HfArgumentParser((ScriptArguments, GOLDConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load the dataset
    dataset = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        split="train_sft",
    )
    if script_args.max_samples is not None:
        dataset = dataset.select(range(min(script_args.max_samples, len(dataset))))

    # Load tokenizer (student uses Llama tokenizer family)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student model
    # For custom architectures, you would load your custom model here:
    # from your_custom_module import CustomSplitLlamaForCausalLM
    # model = CustomSplitLlamaForCausalLM.from_pretrained(...)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=script_args.trust_remote_code,
    )

    # Initialize the trainer
    # Note: teacher_model=None because we're using vLLM teacher
    trainer = GOLDTrainer(
        model=model,
        teacher_model=None,  # Teacher runs on vLLM server
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
