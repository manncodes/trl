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
AIME25 Distillation from gpt-oss-120B to Custom Split LLama

This script demonstrates how to distill math reasoning capabilities from a large
teacher model (gpt-oss-120B) running on a vLLM server to a custom student model
(Custom Split LLama) using the GOLD trainer.

## Setup

Your vLLM server is running at:
- Base URL: http://qpn744-vllm-gptoss120b-svc.llm-pretraining.svc.cluster.local:8000/v1
- Model: openai/gpt-oss-120b

## Data Requirements

For AIME25 distillation, you need a dataset with math problems. Options:
1. Pre-generated teacher completions (offline distillation)
2. On-the-fly teacher generation (online distillation with seq_kd)

## Usage

```bash
# Option 1: Using pre-generated teacher completions
python examples/scripts/gold_aime25_distillation.py \
    --model_name_or_path path/to/custom_split_llama \
    --dataset_path path/to/aime25_with_solutions.json \
    --output_dir ./aime25-distilled-model

# Option 2: Generate teacher completions first, then train
python examples/scripts/gold_aime25_distillation.py \
    --model_name_or_path path/to/custom_split_llama \
    --generate_teacher_data \
    --aime_problems_path path/to/aime25_problems.json \
    --output_dir ./aime25-distilled-model
```
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl.experimental import GOLDConfig, GOLDTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Arguments for the AIME25 distillation script."""

    # Model arguments
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "Path to the student model (your Custom Split LLama)"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"},
    )

    # Data arguments
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Path to dataset with pre-generated teacher completions (JSON/JSONL)"},
    )
    dataset_name: str | None = field(
        default=None,
        metadata={"help": "HuggingFace dataset name (e.g., 'AI-MO/NuminaMath-CoT')"},
    )
    aime_problems_path: str | None = field(
        default=None,
        metadata={"help": "Path to AIME problems JSON file (for generating teacher completions)"},
    )
    generate_teacher_data: bool = field(
        default=False,
        metadata={"help": "Whether to generate teacher completions before training"},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum number of samples to use"},
    )

    # vLLM Teacher server arguments (your gpt-oss-120B server)
    teacher_base_url: str = field(
        default="http://qpn744-vllm-gptoss120b-svc.llm-pretraining.svc.cluster.local:8000/v1",
        metadata={"help": "Base URL of the teacher vLLM server"},
    )
    teacher_model: str = field(
        default="openai/gpt-oss-120b",
        metadata={"help": "Model name on the vLLM server"},
    )
    teacher_tokenizer: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "Tokenizer for the teacher model (Llama family)"},
    )


def load_aime_problems(path: str) -> list[str]:
    """Load AIME problems from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        if isinstance(data[0], str):
            return data
        elif isinstance(data[0], dict):
            # Try common keys
            for key in ["problem", "question", "prompt", "content"]:
                if key in data[0]:
                    return [item[key] for item in data]
    raise ValueError(f"Could not parse AIME problems from {path}")


def generate_teacher_completions(
    problems: list[str],
    base_url: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate teacher completions using the vLLM server."""
    from trl.extras.vllm_openai_client import VLLMOpenAIClient

    logger.info(f"Connecting to teacher vLLM server at {base_url}")
    client = VLLMOpenAIClient(
        base_url=base_url,
        model=model,
    )

    system_prompt = (
        "You are a mathematical reasoning expert. Solve the given problem step by step. "
        "Show your complete reasoning process, then provide the final answer in \\boxed{}."
    )

    logger.info(f"Generating completions for {len(problems)} problems...")
    result = client.get_teacher_completions(
        prompts=problems,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    # Format as training data
    training_data = []
    for prompt, completion in zip(result["prompts"], result["completions"]):
        training_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        })

    return training_data


def load_or_create_dataset(args: ScriptArguments) -> Dataset:
    """Load dataset or create from teacher completions."""

    if args.dataset_path:
        # Load from local file
        path = Path(args.dataset_path)
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        elif path.suffix == ".jsonl":
            dataset = Dataset.from_json(str(path))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    elif args.dataset_name:
        # Load from HuggingFace
        dataset = load_dataset(args.dataset_name, split="train")

        # Convert to messages format if needed
        if "messages" not in dataset.column_names:
            if "problem" in dataset.column_names and "solution" in dataset.column_names:
                # NuminaMath-style format
                def convert_to_messages(example):
                    return {
                        "messages": [
                            {"role": "user", "content": example["problem"]},
                            {"role": "assistant", "content": example["solution"]},
                        ]
                    }
                dataset = dataset.map(convert_to_messages)

    elif args.generate_teacher_data and args.aime_problems_path:
        # Generate teacher completions
        problems = load_aime_problems(args.aime_problems_path)
        training_data = generate_teacher_completions(
            problems=problems,
            base_url=args.teacher_base_url,
            model=args.teacher_model,
        )
        dataset = Dataset.from_list(training_data)

        # Optionally save for later
        output_path = Path(args.aime_problems_path).stem + "_with_solutions.json"
        with open(output_path, "w") as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Saved teacher completions to {output_path}")

    else:
        # Create a sample dataset for demonstration
        logger.warning("No dataset provided. Creating sample AIME-style problems for demonstration.")
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": "Find the number of positive integers n ≤ 1000 such that 15n is a perfect square."},
                    {"role": "assistant", "content": "For 15n to be a perfect square, we need n = 15k² for some positive integer k.\n\nSince n ≤ 1000, we have 15k² ≤ 1000, so k² ≤ 66.67, meaning k ≤ 8.\n\nThe valid values of k are 1, 2, 3, 4, 5, 6, 7, 8.\n\nTherefore, there are \\boxed{8} such positive integers."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Let S be the set of all positive rational numbers r such that the decimal representation of r has a period of exactly 6. Find the sum of all elements in S that are less than 1."},
                    {"role": "assistant", "content": "A rational number has a decimal period of exactly 6 if and only if it can be written as a/999999 where gcd(a, 999999) = 1 and the period is exactly 6 (not a divisor of 6).\n\n999999 = 3³ × 7 × 11 × 13 × 37\n\nFor the period to be exactly 6, we need to exclude fractions whose denominators divide 9, 99, or 999 (periods 1, 2, 3).\n\nUsing inclusion-exclusion and Euler's totient function, we can compute the sum.\n\nThe answer is \\boxed{499999}."},
                ]
            },
        ]
        dataset = Dataset.from_list(sample_data)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    logger.info(f"Loaded dataset with {len(dataset)} examples")
    return dataset


def main():
    parser = HfArgumentParser((ScriptArguments, GOLDConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load dataset
    dataset = load_or_create_dataset(script_args)

    # Load tokenizer (student uses Llama tokenizer family)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # Use a default chat template for Llama-style models
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|system|>\n{{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}"
            "<|user|>\n{{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|assistant|>\n{{ message['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
        )

    # Load student model
    # For your Custom Split LLama, replace this with your custom model loading code:
    # from your_custom_module import CustomSplitLlamaForCausalLM
    # model = CustomSplitLlamaForCausalLM.from_pretrained(...)
    logger.info(f"Loading student model from {script_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=script_args.trust_remote_code,
    )

    # Override vLLM teacher settings from script args
    training_args.use_vllm_teacher = True
    training_args.vllm_teacher_api_type = "openai"
    training_args.vllm_teacher_base_url = script_args.teacher_base_url
    training_args.vllm_teacher_model = script_args.teacher_model
    training_args.teacher_tokenizer_name_or_path = script_args.teacher_tokenizer

    # For OpenAI-compatible API with sequence-level KD
    # We train on teacher-generated completions (already in dataset)
    # No need for logit-level distillation
    training_args.seq_kd = True

    # Initialize the trainer
    logger.info("Initializing GOLDTrainer...")
    trainer = GOLDTrainer(
        model=model,
        teacher_model=None,  # Teacher runs on vLLM server
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
