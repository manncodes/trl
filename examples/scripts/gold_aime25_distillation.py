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

## Data Formats

The script supports multiple data formats:

1. **JSONL format** (recommended): One JSON object per line
   ```
   {"problem": "Find the sum of all integer bases...", "answer": 70, "id": "0"}
   {"problem": "Let S be the set of all positive...", "answer": 42, "id": "1"}
   ```

2. **JSON format**: Array of objects
   ```json
   [{"problem": "...", "answer": 70}, {"problem": "...", "answer": 42}]
   ```

3. **Pre-generated solutions**: Include "solution" field to skip teacher generation
   ```
   {"problem": "...", "solution": "Step 1: ...", "answer": 70}
   ```

## Prompt Template

Problems are formatted with the following template:
```
{question}
Please reason step by step, and put your final answer within \\boxed{}.
```

## Usage

```bash
# Option 1: Using JSONL file with problems only (generates teacher completions)
python examples/scripts/gold_aime25_distillation.py \\
    --model_name_or_path path/to/custom_split_llama \\
    --dataset_path path/to/aime25.jsonl \\
    --output_dir ./aime25-distilled-model

# Option 2: Using pre-generated teacher completions
python examples/scripts/gold_aime25_distillation.py \\
    --model_name_or_path path/to/custom_split_llama \\
    --dataset_path path/to/aime25_with_solutions.json \\
    --output_dir ./aime25-distilled-model

# Option 3: Generate teacher completions from problems file
python examples/scripts/gold_aime25_distillation.py \\
    --model_name_or_path path/to/custom_split_llama \\
    --generate_teacher_data \\
    --aime_problems_path path/to/aime25.jsonl \\
    --output_dir ./aime25-distilled-model
```
"""

import json
import logging
import os
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
        default="openai/gpt-oss-120b",
        metadata={"help": "Tokenizer for the teacher model (gpt-oss-120b)"},
    )


AIME_PROMPT_TEMPLATE = """{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def load_aime_problems(path: str, apply_template: bool = True, return_raw: bool = False) -> list[str] | list[dict]:
    """Load AIME problems from a JSON or JSONL file.

    Args:
        path: Path to the JSON or JSONL file containing problems.
        apply_template: Whether to apply the AIME prompt template.
        return_raw: If True, return raw data dicts with problem/answer fields.

    Returns:
        List of problem strings (with template applied if requested), or
        list of raw data dicts if return_raw=True.
    """
    path_obj = Path(path)

    # Determine format and load data
    if path_obj.suffix == ".jsonl":
        # JSONL format: one JSON object per line
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        # Regular JSON format
        with open(path) as f:
            data = json.load(f)

    if isinstance(data, list) and len(data) == 0:
        raise ValueError(f"Empty dataset: {path}")

    # Return raw data if requested (for verification with answers)
    if return_raw:
        if isinstance(data[0], dict):
            return data
        else:
            # Convert string list to dict format
            return [{"problem": p, "answer": ""} for p in data]

    # Extract problems from data
    problems = []
    if isinstance(data, list):
        if isinstance(data[0], str):
            problems = data
        elif isinstance(data[0], dict):
            # Try common keys
            for key in ["problem", "question", "prompt", "content"]:
                if key in data[0]:
                    problems = [item[key] for item in data]
                    break
            if not problems:
                raise ValueError(f"Could not find problem field in data. Available keys: {list(data[0].keys())}")
    else:
        raise ValueError(f"Expected list of problems, got {type(data)}")

    # Apply prompt template if requested
    if apply_template:
        problems = [AIME_PROMPT_TEMPLATE.format(question=p) for p in problems]

    return problems


def verify_answer(completion: str, gold_answer: str | int | float) -> bool:
    """Verify if the completion contains the correct answer using trl.rewards.accuracy_rewards."""
    try:
        from trl.rewards.accuracy_rewards import reasoning_accuracy_reward
    except ImportError:
        logger.warning("Could not import accuracy_rewards. Make sure math_verify is installed.")
        return False

    try:
        # Format as expected by reasoning_accuracy_reward
        completions = [[{"role": "assistant", "content": completion}]]
        solution = [str(gold_answer)]

        # Use reasoning_accuracy_reward which handles </think> delimiters
        rewards = reasoning_accuracy_reward(
            completions=completions,
            solution=solution,
            reasoning_delimiters=["</think>", "</answer>"],
        )

        # Return True if reward is 1.0 (correct), False otherwise
        return rewards[0] == 1.0
    except Exception as e:
        logger.debug(f"Verification failed: {e}")
        return False


def generate_teacher_completions(
    problems: list[dict],  # List of {"problem": str, "answer": str/int}
    base_url: str,
    model: str,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    n_samples: int = 8,
) -> list[dict]:
    """Generate teacher completions with n samples per problem and math verification.

    For each problem:
    1. Generate n_samples completions
    2. Verify each against the gold answer using math_verify
    3. Select the first correct completion (or best available if none correct)

    vLLM automatically parses reasoning_content for reasoning models (gpt-oss, DeepSeek-R1, etc.)
    and the client combines it into <think>...</think><answer>...</answer> format.
    """
    from trl.extras.vllm_openai_client import VLLMOpenAIClient

    logger.info(f"Connecting to teacher vLLM server at {base_url}")
    client = VLLMOpenAIClient(
        base_url=base_url,
        model=model,
        max_concurrent_requests=256,  # Higher concurrency for n_samples
    )

    # Simple system prompt - reasoning is handled by the model natively
    system_prompt = (
        "You are a mathematical reasoning expert. Solve the given problem step by step. "
        "Show your complete reasoning process, then provide the final answer with \\boxed{}."
    )

    # Expand prompts: each problem gets n_samples copies
    expanded_prompts = []
    problem_indices = []  # Track which problem each prompt belongs to
    for i, item in enumerate(problems):
        prompt = AIME_PROMPT_TEMPLATE.format(question=item["problem"])
        for _ in range(n_samples):
            expanded_prompts.append(prompt)
            problem_indices.append(i)

    logger.info(f"Generating {n_samples} samples for {len(problems)} problems ({len(expanded_prompts)} total requests)...")
    result = client.get_teacher_completions(
        prompts=expanded_prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        include_reasoning=True,
    )

    # Group completions by problem
    completions_by_problem: dict[int, list[str]] = {}
    for idx, completion in zip(problem_indices, result["completions"]):
        if idx not in completions_by_problem:
            completions_by_problem[idx] = []
        completions_by_problem[idx].append(completion)

    # Select best completion for each problem (first correct one, or first non-empty)
    training_data = []
    verified_count = 0
    failed_count = 0

    for i, item in enumerate(problems):
        candidates = completions_by_problem.get(i, [])
        gold_answer = item.get("answer", "")
        prompt = AIME_PROMPT_TEMPLATE.format(question=item["problem"])

        # Filter out failed completions
        valid_candidates = [
            c for c in candidates
            if c is not None and not c.startswith("ERROR:") and c.strip() != ""
        ]

        if not valid_candidates:
            failed_count += 1
            logger.warning(f"No valid completions for problem {i}: {item['problem'][:80]}...")
            continue

        # Try to find a verified correct answer
        selected_completion = None
        for completion in valid_candidates:
            if verify_answer(completion, gold_answer):
                selected_completion = completion
                verified_count += 1
                break

        # If no verified answer, use the first valid completion
        if selected_completion is None:
            selected_completion = valid_candidates[0]
            logger.debug(f"Problem {i}: No verified answer, using first completion")

        training_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": selected_completion},
            ]
        })

    logger.info(f"Verification results: {verified_count}/{len(problems)} problems have verified correct answers")
    if failed_count > 0:
        logger.warning(f"Failed to generate for {failed_count}/{len(problems)} problems")

    # Log format info
    has_reasoning = any("<think>" in d["messages"][2]["content"] for d in training_data[:5])
    if has_reasoning:
        logger.info(f"Successfully generated {len(training_data)} training examples with <think>/<answer> CoT format")
    else:
        logger.info(f"Successfully generated {len(training_data)} training examples (no reasoning_content from vLLM)")

    return training_data


def load_or_create_dataset(args: ScriptArguments) -> Dataset:
    """Load dataset or create from teacher completions."""

    if args.dataset_path:
        # Load from local file
        path = Path(args.dataset_path)

        # Load data based on format
        if path.suffix == ".jsonl":
            # JSONL format: one JSON object per line
            data = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Check if this is already in messages format (pre-generated solutions)
        if isinstance(data[0], dict) and "messages" in data[0]:
            dataset = Dataset.from_list(data)
        elif isinstance(data[0], dict) and "problem" in data[0]:
            # AIME-style format: {"problem": "...", "answer": ..., "id": "..."}
            # This needs teacher completions - either pre-computed or generate them
            if "solution" in data[0]:
                # Has pre-computed solutions, convert to messages format
                def convert_to_messages(item):
                    prompt = AIME_PROMPT_TEMPLATE.format(question=item["problem"])
                    return {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": item["solution"]},
                        ]
                    }
                dataset = Dataset.from_list([convert_to_messages(item) for item in data])
            else:
                # No solutions - generate from teacher with n=8 sampling and verification
                logger.info("Dataset has problems but no solutions. Generating from teacher with verification...")
                training_data = generate_teacher_completions(
                    problems=data,  # Pass full problem data with answers for verification
                    base_url=args.teacher_base_url,
                    model=args.teacher_model,
                    n_samples=8,  # Generate 8 samples per problem
                )
                dataset = Dataset.from_list(training_data)

                # Save for later reuse
                output_path = str(path.stem) + "_with_solutions.json"
                with open(output_path, "w") as f:
                    json.dump(training_data, f, indent=2)
                logger.info(f"Saved teacher completions to {output_path}")
        else:
            raise ValueError(f"Unrecognized data format. Expected 'messages' or 'problem' field.")

    elif args.dataset_name:
        # Load from HuggingFace
        dataset = load_dataset(args.dataset_name, split="train")

        # Convert to messages format if needed
        if "messages" not in dataset.column_names:
            if "problem" in dataset.column_names and "solution" in dataset.column_names:
                # NuminaMath-style format
                def convert_to_messages(example):
                    prompt = AIME_PROMPT_TEMPLATE.format(question=example["problem"])
                    return {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": example["solution"]},
                        ]
                    }
                dataset = dataset.map(convert_to_messages)

    elif args.generate_teacher_data and args.aime_problems_path:
        # Generate teacher completions from problems file with n=8 sampling
        problems_data = load_aime_problems(args.aime_problems_path, apply_template=False, return_raw=True)
        training_data = generate_teacher_completions(
            problems=problems_data,  # Pass full data with answers
            base_url=args.teacher_base_url,
            model=args.teacher_model,
            n_samples=8,
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
        sample_question_1 = "Find the number of positive integers n ≤ 1000 such that 15n is a perfect square."
        sample_question_2 = "Let S be the set of all positive rational numbers r such that the decimal representation of r has a period of exactly 6. Find the sum of all elements in S that are less than 1."
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": AIME_PROMPT_TEMPLATE.format(question=sample_question_1)},
                    {"role": "assistant", "content": "For 15n to be a perfect square, we need n = 15k² for some positive integer k.\n\nSince n ≤ 1000, we have 15k² ≤ 1000, so k² ≤ 66.67, meaning k ≤ 8.\n\nThe valid values of k are 1, 2, 3, 4, 5, 6, 7, 8.\n\nTherefore, there are \\boxed{8} such positive integers."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": AIME_PROMPT_TEMPLATE.format(question=sample_question_2)},
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

    # Check if running distributed
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    # Model loading kwargs for efficient distributed loading
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": script_args.trust_remote_code,
        "attn_implementation": "flash_attention_2",  # Use Flash Attention 2 on H100s
    }

    # For FSDP, we load on CPU first then shard - more memory efficient
    if is_distributed:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        **model_kwargs,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
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
