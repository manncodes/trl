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
GOLD with vLLM Teacher Generation + Transformers Log Probs

This script implements an optimized GOLD recipe where:
1. Teacher generates sequences via vLLM server (fast async batch generation)
2. Both student and teacher compute log probs via local transformers (accurate, full vocab)

┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID GOLD FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TEACHER GENERATION (vLLM - Async Batch)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  prompts ──► VLLMOpenAIClient.chat() ──► completion_texts                   │
│              (async, high concurrency)     (just text, no logits)           │
│                                                                             │
│  Why vLLM for generation?                                                   │
│  - Autoregressive generation is sequential (slow in transformers)           │
│  - vLLM has optimized KV cache, continuous batching, PagedAttention         │
│  - Can handle large teacher models on separate GPU cluster                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: LOG PROB COMPUTATION (Transformers - Batch Prefill)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [prompt + completion] tokenized ──► model.forward() ──► logits             │
│                                      (single pass)       [B, L, V]          │
│                                                                             │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │  Student (local)    │         │  Teacher (local)    │                    │
│  │  forward(input_ids) │         │  forward(input_ids) │                    │
│  │  → student_logits   │         │  → teacher_logits   │                    │
│  │  (grads ON)         │         │  (grads OFF)        │                    │
│  └─────────────────────┘         └─────────────────────┘                    │
│                                                                             │
│  Why transformers for log probs?                                            │
│  - Prefill is O(1) in sequence (parallel matmul, not autoregressive)        │
│  - Full vocabulary logits (not top-k)                                       │
│  - Exact numerical values (no sampling contamination)                       │
│  - Same computation graph for proper gradients                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: LOSS COMPUTATION                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Same tokenizer: loss = JSD(student_logits, teacher_logits)                 │
│  Diff tokenizer: loss = ULD(student_logits, teacher_logits) with alignment  │
│                                                                             │
│  Generalized JSD:                                                           │
│  - β=0.0: Forward KL (mode-seeking)                                         │
│  - β=0.5: Symmetric JSD (balanced)                                          │
│  - β=1.0: Reverse KL (mean-seeking)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

## Performance Comparison

| Approach                    | Generation | Log Probs | Full Vocab | Accurate |
|-----------------------------|------------|-----------|------------|----------|
| vLLM logprobs=-1            | Fast       | O(n) SLOW | Yes        | No*      |
| Transformers generate+fwd   | SLOW       | Fast      | Yes        | Yes      |
| Hybrid (this script)        | Fast       | Fast      | Yes        | Yes      |

*vLLM logprobs affected by sampling params, numerical differences from transformers

## Usage

```bash
# Basic usage (same tokenizer for student/teacher)
python examples/scripts/gold_vllm_teacher.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --vllm_server_url http://localhost:8000/v1 \
    --dataset_name trl-lib/Capybara \
    --output_dir ./gold-hybrid-distilled

# Cross-tokenizer (different model families)
python examples/scripts/gold_vllm_teacher.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-72B-Instruct \
    --teacher_tokenizer_name_or_path Qwen/Qwen2.5-72B-Instruct \
    --vllm_server_url http://teacher-cluster:8000/v1 \
    --output_dir ./gold-cross-tokenizer
```
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)

from trl.extras.vllm_openai_client import VLLMOpenAIClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Arguments for the hybrid GOLD training script."""

    # Student model (local)
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Path to the student model (trained locally)"},
    )

    # Teacher model - local copy for log probs
    teacher_model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the local teacher model for log prob computation"},
    )
    teacher_tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Tokenizer for teacher (if different from student). Enables cross-tokenizer distillation."},
    )

    # vLLM server for teacher generation
    vllm_server_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "URL of the vLLM server for teacher generation"},
    )
    vllm_model_name: str | None = field(
        default=None,
        metadata={"help": "Model name on vLLM server (if different from teacher_model_name_or_path)"},
    )
    vllm_max_concurrent: int = field(
        default=64,
        metadata={"help": "Maximum concurrent requests to vLLM server"},
    )

    # Data
    dataset_name: str = field(
        default="trl-lib/Capybara",
        metadata={"help": "HuggingFace dataset name"},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum samples for debugging"},
    )

    # Training
    output_dir: str = field(
        default="./gold-hybrid-output",
        metadata={"help": "Output directory"},
    )
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=4, metadata={"help": "Training batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate"})
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    max_completion_length: int = field(default=1024, metadata={"help": "Maximum completion length"})

    # GOLD hyperparameters
    beta: float = field(default=0.5, metadata={"help": "JSD interpolation (0=KL, 0.5=JSD, 1=reverse KL)"})
    temperature: float = field(default=1.0, metadata={"help": "Distillation temperature"})
    generation_temperature: float = field(default=0.7, metadata={"help": "Teacher generation temperature"})

    # Misc
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code"})
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Use gradient checkpointing"})


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Generalized Jensen-Shannon Divergence loss.

    Args:
        student_logits: [batch_size, seq_len, vocab_size]
        teacher_logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] with -100 for tokens to ignore
        beta: Interpolation coefficient (0=KL, 0.5=JSD, 1=reverse KL)
        temperature: Softmax temperature

    Returns:
        Scalar loss tensor
    """
    # Temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Log softmax for numerical stability
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        # Forward KL: KL(teacher || student)
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        # Reverse KL: KL(student || teacher)
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        # Generalized JSD
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log1p(-beta_t),
                teacher_log_probs + torch.log(beta_t)
            ]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    # Sum over vocab dimension
    jsd = jsd.sum(dim=-1)

    # Apply mask
    if labels is not None:
        mask = labels != -100
        jsd = jsd * mask
        return jsd.sum() / mask.sum().clamp(min=1)
    else:
        return jsd.mean()


def uld_sorted_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor | None = None,
    teacher_labels: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Universal Logit Distillation loss using sorted probability comparison.

    This enables distillation between models with DIFFERENT vocabulary sizes
    by comparing the SHAPE of probability distributions rather than
    specific token probabilities.

    Algorithm:
    1. Convert logits to probabilities (softmax with temperature)
    2. Sort both distributions in descending order
    3. Pad smaller vocab to match larger vocab (with zeros)
    4. Compute L1 loss between sorted distributions

    This works because:
    - Sorting removes token identity - only distribution shape matters
    - A well-trained student should have similar confidence patterns as teacher
    - "How spread out is the probability mass?" is vocab-agnostic

    Args:
        student_logits: [batch_size, seq_len, student_vocab_size]
        teacher_logits: [batch_size, seq_len, teacher_vocab_size]
        student_labels: [batch_size, seq_len] with -100 for tokens to ignore (student tokenization)
        teacher_labels: [batch_size, seq_len] with -100 for tokens to ignore (teacher tokenization)
        temperature: Softmax temperature for distillation

    Returns:
        Scalar loss tensor
    """
    # Temperature scaling and convert to probabilities
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    batch_size = student_probs.size(0)
    device = student_probs.device

    losses = []

    for i in range(batch_size):
        # Get masks for valid positions (completion tokens only)
        if student_labels is not None:
            student_mask = student_labels[i] != -100
        else:
            student_mask = torch.ones(student_probs.size(1), dtype=torch.bool, device=device)

        if teacher_labels is not None:
            teacher_mask = teacher_labels[i] != -100
        else:
            teacher_mask = torch.ones(teacher_probs.size(1), dtype=torch.bool, device=device)

        # Get valid positions count
        student_valid = student_mask.sum().item()
        teacher_valid = teacher_mask.sum().item()

        if student_valid == 0 or teacher_valid == 0:
            continue

        # Use minimum length for alignment (different tokenizations may have different lengths)
        min_valid = min(student_valid, teacher_valid)

        # Get the first min_valid valid positions from each
        student_valid_indices = torch.where(student_mask)[0][:min_valid]
        teacher_valid_indices = torch.where(teacher_mask)[0][:min_valid]

        # Extract probabilities for valid positions
        student_p = student_probs[i, student_valid_indices]  # [min_valid, student_vocab]
        teacher_p = teacher_probs[i, teacher_valid_indices]  # [min_valid, teacher_vocab]

        # Sort probabilities in descending order (removes token identity)
        student_sorted = student_p.sort(dim=-1, descending=True).values
        teacher_sorted = teacher_p.sort(dim=-1, descending=True).values

        # Pad smaller vocab to match larger
        student_vocab_size = student_sorted.size(-1)
        teacher_vocab_size = teacher_sorted.size(-1)
        max_vocab_size = max(student_vocab_size, teacher_vocab_size)

        if student_vocab_size < max_vocab_size:
            student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
        if teacher_vocab_size < max_vocab_size:
            teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

        # L1 loss on sorted distributions
        loss_i = F.l1_loss(student_sorted, teacher_sorted, reduction="mean")
        losses.append(loss_i)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


@torch.no_grad()
def compute_teacher_logits_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute teacher logits efficiently in eval mode."""
    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return outputs.logits


def compute_student_logits_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute student logits with gradients."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return outputs.logits


class HybridGOLDTrainer:
    """
    Hybrid GOLD trainer that uses vLLM for teacher generation
    and transformers for log prob computation.

    Supports both same-tokenizer and cross-tokenizer distillation:
    - Same tokenizer: Direct JSD loss on aligned logits
    - Cross tokenizer: Separate tokenization, truncate to min length
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        student_tokenizer: AutoTokenizer,
        teacher_tokenizer: AutoTokenizer | None,
        vllm_client: VLLMOpenAIClient,
        args: ScriptArguments,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer or student_tokenizer
        self.vllm_client = vllm_client
        self.args = args

        # Check if we're doing cross-tokenizer distillation
        self.cross_tokenizer = teacher_tokenizer is not None
        if self.cross_tokenizer:
            logger.info("Cross-tokenizer distillation enabled (using ULD sorted loss)")
            logger.info(f"  Student vocab size: {len(self.student_tokenizer)}")
            logger.info(f"  Teacher vocab size: {len(self.teacher_tokenizer)}")
            logger.info("  Loss: ULD sorted probability comparison (vocab-agnostic)")
        else:
            logger.info("Same-tokenizer distillation (using JSD loss)")

        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Device
        self.device = next(student_model.parameters()).device

    def generate_teacher_completions(self, prompts: list[str]) -> list[str]:
        """
        Generate completions from teacher via vLLM server.
        Uses async batch processing for high throughput.
        """
        # Format as chat messages
        messages_list = []
        for prompt in prompts:
            messages_list.append([{"role": "user", "content": prompt}])

        # Async batch generation
        result = self.vllm_client.chat(
            messages=messages_list,
            max_tokens=self.args.max_completion_length,
            temperature=self.args.generation_temperature,
            top_p=0.95,
        )

        return result["completions"]

    def _tokenize_batch(
        self,
        tokenizer: AutoTokenizer,
        prompts: list[str],
        completions: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize prompt + completion pairs for a specific tokenizer.

        Returns dict with:
            - input_ids: [batch, seq_len]
            - attention_mask: [batch, seq_len]
            - labels: [batch, seq_len] with -100 for prompt tokens
            - prompt_lengths: list of prompt lengths for each sample
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        prompt_lengths = []

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for prompt, completion in zip(prompts, completions):
            # Tokenize prompt separately to get its length
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_len = prompt_tokens["input_ids"].shape[1]
            prompt_lengths.append(prompt_len)

            # Tokenize full sequence
            full_text = prompt + completion
            full_tokens = tokenizer(
                full_text,
                add_special_tokens=True,
                max_length=self.args.max_length,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = full_tokens["input_ids"].squeeze(0)
            attention_mask = full_tokens["attention_mask"].squeeze(0)

            # Create labels: -100 for prompt, actual tokens for completion
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        # Pad to same length within this batch
        max_len = max(ids.shape[0] for ids in batch_input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(
            batch_input_ids, batch_attention_mask, batch_labels
        ):
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
                labels = F.pad(labels, (0, pad_len), value=-100)

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

        return {
            "input_ids": torch.stack(padded_input_ids).to(self.device),
            "attention_mask": torch.stack(padded_attention_mask).to(self.device),
            "labels": torch.stack(padded_labels).to(self.device),
            "prompt_lengths": prompt_lengths,
        }

    def prepare_batch_for_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        """
        Tokenize prompt + completion pairs for log prob computation.

        For same-tokenizer: Returns (student_batch, None)
        For cross-tokenizer: Returns (student_batch, teacher_batch)
        """
        # Always tokenize for student
        student_batch = self._tokenize_batch(
            self.student_tokenizer, prompts, completions
        )

        if self.cross_tokenizer:
            # Tokenize separately for teacher
            teacher_batch = self._tokenize_batch(
                self.teacher_tokenizer, prompts, completions
            )
            return student_batch, teacher_batch
        else:
            return student_batch, None

    def training_step(self, prompts: list[str]) -> torch.Tensor:
        """
        Single training step:
        1. Generate completions from teacher via vLLM
        2. Compute log probs from both models via transformers
        3. Compute JSD loss (with cross-tokenizer handling if needed)
        """
        # Phase 1: Teacher generation via vLLM
        completions = self.generate_teacher_completions(prompts)

        # Filter out failed completions
        valid_pairs = [
            (p, c) for p, c in zip(prompts, completions)
            if not c.startswith("ERROR:") and c.strip()
        ]
        if not valid_pairs:
            logger.warning("All completions failed, skipping batch")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        valid_prompts, valid_completions = zip(*valid_pairs)

        # Phase 2: Prepare batch for log prob computation
        student_batch, teacher_batch = self.prepare_batch_for_logprobs(
            list(valid_prompts), list(valid_completions)
        )

        # Phase 3: Compute logits via transformers forward pass
        if self.cross_tokenizer:
            # Cross-tokenizer: Use ULD sorted loss (compares distribution shapes)
            # This works with different vocab sizes by sorting probabilities

            # Compute teacher logits using teacher tokenization
            teacher_logits = compute_teacher_logits_batch(
                self.teacher_model,
                teacher_batch["input_ids"],
                teacher_batch["attention_mask"],
            )

            # Compute student logits using student tokenization
            self.student_model.train()
            student_logits = compute_student_logits_batch(
                self.student_model,
                student_batch["input_ids"],
                student_batch["attention_mask"],
            )

            # Shift logits and labels for next-token prediction
            shifted_student_logits = student_logits[:, :-1, :].contiguous()
            shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
            shifted_student_labels = student_batch["labels"][:, 1:].contiguous()
            shifted_teacher_labels = teacher_batch["labels"][:, 1:].contiguous()

            # Compute ULD loss (sorted probability comparison)
            loss = uld_sorted_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                student_labels=shifted_student_labels,
                teacher_labels=shifted_teacher_labels,
                temperature=self.args.temperature,
            )

        else:
            # Same tokenizer: direct computation
            teacher_logits = compute_teacher_logits_batch(
                self.teacher_model,
                student_batch["input_ids"],
                student_batch["attention_mask"],
            )

            self.student_model.train()
            student_logits = compute_student_logits_batch(
                self.student_model,
                student_batch["input_ids"],
                student_batch["attention_mask"],
            )

            # Shift logits and labels for next-token prediction
            shifted_student_logits = student_logits[:, :-1, :].contiguous()
            shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
            shifted_labels = student_batch["labels"][:, 1:].contiguous()

            # Compute JSD loss
            loss = generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                labels=shifted_labels,
                beta=self.args.beta,
                temperature=self.args.temperature,
            )

        return loss

    def train(self, train_dataset: Dataset):
        """Main training loop."""
        # Extract prompts from dataset
        if "messages" in train_dataset.column_names:
            # Chat format - extract user messages as prompts
            prompts = []
            for item in train_dataset:
                user_msgs = [m["content"] for m in item["messages"] if m["role"] == "user"]
                if user_msgs:
                    prompts.append(user_msgs[0])
        elif "prompt" in train_dataset.column_names:
            prompts = train_dataset["prompt"]
        elif "question" in train_dataset.column_names:
            prompts = train_dataset["question"]
        elif "problem" in train_dataset.column_names:
            # Math dataset format (e.g., AIME) - format as math prompt
            prompts = []
            for item in train_dataset:
                problem = item["problem"]
                # Format as a math reasoning prompt
                formatted_prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                prompts.append(formatted_prompt)
        else:
            raise ValueError(f"Unknown dataset format. Columns: {train_dataset.column_names}")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01,
        )

        # Calculate total steps
        num_batches = len(prompts) // self.args.batch_size
        total_steps = num_batches * self.args.num_epochs // self.args.gradient_accumulation_steps

        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        # Training loop
        global_step = 0
        for epoch in range(self.args.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")

            # Shuffle prompts
            import random
            shuffled_prompts = prompts.copy()
            random.shuffle(shuffled_prompts)

            epoch_loss = 0.0
            num_batches_processed = 0

            progress_bar = tqdm(
                range(0, len(shuffled_prompts), self.args.batch_size),
                desc=f"Epoch {epoch + 1}",
            )

            optimizer.zero_grad()

            for i in progress_bar:
                batch_prompts = shuffled_prompts[i:i + self.args.batch_size]
                if len(batch_prompts) < self.args.batch_size:
                    continue  # Skip incomplete batches

                # Training step
                loss = self.training_step(batch_prompts)
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * self.args.gradient_accumulation_steps
                num_batches_processed += 1

                # Gradient accumulation
                if num_batches_processed % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Update progress bar
                avg_loss = epoch_loss / num_batches_processed
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            logger.info(f"Epoch {epoch + 1} average loss: {epoch_loss / max(1, num_batches_processed):.4f}")

            # Save checkpoint
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.student_model.save_pretrained(checkpoint_dir)
            self.student_tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Save final model
        self.student_model.save_pretrained(self.args.output_dir)
        self.student_tokenizer.save_pretrained(self.args.output_dir)
        logger.info(f"Saved final model to {self.args.output_dir}")


def main():
    parser = HfArgumentParser((ScriptArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info(f"Dataset size: {len(dataset)}")

    # Load student tokenizer
    logger.info(f"Loading student tokenizer from {args.model_name_or_path}")
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # Load teacher tokenizer (if different)
    teacher_tokenizer = None
    if args.teacher_tokenizer_name_or_path:
        logger.info(f"Loading teacher tokenizer from {args.teacher_tokenizer_name_or_path}")
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_tokenizer_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Load student model
    logger.info(f"Loading student model: {args.model_name_or_path}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Load teacher model (local copy for log prob computation)
    logger.info(f"Loading teacher model: {args.teacher_model_name_or_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    teacher_model.eval()

    # Initialize vLLM client for teacher generation
    vllm_model_name = args.vllm_model_name or args.teacher_model_name_or_path
    logger.info(f"Connecting to vLLM server at {args.vllm_server_url}")
    vllm_client = VLLMOpenAIClient(
        base_url=args.vllm_server_url,
        model=vllm_model_name,
        max_concurrent_requests=args.vllm_max_concurrent,
    )

    # Create trainer
    trainer = HybridGOLDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        vllm_client=vllm_client,
        args=args,
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Generation temperature: {args.generation_temperature}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    if teacher_tokenizer is None:
        logger.info(f"  Loss: JSD with beta={args.beta}")
    else:
        logger.info("  Loss: ULD sorted (cross-tokenizer)")

    trainer.train(dataset)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
