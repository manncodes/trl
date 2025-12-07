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
OpenAI-compatible vLLM client for GOLD training.

This client communicates with vLLM servers that expose the OpenAI-compatible API
(e.g., `/v1/completions`, `/v1/chat/completions`).
"""

import logging
import time
from typing import Any

from ..import_utils import is_openai_available


if is_openai_available():
    from openai import OpenAI


logger = logging.getLogger(__name__)


class VLLMOpenAIClient:
    """
    A client class to interact with a vLLM server using the OpenAI-compatible API.

    This client is useful when your vLLM server exposes the standard OpenAI API endpoints
    (e.g., `/v1/completions`, `/v1/chat/completions`) rather than the TRL-specific endpoints.

    Args:
        base_url (`str`):
            Base URL for the vLLM server (e.g., `"http://localhost:8000/v1"`).
        api_key (`str`, *optional*, defaults to `"EMPTY"`):
            API key for authentication. vLLM typically doesn't require a real key.
        model (`str`, *optional*, defaults to `None`):
            Model name to use for requests. If None, must be specified in each request.
        connection_timeout (`float`, *optional*, defaults to `60.0`):
            Timeout for connecting to the server.
        max_retries (`int`, *optional*, defaults to `3`):
            Maximum number of retries for failed requests.

    Examples:
        ```python
        >>> from trl.extras.vllm_openai_client import VLLMOpenAIClient
        >>> client = VLLMOpenAIClient(
        ...     base_url="http://my-vllm-server:8000/v1",
        ...     model="openai/gpt-oss-120b"
        ... )
        >>> response = client.generate(["Hello, world!"], max_tokens=100)
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str | None = None,
        connection_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        if not is_openai_available():
            raise ImportError(
                "OpenAI package is not installed. Please install it with `pip install openai`."
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=connection_timeout,
            max_retries=max_retries,
        )

        # Verify connection
        self._check_connection()

    def _check_connection(self, retry_interval: float = 2.0, max_retries: int = 5):
        """Check if the server is reachable."""
        for attempt in range(max_retries):
            try:
                # Try to list models to verify connection
                self.client.models.list()
                logger.info(f"Connected to vLLM server at {self.base_url}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_interval)
                else:
                    raise ConnectionError(
                        f"Failed to connect to vLLM server at {self.base_url} after {max_retries} attempts: {e}"
                    )

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        n: int = 1,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        echo: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate completions for the given prompts using the OpenAI-compatible API.

        Args:
            prompts (`list[str]`):
                List of prompts to generate completions for.
            max_tokens (`int`, *optional*, defaults to `256`):
                Maximum number of tokens to generate.
            temperature (`float`, *optional*, defaults to `0.7`):
                Sampling temperature.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. -1 means disabled.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate per prompt.
            stop (`list[str]`, *optional*):
                Stop sequences.
            logprobs (`int`, *optional*):
                Number of top log probabilities to return per token.
            echo (`bool`, *optional*, defaults to `False`):
                Whether to echo the prompt in the response.
            **kwargs:
                Additional parameters to pass to the API.

        Returns:
            `dict` with keys:
                - `completions` (`list[str]`): Generated completion texts.
                - `completion_ids` (`list[list[int]]`): Token IDs of completions (if available).
                - `logprobs` (`list[list[float]]`): Log probabilities (if requested).
        """
        if self.model is None and "model" not in kwargs:
            raise ValueError("Model must be specified either in constructor or in generate() call")

        model = kwargs.pop("model", self.model)

        # Build extra parameters for vLLM
        extra_body = {}
        if top_k > 0:
            extra_body["top_k"] = top_k

        all_completions = []
        all_logprobs = []

        for prompt in prompts:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                logprobs=logprobs,
                echo=echo,
                extra_body=extra_body if extra_body else None,
                **kwargs,
            )

            for choice in response.choices:
                all_completions.append(choice.text)
                if choice.logprobs is not None:
                    # Extract token logprobs
                    token_logprobs = choice.logprobs.token_logprobs or []
                    all_logprobs.append(token_logprobs)

        return {
            "completions": all_completions,
            "logprobs": all_logprobs if all_logprobs else None,
        }

    def chat(
        self,
        messages: list[list[dict[str, str]]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate chat completions for the given message lists.

        Args:
            messages (`list[list[dict]]`):
                List of conversation message lists. Each inner list is a conversation.
            max_tokens (`int`, *optional*, defaults to `256`):
                Maximum number of tokens to generate.
            temperature (`float`, *optional*, defaults to `0.7`):
                Sampling temperature.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate per conversation.
            stop (`list[str]`, *optional*):
                Stop sequences.
            logprobs (`bool`, *optional*, defaults to `False`):
                Whether to return log probabilities.
            top_logprobs (`int`, *optional*):
                Number of top log probabilities to return.
            **kwargs:
                Additional parameters to pass to the API.

        Returns:
            `dict` with keys:
                - `completions` (`list[str]`): Generated completion texts.
                - `logprobs` (`list`): Log probability information (if requested).
        """
        if self.model is None and "model" not in kwargs:
            raise ValueError("Model must be specified either in constructor or in chat() call")

        model = kwargs.pop("model", self.model)

        all_completions = []
        all_logprobs = []

        for message_list in messages:
            response = self.client.chat.completions.create(
                model=model,
                messages=message_list,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                **kwargs,
            )

            for choice in response.choices:
                all_completions.append(choice.message.content)
                if choice.logprobs is not None:
                    all_logprobs.append(choice.logprobs)

        return {
            "completions": all_completions,
            "logprobs": all_logprobs if all_logprobs else None,
        }

    def get_teacher_completions(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate teacher completions for distillation.

        This is a convenience method for generating high-quality completions
        from a teacher model for knowledge distillation.

        Args:
            prompts (`list[str]`):
                List of prompts (e.g., math problems) to solve.
            max_tokens (`int`, *optional*, defaults to `2048`):
                Maximum tokens for reasoning.
            temperature (`float`, *optional*, defaults to `0.7`):
                Sampling temperature.
            top_p (`float`, *optional*, defaults to `0.95`):
                Top-p sampling parameter.
            system_prompt (`str`, *optional*):
                System prompt to use.
            **kwargs:
                Additional parameters.

        Returns:
            `dict` with:
                - `completions` (`list[str]`): Teacher's solutions.
                - `prompts` (`list[str]`): Original prompts.
        """
        messages_list = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        result = self.chat(
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        return {
            "completions": result["completions"],
            "prompts": prompts,
        }
