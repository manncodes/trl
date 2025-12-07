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
OpenAI-compatible vLLM client for GOLD training with async batch support.

This client communicates with vLLM servers that expose the OpenAI-compatible API
(e.g., `/v1/completions`, `/v1/chat/completions`) with concurrent async requests.
"""

import asyncio
import logging
import time
from typing import Any

from ..import_utils import is_openai_available

try:
    from tqdm.asyncio import tqdm_asyncio
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import nest_asyncio
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False


if is_openai_available():
    from openai import AsyncOpenAI, OpenAI


logger = logging.getLogger(__name__)


class VLLMOpenAIClient:
    """
    A client class to interact with a vLLM server using the OpenAI-compatible API.

    This client is useful when your vLLM server exposes the standard OpenAI API endpoints
    (e.g., `/v1/completions`, `/v1/chat/completions`) rather than the TRL-specific endpoints.

    Supports async batch processing for high throughput.

    Args:
        base_url (`str`):
            Base URL for the vLLM server (e.g., `"http://localhost:8000/v1"`).
        api_key (`str`, *optional*, defaults to `"EMPTY"`):
            API key for authentication. vLLM typically doesn't require a real key.
        model (`str`, *optional*, defaults to `None`):
            Model name to use for requests. If None, must be specified in each request.
        connection_timeout (`float`, *optional*, defaults to `120.0`):
            Timeout for connecting to the server.
        max_retries (`int`, *optional*, defaults to `3`):
            Maximum number of retries for failed requests.
        max_concurrent_requests (`int`, *optional*, defaults to `32`):
            Maximum number of concurrent async requests.

    Examples:
        ```python
        >>> from trl.extras.vllm_openai_client import VLLMOpenAIClient
        >>> client = VLLMOpenAIClient(
        ...     base_url="http://my-vllm-server:8000/v1",
        ...     model="openai/gpt-oss-120b",
        ...     max_concurrent_requests=64
        ... )
        >>> response = client.generate(["Hello, world!"], max_tokens=100)
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str | None = None,
        connection_timeout: float = 120.0,
        max_retries: int = 3,
        max_concurrent_requests: int = 32,
    ):
        if not is_openai_available():
            raise ImportError(
                "OpenAI package is not installed. Please install it with `pip install openai`."
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests

        # Sync client for connection check
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=connection_timeout,
            max_retries=max_retries,
        )

        # Async client for batch processing
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=connection_timeout,
            max_retries=max_retries,
        )

        # Semaphore for controlling concurrency
        self._semaphore = None

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

    async def _async_chat_single(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        logprobs: bool,
        top_logprobs: int | None,
        semaphore: asyncio.Semaphore,
        idx: int,
        max_retries: int = 3,
        **kwargs,
    ) -> tuple[int, str, Any]:
        """Single async chat completion with semaphore for concurrency control and retries."""
        async with semaphore:
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = await self.async_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        **kwargs,
                    )
                    content = response.choices[0].message.content

                    # Handle null/empty content - retry
                    if content is None or content.strip() == "":
                        last_error = "Empty response from model"
                        if attempt < max_retries - 1:
                            logger.warning(f"Request {idx} got empty response, retrying ({attempt + 1}/{max_retries})...")
                            await asyncio.sleep(1.0 * (attempt + 1))  # backoff
                            continue
                        else:
                            logger.error(f"Request {idx} failed after {max_retries} attempts: {last_error}")
                            return (idx, f"ERROR: {last_error}", None)

                    lp = response.choices[0].logprobs if logprobs else None
                    return (idx, content, lp)
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        logger.warning(f"Request {idx} failed: {e}, retrying ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(1.0 * (attempt + 1))  # backoff
                    else:
                        logger.error(f"Request {idx} failed after {max_retries} attempts: {e}")
                        return (idx, f"ERROR: {e}", None)

            return (idx, f"ERROR: {last_error}", None)

    async def _async_completion_single(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        logprobs: int | None,
        echo: bool,
        extra_body: dict | None,
        semaphore: asyncio.Semaphore,
        idx: int,
        **kwargs,
    ) -> tuple[int, str, Any]:
        """Single async completion with semaphore for concurrency control."""
        async with semaphore:
            try:
                response = await self.async_client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    logprobs=logprobs,
                    echo=echo,
                    extra_body=extra_body,
                    **kwargs,
                )
                text = response.choices[0].text
                lp = response.choices[0].logprobs.token_logprobs if response.choices[0].logprobs else None
                return (idx, text, lp)
            except Exception as e:
                logger.error(f"Request {idx} failed: {e}")
                return (idx, f"ERROR: {e}", None)

    def _run_async(self, coro):
        """Run an async coroutine, handling event loop properly."""
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context, need to use nest_asyncio or run in thread
            if NEST_ASYNCIO_AVAILABLE:
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            else:
                # Fallback: run in a new thread with its own event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(coro)

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
        Generate completions for the given prompts using async batch processing.

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
                - `logprobs` (`list[list[float]]`): Log probabilities (if requested).
        """
        if self.model is None and "model" not in kwargs:
            raise ValueError("Model must be specified either in constructor or in generate() call")

        model = kwargs.pop("model", self.model)

        # Build extra parameters for vLLM
        extra_body = {}
        if top_k > 0:
            extra_body["top_k"] = top_k

        async def run_batch():
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            tasks = [
                self._async_completion_single(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    logprobs=logprobs,
                    echo=echo,
                    extra_body=extra_body if extra_body else None,
                    semaphore=semaphore,
                    idx=i,
                    **kwargs,
                )
                for i, prompt in enumerate(prompts)
            ]
            if TQDM_AVAILABLE:
                return await tqdm_asyncio.gather(*tasks, desc="Generating completions")
            return await asyncio.gather(*tasks)

        logger.info(f"Starting async batch of {len(prompts)} requests with max {self.max_concurrent_requests} concurrent")
        start_time = time.time()

        results = self._run_async(run_batch())

        elapsed = time.time() - start_time
        logger.info(f"Completed {len(prompts)} requests in {elapsed:.2f}s ({len(prompts)/elapsed:.2f} req/s)")

        # Sort by index to maintain order
        results = sorted(results, key=lambda x: x[0])

        all_completions = [r[1] for r in results]
        all_logprobs = [r[2] for r in results if r[2] is not None]

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
        Generate chat completions using async batch processing.

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

        async def run_batch():
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            tasks = [
                self._async_chat_single(
                    messages=msg_list,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    semaphore=semaphore,
                    idx=i,
                    **kwargs,
                )
                for i, msg_list in enumerate(messages)
            ]
            if TQDM_AVAILABLE:
                return await tqdm_asyncio.gather(*tasks, desc="Generating chat completions")
            return await asyncio.gather(*tasks)

        logger.info(f"Starting async batch of {len(messages)} requests with max {self.max_concurrent_requests} concurrent")
        start_time = time.time()

        results = self._run_async(run_batch())

        elapsed = time.time() - start_time
        logger.info(f"Completed {len(messages)} requests in {elapsed:.2f}s ({len(messages)/elapsed:.2f} req/s)")

        # Sort by index to maintain order
        results = sorted(results, key=lambda x: x[0])

        all_completions = [r[1] for r in results]
        all_logprobs = [r[2] for r in results if r[2] is not None]

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
        Generate teacher completions for distillation using async batch processing.

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
