"""
Web Application LLM Adapter.

This module provides an adapter that bridges LangRAG's LLM interface with
the Web Application's OpenAI-compatible LLM client.

Features:
- Configurable timeouts
- Automatic retry with exponential backoff
- Proper error classification
- Comprehensive logging for debugging
"""

import json
import logging
import time
from typing import Any, Generator

import httpx
from openai import AsyncOpenAI

from langrag.errors import (
    AuthenticationError,
    RateLimitError,
    ReadTimeoutError,
    ServiceUnavailableError,
    TimeoutError,
    classify_http_error,
    wrap_exception,
)
from langrag.llm.base import BaseLLM
from langrag.llm.config import LLMConfig, TimeoutConfig
from langrag.utils.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)


class WebLLMAdapter(BaseLLM):
    """
    Adapter that wraps Web App's LLM client to match LangRAG's BaseLLM interface.

    This allows LangRAG core components (Router, Rewriter) to use the LLM managed
    by the Web Application.

    Features:
    - Configurable timeouts for connect/read operations
    - Automatic retry on transient errors with exponential backoff
    - Proper error classification for debugging
    - Comprehensive logging

    Example:
        adapter = WebLLMAdapter(
            client=async_openai_client,
            model="gpt-4",
            config=LLMConfig.resilient()
        )
        response = adapter.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        config: LLMConfig | None = None
    ):
        """
        Initialize the LLM adapter.

        Args:
            client: AsyncOpenAI client from the web application
            model: Model identifier to use for completions
            config: LLM configuration (timeout, retry settings)
        """
        self.client = client
        self.model = model
        self.config = config or LLMConfig.default()

        # Extract connection info from async client
        self.base_url = str(client.base_url)
        self.api_key = client.api_key

        # Build retry configuration from LLM config
        self._retry_config = RetryConfig(
            max_attempts=self.config.max_retries,
            base_delay=self.config.retry_base_delay,
            max_delay=self.config.retry_max_delay,
            exponential_base=self.config.retry_exponential_base,
            on_retry=self._on_retry,
        )

        logger.info(
            f"WebLLMAdapter initialized: model={model}, "
            f"base_url={self.base_url}, "
            f"timeout={self.config.timeout.total}s, "
            f"max_retries={self.config.max_retries}"
        )

    def _on_retry(self, attempt: int, error: Exception, delay: float) -> None:
        """Callback invoked before each retry attempt."""
        logger.warning(
            f"[WebLLMAdapter] Retry {attempt}/{self.config.max_retries} "
            f"after {type(error).__name__}: {error}. "
            f"Waiting {delay:.2f}s before next attempt."
        )

    def _get_headers(self) -> dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_timeout(self) -> httpx.Timeout:
        """Build httpx Timeout from config."""
        return httpx.Timeout(
            connect=self.config.timeout.connect,
            read=self.config.timeout.read,
            write=30.0,  # Fixed write timeout
            pool=None  # No pool timeout
        )

    def _handle_http_error(
        self,
        response: httpx.Response,
        context: str
    ) -> None:
        """
        Handle HTTP error responses by raising appropriate exceptions.

        Args:
            response: The HTTP response object
            context: Context string for logging

        Raises:
            Appropriate LangRAGError subclass based on status code
        """
        status_code = response.status_code

        # Try to extract error message from response
        try:
            error_body = response.json()
            error_message = error_body.get("error", {}).get("message", response.text)
        except (json.JSONDecodeError, KeyError):
            error_message = response.text[:500] if response.text else f"HTTP {status_code}"

        # Get headers for Retry-After
        headers = dict(response.headers)

        logger.error(
            f"[WebLLMAdapter] {context} failed: "
            f"status={status_code}, message={error_message[:200]}"
        )

        raise classify_http_error(status_code, error_message, headers)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed documents (not implemented in this adapter).

        Use a dedicated Embedder for embedding operations.
        """
        logger.debug("embed_documents called on WebLLMAdapter (returns empty)")
        return []

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query (not implemented in this adapter).

        Use a dedicated Embedder for embedding operations.
        """
        logger.debug("embed_query called on WebLLMAdapter (returns empty)")
        return []

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """
        Synchronous chat completion with retry and timeout support.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            **kwargs: Additional parameters
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Maximum tokens in response

        Returns:
            The assistant's response content

        Raises:
            RateLimitError: When API rate limit is exceeded
            TimeoutError: When request times out
            AuthenticationError: When API key is invalid
            ServiceUnavailableError: When service is temporarily unavailable
        """
        request_id = f"chat_{int(time.time() * 1000) % 100000}"

        logger.debug(
            f"[{request_id}] Chat request: model={self.model}, "
            f"messages={len(messages)}, temperature={kwargs.get('temperature', 0.0)}"
        )

        @retry_with_backoff(config=self._retry_config)
        def _do_chat() -> str:
            return self._execute_chat(messages, request_id, **kwargs)

        start_time = time.time()
        try:
            result = _do_chat()
            elapsed = time.time() - start_time

            logger.info(
                f"[{request_id}] Chat completed: "
                f"elapsed={elapsed:.2f}s, response_length={len(result)}"
            )
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] Chat failed after {elapsed:.2f}s: "
                f"{type(e).__name__}: {e}"
            )
            raise

    def _execute_chat(
        self,
        messages: list[dict],
        request_id: str,
        **kwargs: Any
    ) -> str:
        """
        Execute a single chat request.

        This is the internal method that performs the actual HTTP request.
        It's called by chat() with retry wrapper.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
        }

        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        try:
            response = httpx.post(
                f"{self.base_url}chat/completions",
                json=payload,
                headers=self._get_headers(),
                timeout=self._build_timeout()
            )

            # Handle non-2xx responses
            if response.status_code >= 400:
                self._handle_http_error(response, f"[{request_id}] chat")

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return content

        except httpx.TimeoutException as e:
            logger.warning(f"[{request_id}] Request timed out: {e}")
            raise wrap_exception(e, context=f"[{request_id}] chat request")

        except httpx.ConnectError as e:
            logger.warning(f"[{request_id}] Connection failed: {e}")
            raise wrap_exception(e, context=f"[{request_id}] connecting to LLM service")

        except httpx.HTTPStatusError as e:
            # This shouldn't be reached due to earlier handling, but just in case
            self._handle_http_error(e.response, f"[{request_id}] chat")
            raise  # Will never reach here

    def stream_chat(
        self,
        messages: list[dict],
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Streaming chat completion.

        Yields content chunks as they arrive from the LLM.

        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
                - temperature: Sampling temperature

        Yields:
            Content chunks (strings)

        Raises:
            RateLimitError: When API rate limit is exceeded
            TimeoutError: When request or stream times out
            AuthenticationError: When API key is invalid

        Note:
            Streaming does not use retry logic as partial responses
            cannot be safely resumed. Callers should implement their
            own retry logic if needed.
        """
        request_id = f"stream_{int(time.time() * 1000) % 100000}"

        logger.debug(
            f"[{request_id}] Stream chat request: model={self.model}, "
            f"messages={len(messages)}"
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "stream": True
        }

        start_time = time.time()
        chunks_received = 0
        total_content_length = 0

        try:
            with httpx.Client(timeout=self._build_timeout()) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                ) as response:
                    # Check for error response
                    if response.status_code >= 400:
                        # Need to read the body for error details
                        response.read()
                        self._handle_http_error(response, f"[{request_id}] stream_chat")

                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            logger.debug(f"[{request_id}] Stream completed: [DONE]")
                            break

                        try:
                            chunk = json.loads(data_str)
                            content = chunk["choices"][0]["delta"].get("content", "")

                            if content:
                                chunks_received += 1
                                total_content_length += len(content)
                                yield content

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"[{request_id}] Malformed JSON in stream: {e}"
                            )
                            continue
                        except (KeyError, IndexError) as e:
                            logger.warning(
                                f"[{request_id}] Unexpected chunk structure: {e}"
                            )
                            continue

            elapsed = time.time() - start_time
            logger.info(
                f"[{request_id}] Stream completed: "
                f"chunks={chunks_received}, "
                f"total_chars={total_content_length}, "
                f"elapsed={elapsed:.2f}s"
            )

        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] Stream timed out after {elapsed:.2f}s, "
                f"chunks_received={chunks_received}: {e}"
            )
            raise ReadTimeoutError(
                message=f"Stream read timed out after {elapsed:.2f}s",
                timeout=self.config.timeout.read,
                original_error=e
            )

        except httpx.ConnectError as e:
            logger.error(f"[{request_id}] Stream connection failed: {e}")
            raise wrap_exception(e, context=f"[{request_id}] connecting for stream")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] Stream failed after {elapsed:.2f}s: "
                f"{type(e).__name__}: {e}"
            )
            raise
