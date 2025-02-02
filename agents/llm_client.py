import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from exceptions import LLMError
from loggers.llm_logger import LLMLogger

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")


class LLMClient:
    """Centralized interface for LLM communication with comprehensive error handling and monitoring."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        base_wait: float = 1.0,
        max_wait: float = 10.0,
        client: Optional[Any] = None,
    ):
        """Initialize LLM client with configurable retry parameters.

        Args:
            api_key: OpenAI API key
            model: Model identifier to use
            max_retries: Maximum retry attempts
            base_wait: Base delay between retries
            max_wait: Maximum delay between retries
            client: Optional pre-configured client for testing
        """
        self.model = model
        self.client = client or OpenAI(
            api_key=api_key,
            timeout=30.0,  # Add timeout of 30 seconds
        )
        self.async_client = client or AsyncOpenAI(
            api_key=api_key,
            timeout=30.0,  # Add timeout for async client too
        )
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.max_wait = max_wait

        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "failed_queries": 0,
            "retry_count": 0,
            "total_tokens": 0,
            "query_times": [],
            "rate_limit_hits": 0,
            "timeout_errors": 0,
        }

    def query(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_message: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Execute synchronous LLM query with retry logic.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_message: Optional system context

        Returns:
            str: LLM response text

        Raises:
            LLMError: If all retries fail
            ValueError: If input parameters are invalid
        """
        # Input validation
        if not prompt:
            LLMLogger.log_input_validation_error("prompt", prompt)
            raise ValueError("Prompt cannot be empty")
        if not 0 <= temperature <= 1:
            LLMLogger.log_input_validation_error("temperature", temperature)
            raise ValueError("Temperature must be between 0 and 1")
        if max_tokens < 1:
            LLMLogger.log_input_validation_error("max_tokens", max_tokens)
            raise ValueError("max_tokens must be positive")

        try:
            return self._execute_query(
                prompt, temperature, max_tokens, system_message, tags
            )
        except Exception as e:
            LLMLogger.log_query_error(e, "synchronous")
            raise LLMError(f"Query failed after retries: {str(e)}")

    @retry(
        stop=stop_after_attempt(5),  # Increase max attempts
        wait=wait_exponential(multiplier=1, min=2, max=20),  # Adjust wait times
        reraise=True,
    )
    def _execute_query(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Execute the actual query with retry logic."""
        start_time = time.time()
        self.metrics["total_queries"] += 1

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response_text = response.choices[0].message.content

            # Add a small delay after successful response (0.5 seconds)
            time.sleep(0.5)

            # Log the prompt and response with tags
            LLMLogger.log_prompt_and_response(
                prompt=prompt,
                response=response_text,
                system_message=system_message,
                model=self.model,
                tags=", ".join(tags) if tags else None,  # Join list of tags into string
            )

            # Update metrics
            duration = time.time() - start_time
            self.metrics["query_times"].append(duration)
            self.metrics["total_tokens"] += response.usage.total_tokens

            LLMLogger.log_metrics_update(duration, response.usage.total_tokens)
            return response_text

        except (APITimeoutError, APIError) as e:
            self.metrics["retry_count"] += 1
            error_type = "timeout" if isinstance(e, APITimeoutError) else "api_error"
            LLMLogger.log_query_error(f"{error_type}: {str(e)}")

            # Track timeouts separately
            self.metrics.setdefault("timeout_errors", 0)
            self.metrics["timeout_errors"] += 1

            LLMLogger.log_metrics_update(
                time.time() - start_time,
                0,
                success=False,
                error=f"{error_type} occurred. Waiting before retry. Attempt {self.metrics['retry_count']}",
            )
            raise  # Let tenacity handle the retry

        except RateLimitError as e:
            self.metrics["retry_count"] += 1
            LLMLogger.log_query_error(f"Rate limit exceeded: {str(e)}")

            # Add rate limit specific metrics
            self.metrics.setdefault("rate_limit_hits", 0)
            self.metrics["rate_limit_hits"] += 1

            # Log the rate limit hit with more detail
            LLMLogger.log_metrics_update(
                time.time() - start_time,
                0,
                success=False,
                error=f"Rate limit hit. Waiting before retry. Attempt {self.metrics['retry_count']}",
            )
            raise  # Let tenacity handle the retry with exponential backoff

        except Exception as e:
            # Track retry count and failed query for each attempt
            self.metrics["retry_count"] += 1
            self.metrics["failed_queries"] += 1
            LLMLogger.log_query_error(e)

            LLMLogger.log_metrics_update(
                time.time() - start_time, 0, success=False, error=e
            )

            raise  # Let tenacity handle the retry

    async def query_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_message: Optional[str] = None,
    ) -> str:
        """Execute asynchronous LLM query with retry logic."""
        start_time = time.time()
        self.metrics["total_queries"] += 1

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Add a small delay after successful response
            await asyncio.sleep(0.5)

            # Update metrics
            duration = time.time() - start_time
            self.metrics["query_times"].append(duration)
            self.metrics["total_tokens"] += response.usage.total_tokens

            LLMLogger.log_metrics_update(duration, response.usage.total_tokens)
            return response.choices[0].message.content

        except Exception as e:
            self.metrics["failed_queries"] += 1
            LLMLogger.log_async_query_error(e)
            LLMLogger.log_metrics_update(
                time.time() - start_time, 0, success=False, error=e
            )
            raise

    def get_metrics(self) -> Dict[str, Union[int, float, list]]:
        """Return current metrics."""
        metrics = self.metrics.copy()
        if self.metrics["query_times"]:
            metrics["average_query_time"] = sum(self.metrics["query_times"]) / len(
                self.metrics["query_times"]
            )
        # Add rate limit and timeout information
        if "rate_limit_hits" in self.metrics:
            metrics["rate_limit_percentage"] = (
                self.metrics["rate_limit_hits"] / self.metrics["total_queries"] * 100
                if self.metrics["total_queries"] > 0
                else 0
            )
        if "timeout_errors" in self.metrics:
            metrics["timeout_percentage"] = (
                self.metrics["timeout_errors"] / self.metrics["total_queries"] * 100
                if self.metrics["total_queries"] > 0
                else 0
            )
        return metrics

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self.metrics = {
            "total_queries": 0,
            "failed_queries": 0,
            "retry_count": 0,
            "total_tokens": 0,
            "query_times": [],
            "rate_limit_hits": 0,
            "timeout_errors": 0,
        }

    def reasoning_query(
        self,
        prompt: str,
        reasoning_type: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_message: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Execute a reasoning-specific LLM query with retry logic.

        Args:
            prompt: The prompt to send
            reasoning_type: Type of reasoning (deductive, inductive, probabilistic)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_message: Optional system context

        Returns:
            str: LLM response text

        Raises:
            LLMError: If all retries fail
            ValueError: If input parameters are invalid
        """
        # Input validation
        if not prompt:
            LLMLogger.log_input_validation_error("prompt", prompt)
            raise ValueError("Prompt cannot be empty")
        if not 0 <= temperature <= 1:
            LLMLogger.log_input_validation_error("temperature", temperature)
            raise ValueError("Temperature must be between 0 and 1")
        if max_tokens < 1:
            LLMLogger.log_input_validation_error("max_tokens", max_tokens)
            raise ValueError("max_tokens must be positive")

        try:
            return self._execute_reasoning_query(
                prompt, reasoning_type, temperature, max_tokens, system_message, tags
            )
        except Exception as e:
            LLMLogger.log_query_error(e, "reasoning")
            raise LLMError(f"Reasoning query failed after retries: {str(e)}")

    @retry(
        stop=stop_after_attempt(5),  # Increase max attempts
        wait=wait_exponential(multiplier=1, min=2, max=20),  # Adjust wait times
        reraise=True,
    )
    def _execute_reasoning_query(
        self,
        prompt: str,
        reasoning_type: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Execute the actual reasoning query with retry logic."""
        start_time = time.time()
        self.metrics["total_queries"] += 1

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": f"{reasoning_type}: {prompt}"})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response_text = response.choices[0].message.content

            # Add a small delay after successful response (0.5 seconds)
            time.sleep(0.5)

            # Log the prompt and response with tags
            LLMLogger.log_prompt_and_response(
                prompt=prompt,
                response=response_text,
                system_message=system_message,
                model=self.model,
                tags=", ".join(tags) if tags else None,  # Join list of tags into string
            )

            # Update metrics
            duration = time.time() - start_time
            self.metrics["query_times"].append(duration)
            self.metrics["total_tokens"] += response.usage.total_tokens

            LLMLogger.log_metrics_update(duration, response.usage.total_tokens)
            return response_text

        except (APITimeoutError, APIError) as e:
            self.metrics["retry_count"] += 1
            error_type = "timeout" if isinstance(e, APITimeoutError) else "api_error"
            LLMLogger.log_query_error(f"{error_type}: {str(e)}")

            # Track timeouts separately
            self.metrics.setdefault("timeout_errors", 0)
            self.metrics["timeout_errors"] += 1

            LLMLogger.log_metrics_update(
                time.time() - start_time,
                0,
                success=False,
                error=f"{error_type} occurred. Waiting before retry. Attempt {self.metrics['retry_count']}",
            )
            raise  # Let tenacity handle the retry

        except RateLimitError as e:
            self.metrics["retry_count"] += 1
            LLMLogger.log_query_error(f"Rate limit exceeded: {str(e)}")

            # Add rate limit specific metrics
            self.metrics.setdefault("rate_limit_hits", 0)
            self.metrics["rate_limit_hits"] += 1

            # Log the rate limit hit with more detail
            LLMLogger.log_metrics_update(
                time.time() - start_time,
                0,
                success=False,
                error=f"Rate limit hit. Waiting before retry. Attempt {self.metrics['retry_count']}",
            )
            raise  # Let tenacity handle the retry with exponential backoff

        except Exception as e:
            # Track retry count and failed query for each attempt
            self.metrics["retry_count"] += 1
            self.metrics["failed_queries"] += 1
            LLMLogger.log_query_error(e)

            LLMLogger.log_metrics_update(
                time.time() - start_time, 0, success=False, error=e
            )

            raise  # Let tenacity handle the retry
