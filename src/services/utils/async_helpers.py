"""
Asynchronous utilities for service operations.

This module provides helper functions and decorators for working with
asynchronous code, including retries, timeouts, and concurrency control.
"""

import asyncio
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def async_retry(max_retries=3, backoff_factor=1.5, circuit_breaker=None):
    """
    Decorator to retry async functions with exponential backoff.
    Integrates with circuit breaker for fault tolerance.

    Args:
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (float): Factor to increase wait time between retries.
        circuit_breaker (CircuitBreaker): Optional circuit breaker instance.

    Returns:
        callable: Decorated function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            last_exception = None

            while retry_count < max_retries:
                # Check circuit breaker
                if circuit_breaker and circuit_breaker.is_open():
                    logger.warning(f"Circuit open, skipping call to {func.__name__}")
                    raise RuntimeError(f"Service unavailable: circuit is open for {func.__name__}")

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Log execution time for monitoring
                    logger.debug(f"Function {func.__name__} completed: {execution_time:.4f}s")

                    # Record success to circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result
                except (TimeoutError, ConnectionError) as e:
                    # Record failure to circuit breaker for specific errors
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    last_exception = e

                    logger.warning(
                        f"Retry {retry_count}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.2f}s - Cause: {type(e).__name__}: {str(e)}"
                    )

                    # Wait before retrying
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.warning(f"async_retry exception: {e}")
                    # For other exceptions, record failure but don't retry
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    raise

            # If we get here, all retries failed
            logger.error(f"All {max_retries} retries failed for {func.__name__}")
            raise last_exception or RuntimeError(f"All retries failed for {func.__name__}")

        return wrapper

    return decorator


async def limited_task_semaphore(semaphore, func, *args, timeout=60, **kwargs):
    """
    Execute a function with semaphore-based concurrency control and timeout.

    Args:
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.
        func (callable): Function to execute.
        *args: Arguments for the function.
        timeout (int): Maximum execution time in seconds.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: Result of the function.

    Raises:
        TimeoutError: If the function exceeds the timeout period.
        Exception: If the function fails.
    """
    start_time = time.time()

    async with semaphore:
        try:
            if asyncio.iscoroutinefunction(func):
                # Apply timeout to async call
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            else:
                # For sync calls, use wait_for with a task wrapper
                return await asyncio.wait_for(asyncio.create_task(func(*args, **kwargs)), timeout=timeout)
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"Function invocation timed out: {elapsed:.2f}s (limit: {timeout}s)")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Function invocation failed: {elapsed:.2f}s after: {type(e).__name__}: {str(e)}")
            raise
