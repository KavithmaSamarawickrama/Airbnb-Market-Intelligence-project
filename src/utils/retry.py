"""
Retry and backoff utilities for network-resilient operations.

Implements exponential backoff with jitter for transient failures.
"""

import time
import logging
import random
from typing import Callable, TypeVar, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


def exponential_backoff(
    func: Callable[..., T],
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Callable to retry
        max_retries: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Cap on delay (to avoid excessive waiting)
        jitter: Add random jitter to delays

    Returns:
        Function result

    Raises:
        Raises the last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except (TimeoutError, ConnectionError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                if jitter:
                    delay = delay * (0.5 + random.random())
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries} retries exhausted for {func.__name__}: {e}"
                )
    raise last_exception


def retry_on_transient(
    max_retries: int = 4, base_delay: float = 1.0, jitter: bool = True
):
    """
    Decorator for automatic retry logic on transient failures.

    Args:
        max_retries: Maximum number of attempts
        base_delay: Initial delay in seconds
        jitter: Add random jitter to delays

    Returns:
        Decorated function with retry capability
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, OSError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if jitter:
                            delay = delay * (0.5 + random.random())
                        logger.warning(
                            f"[{func.__name__}] Attempt {attempt + 1}/{max_retries} failed: "
                            f"{type(e).__name__}. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{func.__name__}] All retries exhausted: {e}"
                        )
            raise last_exception

        return wrapper

    return decorator
