"""Utility module - NOT a test file.

Expected: file_type=code, language=python, is_test=false
This file is named utils.py, not test_*.py, and is not in a tests/ directory.
"""

from typing import TypeVar, Callable, Any
from functools import wraps
import time

T = TypeVar("T")


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator


def chunk_list(items: list[T], size: int) -> list[list[T]]:
    """Split a list into chunks of the given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten(nested: list[list[T]]) -> list[T]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested for item in sublist]
