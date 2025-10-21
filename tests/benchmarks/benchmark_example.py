"""
Example benchmark demonstrating pytest-benchmark usage.

This file serves as a template for writing performance benchmarks.
Run with: uv run pytest tests/benchmarks/ --benchmark-only
"""

import pytest


def fibonacci(n: int) -> int:
    """Simple recursive fibonacci for demonstration."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Iterative fibonacci implementation."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


@pytest.mark.benchmark
def test_fibonacci_recursive(benchmark):
    """Benchmark recursive fibonacci calculation."""
    result = benchmark(fibonacci, 15)
    assert result == 610


@pytest.mark.benchmark
def test_fibonacci_iterative(benchmark):
    """Benchmark iterative fibonacci calculation."""
    result = benchmark(fibonacci_iterative, 15)
    assert result == 610


@pytest.mark.benchmark
def test_string_concatenation(benchmark):
    """Benchmark string concatenation performance."""

    def concat_strings():
        result = ""
        for i in range(100):
            result += str(i)
        return result

    result = benchmark(concat_strings)
    assert len(result) > 0


@pytest.mark.benchmark
def test_list_comprehension(benchmark):
    """Benchmark list comprehension performance."""

    def list_comp():
        return [i**2 for i in range(1000)]

    result = benchmark(list_comp)
    assert len(result) == 1000
