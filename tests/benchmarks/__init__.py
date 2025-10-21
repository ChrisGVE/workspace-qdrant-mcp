"""
Performance benchmarks for workspace-qdrant-mcp.

This package contains benchmarks for measuring and tracking
performance of critical components.

Run benchmarks with:
    uv run pytest tests/benchmarks/ --benchmark-only

View detailed statistics:
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-verbose

Save baseline:
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

Compare against baseline:
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline
"""
