"""
Context injection latency benchmarks for the LLM Context Injection System.

Measures performance of context injection operations including trigger detection,
context retrieval from vector database, token counting and tracking, context
formatting, and end-to-end injection pipeline latency.

Run with: uv run pytest tests/benchmarks/benchmark_context_injection.py --benchmark-only
"""

import asyncio
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from common.core.context_injection import (
    ClaudeMdFileTrigger,
    RuleFilter,
    RuleRetrieval,
    SessionTrigger,
    TokenBudgetManager,
    TokenCounter,
    TokenizerType,
    TriggerContext,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
)
from common.core.context_injection.formatters import FormatManager
from common.core.memory import AuthorityLevel, MemoryCategory, MemoryManager, MemoryRule
from common.core.ssl_config import suppress_qdrant_ssl_warnings


class ContextInjectionBenchmarkFixtures:
    """Helper class for setting up context injection benchmark test data."""

    @staticmethod
    async def create_memory_collection(
        client: QdrantClient,
        collection_name: str,
        num_rules: int = 20,
    ) -> None:
        """
        Create a memory collection with sample rules for benchmarking.

        Args:
            client: Qdrant client instance
            collection_name: Name for the memory collection
            num_rules: Number of sample memory rules to create
        """
        # Delete collection if it already exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        # Create collection with dense vectors only (memory rules don't use sparse)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=models.Distance.COSINE,
                )
            },
        )

        # Sample rule content templates
        rule_templates = [
            "Always use type hints in Python function signatures for better code clarity",
            "Follow PEP 8 style guidelines for consistent code formatting",
            "Write comprehensive docstrings for all public functions and classes",
            "Use async/await patterns for I/O-bound operations to improve performance",
            "Implement proper error handling with specific exception types",
            "Use dataclasses for simple data structures to reduce boilerplate",
            "Apply dependency injection for better testability and modularity",
            "Write unit tests for all core business logic with edge case coverage",
            "Use logging instead of print statements for production code",
            "Implement circuit breakers for external service calls to prevent cascading failures",
        ]

        categories = [
            MemoryCategory.PREFERENCE,
            MemoryCategory.BEHAVIOR,
            MemoryCategory.AGENT,
        ]

        authorities = [
            AuthorityLevel.ABSOLUTE,
            AuthorityLevel.DEFAULT,
            AuthorityLevel.SUGGESTION,
        ]

        # Generate sample rules
        points = []
        for i in range(num_rules):
            template_idx = i % len(rule_templates)
            category_idx = i % len(categories)
            authority_idx = i % len(authorities)

            content = rule_templates[template_idx]
            category = categories[category_idx]
            authority = authorities[authority_idx]

            # Create a simple embedding (random for benchmark purposes)
            embedding = [0.1] * 384  # Simple vector for testing

            point = models.PointStruct(
                id=i,
                vector={"dense": embedding},
                payload={
                    "content": content,
                    "category": category.value,
                    "authority": authority.value,
                    "scope": "",  # Global rule
                    "tags": [f"tag_{i % 3}"],
                    "priority": 50,
                },
            )
            points.append(point)

        # Upload all points
        client.upsert(collection_name=collection_name, points=points)

    @staticmethod
    def create_sample_rules(num_rules: int) -> List[MemoryRule]:
        """
        Create sample MemoryRule objects for benchmarking.

        Args:
            num_rules: Number of rules to create

        Returns:
            List of MemoryRule objects
        """
        rules = []
        content_templates = [
            "Use type hints in all function signatures",
            "Follow PEP 8 style guidelines consistently",
            "Write comprehensive unit tests with edge cases",
            "Document all public APIs with clear examples",
            "Optimize database queries for performance",
            "Implement proper error handling with specific exceptions",
            "Use async patterns for I/O-bound operations",
            "Apply SOLID principles in class design",
            "Maintain code coverage above 80%",
            "Use dependency injection for testability",
        ]

        for i in range(num_rules):
            content = content_templates[i % len(content_templates)]
            # Add some variation to content
            content = f"{content} (Rule {i + 1})"

            rule = MemoryRule(
                id=f"rule_{i}",
                category=MemoryCategory.PREFERENCE if i % 2 == 0 else MemoryCategory.BEHAVIOR,
                name=f"Rule {i + 1}",
                rule=content,
                authority=AuthorityLevel.DEFAULT if i % 3 == 0 else AuthorityLevel.ABSOLUTE,
                scope=[],
            )
            rules.append(rule)

        return rules


# Helper function for percentile calculation
def calculate_percentiles(benchmark_stats) -> Dict[str, float]:
    """
    Calculate percentile metrics from benchmark stats.

    Args:
        benchmark_stats: pytest-benchmark stats object

    Returns:
        Dict with p50, p95, p99 metrics in milliseconds
    """
    if hasattr(benchmark_stats, "stats") and hasattr(benchmark_stats.stats, "data"):
        # Get raw timing data
        data = benchmark_stats.stats.data
        if data:
            # Convert to milliseconds
            data_ms = [t * 1000 for t in data]
            return {
                "p50_ms": statistics.quantiles(data_ms, n=100)[49],
                "p95_ms": statistics.quantiles(data_ms, n=100)[94],
                "p99_ms": statistics.quantiles(data_ms, n=100)[98],
                "min_ms": min(data_ms),
                "max_ms": max(data_ms),
                "mean_ms": statistics.mean(data_ms),
                "median_ms": statistics.median(data_ms),
            }
    return {}


# Fixtures for test infrastructure
@pytest.fixture(scope="module")
def qdrant_client():
    """Create Qdrant client for benchmarking."""
    with suppress_qdrant_ssl_warnings():
        client = QdrantClient(url="http://localhost:6333")
    yield client
    client.close()


@pytest.fixture(scope="module")
def memory_collection_name():
    """Provide collection name for memory rules."""
    return "benchmark_memory_rules"


@pytest.fixture(scope="module")
def memory_collection(qdrant_client, memory_collection_name):
    """Create and populate memory collection for benchmarks."""
    # Run async setup
    asyncio.run(
        ContextInjectionBenchmarkFixtures.create_memory_collection(
            qdrant_client,
            memory_collection_name,
            num_rules=50,  # Create 50 rules for various scenarios
        )
    )
    yield memory_collection_name
    # Cleanup
    try:
        qdrant_client.delete_collection(memory_collection_name)
    except Exception:
        pass


@pytest.fixture(scope="module")
def memory_manager(qdrant_client, memory_collection):
    """Create MemoryManager instance for benchmarks."""
    manager = MemoryManager(
        qdrant_client=qdrant_client,
        collection_name=memory_collection,
    )
    yield manager


# Sample rule fixtures for different context sizes
@pytest.fixture(scope="module")
def small_rules():
    """Small context: 2 rules."""
    return ContextInjectionBenchmarkFixtures.create_sample_rules(2)


@pytest.fixture(scope="module")
def medium_rules():
    """Medium context: 10 rules."""
    return ContextInjectionBenchmarkFixtures.create_sample_rules(10)


@pytest.fixture(scope="module")
def large_rules():
    """Large context: 25 rules."""
    return ContextInjectionBenchmarkFixtures.create_sample_rules(25)


# ============================================================================
# Rule Retrieval Benchmarks
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_rule_retrieval_small_limit_5_no_cache(
    benchmark, memory_manager, memory_collection
):
    """Benchmark rule retrieval with limit=5, no caching."""
    retrieval = RuleRetrieval(
        memory_manager=memory_manager,
        enable_cache=False,
        enable_indexing=False,
    )

    filter = RuleFilter(limit=5, offset=0)

    def run_retrieval():
        return asyncio.run(retrieval.get_rules(filter))

    result = benchmark(run_retrieval)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nRule retrieval (limit=5, no cache) percentiles: {percentiles}")

    assert len(result.rules) <= 5


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_rule_retrieval_medium_limit_10_no_cache(
    benchmark, memory_manager, memory_collection
):
    """Benchmark rule retrieval with limit=10, no caching."""
    retrieval = RuleRetrieval(
        memory_manager=memory_manager,
        enable_cache=False,
        enable_indexing=False,
    )

    filter = RuleFilter(limit=10, offset=0)

    def run_retrieval():
        return asyncio.run(retrieval.get_rules(filter))

    result = benchmark(run_retrieval)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nRule retrieval (limit=10, no cache) percentiles: {percentiles}")

    assert len(result.rules) <= 10


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_rule_retrieval_large_limit_25_no_cache(
    benchmark, memory_manager, memory_collection
):
    """Benchmark rule retrieval with limit=25, no caching."""
    retrieval = RuleRetrieval(
        memory_manager=memory_manager,
        enable_cache=False,
        enable_indexing=False,
    )

    filter = RuleFilter(limit=25, offset=0)

    def run_retrieval():
        return asyncio.run(retrieval.get_rules(filter))

    result = benchmark(run_retrieval)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nRule retrieval (limit=25, no cache) percentiles: {percentiles}")

    assert len(result.rules) <= 25


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_rule_retrieval_with_cache_cold_start(
    benchmark, memory_manager, memory_collection
):
    """Benchmark rule retrieval with cache (cold start)."""

    def run_cold_retrieval():
        # Create new retrieval instance each time for cold start
        retrieval = RuleRetrieval(
            memory_manager=memory_manager,
            enable_cache=True,
            cache_maxsize=128,
            cache_ttl=300,
            enable_indexing=False,
        )
        filter = RuleFilter(limit=10, offset=0)
        return asyncio.run(retrieval.get_rules(filter))

    result = benchmark(run_cold_retrieval)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nRule retrieval (cache cold start) percentiles: {percentiles}")

    assert len(result.rules) <= 10


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_rule_retrieval_with_cache_warm(benchmark, memory_manager, memory_collection):
    """Benchmark rule retrieval with cache (warm/hit)."""
    retrieval = RuleRetrieval(
        memory_manager=memory_manager,
        enable_cache=True,
        cache_maxsize=128,
        cache_ttl=300,
        enable_indexing=False,
    )
    filter = RuleFilter(limit=10, offset=0)

    # Warm up cache
    for _ in range(3):
        asyncio.run(retrieval.get_rules(filter))

    def run_warm_retrieval():
        return asyncio.run(retrieval.get_rules(filter))

    result = benchmark(run_warm_retrieval)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nRule retrieval (cache warm) percentiles: {percentiles}")

    assert len(result.rules) <= 10


# ============================================================================
# Token Counting Benchmarks
# ============================================================================


@pytest.mark.benchmark
def test_token_counting_small_context_tiktoken(benchmark, small_rules):
    """Benchmark token counting for small context (2 rules) with tiktoken."""

    def count_tokens():
        total = 0
        for rule in small_rules:
            total += TokenCounter.count_tokens_with_model(
                text=rule.rule,
                model_name="gpt-3.5-turbo",
                tokenizer_type=TokenizerType.TIKTOKEN,
            )
        return total

    result = benchmark(count_tokens)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nToken counting (small, tiktoken) percentiles: {percentiles}")

    assert result > 0


@pytest.mark.benchmark
def test_token_counting_medium_context_tiktoken(benchmark, medium_rules):
    """Benchmark token counting for medium context (10 rules) with tiktoken."""

    def count_tokens():
        total = 0
        for rule in medium_rules:
            total += TokenCounter.count_tokens_with_model(
                text=rule.rule,
                model_name="gpt-3.5-turbo",
                tokenizer_type=TokenizerType.TIKTOKEN,
            )
        return total

    result = benchmark(count_tokens)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nToken counting (medium, tiktoken) percentiles: {percentiles}")

    assert result > 0


@pytest.mark.benchmark
def test_token_counting_large_context_tiktoken(benchmark, large_rules):
    """Benchmark token counting for large context (25 rules) with tiktoken."""

    def count_tokens():
        total = 0
        for rule in large_rules:
            total += TokenCounter.count_tokens_with_model(
                text=rule.rule,
                model_name="gpt-3.5-turbo",
                tokenizer_type=TokenizerType.TIKTOKEN,
            )
        return total

    result = benchmark(count_tokens)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nToken counting (large, tiktoken) percentiles: {percentiles}")

    assert result > 0


@pytest.mark.benchmark
def test_token_counting_estimation_fallback(benchmark, medium_rules):
    """Benchmark token counting with estimation fallback (no tiktoken)."""

    def count_tokens():
        total = 0
        for rule in medium_rules:
            total += TokenCounter.count_tokens_with_model(
                text=rule.rule,
                model_name="gpt-3.5-turbo",
                tokenizer_type=TokenizerType.ESTIMATION,
            )
        return total

    result = benchmark(count_tokens)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nToken counting (estimation) percentiles: {percentiles}")

    assert result > 0


# ============================================================================
# Token Budget Allocation Benchmarks
# ============================================================================


@pytest.mark.benchmark
def test_budget_allocation_small_context(benchmark, small_rules):
    """Benchmark token budget allocation for small context (2 rules)."""
    manager = TokenBudgetManager(
        overhead_percentage=0.1,
        enable_cache=False,
    )

    def allocate_budget():
        return manager.allocate_budget(
            rules=small_rules,
            total_budget=10000,
            tool_name="claude",
        )

    result = benchmark(allocate_budget)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nBudget allocation (small) percentiles: {percentiles}")

    assert len(result.rules_included) > 0


@pytest.mark.benchmark
def test_budget_allocation_medium_context(benchmark, medium_rules):
    """Benchmark token budget allocation for medium context (10 rules)."""
    manager = TokenBudgetManager(
        overhead_percentage=0.1,
        enable_cache=False,
    )

    def allocate_budget():
        return manager.allocate_budget(
            rules=medium_rules,
            total_budget=10000,
            tool_name="claude",
        )

    result = benchmark(allocate_budget)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nBudget allocation (medium) percentiles: {percentiles}")

    assert len(result.rules_included) > 0


@pytest.mark.benchmark
def test_budget_allocation_large_context(benchmark, large_rules):
    """Benchmark token budget allocation for large context (25 rules)."""
    manager = TokenBudgetManager(
        overhead_percentage=0.1,
        enable_cache=False,
    )

    def allocate_budget():
        return manager.allocate_budget(
            rules=large_rules,
            total_budget=10000,
            tool_name="claude",
        )

    result = benchmark(allocate_budget)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nBudget allocation (large) percentiles: {percentiles}")

    assert len(result.rules_included) > 0


@pytest.mark.benchmark
def test_budget_allocation_tight_budget(benchmark, large_rules):
    """Benchmark token budget allocation with tight budget constraint."""
    manager = TokenBudgetManager(
        overhead_percentage=0.1,
        enable_cache=False,
    )

    def allocate_budget():
        return manager.allocate_budget(
            rules=large_rules,
            total_budget=2000,  # Tight budget
            tool_name="claude",
        )

    result = benchmark(allocate_budget)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nBudget allocation (tight budget) percentiles: {percentiles}")

    # Some rules may be skipped due to tight budget
    assert len(result.rules_included) >= 0


# ============================================================================
# Context Formatting Benchmarks
# ============================================================================


@pytest.mark.benchmark
def test_format_context_small_claude_code(benchmark, small_rules):
    """Benchmark context formatting for small context with Claude Code format."""
    manager = FormatManager()

    def format_context():
        return manager.format_for_tool(
            tool_name="claude",
            rules=small_rules,
            token_budget=10000,
        )

    result = benchmark(format_context)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nFormat context (small, Claude Code) percentiles: {percentiles}")

    assert result.content is not None


@pytest.mark.benchmark
def test_format_context_medium_claude_code(benchmark, medium_rules):
    """Benchmark context formatting for medium context with Claude Code format."""
    manager = FormatManager()

    def format_context():
        return manager.format_for_tool(
            tool_name="claude",
            rules=medium_rules,
            token_budget=10000,
        )

    result = benchmark(format_context)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nFormat context (medium, Claude Code) percentiles: {percentiles}")

    assert result.content is not None


@pytest.mark.benchmark
def test_format_context_large_claude_code(benchmark, large_rules):
    """Benchmark context formatting for large context with Claude Code format."""
    manager = FormatManager()

    def format_context():
        return manager.format_for_tool(
            tool_name="claude",
            rules=large_rules,
            token_budget=10000,
        )

    result = benchmark(format_context)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nFormat context (large, Claude Code) percentiles: {percentiles}")

    assert result.content is not None


@pytest.mark.benchmark
def test_format_context_github_codex(benchmark, medium_rules):
    """Benchmark context formatting with GitHub Codex format."""
    manager = FormatManager()

    def format_context():
        return manager.format_for_tool(
            tool_name="codex",
            rules=medium_rules,
            token_budget=10000,
        )

    result = benchmark(format_context)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nFormat context (GitHub Codex) percentiles: {percentiles}")

    assert result.content is not None


@pytest.mark.benchmark
def test_format_context_google_gemini(benchmark, medium_rules):
    """Benchmark context formatting with Google Gemini format."""
    manager = FormatManager()

    def format_context():
        return manager.format_for_tool(
            tool_name="gemini",
            rules=medium_rules,
            token_budget=10000,
        )

    result = benchmark(format_context)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nFormat context (Google Gemini) percentiles: {percentiles}")

    assert result.content is not None


# ============================================================================
# End-to-End Context Injection Pipeline Benchmarks
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_end_to_end_injection_small_context(benchmark, memory_manager):
    """Benchmark end-to-end context injection for small context (2 rules)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "claude_md_output.txt"
        project_root = Path(tmpdir)

        trigger = ClaudeMdFileTrigger(
            output_path=output_path,
            token_budget=10000,
            filter=RuleFilter(limit=2),
        )

        # Create mock session
        mock_session = MagicMock()
        context = TriggerContext(
            session=mock_session,
            project_root=project_root,
            memory_manager=memory_manager,
        )

        def run_injection():
            return asyncio.run(trigger.execute(context))

        result = benchmark(run_injection)
        percentiles = calculate_percentiles(benchmark)
        if percentiles:
            print(f"\nEnd-to-end injection (small) percentiles: {percentiles}")

        assert result.success


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_end_to_end_injection_medium_context(benchmark, memory_manager):
    """Benchmark end-to-end context injection for medium context (10 rules)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "claude_md_output.txt"
        project_root = Path(tmpdir)

        trigger = ClaudeMdFileTrigger(
            output_path=output_path,
            token_budget=10000,
            filter=RuleFilter(limit=10),
        )

        mock_session = MagicMock()
        context = TriggerContext(
            session=mock_session,
            project_root=project_root,
            memory_manager=memory_manager,
        )

        def run_injection():
            return asyncio.run(trigger.execute(context))

        result = benchmark(run_injection)
        percentiles = calculate_percentiles(benchmark)
        if percentiles:
            print(f"\nEnd-to-end injection (medium) percentiles: {percentiles}")

        assert result.success


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_end_to_end_injection_large_context(benchmark, memory_manager):
    """Benchmark end-to-end context injection for large context (25 rules)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "claude_md_output.txt"
        project_root = Path(tmpdir)

        trigger = ClaudeMdFileTrigger(
            output_path=output_path,
            token_budget=10000,
            filter=RuleFilter(limit=25),
        )

        mock_session = MagicMock()
        context = TriggerContext(
            session=mock_session,
            project_root=project_root,
            memory_manager=memory_manager,
        )

        def run_injection():
            return asyncio.run(trigger.execute(context))

        result = benchmark(run_injection)
        percentiles = calculate_percentiles(benchmark)
        if percentiles:
            print(f"\nEnd-to-end injection (large) percentiles: {percentiles}")

        assert result.success


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_end_to_end_injection_tight_budget(benchmark, memory_manager):
    """Benchmark end-to-end context injection with tight token budget."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "claude_md_output.txt"
        project_root = Path(tmpdir)

        trigger = ClaudeMdFileTrigger(
            output_path=output_path,
            token_budget=2000,  # Tight budget
            filter=RuleFilter(limit=20),
        )

        mock_session = MagicMock()
        context = TriggerContext(
            session=mock_session,
            project_root=project_root,
            memory_manager=memory_manager,
        )

        def run_injection():
            return asyncio.run(trigger.execute(context))

        result = benchmark(run_injection)
        percentiles = calculate_percentiles(benchmark)
        if percentiles:
            print(f"\nEnd-to-end injection (tight budget) percentiles: {percentiles}")

        assert result.success
