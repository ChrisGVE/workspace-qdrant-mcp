"""
Test Result Aggregation System for workspace-qdrant-mcp.

Provides unified aggregation and analysis of test results from multiple sources:
- pytest (JUnit XML, JSON)
- cargo test (text, JSON)
- Custom benchmarks
- gRPC integration tests

Usage:
    >>> from tests.reporting import TestResultAggregator, TestSource
    >>> aggregator = TestResultAggregator()
    >>> run = aggregator.aggregate_from_file('results.xml', TestSource.PYTEST)
    >>> print(f"Success rate: {run.success_rate:.1f}%")

Query API:
    >>> from tests.reporting import TestResultQuery
    >>> query = TestResultQuery()
    >>> failed = query.get_failed_tests(run.run_id)
    >>> flaky = query.get_flaky_tests(days=30)
"""

from .aggregator import TestResultAggregator, aggregate_test_results
from .models import (
    PerformanceMetrics,
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)
from .query import TestResultQuery
from .storage import TestResultStorage

__all__ = [
    # Main API
    "TestResultAggregator",
    "TestResultQuery",
    "TestResultStorage",
    "aggregate_test_results",
    # Models
    "TestRun",
    "TestSuite",
    "TestCase",
    "TestResult",
    "PerformanceMetrics",
    # Enums
    "TestSource",
    "TestStatus",
    "TestType",
]
