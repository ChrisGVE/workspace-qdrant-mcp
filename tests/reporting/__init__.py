"""
Test Result Aggregation and Reporting System for workspace-qdrant-mcp.

Provides unified aggregation, analysis, and reporting of test results from multiple sources:
- pytest (JUnit XML, JSON)
- cargo test (text, JSON)
- Custom benchmarks
- gRPC integration tests

Aggregation API:
    >>> from tests.reporting import TestResultAggregator, TestSource
    >>> aggregator = TestResultAggregator()
    >>> run = aggregator.aggregate_from_file('results.xml', TestSource.PYTEST)
    >>> print(f"Success rate: {run.success_rate:.1f}%")

Query API:
    >>> from tests.reporting import TestResultQuery
    >>> query = TestResultQuery()
    >>> failed = query.get_failed_tests(run.run_id)
    >>> flaky = query.get_flaky_tests(days=30)

Report Generation:
    >>> from tests.reporting import generate_test_report
    >>> generate_test_report(run.run_id, 'report.html', format='html')
    >>> generate_test_report(run.run_id, 'report.pdf', format='pdf')
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
from .report_generator import ReportGenerator, generate_test_report
from .storage import TestResultStorage

__all__ = [
    # Main API
    "TestResultAggregator",
    "TestResultQuery",
    "TestResultStorage",
    "aggregate_test_results",
    # Reporting
    "ReportGenerator",
    "generate_test_report",
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
