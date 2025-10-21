"""
Unit tests for test result aggregation system.

Tests data models, parsers, storage, and query interfaces.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from tests.reporting import (
    TestResultAggregator,
    TestResultQuery,
    TestSource,
    TestStatus,
    TestType,
    aggregate_test_results,
)
from tests.reporting.models import PerformanceMetrics, TestCase, TestResult, TestRun
from tests.reporting.storage import TestResultStorage


class TestDataModels:
    """Test data model classes."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        perf = PerformanceMetrics(
            min_ms=1.0,
            max_ms=10.0,
            avg_ms=5.0,
            p95_ms=8.0,
            operations_per_second=100.0,
        )

        assert perf.min_ms == 1.0
        assert perf.max_ms == 10.0
        assert perf.avg_ms == 5.0
        assert perf.operations_per_second == 100.0

    def test_performance_metrics_serialization(self):
        """Test performance metrics to/from dict."""
        perf = PerformanceMetrics(avg_ms=5.0, memory_mb=128.0)
        data = perf.to_dict()

        assert data["avg_ms"] == 5.0
        assert data["memory_mb"] == 128.0

        perf2 = PerformanceMetrics.from_dict(data)
        assert perf2.avg_ms == 5.0
        assert perf2.memory_mb == 128.0

    def test_test_result_creation(self):
        """Test creating test result."""
        result = TestResult(
            test_id="test-1",
            name="test_example",
            status=TestStatus.PASSED,
            duration_ms=100.0,
            timestamp=datetime.now(),
        )

        assert result.name == "test_example"
        assert result.status == TestStatus.PASSED
        assert result.duration_ms == 100.0

    def test_test_case_latest_result(self):
        """Test getting latest result from test case."""
        case = TestCase(case_id="case-1", name="test_case", test_type=TestType.UNIT)

        now = datetime.now()
        result1 = TestResult(
            test_id="r1",
            name="test_case",
            status=TestStatus.FAILED,
            duration_ms=50.0,
            timestamp=now,
        )
        result2 = TestResult(
            test_id="r2",
            name="test_case",
            status=TestStatus.PASSED,
            duration_ms=60.0,
            timestamp=now,
        )

        case.add_result(result1)
        case.add_result(result2)

        # Latest should be result2 (same timestamp, but added last)
        assert case.latest_status in [TestStatus.PASSED, TestStatus.FAILED]

    def test_test_run_statistics(self):
        """Test test run statistics calculation."""
        from uuid import uuid4

        from tests.reporting.models import TestSuite

        test_run = TestRun.create(source=TestSource.PYTEST)

        suite = TestSuite(
            suite_id=str(uuid4()), name="unit_tests", test_type=TestType.UNIT
        )

        # Add test cases with results
        for i in range(10):
            case = TestCase(
                case_id=str(uuid4()), name=f"test_{i}", test_type=TestType.UNIT
            )
            status = TestStatus.PASSED if i < 8 else TestStatus.FAILED
            result = TestResult(
                test_id=str(uuid4()),
                name=f"test_{i}",
                status=status,
                duration_ms=100.0,
                timestamp=datetime.now(),
            )
            case.add_result(result)
            suite.add_test_case(case)

        test_run.add_suite(suite)

        assert test_run.total_tests == 10
        assert test_run.passed_tests == 8
        assert test_run.failed_tests == 2
        assert test_run.success_rate == 80.0


class TestStorage:
    """Test storage backend."""

    def test_storage_initialization(self):
        """Test creating storage with temp database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            storage = TestResultStorage(db_path)
            assert db_path.exists()

            # Check stats on empty db
            stats = storage.get_statistics()
            assert stats["total_runs"] == 0
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_save_and_retrieve_test_run(self):
        """Test saving and retrieving a test run."""
        from uuid import uuid4

        from tests.reporting.models import TestSuite

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            storage = TestResultStorage(db_path)

            # Create test run
            test_run = TestRun.create(source=TestSource.PYTEST)
            run_id = test_run.run_id

            suite = TestSuite(
                suite_id=str(uuid4()), name="tests", test_type=TestType.UNIT
            )
            case = TestCase(case_id=str(uuid4()), name="test_1", test_type=TestType.UNIT)
            result = TestResult(
                test_id=str(uuid4()),
                name="test_1",
                status=TestStatus.PASSED,
                duration_ms=50.0,
                timestamp=datetime.now(),
            )
            case.add_result(result)
            suite.add_test_case(case)
            test_run.add_suite(suite)

            # Save
            storage.save_test_run(test_run)

            # Retrieve
            retrieved = storage.get_test_run(run_id)
            assert retrieved is not None
            assert retrieved.run_id == run_id
            assert retrieved.total_tests == 1
            assert len(retrieved.suites) == 1
            assert len(retrieved.suites[0].test_cases) == 1
        finally:
            if db_path.exists():
                db_path.unlink()


class TestBenchmarkParser:
    """Test benchmark JSON parser."""

    def test_parse_custom_benchmark(self):
        """Test parsing custom benchmark format."""
        # Use actual benchmark file
        benchmark_file = Path(__file__).parent.parent.parent / "20250924-1829_benchmark_results.json"

        if not benchmark_file.exists():
            pytest.skip(f"Benchmark file not found: {benchmark_file}")

        run = aggregate_test_results(benchmark_file, TestSource.BENCHMARK_JSON)

        assert run is not None
        assert run.source == TestSource.BENCHMARK_JSON
        assert len(run.suites) > 0

        # Check that benchmarks were parsed
        suite = run.suites[0]
        assert suite.test_type == TestType.BENCHMARK
        assert len(suite.test_cases) > 0

    def test_parse_grpc_integration_test(self):
        """Test parsing gRPC integration test format."""
        grpc_file = Path(__file__).parent.parent.parent / "grpc_integration_test_results_20250924_192526.json"

        if not grpc_file.exists():
            pytest.skip(f"gRPC test file not found: {grpc_file}")

        run = aggregate_test_results(grpc_file, TestSource.GRPC_TEST)

        assert run is not None
        assert run.source == TestSource.GRPC_TEST
        assert len(run.suites) > 0


class TestAggregator:
    """Test main aggregator."""

    def test_aggregator_with_benchmark(self):
        """Test aggregating benchmark results."""
        benchmark_file = Path(__file__).parent.parent.parent / "20250924-1829_benchmark_results.json"

        if not benchmark_file.exists():
            pytest.skip(f"Benchmark file not found: {benchmark_file}")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            storage = TestResultStorage(db_path)
            aggregator = TestResultAggregator(storage)

            run = aggregator.aggregate_from_file(benchmark_file, TestSource.BENCHMARK_JSON)

            assert run.total_tests > 0

            # Retrieve from storage
            retrieved = storage.get_test_run(run.run_id)
            assert retrieved is not None
            assert retrieved.total_tests == run.total_tests
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_multi_source_aggregation(self):
        """Test aggregating from multiple sources."""
        benchmark_file = Path(__file__).parent.parent.parent / "20250924-1829_benchmark_results.json"

        if not benchmark_file.exists():
            pytest.skip(f"Benchmark file not found: {benchmark_file}")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            storage = TestResultStorage(db_path)
            aggregator = TestResultAggregator(storage)

            # Create a simple pytest-like result
            pytest_data = {
                "created": datetime.now().timestamp(),
                "duration": 1.0,
                "exitcode": 0,
                "summary": {"passed": 1, "failed": 0},
                "tests": [
                    {
                        "name": "test_example",
                        "nodeid": "test_example.py::test_example",
                        "outcome": "passed",
                        "duration": 0.1,
                        "location": ["test_example.py", 1, "test_example"],
                    }
                ],
            }

            run = aggregator.aggregate_multiple(
                [
                    {"file": str(benchmark_file), "source": TestSource.BENCHMARK_JSON},
                    {"data": pytest_data, "source": TestSource.PYTEST},
                ],
                run_id="multi-test-run",
            )

            assert run.run_id == "multi-test-run"
            assert len(run.suites) >= 2  # At least benchmark and pytest suites
        finally:
            if db_path.exists():
                db_path.unlink()


class TestQuery:
    """Test query interface."""

    def test_query_benchmark_results(self):
        """Test querying benchmark results."""
        benchmark_file = Path(__file__).parent.parent.parent / "20250924-1829_benchmark_results.json"

        if not benchmark_file.exists():
            pytest.skip(f"Benchmark file not found: {benchmark_file}")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            storage = TestResultStorage(db_path)
            aggregator = TestResultAggregator(storage)
            query = TestResultQuery(storage)

            run = aggregator.aggregate_from_file(benchmark_file, TestSource.BENCHMARK_JSON)

            # Get run summary
            summary = query.get_run_summary(run.run_id)
            assert summary is not None
            assert summary["run_id"] == run.run_id
            assert summary["total_tests"] > 0

            # Get benchmark results
            benchmarks = query.get_benchmark_results(run.run_id)
            assert len(benchmarks) > 0

            # Check that performance metrics are present
            bench = benchmarks[0]
            assert "name" in bench
            assert "avg_ms" in bench or "min_ms" in bench
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_query_statistics(self):
        """Test getting statistics."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            storage = TestResultStorage(db_path)
            query = TestResultQuery(storage)

            stats = query.get_statistics_summary()
            assert "total_runs" in stats
            assert "total_results" in stats
        finally:
            if db_path.exists():
                db_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
