"""
Parser for custom benchmark JSON results.

Parses benchmark results in the format used by workspace-qdrant-mcp performance tests.
Example files: 20250924-1829_benchmark_results.json, grpc_integration_test_results_*.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

from ..models import (
    PerformanceMetrics,
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)
from .base import BaseParser


class BenchmarkJsonParser(BaseParser):
    """Parser for custom benchmark JSON format."""

    def parse(self, source: str | Path | dict) -> TestRun:
        """
        Parse benchmark JSON results into TestRun.

        Args:
            source: Path to JSON file or dict with benchmark data

        Returns:
            TestRun object with benchmark results
        """
        # Load data
        if isinstance(source, dict):
            data = source
        else:
            path = self._ensure_path(source)
            with open(path) as f:
                data = json.load(f)

        # Determine benchmark type and create appropriate test run
        if "test_summary" in data:
            # gRPC integration test format
            return self._parse_grpc_integration_test(data)
        elif "summary" in data:
            # Custom benchmark format (20250924-1829_benchmark_results.json)
            return self._parse_custom_benchmark(data)
        else:
            raise ValueError("Unknown benchmark JSON format")

    def _parse_custom_benchmark(self, data: dict[str, Any]) -> TestRun:
        """Parse custom benchmark format (e.g., 20250924-1829_benchmark_results.json)."""
        # Extract timestamp from summary
        timestamp_unix = data.get("summary", {}).get("timestamp")
        if timestamp_unix:
            timestamp = datetime.fromtimestamp(timestamp_unix)
        else:
            timestamp = datetime.now()

        # Create test run
        test_run = TestRun.create(
            source=TestSource.BENCHMARK_JSON,
            timestamp=timestamp,
            metadata={
                "format": "custom_benchmark",
                "total_benchmarks": data.get("summary", {}).get("total_benchmarks", 0),
                "completed_benchmarks": data.get("summary", {}).get("completed_benchmarks", 0),
                "failed_benchmarks": data.get("summary", {}).get("failed_benchmarks", 0),
                "overall_success_rate": data.get("summary", {}).get("overall_success_rate", 0),
            },
        )

        # Create benchmark suite
        suite = TestSuite(
            suite_id=str(uuid4()),
            name="benchmarks",
            test_type=TestType.BENCHMARK,
        )

        # Parse each benchmark category
        for bench_name, bench_data in data.items():
            if bench_name == "summary":
                continue

            test_case = self._parse_benchmark_category(bench_name, bench_data, timestamp)
            suite.add_test_case(test_case)

        test_run.add_suite(suite)
        return test_run

    def _parse_benchmark_category(
        self, name: str, data: dict[str, Any], timestamp: datetime
    ) -> TestCase:
        """Parse a single benchmark category."""
        case = TestCase(
            case_id=str(uuid4()),
            name=name,
            test_type=TestType.BENCHMARK,
            metadata={"category": name},
        )

        # Determine status
        status_str = data.get("status", "unknown")
        if status_str == "completed":
            status = TestStatus.PASSED
        elif status_str == "skipped":
            status = TestStatus.SKIPPED
        elif status_str == "failed":
            status = TestStatus.FAILED
        else:
            status = TestStatus.ERROR

        # Extract performance metrics
        perf_metrics = PerformanceMetrics()

        # Duration
        duration_ms = data.get("benchmark_duration_ms", 0)

        # Look for timing stats in various places
        for stats_key in [
            "parse_time_stats",
            "validation_time_stats",
            "query_time_stats",
            "load_time_stats",
            "reload_time_stats",
        ]:
            if stats_key in data:
                stats = data[stats_key]
                if "min_ms" in stats:
                    perf_metrics.min_ms = stats["min_ms"]
                elif "min" in stats:
                    perf_metrics.min_ms = stats["min"] * 1000
                if "max_ms" in stats:
                    perf_metrics.max_ms = stats["max_ms"]
                elif "max" in stats:
                    perf_metrics.max_ms = stats["max"] * 1000
                if "avg_ms" in stats:
                    perf_metrics.avg_ms = stats["avg_ms"]
                elif "avg" in stats:
                    perf_metrics.avg_ms = stats["avg"] * 1000
                if "p95_ms" in stats:
                    perf_metrics.p95_ms = stats["p95_ms"]
                if "p99_ms" in stats:
                    perf_metrics.p99_ms = stats["p99_ms"]

        # Memory metrics
        if "initial_memory_mb" in data:
            perf_metrics.memory_mb = data.get("peak_memory_mb", 0) - data.get(
                "initial_memory_mb", 0
            )

        # Custom metrics
        perf_metrics.custom_metrics = {
            k: v for k, v in data.items() if k not in ["status", "benchmark_duration_ms"]
        }

        # Create result
        result = TestResult(
            test_id=str(uuid4()),
            name=name,
            status=status,
            duration_ms=duration_ms,
            timestamp=timestamp,
            performance=perf_metrics,
            metadata={"benchmark_data": data},
        )

        case.add_result(result)
        return case

    def _parse_grpc_integration_test(self, data: dict[str, Any]) -> TestRun:
        """Parse gRPC integration test format."""
        # Extract timestamp
        timestamp_unix = data.get("test_summary", {}).get("start_time")
        if timestamp_unix:
            timestamp = datetime.fromtimestamp(timestamp_unix)
        else:
            timestamp = datetime.now()

        # Create test run
        test_run = TestRun.create(
            source=TestSource.GRPC_TEST,
            timestamp=timestamp,
            metadata={
                "format": "grpc_integration_test",
                "configuration": data.get("test_summary", {}).get("configuration", {}),
                "overall_metrics": data.get("overall_metrics", {}),
                "test_validation": data.get("test_validation", {}),
            },
        )

        # Parse service tests
        service_tests = data.get("service_tests", {})
        if service_tests:
            suite = TestSuite(
                suite_id=str(uuid4()),
                name="service_tests",
                test_type=TestType.INTEGRATION,
            )

            for service_name, service_data in service_tests.items():
                test_case = self._parse_grpc_service(
                    service_name, service_data, timestamp
                )
                suite.add_test_case(test_case)

            test_run.add_suite(suite)

        # Parse integration tests
        integration_tests = data.get("integration_tests", {})
        if integration_tests:
            suite = TestSuite(
                suite_id=str(uuid4()),
                name="integration_tests",
                test_type=TestType.INTEGRATION,
            )

            for test_name, test_data in integration_tests.items():
                test_case = self._parse_grpc_integration(
                    test_name, test_data, timestamp
                )
                suite.add_test_case(test_case)

            test_run.add_suite(suite)

        # Parse performance tests
        performance_tests = data.get("performance_tests", {})
        if performance_tests:
            suite = TestSuite(
                suite_id=str(uuid4()),
                name="performance_tests",
                test_type=TestType.PERFORMANCE,
            )

            for test_name, test_data in performance_tests.items():
                test_case = self._parse_grpc_performance(
                    test_name, test_data, timestamp
                )
                suite.add_test_case(test_case)

            test_run.add_suite(suite)

        return test_run

    def _parse_grpc_service(
        self, service_name: str, data: dict[str, Any], timestamp: datetime
    ) -> TestCase:
        """Parse a gRPC service test."""
        case = TestCase(
            case_id=str(uuid4()),
            name=service_name,
            test_type=TestType.INTEGRATION,
            metadata={
                "service": service_name,
                "methods_tested": data.get("methods_tested", []),
                "error_scenarios": data.get("error_scenarios", {}),
                "edge_cases": data.get("edge_cases", {}),
            },
        )

        # Parse performance metrics for each method
        perf_metrics_data = data.get("performance_metrics", {})
        for method_name, method_data in perf_metrics_data.items():
            # Create result for each method
            perf_metrics = PerformanceMetrics(
                avg_ms=method_data.get("execution_time_ms", 0),
                custom_metrics=method_data,
            )

            result = TestResult(
                test_id=str(uuid4()),
                name=f"{service_name}.{method_name}",
                status=TestStatus.PASSED,  # Assume passed if in results
                duration_ms=method_data.get("execution_time_ms", 0),
                timestamp=timestamp,
                performance=perf_metrics,
                metadata={"method": method_name, "method_data": method_data},
            )

            case.add_result(result)

        return case

    def _parse_grpc_integration(
        self, test_name: str, data: dict[str, Any], timestamp: datetime
    ) -> TestCase:
        """Parse a gRPC integration test."""
        case = TestCase(
            case_id=str(uuid4()),
            name=test_name,
            test_type=TestType.INTEGRATION,
            metadata={"test_type": data.get("test_type", ""), "test_data": data},
        )

        # Look for concurrency levels
        concurrency_levels = data.get("concurrency_levels", [])
        for level_data in concurrency_levels:
            perf_metrics = PerformanceMetrics(
                avg_ms=level_data.get("average_operation_time_ms", 0),
                min_ms=level_data.get("total_time_ms", 0),
                operations_per_second=level_data.get("operations_per_second", 0),
                custom_metrics=level_data,
            )

            # Determine status
            success_rate = level_data.get("success_rate", 0)
            status = TestStatus.PASSED if success_rate == 100.0 else TestStatus.FAILED

            result = TestResult(
                test_id=str(uuid4()),
                name=f"{test_name}_concurrency_{level_data.get('concurrency_level', 0)}",
                status=status,
                duration_ms=level_data.get("total_time_ms", 0),
                timestamp=timestamp,
                performance=perf_metrics,
                metadata={"concurrency_data": level_data},
            )

            case.add_result(result)

        return case

    def _parse_grpc_performance(
        self, test_name: str, data: dict[str, Any], timestamp: datetime
    ) -> TestCase:
        """Parse a gRPC performance test."""
        case = TestCase(
            case_id=str(uuid4()),
            name=test_name,
            test_type=TestType.PERFORMANCE,
            metadata={"test_type": data.get("test_type", ""), "test_data": data},
        )

        actual_perf = data.get("actual_performance", {})

        perf_metrics = PerformanceMetrics(
            avg_ms=actual_perf.get("average_response_time_ms", 0),
            median_ms=actual_perf.get("median_response_time_ms", 0),
            min_ms=actual_perf.get("min_response_time_ms", 0),
            max_ms=actual_perf.get("max_response_time_ms", 0),
            p95_ms=actual_perf.get("p95_response_time_ms", 0),
            operations_per_second=actual_perf.get("operations_per_second", 0),
            custom_metrics=actual_perf,
        )

        # Resource usage
        resource_usage = data.get("resource_usage", {})
        if resource_usage:
            perf_metrics.cpu_percent = resource_usage.get("peak_cpu_percent", 0)
            perf_metrics.memory_mb = resource_usage.get("peak_memory_mb", 0)

        # Determine status
        success_rate = actual_perf.get("success_rate", 0)
        status = TestStatus.PASSED if success_rate == 100.0 else TestStatus.FAILED

        result = TestResult(
            test_id=str(uuid4()),
            name=test_name,
            status=status,
            duration_ms=actual_perf.get("duration_seconds", 0) * 1000,
            timestamp=timestamp,
            performance=perf_metrics,
            metadata={"performance_data": data},
        )

        case.add_result(result)
        return case
