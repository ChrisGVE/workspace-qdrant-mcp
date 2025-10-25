"""
Query interface for analyzing aggregated test results.

Provides high-level queries for test result analysis, trend detection,
and reporting.
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from .models import TestStatus, TestType
from .storage import TestResultStorage


class TestResultQuery:
    """High-level query interface for test results."""

    def __init__(self, storage: TestResultStorage | None = None):
        """
        Initialize query interface.

        Args:
            storage: Storage backend. If None, creates default storage.
        """
        self.storage = storage or TestResultStorage()

    def get_latest_run(self) -> dict[str, Any] | None:
        """
        Get the most recent test run.

        Returns:
            Test run summary dict or None
        """
        runs = self.storage.list_test_runs(limit=1, offset=0)
        return runs[0] if runs else None

    def get_run_summary(self, run_id: str) -> dict[str, Any] | None:
        """
        Get summary of a specific test run.

        Args:
            run_id: Test run ID

        Returns:
            Summary dict with counts, success rate, duration, etc.
        """
        test_run = self.storage.get_test_run(run_id)
        if not test_run:
            return None

        # Calculate total duration
        total_duration_ms = sum(
            result.duration_ms
            for suite in test_run.suites
            for case in suite.test_cases
            for result in case.results
        )

        # Count by test type
        counts_by_type = {}
        for suite in test_run.suites:
            test_type = suite.test_type.value
            counts_by_type.setdefault(test_type, 0)
            counts_by_type[test_type] += suite.total_tests

        return {
            "run_id": test_run.run_id,
            "timestamp": test_run.timestamp.isoformat(),
            "source": test_run.source.value,
            "total_tests": test_run.total_tests,
            "passed": test_run.passed_tests,
            "failed": test_run.failed_tests,
            "skipped": test_run.skipped_tests,
            "errors": test_run.error_tests,
            "success_rate": test_run.success_rate,
            "total_duration_ms": total_duration_ms,
            "counts_by_type": counts_by_type,
            "suites": len(test_run.suites),
            "metadata": test_run.metadata,
        }

    def get_failed_tests(self, run_id: str) -> list[dict[str, Any]]:
        """
        Get all failed tests from a run.

        Args:
            run_id: Test run ID

        Returns:
            List of failed test details
        """
        test_run = self.storage.get_test_run(run_id)
        if not test_run:
            return []

        failed_tests = []
        for suite in test_run.suites:
            for case in suite.test_cases:
                for result in case.results:
                    if result.status == TestStatus.FAILED:
                        failed_tests.append(
                            {
                                "name": result.name,
                                "suite": suite.name,
                                "test_type": suite.test_type.value,
                                "duration_ms": result.duration_ms,
                                "error_message": result.error_message,
                                "file_path": case.file_path,
                                "line_number": case.line_number,
                            }
                        )

        return failed_tests

    def get_slow_tests(
        self, run_id: str, threshold_ms: float = 1000, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get slowest tests from a run.

        Args:
            run_id: Test run ID
            threshold_ms: Only include tests slower than this
            limit: Maximum number of results

        Returns:
            List of slow test details, sorted by duration descending
        """
        test_run = self.storage.get_test_run(run_id)
        if not test_run:
            return []

        all_tests = []
        for suite in test_run.suites:
            for case in suite.test_cases:
                for result in case.results:
                    if result.duration_ms >= threshold_ms:
                        all_tests.append(
                            {
                                "name": result.name,
                                "suite": suite.name,
                                "test_type": suite.test_type.value,
                                "duration_ms": result.duration_ms,
                                "file_path": case.file_path,
                            }
                        )

        # Sort by duration descending
        all_tests.sort(key=lambda x: x["duration_ms"], reverse=True)
        return all_tests[:limit]

    def get_benchmark_results(self, run_id: str) -> list[dict[str, Any]]:
        """
        Get all benchmark/performance test results from a run.

        Args:
            run_id: Test run ID

        Returns:
            List of benchmark results with performance metrics
        """
        test_run = self.storage.get_test_run(run_id)
        if not test_run:
            return []

        benchmarks = []
        for suite in test_run.suites:
            if suite.test_type in [TestType.BENCHMARK, TestType.PERFORMANCE]:
                for case in suite.test_cases:
                    for result in case.results:
                        if result.performance:
                            benchmarks.append(
                                {
                                    "name": result.name,
                                    "suite": suite.name,
                                    "test_type": suite.test_type.value,
                                    "min_ms": result.performance.min_ms,
                                    "max_ms": result.performance.max_ms,
                                    "avg_ms": result.performance.avg_ms,
                                    "median_ms": result.performance.median_ms,
                                    "p95_ms": result.performance.p95_ms,
                                    "p99_ms": result.performance.p99_ms,
                                    "ops_per_sec": result.performance.operations_per_second,
                                    "memory_mb": result.performance.memory_mb,
                                    "cpu_percent": result.performance.cpu_percent,
                                }
                            )

        return benchmarks

    def compare_runs(self, run_id_1: str, run_id_2: str) -> dict[str, Any]:
        """
        Compare two test runs.

        Args:
            run_id_1: First test run ID
            run_id_2: Second test run ID

        Returns:
            Comparison summary with differences
        """
        run1 = self.storage.get_test_run(run_id_1)
        run2 = self.storage.get_test_run(run_id_2)

        if not run1 or not run2:
            return {"error": "One or both runs not found"}

        return {
            "run1": {
                "run_id": run1.run_id,
                "timestamp": run1.timestamp.isoformat(),
                "total_tests": run1.total_tests,
                "passed": run1.passed_tests,
                "failed": run1.failed_tests,
                "success_rate": run1.success_rate,
            },
            "run2": {
                "run_id": run2.run_id,
                "timestamp": run2.timestamp.isoformat(),
                "total_tests": run2.total_tests,
                "passed": run2.passed_tests,
                "failed": run2.failed_tests,
                "success_rate": run2.success_rate,
            },
            "differences": {
                "total_tests_delta": run2.total_tests - run1.total_tests,
                "passed_delta": run2.passed_tests - run1.passed_tests,
                "failed_delta": run2.failed_tests - run1.failed_tests,
                "success_rate_delta": run2.success_rate - run1.success_rate,
            },
        }

    def get_test_history(
        self, test_name: str, days: int = 30, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Get execution history for a specific test.

        Args:
            test_name: Test name to search for
            days: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of test executions with status and metrics
        """
        start_date = datetime.now() - timedelta(days=days)
        runs = self.storage.list_test_runs(
            limit=limit, start_date=start_date, offset=0
        )

        history = []
        for run_summary in runs:
            run = self.storage.get_test_run(run_summary["run_id"])
            if not run:
                continue

            # Search for test in this run
            for suite in run.suites:
                for case in suite.test_cases:
                    if case.name == test_name or test_name in case.name:
                        for result in case.results:
                            history.append(
                                {
                                    "run_id": run.run_id,
                                    "timestamp": run.timestamp.isoformat(),
                                    "status": result.status.value,
                                    "duration_ms": result.duration_ms,
                                    "suite": suite.name,
                                    "test_type": suite.test_type.value,
                                }
                            )

        return history

    def get_flaky_tests(
        self, days: int = 30, min_runs: int = 5, flaky_threshold: float = 0.2
    ) -> list[dict[str, Any]]:
        """
        Identify potentially flaky tests (tests with inconsistent results).

        Args:
            days: Number of days to analyze
            min_runs: Minimum number of runs to consider
            flaky_threshold: Failure rate threshold (0.2 = 20% failure rate)

        Returns:
            List of potentially flaky tests with statistics
        """
        start_date = datetime.now() - timedelta(days=days)
        runs = self.storage.list_test_runs(limit=1000, start_date=start_date, offset=0)

        # Collect test statistics
        test_stats: dict[str, dict[str, Any]] = {}

        for run_summary in runs:
            run = self.storage.get_test_run(run_summary["run_id"])
            if not run:
                continue

            for suite in run.suites:
                for case in suite.test_cases:
                    test_key = f"{suite.name}::{case.name}"

                    if test_key not in test_stats:
                        test_stats[test_key] = {
                            "name": case.name,
                            "suite": suite.name,
                            "total_runs": 0,
                            "passed": 0,
                            "failed": 0,
                            "file_path": case.file_path,
                        }

                    for result in case.results:
                        test_stats[test_key]["total_runs"] += 1
                        if result.status == TestStatus.PASSED:
                            test_stats[test_key]["passed"] += 1
                        elif result.status == TestStatus.FAILED:
                            test_stats[test_key]["failed"] += 1

        # Identify flaky tests
        flaky_tests = []
        for test_key, stats in test_stats.items():
            total = stats["total_runs"]
            if total < min_runs:
                continue

            failed = stats["failed"]
            failure_rate = failed / total

            # Flaky: some failures but not complete failure
            if 0 < failure_rate <= flaky_threshold:
                flaky_tests.append(
                    {
                        "name": stats["name"],
                        "suite": stats["suite"],
                        "file_path": stats["file_path"],
                        "total_runs": total,
                        "passed": stats["passed"],
                        "failed": failed,
                        "failure_rate": failure_rate,
                    }
                )

        # Sort by failure rate descending
        flaky_tests.sort(key=lambda x: x["failure_rate"], reverse=True)
        return flaky_tests

    def get_statistics_summary(self) -> dict[str, Any]:
        """
        Get overall statistics summary.

        Returns:
            Statistics dictionary
        """
        return self.storage.get_statistics()

    def get_trend_data(
        self, days: int = 30, metric: str = "success_rate"
    ) -> list[dict[str, Any]]:
        """
        Get trend data for a specific metric over time.

        Args:
            days: Number of days to analyze
            metric: Metric to track ('success_rate', 'total_tests', 'failed_tests')

        Returns:
            List of data points with timestamp and metric value
        """
        start_date = datetime.now() - timedelta(days=days)
        runs = self.storage.list_test_runs(limit=1000, start_date=start_date, offset=0)

        trend_data = []
        for run in runs:
            data_point = {
                "timestamp": run["timestamp"],
                "run_id": run["run_id"],
            }

            if metric == "success_rate":
                data_point["value"] = run["success_rate"]
            elif metric == "total_tests":
                data_point["value"] = run["total_tests"]
            elif metric == "failed_tests":
                data_point["value"] = run["failed_tests"]
            elif metric == "passed_tests":
                data_point["value"] = run["passed_tests"]
            else:
                data_point["value"] = 0

            trend_data.append(data_point)

        return trend_data
