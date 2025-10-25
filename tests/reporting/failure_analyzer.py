"""
Failure pattern analysis and flakiness detection for test results.

Analyzes test failures across multiple runs to:
- Detect flaky tests (inconsistent pass/fail results)
- Categorize failure types
- Group similar error patterns
- Track failure trends over time
"""

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from .models import (
    FailureAnalysisReport,
    FailureCategory,
    FailurePattern,
    FlakinessMetrics,
    TestCase,
    TestResult,
    TestRun,
    TestStatus,
)
from .storage import TestResultStorage


class FailureAnalyzer:
    """
    Analyze test failures to detect patterns and flakiness.

    Uses error message analysis, failure categorization, and statistical
    methods to identify problematic tests requiring attention.
    """

    # Error patterns for categorization
    ASSERTION_PATTERNS = [
        r"AssertionError",
        r"assert\s+",
        r"expected.*got",
        r"should be.*but was",
        r"!= .*expected",
    ]

    TIMEOUT_PATTERNS = [
        r"TimeoutError",
        r"timeout",
        r"timed out",
        r"hang",
        r"deadline exceeded",
        r"took too long",
    ]

    SETUP_TEARDOWN_PATTERNS = [
        r"fixture",
        r"setUp",
        r"tearDown",
        r"cleanup",
        r"@pytest\.fixture",
        r"beforeEach",
        r"afterEach",
    ]

    EXTERNAL_DEPENDENCY_PATTERNS = [
        r"ConnectionError",
        r"connection refused",
        r"network",
        r"database",
        r"service unavailable",
        r"API.*failed",
        r"HTTPError",
        r"RequestException",
    ]

    RESOURCE_EXHAUSTION_PATTERNS = [
        r"MemoryError",
        r"OutOfMemory",
        r"OOM",
        r"disk.*full",
        r"too many open files",
        r"resource exhausted",
        r"file handles",
    ]

    def __init__(self, storage: TestResultStorage | None = None):
        """
        Initialize failure analyzer.

        Args:
            storage: Storage backend for test results
        """
        self.storage = storage or TestResultStorage()

    def categorize_failure(
        self, error_message: str | None, error_traceback: str | None
    ) -> FailureCategory:
        """
        Categorize a failure based on error message and traceback.

        Args:
            error_message: Error message from test result
            error_traceback: Error traceback from test result

        Returns:
            FailureCategory enum value
        """
        if not error_message and not error_traceback:
            return FailureCategory.UNKNOWN

        # Combine message and traceback for analysis
        text = f"{error_message or ''}\n{error_traceback or ''}"

        # Check patterns in order of specificity
        if self._matches_patterns(text, self.TIMEOUT_PATTERNS):
            return FailureCategory.TIMEOUT
        if self._matches_patterns(text, self.SETUP_TEARDOWN_PATTERNS):
            return FailureCategory.SETUP_TEARDOWN
        if self._matches_patterns(text, self.EXTERNAL_DEPENDENCY_PATTERNS):
            return FailureCategory.EXTERNAL_DEPENDENCY
        if self._matches_patterns(text, self.RESOURCE_EXHAUSTION_PATTERNS):
            return FailureCategory.RESOURCE_EXHAUSTION
        if self._matches_patterns(text, self.ASSERTION_PATTERNS):
            return FailureCategory.ASSERTION

        return FailureCategory.UNKNOWN

    def extract_error_signature(
        self, error_message: str | None, error_traceback: str | None
    ) -> str:
        """
        Extract a normalized error signature for pattern grouping.

        Removes variable parts like file paths, line numbers, timestamps,
        and specific values while preserving the error type and structure.

        Args:
            error_message: Error message from test result
            error_traceback: Error traceback from test result

        Returns:
            Normalized error signature string
        """
        if not error_message:
            return "NO_ERROR_MESSAGE"

        signature = error_message

        # Extract exception type if present
        exception_match = re.search(r"(\w+Error|\w+Exception):", signature)
        if exception_match:
            exc_type = exception_match.group(1)
        else:
            exc_type = "UnknownError"

        # Normalize common patterns
        # Remove specific file paths (keep just filename)
        signature = re.sub(r"/[a-zA-Z0-9_/.-]+/([a-zA-Z0-9_.-]+\.py)", r"\1", signature)

        # Remove line numbers
        signature = re.sub(r":\d+:", ":N:", signature)
        signature = re.sub(r"line \d+", "line N", signature)

        # Remove specific values in assertions
        signature = re.sub(r"expected: \S+", "expected: X", signature)
        signature = re.sub(r"got: \S+", "got: Y", signature)
        signature = re.sub(r"!= \S+", "!= X", signature)
        signature = re.sub(r"== \S+", "== X", signature)

        # Remove timestamps
        signature = re.sub(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "TIMESTAMP", signature
        )

        # Remove memory addresses
        signature = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", signature)

        # Truncate very long signatures
        if len(signature) > 200:
            signature = signature[:200] + "..."

        return f"{exc_type}: {signature}"

    def compute_flakiness_score(
        self,
        pass_count: int,
        fail_count: int,
        error_count: int = 0,
        skip_count: int = 0,
    ) -> float:
        """
        Compute flakiness score for a test.

        Flakiness score measures inconsistency in test results.
        A test is flaky if it has both passes and failures.

        Formula: (min(pass_count, fail_count + error_count) / total_runs) * 100

        Args:
            pass_count: Number of passing executions
            fail_count: Number of failing executions
            error_count: Number of errored executions
            skip_count: Number of skipped executions

        Returns:
            Flakiness score from 0.0 to 100.0
            Returns 0.0 if test always passes or always fails
        """
        total_runs = pass_count + fail_count + error_count + skip_count
        if total_runs == 0:
            return 0.0

        # Consider both failures and errors as "bad" outcomes
        bad_outcomes = fail_count + error_count

        # Only flaky if has BOTH passes and bad outcomes
        if pass_count == 0 or bad_outcomes == 0:
            return 0.0

        # Score based on minority outcome percentage
        minority_count = min(pass_count, bad_outcomes)
        return (minority_count / total_runs) * 100.0

    def analyze_test_case(
        self, test_case: TestCase
    ) -> FlakinessMetrics | None:
        """
        Analyze a single test case across its execution history.

        Args:
            test_case: TestCase with multiple results

        Returns:
            FlakinessMetrics if test has multiple results, None otherwise
        """
        if not test_case.results:
            return None

        # Count outcomes
        status_counts = defaultdict(int)
        for result in test_case.results:
            status_counts[result.status] += 1

        pass_count = status_counts[TestStatus.PASSED] + status_counts[TestStatus.XPASSED]
        fail_count = status_counts[TestStatus.FAILED]
        error_count = status_counts[TestStatus.ERROR]
        skip_count = status_counts[TestStatus.SKIPPED] + status_counts[TestStatus.XFAILED]

        total_runs = len(test_case.results)

        # Calculate metrics
        flakiness_score = self.compute_flakiness_score(
            pass_count, fail_count, error_count, skip_count
        )
        pass_rate = (pass_count / total_runs * 100.0) if total_runs > 0 else 0.0

        # Analyze failures
        failure_categories: dict[str, int] = defaultdict(int)
        error_signatures: list[str] = []

        for result in test_case.results:
            if result.status in (TestStatus.FAILED, TestStatus.ERROR):
                category = self.categorize_failure(
                    result.error_message, result.error_traceback
                )
                failure_categories[category.value] += 1

                signature = self.extract_error_signature(
                    result.error_message, result.error_traceback
                )
                error_signatures.append(signature)

        # Get most common error signatures (top 5)
        signature_counts: dict[str, int] = defaultdict(int)
        for sig in error_signatures:
            signature_counts[sig] += 1

        common_signatures = sorted(
            signature_counts.keys(), key=lambda s: signature_counts[s], reverse=True
        )[:5]

        # Time window
        timestamps = [r.timestamp for r in test_case.results]
        first_run = min(timestamps) if timestamps else None
        last_run = max(timestamps) if timestamps else None

        return FlakinessMetrics(
            test_case_name=test_case.name,
            total_runs=total_runs,
            pass_count=pass_count,
            fail_count=fail_count,
            skip_count=skip_count,
            error_count=error_count,
            flakiness_score=flakiness_score,
            pass_rate=pass_rate,
            failure_categories=dict(failure_categories),
            common_error_signatures=common_signatures,
            first_run=first_run,
            last_run=last_run,
        )

    def detect_failure_patterns(
        self, test_runs: list[TestRun]
    ) -> list[FailurePattern]:
        """
        Detect common failure patterns across multiple test runs.

        Groups failures by error signature and categorizes them.

        Args:
            test_runs: List of test runs to analyze

        Returns:
            List of FailurePattern objects sorted by occurrence count
        """
        # Collect all failures
        pattern_data: dict[str, dict[str, Any]] = {}

        for run in test_runs:
            for suite in run.suites:
                for case in suite.test_cases:
                    for result in case.results:
                        if result.status not in (TestStatus.FAILED, TestStatus.ERROR):
                            continue

                        # Extract signature and category
                        signature = self.extract_error_signature(
                            result.error_message, result.error_traceback
                        )
                        category = self.categorize_failure(
                            result.error_message, result.error_traceback
                        )

                        # Create pattern ID from signature hash
                        pattern_id = hashlib.md5(signature.encode()).hexdigest()[:16]

                        if pattern_id not in pattern_data:
                            pattern_data[pattern_id] = {
                                "signature": signature,
                                "category": category,
                                "occurrences": 0,
                                "affected_tests": set(),
                                "timestamps": [],
                                "sample_message": result.error_message,
                                "sample_traceback": result.error_traceback,
                            }

                        pattern_data[pattern_id]["occurrences"] += 1
                        pattern_data[pattern_id]["affected_tests"].add(case.name)
                        pattern_data[pattern_id]["timestamps"].append(result.timestamp)

        # Convert to FailurePattern objects
        patterns = []
        for pattern_id, data in pattern_data.items():
            timestamps = data["timestamps"]
            pattern = FailurePattern(
                pattern_id=pattern_id,
                error_signature=data["signature"],
                category=data["category"],
                occurrences=data["occurrences"],
                affected_tests=sorted(data["affected_tests"]),
                first_seen=min(timestamps) if timestamps else None,
                last_seen=max(timestamps) if timestamps else None,
                sample_error_message=data["sample_message"],
                sample_error_traceback=data["sample_traceback"],
            )
            patterns.append(pattern)

        # Sort by occurrence count (most common first)
        patterns.sort(key=lambda p: p.occurrences, reverse=True)

        return patterns

    def analyze_test_runs(
        self,
        run_ids: list[str] | None = None,
        days: int | None = None,
        min_flakiness_score: float = 5.0,
    ) -> FailureAnalysisReport:
        """
        Analyze test runs to detect flaky tests and failure patterns.

        Args:
            run_ids: Specific run IDs to analyze (if None, uses time window)
            days: Number of days to analyze (default: 30, ignored if run_ids provided)
            min_flakiness_score: Minimum flakiness score to include in report

        Returns:
            FailureAnalysisReport with complete analysis
        """
        # Determine which runs to analyze
        if run_ids:
            test_runs = [self.storage.get_test_run(rid) for rid in run_ids]
            test_runs = [r for r in test_runs if r is not None]
        else:
            # Get runs from last N days
            if days is None:
                days = 30
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            run_summaries = self.storage.list_test_runs(
                start_date=start_date, end_date=end_date, limit=1000
            )
            test_runs = [
                self.storage.get_test_run(summary["run_id"])
                for summary in run_summaries
            ]
            test_runs = [r for r in test_runs if r is not None]

        if not test_runs:
            # Return empty report
            return FailureAnalysisReport(
                report_id=str(uuid4()),
                timestamp=datetime.now(),
                analyzed_runs=[],
            )

        # Collect all test cases by name across runs
        test_cases_by_name: dict[str, list[TestCase]] = defaultdict(list)

        for run in test_runs:
            for suite in run.suites:
                for case in suite.test_cases:
                    test_cases_by_name[case.name].append(case)

        # Merge results for each test case
        merged_cases: list[TestCase] = []
        for name, cases in test_cases_by_name.items():
            # Merge all results into single test case
            merged = TestCase(
                case_id=cases[0].case_id,
                name=name,
                file_path=cases[0].file_path,
                line_number=cases[0].line_number,
                test_type=cases[0].test_type,
                markers=cases[0].markers,
            )
            for case in cases:
                merged.results.extend(case.results)
            merged_cases.append(merged)

        # Analyze each test case for flakiness
        flaky_tests: list[FlakinessMetrics] = []
        for case in merged_cases:
            metrics = self.analyze_test_case(case)
            if metrics and metrics.flakiness_score >= min_flakiness_score:
                flaky_tests.append(metrics)

        # Sort by flakiness score (highest first)
        flaky_tests.sort(key=lambda m: m.flakiness_score, reverse=True)

        # Detect failure patterns
        failure_patterns = self.detect_failure_patterns(test_runs)

        # Calculate category distribution
        category_distribution: dict[str, int] = defaultdict(int)
        for pattern in failure_patterns:
            category_distribution[pattern.category.value] += pattern.occurrences

        # Determine failure trend (simple heuristic based on recent vs older failures)
        failure_trend = self._compute_failure_trend(test_runs)

        # Time window
        run_timestamps = [r.timestamp for r in test_runs]
        time_window_start = min(run_timestamps) if run_timestamps else None
        time_window_end = max(run_timestamps) if run_timestamps else None

        return FailureAnalysisReport(
            report_id=str(uuid4()),
            timestamp=datetime.now(),
            analyzed_runs=[r.run_id for r in test_runs],
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            flaky_tests=flaky_tests,
            total_flaky_tests=len(flaky_tests),
            failure_patterns=failure_patterns,
            total_failure_patterns=len(failure_patterns),
            category_distribution=dict(category_distribution),
            failure_trend=failure_trend,
        )

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the given regex patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _compute_failure_trend(self, test_runs: list[TestRun]) -> str:
        """
        Compute failure trend based on recent vs older failures.

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(test_runs) < 2:
            return "stable"

        # Sort by timestamp
        sorted_runs = sorted(test_runs, key=lambda r: r.timestamp)

        # Split into two halves
        mid = len(sorted_runs) // 2
        older_runs = sorted_runs[:mid]
        recent_runs = sorted_runs[mid:]

        # Calculate average failure rate for each half
        def avg_failure_rate(runs: list[TestRun]) -> float:
            if not runs:
                return 0.0
            total_failed = sum(r.failed_tests + r.error_tests for r in runs)
            total_tests = sum(r.total_tests for r in runs)
            return (total_failed / total_tests * 100.0) if total_tests > 0 else 0.0

        older_rate = avg_failure_rate(older_runs)
        recent_rate = avg_failure_rate(recent_runs)

        # Compare with threshold
        threshold = 5.0  # 5% difference
        if recent_rate > older_rate + threshold:
            return "increasing"
        elif recent_rate < older_rate - threshold:
            return "decreasing"
        else:
            return "stable"
