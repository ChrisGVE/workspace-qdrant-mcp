"""
Unit tests for failure pattern analysis and flakiness detection.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.reporting.failure_analyzer import FailureAnalyzer
from tests.reporting.models import (
    FailureCategory,
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)
from tests.reporting.storage import TestResultStorage


@pytest.fixture
def temp_storage():
    """Create temporary storage for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_results.db"
        storage = TestResultStorage(db_path)
        yield storage


@pytest.fixture
def analyzer(temp_storage):
    """Create failure analyzer with temporary storage."""
    return FailureAnalyzer(temp_storage)


class TestFailureCategorization:
    """Test failure categorization logic."""

    def test_categorize_assertion_error(self, analyzer):
        """Test categorization of assertion failures."""
        error_msg = "AssertionError: expected 5 but got 3"
        traceback = "test_foo.py:42: AssertionError"

        category = analyzer.categorize_failure(error_msg, traceback)
        assert category == FailureCategory.ASSERTION

    def test_categorize_timeout_error(self, analyzer):
        """Test categorization of timeout failures."""
        error_msg = "TimeoutError: operation timed out after 30s"
        traceback = "test_slow.py:15: TimeoutError"

        category = analyzer.categorize_failure(error_msg, traceback)
        assert category == FailureCategory.TIMEOUT

    def test_categorize_setup_error(self, analyzer):
        """Test categorization of setup/teardown failures."""
        error_msg = "fixture 'db_connection' not found"
        traceback = "conftest.py:20: fixture error"

        category = analyzer.categorize_failure(error_msg, traceback)
        assert category == FailureCategory.SETUP_TEARDOWN

    def test_categorize_external_dependency(self, analyzer):
        """Test categorization of external dependency failures."""
        error_msg = "ConnectionError: database connection refused"
        traceback = "test_api.py:55: ConnectionError"

        category = analyzer.categorize_failure(error_msg, traceback)
        assert category == FailureCategory.EXTERNAL_DEPENDENCY

    def test_categorize_resource_exhaustion(self, analyzer):
        """Test categorization of resource exhaustion failures."""
        error_msg = "MemoryError: out of memory"
        traceback = "test_large.py:100: MemoryError"

        category = analyzer.categorize_failure(error_msg, traceback)
        assert category == FailureCategory.RESOURCE_EXHAUSTION

    def test_categorize_unknown(self, analyzer):
        """Test categorization of unknown failure types."""
        error_msg = "SomeWeirdError: something went wrong"
        traceback = "test_mystery.py:1: SomeWeirdError"

        category = analyzer.categorize_failure(error_msg, traceback)
        assert category == FailureCategory.UNKNOWN

    def test_categorize_none_inputs(self, analyzer):
        """Test categorization with None inputs."""
        category = analyzer.categorize_failure(None, None)
        assert category == FailureCategory.UNKNOWN


class TestErrorSignatureExtraction:
    """Test error signature extraction and normalization."""

    def test_extract_basic_signature(self, analyzer):
        """Test basic error signature extraction."""
        error_msg = "AssertionError: expected 5 but got 3"
        signature = analyzer.extract_error_signature(error_msg, None)

        assert "AssertionError" in signature
        # Signature should be extracted (not empty)
        assert len(signature) > 0

    def test_extract_signature_with_file_paths(self, analyzer):
        """Test signature extraction normalizes file paths."""
        error_msg = "Error in /long/path/to/test_file.py:42"
        signature = analyzer.extract_error_signature(error_msg, None)

        # Should keep only filename, remove full path
        assert "test_file.py" in signature
        assert "/long/path/to/" not in signature

    def test_extract_signature_with_line_numbers(self, analyzer):
        """Test signature extraction normalizes line numbers."""
        error_msg = "Error at line 123: AssertionError"
        signature = analyzer.extract_error_signature(error_msg, None)

        # Should normalize line numbers
        assert "line N" in signature or "123" not in signature

    def test_extract_signature_none_input(self, analyzer):
        """Test signature extraction with None input."""
        signature = analyzer.extract_error_signature(None, None)
        assert signature == "NO_ERROR_MESSAGE"


class TestFlakinessScoreCalculation:
    """Test flakiness score calculation."""

    def test_score_always_passes(self, analyzer):
        """Test score for test that always passes."""
        score = analyzer.compute_flakiness_score(
            pass_count=10, fail_count=0, error_count=0
        )
        assert score == 0.0

    def test_score_always_fails(self, analyzer):
        """Test score for test that always fails."""
        score = analyzer.compute_flakiness_score(
            pass_count=0, fail_count=10, error_count=0
        )
        assert score == 0.0

    def test_score_50_50_split(self, analyzer):
        """Test score for test with 50/50 pass/fail split."""
        score = analyzer.compute_flakiness_score(
            pass_count=5, fail_count=5, error_count=0
        )
        # 50% flakiness (5 minority / 10 total * 100)
        assert score == 50.0

    def test_score_with_errors(self, analyzer):
        """Test score includes errors in bad outcomes."""
        score = analyzer.compute_flakiness_score(
            pass_count=7, fail_count=2, error_count=1
        )
        # 3 bad outcomes (2 fail + 1 error) vs 7 passes
        # min(7, 3) / 10 * 100 = 30%
        assert score == 30.0

    def test_score_mostly_passes(self, analyzer):
        """Test score for mostly passing test."""
        score = analyzer.compute_flakiness_score(
            pass_count=9, fail_count=1, error_count=0
        )
        # 1 minority / 10 total * 100 = 10%
        assert score == 10.0

    def test_score_with_skips(self, analyzer):
        """Test score calculation with skipped tests."""
        score = analyzer.compute_flakiness_score(
            pass_count=5, fail_count=3, error_count=0, skip_count=2
        )
        # Total 10 runs, minority is 3 (failures) vs 5 passes
        # min(5, 3) / 10 * 100 = 30%
        assert score == 30.0


class TestCaseAnalysis:
    """Test analyzing individual test cases."""

    def test_analyze_single_test_case(self, analyzer):
        """Test analysis of a single test case with multiple results."""
        test_case = TestCase(
            case_id="test-1",
            name="test_flaky_example",
            test_type=TestType.UNIT,
        )

        # Add mixed results (flaky test)
        base_time = datetime.now()
        for i in range(10):
            status = TestStatus.PASSED if i % 2 == 0 else TestStatus.FAILED
            error_msg = "AssertionError: flaky" if status == TestStatus.FAILED else None

            result = TestResult(
                test_id=f"result-{i}",
                name="test_flaky_example",
                status=status,
                duration_ms=100.0,
                timestamp=base_time + timedelta(seconds=i),
                error_message=error_msg,
            )
            test_case.add_result(result)

        metrics = analyzer.analyze_test_case(test_case)

        assert metrics is not None
        assert metrics.test_case_name == "test_flaky_example"
        assert metrics.total_runs == 10
        assert metrics.pass_count == 5
        assert metrics.fail_count == 5
        assert metrics.flakiness_score == 50.0
        assert metrics.pass_rate == 50.0

    def test_analyze_no_results(self, analyzer):
        """Test analysis of test case with no results."""
        test_case = TestCase(
            case_id="test-2",
            name="test_empty",
            test_type=TestType.UNIT,
        )

        metrics = analyzer.analyze_test_case(test_case)
        assert metrics is None


class TestFailurePatternDetection:
    """Test failure pattern detection across test runs."""

    def test_detect_patterns(self, analyzer, temp_storage):
        """Test detection of common failure patterns."""
        # Create two test runs with similar failures
        test_runs = []

        for run_num in range(2):
            test_run = TestRun.create(source=TestSource.PYTEST)

            suite = TestSuite(
                suite_id=f"suite-{run_num}",
                name="unit",
                test_type=TestType.UNIT,
            )

            # Add test cases with same error pattern
            for test_num in range(3):
                case = TestCase(
                    case_id=f"case-{run_num}-{test_num}",
                    name=f"test_{test_num}",
                    test_type=TestType.UNIT,
                )

                # All have same error pattern
                result = TestResult(
                    test_id=f"result-{run_num}-{test_num}",
                    name=f"test_{test_num}",
                    status=TestStatus.FAILED,
                    duration_ms=50.0,
                    timestamp=datetime.now(),
                    error_message="AssertionError: expected 5 but got 3",
                    error_traceback="test.py:10: AssertionError",
                )
                case.add_result(result)
                suite.add_test_case(case)

            test_run.add_suite(suite)
            test_runs.append(test_run)

        patterns = analyzer.detect_failure_patterns(test_runs)

        # Should detect one common pattern (6 total occurrences)
        assert len(patterns) > 0
        assert patterns[0].occurrences == 6
        assert patterns[0].category == FailureCategory.ASSERTION
        assert len(patterns[0].affected_tests) == 3  # test_0, test_1, test_2


class TestFullAnalysis:
    """Test end-to-end failure analysis."""

    def test_analyze_test_runs(self, analyzer, temp_storage):
        """Test complete analysis of multiple test runs."""
        # Create test runs with flaky tests and patterns
        test_runs = []

        for run_num in range(5):
            test_run = TestRun.create(source=TestSource.PYTEST)

            suite = TestSuite(
                suite_id=f"suite-{run_num}",
                name="unit",
                test_type=TestType.UNIT,
            )

            # Add a flaky test (alternates pass/fail)
            flaky_case = TestCase(
                case_id=f"flaky-{run_num}",
                name="test_flaky",
                test_type=TestType.UNIT,
            )

            status = TestStatus.PASSED if run_num % 2 == 0 else TestStatus.FAILED
            error_msg = "AssertionError: flaky" if status == TestStatus.FAILED else None

            result = TestResult(
                test_id=f"flaky-result-{run_num}",
                name="test_flaky",
                status=status,
                duration_ms=100.0,
                timestamp=datetime.now() + timedelta(seconds=run_num),
                error_message=error_msg,
            )
            flaky_case.add_result(result)
            suite.add_test_case(flaky_case)

            # Add a consistently failing test
            failing_case = TestCase(
                case_id=f"failing-{run_num}",
                name="test_always_fails",
                test_type=TestType.UNIT,
            )

            result = TestResult(
                test_id=f"failing-result-{run_num}",
                name="test_always_fails",
                status=TestStatus.FAILED,
                duration_ms=50.0,
                timestamp=datetime.now() + timedelta(seconds=run_num),
                error_message="TimeoutError: operation timed out",
            )
            failing_case.add_result(result)
            suite.add_test_case(failing_case)

            test_run.add_suite(suite)
            temp_storage.save_test_run(test_run)
            test_runs.append(test_run)

        # Analyze all runs
        report = analyzer.analyze_test_runs(
            run_ids=[r.run_id for r in test_runs],
            min_flakiness_score=5.0,
        )

        # Verify report contents
        assert report.total_flaky_tests >= 1  # Should detect test_flaky
        assert report.total_failure_patterns >= 1  # Should detect patterns

        # Check flaky test detection
        flaky_test_names = [f.test_case_name for f in report.flaky_tests]
        assert "test_flaky" in flaky_test_names

        # Check failure patterns
        assert len(report.failure_patterns) > 0
        assert report.category_distribution  # Should have category counts

    def test_analyze_empty_runs(self, analyzer):
        """Test analysis with no test runs."""
        report = analyzer.analyze_test_runs(run_ids=[])

        assert report.total_flaky_tests == 0
        assert report.total_failure_patterns == 0
        assert len(report.analyzed_runs) == 0


class TestStorageIntegration:
    """Test storage and retrieval of failure analysis reports."""

    def test_save_and_retrieve_report(self, analyzer, temp_storage):
        """Test saving and retrieving failure analysis reports."""
        # Create a simple test run
        test_run = TestRun.create(source=TestSource.PYTEST)
        suite = TestSuite(suite_id="suite-1", name="unit", test_type=TestType.UNIT)
        test_run.add_suite(suite)
        temp_storage.save_test_run(test_run)

        # Analyze and save report
        report = analyzer.analyze_test_runs(run_ids=[test_run.run_id])
        temp_storage.save_failure_analysis_report(report)

        # Retrieve report
        retrieved = temp_storage.get_failure_analysis_report(report.report_id)

        assert retrieved is not None
        assert retrieved.report_id == report.report_id
        assert retrieved.analyzed_runs == report.analyzed_runs

    def test_list_failure_reports(self, analyzer, temp_storage):
        """Test listing failure analysis reports."""
        # Create and save multiple reports
        for i in range(3):
            test_run = TestRun.create(source=TestSource.PYTEST)
            suite = TestSuite(suite_id=f"suite-{i}", name="unit", test_type=TestType.UNIT)
            test_run.add_suite(suite)
            temp_storage.save_test_run(test_run)

            report = analyzer.analyze_test_runs(run_ids=[test_run.run_id])
            temp_storage.save_failure_analysis_report(report)

        # List reports
        reports = temp_storage.list_failure_analysis_reports(limit=10)

        assert len(reports) == 3
        assert all("report_id" in r for r in reports)
        assert all("timestamp" in r for r in reports)
