"""
Data models for test result aggregation.

Provides a unified data structure for representing test results from multiple sources
(pytest, cargo test, custom benchmarks, gRPC tests, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TestStatus(str, Enum):
    """Status of a test execution."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    XFAILED = "xfailed"  # Expected failure
    XPASSED = "xpassed"  # Unexpected pass


class FailureCategory(str, Enum):
    """Category of test failure based on error analysis."""

    ASSERTION = "assertion"  # Assertion errors, value mismatches
    TIMEOUT = "timeout"  # Timeout errors, hangs
    SETUP_TEARDOWN = "setup_teardown"  # Fixture failures, cleanup errors
    EXTERNAL_DEPENDENCY = "external_dependency"  # Network, database, service failures
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # OOM, disk space, file handles
    UNKNOWN = "unknown"  # Unclassified errors


class TestType(str, Enum):
    """Type of test."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    FUNCTIONAL = "functional"
    BENCHMARK = "benchmark"
    PERFORMANCE = "performance"
    STRESS = "stress"
    LOAD = "load"
    SECURITY = "security"


class TestSource(str, Enum):
    """Source of test results."""

    PYTEST = "pytest"
    CARGO = "cargo"
    BENCHMARK_JSON = "benchmark_json"
    GRPC_TEST = "grpc_test"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmark and performance tests."""

    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    avg_ms: Optional[float] = None
    median_ms: Optional[float] = None
    std_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    operations_per_second: Optional[float] = None

    # Additional metrics
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None

    # Custom metrics (stored as JSON)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "avg_ms": self.avg_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "operations_per_second": self.operations_per_second,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "custom_metrics": self.custom_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary."""
        return cls(
            min_ms=data.get("min_ms"),
            max_ms=data.get("max_ms"),
            avg_ms=data.get("avg_ms"),
            median_ms=data.get("median_ms"),
            std_ms=data.get("std_ms"),
            p95_ms=data.get("p95_ms"),
            p99_ms=data.get("p99_ms"),
            operations_per_second=data.get("operations_per_second"),
            memory_mb=data.get("memory_mb"),
            cpu_percent=data.get("cpu_percent"),
            custom_metrics=data.get("custom_metrics", {}),
        )


@dataclass
class TestResult:
    """Individual test result with metrics."""

    test_id: str  # Unique ID for this test result
    name: str  # Test name (e.g., "test_hybrid_search")
    status: TestStatus
    duration_ms: float
    timestamp: datetime

    # Optional fields
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Performance metrics (for benchmarks/performance tests)
    performance: Optional[PerformanceMetrics] = None

    # Flexible metadata for source-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "performance": self.performance.to_dict() if self.performance else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
        """Create from dictionary."""
        return cls(
            test_id=data["test_id"],
            name=data["name"],
            status=TestStatus(data["status"]),
            duration_ms=data["duration_ms"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error_message=data.get("error_message"),
            error_traceback=data.get("error_traceback"),
            performance=PerformanceMetrics.from_dict(data["performance"])
            if data.get("performance")
            else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class TestCase:
    """A test case (possibly containing multiple executions/results)."""

    case_id: str  # Unique ID for this test case
    name: str  # Test case name
    file_path: Optional[str] = None  # Source file path
    line_number: Optional[int] = None  # Line number in source file

    # Test classification
    test_type: Optional[TestType] = None
    markers: List[str] = field(default_factory=list)  # pytest markers or tags

    # Results from this test case
    results: List[TestResult] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: TestResult) -> None:
        """Add a test result to this case."""
        self.results.append(result)

    @property
    def latest_result(self) -> Optional[TestResult]:
        """Get the most recent test result."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.timestamp)

    @property
    def latest_status(self) -> Optional[TestStatus]:
        """Get the status of the most recent test result."""
        latest = self.latest_result
        return latest.status if latest else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "case_id": self.case_id,
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "test_type": self.test_type.value if self.test_type else None,
            "markers": self.markers,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary."""
        return cls(
            case_id=data["case_id"],
            name=data["name"],
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            test_type=TestType(data["test_type"]) if data.get("test_type") else None,
            markers=data.get("markers", []),
            results=[TestResult.from_dict(r) for r in data.get("results", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class TestSuite:
    """A suite of related test cases (e.g., unit tests, integration tests)."""

    suite_id: str  # Unique ID for this suite
    name: str  # Suite name (e.g., "unit", "integration", "benchmarks")
    test_type: TestType

    # Test cases in this suite
    test_cases: List[TestCase] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to this suite."""
        self.test_cases.append(test_case)

    @property
    def total_tests(self) -> int:
        """Total number of test results across all cases."""
        return sum(len(case.results) for case in self.test_cases)

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        return sum(
            1
            for case in self.test_cases
            for result in case.results
            if result.status == TestStatus.PASSED
        )

    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        return sum(
            1
            for case in self.test_cases
            for result in case.results
            if result.status == TestStatus.FAILED
        )

    @property
    def skipped_tests(self) -> int:
        """Number of skipped tests."""
        return sum(
            1
            for case in self.test_cases
            for result in case.results
            if result.status == TestStatus.SKIPPED
        )

    @property
    def error_tests(self) -> int:
        """Number of errored tests."""
        return sum(
            1
            for case in self.test_cases
            for result in case.results
            if result.status == TestStatus.ERROR
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "test_type": self.test_type.value,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSuite":
        """Create from dictionary."""
        return cls(
            suite_id=data["suite_id"],
            name=data["name"],
            test_type=TestType(data["test_type"]),
            test_cases=[TestCase.from_dict(tc) for tc in data.get("test_cases", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileCoverage:
    """Coverage metrics for a single source file."""

    file_path: str
    lines_covered: int
    lines_total: int
    line_coverage_percent: float
    uncovered_lines: List[int] = field(default_factory=list)  # Line numbers not covered
    functions_covered: Optional[int] = None
    functions_total: Optional[int] = None
    branches_covered: Optional[int] = None
    branches_total: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "file_path": self.file_path,
            "lines_covered": self.lines_covered,
            "lines_total": self.lines_total,
            "line_coverage_percent": self.line_coverage_percent,
            "uncovered_lines": self.uncovered_lines,
            "functions_covered": self.functions_covered,
            "functions_total": self.functions_total,
            "branches_covered": self.branches_covered,
            "branches_total": self.branches_total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileCoverage":
        """Create from dictionary."""
        return cls(
            file_path=data["file_path"],
            lines_covered=data["lines_covered"],
            lines_total=data["lines_total"],
            line_coverage_percent=data["line_coverage_percent"],
            uncovered_lines=data.get("uncovered_lines", []),
            functions_covered=data.get("functions_covered"),
            functions_total=data.get("functions_total"),
            branches_covered=data.get("branches_covered"),
            branches_total=data.get("branches_total"),
        )


@dataclass
class CoverageMetrics:
    """Code coverage metrics for a test run."""

    # Overall coverage
    line_coverage_percent: float
    lines_covered: int
    lines_total: int

    # Function/method coverage
    function_coverage_percent: Optional[float] = None
    functions_covered: Optional[int] = None
    functions_total: Optional[int] = None

    # Branch coverage (for languages that support it)
    branch_coverage_percent: Optional[float] = None
    branches_covered: Optional[int] = None
    branches_total: Optional[int] = None

    # Per-file coverage breakdown
    file_coverage: List[FileCoverage] = field(default_factory=list)

    # Coverage source (e.g., "coverage.py", "tarpaulin", "llvm-cov")
    coverage_tool: Optional[str] = None

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "line_coverage_percent": self.line_coverage_percent,
            "lines_covered": self.lines_covered,
            "lines_total": self.lines_total,
            "function_coverage_percent": self.function_coverage_percent,
            "functions_covered": self.functions_covered,
            "functions_total": self.functions_total,
            "branch_coverage_percent": self.branch_coverage_percent,
            "branches_covered": self.branches_covered,
            "branches_total": self.branches_total,
            "file_coverage": [f.to_dict() for f in self.file_coverage],
            "coverage_tool": self.coverage_tool,
            "custom_metrics": self.custom_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoverageMetrics":
        """Create from dictionary."""
        return cls(
            line_coverage_percent=data["line_coverage_percent"],
            lines_covered=data["lines_covered"],
            lines_total=data["lines_total"],
            function_coverage_percent=data.get("function_coverage_percent"),
            functions_covered=data.get("functions_covered"),
            functions_total=data.get("functions_total"),
            branch_coverage_percent=data.get("branch_coverage_percent"),
            branches_covered=data.get("branches_covered"),
            branches_total=data.get("branches_total"),
            file_coverage=[
                FileCoverage.from_dict(f) for f in data.get("file_coverage", [])
            ],
            coverage_tool=data.get("coverage_tool"),
            custom_metrics=data.get("custom_metrics", {}),
        )


@dataclass
class TestRun:
    """
    A complete test run aggregating results from multiple sources.

    Represents a single execution cycle (e.g., CI/CD pipeline run, local test execution).
    Can be built incrementally as different test stages complete.
    """

    run_id: str  # Unique ID for this test run
    timestamp: datetime
    source: TestSource

    # Test suites in this run
    suites: List[TestSuite] = field(default_factory=list)

    # Code coverage metrics
    coverage: Optional[CoverageMetrics] = None

    # Run metadata
    environment: Dict[str, Any] = field(default_factory=dict)  # Python version, OS, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # Branch, commit, CI info, etc.

    @classmethod
    def create(
        cls,
        source: TestSource,
        run_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        environment: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TestRun":
        """Create a new test run with auto-generated ID if not provided."""
        return cls(
            run_id=run_id or str(uuid4()),
            timestamp=timestamp or datetime.now(),
            source=source,
            environment=environment or {},
            metadata=metadata or {},
        )

    def add_suite(self, suite: TestSuite) -> None:
        """Add a test suite to this run."""
        self.suites.append(suite)

    @property
    def total_tests(self) -> int:
        """Total number of test results across all suites."""
        return sum(suite.total_tests for suite in self.suites)

    @property
    def passed_tests(self) -> int:
        """Number of passed tests across all suites."""
        return sum(suite.passed_tests for suite in self.suites)

    @property
    def failed_tests(self) -> int:
        """Number of failed tests across all suites."""
        return sum(suite.failed_tests for suite in self.suites)

    @property
    def skipped_tests(self) -> int:
        """Number of skipped tests across all suites."""
        return sum(suite.skipped_tests for suite in self.suites)

    @property
    def error_tests(self) -> int:
        """Number of errored tests across all suites."""
        return sum(suite.error_tests for suite in self.suites)

    @property
    def success_rate(self) -> float:
        """Success rate (passed / total) as percentage."""
        total = self.total_tests
        if total == 0:
            return 0.0
        return (self.passed_tests / total) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "suites": [s.to_dict() for s in self.suites],
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "environment": self.environment,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestRun":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=TestSource(data["source"]),
            suites=[TestSuite.from_dict(s) for s in data.get("suites", [])],
            coverage=CoverageMetrics.from_dict(data["coverage"])
            if data.get("coverage")
            else None,
            environment=data.get("environment", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FailurePattern:
    """
    A pattern of similar failures across multiple test executions.

    Groups failures with similar error messages or characteristics.
    """

    pattern_id: str  # Unique identifier for this pattern
    error_signature: str  # Normalized error signature (e.g., "AssertionError: expected X got Y")
    category: FailureCategory  # Classification of failure type
    occurrences: int  # Number of times this pattern occurred
    affected_tests: List[str] = field(default_factory=list)  # Test case names
    first_seen: Optional[datetime] = None  # When first observed
    last_seen: Optional[datetime] = None  # When last observed

    # Sample error details
    sample_error_message: Optional[str] = None
    sample_error_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "error_signature": self.error_signature,
            "category": self.category.value,
            "occurrences": self.occurrences,
            "affected_tests": self.affected_tests,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "sample_error_message": self.sample_error_message,
            "sample_error_traceback": self.sample_error_traceback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailurePattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            error_signature=data["error_signature"],
            category=FailureCategory(data["category"]),
            occurrences=data["occurrences"],
            affected_tests=data.get("affected_tests", []),
            first_seen=datetime.fromisoformat(data["first_seen"]) if data.get("first_seen") else None,
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
            sample_error_message=data.get("sample_error_message"),
            sample_error_traceback=data.get("sample_error_traceback"),
        )


@dataclass
class FlakinessMetrics:
    """
    Metrics for test flakiness detection.

    A test is considered flaky if it has inconsistent results across multiple runs
    (i.e., sometimes passes, sometimes fails without code changes).
    """

    test_case_name: str  # Name of the test case
    total_runs: int  # Total number of test executions analyzed
    pass_count: int  # Number of passing executions
    fail_count: int  # Number of failing executions
    skip_count: int  # Number of skipped executions
    error_count: int  # Number of errored executions

    # Flakiness score: 0.0 (never flaky) to 100.0 (extremely flaky)
    # Calculated as: (min(pass_count, fail_count) / total_runs) * 100
    # Only non-zero if test has BOTH passes and failures
    flakiness_score: float

    # Pass rate percentage
    pass_rate: float

    # Failure analysis
    failure_categories: Dict[str, int] = field(default_factory=dict)  # Category -> count
    common_error_signatures: List[str] = field(default_factory=list)  # Most common errors

    # Time window analyzed
    first_run: Optional[datetime] = None
    last_run: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "test_case_name": self.test_case_name,
            "total_runs": self.total_runs,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "error_count": self.error_count,
            "flakiness_score": self.flakiness_score,
            "pass_rate": self.pass_rate,
            "failure_categories": self.failure_categories,
            "common_error_signatures": self.common_error_signatures,
            "first_run": self.first_run.isoformat() if self.first_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlakinessMetrics":
        """Create from dictionary."""
        return cls(
            test_case_name=data["test_case_name"],
            total_runs=data["total_runs"],
            pass_count=data["pass_count"],
            fail_count=data["fail_count"],
            skip_count=data["skip_count"],
            error_count=data["error_count"],
            flakiness_score=data["flakiness_score"],
            pass_rate=data["pass_rate"],
            failure_categories=data.get("failure_categories", {}),
            common_error_signatures=data.get("common_error_signatures", []),
            first_run=datetime.fromisoformat(data["first_run"]) if data.get("first_run") else None,
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
        )


@dataclass
class FailureAnalysisReport:
    """
    Complete failure analysis report for a test run or set of runs.

    Contains flakiness metrics, failure patterns, and categorized failures.
    """

    report_id: str  # Unique identifier
    timestamp: datetime  # When analysis was performed

    # Analysis scope
    analyzed_runs: List[str] = field(default_factory=list)  # Run IDs analyzed
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    # Flakiness analysis
    flaky_tests: List[FlakinessMetrics] = field(default_factory=list)  # Sorted by flakiness score
    total_flaky_tests: int = 0

    # Failure patterns
    failure_patterns: List[FailurePattern] = field(default_factory=list)  # Sorted by occurrence
    total_failure_patterns: int = 0

    # Failure categories distribution
    category_distribution: Dict[str, int] = field(default_factory=dict)  # Category -> count

    # Trends
    failure_trend: Optional[str] = None  # "increasing", "decreasing", "stable"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "analyzed_runs": self.analyzed_runs,
            "time_window_start": self.time_window_start.isoformat() if self.time_window_start else None,
            "time_window_end": self.time_window_end.isoformat() if self.time_window_end else None,
            "flaky_tests": [f.to_dict() for f in self.flaky_tests],
            "total_flaky_tests": self.total_flaky_tests,
            "failure_patterns": [p.to_dict() for p in self.failure_patterns],
            "total_failure_patterns": self.total_failure_patterns,
            "category_distribution": self.category_distribution,
            "failure_trend": self.failure_trend,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureAnalysisReport":
        """Create from dictionary."""
        return cls(
            report_id=data["report_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            analyzed_runs=data.get("analyzed_runs", []),
            time_window_start=datetime.fromisoformat(data["time_window_start"]) if data.get("time_window_start") else None,
            time_window_end=datetime.fromisoformat(data["time_window_end"]) if data.get("time_window_end") else None,
            flaky_tests=[FlakinessMetrics.from_dict(f) for f in data.get("flaky_tests", [])],
            total_flaky_tests=data.get("total_flaky_tests", 0),
            failure_patterns=[FailurePattern.from_dict(p) for p in data.get("failure_patterns", [])],
            total_failure_patterns=data.get("total_failure_patterns", 0),
            category_distribution=data.get("category_distribution", {}),
            failure_trend=data.get("failure_trend"),
            metadata=data.get("metadata", {}),
        )
