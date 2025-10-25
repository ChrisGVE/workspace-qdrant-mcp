"""
Automated Test Infrastructure and Reporting

Comprehensive automated test execution pipeline, reporting and metrics collection,
CI/CD integration, and performance regression testing infrastructure.

This module implements subtask 203.8 of the End-to-End Functional Testing Framework.
"""

import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import pytest


@dataclass
class TestResult:
    """Represents a single test result."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    timestamp: str | None = None
    test_file: str | None = None
    test_category: str | None = None


@dataclass
class TestSuiteResult:
    """Represents results from a complete test suite run."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    timestamp: str
    platform_info: dict[str, Any]
    test_results: list[TestResult]
    performance_metrics: dict[str, Any] | None = None


class TestInfrastructureManager:
    """Manages automated test infrastructure and execution."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "test_results" / "reports"
        self.metrics_dir = project_root / "test_results" / "metrics"
        self.logs_dir = project_root / "test_results" / "logs"
        self.database_path = project_root / "test_results" / "test_history.db"

        self.setup_infrastructure()

    def setup_infrastructure(self):
        """Set up test infrastructure directories and database."""
        # Create directories
        for directory in [self.reports_dir, self.metrics_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for test history tracking."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_suites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_tests INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    skipped INTEGER,
                    errors INTEGER,
                    total_duration REAL,
                    platform_system TEXT,
                    platform_python_version TEXT,
                    git_commit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_id INTEGER,
                    test_name TEXT NOT NULL,
                    test_file TEXT,
                    test_category TEXT,
                    status TEXT NOT NULL,
                    duration REAL,
                    error_message TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (suite_id) REFERENCES test_suites (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_id INTEGER,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_unit TEXT,
                    test_name TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (suite_id) REFERENCES test_suites (id)
                )
            """)

            conn.commit()

    def get_platform_info(self) -> dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node()
        }

    def get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def run_test_suite(
        self,
        test_pattern: str = "tests/functional/",
        markers: list[str] | None = None,
        parallel: bool = False,
        coverage: bool = True
    ) -> TestSuiteResult:
        """Run a test suite and collect results."""
        # Prepare pytest command
        cmd = ["python", "-m", "pytest", test_pattern, "-v", "--tb=short"]

        # Add markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])

        # Add coverage
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=json", "--cov-report=html"])

        # Add JSON report
        json_report_path = self.reports_dir / f"pytest_report_{int(time.time())}.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report_path}"])

        # Set environment
        env = os.environ.copy()
        env.update({
            "PYTEST_CURRENT_TEST": "1",
            "PYTHONPATH": str(self.project_root),
        })

        # Execute tests
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                env=env,
                timeout=1800  # 30 minutes timeout
            )

            total_duration = time.time() - start_time

            # Parse results
            suite_result = self._parse_pytest_results(
                json_report_path, timestamp, total_duration, result
            )

            # Store results
            self.store_test_results(suite_result)

            return suite_result

        except subprocess.TimeoutExpired:
            total_duration = time.time() - start_time
            return TestSuiteResult(
                suite_name="timeout",
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=0,
                total_duration=total_duration,
                timestamp=timestamp,
                platform_info=self.get_platform_info(),
                test_results=[TestResult(
                    test_name="test_suite_execution",
                    status="error",
                    duration=total_duration,
                    error_message="Test suite execution timed out"
                )]
            )

    def _parse_pytest_results(
        self,
        json_report_path: Path,
        timestamp: str,
        total_duration: float,
        process_result: subprocess.CompletedProcess
    ) -> TestSuiteResult:
        """Parse pytest JSON results."""
        test_results = []
        summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }

        # Try to parse JSON report
        if json_report_path.exists():
            try:
                with open(json_report_path) as f:
                    pytest_data = json.load(f)

                # Parse individual test results
                for test in pytest_data.get("tests", []):
                    status = test.get("outcome", "unknown")
                    test_result = TestResult(
                        test_name=test.get("nodeid", "unknown"),
                        status=status,
                        duration=test.get("duration", 0.0),
                        error_message=self._extract_error_message(test),
                        timestamp=timestamp,
                        test_file=self._extract_test_file(test.get("nodeid", "")),
                        test_category=self._extract_test_category(test.get("nodeid", ""))
                    )
                    test_results.append(test_result)

                    # Update summary
                    summary["total"] += 1
                    if status == "passed":
                        summary["passed"] += 1
                    elif status == "failed":
                        summary["failed"] += 1
                    elif status == "skipped":
                        summary["skipped"] += 1
                    else:
                        summary["errors"] += 1

            except (json.JSONDecodeError, FileNotFoundError):
                # Fallback to process output parsing
                summary = self._parse_pytest_output(process_result.stdout)
        else:
            # Fallback parsing
            summary = self._parse_pytest_output(process_result.stdout)

        return TestSuiteResult(
            suite_name="functional_tests",
            total_tests=summary["total"],
            passed=summary["passed"],
            failed=summary["failed"],
            skipped=summary["skipped"],
            errors=summary["errors"],
            total_duration=total_duration,
            timestamp=timestamp,
            platform_info=self.get_platform_info(),
            test_results=test_results
        )

    def _extract_error_message(self, test_data: dict[str, Any]) -> str | None:
        """Extract error message from test data."""
        if "call" in test_data and "longrepr" in test_data["call"]:
            return str(test_data["call"]["longrepr"])[:500]  # Truncate long messages
        return None

    def _extract_test_file(self, nodeid: str) -> str:
        """Extract test file from node ID."""
        if "::" in nodeid:
            return nodeid.split("::")[0]
        return nodeid

    def _extract_test_category(self, nodeid: str) -> str:
        """Extract test category from node ID."""
        if "functional" in nodeid:
            return "functional"
        elif "unit" in nodeid:
            return "unit"
        elif "integration" in nodeid:
            return "integration"
        else:
            return "other"

    def _parse_pytest_output(self, output: str) -> dict[str, int]:
        """Parse pytest output for test counts."""
        summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Look for summary line
        lines = output.split('\n')
        for line in lines:
            if "passed" in line or "failed" in line or "skipped" in line:
                # Parse various pytest summary formats
                words = line.split()
                for i, word in enumerate(words):
                    if word == "passed" and i > 0:
                        try:
                            summary["passed"] = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif word == "failed" and i > 0:
                        try:
                            summary["failed"] = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif word == "skipped" and i > 0:
                        try:
                            summary["skipped"] = int(words[i-1])
                        except (ValueError, IndexError):
                            pass

        summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"] + summary["errors"]
        return summary

    def store_test_results(self, suite_result: TestSuiteResult):
        """Store test results in database."""
        with sqlite3.connect(self.database_path) as conn:
            # Insert suite record
            cursor = conn.execute("""
                INSERT INTO test_suites (
                    suite_name, timestamp, total_tests, passed, failed,
                    skipped, errors, total_duration, platform_system,
                    platform_python_version, git_commit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suite_result.suite_name,
                suite_result.timestamp,
                suite_result.total_tests,
                suite_result.passed,
                suite_result.failed,
                suite_result.skipped,
                suite_result.errors,
                suite_result.total_duration,
                suite_result.platform_info.get("system"),
                suite_result.platform_info.get("python_version"),
                self.get_git_commit()
            ))

            suite_id = cursor.lastrowid

            # Insert individual test results
            for test_result in suite_result.test_results:
                conn.execute("""
                    INSERT INTO test_results (
                        suite_id, test_name, test_file, test_category,
                        status, duration, error_message, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    suite_id,
                    test_result.test_name,
                    test_result.test_file,
                    test_result.test_category,
                    test_result.status,
                    test_result.duration,
                    test_result.error_message,
                    test_result.timestamp
                ))

            conn.commit()

    def generate_test_report(self, format: str = "html") -> Path:
        """Generate comprehensive test report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "html":
            return self._generate_html_report(timestamp)
        elif format == "json":
            return self._generate_json_report(timestamp)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_html_report(self, timestamp: str) -> Path:
        """Generate HTML test report."""
        report_path = self.reports_dir / f"test_report_{timestamp}.html"

        # Get recent test data
        with sqlite3.connect(self.database_path) as conn:
            # Get latest suite
            suite_data = conn.execute("""
                SELECT * FROM test_suites
                ORDER BY created_at DESC
                LIMIT 1
            """).fetchone()

            if not suite_data:
                # Create empty report
                html_content = "<html><body><h1>No test data available</h1></body></html>"
            else:
                # Get test results for latest suite
                test_results = conn.execute("""
                    SELECT * FROM test_results
                    WHERE suite_id = ?
                    ORDER BY test_name
                """, (suite_data[0],)).fetchall()

                html_content = self._create_html_content(suite_data, test_results)

        # Write HTML report
        with open(report_path, 'w') as f:
            f.write(html_content)

        return report_path

    def _create_html_content(self, suite_data: tuple, test_results: list[tuple]) -> str:
        """Create HTML content for test report."""
        # Basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - {suite_data[1]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
                .error {{ color: darkred; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .test-name {{ font-family: monospace; }}
            </style>
        </head>
        <body>
            <h1>Functional Test Report</h1>
            <div class="summary">
                <h2>Test Suite Summary</h2>
                <p><strong>Suite:</strong> {suite_data[1]}</p>
                <p><strong>Timestamp:</strong> {suite_data[2]}</p>
                <p><strong>Platform:</strong> {suite_data[9]} (Python {suite_data[10]})</p>
                <p><strong>Duration:</strong> {suite_data[8]:.2f} seconds</p>
                <p><strong>Total Tests:</strong> {suite_data[3]}</p>
                <p><strong class="passed">Passed:</strong> {suite_data[4]}</p>
                <p><strong class="failed">Failed:</strong> {suite_data[5]}</p>
                <p><strong class="skipped">Skipped:</strong> {suite_data[6]}</p>
                <p><strong class="error">Errors:</strong> {suite_data[7]}</p>
            </div>

            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Category</th>
                    <th>Error Message</th>
                </tr>
        """

        # Add test result rows
        for result in test_results:
            status_class = result[5].lower()  # status
            error_msg = result[7][:100] + "..." if result[7] and len(result[7]) > 100 else (result[7] or "")

            html += f"""
                <tr>
                    <td class="test-name">{result[2]}</td>
                    <td class="{status_class}">{result[5].upper()}</td>
                    <td>{result[6]:.3f}s</td>
                    <td>{result[4] or 'N/A'}</td>
                    <td>{error_msg}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def _generate_json_report(self, timestamp: str) -> Path:
        """Generate JSON test report."""
        report_path = self.reports_dir / f"test_report_{timestamp}.json"

        with sqlite3.connect(self.database_path) as conn:
            # Get latest suite
            suite_data = conn.execute("""
                SELECT * FROM test_suites
                ORDER BY created_at DESC
                LIMIT 1
            """).fetchone()

            if suite_data:
                # Get test results
                test_results = conn.execute("""
                    SELECT * FROM test_results
                    WHERE suite_id = ?
                """, (suite_data[0],)).fetchall()

                report_data = {
                    "suite_name": suite_data[1],
                    "timestamp": suite_data[2],
                    "summary": {
                        "total_tests": suite_data[3],
                        "passed": suite_data[4],
                        "failed": suite_data[5],
                        "skipped": suite_data[6],
                        "errors": suite_data[7],
                        "duration": suite_data[8]
                    },
                    "platform": {
                        "system": suite_data[9],
                        "python_version": suite_data[10]
                    },
                    "git_commit": suite_data[11],
                    "test_results": [
                        {
                            "test_name": result[2],
                            "test_file": result[3],
                            "test_category": result[4],
                            "status": result[5],
                            "duration": result[6],
                            "error_message": result[7]
                        }
                        for result in test_results
                    ]
                }
            else:
                report_data = {"error": "No test data available"}

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return report_path

    def check_performance_regression(self, threshold: float = 0.2) -> dict[str, Any]:
        """Check for performance regressions compared to historical data."""
        with sqlite3.connect(self.database_path) as conn:
            # Get recent test suite durations
            recent_suites = conn.execute("""
                SELECT total_duration, timestamp
                FROM test_suites
                ORDER BY created_at DESC
                LIMIT 10
            """).fetchall()

            if len(recent_suites) < 2:
                return {"status": "insufficient_data", "message": "Not enough historical data"}

            # Calculate baseline (average of previous runs excluding latest)
            latest_duration = recent_suites[0][0]
            historical_durations = [suite[0] for suite in recent_suites[1:]]
            baseline_duration = statistics.mean(historical_durations)

            # Check for regression
            regression_ratio = (latest_duration - baseline_duration) / baseline_duration

            return {
                "status": "regression" if regression_ratio > threshold else "normal",
                "latest_duration": latest_duration,
                "baseline_duration": baseline_duration,
                "regression_ratio": regression_ratio,
                "threshold": threshold,
                "message": f"Latest run: {latest_duration:.2f}s, Baseline: {baseline_duration:.2f}s, Change: {regression_ratio:.1%}"
            }

    def generate_ci_summary(self) -> dict[str, Any]:
        """Generate CI/CD friendly summary."""
        with sqlite3.connect(self.database_path) as conn:
            latest_suite = conn.execute("""
                SELECT * FROM test_suites
                ORDER BY created_at DESC
                LIMIT 1
            """).fetchone()

            if not latest_suite:
                return {"status": "no_data", "message": "No test data available"}

            # Calculate success rate
            total_tests = latest_suite[3]
            passed_tests = latest_suite[4]
            success_rate = (passed_tests / total_tests) if total_tests > 0 else 0

            # Determine overall status
            if latest_suite[5] > 0 or latest_suite[7] > 0:  # failures or errors
                overall_status = "failed"
            elif success_rate < 0.95:  # Less than 95% success
                overall_status = "unstable"
            else:
                overall_status = "passed"

            return {
                "status": overall_status,
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": latest_suite[5],
                "skipped": latest_suite[6],
                "errors": latest_suite[7],
                "success_rate": success_rate,
                "duration": latest_suite[8],
                "timestamp": latest_suite[2],
                "platform": f"{latest_suite[9]} Python {latest_suite[10]}",
                "git_commit": latest_suite[11]
            }


@pytest.mark.functional
@pytest.mark.test_infrastructure
class TestAutomatedInfrastructureAndReporting:
    """Test the automated test infrastructure and reporting system."""

    @pytest.fixture
    def infra_manager(self, tmp_path):
        """Create test infrastructure manager."""
        # Use a temporary path for testing
        return TestInfrastructureManager(tmp_path)

    def test_infrastructure_setup(self, infra_manager):
        """Test infrastructure setup and initialization."""
        # Verify directories were created
        assert infra_manager.reports_dir.exists()
        assert infra_manager.metrics_dir.exists()
        assert infra_manager.logs_dir.exists()

        # Verify database was initialized
        assert infra_manager.database_path.exists()

        # Test database schema
        with sqlite3.connect(infra_manager.database_path) as conn:
            tables = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
            """).fetchall()

            table_names = [table[0] for table in tables]
            assert "test_suites" in table_names
            assert "test_results" in table_names
            assert "performance_metrics" in table_names

    def test_platform_info_collection(self, infra_manager):
        """Test platform information collection."""
        platform_info = infra_manager.get_platform_info()

        # Verify required fields
        required_fields = [
            "system", "python_version", "architecture", "hostname"
        ]

        for field in required_fields:
            assert field in platform_info
            assert platform_info[field] is not None
            assert len(str(platform_info[field])) > 0

    def test_git_commit_detection(self, infra_manager):
        """Test git commit hash detection."""
        commit_hash = infra_manager.get_git_commit()

        # Should either get a commit hash or None
        if commit_hash is not None:
            assert isinstance(commit_hash, str)
            assert len(commit_hash) >= 7  # Short commit hash minimum

    def test_test_result_storage(self, infra_manager):
        """Test storing and retrieving test results."""
        # Create mock test suite result
        test_results = [
            TestResult(
                test_name="test_example_1",
                status="passed",
                duration=1.5,
                test_file="test_example.py",
                test_category="functional"
            ),
            TestResult(
                test_name="test_example_2",
                status="failed",
                duration=0.8,
                error_message="Assertion failed",
                test_file="test_example.py",
                test_category="functional"
            )
        ]

        suite_result = TestSuiteResult(
            suite_name="test_storage",
            total_tests=2,
            passed=1,
            failed=1,
            skipped=0,
            errors=0,
            total_duration=2.3,
            timestamp=datetime.now().isoformat(),
            platform_info=infra_manager.get_platform_info(),
            test_results=test_results
        )

        # Store results
        infra_manager.store_test_results(suite_result)

        # Verify storage
        with sqlite3.connect(infra_manager.database_path) as conn:
            suite_count = conn.execute("SELECT COUNT(*) FROM test_suites").fetchone()[0]
            result_count = conn.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]

            assert suite_count >= 1
            assert result_count >= 2

    def test_html_report_generation(self, infra_manager):
        """Test HTML report generation."""
        # Store some test data first
        test_result = TestResult(
            test_name="test_report_generation",
            status="passed",
            duration=1.0,
            test_category="infrastructure"
        )

        suite_result = TestSuiteResult(
            suite_name="report_test",
            total_tests=1,
            passed=1,
            failed=0,
            skipped=0,
            errors=0,
            total_duration=1.0,
            timestamp=datetime.now().isoformat(),
            platform_info=infra_manager.get_platform_info(),
            test_results=[test_result]
        )

        infra_manager.store_test_results(suite_result)

        # Generate HTML report
        report_path = infra_manager.generate_test_report("html")

        # Verify report was created
        assert report_path.exists()
        assert report_path.suffix == ".html"

        # Verify report content
        content = report_path.read_text()
        assert "Test Report" in content
        assert "test_report_generation" in content
        assert "PASSED" in content

    def test_json_report_generation(self, infra_manager):
        """Test JSON report generation."""
        # Store test data
        test_result = TestResult(
            test_name="test_json_report",
            status="passed",
            duration=0.5,
            test_category="infrastructure"
        )

        suite_result = TestSuiteResult(
            suite_name="json_test",
            total_tests=1,
            passed=1,
            failed=0,
            skipped=0,
            errors=0,
            total_duration=0.5,
            timestamp=datetime.now().isoformat(),
            platform_info=infra_manager.get_platform_info(),
            test_results=[test_result]
        )

        infra_manager.store_test_results(suite_result)

        # Generate JSON report
        report_path = infra_manager.generate_test_report("json")

        # Verify report
        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Verify JSON content
        with open(report_path) as f:
            report_data = json.load(f)

        assert "suite_name" in report_data
        assert "summary" in report_data
        assert "test_results" in report_data
        assert report_data["summary"]["total_tests"] == 1
        assert report_data["summary"]["passed"] == 1

    def test_performance_regression_detection(self, infra_manager):
        """Test performance regression detection."""
        # Store multiple test suite results with different durations
        base_time = time.time()

        for i, duration in enumerate([10.0, 10.5, 9.8, 10.2, 15.0]):  # Last one is regression
            test_result = TestResult(
                test_name=f"test_perf_{i}",
                status="passed",
                duration=1.0,
                test_category="performance"
            )

            suite_result = TestSuiteResult(
                suite_name=f"perf_test_{i}",
                total_tests=1,
                passed=1,
                failed=0,
                skipped=0,
                errors=0,
                total_duration=duration,
                timestamp=datetime.fromtimestamp(base_time + i * 3600).isoformat(),
                platform_info=infra_manager.get_platform_info(),
                test_results=[test_result]
            )

            infra_manager.store_test_results(suite_result)
            time.sleep(0.1)  # Small delay to ensure different timestamps

        # Check for regression
        regression_result = infra_manager.check_performance_regression(threshold=0.2)

        # Should detect regression in latest run
        assert "status" in regression_result
        assert "latest_duration" in regression_result
        assert "baseline_duration" in regression_result
        assert "regression_ratio" in regression_result

        # With our test data, should detect regression
        if regression_result["status"] != "insufficient_data":
            assert regression_result["regression_ratio"] > 0  # Should be positive (slower)

    def test_ci_summary_generation(self, infra_manager):
        """Test CI/CD summary generation."""
        # Store test results
        test_results = [
            TestResult(
                test_name="test_ci_1",
                status="passed",
                duration=1.0,
                test_category="ci"
            ),
            TestResult(
                test_name="test_ci_2",
                status="failed",
                duration=0.5,
                error_message="CI test failure",
                test_category="ci"
            )
        ]

        suite_result = TestSuiteResult(
            suite_name="ci_test",
            total_tests=2,
            passed=1,
            failed=1,
            skipped=0,
            errors=0,
            total_duration=1.5,
            timestamp=datetime.now().isoformat(),
            platform_info=infra_manager.get_platform_info(),
            test_results=test_results
        )

        infra_manager.store_test_results(suite_result)

        # Generate CI summary
        ci_summary = infra_manager.generate_ci_summary()

        # Verify summary structure
        required_fields = [
            "status", "total_tests", "passed", "failed",
            "success_rate", "duration", "platform"
        ]

        for field in required_fields:
            assert field in ci_summary

        # Verify values
        assert ci_summary["total_tests"] == 2
        assert ci_summary["passed"] == 1
        assert ci_summary["failed"] == 1
        assert ci_summary["status"] == "failed"  # Should be failed due to failures
        assert 0 <= ci_summary["success_rate"] <= 1

    def test_test_categorization(self, infra_manager):
        """Test test categorization and filtering."""
        # Test category extraction
        test_cases = [
            ("tests/functional/test_example.py::test_func", "functional"),
            ("tests/unit/test_util.py::test_method", "unit"),
            ("tests/integration/test_api.py::test_endpoint", "integration"),
            ("tests/other/test_misc.py::test_something", "other")
        ]

        for nodeid, expected_category in test_cases:
            category = infra_manager._extract_test_category(nodeid)
            assert category == expected_category

    def test_error_message_extraction(self, infra_manager):
        """Test error message extraction and truncation."""
        # Test with mock test data
        test_data = {
            "call": {
                "longrepr": "Very long error message " + "x" * 1000
            }
        }

        error_msg = infra_manager._extract_error_message(test_data)

        # Should extract and truncate
        assert error_msg is not None
        assert len(error_msg) <= 500
        assert "Very long error message" in error_msg

    def test_database_migration_safety(self, infra_manager):
        """Test that database operations are safe for migration."""
        # Test multiple initializations (should not fail)
        infra_manager.init_database()
        infra_manager.init_database()  # Should not raise error

        # Verify tables still exist and are functional
        with sqlite3.connect(infra_manager.database_path) as conn:
            tables = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
            """).fetchall()

            table_names = [table[0] for table in tables]
            assert "test_suites" in table_names
            assert "test_results" in table_names
            assert "performance_metrics" in table_names

            # Test inserting data still works
            conn.execute("""
                INSERT INTO test_suites (
                    suite_name, timestamp, total_tests, passed,
                    failed, skipped, errors, total_duration,
                    platform_system, platform_python_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "migration_test", datetime.now().isoformat(),
                1, 1, 0, 0, 0, 1.0, "test_system", "3.9.0"
            ))

            # Verify insertion worked
            count = conn.execute("SELECT COUNT(*) FROM test_suites").fetchone()[0]
            assert count > 0
