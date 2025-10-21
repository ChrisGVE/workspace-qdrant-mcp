"""
SQLite storage backend for test result aggregation.

Provides persistent storage for test runs, suites, cases, and results with
query capabilities. Supports incremental result addition for CI/CD pipelines.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    CoverageMetrics,
    FailureAnalysisReport,
    FileCoverage,
    PerformanceMetrics,
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)


class TestResultStorage:
    """SQLite storage backend for test results."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize storage backend.

        Args:
            db_path: Path to SQLite database file. Defaults to tests/reporting/test_results.db
        """
        if db_path is None:
            # Default to tests/reporting/test_results.db
            self.db_path = Path(__file__).parent / "test_results.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")

            # Test runs table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    environment TEXT,  -- JSON
                    metadata TEXT,  -- JSON
                    total_tests INTEGER DEFAULT 0,
                    passed_tests INTEGER DEFAULT 0,
                    failed_tests INTEGER DEFAULT 0,
                    skipped_tests INTEGER DEFAULT 0,
                    error_tests INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Coverage metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coverage_metrics (
                    run_id TEXT PRIMARY KEY,
                    line_coverage_percent REAL NOT NULL,
                    lines_covered INTEGER NOT NULL,
                    lines_total INTEGER NOT NULL,
                    function_coverage_percent REAL,
                    functions_covered INTEGER,
                    functions_total INTEGER,
                    branch_coverage_percent REAL,
                    branches_covered INTEGER,
                    branches_total INTEGER,
                    coverage_tool TEXT,
                    custom_metrics TEXT,  -- JSON
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id) ON DELETE CASCADE
                )
                """
            )

            # File coverage table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    lines_covered INTEGER NOT NULL,
                    lines_total INTEGER NOT NULL,
                    line_coverage_percent REAL NOT NULL,
                    uncovered_lines TEXT,  -- JSON array
                    functions_covered INTEGER,
                    functions_total INTEGER,
                    branches_covered INTEGER,
                    branches_total INTEGER,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id) ON DELETE CASCADE
                )
                """
            )

            # Test suites table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_suites (
                    suite_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id) ON DELETE CASCADE
                )
                """
            )

            # Test cases table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_cases (
                    case_id TEXT PRIMARY KEY,
                    suite_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    file_path TEXT,
                    line_number INTEGER,
                    test_type TEXT,
                    markers TEXT,  -- JSON array
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (suite_id) REFERENCES test_suites(suite_id) ON DELETE CASCADE
                )
                """
            )

            # Test results table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    test_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    error_message TEXT,
                    error_traceback TEXT,
                    performance_metrics TEXT,  -- JSON
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (case_id) REFERENCES test_cases(case_id) ON DELETE CASCADE
                )
                """
            )

            # Indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON test_runs(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_source ON test_runs(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_suites_run ON test_suites(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_suites_type ON test_suites(test_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cases_suite ON test_cases(suite_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cases_name ON test_cases(name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_case ON test_results(case_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_status ON test_results(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_timestamp ON test_results(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_coverage_run ON file_coverage(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_coverage_path ON file_coverage(file_path)"
            )

            # Failure analysis reports table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_analysis_reports (
                    report_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    analyzed_runs TEXT NOT NULL,  -- JSON array of run IDs
                    time_window_start TEXT,
                    time_window_end TEXT,
                    flaky_tests TEXT NOT NULL,  -- JSON array of FlakinessMetrics
                    total_flaky_tests INTEGER DEFAULT 0,
                    failure_patterns TEXT NOT NULL,  -- JSON array of FailurePattern
                    total_failure_patterns INTEGER DEFAULT 0,
                    category_distribution TEXT,  -- JSON object
                    failure_trend TEXT,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failure_reports_timestamp ON failure_analysis_reports(timestamp)"
            )

            conn.commit()

    def save_test_run(self, test_run: TestRun) -> None:
        """
        Save a complete test run with all suites, cases, and results.

        Args:
            test_run: TestRun to save
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # Save test run
            conn.execute(
                """
                INSERT OR REPLACE INTO test_runs
                (run_id, timestamp, source, environment, metadata,
                 total_tests, passed_tests, failed_tests, skipped_tests, error_tests, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test_run.run_id,
                    test_run.timestamp.isoformat(),
                    test_run.source.value,
                    json.dumps(test_run.environment),
                    json.dumps(test_run.metadata),
                    test_run.total_tests,
                    test_run.passed_tests,
                    test_run.failed_tests,
                    test_run.skipped_tests,
                    test_run.error_tests,
                    test_run.success_rate,
                ),
            )

            # Save suites
            for suite in test_run.suites:
                self._save_suite(conn, test_run.run_id, suite)

            # Save coverage if present
            if test_run.coverage:
                self._save_coverage(conn, test_run.run_id, test_run.coverage)

            conn.commit()

    def _save_suite(
        self, conn: sqlite3.Connection, run_id: str, suite: TestSuite
    ) -> None:
        """Save a test suite and its test cases."""
        conn.execute(
            """
            INSERT OR REPLACE INTO test_suites
            (suite_id, run_id, name, test_type, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                suite.suite_id,
                run_id,
                suite.name,
                suite.test_type.value,
                json.dumps(suite.metadata),
            ),
        )

        # Save test cases
        for case in suite.test_cases:
            self._save_case(conn, suite.suite_id, case)

    def _save_case(
        self, conn: sqlite3.Connection, suite_id: str, case: TestCase
    ) -> None:
        """Save a test case and its results."""
        conn.execute(
            """
            INSERT OR REPLACE INTO test_cases
            (case_id, suite_id, name, file_path, line_number, test_type, markers, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case.case_id,
                suite_id,
                case.name,
                case.file_path,
                case.line_number,
                case.test_type.value if case.test_type else None,
                json.dumps(case.markers),
                json.dumps(case.metadata),
            ),
        )

        # Save results
        for result in case.results:
            self._save_result(conn, case.case_id, result)

    def _save_result(
        self, conn: sqlite3.Connection, case_id: str, result: TestResult
    ) -> None:
        """Save a test result."""
        conn.execute(
            """
            INSERT OR REPLACE INTO test_results
            (test_id, case_id, name, status, duration_ms, timestamp,
             error_message, error_traceback, performance_metrics, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.test_id,
                case_id,
                result.name,
                result.status.value,
                result.duration_ms,
                result.timestamp.isoformat(),
                result.error_message,
                result.error_traceback,
                json.dumps(result.performance.to_dict())
                if result.performance
                else None,
                json.dumps(result.metadata),
            ),
        )

    def _save_coverage(
        self, conn: sqlite3.Connection, run_id: str, coverage: CoverageMetrics
    ) -> None:
        """Save coverage metrics."""
        # Save overall coverage metrics
        conn.execute(
            """
            INSERT OR REPLACE INTO coverage_metrics
            (run_id, line_coverage_percent, lines_covered, lines_total,
             function_coverage_percent, functions_covered, functions_total,
             branch_coverage_percent, branches_covered, branches_total,
             coverage_tool, custom_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                coverage.line_coverage_percent,
                coverage.lines_covered,
                coverage.lines_total,
                coverage.function_coverage_percent,
                coverage.functions_covered,
                coverage.functions_total,
                coverage.branch_coverage_percent,
                coverage.branches_covered,
                coverage.branches_total,
                coverage.coverage_tool,
                json.dumps(coverage.custom_metrics),
            ),
        )

        # Save per-file coverage
        for file_cov in coverage.file_coverage:
            conn.execute(
                """
                INSERT INTO file_coverage
                (run_id, file_path, lines_covered, lines_total, line_coverage_percent,
                 uncovered_lines, functions_covered, functions_total,
                 branches_covered, branches_total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    file_cov.file_path,
                    file_cov.lines_covered,
                    file_cov.lines_total,
                    file_cov.line_coverage_percent,
                    json.dumps(file_cov.uncovered_lines),
                    file_cov.functions_covered,
                    file_cov.functions_total,
                    file_cov.branches_covered,
                    file_cov.branches_total,
                ),
            )

    def get_test_run(self, run_id: str) -> Optional[TestRun]:
        """
        Retrieve a complete test run by ID.

        Args:
            run_id: Test run ID

        Returns:
            TestRun object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get test run
            row = conn.execute(
                "SELECT * FROM test_runs WHERE run_id = ?", (run_id,)
            ).fetchone()

            if not row:
                return None

            # Construct TestRun
            test_run = TestRun(
                run_id=row["run_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                source=TestSource(row["source"]),
                environment=json.loads(row["environment"]) if row["environment"] else {},
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )

            # Load suites
            test_run.suites = self._load_suites(conn, run_id)

            # Load coverage
            test_run.coverage = self._load_coverage(conn, run_id)

            return test_run

    def _load_suites(
        self, conn: sqlite3.Connection, run_id: str
    ) -> List[TestSuite]:
        """Load all suites for a test run."""
        rows = conn.execute(
            "SELECT * FROM test_suites WHERE run_id = ?", (run_id,)
        ).fetchall()

        suites = []
        for row in rows:
            suite = TestSuite(
                suite_id=row["suite_id"],
                name=row["name"],
                test_type=TestType(row["test_type"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            suite.test_cases = self._load_cases(conn, row["suite_id"])
            suites.append(suite)

        return suites

    def _load_cases(self, conn: sqlite3.Connection, suite_id: str) -> List[TestCase]:
        """Load all test cases for a suite."""
        rows = conn.execute(
            "SELECT * FROM test_cases WHERE suite_id = ?", (suite_id,)
        ).fetchall()

        cases = []
        for row in rows:
            case = TestCase(
                case_id=row["case_id"],
                name=row["name"],
                file_path=row["file_path"],
                line_number=row["line_number"],
                test_type=TestType(row["test_type"]) if row["test_type"] else None,
                markers=json.loads(row["markers"]) if row["markers"] else [],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            case.results = self._load_results(conn, row["case_id"])
            cases.append(case)

        return cases

    def _load_results(
        self, conn: sqlite3.Connection, case_id: str
    ) -> List[TestResult]:
        """Load all results for a test case."""
        rows = conn.execute(
            "SELECT * FROM test_results WHERE case_id = ? ORDER BY timestamp",
            (case_id,),
        ).fetchall()

        results = []
        for row in rows:
            result = TestResult(
                test_id=row["test_id"],
                name=row["name"],
                status=TestStatus(row["status"]),
                duration_ms=row["duration_ms"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                error_message=row["error_message"],
                error_traceback=row["error_traceback"],
                performance=PerformanceMetrics.from_dict(
                    json.loads(row["performance_metrics"])
                )
                if row["performance_metrics"]
                else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            results.append(result)

        return results

    def _load_coverage(
        self, conn: sqlite3.Connection, run_id: str
    ) -> Optional[CoverageMetrics]:
        """Load coverage metrics for a test run."""
        row = conn.execute(
            "SELECT * FROM coverage_metrics WHERE run_id = ?", (run_id,)
        ).fetchone()

        if not row:
            return None

        # Load file coverage
        file_rows = conn.execute(
            "SELECT * FROM file_coverage WHERE run_id = ? ORDER BY file_path",
            (run_id,),
        ).fetchall()

        file_coverage = []
        for file_row in file_rows:
            file_coverage.append(
                FileCoverage(
                    file_path=file_row["file_path"],
                    lines_covered=file_row["lines_covered"],
                    lines_total=file_row["lines_total"],
                    line_coverage_percent=file_row["line_coverage_percent"],
                    uncovered_lines=json.loads(file_row["uncovered_lines"])
                    if file_row["uncovered_lines"]
                    else [],
                    functions_covered=file_row["functions_covered"],
                    functions_total=file_row["functions_total"],
                    branches_covered=file_row["branches_covered"],
                    branches_total=file_row["branches_total"],
                )
            )

        return CoverageMetrics(
            line_coverage_percent=row["line_coverage_percent"],
            lines_covered=row["lines_covered"],
            lines_total=row["lines_total"],
            function_coverage_percent=row["function_coverage_percent"],
            functions_covered=row["functions_covered"],
            functions_total=row["functions_total"],
            branch_coverage_percent=row["branch_coverage_percent"],
            branches_covered=row["branches_covered"],
            branches_total=row["branches_total"],
            coverage_tool=row["coverage_tool"],
            custom_metrics=json.loads(row["custom_metrics"])
            if row["custom_metrics"]
            else {},
            file_coverage=file_coverage,
        )

    def list_test_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        source: Optional[TestSource] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List test runs with optional filtering.

        Args:
            limit: Maximum number of runs to return
            offset: Offset for pagination
            source: Filter by test source
            start_date: Filter runs after this date
            end_date: Filter runs before this date

        Returns:
            List of test run summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM test_runs WHERE 1=1"
            params = []

            if source:
                query += " AND source = ?"
                params.append(source.value)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()

            return [dict(row) for row in rows]

    def delete_test_run(self, run_id: str) -> bool:
        """
        Delete a test run and all associated data.

        Args:
            run_id: Test run ID to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM test_runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics from the database.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            stats = {}

            # Total counts
            stats["total_runs"] = conn.execute(
                "SELECT COUNT(*) as count FROM test_runs"
            ).fetchone()["count"]
            stats["total_suites"] = conn.execute(
                "SELECT COUNT(*) as count FROM test_suites"
            ).fetchone()["count"]
            stats["total_cases"] = conn.execute(
                "SELECT COUNT(*) as count FROM test_cases"
            ).fetchone()["count"]
            stats["total_results"] = conn.execute(
                "SELECT COUNT(*) as count FROM test_results"
            ).fetchone()["count"]

            # Results by status
            status_counts = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM test_results
                GROUP BY status
                """
            ).fetchall()
            stats["results_by_status"] = {row["status"]: row["count"] for row in status_counts}

            # Runs by source
            source_counts = conn.execute(
                """
                SELECT source, COUNT(*) as count
                FROM test_runs
                GROUP BY source
                """
            ).fetchall()
            stats["runs_by_source"] = {row["source"]: row["count"] for row in source_counts}

            # Average success rate
            avg_success = conn.execute(
                "SELECT AVG(success_rate) as avg_rate FROM test_runs"
            ).fetchone()
            stats["average_success_rate"] = avg_success["avg_rate"] or 0.0

            return stats

    def save_failure_analysis_report(self, report: FailureAnalysisReport) -> None:
        """
        Save a failure analysis report.

        Args:
            report: FailureAnalysisReport to save
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO failure_analysis_reports
                (report_id, timestamp, analyzed_runs, time_window_start, time_window_end,
                 flaky_tests, total_flaky_tests, failure_patterns, total_failure_patterns,
                 category_distribution, failure_trend, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.timestamp.isoformat(),
                    json.dumps(report.analyzed_runs),
                    report.time_window_start.isoformat() if report.time_window_start else None,
                    report.time_window_end.isoformat() if report.time_window_end else None,
                    json.dumps([f.to_dict() for f in report.flaky_tests]),
                    report.total_flaky_tests,
                    json.dumps([p.to_dict() for p in report.failure_patterns]),
                    report.total_failure_patterns,
                    json.dumps(report.category_distribution),
                    report.failure_trend,
                    json.dumps(report.metadata),
                ),
            )
            conn.commit()

    def get_failure_analysis_report(self, report_id: str) -> Optional[FailureAnalysisReport]:
        """
        Retrieve a failure analysis report by ID.

        Args:
            report_id: Report ID

        Returns:
            FailureAnalysisReport or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT * FROM failure_analysis_reports WHERE report_id = ?",
                (report_id,)
            ).fetchone()

            if not row:
                return None

            return FailureAnalysisReport.from_dict({
                "report_id": row["report_id"],
                "timestamp": row["timestamp"],
                "analyzed_runs": json.loads(row["analyzed_runs"]),
                "time_window_start": row["time_window_start"],
                "time_window_end": row["time_window_end"],
                "flaky_tests": json.loads(row["flaky_tests"]),
                "total_flaky_tests": row["total_flaky_tests"],
                "failure_patterns": json.loads(row["failure_patterns"]),
                "total_failure_patterns": row["total_failure_patterns"],
                "category_distribution": json.loads(row["category_distribution"]) if row["category_distribution"] else {},
                "failure_trend": row["failure_trend"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })

    def list_failure_analysis_reports(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List failure analysis reports.

        Args:
            limit: Maximum number of reports to return
            offset: Offset for pagination

        Returns:
            List of report summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT report_id, timestamp, total_flaky_tests, total_failure_patterns,
                       failure_trend, time_window_start, time_window_end
                FROM failure_analysis_reports
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """

            rows = conn.execute(query, (limit, offset)).fetchall()

            return [dict(row) for row in rows]
