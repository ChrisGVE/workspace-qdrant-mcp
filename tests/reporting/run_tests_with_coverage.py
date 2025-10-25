#!/usr/bin/env python3
"""
Run tests with coverage tracking and generate comprehensive reports.

This script:
1. Runs Python tests with coverage.py
2. Runs Rust tests with tarpaulin (if available)
3. Parses coverage data
4. Aggregates test results
5. Checks coverage thresholds
6. Generates HTML/PDF reports

Usage:
    python tests/reporting/run_tests_with_coverage.py
    python tests/reporting/run_tests_with_coverage.py --threshold=strict
    python tests/reporting/run_tests_with_coverage.py --python-only
    python tests/reporting/run_tests_with_coverage.py --output=report.html
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .aggregator import TestResultAggregator
from .coverage_checker import (
    CoverageChecker,
    CoverageThresholds,
    check_coverage_thresholds,
)
from .models import TestRun, TestSource
from .parsers.coverage_py_parser import parse_coverage_xml
from .parsers.pytest_parser import PytestParser
from .parsers.tarpaulin_parser import parse_tarpaulin_json, parse_tarpaulin_lcov
from .report_generator import ReportGenerator
from .storage import TestResultStorage


class CoverageTestRunner:
    """Run tests with coverage tracking and reporting."""

    def __init__(
        self,
        project_root: Path | None = None,
        threshold_preset: str = "default",
    ):
        """
        Initialize coverage test runner.

        Args:
            project_root: Project root directory (auto-detected if None)
            threshold_preset: Coverage threshold preset ("default" or "strict")
        """
        self.project_root = project_root or Path.cwd()
        self.coverage_dir = self.project_root / "coverage_reports"
        self.coverage_dir.mkdir(parents=True, exist_ok=True)

        # Set up thresholds
        if threshold_preset == "strict":
            self.thresholds = CoverageThresholds.strict()
        else:
            self.thresholds = CoverageThresholds.default()

        # Set up storage and aggregator
        self.storage = TestResultStorage()
        self.aggregator = TestResultAggregator(self.storage)

    def run_python_tests_with_coverage(self) -> Path | None:
        """
        Run Python tests with coverage.py.

        Returns:
            Path to coverage.xml file or None if failed
        """
        print("Running Python tests with coverage...")

        # Run pytest with coverage
        coverage_xml = self.coverage_dir / "python_coverage.xml"
        coverage_html = self.coverage_dir / "python_htmlcov"

        cmd = [
            "pytest",
            "--cov=src",
            f"--cov-report=xml:{coverage_xml}",
            f"--cov-report=html:{coverage_html}",
            "--cov-report=term",
            "--tb=short",
            "-v",
        ]

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if coverage_xml.exists():
                print(f"Python coverage report generated: {coverage_xml}")
                return coverage_xml
            else:
                print("Warning: Python coverage XML not generated")
                return None

        except FileNotFoundError:
            print("Error: pytest not found. Install with: pip install pytest pytest-cov")
            return None
        except Exception as e:
            print(f"Error running Python tests: {e}")
            return None

    def run_rust_tests_with_coverage(self) -> Path | None:
        """
        Run Rust tests with tarpaulin.

        Returns:
            Path to coverage file or None if failed/not available
        """
        rust_engine = self.project_root / "src" / "rust" / "daemon"
        if not rust_engine.exists():
            print("Rust engine directory not found, skipping Rust coverage")
            return None

        print("Running Rust tests with tarpaulin...")

        # Try to run tarpaulin
        coverage_json = self.coverage_dir / "rust_coverage.json"
        coverage_lcov = self.coverage_dir / "rust_lcov.info"

        # Try JSON output first
        cmd = [
            "cargo",
            "tarpaulin",
            "--out",
            "Json",
            "--output-dir",
            str(self.coverage_dir),
        ]

        try:
            result = subprocess.run(
                cmd, cwd=rust_engine, capture_output=True, text=True, timeout=300
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            # Check for JSON output
            if coverage_json.exists():
                print(f"Rust coverage report generated: {coverage_json}")
                return coverage_json

            # Fallback to LCOV
            cmd[3] = "Lcov"
            result = subprocess.run(
                cmd, cwd=rust_engine, capture_output=True, text=True, timeout=300
            )

            if coverage_lcov.exists():
                print(f"Rust coverage report generated: {coverage_lcov}")
                return coverage_lcov

            print("Warning: Rust coverage not generated")
            return None

        except FileNotFoundError:
            print(
                "Warning: cargo-tarpaulin not found. "
                "Install with: cargo install cargo-tarpaulin"
            )
            return None
        except subprocess.TimeoutExpired:
            print("Warning: Rust coverage generation timed out")
            return None
        except Exception as e:
            print(f"Error running Rust tests: {e}")
            return None

    def parse_coverage_data(
        self, python_coverage: Path | None, rust_coverage: Path | None
    ):
        """
        Parse coverage data from Python and/or Rust.

        Args:
            python_coverage: Path to Python coverage.xml or None
            rust_coverage: Path to Rust coverage file or None

        Returns:
            CoverageMetrics or None
        """
        # For now, only use Python coverage (most comprehensive)
        # In the future, we could merge Python and Rust coverage
        if python_coverage:
            try:
                print(f"Parsing Python coverage from {python_coverage}")
                coverage = parse_coverage_xml(
                    python_coverage, source_root=self.project_root
                )
                return coverage
            except Exception as e:
                print(f"Error parsing Python coverage: {e}")
                return None

        if rust_coverage:
            try:
                print(f"Parsing Rust coverage from {rust_coverage}")
                if rust_coverage.suffix == ".json":
                    coverage = parse_tarpaulin_json(
                        rust_coverage, source_root=self.project_root
                    )
                else:
                    coverage = parse_tarpaulin_lcov(
                        rust_coverage, source_root=self.project_root
                    )
                return coverage
            except Exception as e:
                print(f"Error parsing Rust coverage: {e}")
                return None

        return None

    def check_thresholds(self, coverage):
        """
        Check coverage against thresholds.

        Args:
            coverage: CoverageMetrics

        Returns:
            CoverageCheckResult
        """
        print("\nChecking coverage thresholds...")
        result = check_coverage_thresholds(coverage, self.thresholds)

        print(f"\nStatus: {result.message}")

        if result.violations:
            print(f"\nFAILURES ({len(result.violations)}):")
            for violation in result.violations:
                print(f"  - {violation.message}")

        if result.warnings:
            print(f"\nWARNINGS ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning.message}")

        return result

    def generate_report(self, test_run, output_path: Path | None = None):
        """
        Generate HTML report.

        Args:
            test_run: TestRun with coverage data
            output_path: Path to save report (auto-generated if None)

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.coverage_dir / f"test_report_{timestamp}.html"

        print(f"\nGenerating report: {output_path}")

        generator = ReportGenerator(self.storage)
        generator.generate_html_report(
            run_id=test_run.run_id,
            output_path=output_path,
            include_charts=True,
            include_trends=False,  # Not enough historical data yet
        )

        print(f"Report generated: {output_path}")
        return output_path

    def run(
        self,
        python_only: bool = False,
        rust_only: bool = False,
        output_path: Path | None = None,
        skip_threshold_check: bool = False,
    ) -> int:
        """
        Run complete test suite with coverage.

        Args:
            python_only: Only run Python tests
            rust_only: Only run Rust tests
            output_path: Path for HTML report
            skip_threshold_check: Skip threshold checking

        Returns:
            Exit code (0 = success, 1 = threshold failures)
        """
        print("=" * 80)
        print("Running tests with coverage tracking")
        print("=" * 80)

        # Run tests
        python_coverage = None
        rust_coverage = None

        if not rust_only:
            python_coverage = self.run_python_tests_with_coverage()

        if not python_only:
            rust_coverage = self.run_rust_tests_with_coverage()

        # Parse coverage
        coverage = self.parse_coverage_data(python_coverage, rust_coverage)

        if coverage is None:
            print("\nError: No coverage data available")
            return 1

        print(
            f"\nOverall Coverage: {coverage.line_coverage_percent:.2f}% "
            f"({coverage.lines_covered}/{coverage.lines_total} lines)"
        )

        # Create test run (simplified - in real usage would parse pytest JSON output)
        test_run = TestRun.create(
            source=TestSource.PYTEST,
            metadata={
                "coverage_enabled": True,
                "python_coverage": str(python_coverage) if python_coverage else None,
                "rust_coverage": str(rust_coverage) if rust_coverage else None,
            },
        )
        test_run.coverage = coverage

        # Save to storage
        self.storage.save_test_run(test_run)
        print(f"Test run saved: {test_run.run_id}")

        # Check thresholds
        threshold_result = None
        if not skip_threshold_check:
            threshold_result = self.check_thresholds(coverage)

        # Generate report
        report_path = self.generate_report(test_run, output_path)

        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  Coverage: {coverage.line_coverage_percent:.2f}%")
        print(f"  Report: {report_path}")
        if threshold_result:
            print(f"  Threshold Check: {threshold_result.message}")
        print("=" * 80)

        # Return exit code based on threshold check
        if threshold_result and not threshold_result.passed:
            return 1
        return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests with coverage tracking and reporting"
    )
    parser.add_argument(
        "--threshold",
        choices=["default", "strict"],
        default="default",
        help="Coverage threshold preset",
    )
    parser.add_argument(
        "--python-only", action="store_true", help="Only run Python tests"
    )
    parser.add_argument(
        "--rust-only", action="store_true", help="Only run Rust tests"
    )
    parser.add_argument(
        "--output", type=Path, help="Output path for HTML report"
    )
    parser.add_argument(
        "--skip-threshold-check",
        action="store_true",
        help="Skip threshold checking (always exit 0)",
    )

    args = parser.parse_args()

    runner = CoverageTestRunner(threshold_preset=args.threshold)
    exit_code = runner.run(
        python_only=args.python_only,
        rust_only=args.rust_only,
        output_path=args.output,
        skip_threshold_check=args.skip_threshold_check,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
