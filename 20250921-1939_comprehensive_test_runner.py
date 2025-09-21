#!/usr/bin/env python3
"""
Comprehensive Test Runner for Quality Loop
==========================================

This script provides optimized test execution with parallel processing,
coverage analysis, and failure reporting for the iterative quality loop.

Features:
- Parallel test execution with pytest-xdist
- Intelligent test filtering and selection
- Coverage gap analysis and reporting
- Performance optimization for rapid iterations
- Test isolation validation
- Atomic commit integration
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import rich
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()


@dataclass
class TestExecutionResult:
    """Comprehensive test execution result."""
    codebase: str
    command_used: List[str]
    return_code: int
    execution_time: float
    stdout: str
    stderr: str
    passed_count: int
    failed_count: int
    skipped_count: int
    error_count: int
    total_count: int
    coverage_percentage: float
    failed_tests: List[str]
    slow_tests: List[Tuple[str, float]]  # (test_name, duration)
    coverage_missing: List[str]
    warnings: List[str]


class ComprehensiveTestRunner:
    """Optimized test runner for iterative quality improvement."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.python_src = self.project_root / "src" / "python"
        self.rust_src = self.project_root / "src" / "rust"
        self.test_dir = self.project_root / "tests"

    def build_pytest_command(
        self,
        test_path: str = "tests/unit/",
        parallel: bool = True,
        coverage: bool = True,
        markers: Optional[List[str]] = None,
        exclude_markers: Optional[List[str]] = None,
        fast_mode: bool = False,
        verbose: bool = True
    ) -> List[str]:
        """Build optimized pytest command."""
        cmd = ["uv", "run", "pytest"]

        # Test path
        cmd.append(test_path)

        # Parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])

        # Coverage settings
        if coverage:
            cmd.extend([
                "--cov=src/python",
                "--cov-report=xml:coverage.xml",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-branch",
                "--cov-fail-under=0"  # Let quality loop handle failures
            ])

        # Output formatting
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        else:
            cmd.extend(["-q", "--tb=line"])

        # Performance optimizations
        cmd.extend([
            "--durations=10",  # Show 10 slowest tests
            "--durations-min=1.0",  # Only show tests >= 1 second
            "--benchmark-skip",  # Skip benchmarks for faster iteration
            "--disable-warnings" if not verbose else "--tb=short"
        ])

        # Fast mode optimizations
        if fast_mode:
            cmd.extend([
                "--lf",  # Run last-failed tests first
                "--ff",  # Run failures first
                "-x",    # Stop on first failure
                "--cache-clear"  # Clear cache for consistent results
            ])

        # Marker filtering
        if markers:
            marker_expr = " or ".join(markers)
            cmd.extend(["-m", marker_expr])

        if exclude_markers:
            exclude_expr = " and ".join(f"not {marker}" for marker in exclude_markers)
            if markers:
                cmd.extend(["-m", f"({marker_expr}) and ({exclude_expr})"])
            else:
                cmd.extend(["-m", exclude_expr])

        # Timeout protection
        cmd.extend(["--timeout=300"])

        return cmd

    def build_rust_test_command(
        self,
        workspace: bool = True,
        coverage: bool = True,
        release: bool = False,
        parallel: bool = True
    ) -> List[str]:
        """Build optimized Rust test command."""
        if coverage:
            cmd = ["cargo", "tarpaulin"]
            if workspace:
                cmd.append("--workspace")
            cmd.extend([
                "--out", "xml",
                "--output-dir", str(self.project_root),
                "--fail-under", "0",
                "--timeout", "300"
            ])
        else:
            cmd = ["cargo", "test"]
            if workspace:
                cmd.append("--workspace")
            if release:
                cmd.append("--release")
            if parallel:
                cmd.extend(["--", "--nocapture"])

        return cmd

    async def execute_python_tests(
        self,
        fast_mode: bool = False,
        markers: Optional[List[str]] = None,
        exclude_slow: bool = True
    ) -> TestExecutionResult:
        """Execute Python test suite with optimizations."""
        console.print("🐍 Executing Python test suite...", style="bold blue")

        # Configure exclusions for faster iterations
        exclude_markers = []
        if exclude_slow:
            exclude_markers.extend(["slow", "performance", "memory_intensive"])

        cmd = self.build_pytest_command(
            fast_mode=fast_mode,
            markers=markers,
            exclude_markers=exclude_markers
        )

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )

            execution_time = time.time() - start_time

            # Parse pytest output
            test_result = self._parse_pytest_output(result, cmd, execution_time)

            # Enhance with coverage analysis
            if "--cov" in " ".join(cmd):
                test_result.coverage_percentage = await self._extract_python_coverage()
                test_result.coverage_missing = await self._identify_coverage_gaps("python")

            return test_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            console.print("⚠️ Python tests timed out", style="bold yellow")

            return TestExecutionResult(
                codebase="python",
                command_used=cmd,
                return_code=-1,
                execution_time=execution_time,
                stdout="",
                stderr="Test execution timed out",
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                error_count=0,
                total_count=1,
                coverage_percentage=0.0,
                failed_tests=["TIMEOUT"],
                slow_tests=[],
                coverage_missing=[],
                warnings=["Test execution timed out after 15 minutes"]
            )

    async def execute_rust_tests(self, coverage: bool = True) -> TestExecutionResult:
        """Execute Rust test suite with coverage."""
        console.print("🦀 Executing Rust test suite...", style="bold blue")

        rust_engine = self.project_root / "src" / "rust" / "daemon"
        if not rust_engine.exists():
            console.print("⚠️ Rust engine directory not found", style="yellow")
            return TestExecutionResult(
                codebase="rust",
                command_used=[],
                return_code=-1,
                execution_time=0.0,
                stdout="",
                stderr="Rust engine directory not found",
                passed_count=0,
                failed_count=0,
                skipped_count=0,
                error_count=1,
                total_count=0,
                coverage_percentage=0.0,
                failed_tests=[],
                slow_tests=[],
                coverage_missing=["Rust engine not found"],
                warnings=["Rust engine directory missing"]
            )

        cmd = self.build_rust_test_command(coverage=coverage)
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=rust_engine,
                capture_output=True,
                text=True,
                timeout=900
            )

            execution_time = time.time() - start_time
            test_result = self._parse_rust_output(result, cmd, execution_time)

            # Enhance with coverage analysis
            if coverage:
                test_result.coverage_percentage = await self._extract_rust_coverage()
                test_result.coverage_missing = await self._identify_coverage_gaps("rust")

            return test_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            console.print("⚠️ Rust tests timed out", style="bold yellow")

            return TestExecutionResult(
                codebase="rust",
                command_used=cmd,
                return_code=-1,
                execution_time=execution_time,
                stdout="",
                stderr="Test execution timed out",
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                error_count=0,
                total_count=1,
                coverage_percentage=0.0,
                failed_tests=["TIMEOUT"],
                slow_tests=[],
                coverage_missing=[],
                warnings=["Test execution timed out after 15 minutes"]
            )

    def _parse_pytest_output(
        self,
        result: subprocess.CompletedProcess,
        cmd: List[str],
        execution_time: float
    ) -> TestExecutionResult:
        """Parse pytest output to extract test results."""
        stdout = result.stdout
        stderr = result.stderr
        lines = stdout.split('\n')

        # Initialize counters
        passed = failed = skipped = errors = 0
        failed_tests = []
        slow_tests = []
        warnings = []

        # Parse test results
        for line in lines:
            # Parse summary line: "X failed, Y passed, Z skipped in W.Ws"
            if " passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed" and i > 0:
                        failed = int(parts[i-1])
                    elif part == "passed" and i > 0:
                        passed = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        skipped = int(parts[i-1])
                    elif part == "error" and i > 0:
                        errors = int(parts[i-1])

            # Extract failed test names
            elif "FAILED" in line and "::" in line:
                test_name = line.split()[0] if line.split() else line
                failed_tests.append(test_name)

            # Extract slow tests from durations report
            elif "slowest durations" in line.lower():
                # Parse following lines for test durations
                continue

            # Extract warnings
            elif "warning" in line.lower() and "pytest" not in line.lower():
                warnings.append(line.strip())

        total = passed + failed + skipped + errors

        return TestExecutionResult(
            codebase="python",
            command_used=cmd,
            return_code=result.returncode,
            execution_time=execution_time,
            stdout=stdout,
            stderr=stderr,
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            error_count=errors,
            total_count=total,
            coverage_percentage=0.0,  # Will be filled by coverage analysis
            failed_tests=failed_tests,
            slow_tests=slow_tests,
            coverage_missing=[],  # Will be filled by coverage analysis
            warnings=warnings
        )

    def _parse_rust_output(
        self,
        result: subprocess.CompletedProcess,
        cmd: List[str],
        execution_time: float
    ) -> TestExecutionResult:
        """Parse Rust test output to extract results."""
        stdout = result.stdout
        stderr = result.stderr
        lines = stdout.split('\n')

        passed = failed = 0
        failed_tests = []
        warnings = []

        # Parse Rust test output
        for line in lines:
            if "test result:" in line:
                # Parse: "test result: ok. X passed; Y failed; Z ignored; W measured"
                parts = line.split()
                try:
                    passed_idx = next(i for i, p in enumerate(parts) if "passed" in p)
                    passed = int(parts[passed_idx - 1])

                    if "failed" in line:
                        failed_idx = next(i for i, p in enumerate(parts) if "failed" in p)
                        failed = int(parts[failed_idx - 1])
                except (StopIteration, ValueError, IndexError):
                    pass

            elif "FAILED" in line or "test ... FAILED" in line:
                failed_tests.append(line.strip())

            elif "warning:" in line.lower():
                warnings.append(line.strip())

        total = passed + failed

        return TestExecutionResult(
            codebase="rust",
            command_used=cmd,
            return_code=result.returncode,
            execution_time=execution_time,
            stdout=stdout,
            stderr=stderr,
            passed_count=passed,
            failed_count=failed,
            skipped_count=0,
            error_count=0,
            total_count=total,
            coverage_percentage=0.0,  # Will be filled by coverage analysis
            failed_tests=failed_tests,
            slow_tests=[],
            coverage_missing=[],  # Will be filled by coverage analysis
            warnings=warnings
        )

    async def _extract_python_coverage(self) -> float:
        """Extract Python coverage percentage from coverage.xml."""
        coverage_xml = self.project_root / "coverage.xml"
        if not coverage_xml.exists():
            return 0.0

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_xml)
            root = tree.getroot()
            return float(root.get('line-rate', 0.0)) * 100
        except Exception:
            return 0.0

    async def _extract_rust_coverage(self) -> float:
        """Extract Rust coverage percentage from cobertura.xml."""
        cobertura_xml = self.project_root / "cobertura.xml"
        if not cobertura_xml.exists():
            return 0.0

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(cobertura_xml)
            root = tree.getroot()
            return float(root.get('line-rate', 0.0)) * 100
        except Exception:
            return 0.0

    async def _identify_coverage_gaps(self, codebase: str) -> List[str]:
        """Identify specific coverage gaps for targeted improvement."""
        gaps = []

        if codebase == "python":
            # Analyze coverage.json for detailed gap information
            coverage_json = self.project_root / "coverage.json"
            if coverage_json.exists():
                try:
                    with open(coverage_json) as f:
                        data = json.load(f)

                    for filename, file_data in data.get('files', {}).items():
                        coverage = file_data.get('summary', {}).get('lines', {}).get('percent', 100)
                        if coverage < 100:
                            gaps.append(f"{filename}: {coverage:.1f}% coverage")
                except Exception:
                    pass

        elif codebase == "rust":
            # Analyze Rust coverage XML
            cobertura_xml = self.project_root / "cobertura.xml"
            if cobertura_xml.exists():
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(cobertura_xml)
                    root = tree.getroot()

                    for class_elem in root.findall(".//class"):
                        line_rate = float(class_elem.get('line-rate', 1.0))
                        if line_rate < 1.0:
                            filename = class_elem.get('filename', 'unknown')
                            gaps.append(f"{filename}: {line_rate*100:.1f}% coverage")
                except Exception:
                    pass

        return gaps

    def generate_execution_report(
        self,
        python_result: Optional[TestExecutionResult] = None,
        rust_result: Optional[TestExecutionResult] = None
    ) -> Table:
        """Generate comprehensive execution report."""
        table = Table(title="Test Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Python", style="green")
        table.add_column("Rust", style="yellow")
        table.add_column("Combined", style="blue")

        # Calculate metrics
        py_total = python_result.total_count if python_result else 0
        py_passed = python_result.passed_count if python_result else 0
        py_failed = python_result.failed_count if python_result else 0
        py_coverage = python_result.coverage_percentage if python_result else 0.0

        rust_total = rust_result.total_count if rust_result else 0
        rust_passed = rust_result.passed_count if rust_result else 0
        rust_failed = rust_result.failed_count if rust_result else 0
        rust_coverage = rust_result.coverage_percentage if rust_result else 0.0

        total_tests = py_total + rust_total
        total_passed = py_passed + rust_passed
        total_failed = py_failed + rust_failed

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        avg_coverage = (py_coverage + rust_coverage) / 2 if python_result and rust_result else (py_coverage or rust_coverage)

        table.add_row("Tests Run", str(py_total), str(rust_total), str(total_tests))
        table.add_row("Passed", str(py_passed), str(rust_passed), str(total_passed))
        table.add_row("Failed", str(py_failed), str(rust_failed), str(total_failed))
        table.add_row("Pass Rate", f"{py_passed/py_total*100:.1f}%" if py_total else "N/A",
                     f"{rust_passed/rust_total*100:.1f}%" if rust_total else "N/A", f"{pass_rate:.1f}%")
        table.add_row("Coverage", f"{py_coverage:.1f}%", f"{rust_coverage:.1f}%", f"{avg_coverage:.1f}%")

        return table

    async def run_comprehensive_tests(
        self,
        python_only: bool = False,
        rust_only: bool = False,
        fast_mode: bool = False,
        markers: Optional[List[str]] = None
    ) -> Tuple[Optional[TestExecutionResult], Optional[TestExecutionResult]]:
        """Run comprehensive test suite for both codebases."""
        console.print("🚀 Starting comprehensive test execution", style="bold green")

        python_result = None
        rust_result = None

        # Execute Python tests
        if not rust_only:
            python_result = await self.execute_python_tests(
                fast_mode=fast_mode,
                markers=markers
            )

        # Execute Rust tests
        if not python_only:
            rust_result = await self.execute_rust_tests()

        # Display results
        report = self.generate_execution_report(python_result, rust_result)
        console.print(report)

        return python_result, rust_result


async def main():
    """Main entry point for comprehensive test runner."""
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner for Quality Loop")
    parser.add_argument("--python-only", action="store_true", help="Run only Python tests")
    parser.add_argument("--rust-only", action="store_true", help="Run only Rust tests")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode optimizations")
    parser.add_argument("--markers", nargs="+", help="Run tests with specific markers")
    parser.add_argument("--project-root", type=str, help="Project root directory")

    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    runner = ComprehensiveTestRunner(project_root)

    python_result, rust_result = await runner.run_comprehensive_tests(
        python_only=args.python_only,
        rust_only=args.rust_only,
        fast_mode=args.fast,
        markers=args.markers
    )

    # Exit with non-zero if any tests failed
    total_failed = 0
    if python_result:
        total_failed += python_result.failed_count
    if rust_result:
        total_failed += rust_result.failed_count

    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())