#!/usr/bin/env python3
"""
Quality Loop Framework for Iterative Testing
============================================

This script implements the continuous testing loop for Phase 3 of task 267.3.
It runs tests, identifies failures, analyzes coverage gaps, and provides
actionable feedback for achieving 100% coverage and 100% pass rate.

Usage:
    python 20250921-1939_quality_loop_framework.py [--cycles N] [--python-only] [--rust-only]

Features:
- Parallel test execution with pytest-xdist
- Coverage analysis for both Python and Rust
- Failure analysis and gap identification
- Progress tracking with atomic commits
- TDD workflow support with test isolation
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rich
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()


@dataclass
class TestResult:
    """Test execution result with coverage and failure data."""
    codebase: str  # "python" or "rust"
    passed: int
    failed: int
    skipped: int
    total: int
    coverage_percent: float
    execution_time: float
    failed_tests: List[str]
    coverage_gaps: List[str]
    timestamp: str


@dataclass
class QualityLoopState:
    """State tracking for the quality improvement loop."""
    cycle: int
    python_results: Optional[TestResult]
    rust_results: Optional[TestResult]
    overall_pass_rate: float
    overall_coverage: float
    target_achieved: bool
    improvements_made: List[str]


class QualityLoopFramework:
    """Framework for iterative testing and quality improvement."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.python_src = self.project_root / "src" / "python"
        self.rust_src = self.project_root / "src" / "rust"
        self.test_dir = self.project_root / "tests"
        self.coverage_target = 100.0
        self.pass_rate_target = 100.0

    async def run_python_tests(self, parallel: bool = True) -> TestResult:
        """Run Python test suite with coverage analysis."""
        console.print("🐍 Running Python test suite...", style="bold blue")

        # Configure pytest command with optimal settings
        cmd = [
            "uv", "run", "pytest",
            "tests/unit/",
            "--cov=src/python",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=0",  # Don't fail on coverage, we'll handle it
            "--tb=short",
            "-v",
            "--benchmark-skip",  # Skip benchmarks in quality loop
        ]

        if parallel:
            cmd.extend(["-n", "auto"])  # pytest-xdist parallel execution

        # Add markers for focused testing
        cmd.extend(["-m", "not slow"])  # Skip slow tests for faster iteration

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            # Parse pytest output
            lines = result.stdout.split('\n')
            failed_tests = []
            coverage_percent = 0.0

            # Extract test counts from pytest summary
            for line in lines:
                if "failed" in line and "passed" in line:
                    # Parse: "X failed, Y passed in Z.ZZs"
                    parts = line.split()
                    failed = int(parts[0]) if "failed" in line else 0
                    passed_idx = next(i for i, p in enumerate(parts) if "passed" in p)
                    passed = int(parts[passed_idx - 1])
                    total = failed + passed
                    break
            else:
                # Fallback parsing
                failed = result.stdout.count("FAILED")
                passed = result.stdout.count("PASSED")
                total = failed + passed

            # Extract failed test names
            for line in lines:
                if "FAILED" in line and "::" in line:
                    failed_tests.append(line.split()[0])

            # Parse coverage from XML report
            coverage_xml = self.project_root / "coverage.xml"
            if coverage_xml.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_xml)
                root = tree.getroot()
                coverage_percent = float(root.get('line-rate', 0.0)) * 100

            # Identify coverage gaps from HTML report
            coverage_gaps = await self._analyze_coverage_gaps("python")

            return TestResult(
                codebase="python",
                passed=passed,
                failed=failed,
                skipped=0,  # Will parse if needed
                total=total,
                coverage_percent=coverage_percent,
                execution_time=self._extract_execution_time(result.stdout),
                failed_tests=failed_tests,
                coverage_gaps=coverage_gaps,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        except subprocess.TimeoutExpired:
            console.print("❌ Python tests timed out", style="bold red")
            return TestResult(
                codebase="python",
                passed=0, failed=0, skipped=0, total=0,
                coverage_percent=0.0, execution_time=600.0,
                failed_tests=["TIMEOUT"], coverage_gaps=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            console.print(f"❌ Python test execution failed: {e}", style="bold red")
            return TestResult(
                codebase="python",
                passed=0, failed=1, skipped=0, total=1,
                coverage_percent=0.0, execution_time=0.0,
                failed_tests=[str(e)], coverage_gaps=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

    async def run_rust_tests(self) -> TestResult:
        """Run Rust test suite with coverage analysis."""
        console.print("🦀 Running Rust test suite...", style="bold blue")

        rust_engine = self.project_root / "src" / "rust" / "daemon"
        if not rust_engine.exists():
            console.print("⚠️ Rust engine directory not found", style="yellow")
            return TestResult(
                codebase="rust",
                passed=0, failed=0, skipped=0, total=0,
                coverage_percent=0.0, execution_time=0.0,
                failed_tests=[], coverage_gaps=["Rust engine not found"],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        try:
            # Run tests with coverage using cargo-tarpaulin
            cmd = [
                "cargo", "tarpaulin",
                "--workspace",
                "--out", "xml",
                "--output-dir", str(self.project_root),
                "--fail-under", "0",  # Don't fail on coverage, we'll handle it
                "--timeout", "300"
            ]

            result = subprocess.run(
                cmd,
                cwd=rust_engine,
                capture_output=True,
                text=True,
                timeout=600
            )

            # Parse cargo test output
            lines = result.stdout.split('\n')
            passed = 0
            failed = 0

            for line in lines:
                if "test result:" in line:
                    # Parse: "test result: ok. X passed; Y failed; Z ignored"
                    parts = line.split()
                    passed_idx = next(i for i, p in enumerate(parts) if "passed" in p)
                    passed = int(parts[passed_idx - 1])
                    if "failed" in line:
                        failed_idx = next(i for i, p in enumerate(parts) if "failed" in p)
                        failed = int(parts[failed_idx - 1])
                    break

            total = passed + failed

            # Extract failed test names
            failed_tests = []
            for line in lines:
                if "FAILED" in line or "test ... FAILED" in line:
                    failed_tests.append(line.strip())

            # Parse coverage from tarpaulin XML
            coverage_percent = 0.0
            cobertura_xml = self.project_root / "cobertura.xml"
            if cobertura_xml.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(cobertura_xml)
                root = tree.getroot()
                coverage_percent = float(root.get('line-rate', 0.0)) * 100

            coverage_gaps = await self._analyze_coverage_gaps("rust")

            return TestResult(
                codebase="rust",
                passed=passed,
                failed=failed,
                skipped=0,
                total=total,
                coverage_percent=coverage_percent,
                execution_time=self._extract_execution_time(result.stdout),
                failed_tests=failed_tests,
                coverage_gaps=coverage_gaps,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        except subprocess.TimeoutExpired:
            console.print("❌ Rust tests timed out", style="bold red")
            return TestResult(
                codebase="rust",
                passed=0, failed=0, skipped=0, total=0,
                coverage_percent=0.0, execution_time=600.0,
                failed_tests=["TIMEOUT"], coverage_gaps=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            console.print(f"❌ Rust test execution failed: {e}", style="bold red")
            return TestResult(
                codebase="rust",
                passed=0, failed=1, skipped=0, total=1,
                coverage_percent=0.0, execution_time=0.0,
                failed_tests=[str(e)], coverage_gaps=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

    async def _analyze_coverage_gaps(self, codebase: str) -> List[str]:
        """Analyze coverage reports to identify uncovered code sections."""
        gaps = []

        if codebase == "python":
            # Analyze HTML coverage report
            htmlcov_dir = self.project_root / "htmlcov"
            if htmlcov_dir.exists():
                for html_file in htmlcov_dir.glob("*.html"):
                    if "index" not in html_file.name:
                        # Could parse HTML to find uncovered lines
                        # For now, just identify files with low coverage
                        gaps.append(f"Low coverage in {html_file.stem}")

        elif codebase == "rust":
            # Analyze Rust coverage report
            cobertura_xml = self.project_root / "cobertura.xml"
            if cobertura_xml.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(cobertura_xml)
                root = tree.getroot()
                for class_elem in root.findall(".//class"):
                    line_rate = float(class_elem.get('line-rate', 0.0))
                    if line_rate < 1.0:
                        filename = class_elem.get('filename', 'unknown')
                        gaps.append(f"Incomplete coverage in {filename}")

        return gaps

    def _extract_execution_time(self, output: str) -> float:
        """Extract test execution time from output."""
        import re
        # Look for patterns like "in 12.34s" or "12.34 seconds"
        patterns = [
            r"in ([\d.]+)s",
            r"([\d.]+) seconds",
            r"Finished in ([\d.]+) seconds"
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))

        return 0.0

    def generate_quality_report(self, state: QualityLoopState) -> str:
        """Generate detailed quality improvement report."""
        table = Table(title=f"Quality Loop Cycle {state.cycle}")
        table.add_column("Metric", style="cyan")
        table.add_column("Python", style="green")
        table.add_column("Rust", style="yellow")
        table.add_column("Target", style="red")

        if state.python_results:
            py_pass_rate = (state.python_results.passed / state.python_results.total * 100) if state.python_results.total > 0 else 0
            py_coverage = state.python_results.coverage_percent
        else:
            py_pass_rate = 0
            py_coverage = 0

        if state.rust_results:
            rust_pass_rate = (state.rust_results.passed / state.rust_results.total * 100) if state.rust_results.total > 0 else 0
            rust_coverage = state.rust_results.coverage_percent
        else:
            rust_pass_rate = 0
            rust_coverage = 0

        table.add_row("Pass Rate", f"{py_pass_rate:.1f}%", f"{rust_pass_rate:.1f}%", "100%")
        table.add_row("Coverage", f"{py_coverage:.1f}%", f"{rust_coverage:.1f}%", "100%")

        return table

    def identify_next_actions(self, state: QualityLoopState) -> List[str]:
        """Identify next actions for quality improvement."""
        actions = []

        # Analyze Python issues
        if state.python_results:
            if state.python_results.failed > 0:
                actions.append(f"Fix {state.python_results.failed} failing Python tests:")
                actions.extend(f"  - {test}" for test in state.python_results.failed_tests[:5])

            if state.python_results.coverage_percent < self.coverage_target:
                actions.append(f"Improve Python coverage from {state.python_results.coverage_percent:.1f}% to {self.coverage_target}%")
                actions.extend(f"  - {gap}" for gap in state.python_results.coverage_gaps[:3])

        # Analyze Rust issues
        if state.rust_results:
            if state.rust_results.failed > 0:
                actions.append(f"Fix {state.rust_results.failed} failing Rust tests:")
                actions.extend(f"  - {test}" for test in state.rust_results.failed_tests[:5])

            if state.rust_results.coverage_percent < self.coverage_target:
                actions.append(f"Improve Rust coverage from {state.rust_results.coverage_percent:.1f}% to {self.coverage_target}%")
                actions.extend(f"  - {gap}" for gap in state.rust_results.coverage_gaps[:3])

        return actions

    async def execute_quality_loop(self, max_cycles: int = 10, python_only: bool = False, rust_only: bool = False) -> bool:
        """Execute the iterative quality improvement loop."""
        console.print("🚀 Starting Quality Loop Framework", style="bold green")
        console.print(f"Target: {self.coverage_target}% coverage, {self.pass_rate_target}% pass rate")

        for cycle in range(1, max_cycles + 1):
            console.print(f"\n📊 Quality Loop Cycle {cycle}/{max_cycles}", style="bold cyan")

            # Initialize cycle state
            state = QualityLoopState(
                cycle=cycle,
                python_results=None,
                rust_results=None,
                overall_pass_rate=0.0,
                overall_coverage=0.0,
                target_achieved=False,
                improvements_made=[]
            )

            # Run tests based on configuration
            if not rust_only:
                state.python_results = await self.run_python_tests()

            if not python_only:
                state.rust_results = await self.run_rust_tests()

            # Calculate overall metrics
            total_passed = 0
            total_tests = 0
            total_coverage = 0.0
            coverage_count = 0

            if state.python_results:
                total_passed += state.python_results.passed
                total_tests += state.python_results.total
                total_coverage += state.python_results.coverage_percent
                coverage_count += 1

            if state.rust_results:
                total_passed += state.rust_results.passed
                total_tests += state.rust_results.total
                total_coverage += state.rust_results.coverage_percent
                coverage_count += 1

            state.overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            state.overall_coverage = (total_coverage / coverage_count) if coverage_count > 0 else 0

            # Check if targets achieved
            state.target_achieved = (
                state.overall_pass_rate >= self.pass_rate_target and
                state.overall_coverage >= self.coverage_target
            )

            # Display results
            console.print(self.generate_quality_report(state))

            if state.target_achieved:
                console.print("🎉 Quality targets achieved!", style="bold green")
                await self._commit_quality_milestone(cycle, state)
                return True

            # Identify and display next actions
            actions = self.identify_next_actions(state)
            if actions:
                console.print("\n📋 Next Actions:", style="bold yellow")
                for action in actions:
                    console.print(f"  {action}")

            # Save cycle state for analysis
            await self._save_cycle_state(state)

            # Pause for review or automatic continuation
            if cycle < max_cycles:
                console.print(f"\n⏸️ Cycle {cycle} complete. Ready for fixes...")
                if not await self._should_continue():
                    break

        console.print("❌ Quality targets not achieved within cycle limit", style="bold red")
        return False

    async def _save_cycle_state(self, state: QualityLoopState):
        """Save cycle state for tracking and analysis."""
        state_file = self.project_root / f"20250921-1939_quality_cycle_{state.cycle}.json"

        with open(state_file, 'w') as f:
            json.dump(asdict(state), f, indent=2, default=str)

        console.print(f"💾 Cycle state saved to {state_file.name}", style="dim")

    async def _commit_quality_milestone(self, cycle: int, state: QualityLoopState):
        """Make atomic commit for quality milestone."""
        try:
            commit_msg = f"test: achieve quality targets in cycle {cycle}\n\n"
            commit_msg += f"- Overall pass rate: {state.overall_pass_rate:.1f}%\n"
            commit_msg += f"- Overall coverage: {state.overall_coverage:.1f}%\n"
            commit_msg += f"- Python tests: {state.python_results.passed}/{state.python_results.total} passed\n" if state.python_results else ""
            commit_msg += f"- Rust tests: {state.rust_results.passed}/{state.rust_results.total} passed\n" if state.rust_results else ""
            commit_msg += "\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

            subprocess.run([
                "git", "add", "coverage.xml", "htmlcov/", "cobertura.xml"
            ], cwd=self.project_root, check=False)

            subprocess.run([
                "git", "commit", "-m", commit_msg
            ], cwd=self.project_root, check=True)

            console.print("✅ Quality milestone committed", style="bold green")

        except subprocess.CalledProcessError as e:
            console.print(f"⚠️ Failed to commit milestone: {e}", style="yellow")

    async def _should_continue(self) -> bool:
        """Determine if quality loop should continue (for automation or manual control)."""
        # In automated mode, always continue
        # In manual mode, could prompt user
        return True


async def main():
    """Main entry point for quality loop framework."""
    parser = argparse.ArgumentParser(description="Quality Loop Framework for Iterative Testing")
    parser.add_argument("--cycles", type=int, default=10, help="Maximum cycles to run")
    parser.add_argument("--python-only", action="store_true", help="Run only Python tests")
    parser.add_argument("--rust-only", action="store_true", help="Run only Rust tests")
    parser.add_argument("--project-root", type=str, help="Project root directory")

    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    framework = QualityLoopFramework(project_root)

    success = await framework.execute_quality_loop(
        max_cycles=args.cycles,
        python_only=args.python_only,
        rust_only=args.rust_only
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())