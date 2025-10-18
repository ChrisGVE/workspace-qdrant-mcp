#!/usr/bin/env python3
"""
Comprehensive test orchestration and reporting system.

Coordinates all integration test scenarios, generates detailed reports,
integrates with CI/CD pipelines, and provides failure analysis.

Features:
- Test suite orchestration
- Docker Compose management
- Comprehensive HTML/JSON reporting
- Performance metrics aggregation
- Failure analysis and debugging
- Test data cleanup
- Environment reset
- CI/CD integration

Task: #290.10 - Implement comprehensive test orchestration and reporting
Parent: #290 - Build MCP-daemon integration test framework
"""

import argparse
import asyncio
import json
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil


class TestOrchestrator:
    """
    Orchestrates integration test execution with comprehensive reporting.
    """

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        docker_compose_path: Optional[Path] = None,
    ):
        """
        Initialize test orchestrator.

        Args:
            project_root: Root directory of the project
            output_dir: Directory for test results and reports
            docker_compose_path: Path to docker-compose.yml
        """
        self.project_root = project_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.docker_compose_path = docker_compose_path or (
            project_root / "docker" / "integration-tests" / "docker-compose.yml"
        )

        self.test_results: Dict[str, Any] = {}
        self.start_time: float = 0
        self.end_time: float = 0

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """
        Run shell command and capture output.

        Args:
            cmd: Command and arguments
            cwd: Working directory

        Returns:
            CompletedProcess with results
        """
        print(f"Running: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            cwd=cwd or self.project_root,
            capture_output=True,
            text=True,
        )

    def start_docker_services(self) -> bool:
        """
        Start Docker Compose services.

        Returns:
            True if services started successfully
        """
        print("\n" + "=" * 80)
        print("STARTING DOCKER SERVICES")
        print("=" * 80)

        compose_dir = self.docker_compose_path.parent

        # Stop any existing services
        self.run_command(
            ["docker-compose", "down", "-v"],
            cwd=compose_dir
        )

        # Start services
        result = self.run_command(
            ["docker-compose", "up", "-d"],
            cwd=compose_dir
        )

        if result.returncode != 0:
            print(f"ERROR: Failed to start Docker services")
            print(result.stderr)
            return False

        # Wait for services to be healthy
        print("Waiting for services to be healthy...")
        time.sleep(10)

        return True

    def stop_docker_services(self):
        """Stop Docker Compose services."""
        print("\n" + "=" * 80)
        print("STOPPING DOCKER SERVICES")
        print("=" * 80)

        compose_dir = self.docker_compose_path.parent
        self.run_command(
            ["docker-compose", "down", "-v"],
            cwd=compose_dir
        )

    def run_test_suite(
        self,
        test_path: str,
        suite_name: str,
        markers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a test suite and collect results.

        Args:
            test_path: Path to test file or directory
            suite_name: Name of the test suite
            markers: Pytest markers to filter tests

        Returns:
            Test results dictionary
        """
        print("\n" + "=" * 80)
        print(f"RUNNING TEST SUITE: {suite_name}")
        print("=" * 80)

        # Build pytest command
        cmd = [
            "uv", "run", "pytest",
            test_path,
            "-v",
            "--tb=short",
            f"--junit-xml={self.output_dir / f'{suite_name}_junit.xml'}",
            f"--html={self.output_dir / f'{suite_name}_report.html'}",
            "--self-contained-html",
        ]

        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        # Run tests
        start_time = time.time()
        result = self.run_command(cmd)
        duration = time.time() - start_time

        # Parse results
        suite_results = {
            "suite_name": suite_name,
            "test_path": test_path,
            "duration_seconds": duration,
            "exit_code": result.returncode,
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        # Extract test counts from output
        if "passed" in result.stdout:
            # Parse pytest summary
            for line in result.stdout.split("\n"):
                if "passed" in line or "failed" in line:
                    suite_results["summary_line"] = line.strip()
                    break

        self.test_results[suite_name] = suite_results
        return suite_results

    def cleanup_test_data(self):
        """Clean up test data and temporary files."""
        print("\n" + "=" * 80)
        print("CLEANING UP TEST DATA")
        print("=" * 80)

        # Clean up common test directories
        cleanup_dirs = [
            self.project_root / "test_results",
            self.project_root / "htmlcov",
            self.project_root / ".pytest_cache",
        ]

        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                print(f"Removing {cleanup_dir}")
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        # Clean up __pycache__ directories
        for pycache in self.project_root.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)

    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive HTML report.

        Returns:
            Path to generated report
        """
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)

        total_duration = self.end_time - self.start_time

        # Calculate overall statistics
        total_suites = len(self.test_results)
        passed_suites = sum(1 for r in self.test_results.values() if r["passed"])
        failed_suites = total_suites - passed_suites

        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Integration Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .suite {{
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .suite.passed {{
            border-left-color: #27ae60;
        }}
        .suite.failed {{
            border-left-color: #e74c3c;
        }}
        .metric {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            background-color: #ecf0f1;
            border-radius: 3px;
        }}
        .passed {{ color: #27ae60; font-weight: bold; }}
        .failed {{ color: #e74c3c; font-weight: bold; }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Integration Test Comprehensive Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <h2>Test Execution Summary</h2>
        <div class="metric">Total Duration: {total_duration:.2f}s</div>
        <div class="metric">Total Suites: {total_suites}</div>
        <div class="metric"><span class="passed">Passed: {passed_suites}</span></div>
        <div class="metric"><span class="failed">Failed: {failed_suites}</span></div>
    </div>

    <h2>Test Suite Results</h2>
"""

        # Add each suite
        for suite_name, results in self.test_results.items():
            status_class = "passed" if results["passed"] else "failed"
            status_text = "PASSED" if results["passed"] else "FAILED"

            html_content += f"""
    <div class="suite {status_class}">
        <h3>{suite_name} - <span class="{status_class}">{status_text}</span></h3>
        <p>Duration: {results['duration_seconds']:.2f}s</p>
        <p>Test Path: {results['test_path']}</p>
        {f"<p>{results.get('summary_line', '')}</p>" if results.get('summary_line') else ""}
"""

            if not results["passed"] and results.get("stderr"):
                html_content += f"""
        <h4>Error Output:</h4>
        <pre>{results['stderr'][:1000]}</pre>
"""

            html_content += "    </div>\n"

        html_content += """
</body>
</html>
"""

        # Write report
        report_path = self.output_dir / "comprehensive_report.html"
        report_path.write_text(html_content)

        print(f"Report generated: {report_path}")
        return str(report_path)

    def generate_json_report(self) -> str:
        """
        Generate JSON report for CI/CD integration.

        Returns:
            Path to JSON report
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": self.end_time - self.start_time,
            "test_suites": self.test_results,
            "summary": {
                "total_suites": len(self.test_results),
                "passed_suites": sum(1 for r in self.test_results.values() if r["passed"]),
                "failed_suites": sum(1 for r in self.test_results.values() if not r["passed"]),
            }
        }

        json_path = self.output_dir / "test_results.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"JSON report generated: {json_path}")
        return str(json_path)

    def run_all_tests(self, skip_docker: bool = False) -> bool:
        """
        Run all integration test suites.

        Args:
            skip_docker: Skip Docker service management

        Returns:
            True if all tests passed
        """
        self.start_time = time.time()

        try:
            # Start Docker services
            if not skip_docker:
                if not self.start_docker_services():
                    return False

            # Run test suites
            test_suites = [
                {
                    "path": "tests/integration/test_mcp_daemon_docker_integration.py",
                    "name": "mcp_daemon_communication",
                },
                {
                    "path": "tests/integration/test_e2e_ingestion_workflow.py",
                    "name": "e2e_ingestion_workflow",
                },
                {
                    "path": "tests/integration/test_realtime_file_watching.py",
                    "name": "realtime_file_watching",
                },
                {
                    "path": "tests/integration/test_grpc_load_stress.py",
                    "name": "grpc_load_stress",
                },
                {
                    "path": "tests/integration/test_connection_failure_recovery.py",
                    "name": "connection_failure_recovery",
                },
                {
                    "path": "tests/integration/test_state_consistency.py",
                    "name": "state_consistency",
                },
                {
                    "path": "tests/integration/test_concurrent_operations.py",
                    "name": "concurrent_operations",
                },
                {
                    "path": "tests/integration/test_performance_monitoring.py",
                    "name": "performance_monitoring",
                },
            ]

            for suite in test_suites:
                self.run_test_suite(
                    test_path=suite["path"],
                    suite_name=suite["name"],
                )

        finally:
            self.end_time = time.time()

            # Stop Docker services
            if not skip_docker:
                self.stop_docker_services()

            # Generate reports
            self.generate_comprehensive_report()
            self.generate_json_report()

        # Return overall success
        return all(r["passed"] for r in self.test_results.values())


def main():
    """Main entry point for test orchestration."""
    parser = argparse.ArgumentParser(
        description="Run integration tests with comprehensive reporting"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Output directory for test results",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker service management (assume services running)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test data after execution",
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Run specific test suite only",
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent

    # Create orchestrator
    orchestrator = TestOrchestrator(
        project_root=project_root,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("INTEGRATION TEST ORCHESTRATION")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Skip Docker: {args.skip_docker}")
    print("=" * 80)

    # Run tests
    if args.suite:
        # Run specific suite
        suite_path = f"tests/integration/{args.suite}"
        orchestrator.run_test_suite(
            test_path=suite_path,
            suite_name=args.suite,
        )
    else:
        # Run all suites
        success = orchestrator.run_all_tests(skip_docker=args.skip_docker)

    # Cleanup if requested
    if args.cleanup:
        orchestrator.cleanup_test_data()

    # Print final summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION COMPLETE")
    print("=" * 80)

    total_suites = len(orchestrator.test_results)
    passed_suites = sum(1 for r in orchestrator.test_results.values() if r["passed"])
    failed_suites = total_suites - passed_suites

    print(f"Total Suites: {total_suites}")
    print(f"Passed: {passed_suites}")
    print(f"Failed: {failed_suites}")
    print(f"Duration: {orchestrator.end_time - orchestrator.start_time:.2f}s")
    print(f"Reports: {args.output_dir}")
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if failed_suites == 0 else 1)


if __name__ == "__main__":
    main()
