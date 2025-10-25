#!/usr/bin/env python3
"""
Multi-Tenant Architecture Test Runner.

This script executes the comprehensive test suite for the multi-tenant
architecture and generates detailed reports on test results, performance
metrics, and validation summaries.

Usage:
    python tests/run_multitenant_tests.py [options]

Options:
    --unit-only       Run only unit tests
    --integration     Run integration tests
    --performance     Run performance tests
    --migration       Run migration tests
    --all             Run all test categories (default)
    --report-format   Format for reports (json, html, text)
    --output-dir      Directory for test reports
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest


class MultiTenantTestRunner:
    """Comprehensive test runner for multi-tenant architecture."""

    def __init__(self, output_dir: str = "test-reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = None

    def run_test_suite(self, test_categories: list[str], report_format: str = "json") -> dict[str, Any]:
        """Run comprehensive test suite for multi-tenant architecture."""
        self.start_time = time.time()

        print("🚀 Starting Multi-Tenant Architecture Test Suite")
        print("=" * 60)

        # Define test configurations
        test_configs = {
            "unit": {
                "description": "Unit tests for multi-tenant components",
                "paths": [
                    "tests/unit/test_collision_detection.py",
                    "tests/unit/test_metadata_filtering.py",
                    "tests/unit/test_metadata_schema.py",
                    "tests/unit/test_migration_utils.py"
                ],
                "markers": ["not slow", "not performance"],
                "timeout": 300
            },
            "integration": {
                "description": "Integration tests with existing systems",
                "paths": [
                    "tests/integration/test_multitenant_architecture_comprehensive.py",
                    "tests/integration/test_multitenant_migration_scenarios.py"
                ],
                "markers": ["integration", "not performance"],
                "timeout": 1800
            },
            "performance": {
                "description": "Performance and scalability tests",
                "paths": [
                    "tests/performance/test_multitenant_performance.py"
                ],
                "markers": ["performance"],
                "timeout": 3600
            },
            "migration": {
                "description": "Migration scenario validation",
                "paths": [
                    "tests/integration/test_multitenant_migration_scenarios.py"
                ],
                "markers": ["migration", "not performance"],
                "timeout": 1200
            }
        }

        # Run test categories
        overall_results = {
            "test_suite": "multi_tenant_architecture",
            "execution_time": None,
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0
            }
        }

        for category in test_categories:
            if category not in test_configs:
                print(f"⚠️  Unknown test category: {category}")
                continue

            print(f"\n📋 Running {category.upper()} tests...")
            print(f"   {test_configs[category]['description']}")

            category_result = self._run_test_category(category, test_configs[category])
            overall_results["categories"][category] = category_result

            # Update summary
            if category_result.get("success", False):
                summary = category_result.get("summary", {})
                overall_results["summary"]["total_tests"] += summary.get("total", 0)
                overall_results["summary"]["passed"] += summary.get("passed", 0)
                overall_results["summary"]["failed"] += summary.get("failed", 0)
                overall_results["summary"]["skipped"] += summary.get("skipped", 0)
                overall_results["summary"]["errors"] += summary.get("errors", 0)

        overall_results["execution_time"] = time.time() - self.start_time

        # Generate reports
        self._generate_reports(overall_results, report_format)

        # Print summary
        self._print_test_summary(overall_results)

        return overall_results

    def _run_test_category(self, category: str, config: dict[str, Any]) -> dict[str, Any]:
        """Run a specific test category."""
        start_time = time.time()

        # Build pytest command
        pytest_args = [
            "--tb=short",
            "--verbose",
            f"--timeout={config['timeout']}",
            "--junit-xml=" + str(self.output_dir / f"{category}_results.xml"),
            "--json-report",
            "--json-report-file=" + str(self.output_dir / f"{category}_report.json")
        ]

        # Add markers
        if config.get("markers"):
            for marker in config["markers"]:
                pytest_args.extend(["-m", marker])

        # Add coverage for unit tests
        if category == "unit":
            pytest_args.extend([
                "--cov=src/python/common/core",
                "--cov=src/python/workspace_qdrant_mcp/tools",
                "--cov-report=html:" + str(self.output_dir / "coverage_html"),
                "--cov-report=json:" + str(self.output_dir / "coverage.json")
            ])

        # Add test paths
        pytest_args.extend(config["paths"])

        try:
            print(f"   Running: pytest {' '.join(pytest_args)}")

            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest"] + pytest_args,
                capture_output=True,
                text=True,
                timeout=config["timeout"]
            )

            execution_time = time.time() - start_time

            # Parse results
            category_result = {
                "category": category,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "command": " ".join(["pytest"] + pytest_args),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }

            # Try to parse JSON report for detailed results
            json_report_path = self.output_dir / f"{category}_report.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path) as f:
                        json_report = json.load(f)

                    category_result["summary"] = json_report.get("summary", {})
                    category_result["test_details"] = json_report.get("tests", [])

                    # Extract performance metrics if available
                    if category == "performance":
                        category_result["performance_metrics"] = self._extract_performance_metrics(
                            json_report
                        )

                except Exception as e:
                    print(f"   ⚠️  Could not parse JSON report: {e}")

            if result.returncode == 0:
                print(f"   ✅ {category.upper()} tests completed successfully in {execution_time:.1f}s")
            else:
                print(f"   ❌ {category.upper()} tests failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error output: {result.stderr[:500]}...")

            return category_result

        except subprocess.TimeoutExpired:
            print(f"   ⏰ {category.upper()} tests timed out after {config['timeout']}s")
            return {
                "category": category,
                "success": False,
                "execution_time": config["timeout"],
                "error": "Test execution timed out",
                "timeout": True
            }

        except Exception as e:
            print(f"   💥 Error running {category.upper()} tests: {e}")
            return {
                "category": category,
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "exception": True
            }

    def _extract_performance_metrics(self, json_report: dict[str, Any]) -> dict[str, Any]:
        """Extract performance metrics from test results."""
        metrics = {
            "test_performance": {},
            "resource_usage": {},
            "thresholds": {}
        }

        # Extract test-specific performance data
        for test in json_report.get("tests", []):
            test_name = test.get("nodeid", "")

            if "performance" in test_name or "scale" in test_name:
                duration = test.get("duration", 0)
                outcome = test.get("outcome", "unknown")

                metrics["test_performance"][test_name] = {
                    "duration": duration,
                    "outcome": outcome,
                    "setup_duration": test.get("setup", {}).get("duration", 0),
                    "call_duration": test.get("call", {}).get("duration", 0),
                    "teardown_duration": test.get("teardown", {}).get("duration", 0)
                }

        return metrics

    def _generate_reports(self, results: dict[str, Any], report_format: str):
        """Generate test reports in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if report_format in ["json", "all"]:
            json_file = self.output_dir / f"multitenant_test_report_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📄 JSON report saved to: {json_file}")

        if report_format in ["html", "all"]:
            html_file = self.output_dir / f"multitenant_test_report_{timestamp}.html"
            self._generate_html_report(results, html_file)
            print(f"📄 HTML report saved to: {html_file}")

        if report_format in ["text", "all"]:
            text_file = self.output_dir / f"multitenant_test_report_{timestamp}.txt"
            self._generate_text_report(results, text_file)
            print(f"📄 Text report saved to: {text_file}")

    def _generate_html_report(self, results: dict[str, Any], output_file: Path):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Tenant Architecture Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ background: #d4edda; }}
        .failure {{ background: #f8d7da; }}
        .metrics {{ background: #f8f9fa; padding: 10px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Tenant Architecture Test Report</h1>
        <p><strong>Execution Date:</strong> {results['timestamp']}</p>
        <p><strong>Total Execution Time:</strong> {results['execution_time']:.1f} seconds</p>
    </div>

    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Metric</th><th>Count</th></tr>
            <tr><td>Total Tests</td><td>{results['summary']['total_tests']}</td></tr>
            <tr><td>Passed</td><td>{results['summary']['passed']}</td></tr>
            <tr><td>Failed</td><td>{results['summary']['failed']}</td></tr>
            <tr><td>Skipped</td><td>{results['summary']['skipped']}</td></tr>
            <tr><td>Errors</td><td>{results['summary']['errors']}</td></tr>
        </table>
    </div>

    <h2>Test Categories</h2>
"""

        for category, result in results["categories"].items():
            success_class = "success" if result.get("success", False) else "failure"
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"

            html_content += f"""
    <div class="category {success_class}">
        <h3>{category.upper()} Tests - {status}</h3>
        <p><strong>Execution Time:</strong> {result.get('execution_time', 0):.1f} seconds</p>
"""

            if result.get("summary"):
                summary = result["summary"]
                html_content += f"""
        <table>
            <tr><th>Tests</th><th>Passed</th><th>Failed</th><th>Skipped</th></tr>
            <tr>
                <td>{summary.get('total', 0)}</td>
                <td>{summary.get('passed', 0)}</td>
                <td>{summary.get('failed', 0)}</td>
                <td>{summary.get('skipped', 0)}</td>
            </tr>
        </table>
"""

            if result.get("performance_metrics"):
                html_content += """
        <div class="metrics">
            <h4>Performance Metrics</h4>
            <p>Performance data available in detailed JSON report.</p>
        </div>
"""

            html_content += "</div>"

        html_content += """
    <div class="footer">
        <p><em>Generated by workspace-qdrant-mcp test suite</em></p>
    </div>
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html_content)

    def _generate_text_report(self, results: dict[str, Any], output_file: Path):
        """Generate text test report."""
        report_lines = [
            "=" * 80,
            "MULTI-TENANT ARCHITECTURE TEST REPORT",
            "=" * 80,
            f"Execution Date: {results['timestamp']}",
            f"Total Execution Time: {results['execution_time']:.1f} seconds",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Tests: {results['summary']['total_tests']}",
            f"Passed: {results['summary']['passed']}",
            f"Failed: {results['summary']['failed']}",
            f"Skipped: {results['summary']['skipped']}",
            f"Errors: {results['summary']['errors']}",
            "",
            "TEST CATEGORIES",
            "-" * 40
        ]

        for category, result in results["categories"].items():
            status = "PASSED" if result.get("success", False) else "FAILED"
            report_lines.extend([
                "",
                f"{category.upper()} Tests: {status}",
                f"  Execution Time: {result.get('execution_time', 0):.1f} seconds"
            ])

            if result.get("summary"):
                summary = result["summary"]
                report_lines.extend([
                    f"  Tests Run: {summary.get('total', 0)}",
                    f"  Passed: {summary.get('passed', 0)}",
                    f"  Failed: {summary.get('failed', 0)}",
                    f"  Skipped: {summary.get('skipped', 0)}"
                ])

            if result.get("error"):
                report_lines.append(f"  Error: {result['error']}")

        report_lines.extend([
            "",
            "=" * 80,
            "End of Report"
        ])

        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))

    def _print_test_summary(self, results: dict[str, Any]):
        """Print test execution summary to console."""
        print("\n" + "=" * 60)
        print("🎯 MULTI-TENANT ARCHITECTURE TEST RESULTS")
        print("=" * 60)

        summary = results["summary"]
        total_tests = summary["total_tests"]
        passed = summary["passed"]
        failed = summary["failed"]

        if total_tests > 0:
            pass_rate = (passed / total_tests) * 100
            print(f"📊 Overall Results: {passed}/{total_tests} tests passed ({pass_rate:.1f}%)")
        else:
            print("📊 No tests were executed")

        print(f"⏱️  Total Execution Time: {results['execution_time']:.1f} seconds")

        # Category breakdown
        print("\n📋 Category Results:")
        for category, result in results["categories"].items():
            status_icon = "✅" if result.get("success", False) else "❌"
            time_str = f"{result.get('execution_time', 0):.1f}s"
            print(f"   {status_icon} {category.upper():<12} ({time_str})")

        # Recommendations
        print("\n💡 Recommendations:")

        if failed > 0:
            print("   - Review failed tests and fix underlying issues")
            print("   - Check test logs for detailed error information")

        if summary.get("skipped", 0) > 0:
            print("   - Investigate skipped tests - may indicate missing dependencies")

        print("   - Run performance tests regularly to detect regressions")
        print("   - Ensure migration tests pass before deploying schema changes")

        # Final verdict
        if failed == 0 and total_tests > 0:
            print("\n🎉 All tests passed! Multi-tenant architecture is validated.")
        elif total_tests == 0:
            print("\n⚠️  No tests were executed. Check test configuration.")
        else:
            print(f"\n🔥 {failed} test(s) failed. Multi-tenant architecture needs attention.")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive multi-tenant architecture tests"
    )

    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests"
    )
    parser.add_argument(
        "--migration",
        action="store_true",
        help="Run migration tests"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test categories (default)"
    )
    parser.add_argument(
        "--report-format",
        choices=["json", "html", "text", "all"],
        default="json",
        help="Format for test reports"
    )
    parser.add_argument(
        "--output-dir",
        default="test-reports",
        help="Directory for test reports"
    )

    args = parser.parse_args()

    # Determine test categories to run
    categories = []
    if args.unit_only:
        categories = ["unit"]
    elif args.integration:
        categories = ["integration"]
    elif args.performance:
        categories = ["performance"]
    elif args.migration:
        categories = ["migration"]
    else:
        # Default: run all categories
        categories = ["unit", "integration", "migration", "performance"]

    # Run tests
    runner = MultiTenantTestRunner(args.output_dir)
    results = runner.run_test_suite(categories, args.report_format)

    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
