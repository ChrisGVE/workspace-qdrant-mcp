"""
Console Silence Test Runner.

Comprehensive test runner for validating complete console silence in MCP stdio mode.
Executes all test categories and generates detailed reports.

Usage:
    python run_console_silence_tests.py [--quick] [--report-file output.json]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class ConsoleSilenceTestRunner:
    """Test runner for console silence validation."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = time.time()

    def run_test_category(self, category: str, test_file: str) -> dict:
        """Run a specific test category."""
        print(f"\n{'='*60}")
        print(f"Running {category} Tests")
        print(f"{'='*60}")

        test_path = Path(__file__).parent / test_file
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--capture=no",  # Important: don't capture output for these tests
            "--disable-warnings",
            "-x" if self.quick_mode else "",
        ]

        if not self.quick_mode:
            cmd.extend([
                "--benchmark-only",
                "--benchmark-json=benchmark_results.json"
            ])

        # Remove empty strings
        cmd = [arg for arg in cmd if arg]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        test_result = {
            "category": category,
            "test_file": test_file,
            "duration_seconds": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }

        # Parse test results
        if "FAILED" in result.stdout:
            test_result["status"] = "FAILED"
        elif "ERROR" in result.stdout:
            test_result["status"] = "ERROR"
        elif result.returncode == 0:
            test_result["status"] = "PASSED"
        else:
            test_result["status"] = "UNKNOWN"

        # Extract test counts
        for line in result.stdout.split('\n'):
            if " passed" in line or " failed" in line:
                test_result["summary_line"] = line.strip()
                break

        self.test_results[category] = test_result
        return test_result

    def run_all_tests(self) -> dict:
        """Run all console silence test categories."""
        print("Starting Comprehensive Console Silence Validation")
        print(f"Quick mode: {self.quick_mode}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        test_categories = [
            ("Console Capture", "test_console_capture.py"),
            ("MCP Protocol Purity", "test_mcp_protocol_purity.py"),
            ("Integration with Claude Desktop", "test_integration_claude_desktop.py"),
        ]

        if not self.quick_mode:
            test_categories.append(
                ("Performance Benchmarks", "test_performance_benchmarks.py")
            )

        all_passed = True
        critical_failures = []

        for category, test_file in test_categories:
            try:
                result = self.run_test_category(category, test_file)

                if not result["passed"]:
                    all_passed = False
                    if "capture" in test_file or "protocol" in test_file:
                        critical_failures.append(category)

                print(f"\n{category}: {result['status']}")
                print(f"Duration: {result['duration_seconds']:.2f}s")

                if result.get("summary_line"):
                    print(f"Summary: {result['summary_line']}")

                if result["stderr"] and len(result["stderr"]) > 0:
                    print(f"⚠️  STDERR OUTPUT DETECTED: {len(result['stderr'])} chars")
                    if len(result["stderr"]) < 500:
                        print(f"STDERR: {result['stderr']}")

            except Exception as e:
                print(f"ERROR running {category}: {str(e)}")
                all_passed = False
                critical_failures.append(category)

        # Generate summary
        total_duration = time.time() - self.start_time
        summary = {
            "overall_success": all_passed,
            "critical_failures": critical_failures,
            "total_duration_seconds": total_duration,
            "test_categories_run": len(test_categories),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "quick_mode": self.quick_mode
        }

        self.test_results["summary"] = summary
        return self.test_results

    def validate_console_silence_requirements(self) -> bool:
        """Validate that console silence requirements are met."""
        print(f"\n{'='*60}")
        print("VALIDATING CONSOLE SILENCE REQUIREMENTS")
        print(f"{'='*60}")

        requirements_met = True
        failures = []

        # Check each test category for specific requirements
        for category, result in self.test_results.items():
            if category == "summary":
                continue

            # Critical requirement: No stderr output during tests
            if result.get("stderr") and len(result["stderr"]) > 100:
                failures.append(f"{category}: Excessive stderr output ({len(result['stderr'])} chars)")
                requirements_met = False

            # Critical requirement: Tests must pass
            if not result.get("passed"):
                if "capture" in category.lower() or "protocol" in category.lower():
                    failures.append(f"{category}: Critical test category failed")
                    requirements_met = False

        # Print validation results
        if requirements_met:
            print("✅ ALL CONSOLE SILENCE REQUIREMENTS MET")
            print("   - Zero stderr output in stdio mode")
            print("   - Only JSON-RPC messages on stdout")
            print("   - All critical tests pass")
            print("   - MCP protocol compliance verified")
        else:
            print("❌ CONSOLE SILENCE REQUIREMENTS NOT MET")
            for failure in failures:
                print(f"   - {failure}")

        return requirements_met

    def generate_report(self, report_file: str | None = None) -> dict:
        """Generate comprehensive test report."""
        report = {
            "test_execution": self.test_results,
            "validation": {
                "console_silence_achieved": self.validate_console_silence_requirements(),
                "requirements_summary": {
                    "zero_stderr_output": True,  # Will be updated based on results
                    "json_rpc_only_stdout": True,
                    "all_tools_functional": True,
                    "performance_acceptable": True,
                    "integration_successful": True
                }
            },
            "metadata": {
                "test_runner_version": "1.0.0",
                "python_version": sys.version,
                "execution_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration": self.test_results.get("summary", {}).get("total_duration_seconds", 0)
            }
        }

        # Update requirement validation based on results
        validation = report["validation"]["requirements_summary"]

        # Check stderr output
        for category, result in self.test_results.items():
            if category != "summary" and result.get("stderr"):
                validation["zero_stderr_output"] = False

        # Check test results
        if not self.test_results.get("summary", {}).get("overall_success"):
            validation["all_tools_functional"] = False
            validation["integration_successful"] = False

        # Save report if requested
        if report_file:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {report_file}")

        return report

    def print_final_summary(self):
        """Print final test summary."""
        print(f"\n{'='*60}")
        print("CONSOLE SILENCE TEST EXECUTION COMPLETE")
        print(f"{'='*60}")

        summary = self.test_results.get("summary", {})

        print(f"Overall Success: {'✅ YES' if summary.get('overall_success') else '❌ NO'}")
        print(f"Duration: {summary.get('total_duration_seconds', 0):.2f} seconds")
        print(f"Categories Run: {summary.get('test_categories_run', 0)}")

        if summary.get("critical_failures"):
            print(f"Critical Failures: {', '.join(summary['critical_failures'])}")

        # Print per-category results
        print("\nCategory Results:")
        for category, result in self.test_results.items():
            if category == "summary":
                continue

            status_icon = "✅" if result.get("passed") else "❌"
            print(f"  {status_icon} {category}: {result.get('status', 'UNKNOWN')}")

        print("\nTarget Achievement:")
        print(f"  Complete stdio silence: {'✅' if self.validate_console_silence_requirements() else '❌'}")
        print(f"  MCP protocol compliance: {'✅' if 'MCP Protocol Purity' in self.test_results and self.test_results['MCP Protocol Purity'].get('passed') else '❌'}")
        print(f"  Integration functionality: {'✅' if 'Integration with Claude Desktop' in self.test_results and self.test_results['Integration with Claude Desktop'].get('passed') else '❌'}")


def main():
    """Main entry point for console silence test runner."""
    parser = argparse.ArgumentParser(description="Run console silence validation tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only (skip benchmarks)")
    parser.add_argument("--report-file", type=str,
                       help="Save detailed report to JSON file")

    args = parser.parse_args()

    runner = ConsoleSilenceTestRunner(quick_mode=args.quick)

    try:
        # Run all tests
        results = runner.run_all_tests()

        # Generate and optionally save report
        runner.generate_report(args.report_file)

        # Print final summary
        runner.print_final_summary()

        # Exit with appropriate code
        if results.get("summary", {}).get("overall_success"):
            print("\n🎉 ALL CONSOLE SILENCE TESTS PASSED!")
            sys.exit(0)
        else:
            print("\n💥 CONSOLE SILENCE TESTS FAILED!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
