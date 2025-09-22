#!/usr/bin/env python3
"""
Continuous Test Monitoring Framework
====================================

Monitors Python test coverage and Rust compilation progress toward 100% targets.

Requirements:
- Python coverage: Current 8.41% → Target 100%
- Rust compilation: 65+ errors → Target 0 errors
- Test passing rate: Target 100%

Monitoring intervals:
- Python tests: Every 10 minutes
- Rust checks: Every 10 minutes
- Progress reports: Every 30 minutes
"""

import json
import time
import subprocess
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
import threading


class ContinuousTestMonitor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.monitoring_data = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "baseline": {},
            "measurements": [],
            "progress_reports": [],
            "current_cycle": 0
        }
        self.data_file = self.project_root / "20250921-2216_monitoring_data.json"
        self.target_coverage = 100.0
        self.target_rust_errors = 0
        self.target_pass_rate = 100.0

    def load_existing_data(self):
        """Load existing monitoring data if available"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    existing_data = json.load(f)
                    self.monitoring_data.update(existing_data)
                    print(f"Loaded existing monitoring data with {len(self.monitoring_data['measurements'])} measurements")
            except Exception as e:
                print(f"Error loading existing data: {e}")

    def save_data(self):
        """Save monitoring data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.monitoring_data, f, indent=2)
        except Exception as e:
            print(f"Error saving monitoring data: {e}")

    def run_python_tests_with_coverage(self) -> Dict[str, Any]:
        """Run Python tests with coverage analysis"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Python tests with coverage...")

        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["uv", "run", "pytest", "--cov=src", "--cov-report=term-missing", "--cov-report=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Parse coverage from JSON report
            coverage_file = self.project_root / "coverage.json"
            coverage_percent = 0.0
            covered_lines = 0
            total_lines = 0
            missing_lines = 0

            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0.0)
                        covered_lines = coverage_data.get('totals', {}).get('covered_lines', 0)
                        total_lines = coverage_data.get('totals', {}).get('num_statements', 0)
                        missing_lines = total_lines - covered_lines
                except Exception as e:
                    print(f"Error parsing coverage.json: {e}")

            # Extract test results
            test_output = result.stdout + result.stderr

            # Count passed/failed tests
            passed_tests = len(re.findall(r'PASSED', test_output))
            failed_tests = len(re.findall(r'FAILED', test_output))
            error_tests = len(re.findall(r'ERROR', test_output))
            skipped_tests = len(re.findall(r'SKIPPED', test_output))

            total_tests = passed_tests + failed_tests + error_tests + skipped_tests
            pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

            # Extract specific failures
            failure_matches = re.findall(r'FAILED (.*?) -', test_output)
            error_matches = re.findall(r'ERROR (.*?) -', test_output)

            return {
                "coverage_percent": coverage_percent,
                "covered_lines": covered_lines,
                "total_lines": total_lines,
                "missing_lines": missing_lines,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "skipped_tests": skipped_tests,
                "total_tests": total_tests,
                "pass_rate": pass_rate,
                "failures": failure_matches,
                "errors": error_matches,
                "exit_code": result.returncode,
                "output": test_output[-2000:],  # Last 2000 chars
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except subprocess.TimeoutExpired:
            return {
                "error": "Python tests timed out after 5 minutes",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "error": f"Python test execution failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def run_rust_checks(self) -> Dict[str, Any]:
        """Run Rust compilation checks"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Rust compilation checks...")

        rust_dir = self.project_root / "rust-engine"
        if not rust_dir.exists():
            return {
                "error": "rust-engine directory not found",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        try:
            # Run cargo check
            result = subprocess.run(
                ["cargo", "check"],
                cwd=rust_dir,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )

            output = result.stdout + result.stderr

            # Count errors and warnings
            error_count = len(re.findall(r'error:', output))
            warning_count = len(re.findall(r'warning:', output))

            # Extract specific errors
            error_matches = re.findall(r'error\[E\d+\]: (.*?)(?:\n|$)', output)
            warning_matches = re.findall(r'warning: (.*?)(?:\n|$)', output)

            # Check if compilation succeeded
            compilation_success = result.returncode == 0 and error_count == 0

            return {
                "error_count": error_count,
                "warning_count": warning_count,
                "compilation_success": compilation_success,
                "errors": error_matches[:10],  # First 10 errors
                "warnings": warning_matches[:10],  # First 10 warnings
                "exit_code": result.returncode,
                "output": output[-1500:],  # Last 1500 chars
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except subprocess.TimeoutExpired:
            return {
                "error": "Rust compilation check timed out after 3 minutes",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "error": f"Rust check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def run_rust_tests(self) -> Dict[str, Any]:
        """Run Rust tests (only if compilation succeeds)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Rust tests...")

        rust_dir = self.project_root / "rust-engine"
        if not rust_dir.exists():
            return {
                "error": "rust-engine directory not found",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        try:
            # Run cargo test
            result = subprocess.run(
                ["cargo", "test"],
                cwd=rust_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            # Parse test results
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            ignored_match = re.search(r'(\d+) ignored', output)

            passed_tests = int(passed_match.group(1)) if passed_match else 0
            failed_tests = int(failed_match.group(1)) if failed_match else 0
            ignored_tests = int(ignored_match.group(1)) if ignored_match else 0

            total_tests = passed_tests + failed_tests + ignored_tests
            pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

            return {
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "ignored_tests": ignored_tests,
                "total_tests": total_tests,
                "pass_rate": pass_rate,
                "exit_code": result.returncode,
                "output": output[-1500:],  # Last 1500 chars
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except subprocess.TimeoutExpired:
            return {
                "error": "Rust tests timed out after 5 minutes",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "error": f"Rust tests failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def establish_baseline(self):
        """Establish baseline measurements"""
        print("=" * 60)
        print("ESTABLISHING BASELINE MEASUREMENTS")
        print("=" * 60)

        # Get Python baseline
        python_baseline = self.run_python_tests_with_coverage()

        # Get Rust baseline
        rust_baseline = self.run_rust_checks()

        # Only run Rust tests if compilation succeeds
        rust_test_baseline = {}
        if rust_baseline.get('compilation_success', False):
            rust_test_baseline = self.run_rust_tests()
        else:
            rust_test_baseline = {"note": "Skipped due to compilation errors"}

        self.monitoring_data["baseline"] = {
            "python": python_baseline,
            "rust_check": rust_baseline,
            "rust_tests": rust_test_baseline,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.save_data()
        self.print_baseline_report()

    def print_baseline_report(self):
        """Print baseline measurements report"""
        baseline = self.monitoring_data["baseline"]
        print("\n" + "=" * 60)
        print("BASELINE REPORT")
        print("=" * 60)

        # Python baseline
        python = baseline.get("python", {})
        if "error" not in python:
            print(f"Python Coverage: {python.get('coverage_percent', 0):.2f}%")
            print(f"Python Tests: {python.get('passed_tests', 0)}/{python.get('total_tests', 0)} passed ({python.get('pass_rate', 0):.1f}%)")
            print(f"Python Failures: {python.get('failed_tests', 0)} failed, {python.get('error_tests', 0)} errors")
        else:
            print(f"Python Error: {python['error']}")

        # Rust baseline
        rust = baseline.get("rust_check", {})
        if "error" not in rust:
            print(f"Rust Errors: {rust.get('error_count', 0)}")
            print(f"Rust Warnings: {rust.get('warning_count', 0)}")
            print(f"Rust Compilation: {'SUCCESS' if rust.get('compilation_success', False) else 'FAILED'}")
        else:
            print(f"Rust Error: {rust['error']}")

        # Rust tests baseline
        rust_tests = baseline.get("rust_tests", {})
        if "error" not in rust_tests and "note" not in rust_tests:
            print(f"Rust Tests: {rust_tests.get('passed_tests', 0)}/{rust_tests.get('total_tests', 0)} passed ({rust_tests.get('pass_rate', 0):.1f}%)")
        elif "note" in rust_tests:
            print(f"Rust Tests: {rust_tests['note']}")
        else:
            print(f"Rust Tests Error: {rust_tests.get('error', 'Unknown error')}")

        print("=" * 60)

    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        self.monitoring_data["current_cycle"] += 1
        cycle = self.monitoring_data["current_cycle"]

        print(f"\n[CYCLE {cycle}] Starting monitoring cycle at {datetime.now().strftime('%H:%M:%S')}")

        # Run Python tests
        python_results = self.run_python_tests_with_coverage()

        # Run Rust checks
        rust_results = self.run_rust_checks()

        # Run Rust tests if compilation succeeds
        rust_test_results = {}
        if rust_results.get('compilation_success', False):
            rust_test_results = self.run_rust_tests()
        else:
            rust_test_results = {"note": "Skipped due to compilation errors"}

        # Store measurement
        measurement = {
            "cycle": cycle,
            "python": python_results,
            "rust_check": rust_results,
            "rust_tests": rust_test_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.monitoring_data["measurements"].append(measurement)
        self.save_data()

        # Print cycle summary
        self.print_cycle_summary(measurement)

        # Check for regressions
        self.check_for_regressions(measurement)

        return measurement

    def print_cycle_summary(self, measurement: Dict[str, Any]):
        """Print summary of current cycle"""
        cycle = measurement["cycle"]
        python = measurement.get("python", {})
        rust = measurement.get("rust_check", {})
        rust_tests = measurement.get("rust_tests", {})

        print(f"\n[CYCLE {cycle}] SUMMARY:")

        if "error" not in python:
            coverage = python.get('coverage_percent', 0)
            pass_rate = python.get('pass_rate', 0)
            print(f"  Python: {coverage:.2f}% coverage, {pass_rate:.1f}% tests passing")

            # Show progress toward target
            coverage_progress = (coverage / self.target_coverage) * 100
            print(f"  Coverage Progress: {coverage_progress:.1f}% of target")
        else:
            print(f"  Python Error: {python['error']}")

        if "error" not in rust:
            errors = rust.get('error_count', 0)
            warnings = rust.get('warning_count', 0)
            success = rust.get('compilation_success', False)
            print(f"  Rust: {errors} errors, {warnings} warnings, {'SUCCESS' if success else 'FAILED'}")

            # Show progress toward target (fewer errors is better)
            if errors == 0:
                print(f"  Rust Progress: 100.0% of target (0 errors achieved)")
            else:
                print(f"  Rust Progress: {errors} errors remaining")
        else:
            print(f"  Rust Error: {rust['error']}")

        if "error" not in rust_tests and "note" not in rust_tests:
            rust_pass_rate = rust_tests.get('pass_rate', 0)
            print(f"  Rust Tests: {rust_pass_rate:.1f}% passing")
        elif "note" in rust_tests:
            print(f"  Rust Tests: {rust_tests['note']}")

    def check_for_regressions(self, measurement: Dict[str, Any]):
        """Check for regressions compared to previous measurements"""
        if len(self.monitoring_data["measurements"]) < 2:
            return  # Need at least 2 measurements to compare

        prev_measurement = self.monitoring_data["measurements"][-2]
        current = measurement

        # Check Python coverage regression
        prev_coverage = prev_measurement.get("python", {}).get("coverage_percent", 0)
        curr_coverage = current.get("python", {}).get("coverage_percent", 0)

        if curr_coverage < prev_coverage:
            print(f"  ⚠️  REGRESSION: Python coverage decreased from {prev_coverage:.2f}% to {curr_coverage:.2f}%")

        # Check Python test pass rate regression
        prev_pass_rate = prev_measurement.get("python", {}).get("pass_rate", 0)
        curr_pass_rate = current.get("python", {}).get("pass_rate", 0)

        if curr_pass_rate < prev_pass_rate:
            print(f"  ⚠️  REGRESSION: Python pass rate decreased from {prev_pass_rate:.1f}% to {curr_pass_rate:.1f}%")

        # Check Rust error count regression
        prev_errors = prev_measurement.get("rust_check", {}).get("error_count", 0)
        curr_errors = current.get("rust_check", {}).get("error_count", 0)

        if curr_errors > prev_errors:
            print(f"  ⚠️  REGRESSION: Rust errors increased from {prev_errors} to {curr_errors}")

    def generate_progress_report(self):
        """Generate comprehensive progress report"""
        if not self.monitoring_data["measurements"]:
            print("No measurements available for progress report")
            return

        baseline = self.monitoring_data["baseline"]
        latest = self.monitoring_data["measurements"][-1]

        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "monitoring_duration_minutes": len(self.monitoring_data["measurements"]) * 10,
            "total_cycles": len(self.monitoring_data["measurements"]),
            "python_progress": self.calculate_python_progress(baseline, latest),
            "rust_progress": self.calculate_rust_progress(baseline, latest),
            "overall_status": self.calculate_overall_status(latest)
        }

        self.monitoring_data["progress_reports"].append(report)
        self.save_data()
        self.print_progress_report(report)

        return report

    def calculate_python_progress(self, baseline: Dict, latest: Dict) -> Dict:
        """Calculate Python testing progress"""
        baseline_python = baseline.get("python", {})
        latest_python = latest.get("python", {})

        baseline_coverage = baseline_python.get("coverage_percent", 0)
        latest_coverage = latest_python.get("coverage_percent", 0)

        baseline_pass_rate = baseline_python.get("pass_rate", 0)
        latest_pass_rate = latest_python.get("pass_rate", 0)

        coverage_improvement = latest_coverage - baseline_coverage
        pass_rate_improvement = latest_pass_rate - baseline_pass_rate

        coverage_target_progress = (latest_coverage / self.target_coverage) * 100
        pass_rate_target_progress = (latest_pass_rate / self.target_pass_rate) * 100

        return {
            "baseline_coverage": baseline_coverage,
            "current_coverage": latest_coverage,
            "coverage_improvement": coverage_improvement,
            "coverage_target_progress": coverage_target_progress,
            "baseline_pass_rate": baseline_pass_rate,
            "current_pass_rate": latest_pass_rate,
            "pass_rate_improvement": pass_rate_improvement,
            "pass_rate_target_progress": pass_rate_target_progress,
            "coverage_target_reached": latest_coverage >= self.target_coverage,
            "pass_rate_target_reached": latest_pass_rate >= self.target_pass_rate
        }

    def calculate_rust_progress(self, baseline: Dict, latest: Dict) -> Dict:
        """Calculate Rust compilation progress"""
        baseline_rust = baseline.get("rust_check", {})
        latest_rust = latest.get("rust_check", {})

        baseline_errors = baseline_rust.get("error_count", 0)
        latest_errors = latest_rust.get("error_count", 0)

        baseline_warnings = baseline_rust.get("warning_count", 0)
        latest_warnings = latest_rust.get("warning_count", 0)

        error_improvement = baseline_errors - latest_errors
        warning_improvement = baseline_warnings - latest_warnings

        compilation_success = latest_rust.get("compilation_success", False)

        return {
            "baseline_errors": baseline_errors,
            "current_errors": latest_errors,
            "error_improvement": error_improvement,
            "baseline_warnings": baseline_warnings,
            "current_warnings": latest_warnings,
            "warning_improvement": warning_improvement,
            "compilation_success": compilation_success,
            "target_reached": latest_errors == self.target_rust_errors
        }

    def calculate_overall_status(self, latest: Dict) -> Dict:
        """Calculate overall progress status"""
        python = latest.get("python", {})
        rust = latest.get("rust_check", {})

        coverage = python.get("coverage_percent", 0)
        python_pass_rate = python.get("pass_rate", 0)
        rust_errors = rust.get("error_count", 0)
        rust_success = rust.get("compilation_success", False)

        # Calculate overall completion percentage
        coverage_completion = min((coverage / self.target_coverage) * 100, 100)
        pass_rate_completion = min((python_pass_rate / self.target_pass_rate) * 100, 100)
        rust_completion = 100 if rust_errors == 0 else max(0, 100 - (rust_errors * 2))  # Rough estimate

        overall_completion = (coverage_completion + pass_rate_completion + rust_completion) / 3

        all_targets_met = (
            coverage >= self.target_coverage and
            python_pass_rate >= self.target_pass_rate and
            rust_errors <= self.target_rust_errors
        )

        return {
            "overall_completion_percent": overall_completion,
            "coverage_target_met": coverage >= self.target_coverage,
            "pass_rate_target_met": python_pass_rate >= self.target_pass_rate,
            "rust_target_met": rust_errors <= self.target_rust_errors,
            "all_targets_met": all_targets_met,
            "ready_for_completion": all_targets_met
        }

    def print_progress_report(self, report: Dict):
        """Print formatted progress report"""
        print("\n" + "=" * 80)
        print("PROGRESS REPORT")
        print("=" * 80)

        print(f"Monitoring Duration: {report['monitoring_duration_minutes']} minutes ({report['total_cycles']} cycles)")
        print(f"Report Generated: {datetime.fromisoformat(report['report_timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Python progress
        python = report["python_progress"]
        print(f"\nPYTHON PROGRESS:")
        print(f"  Coverage: {python['baseline_coverage']:.2f}% → {python['current_coverage']:.2f}% ({python['coverage_improvement']:+.2f}%)")
        print(f"  Target Progress: {python['coverage_target_progress']:.1f}% of {self.target_coverage}%")
        print(f"  Pass Rate: {python['baseline_pass_rate']:.1f}% → {python['current_pass_rate']:.1f}% ({python['pass_rate_improvement']:+.1f}%)")
        print(f"  Target Progress: {python['pass_rate_target_progress']:.1f}% of {self.target_pass_rate}%")
        print(f"  Targets Met: Coverage={'✓' if python['coverage_target_reached'] else '✗'}, Pass Rate={'✓' if python['pass_rate_target_reached'] else '✗'}")

        # Rust progress
        rust = report["rust_progress"]
        print(f"\nRUST PROGRESS:")
        print(f"  Errors: {rust['baseline_errors']} → {rust['current_errors']} ({rust['error_improvement']:+d})")
        print(f"  Warnings: {rust['baseline_warnings']} → {rust['current_warnings']} ({rust['warning_improvement']:+d})")
        print(f"  Compilation: {'SUCCESS' if rust['compilation_success'] else 'FAILED'}")
        print(f"  Target Met: {'✓' if rust['target_reached'] else '✗'}")

        # Overall status
        overall = report["overall_status"]
        print(f"\nOVERALL STATUS:")
        print(f"  Completion: {overall['overall_completion_percent']:.1f}%")
        print(f"  All Targets Met: {'✓ READY FOR COMPLETION' if overall['all_targets_met'] else '✗ CONTINUE MONITORING'}")

        print("=" * 80)

    def start_continuous_monitoring(self, total_minutes: int = 480):  # 8 hours default
        """Start continuous monitoring loop"""
        print(f"Starting continuous monitoring for {total_minutes} minutes...")
        print("Test cycles every 10 minutes, progress reports every 30 minutes")
        print("Press Ctrl+C to stop monitoring\n")

        # Establish baseline if not exists
        if not self.monitoring_data["baseline"]:
            self.establish_baseline()

        start_time = time.time()
        end_time = start_time + (total_minutes * 60)
        last_report_time = start_time

        try:
            while time.time() < end_time:
                cycle_start = time.time()

                # Run monitoring cycle
                measurement = self.run_monitoring_cycle()

                # Generate progress report every 30 minutes
                if time.time() - last_report_time >= 1800:  # 30 minutes
                    self.generate_progress_report()
                    last_report_time = time.time()

                # Check if all targets are met
                overall_status = self.calculate_overall_status(measurement)
                if overall_status["all_targets_met"]:
                    print("\n🎉 ALL TARGETS ACHIEVED! 🎉")
                    print("✅ 100% Python coverage")
                    print("✅ 100% test passing rate")
                    print("✅ 0 Rust compilation errors")
                    self.generate_progress_report()
                    print("Monitoring complete - all success criteria met!")
                    break

                # Wait for next cycle (10 minutes total)
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, 600 - cycle_duration)  # 10 minutes = 600 seconds

                if sleep_time > 0:
                    print(f"Waiting {sleep_time:.0f} seconds until next cycle...")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.generate_progress_report()

        except Exception as e:
            print(f"\nMonitoring error: {e}")
            self.generate_progress_report()


def main():
    """Main entry point"""
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"

    monitor = ContinuousTestMonitor(project_root)
    monitor.load_existing_data()

    # Start continuous monitoring
    monitor.start_continuous_monitoring(total_minutes=480)  # 8 hours


if __name__ == "__main__":
    main()