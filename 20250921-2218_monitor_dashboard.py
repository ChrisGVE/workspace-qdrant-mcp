#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard
==============================

Displays real-time progress from the continuous monitoring system.
Reads monitoring data and provides live updates.
"""

import json
import time
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class MonitoringDashboard:
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.last_cycle = 0

    def load_data(self):
        """Load current monitoring data"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def display_current_status(self):
        """Display current monitoring status"""
        data = self.load_data()
        if not data:
            print("No monitoring data available")
            return

        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 80)
        print("CONTINUOUS TEST MONITORING DASHBOARD")
        print("=" * 80)

        # Show monitoring start time
        start_time = datetime.fromisoformat(data["start_time"])
        elapsed = datetime.now(timezone.utc) - start_time
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Elapsed: {elapsed.total_seconds() / 60:.1f} minutes")
        print(f"Current Cycle: {data['current_cycle']}")

        # Show baseline
        baseline = data.get("baseline", {})
        if baseline:
            print(f"\nBASELINE MEASUREMENTS:")
            python_baseline = baseline.get("python", {})
            rust_baseline = baseline.get("rust_check", {})

            if "error" not in python_baseline:
                print(f"  Python Coverage: {python_baseline.get('coverage_percent', 0):.2f}%")
                print(f"  Python Test Pass Rate: {python_baseline.get('pass_rate', 0):.1f}%")
                print(f"  Python Tests: {python_baseline.get('passed_tests', 0)}/{python_baseline.get('total_tests', 0)} passed")
                if python_baseline.get('error_tests', 0) > 0:
                    print(f"  Python Errors: {python_baseline.get('error_tests', 0)} collection errors")
            else:
                print(f"  Python: ERROR - {python_baseline['error']}")

            if "error" not in rust_baseline:
                print(f"  Rust Errors: {rust_baseline.get('error_count', 0)}")
                print(f"  Rust Warnings: {rust_baseline.get('warning_count', 0)}")
                print(f"  Rust Compilation: {'SUCCESS' if rust_baseline.get('compilation_success', False) else 'FAILED'}")
            else:
                print(f"  Rust: ERROR - {rust_baseline['error']}")

        # Show latest measurement if available
        measurements = data.get("measurements", [])
        if measurements:
            latest = measurements[-1]
            print(f"\nLATEST MEASUREMENT (Cycle {latest['cycle']}):")

            python_latest = latest.get("python", {})
            rust_latest = latest.get("rust_check", {})

            if "error" not in python_latest:
                coverage = python_latest.get('coverage_percent', 0)
                pass_rate = python_latest.get('pass_rate', 0)
                print(f"  Python Coverage: {coverage:.2f}% (Target: 100%)")
                print(f"  Python Pass Rate: {pass_rate:.1f}% (Target: 100%)")

                # Progress bars
                coverage_progress = min(coverage, 100)
                pass_rate_progress = min(pass_rate, 100)
                print(f"  Coverage Progress: {'█' * int(coverage_progress/5)}{'░' * (20-int(coverage_progress/5))} {coverage_progress:.1f}%")
                print(f"  Pass Rate Progress: {'█' * int(pass_rate_progress/5)}{'░' * (20-int(pass_rate_progress/5))} {pass_rate_progress:.1f}%")

                # Show improvement
                if baseline and "python" in baseline:
                    baseline_coverage = baseline["python"].get("coverage_percent", 0)
                    baseline_pass_rate = baseline["python"].get("pass_rate", 0)
                    coverage_improvement = coverage - baseline_coverage
                    pass_rate_improvement = pass_rate - baseline_pass_rate
                    print(f"  Coverage Change: {coverage_improvement:+.2f}%")
                    print(f"  Pass Rate Change: {pass_rate_improvement:+.1f}%")
            else:
                print(f"  Python: ERROR - {python_latest['error']}")

            if "error" not in rust_latest:
                errors = rust_latest.get('error_count', 0)
                warnings = rust_latest.get('warning_count', 0)
                success = rust_latest.get('compilation_success', False)
                print(f"  Rust Errors: {errors} (Target: 0)")
                print(f"  Rust Warnings: {warnings}")
                print(f"  Rust Status: {'SUCCESS' if success else 'FAILED'}")

                # Show improvement
                if baseline and "rust_check" in baseline:
                    baseline_errors = baseline["rust_check"].get("error_count", 0)
                    baseline_warnings = baseline["rust_check"].get("warning_count", 0)
                    error_improvement = baseline_errors - errors
                    warning_improvement = baseline_warnings - warnings
                    print(f"  Error Change: {error_improvement:+d}")
                    print(f"  Warning Change: {warning_improvement:+d}")
            else:
                print(f"  Rust: ERROR - {rust_latest['error']}")

        # Show recent failures for quick reference
        if measurements:
            latest = measurements[-1]
            python = latest.get("python", {})
            failures = python.get("failures", [])
            errors = python.get("errors", [])

            if failures or errors:
                print(f"\nRECENT TEST ISSUES:")
                for failure in failures[:5]:  # Show first 5
                    print(f"  FAILED: {failure}")
                for error in errors[:5]:  # Show first 5
                    print(f"  ERROR: {error}")

        # Show targets status
        print(f"\nTARGET STATUS:")
        if measurements:
            latest = measurements[-1]
            python = latest.get("python", {})
            rust = latest.get("rust_check", {})

            coverage_target_met = python.get("coverage_percent", 0) >= 100
            pass_rate_target_met = python.get("pass_rate", 0) >= 100
            rust_target_met = rust.get("error_count", 999) == 0

            print(f"  Coverage 100%: {'✅' if coverage_target_met else '❌'}")
            print(f"  Pass Rate 100%: {'✅' if pass_rate_target_met else '❌'}")
            print(f"  Rust 0 Errors: {'✅' if rust_target_met else '❌'}")

            if coverage_target_met and pass_rate_target_met and rust_target_met:
                print(f"\n🎉 ALL TARGETS ACHIEVED! 🎉")

        # Show next check time
        if data['current_cycle'] > 0:
            print(f"\nNext check in approximately {10 - (elapsed.total_seconds() % 600) / 60:.1f} minutes")

        print("=" * 80)
        print("Press Ctrl+C to exit dashboard")

    def run_dashboard(self):
        """Run live dashboard"""
        try:
            while True:
                self.display_current_status()
                time.sleep(30)  # Update every 30 seconds
        except KeyboardInterrupt:
            print("\nDashboard stopped")


def main():
    data_file = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/20250921-2216_monitoring_data.json"
    dashboard = MonitoringDashboard(data_file)
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()