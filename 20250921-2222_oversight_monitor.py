#!/usr/bin/env python3
"""
Oversight Monitor
================

Monitors the monitoring system itself and provides oversight reports.
Ensures continuous execution is proceeding correctly toward 100% targets.
"""

import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class OversightMonitor:
    def __init__(self):
        self.data_file = Path("20250921-2216_monitoring_data.json")
        self.reports_dir = Path("20250921-2222_oversight_reports")
        self.reports_dir.mkdir(exist_ok=True)

    def check_monitoring_health(self):
        """Check if monitoring system is healthy"""
        try:
            # Check if data file exists and is recent
            if not self.data_file.exists():
                return False, "Monitoring data file not found"

            # Check file modification time
            mod_time = datetime.fromtimestamp(self.data_file.stat().st_mtime, timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - mod_time).total_seconds() / 60

            if age_minutes > 15:
                return False, f"Monitoring data is {age_minutes:.1f} minutes old (stale)"

            # Load and validate data
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            if not data.get("baseline"):
                return False, "No baseline measurements found"

            # Check if monitoring is progressing
            start_time = datetime.fromisoformat(data["start_time"])
            elapsed_minutes = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
            expected_cycles = max(1, int(elapsed_minutes / 10))
            actual_cycles = len(data.get("measurements", []))

            if actual_cycles < expected_cycles - 1:
                return False, f"Expected {expected_cycles} cycles, found {actual_cycles}"

            return True, f"Healthy - {actual_cycles} cycles completed in {elapsed_minutes:.1f} minutes"

        except Exception as e:
            return False, f"Health check error: {str(e)}"

    def generate_oversight_report(self):
        """Generate oversight report"""
        timestamp = datetime.now(timezone.utc)

        # Load monitoring data
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except:
            return self.generate_error_report("Cannot load monitoring data")

        # Calculate metrics
        start_time = datetime.fromisoformat(data["start_time"])
        elapsed_minutes = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

        baseline = data.get("baseline", {})
        measurements = data.get("measurements", [])

        report = {
            "timestamp": timestamp.isoformat(),
            "elapsed_minutes": elapsed_minutes,
            "monitoring_health": self.check_monitoring_health(),
            "progress_summary": self.calculate_progress_summary(baseline, measurements),
            "target_assessment": self.assess_target_progress(measurements),
            "recommendations": self.generate_recommendations(baseline, measurements),
            "next_checkpoints": self.calculate_next_checkpoints(elapsed_minutes, measurements)
        }

        # Save report
        report_file = self.reports_dir / f"oversight_report_{timestamp.strftime('%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def calculate_progress_summary(self, baseline, measurements):
        """Calculate overall progress summary"""
        if not measurements:
            return {"status": "no_measurements", "note": "Waiting for first measurement cycle"}

        latest = measurements[-1]

        # Python progress
        baseline_python = baseline.get("python", {})
        latest_python = latest.get("python", {})

        coverage_change = latest_python.get("coverage_percent", 0) - baseline_python.get("coverage_percent", 0)
        pass_rate_change = latest_python.get("pass_rate", 0) - baseline_python.get("pass_rate", 0)
        collection_error_change = baseline_python.get("error_tests", 0) - latest_python.get("error_tests", 0)

        # Rust progress
        baseline_rust = baseline.get("rust_check", {})
        latest_rust = latest.get("rust_check", {})

        rust_error_change = baseline_rust.get("error_count", 0) - latest_rust.get("error_count", 0)
        rust_warning_change = baseline_rust.get("warning_count", 0) - latest_rust.get("warning_count", 0)

        return {
            "total_cycles": len(measurements),
            "python": {
                "coverage_change": coverage_change,
                "pass_rate_change": pass_rate_change,
                "collection_error_reduction": collection_error_change,
                "current_coverage": latest_python.get("coverage_percent", 0),
                "current_pass_rate": latest_python.get("pass_rate", 0)
            },
            "rust": {
                "error_reduction": rust_error_change,
                "warning_change": rust_warning_change,
                "current_errors": latest_rust.get("error_count", 0),
                "compilation_success": latest_rust.get("compilation_success", False)
            }
        }

    def assess_target_progress(self, measurements):
        """Assess progress toward 100% targets"""
        if not measurements:
            return {"status": "pending", "note": "No measurements available"}

        latest = measurements[-1]
        python = latest.get("python", {})
        rust = latest.get("rust_check", {})

        coverage = python.get("coverage_percent", 0)
        pass_rate = python.get("pass_rate", 0)
        rust_errors = rust.get("error_count", 999)

        # Calculate completion percentages
        coverage_completion = min(coverage, 100)
        pass_rate_completion = min(pass_rate, 100)
        rust_completion = 100 if rust_errors == 0 else max(0, 100 - (rust_errors * 1.5))

        overall_completion = (coverage_completion + pass_rate_completion + rust_completion) / 3

        targets_met = {
            "coverage_100": coverage >= 100,
            "pass_rate_100": pass_rate >= 100,
            "rust_0_errors": rust_errors == 0
        }

        return {
            "overall_completion_percent": overall_completion,
            "individual_completions": {
                "coverage": coverage_completion,
                "pass_rate": pass_rate_completion,
                "rust": rust_completion
            },
            "targets_met": targets_met,
            "all_targets_met": all(targets_met.values()),
            "targets_remaining": sum(1 for met in targets_met.values() if not met)
        }

    def generate_recommendations(self, baseline, measurements):
        """Generate actionable recommendations"""
        recommendations = []

        if not measurements:
            recommendations.append({
                "priority": "high",
                "action": "Wait for first measurement cycle to complete",
                "timeline": "immediate"
            })
            return recommendations

        latest = measurements[-1]
        python = latest.get("python", {})
        rust = latest.get("rust_check", {})

        # Python recommendations
        collection_errors = python.get("error_tests", 0)
        if collection_errors > 0:
            recommendations.append({
                "priority": "critical",
                "action": f"Address {collection_errors} Python collection errors blocking test execution",
                "timeline": "next_cycle",
                "details": "Collection errors prevent all Python tests from running"
            })

        coverage = python.get("coverage_percent", 0)
        if coverage < 50:
            recommendations.append({
                "priority": "high",
                "action": f"Increase Python test coverage from {coverage:.1f}% toward 100%",
                "timeline": "ongoing",
                "details": "Target incremental improvements each cycle"
            })

        # Rust recommendations
        rust_errors = rust.get("error_count", 0)
        if rust_errors > 0:
            recommendations.append({
                "priority": "high",
                "action": f"Resolve {rust_errors} Rust compilation errors",
                "timeline": "next_30_minutes",
                "details": "Focus on tonic gRPC framework compatibility issues"
            })

        # Progress recommendations
        if len(measurements) >= 2:
            prev = measurements[-2]
            curr = measurements[-1]

            # Check for stagnation
            prev_coverage = prev.get("python", {}).get("coverage_percent", 0)
            curr_coverage = curr.get("python", {}).get("coverage_percent", 0)

            if abs(curr_coverage - prev_coverage) < 0.1:
                recommendations.append({
                    "priority": "medium",
                    "action": "Python coverage appears stagnant - investigate test execution issues",
                    "timeline": "next_cycle"
                })

        return recommendations

    def calculate_next_checkpoints(self, elapsed_minutes, measurements):
        """Calculate next monitoring checkpoints"""
        next_cycle_minutes = ((len(measurements)) * 10) + 10
        next_report_minutes = ((elapsed_minutes // 30) + 1) * 30

        return {
            "next_test_cycle": {
                "expected_time_minutes": next_cycle_minutes,
                "time_remaining_minutes": max(0, next_cycle_minutes - elapsed_minutes)
            },
            "next_progress_report": {
                "expected_time_minutes": next_report_minutes,
                "time_remaining_minutes": max(0, next_report_minutes - elapsed_minutes)
            },
            "checkpoints": [
                {"time_minutes": 30, "milestone": "Initial progress assessment"},
                {"time_minutes": 60, "milestone": "Major issue resolution target"},
                {"time_minutes": 120, "milestone": "50% progress checkpoint"},
                {"time_minutes": 240, "milestone": "75% progress checkpoint"},
                {"time_minutes": 480, "milestone": "100% target deadline"}
            ]
        }

    def print_oversight_summary(self, report):
        """Print oversight summary"""
        print("\n" + "=" * 70)
        print("OVERSIGHT MONITOR REPORT")
        print("=" * 70)

        print(f"Time: {datetime.fromisoformat(report['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Elapsed: {report['elapsed_minutes']:.1f} minutes")

        # Health status
        health_status, health_msg = report["monitoring_health"]
        health_icon = "✅" if health_status else "❌"
        print(f"Monitoring Health: {health_icon} {health_msg}")

        # Progress summary
        progress = report["progress_summary"]
        if "python" in progress:
            python = progress["python"]
            rust = progress["rust"]

            print(f"\nPROGRESS (Cycle {progress['total_cycles']}):")
            print(f"  Python Coverage: {python['current_coverage']:.2f}% ({python['coverage_change']:+.2f}%)")
            print(f"  Python Pass Rate: {python['current_pass_rate']:.1f}% ({python['pass_rate_change']:+.1f}%)")
            print(f"  Rust Errors: {rust['current_errors']} ({rust['error_reduction']:+d})")
            print(f"  Rust Compilation: {'✅' if rust['compilation_success'] else '❌'}")

        # Target assessment
        targets = report["target_assessment"]
        if "overall_completion_percent" in targets:
            print(f"\nTARGET PROGRESS: {targets['overall_completion_percent']:.1f}% complete")
            print(f"Targets Remaining: {targets['targets_remaining']}/3")

            if targets["all_targets_met"]:
                print("🎉 ALL TARGETS ACHIEVED! 🎉")

        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in recommendations:
                priority_icon = {"critical": "🚨", "high": "⚠️", "medium": "📋"}.get(rec["priority"], "📝")
                print(f"  {priority_icon} {rec['action']}")

        # Next checkpoints
        checkpoints = report["next_checkpoints"]
        next_cycle = checkpoints["next_test_cycle"]["time_remaining_minutes"]
        print(f"\nNext test cycle in: {next_cycle:.1f} minutes")

        print("=" * 70)

    def run_oversight_monitoring(self):
        """Run continuous oversight monitoring"""
        print("Starting oversight monitoring...")
        print("Reports every 5 minutes. Press Ctrl+C to stop.")

        try:
            while True:
                report = self.generate_oversight_report()
                self.print_oversight_summary(report)

                # Check if all targets are met
                if report.get("target_assessment", {}).get("all_targets_met", False):
                    print("\n🎉 MONITORING COMPLETE - ALL TARGETS ACHIEVED! 🎉")
                    break

                time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\nOversight monitoring stopped")


def main():
    monitor = OversightMonitor()
    monitor.run_oversight_monitoring()


if __name__ == "__main__":
    main()