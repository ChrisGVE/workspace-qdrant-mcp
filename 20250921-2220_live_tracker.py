#!/usr/bin/env python3
"""
Live Progress Tracker
====================

Provides continuous updates on monitoring progress every 2 minutes.
Shows detailed progress toward targets and identifies regressions.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def load_monitoring_data():
    """Load current monitoring data"""
    data_file = Path("20250921-2216_monitoring_data.json")
    try:
        with open(data_file, 'r') as f:
            return json.load(f)
    except:
        return None


def print_detailed_status():
    """Print comprehensive status update"""
    data = load_monitoring_data()
    if not data:
        print("❌ No monitoring data available")
        return

    # Calculate elapsed time
    start_time = datetime.fromisoformat(data["start_time"])
    elapsed = datetime.now(timezone.utc) - start_time
    elapsed_minutes = elapsed.total_seconds() / 60

    print("\n" + "=" * 60)
    print(f"LIVE PROGRESS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print(f"Monitoring Duration: {elapsed_minutes:.1f} minutes")
    print(f"Current Cycle: {data['current_cycle']}")
    print(f"Total Measurements: {len(data['measurements'])}")

    # Show baseline
    baseline = data.get("baseline", {})
    if baseline:
        baseline_python = baseline.get("python", {})
        baseline_rust = baseline.get("rust_check", {})

        print(f"\nBASELINE:")
        print(f"  Python Coverage: {baseline_python.get('coverage_percent', 0):.2f}%")
        print(f"  Python Pass Rate: {baseline_python.get('pass_rate', 0):.1f}%")
        print(f"  Python Collection Errors: {baseline_python.get('error_tests', 0)}")
        print(f"  Rust Errors: {baseline_rust.get('error_count', 0)}")
        print(f"  Rust Warnings: {baseline_rust.get('warning_count', 0)}")

    # Show latest if available
    if data['measurements']:
        latest = data['measurements'][-1]
        latest_python = latest.get("python", {})
        latest_rust = latest.get("rust_check", {})

        print(f"\nLATEST (Cycle {latest['cycle']}):")

        # Python metrics
        coverage = latest_python.get('coverage_percent', 0)
        pass_rate = latest_python.get('pass_rate', 0)
        collection_errors = latest_python.get('error_tests', 0)

        print(f"  Python Coverage: {coverage:.2f}% (Target: 100%)")
        print(f"  Python Pass Rate: {pass_rate:.1f}% (Target: 100%)")
        print(f"  Python Collection Errors: {collection_errors}")

        # Rust metrics
        rust_errors = latest_rust.get('error_count', 0)
        rust_warnings = latest_rust.get('warning_count', 0)
        rust_success = latest_rust.get('compilation_success', False)

        print(f"  Rust Errors: {rust_errors} (Target: 0)")
        print(f"  Rust Warnings: {rust_warnings}")
        print(f"  Rust Compilation: {'✅ SUCCESS' if rust_success else '❌ FAILED'}")

        # Calculate progress since baseline
        if baseline:
            baseline_python = baseline.get("python", {})
            baseline_rust = baseline.get("rust_check", {})

            coverage_change = coverage - baseline_python.get('coverage_percent', 0)
            pass_rate_change = pass_rate - baseline_python.get('pass_rate', 0)
            collection_error_change = baseline_python.get('error_tests', 0) - collection_errors
            rust_error_change = baseline_rust.get('error_count', 0) - rust_errors
            rust_warning_change = baseline_rust.get('warning_count', 0) - rust_warnings

            print(f"\nPROGRESS SINCE BASELINE:")
            print(f"  Coverage: {coverage_change:+.2f}%")
            print(f"  Pass Rate: {pass_rate_change:+.1f}%")
            print(f"  Collection Errors: {collection_error_change:+d}")
            print(f"  Rust Errors: {rust_error_change:+d}")
            print(f"  Rust Warnings: {rust_warning_change:+d}")

            # Target progress
            coverage_target_progress = (coverage / 100) * 100
            rust_target_progress = 100 if rust_errors == 0 else max(0, 100 - (rust_errors * 2))

            print(f"\nTARGET PROGRESS:")
            print(f"  Coverage: {coverage_target_progress:.1f}% of 100%")
            print(f"  Rust Success: {rust_target_progress:.1f}% of 100%")

    # Show trend if multiple measurements
    if len(data['measurements']) >= 2:
        prev = data['measurements'][-2]
        curr = data['measurements'][-1]

        prev_python = prev.get("python", {})
        curr_python = curr.get("python", {})
        prev_rust = prev.get("rust_check", {})
        curr_rust = curr.get("rust_check", {})

        print(f"\nTREND (Cycle {prev['cycle']} → {curr['cycle']}):")

        coverage_trend = curr_python.get('coverage_percent', 0) - prev_python.get('coverage_percent', 0)
        pass_rate_trend = curr_python.get('pass_rate', 0) - prev_python.get('pass_rate', 0)
        rust_error_trend = prev_rust.get('error_count', 0) - curr_rust.get('error_count', 0)

        print(f"  Coverage: {coverage_trend:+.2f}%")
        print(f"  Pass Rate: {pass_rate_trend:+.1f}%")
        print(f"  Rust Errors: {rust_error_trend:+d}")

        # Detect regressions
        regressions = []
        if coverage_trend < -0.1:
            regressions.append("Coverage decreased")
        if pass_rate_trend < -1:
            regressions.append("Pass rate decreased")
        if rust_error_trend < 0:
            regressions.append("Rust errors increased")

        if regressions:
            print(f"  ⚠️ REGRESSIONS: {', '.join(regressions)}")
        else:
            print(f"  ✅ No regressions detected")

    # Expected next cycle time
    if elapsed_minutes < 10:
        next_cycle_in = 10 - elapsed_minutes
        print(f"\nNext cycle expected in: {next_cycle_in:.1f} minutes")
    elif data['current_cycle'] > 0:
        cycles_elapsed = elapsed_minutes / 10
        next_cycle_expected = (int(cycles_elapsed) + 1) * 10
        next_cycle_in = next_cycle_expected - elapsed_minutes
        print(f"\nNext cycle expected in: {next_cycle_in:.1f} minutes")

    # Success criteria check
    if data['measurements']:
        latest = data['measurements'][-1]
        python = latest.get("python", {})
        rust = latest.get("rust_check", {})

        coverage_target_met = python.get("coverage_percent", 0) >= 100
        pass_rate_target_met = python.get("pass_rate", 0) >= 100
        rust_target_met = rust.get("error_count", 999) == 0

        print(f"\nSUCCESS CRITERIA STATUS:")
        print(f"  ✅ Coverage 100%: {'ACHIEVED' if coverage_target_met else 'PENDING'}")
        print(f"  ✅ Pass Rate 100%: {'ACHIEVED' if pass_rate_target_met else 'PENDING'}")
        print(f"  ✅ Rust 0 Errors: {'ACHIEVED' if rust_target_met else 'PENDING'}")

        if coverage_target_met and pass_rate_target_met and rust_target_met:
            print(f"\n🎉 ALL TARGETS ACHIEVED! MONITORING COMPLETE! 🎉")
        else:
            pending_count = sum([not coverage_target_met, not pass_rate_target_met, not rust_target_met])
            print(f"\n📊 {pending_count}/3 targets remaining")

    print("=" * 60)


def continuous_live_tracking():
    """Run continuous live tracking"""
    print("Starting live progress tracking...")
    print("Updates every 2 minutes. Press Ctrl+C to stop.")

    try:
        while True:
            print_detailed_status()
            time.sleep(120)  # 2 minutes
    except KeyboardInterrupt:
        print("\nLive tracking stopped")


if __name__ == "__main__":
    continuous_live_tracking()