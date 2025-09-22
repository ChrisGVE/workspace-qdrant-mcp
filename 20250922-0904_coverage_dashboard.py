#!/usr/bin/env python3
"""
Coverage Dashboard - Real-time view of coverage monitoring data
"""

import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

def load_coverage_data(data_file: Path):
    """Load current coverage data"""
    try:
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def format_trend(history, key='coverage'):
    """Format trend indicator"""
    if len(history) < 2:
        return "➡️"

    current = history[-1].get(key)
    previous = history[-2].get(key)

    if current is None or previous is None:
        return "❓"

    if current > previous:
        return f"📈 +{current - previous:.1f}"
    elif current < previous:
        return f"📉 {current - previous:.1f}"
    else:
        return "➡️ 0.0"

def display_coverage_summary(data):
    """Display comprehensive coverage summary"""
    if not data:
        print("❌ No coverage data available")
        return

    python_history = data.get('python_coverage', [])
    rust_history = data.get('rust_coverage', [])
    import_history = data.get('import_errors', [])
    last_updated = data.get('last_updated', 'Unknown')

    print("\n" + "="*100)
    print(f"🎯 COVERAGE PERFORMANCE DASHBOARD")
    print(f"📅 Last Updated: {last_updated}")
    print("="*100)

    # Current Status
    if python_history:
        latest_python = python_history[-1]
        python_cov = latest_python.get('coverage')
        python_errors = latest_python.get('import_errors', 0)

        if python_cov is not None:
            status = "🟢 TARGET MET" if python_cov >= 100 else "🟡 APPROACHING" if python_cov >= 95 else "🔴 IN PROGRESS"
            print(f"🐍 Python Coverage:  {status} {python_cov:.1f}% | Trend: {format_trend(python_history)}")
        else:
            print(f"🐍 Python Coverage:  ❌ UNABLE TO MEASURE")

        print(f"🔧 Import Errors:    {'🟢 CLEAN' if python_errors == 0 else f'🔴 {python_errors} ERRORS'}")

    if rust_history:
        latest_rust = rust_history[-1]
        rust_cov = latest_rust.get('coverage')

        if rust_cov is not None:
            status = "🟢 TARGET MET" if rust_cov >= 100 else "🟡 APPROACHING" if rust_cov >= 95 else "🔴 IN PROGRESS"
            print(f"🦀 Rust Coverage:    {status} {rust_cov:.1f}% | Trend: {format_trend(rust_history)}")
        else:
            print(f"🦀 Rust Coverage:    ❌ UNABLE TO MEASURE")

    # Progress Timeline (last 10 measurements)
    print("\n📊 RECENT PROGRESS TIMELINE")
    print("-" * 100)

    if python_history:
        print("🐍 Python Coverage History (last 10):")
        for entry in python_history[-10:]:
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            coverage = entry.get('coverage')
            errors = entry.get('import_errors', 0)

            if coverage is not None:
                print(f"   {timestamp} | Coverage: {coverage:5.1f}% | Errors: {errors:2d}")
            else:
                print(f"   {timestamp} | Coverage: ERROR | Errors: {errors:2d}")

    if rust_history:
        print("\n🦀 Rust Coverage History (last 10):")
        for entry in rust_history[-10:]:
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            coverage = entry.get('coverage')

            if coverage is not None:
                print(f"   {timestamp} | Coverage: {coverage:5.1f}%")
            else:
                print(f"   {timestamp} | Coverage: ERROR")

    # Import Error Details
    if import_history:
        latest_errors = import_history[-1]
        error_details = latest_errors.get('details', [])

        if error_details and latest_errors.get('count', 0) > 0:
            print(f"\n🔧 CURRENT IMPORT ERRORS ({len(error_details)} total):")
            print("-" * 100)
            for i, error in enumerate(error_details[:20], 1):  # Show up to 20 errors
                print(f"   {i:2d}. {error}")
            if len(error_details) > 20:
                print(f"   ... and {len(error_details) - 20} more errors")

    # Targets & Alerts
    print(f"\n🎯 TARGETS & ALERTS")
    print("-" * 100)
    print(f"🐍 Python Target: 100.0% | Alert Threshold: 95.0%")
    print(f"🦀 Rust Target:   100.0% | Alert Threshold: 95.0%")

    # Calculate time to target (rough estimate)
    if python_history and len(python_history) >= 5:
        recent_python = [h.get('coverage') for h in python_history[-5:] if h.get('coverage') is not None]
        if len(recent_python) >= 2:
            trend_rate = (recent_python[-1] - recent_python[0]) / len(recent_python)
            if trend_rate > 0:
                remaining = 100.0 - recent_python[-1]
                eta_checks = remaining / trend_rate
                eta_hours = (eta_checks * 2) / 60  # 2 minutes per check
                print(f"📈 Python ETA to 100%: ~{eta_hours:.1f} hours (based on recent trend: +{trend_rate:.2f}%/check)")

    print("="*100)

def watch_mode():
    """Watch mode - continuous dashboard updates"""
    data_file = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/coverage_data.json")

    print("👁️  Starting dashboard watch mode (Ctrl+C to exit)")
    print("🔄 Updates every 30 seconds...")

    try:
        while True:
            # Clear screen (ANSI escape codes)
            print("\033[2J\033[H", end="")

            data = load_coverage_data(data_file)
            display_coverage_summary(data)

            print(f"\n⏰ Next update in 30 seconds... (Last check: {datetime.now().strftime('%H:%M:%S')})")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n👋 Dashboard watch stopped")

def main():
    """Main dashboard function"""
    data_file = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/coverage_data.json")

    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        watch_mode()
    else:
        # Single shot display
        data = load_coverage_data(data_file)
        display_coverage_summary(data)
        print(f"\n💡 Use '{sys.argv[0]} --watch' for continuous updates")

if __name__ == "__main__":
    main()