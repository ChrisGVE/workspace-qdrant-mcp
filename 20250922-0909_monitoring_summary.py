#!/usr/bin/env python3
"""
Performance Monitoring Summary - Emergency Coverage Tracking
Provides real-time insights and alerts for 100% coverage targets
"""

import os
import time
import subprocess
from datetime import datetime
from pathlib import Path

class EmergencyMonitor:
    def __init__(self):
        self.project_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
        self.alert_file = self.project_root / "coverage_alerts.log"
        self.progress_file = self.project_root / "coverage_progress.log"
        self.start_time = datetime.now()

    def check_monitoring_status(self):
        """Check if monitoring is active and get latest status"""
        try:
            # Check if monitoring process is running
            result = subprocess.run(['pgrep', '-f', 'coverage_alerts.sh'],
                                  capture_output=True, text=True)
            monitoring_active = len(result.stdout.strip()) > 0

            return monitoring_active
        except:
            return False

    def get_latest_metrics(self):
        """Extract latest metrics from log files"""
        metrics = {
            'python_coverage': None,
            'rust_coverage': None,
            'import_errors': 0,
            'last_check': None,
            'targets_achieved': False
        }

        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    lines = f.readlines()

                for line in reversed(lines):
                    # Parse Python coverage
                    if 'Python Coverage:' in line and '%' in line:
                        try:
                            pct = line.split('Python Coverage:')[1].strip().split('%')[0]
                            metrics['python_coverage'] = float(pct)
                        except:
                            pass

                    # Parse Rust coverage
                    if 'Rust Tests:' in line and 'estimated' in line:
                        try:
                            if 'PASSING' in line:
                                metrics['rust_coverage'] = 85.0
                            elif 'FAILING' in line:
                                metrics['rust_coverage'] = 40.0
                        except:
                            pass

                    # Parse import errors
                    if 'Import errors blocking coverage:' in line:
                        try:
                            errors = int(line.split('blocking coverage:')[1].strip())
                            metrics['import_errors'] = errors
                        except:
                            pass

                    # Parse timestamp
                    if 'Coverage Check #' in line:
                        try:
                            timestamp_str = line.split(' - ')[1].strip()
                            metrics['last_check'] = timestamp_str
                            break  # We found the latest check
                        except:
                            pass
        except Exception as e:
            print(f"Warning: Could not parse logs: {e}")

        # Check for target achievement
        if self.alert_file.exists():
            try:
                with open(self.alert_file, 'r') as f:
                    content = f.read()
                    if "ALL TARGETS ACHIEVED" in content or "100% TARGET ACHIEVED" in content:
                        metrics['targets_achieved'] = True
            except:
                pass

        return metrics

    def display_emergency_dashboard(self):
        """Display emergency monitoring dashboard"""
        monitoring_active = self.check_monitoring_status()
        metrics = self.get_latest_metrics()

        current_time = datetime.now().strftime("%H:%M:%S")

        print("\n" + "🚨" * 40)
        print(f"⚡ EMERGENCY COVERAGE MONITORING - {current_time}")
        print("🚨" * 40)

        # Monitoring Status
        status_icon = "🟢" if monitoring_active else "🔴"
        print(f"📡 Monitoring Status: {status_icon} {'ACTIVE' if monitoring_active else 'STOPPED'}")

        if metrics['last_check']:
            print(f"🕐 Last Check: {metrics['last_check']}")

        print("")

        # Coverage Status
        print("🎯 COVERAGE TARGETS (100% Required):")
        print("-" * 50)

        # Python Status
        if metrics['python_coverage'] is not None:
            python_status = "🟢 ACHIEVED" if metrics['python_coverage'] >= 100 else \
                          "🟡 CLOSE" if metrics['python_coverage'] >= 95 else "🔴 WORKING"
            print(f"🐍 Python: {python_status} {metrics['python_coverage']:.1f}%")
        else:
            print(f"🐍 Python: ❌ BLOCKED BY IMPORT ERRORS")

        # Import Errors
        if metrics['import_errors'] > 0:
            print(f"🔧 Import Errors: 🔴 {metrics['import_errors']} BLOCKING")
        else:
            print(f"🔧 Import Errors: 🟢 CLEAN")

        # Rust Status
        if metrics['rust_coverage'] is not None:
            rust_status = "🟢 ACHIEVED" if metrics['rust_coverage'] >= 100 else \
                         "🟡 CLOSE" if metrics['rust_coverage'] >= 95 else "🔴 WORKING"
            print(f"🦀 Rust: {rust_status} {metrics['rust_coverage']:.1f}%")
        else:
            print(f"🦀 Rust: ❌ UNABLE TO MEASURE")

        print("")

        # Critical Alerts
        if metrics['targets_achieved']:
            print("🎉 ALERT: ALL TARGETS ACHIEVED! 100% COVERAGE REACHED!")
        elif metrics['python_coverage'] and metrics['python_coverage'] >= 100:
            print("🎯 ALERT: PYTHON 100% TARGET ACHIEVED!")
        elif metrics['rust_coverage'] and metrics['rust_coverage'] >= 100:
            print("🎯 ALERT: RUST 100% TARGET ACHIEVED!")
        elif metrics['import_errors'] > 0:
            print(f"🚨 CRITICAL: {metrics['import_errors']} IMPORT ERRORS BLOCKING PROGRESS")
        else:
            print("🚧 STATUS: WORK IN PROGRESS - MONITORING ACTIVE")

        print("🚨" * 40)

        # Instructions
        print("💡 MONITORING COMMANDS:")
        print(f"   Status: {self.project_root}/20250922-0907_coverage_status.sh")
        print(f"   Logs:   tail -f {self.progress_file}")
        print(f"   Alerts: tail -f {self.alert_file}")

        return metrics

    def check_alerts(self, metrics):
        """Check for critical alerts"""
        alerts = []

        if metrics['targets_achieved']:
            alerts.append("🎉 ALL TARGETS ACHIEVED!")

        if metrics['python_coverage'] and metrics['python_coverage'] >= 100:
            alerts.append("🎯 Python 100% achieved!")

        if metrics['rust_coverage'] and metrics['rust_coverage'] >= 100:
            alerts.append("🎯 Rust 100% achieved!")

        if metrics['import_errors'] > 50:
            alerts.append(f"🚨 Critical: {metrics['import_errors']} import errors!")

        return alerts

def main():
    """Main monitoring summary"""
    monitor = EmergencyMonitor()
    metrics = monitor.display_emergency_dashboard()

    # Check for immediate alerts
    alerts = monitor.check_alerts(metrics)
    if alerts:
        print("\n⚡ IMMEDIATE ALERTS:")
        for alert in alerts:
            print(f"   {alert}")

    return metrics['targets_achieved']

if __name__ == "__main__":
    targets_achieved = main()
    exit(0 if targets_achieved else 1)