#!/usr/bin/env python3
"""
EMERGENCY COVERAGE DASHBOARD
Real-time display of coverage progression and alerts
"""

import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

class EmergencyDashboard:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.monitoring_db = self.project_root / "20250921-2234_emergency_monitoring.db"
        self.report_file = self.project_root / "20250921-2234_emergency_coverage_report.json"

    def get_latest_metrics(self) -> Dict:
        """Get the latest metrics from database"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.execute(
                "SELECT * FROM coverage_metrics ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'timestamp': row[0],
                    'total_tests': row[1],
                    'executable_tests': row[2],
                    'passing_tests': row[3],
                    'coverage_percent': row[4],
                    'lines_covered': row[5],
                    'total_lines': row[6],
                    'import_errors': row[7],
                    'rust_tests': row[8],
                    'rust_passing': row[9],
                    'coverage_delta': row[10]
                }
        except Exception:
            pass

        return {}

    def get_trend_data(self, hours: int = 1) -> List[Dict]:
        """Get trend data for the last N hours"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.execute(f"""
                SELECT * FROM coverage_metrics
                WHERE timestamp > datetime('now', '-{hours} hours')
                ORDER BY timestamp
            """)
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'timestamp': row[0],
                    'coverage_percent': row[4],
                    'executable_tests': row[2],
                    'passing_tests': row[3],
                    'import_errors': row[7]
                } for row in rows
            ]
        except Exception:
            return []

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'timestamp': row[0],
                    'type': row[1],
                    'message': row[2]
                } for row in rows
            ]
        except Exception:
            return []

    def calculate_velocity(self) -> Dict:
        """Calculate coverage velocity and projections"""
        trend_data = self.get_trend_data(2)  # Last 2 hours

        if len(trend_data) < 2:
            return {
                'coverage_velocity': 0.0,
                'test_velocity': 0.0,
                'projected_100_percent': "Unknown"
            }

        # Calculate coverage velocity (% per hour)
        latest = trend_data[-1]
        earliest = trend_data[0]

        time_diff = (datetime.fromisoformat(latest['timestamp']) -
                    datetime.fromisoformat(earliest['timestamp'])).total_seconds() / 3600

        if time_diff > 0:
            coverage_velocity = (latest['coverage_percent'] - earliest['coverage_percent']) / time_diff
            test_velocity = (latest['executable_tests'] - earliest['executable_tests']) / time_diff

            # Project when we might reach 100%
            if coverage_velocity > 0:
                remaining_coverage = 100 - latest['coverage_percent']
                hours_to_100 = remaining_coverage / coverage_velocity
                projected_time = datetime.now() + timedelta(hours=hours_to_100)
                projected_100_percent = projected_time.strftime("%Y-%m-%d %H:%M")
            else:
                projected_100_percent = "No positive trend"
        else:
            coverage_velocity = 0.0
            test_velocity = 0.0
            projected_100_percent = "Insufficient data"

        return {
            'coverage_velocity': coverage_velocity,
            'test_velocity': test_velocity,
            'projected_100_percent': projected_100_percent
        }

    def display_dashboard(self):
        """Display the live dashboard"""
        os.system('clear')  # Clear screen

        print("🚨" * 20)
        print("    EMERGENCY COVERAGE MONITORING DASHBOARD")
        print("🚨" * 20)
        print()

        # Latest metrics
        metrics = self.get_latest_metrics()
        if not metrics:
            print("❌ No metrics available yet. Monitor may be starting...")
            return

        print(f"📊 CURRENT STATUS - {metrics['timestamp']}")
        print("=" * 60)

        # Coverage status with visual bar
        coverage = metrics['coverage_percent']
        bar_length = 30
        filled_length = int(coverage / 100 * bar_length)
        bar = "█" * filled_length + "▒" * (bar_length - filled_length)

        print(f"📈 COVERAGE: {coverage:.2f}% (Δ {metrics['coverage_delta']:+.2f}%)")
        print(f"   [{bar}] {metrics['lines_covered']}/{metrics['total_lines']} lines")
        print()

        # Test execution status
        test_success_rate = 0
        if metrics['executable_tests'] > 0:
            test_success_rate = (metrics['passing_tests'] / metrics['executable_tests']) * 100

        print(f"🧪 TESTS:")
        print(f"   Total Discovered: {metrics['total_tests']}")
        print(f"   Executable: {metrics['executable_tests']} (⚠️ {metrics['import_errors']} import errors)")
        print(f"   Passing: {metrics['passing_tests']} ({test_success_rate:.1f}% success rate)")
        print(f"   Rust Tests: {metrics['rust_passing']}/{metrics['rust_tests']} passing")
        print()

        # Velocity and projections
        velocity = self.calculate_velocity()
        print(f"🚀 VELOCITY ANALYSIS:")
        print(f"   Coverage Velocity: {velocity['coverage_velocity']:+.2f}% per hour")
        print(f"   Test Discovery Rate: {velocity['test_velocity']:+.1f} tests per hour")
        print(f"   Projected 100%: {velocity['projected_100_percent']}")
        print()

        # Recent alerts
        alerts = self.get_recent_alerts(5)
        if alerts:
            print("🚨 RECENT ALERTS:")
            for alert in alerts[:3]:
                timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
                print(f"   {timestamp} [{alert['type']}] {alert['message']}")
        else:
            print("🔇 No alerts yet")

        print()

        # Trend visualization
        trend_data = self.get_trend_data(1)
        if len(trend_data) >= 3:
            print("📈 COVERAGE TREND (Last Hour):")
            trend_line = ""
            for data in trend_data[-10:]:  # Last 10 points
                if data['coverage_percent'] > 20:
                    trend_line += "█"
                elif data['coverage_percent'] > 10:
                    trend_line += "▓"
                elif data['coverage_percent'] > 5:
                    trend_line += "▒"
                else:
                    trend_line += "░"
            print(f"   {trend_line}")

        print()
        print("=" * 60)
        print(f"⏰ Last Update: {datetime.now().strftime('%H:%M:%S')} | Next: {(datetime.now() + timedelta(seconds=10)).strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")

    def run_dashboard(self):
        """Run the live dashboard with updates every 10 seconds"""
        try:
            while True:
                self.display_dashboard()
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")

if __name__ == "__main__":
    dashboard = EmergencyDashboard()
    dashboard.run_dashboard()