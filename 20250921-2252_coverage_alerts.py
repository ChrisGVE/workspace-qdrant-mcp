#!/usr/bin/env python3
"""
Real-time Coverage Alert System
Monitors coverage progression and sends immediate alerts for changes.
"""

import json
import time
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import threading


class CoverageAlertSystem:
    """Real-time alert system for coverage monitoring"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "20250921-2252_coverage_monitoring.db"
        self.alert_log_path = self.project_root / "20250921-2252_coverage_alerts.log"
        self.monitoring_active = False
        self.last_coverage = 0.0
        self.setup_alert_system()

    def setup_alert_system(self):
        """Initialize alert system"""
        # Ensure database exists
        if not self.db_path.exists():
            self.create_monitoring_database()

        # Clear old alerts log
        with open(self.alert_log_path, 'w') as f:
            f.write(f"Coverage Alert System Started: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

    def create_monitoring_database(self):
        """Create monitoring database if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                coverage_before REAL,
                coverage_after REAL,
                change_amount REAL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                milestone_type TEXT NOT NULL,
                coverage_percent REAL,
                description TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def get_current_coverage(self) -> Dict:
        """Get current coverage percentage"""
        try:
            result = subprocess.run([
                'uv', 'run', 'pytest', '--cov=src', '--cov-report=json',
                '--tb=short', '-q'
            ], capture_output=True, text=True, cwd=self.project_root)

            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    coverage_data = json.load(f)

                totals = coverage_data.get('totals', {})
                return {
                    'coverage': totals.get('percent_covered', 0.0),
                    'covered_lines': totals.get('covered_lines', 0),
                    'missing_lines': totals.get('missing_lines', 0),
                    'total_lines': totals.get('num_statements', 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'No coverage data'}

        except Exception as e:
            return {'error': f'Coverage failed: {e}'}

    def log_alert(self, alert_type: str, severity: str, message: str,
                  coverage_before: float = 0, coverage_after: float = 0):
        """Log alert to database and file"""
        timestamp = datetime.now().isoformat()
        change_amount = coverage_after - coverage_before

        # Database log
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO coverage_alerts
            (timestamp, alert_type, severity, coverage_before, coverage_after,
             change_amount, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, alert_type, severity, coverage_before,
              coverage_after, change_amount, message))

        conn.commit()
        conn.close()

        # File log
        with open(self.alert_log_path, 'a') as f:
            f.write(f"[{timestamp}] {severity} - {alert_type}\n")
            f.write(f"Message: {message}\n")
            if coverage_before or coverage_after:
                f.write(f"Coverage: {coverage_before:.2f}% → {coverage_after:.2f}% ({change_amount:+.2f}%)\n")
            f.write("-" * 40 + "\n\n")

        # Console alert
        alert_emoji = {
            'CRITICAL': '🚨',
            'HIGH': '⚠️',
            'MEDIUM': '🔔',
            'LOW': 'ℹ️'
        }

        print(f"\n{alert_emoji.get(severity, '🔔')} {severity} ALERT: {alert_type}")
        print(f"📝 {message}")
        if coverage_before or coverage_after:
            print(f"📊 Coverage: {coverage_before:.2f}% → {coverage_after:.2f}% ({change_amount:+.2f}%)")
        print(f"⏰ Time: {timestamp}")
        print("-" * 50)

    def check_milestone_progress(self, current_coverage: float):
        """Check for milestone achievements"""
        milestones = [25, 50, 75, 90, 95, 99, 100]

        for milestone in milestones:
            if self.last_coverage < milestone <= current_coverage:
                self.log_milestone(milestone, current_coverage)
                self.log_alert(
                    'MILESTONE_ACHIEVED',
                    'HIGH',
                    f"🎯 {milestone}% coverage milestone achieved!",
                    self.last_coverage,
                    current_coverage
                )

    def log_milestone(self, milestone: int, coverage: float):
        """Log milestone achievement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO milestones (timestamp, milestone_type, coverage_percent, description)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            f"{milestone}%",
            coverage,
            f"Achieved {milestone}% coverage milestone"
        ))

        conn.commit()
        conn.close()

    def check_coverage_regression(self, current_coverage: float):
        """Check for coverage regression"""
        if self.last_coverage > 0:
            regression_threshold = 0.5  # Alert if coverage drops by 0.5%

            if current_coverage < self.last_coverage - regression_threshold:
                self.log_alert(
                    'COVERAGE_REGRESSION',
                    'CRITICAL',
                    f"Coverage dropped significantly! Immediate attention required.",
                    self.last_coverage,
                    current_coverage
                )
            elif current_coverage < self.last_coverage:
                self.log_alert(
                    'COVERAGE_DECREASE',
                    'MEDIUM',
                    f"Coverage decreased slightly. Monitor for trends.",
                    self.last_coverage,
                    current_coverage
                )

    def check_coverage_improvement(self, current_coverage: float):
        """Check for coverage improvements"""
        if self.last_coverage > 0:
            improvement_threshold = 1.0  # Alert for 1%+ improvement

            if current_coverage > self.last_coverage + improvement_threshold:
                self.log_alert(
                    'SIGNIFICANT_IMPROVEMENT',
                    'HIGH',
                    f"Excellent progress! Significant coverage improvement detected.",
                    self.last_coverage,
                    current_coverage
                )
            elif current_coverage > self.last_coverage:
                self.log_alert(
                    'COVERAGE_IMPROVEMENT',
                    'LOW',
                    f"Coverage improved. Keep up the good work!",
                    self.last_coverage,
                    current_coverage
                )

    def check_stagnation(self):
        """Check for coverage stagnation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for no improvement in last 30 minutes
        cutoff_time = datetime.now() - timedelta(minutes=30)
        cursor.execute('''
            SELECT COUNT(*) FROM coverage_alerts
            WHERE alert_type IN ('COVERAGE_IMPROVEMENT', 'SIGNIFICANT_IMPROVEMENT')
            AND datetime(timestamp) > datetime(?)
        ''', (cutoff_time.isoformat(),))

        recent_improvements = cursor.fetchone()[0]

        if recent_improvements == 0 and self.last_coverage < 100:
            self.log_alert(
                'COVERAGE_STAGNATION',
                'MEDIUM',
                f"No coverage improvement detected in 30 minutes. Current: {self.last_coverage:.2f}%"
            )

        conn.close()

    def generate_progress_summary(self) -> str:
        """Generate current progress summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent alerts
        cursor.execute('''
            SELECT alert_type, severity, timestamp, message
            FROM coverage_alerts
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
        ''')

        recent_alerts = cursor.fetchall()

        # Get milestones achieved
        cursor.execute('''
            SELECT milestone_type, timestamp
            FROM milestones
            ORDER BY timestamp DESC
            LIMIT 5
        ''')

        milestones = cursor.fetchall()

        summary = f"""
🎯 COVERAGE MONITORING SUMMARY
=============================
Current Coverage: {self.last_coverage:.2f}%
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RECENT ACTIVITY (1 hour):
"""

        if recent_alerts:
            for alert in recent_alerts:
                summary += f"• {alert[1]}: {alert[3]}\n"
        else:
            summary += "• No recent alerts\n"

        summary += "\nMILESTONES ACHIEVED:\n"
        if milestones:
            for milestone in milestones:
                summary += f"🏆 {milestone[0]} - {milestone[1]}\n"
        else:
            summary += "• No milestones achieved yet\n"

        conn.close()
        return summary

    def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous coverage monitoring with alerts"""
        self.monitoring_active = True

        def monitoring_loop():
            self.log_alert(
                'MONITORING_STARTED',
                'LOW',
                f"Coverage monitoring started with {interval_minutes}-minute intervals"
            )

            while self.monitoring_active:
                # Get current coverage
                coverage_data = self.get_current_coverage()

                if 'error' not in coverage_data:
                    current_coverage = coverage_data['coverage']

                    # Check for various alert conditions
                    self.check_coverage_regression(current_coverage)
                    self.check_coverage_improvement(current_coverage)
                    self.check_milestone_progress(current_coverage)

                    # Check for stagnation every 6 cycles (30 minutes)
                    if int(time.time()) % (interval_minutes * 6 * 60) < 60:
                        self.check_stagnation()

                    self.last_coverage = current_coverage

                    # Log current status
                    print(f"📊 Coverage Monitor: {current_coverage:.2f}% ({datetime.now().strftime('%H:%M:%S')})")

                else:
                    self.log_alert(
                        'MONITORING_ERROR',
                        'MEDIUM',
                        f"Failed to get coverage data: {coverage_data['error']}"
                    )

                # Wait for next cycle
                time.sleep(interval_minutes * 60)

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

        return monitor_thread

    def stop_monitoring(self):
        """Stop monitoring and log shutdown"""
        self.monitoring_active = False
        self.log_alert(
            'MONITORING_STOPPED',
            'LOW',
            "Coverage monitoring stopped"
        )

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get alert history for specified hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = datetime.now() - timedelta(hours=hours)
        cursor.execute('''
            SELECT timestamp, alert_type, severity, coverage_before,
                   coverage_after, change_amount, message
            FROM coverage_alerts
            WHERE datetime(timestamp) > datetime(?)
            ORDER BY timestamp DESC
        ''', (cutoff_time.isoformat(),))

        alerts = cursor.fetchall()
        conn.close()

        return [
            {
                'timestamp': alert[0],
                'type': alert[1],
                'severity': alert[2],
                'coverage_before': alert[3],
                'coverage_after': alert[4],
                'change': alert[5],
                'message': alert[6]
            }
            for alert in alerts
        ]


def main():
    """Main alert system execution"""
    alert_system = CoverageAlertSystem()

    print("🚨 Coverage Alert System Starting...")
    print("Target: 100% coverage with real-time alerts")
    print("Monitoring intervals: Every 5 minutes")
    print("=" * 60)

    # Start monitoring
    monitor_thread = alert_system.start_monitoring(interval_minutes=5)

    try:
        # Keep main thread alive and display periodic summaries
        while True:
            time.sleep(900)  # 15 minutes

            print("\n" + "="*60)
            summary = alert_system.generate_progress_summary()
            print(summary)
            print("="*60)

    except KeyboardInterrupt:
        print("\n🛑 Stopping coverage alert system...")
        alert_system.stop_monitoring()


if __name__ == "__main__":
    main()