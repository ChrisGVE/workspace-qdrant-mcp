#!/usr/bin/env python3
"""
EMERGENCY COVERAGE MONITORING SYSTEM
Continuous monitoring and forcing coverage progression despite obstacles
"""

import os
import sys
import time
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
import threading
from typing import Dict, List, Any, Optional

class EmergencyCoverageMonitor:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.monitoring_db = self.project_root / "20250921-2234_emergency_monitoring.db"
        self.report_file = self.project_root / "20250921-2234_emergency_coverage_report.json"
        self.alert_log = self.project_root / "20250921-2234_emergency_alerts.log"
        self.init_database()
        self.monitoring_active = True

    def init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.monitoring_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS coverage_metrics (
                timestamp TEXT PRIMARY KEY,
                total_tests INTEGER,
                executable_tests INTEGER,
                passing_tests INTEGER,
                coverage_percent REAL,
                lines_covered INTEGER,
                total_lines INTEGER,
                import_errors INTEGER,
                rust_tests INTEGER,
                rust_passing INTEGER,
                coverage_delta REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                timestamp TEXT PRIMARY KEY,
                alert_type TEXT,
                message TEXT,
                metrics TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_alert(self, alert_type: str, message: str, metrics: Dict = None):
        """Log an alert with timestamp"""
        timestamp = datetime.now().isoformat()
        alert_msg = f"[{timestamp}] {alert_type}: {message}"

        # Log to file
        with open(self.alert_log, "a") as f:
            f.write(alert_msg + "\n")

        # Log to database
        conn = sqlite3.connect(self.monitoring_db)
        conn.execute(
            "INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?)",
            (timestamp, alert_type, message, json.dumps(metrics or {}))
        )
        conn.commit()
        conn.close()

        # Print to console
        print(f"🚨 ALERT: {alert_msg}")

    def discover_tests(self) -> Dict[str, Any]:
        """Discover all tests and categorize by execution status"""
        os.chdir(self.project_root)

        # Python test discovery
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--collect-only", "-q"
            ], capture_output=True, text=True, timeout=60)

            total_tests = 0
            import_errors = 0

            for line in result.stdout.split('\n'):
                if 'collected' in line and 'items' in line:
                    try:
                        total_tests = int(line.split()[0])
                    except:
                        pass

            for line in result.stderr.split('\n'):
                if 'ERROR' in line and ('import' in line or 'ModuleNotFoundError' in line):
                    import_errors += 1

        except Exception as e:
            total_tests = 0
            import_errors = 100  # Assume high error rate if discovery fails

        # Try to run a subset of tests to check executability
        executable_tests = 0
        passing_tests = 0

        try:
            # Try running unit tests first (most likely to work)
            unit_result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=no"
            ], capture_output=True, text=True, timeout=120)

            for line in unit_result.stdout.split('\n'):
                if '::' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line):
                    executable_tests += 1
                    if 'PASSED' in line:
                        passing_tests += 1

        except Exception:
            pass

        # Rust test discovery
        rust_tests = 0
        rust_passing = 0

        try:
            rust_dir = self.project_root / "rust-engine"
            if rust_dir.exists():
                os.chdir(rust_dir)
                result = subprocess.run([
                    "cargo", "test", "--", "--list"
                ], capture_output=True, text=True, timeout=60)

                rust_tests = len([line for line in result.stdout.split('\n')
                                if line.strip().endswith(': test')])

                # Try running Rust tests
                test_result = subprocess.run([
                    "cargo", "test"
                ], capture_output=True, text=True, timeout=120)

                for line in test_result.stdout.split('\n'):
                    if 'test result:' in line:
                        parts = line.split()
                        if 'passed' in parts:
                            try:
                                rust_passing = int(parts[parts.index('passed') - 1])
                            except:
                                pass

        except Exception:
            pass
        finally:
            os.chdir(self.project_root)

        return {
            'total_tests': total_tests,
            'executable_tests': executable_tests,
            'passing_tests': passing_tests,
            'import_errors': import_errors,
            'rust_tests': rust_tests,
            'rust_passing': rust_passing
        }

    def get_coverage_metrics(self) -> Dict[str, Any]:
        """Get current coverage metrics"""
        os.chdir(self.project_root)

        try:
            # Run coverage analysis on executable tests
            subprocess.run([
                sys.executable, "-m", "pytest", "tests/unit/",
                "--cov=src", "--cov-report=json:coverage_temp.json", "--tb=no"
            ], capture_output=True, timeout=180)

            coverage_file = self.project_root / "coverage_temp.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    data = json.load(f)
                    return {
                        'coverage_percent': data['totals']['percent_covered'],
                        'lines_covered': data['totals']['covered_lines'],
                        'total_lines': data['totals']['num_statements']
                    }

        except Exception:
            pass

        return {
            'coverage_percent': 0.0,
            'lines_covered': 0,
            'total_lines': 1000  # Estimate
        }

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics"""
        timestamp = datetime.now().isoformat()

        print(f"📊 Collecting metrics at {timestamp}")

        test_metrics = self.discover_tests()
        coverage_metrics = self.get_coverage_metrics()

        metrics = {
            'timestamp': timestamp,
            **test_metrics,
            **coverage_metrics
        }

        # Calculate coverage delta from previous measurement
        metrics['coverage_delta'] = self.calculate_coverage_delta(metrics['coverage_percent'])

        return metrics

    def calculate_coverage_delta(self, current_coverage: float) -> float:
        """Calculate coverage change from previous measurement"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.execute(
                "SELECT coverage_percent FROM coverage_metrics ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                previous_coverage = row[0]
                return current_coverage - previous_coverage

        except Exception:
            pass

        return 0.0

    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in database"""
        conn = sqlite3.connect(self.monitoring_db)
        conn.execute("""
            INSERT OR REPLACE INTO coverage_metrics VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics['timestamp'],
            metrics['total_tests'],
            metrics['executable_tests'],
            metrics['passing_tests'],
            metrics['coverage_percent'],
            metrics['lines_covered'],
            metrics['total_lines'],
            metrics['import_errors'],
            metrics['rust_tests'],
            metrics['rust_passing'],
            metrics['coverage_delta']
        ))
        conn.commit()
        conn.close()

    def check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""

        # Coverage improvement alert
        if metrics['coverage_delta'] > 0.1:
            self.log_alert(
                "COVERAGE_IMPROVEMENT",
                f"Coverage increased by {metrics['coverage_delta']:.2f}% to {metrics['coverage_percent']:.2f}%",
                metrics
            )

        # New executable tests
        if metrics['executable_tests'] > 0:
            self.log_alert(
                "EXECUTABLE_TESTS",
                f"Found {metrics['executable_tests']} executable tests, {metrics['passing_tests']} passing",
                metrics
            )

        # Rust test success
        if metrics['rust_tests'] > 0:
            self.log_alert(
                "RUST_TESTS",
                f"Found {metrics['rust_tests']} Rust tests, {metrics['rust_passing']} passing",
                metrics
            )

        # Import error reduction
        if metrics['import_errors'] < 50:  # Improvement threshold
            self.log_alert(
                "IMPORT_ERRORS_REDUCED",
                f"Import errors reduced to {metrics['import_errors']}",
                metrics
            )

    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report"""
        conn = sqlite3.connect(self.monitoring_db)

        # Get latest metrics
        cursor = conn.execute(
            "SELECT * FROM coverage_metrics ORDER BY timestamp DESC LIMIT 1"
        )
        latest = cursor.fetchone()

        # Get historical trend (last hour)
        cursor = conn.execute("""
            SELECT * FROM coverage_metrics
            WHERE timestamp > datetime('now', '-1 hour')
            ORDER BY timestamp
        """)
        history = cursor.fetchall()

        # Get all alerts
        cursor = conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10"
        )
        recent_alerts = cursor.fetchall()

        conn.close()

        if not latest:
            return {"error": "No metrics available"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "current_status": {
                "coverage_percent": latest[4],
                "total_tests": latest[1],
                "executable_tests": latest[2],
                "passing_tests": latest[3],
                "import_errors": latest[7],
                "rust_tests": latest[8],
                "rust_passing": latest[9]
            },
            "trend_analysis": {
                "measurements_count": len(history),
                "coverage_trend": [row[4] for row in history],
                "executable_tests_trend": [row[2] for row in history]
            },
            "recent_alerts": [
                {
                    "timestamp": alert[0],
                    "type": alert[1],
                    "message": alert[2]
                } for alert in recent_alerts
            ]
        }

        return report

    def monitoring_loop(self):
        """Main monitoring loop - runs every 2 minutes"""
        print("🚀 Starting Emergency Coverage Monitoring")
        print("📍 Monitoring every 2 minutes for coverage progression")

        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.collect_metrics()

                # Store metrics
                self.store_metrics(metrics)

                # Check for alerts
                self.check_alerts(metrics)

                # Generate and save report
                report = self.generate_progress_report()
                with open(self.report_file, 'w') as f:
                    json.dump(report, f, indent=2)

                # Print current status
                print(f"""
📊 EMERGENCY MONITORING STATUS - {metrics['timestamp']}
Coverage: {metrics['coverage_percent']:.2f}% (Δ {metrics['coverage_delta']:+.2f}%)
Tests: {metrics['executable_tests']}/{metrics['total_tests']} executable, {metrics['passing_tests']} passing
Import Errors: {metrics['import_errors']}
Rust: {metrics['rust_passing']}/{metrics['rust_tests']} tests passing
Lines: {metrics['lines_covered']}/{metrics['total_lines']}
""")

                # Wait 2 minutes before next cycle
                time.sleep(120)

            except KeyboardInterrupt:
                print("🛑 Monitoring stopped by user")
                self.monitoring_active = False
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def run_emergency_monitoring(self):
        """Start emergency monitoring with immediate baseline"""
        print("🚨 EMERGENCY COVERAGE MONITORING ACTIVATED")

        # Get immediate baseline
        baseline = self.collect_metrics()
        self.store_metrics(baseline)

        print(f"""
🔍 BASELINE ESTABLISHED:
Coverage: {baseline['coverage_percent']:.2f}%
Tests: {baseline['executable_tests']}/{baseline['total_tests']} executable
Passing: {baseline['passing_tests']}
Import Errors: {baseline['import_errors']}
Rust Tests: {baseline['rust_tests']} ({baseline['rust_passing']} passing)
""")

        # Start monitoring loop
        self.monitoring_loop()

if __name__ == "__main__":
    monitor = EmergencyCoverageMonitor()
    monitor.run_emergency_monitoring()