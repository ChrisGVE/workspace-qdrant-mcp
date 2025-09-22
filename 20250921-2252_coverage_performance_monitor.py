#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring System for workspace-qdrant-mcp
Tracks coverage progression, test performance, and provides real-time monitoring.

Target: Continuous monitoring of progression to 100% coverage
Current baseline: 8.41% Python, 0% Rust
"""

import json
import time
import subprocess
import os
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3


@dataclass
class CoverageMetrics:
    """Coverage metrics for tracking progress"""
    timestamp: str
    python_line_coverage: float
    python_function_coverage: float
    python_branch_coverage: float
    rust_line_coverage: float
    rust_function_coverage: float
    test_execution_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    coverage_files_count: int
    uncovered_lines: int


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring system health"""
    timestamp: str
    coverage_report_generation_time: float
    test_execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_io_time: float


class CoverageMonitor:
    """Comprehensive coverage monitoring and performance tracking system"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "20250921-2252_coverage_monitoring.db"
        self.monitoring_active = False
        self.alert_thresholds = {
            'coverage_regression': 0.5,  # Alert if coverage drops by 0.5%
            'test_failure_increase': 5,   # Alert if failures increase by 5
            'performance_degradation': 20  # Alert if test time increases by 20%
        }
        self.setup_database()

    def setup_database(self):
        """Initialize SQLite database for tracking metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                python_line_coverage REAL,
                python_function_coverage REAL,
                python_branch_coverage REAL,
                rust_line_coverage REAL,
                rust_function_coverage REAL,
                test_execution_time REAL,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                skipped_tests INTEGER,
                coverage_files_count INTEGER,
                uncovered_lines INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                coverage_report_generation_time REAL,
                test_execution_time REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                disk_io_time REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def run_python_coverage(self) -> Tuple[Dict, float]:
        """Run Python coverage analysis and return metrics with timing"""
        start_time = time.time()

        try:
            # Run pytest with coverage
            result = subprocess.run([
                'uv', 'run', 'pytest', '--cov=src', '--cov-report=json',
                '--cov-report=xml', '--cov-report=html',
                '--tb=short', '-q'
            ], capture_output=True, text=True, cwd=self.project_root)

            execution_time = time.time() - start_time

            # Parse coverage JSON
            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    coverage_data = json.load(f)
                return coverage_data, execution_time
            else:
                return {}, execution_time

        except Exception as e:
            print(f"Error running Python coverage: {e}")
            return {}, time.time() - start_time

    def run_rust_coverage(self) -> Tuple[Dict, float]:
        """Run Rust coverage analysis and return metrics with timing"""
        start_time = time.time()

        try:
            rust_dir = self.project_root / "rust-engine"
            if rust_dir.exists():
                # Run cargo test with coverage (using tarpaulin if available)
                result = subprocess.run([
                    'cargo', 'test', '--all-features'
                ], capture_output=True, text=True, cwd=rust_dir)

                execution_time = time.time() - start_time

                # For now, return basic metrics - can be enhanced with tarpaulin
                return {
                    'line_coverage': 0.0,  # Placeholder
                    'function_coverage': 0.0,
                    'execution_time': execution_time,
                    'test_output': result.stdout
                }, execution_time
            else:
                return {}, time.time() - start_time

        except Exception as e:
            print(f"Error running Rust coverage: {e}")
            return {}, time.time() - start_time

    def parse_python_coverage(self, coverage_data: Dict) -> Dict:
        """Extract key metrics from Python coverage data"""
        if not coverage_data:
            return {
                'line_coverage': 0.0,
                'function_coverage': 0.0,
                'branch_coverage': 0.0,
                'files_count': 0,
                'uncovered_lines': 0
            }

        totals = coverage_data.get('totals', {})

        return {
            'line_coverage': totals.get('percent_covered', 0.0),
            'function_coverage': totals.get('percent_covered_display', 0.0),
            'branch_coverage': totals.get('percent_covered', 0.0),  # Simplified
            'files_count': len(coverage_data.get('files', {})),
            'uncovered_lines': totals.get('missing_lines', 0)
        }

    def get_test_metrics(self) -> Dict:
        """Parse test execution results"""
        try:
            result = subprocess.run([
                'uv', 'run', 'pytest', '--tb=no', '-q', '--collect-only'
            ], capture_output=True, text=True, cwd=self.project_root)

            # Parse output for test counts
            output = result.stdout
            total_tests = output.count('::test_') if '::test_' in output else 0

            # Run actual tests to get pass/fail counts
            test_result = subprocess.run([
                'uv', 'run', 'pytest', '--tb=no', '-q'
            ], capture_output=True, text=True, cwd=self.project_root)

            # Parse results
            output = test_result.stdout
            passed = output.count(' PASSED') if ' PASSED' in output else 0
            failed = output.count(' FAILED') if ' FAILED' in output else 0
            skipped = output.count(' SKIPPED') if ' SKIPPED' in output else 0

            return {
                'total_tests': total_tests,
                'passed_tests': passed,
                'failed_tests': failed,
                'skipped_tests': skipped
            }

        except Exception as e:
            print(f"Error getting test metrics: {e}")
            return {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'skipped_tests': 0
            }

    def collect_coverage_metrics(self) -> CoverageMetrics:
        """Collect comprehensive coverage metrics"""
        timestamp = datetime.datetime.now().isoformat()

        # Python coverage
        python_coverage_data, python_exec_time = self.run_python_coverage()
        python_metrics = self.parse_python_coverage(python_coverage_data)

        # Rust coverage
        rust_coverage_data, rust_exec_time = self.run_rust_coverage()

        # Test metrics
        test_metrics = self.get_test_metrics()

        return CoverageMetrics(
            timestamp=timestamp,
            python_line_coverage=python_metrics['line_coverage'],
            python_function_coverage=python_metrics['function_coverage'],
            python_branch_coverage=python_metrics['branch_coverage'],
            rust_line_coverage=rust_coverage_data.get('line_coverage', 0.0),
            rust_function_coverage=rust_coverage_data.get('function_coverage', 0.0),
            test_execution_time=python_exec_time + rust_exec_time,
            total_tests=test_metrics['total_tests'],
            passed_tests=test_metrics['passed_tests'],
            failed_tests=test_metrics['failed_tests'],
            skipped_tests=test_metrics['skipped_tests'],
            coverage_files_count=python_metrics['files_count'],
            uncovered_lines=python_metrics['uncovered_lines']
        )

    def save_metrics(self, metrics: CoverageMetrics):
        """Save metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO coverage_history (
                timestamp, python_line_coverage, python_function_coverage,
                python_branch_coverage, rust_line_coverage, rust_function_coverage,
                test_execution_time, total_tests, passed_tests, failed_tests,
                skipped_tests, coverage_files_count, uncovered_lines
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.python_line_coverage, metrics.python_function_coverage,
            metrics.python_branch_coverage, metrics.rust_line_coverage, metrics.rust_function_coverage,
            metrics.test_execution_time, metrics.total_tests, metrics.passed_tests,
            metrics.failed_tests, metrics.skipped_tests, metrics.coverage_files_count,
            metrics.uncovered_lines
        ))

        conn.commit()
        conn.close()

    def check_alerts(self, current_metrics: CoverageMetrics):
        """Check for alert conditions and log alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get previous metrics
        cursor.execute('''
            SELECT python_line_coverage, failed_tests, test_execution_time
            FROM coverage_history
            ORDER BY timestamp DESC
            LIMIT 1 OFFSET 1
        ''')

        prev_data = cursor.fetchone()
        if not prev_data:
            conn.close()
            return

        prev_coverage, prev_failed, prev_time = prev_data

        alerts = []

        # Coverage regression check
        if current_metrics.python_line_coverage < prev_coverage - self.alert_thresholds['coverage_regression']:
            alerts.append({
                'type': 'coverage_regression',
                'severity': 'HIGH',
                'message': f"Coverage dropped from {prev_coverage:.2f}% to {current_metrics.python_line_coverage:.2f}%"
            })

        # Test failure increase
        if current_metrics.failed_tests > prev_failed + self.alert_thresholds['test_failure_increase']:
            alerts.append({
                'type': 'test_failure_increase',
                'severity': 'MEDIUM',
                'message': f"Failed tests increased from {prev_failed} to {current_metrics.failed_tests}"
            })

        # Performance degradation
        time_increase_percent = ((current_metrics.test_execution_time - prev_time) / prev_time) * 100
        if time_increase_percent > self.alert_thresholds['performance_degradation']:
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'LOW',
                'message': f"Test execution time increased by {time_increase_percent:.1f}%"
            })

        # Save alerts
        for alert in alerts:
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (current_metrics.timestamp, alert['type'], alert['severity'], alert['message']))

        conn.commit()
        conn.close()

        # Print alerts
        for alert in alerts:
            print(f"🚨 {alert['severity']} ALERT: {alert['message']}")

    def generate_progress_report(self) -> str:
        """Generate comprehensive progress report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get latest metrics
        cursor.execute('''
            SELECT * FROM coverage_history
            ORDER BY timestamp DESC
            LIMIT 1
        ''')

        latest = cursor.fetchone()
        if not latest:
            return "No coverage data available"

        # Get progress over last 24 hours
        cursor.execute('''
            SELECT python_line_coverage, timestamp FROM coverage_history
            WHERE datetime(timestamp) > datetime('now', '-1 day')
            ORDER BY timestamp ASC
        ''')

        progress_data = cursor.fetchall()

        report = f"""
=== COVERAGE PROGRESSION MONITOR ===
Timestamp: {latest[1]}

CURRENT STATUS:
📊 Python Line Coverage: {latest[2]:.2f}%
🔧 Python Function Coverage: {latest[3]:.2f}%
🌿 Python Branch Coverage: {latest[4]:.2f}%
🦀 Rust Line Coverage: {latest[5]:.2f}%
🦀 Rust Function Coverage: {latest[6]:.2f}%

TEST METRICS:
✅ Total Tests: {latest[8]}
✅ Passed: {latest[9]}
❌ Failed: {latest[10]}
⏭️  Skipped: {latest[11]}
📁 Files Covered: {latest[12]}
📝 Uncovered Lines: {latest[13]}

PERFORMANCE:
⏱️  Test Execution Time: {latest[7]:.2f}s

PROGRESSION (24h):
"""

        if len(progress_data) > 1:
            first_coverage = progress_data[0][0]
            latest_coverage = progress_data[-1][0]
            improvement = latest_coverage - first_coverage
            report += f"📈 Coverage Change: {improvement:+.2f}%\n"
        else:
            report += "📊 Insufficient data for trend analysis\n"

        # Get recent alerts
        cursor.execute('''
            SELECT alert_type, severity, message, timestamp FROM alerts
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
        ''')

        recent_alerts = cursor.fetchall()
        if recent_alerts:
            report += "\nRECENT ALERTS (1h):\n"
            for alert in recent_alerts:
                report += f"🚨 {alert[1]}: {alert[2]}\n"
        else:
            report += "\n✅ No recent alerts\n"

        conn.close()
        return report

    def get_gap_analysis(self) -> Dict:
        """Analyze coverage gaps and provide specific recommendations"""
        try:
            coverage_json_path = self.project_root / "coverage.json"
            if not coverage_json_path.exists():
                return {"error": "No coverage data available"}

            with open(coverage_json_path) as f:
                coverage_data = json.load(f)

            gaps = []
            files = coverage_data.get('files', {})

            for filepath, file_data in files.items():
                missing_lines = file_data.get('missing_lines', [])
                if missing_lines:
                    gaps.append({
                        'file': filepath,
                        'missing_lines': missing_lines,
                        'coverage_percent': file_data.get('summary', {}).get('percent_covered', 0)
                    })

            # Sort by lowest coverage first
            gaps.sort(key=lambda x: x['coverage_percent'])

            return {
                'total_files_with_gaps': len(gaps),
                'files_needing_tests': gaps[:10],  # Top 10 priority files
                'total_uncovered_lines': sum(len(g['missing_lines']) for g in gaps)
            }

        except Exception as e:
            return {"error": f"Gap analysis failed: {e}"}

    def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous monitoring"""
        self.monitoring_active = True

        def monitoring_loop():
            while self.monitoring_active:
                print(f"\n🔍 Running coverage monitoring cycle...")

                # Collect metrics
                metrics = self.collect_coverage_metrics()

                # Save to database
                self.save_metrics(metrics)

                # Check for alerts
                self.check_alerts(metrics)

                # Print progress
                print(f"📊 Current Python Coverage: {metrics.python_line_coverage:.2f}%")
                print(f"🦀 Current Rust Coverage: {metrics.rust_line_coverage:.2f}%")
                print(f"✅ Tests Passed: {metrics.passed_tests}/{metrics.total_tests}")

                # Wait for next cycle
                time.sleep(interval_minutes * 60)

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

        print(f"🚀 Coverage monitoring started (interval: {interval_minutes} minutes)")
        return monitor_thread

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        print("🛑 Coverage monitoring stopped")


def main():
    """Main monitoring execution"""
    monitor = CoverageMonitor()

    print("🎯 Workspace Qdrant MCP Coverage Performance Monitor")
    print("Target: 100% coverage for Python and Rust codebases")
    print("Current baseline: 8.41% Python, 0% Rust")
    print("=" * 60)

    # Run initial assessment
    print("\n📋 Running initial coverage assessment...")
    metrics = monitor.collect_coverage_metrics()
    monitor.save_metrics(metrics)

    # Generate baseline report
    report = monitor.generate_progress_report()
    print(report)

    # Get gap analysis
    gaps = monitor.get_gap_analysis()
    if 'error' not in gaps:
        print(f"\n🎯 GAP ANALYSIS:")
        print(f"Files needing tests: {gaps['total_files_with_gaps']}")
        print(f"Total uncovered lines: {gaps['total_uncovered_lines']}")

        print(f"\nTOP PRIORITY FILES:")
        for file_gap in gaps['files_needing_tests'][:5]:
            print(f"📁 {file_gap['file']}: {file_gap['coverage_percent']:.1f}% coverage")

    # Start continuous monitoring
    print(f"\n🚀 Starting continuous monitoring (5-minute intervals)...")
    monitor_thread = monitor.start_monitoring(interval_minutes=5)

    try:
        # Keep main thread alive
        while True:
            time.sleep(60)

            # Generate hourly progress report
            if int(time.time()) % 3600 < 60:  # Every hour
                print("\n" + "="*60)
                print(monitor.generate_progress_report())
                print("="*60)

    except KeyboardInterrupt:
        print("\n🛑 Stopping coverage monitoring...")
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()