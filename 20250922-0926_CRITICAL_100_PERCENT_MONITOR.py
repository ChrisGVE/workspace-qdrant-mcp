#!/usr/bin/env python3
"""
🚨 CRITICAL EMERGENCY: 100% COVERAGE ACHIEVEMENT MONITOR 🚨

MISSION: Monitor progress toward 100% coverage achievement
ALERT FREQUENCY: Every 2 minutes as requested
TARGET: Both Python 100% AND Rust 100% coverage
URGENCY: CRITICAL - Alert immediately when mission accomplished!

Performance Monitor Agent: ACTIVE
"""

import subprocess
import time
import json
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple

# Configure emergency logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🚨 EMERGENCY - %(message)s',
    handlers=[
        logging.FileHandler('20250922-0926_CRITICAL_ALERTS.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CriticalCoverageMonitor:
    """Emergency monitor for 100% coverage achievement"""

    def __init__(self):
        self.db_path = "20250922-0926_critical_monitoring.db"
        self.target_python_coverage = 100.0
        self.target_rust_coverage = 100.0
        self.alert_interval = 120  # 2 minutes as requested
        self.init_database()
        self.last_alert_time = None

        # Mission control metrics
        self.mission_start_time = datetime.now()
        self.alerts_sent = 0
        self.coverage_checks = 0

        logger.info("🎯 CRITICAL MISSION INITIATED: Monitor until 100% coverage achieved!")
        logger.info(f"⏰ Alert interval: {self.alert_interval} seconds (2 minutes)")

    def init_database(self):
        """Initialize emergency monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS critical_coverage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                python_coverage REAL,
                rust_coverage REAL,
                python_tests_passed INTEGER,
                python_tests_failed INTEGER,
                rust_tests_passed INTEGER,
                rust_tests_failed INTEGER,
                blocking_issues TEXT,
                alert_level TEXT,
                mission_status TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mission_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                urgency_level TEXT NOT NULL,
                coverage_python REAL,
                coverage_rust REAL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def get_python_coverage(self) -> Tuple[float, int, int, str]:
        """Get current Python coverage with enhanced error tracking"""
        try:
            logger.info("🔍 Checking Python coverage...")

            # Run coverage with enhanced error capture
            cmd = ["uv", "run", "pytest", "--cov=src", "--cov-report=term", "--tb=no", "-q"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            output = result.stdout + result.stderr

            # Extract coverage percentage
            coverage = 0.0
            tests_passed = 0
            tests_failed = 0
            issues = ""

            if "% coverage" in output:
                for line in output.split('\n'):
                    if "% coverage" in line:
                        try:
                            coverage = float(line.split('%')[0].split()[-1])
                            break
                        except (ValueError, IndexError):
                            continue

            # Count test results
            if result.returncode == 0:
                for line in output.split('\n'):
                    if "passed" in line and "failed" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed":
                                try:
                                    tests_passed = int(parts[i-1])
                                except (ValueError, IndexError):
                                    pass
                            elif part == "failed":
                                try:
                                    tests_failed = int(parts[i-1])
                                except (ValueError, IndexError):
                                    pass
            else:
                # Extract error count from collection failures
                if "errors during collection" in output:
                    for line in output.split('\n'):
                        if "errors during collection" in line:
                            try:
                                error_count = int(line.split()[1])
                                tests_failed = error_count
                                issues = f"Collection errors: {error_count}"
                            except (ValueError, IndexError):
                                issues = "Test collection failed"
                            break
                else:
                    issues = "Test execution failed"

            logger.info(f"📊 Python Coverage: {coverage}% ({tests_passed} passed, {tests_failed} failed)")
            if issues:
                logger.warning(f"⚠️ Issues: {issues}")

            return coverage, tests_passed, tests_failed, issues

        except subprocess.TimeoutExpired:
            logger.error("⏰ Python coverage check timed out!")
            return 0.0, 0, 1, "Coverage check timeout"
        except Exception as e:
            logger.error(f"💥 Python coverage check failed: {e}")
            return 0.0, 0, 1, f"Coverage check error: {str(e)}"

    def get_rust_coverage(self) -> Tuple[float, int, int, str]:
        """Get current Rust coverage with setup detection"""
        try:
            logger.info("🔍 Checking Rust coverage...")

            # Check if we're in the rust-engine directory
            rust_dir = Path("rust-engine")
            if not rust_dir.exists():
                return 0.0, 0, 0, "Rust engine directory not found"

            # Try to run Rust tests first
            os.chdir(rust_dir)

            # Check if tarpaulin is available
            tarpaulin_check = subprocess.run(
                ["cargo", "tarpaulin", "--version"],
                capture_output=True, text=True
            )

            if tarpaulin_check.returncode != 0:
                # Tarpaulin not installed, try basic cargo test
                logger.warning("📦 cargo-tarpaulin not installed, checking basic tests...")
                test_result = subprocess.run(
                    ["cargo", "test"],
                    capture_output=True, text=True, timeout=300
                )

                # Parse basic test output
                tests_passed = 0
                tests_failed = 0
                issues = "No coverage tool (tarpaulin) installed"

                if test_result.returncode == 0:
                    output = test_result.stdout
                    if "test result:" in output:
                        for line in output.split('\n'):
                            if "test result:" in line:
                                parts = line.split()
                                try:
                                    if "passed" in line:
                                        tests_passed = int(parts[parts.index("passed") - 1])
                                    if "failed" in line:
                                        tests_failed = int(parts[parts.index("failed") - 1])
                                except (ValueError, IndexError):
                                    pass

                os.chdir("..")
                return 0.0, tests_passed, tests_failed, issues

            # Run tarpaulin for coverage
            cmd = ["cargo", "tarpaulin", "--all", "--out", "Stdout"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            output = result.stdout + result.stderr
            coverage = 0.0
            tests_passed = 0
            tests_failed = 0
            issues = ""

            # Parse tarpaulin output
            if result.returncode == 0:
                for line in output.split('\n'):
                    if "Coverage Results:" in line or "%" in line:
                        try:
                            if line.strip().endswith('%'):
                                coverage = float(line.strip().rstrip('%').split()[-1])
                        except (ValueError, IndexError):
                            continue
                    elif "test result:" in line:
                        parts = line.split()
                        try:
                            if "passed" in line:
                                tests_passed = int(parts[parts.index("passed") - 1])
                            if "failed" in line:
                                tests_failed = int(parts[parts.index("failed") - 1])
                        except (ValueError, IndexError):
                            pass
            else:
                issues = "Rust coverage generation failed"

            os.chdir("..")
            logger.info(f"📊 Rust Coverage: {coverage}% ({tests_passed} passed, {tests_failed} failed)")
            if issues:
                logger.warning(f"⚠️ Issues: {issues}")

            return coverage, tests_passed, tests_failed, issues

        except subprocess.TimeoutExpired:
            os.chdir("..")
            logger.error("⏰ Rust coverage check timed out!")
            return 0.0, 0, 1, "Rust coverage timeout"
        except Exception as e:
            try:
                os.chdir("..")
            except:
                pass
            logger.error(f"💥 Rust coverage check failed: {e}")
            return 0.0, 0, 1, f"Rust coverage error: {str(e)}"

    def log_coverage_data(self, python_data: Tuple, rust_data: Tuple, alert_level: str, mission_status: str):
        """Log coverage data to emergency database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        python_cov, python_pass, python_fail, python_issues = python_data
        rust_cov, rust_pass, rust_fail, rust_issues = rust_data

        all_issues = f"Python: {python_issues}; Rust: {rust_issues}" if python_issues or rust_issues else ""

        cursor.execute('''
            INSERT INTO critical_coverage_log
            (timestamp, python_coverage, rust_coverage, python_tests_passed,
             python_tests_failed, rust_tests_passed, rust_tests_failed,
             blocking_issues, alert_level, mission_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            python_cov, rust_cov,
            python_pass, python_fail,
            rust_pass, rust_fail,
            all_issues, alert_level, mission_status
        ))

        conn.commit()
        conn.close()

    def send_critical_alert(self, alert_type: str, message: str, urgency: str, python_cov: float, rust_cov: float):
        """Send critical mission alert"""
        timestamp = datetime.now().isoformat()

        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO mission_alerts
            (timestamp, alert_type, message, urgency_level, coverage_python, coverage_rust)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, alert_type, message, urgency, python_cov, rust_cov))
        conn.commit()
        conn.close()

        # Alert to console
        alert_prefix = {
            'CRITICAL': '🚨🚨🚨 CRITICAL ALERT',
            'SUCCESS': '🎉🎉🎉 MISSION SUCCESS',
            'WARNING': '⚠️⚠️⚠️ WARNING',
            'INFO': 'ℹ️ INFO'
        }.get(urgency, '📢 ALERT')

        logger.critical(f"{alert_prefix}: {message}")

        # Write to alert file
        with open("20250922-0926_CRITICAL_ALERTS.log", "a") as f:
            f.write(f"\n{timestamp} | {urgency} | {alert_type} | {message}\n")
            f.write(f"Coverage Status: Python {python_cov}%, Rust {rust_cov}%\n")
            f.write("=" * 80 + "\n")

        self.alerts_sent += 1

    def check_mission_completion(self, python_cov: float, rust_cov: float) -> bool:
        """Check if 100% coverage mission is complete"""
        if python_cov >= self.target_python_coverage and rust_cov >= self.target_rust_coverage:
            self.send_critical_alert(
                "MISSION_ACCOMPLISHED",
                f"🎯 TARGET ACHIEVED! Python: {python_cov}%, Rust: {rust_cov}% - BOTH AT 100%!",
                "SUCCESS",
                python_cov, rust_cov
            )
            return True
        return False

    def analyze_progress(self, python_cov: float, rust_cov: float, python_issues: str, rust_issues: str):
        """Analyze progress and send alerts as needed"""
        current_time = datetime.now()

        # Check for mission completion
        if self.check_mission_completion(python_cov, rust_cov):
            return "MISSION_COMPLETED"

        # Determine mission status
        avg_coverage = (python_cov + rust_cov) / 2

        if avg_coverage < 25:
            mission_status = "FOUNDATION_PHASE"
        elif avg_coverage < 50:
            mission_status = "BUILDING_PHASE"
        elif avg_coverage < 75:
            mission_status = "ACCELERATION_PHASE"
        elif avg_coverage < 90:
            mission_status = "OPTIMIZATION_PHASE"
        else:
            mission_status = "FINAL_PUSH_PHASE"

        # Alert on significant milestones
        if python_cov >= 25 and not hasattr(self, 'python_25_alerted'):
            self.send_critical_alert(
                "MILESTONE_PYTHON_25",
                f"🎯 Python reached 25% coverage milestone! Current: {python_cov}%",
                "INFO", python_cov, rust_cov
            )
            self.python_25_alerted = True

        if rust_cov >= 25 and not hasattr(self, 'rust_25_alerted'):
            self.send_critical_alert(
                "MILESTONE_RUST_25",
                f"🎯 Rust reached 25% coverage milestone! Current: {rust_cov}%",
                "INFO", python_cov, rust_cov
            )
            self.rust_25_alerted = True

        # Alert on critical blocking issues
        if python_issues and "Collection errors" in python_issues:
            self.send_critical_alert(
                "PYTHON_BLOCKED",
                f"🚫 Python tests blocked by collection errors: {python_issues}",
                "CRITICAL", python_cov, rust_cov
            )

        if rust_issues and "not installed" in rust_issues:
            self.send_critical_alert(
                "RUST_INFRASTRUCTURE",
                f"🔧 Rust coverage infrastructure needs setup: {rust_issues}",
                "WARNING", python_cov, rust_cov
            )

        return mission_status

    def generate_progress_report(self, python_data: Tuple, rust_data: Tuple):
        """Generate comprehensive progress report"""
        python_cov, python_pass, python_fail, python_issues = python_data
        rust_cov, rust_pass, rust_fail, rust_issues = rust_data

        elapsed = datetime.now() - self.mission_start_time

        report = f"""
🚨 CRITICAL EMERGENCY MONITORING REPORT 🚨
⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🕐 Mission Duration: {elapsed}
📊 Coverage Checks: {self.coverage_checks}
🚨 Alerts Sent: {self.alerts_sent}

📈 PYTHON COMPONENT STATUS:
   Coverage: {python_cov}% (Target: 100%)
   Gap: {100 - python_cov:.1f}% remaining
   Tests: {python_pass} passed, {python_fail} failed
   Progress: {'█' * int(python_cov/5)}{'░' * (20-int(python_cov/5))} {python_cov}%
   Issues: {python_issues or 'None'}

📈 RUST COMPONENT STATUS:
   Coverage: {rust_cov}% (Target: 100%)
   Gap: {100 - rust_cov:.1f}% remaining
   Tests: {rust_pass} passed, {rust_fail} failed
   Progress: {'█' * int(rust_cov/5)}{'░' * (20-int(rust_cov/5))} {rust_cov}%
   Issues: {rust_issues or 'None'}

🎯 OVERALL MISSION STATUS:
   Average Coverage: {(python_cov + rust_cov)/2:.1f}%
   Target Achievement: {'✅ COMPLETED' if python_cov >= 100 and rust_cov >= 100 else '🔄 IN PROGRESS'}

🚨 CRITICAL NEXT ACTIONS:
   1. {'✅ Python 100%' if python_cov >= 100 else f'🎯 Python needs {100-python_cov:.1f}% more'}
   2. {'✅ Rust 100%' if rust_cov >= 100 else f'🎯 Rust needs {100-rust_cov:.1f}% more'}

⏰ Next check in 2 minutes...
================================================================================
"""
        return report

    def run_continuous_monitoring(self):
        """Run continuous monitoring with 2-minute intervals"""
        logger.info("🚀 Starting CRITICAL EMERGENCY monitoring...")
        logger.info("🎯 Mission: Achieve 100% coverage for BOTH Python AND Rust")
        logger.info(f"⏰ Monitoring interval: {self.alert_interval} seconds")

        while True:
            try:
                self.coverage_checks += 1
                logger.info(f"🔍 Coverage Check #{self.coverage_checks} - Mission Critical Scan")

                # Get current coverage data
                python_data = self.get_python_coverage()
                rust_data = self.get_rust_coverage()

                python_cov = python_data[0]
                rust_cov = rust_data[0]

                # Analyze progress and determine mission status
                mission_status = self.analyze_progress(
                    python_cov, rust_cov, python_data[3], rust_data[3]
                )

                # Determine alert level
                if python_cov >= 100 and rust_cov >= 100:
                    alert_level = "SUCCESS"
                elif python_data[3] or rust_data[3]:
                    alert_level = "CRITICAL"
                elif python_cov < 25 or rust_cov < 25:
                    alert_level = "WARNING"
                else:
                    alert_level = "INFO"

                # Log data
                self.log_coverage_data(python_data, rust_data, alert_level, mission_status)

                # Generate and display report
                report = self.generate_progress_report(python_data, rust_data)
                print(report)

                # Check if mission is complete
                if mission_status == "MISSION_COMPLETED":
                    logger.critical("🎉🎉🎉 MISSION ACCOMPLISHED! 100% COVERAGE ACHIEVED!")
                    break

                # Wait for next check
                logger.info(f"⏰ Waiting {self.alert_interval} seconds until next critical check...")
                time.sleep(self.alert_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Emergency monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"💥 Critical monitoring error: {e}")
                time.sleep(30)  # Brief pause on error

        logger.info("🏁 Critical coverage monitoring session ended")

if __name__ == "__main__":
    monitor = CriticalCoverageMonitor()
    monitor.run_continuous_monitoring()