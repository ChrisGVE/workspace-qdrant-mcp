#!/usr/bin/env python3
"""
Real-time coverage monitoring during emergency test execution
"""

import time
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime


class CoverageMonitor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.coverage_history = []

    def get_current_coverage(self) -> float:
        """Get current coverage from coverage.xml"""
        xml_path = self.project_root / "coverage.xml"
        if not xml_path.exists():
            return 0.0

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            coverage_attr = root.get('line-rate')
            if coverage_attr:
                return float(coverage_attr) * 100

        except Exception:
            return 0.0

        return 0.0

    def monitor_continuous(self, duration_minutes: int = 30):
        """Monitor coverage continuously for specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        print(f"Starting coverage monitoring for {duration_minutes} minutes...")
        print("Time\t\tCoverage\tChange")
        print("-" * 40)

        last_coverage = 0.0

        while time.time() < end_time:
            current_coverage = self.get_current_coverage()
            current_time = datetime.now().strftime("%H:%M:%S")

            change = current_coverage - last_coverage
            change_str = f"+{change:.2f}%" if change > 0 else f"{change:.2f}%" if change < 0 else "±0.00%"

            print(f"{current_time}\t{current_coverage:.2f}%\t\t{change_str}")

            self.coverage_history.append({
                'timestamp': time.time(),
                'coverage': current_coverage,
                'change': change
            })

            last_coverage = current_coverage
            time.sleep(10)  # Check every 10 seconds

        self.save_monitoring_results()

    def save_monitoring_results(self):
        """Save monitoring results"""
        timestamp = time.strftime("%Y%m%d-%H%M")
        results_file = self.project_root / f"{timestamp}_coverage_monitoring.json"

        results = {
            'monitoring_start': self.coverage_history[0]['timestamp'] if self.coverage_history else time.time(),
            'monitoring_end': time.time(),
            'coverage_history': self.coverage_history,
            'max_coverage': max(item['coverage'] for item in self.coverage_history) if self.coverage_history else 0.0,
            'final_coverage': self.coverage_history[-1]['coverage'] if self.coverage_history else 0.0
        }

        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nMonitoring results saved to: {results_file}")
        except Exception as e:
            print(f"Failed to save monitoring results: {e}")


def main():
    project_root = os.getcwd()
    monitor = CoverageMonitor(project_root)

    print("EMERGENCY COVERAGE MONITORING")
    print(f"Project: {project_root}")
    print(f"Target: Track coverage progression to 100%")

    monitor.monitor_continuous(30)  # Monitor for 30 minutes


if __name__ == "__main__":
    main()