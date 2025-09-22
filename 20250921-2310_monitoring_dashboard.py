#!/usr/bin/env python3
"""
Real-time Coverage Monitoring Dashboard
Provides live updates on coverage progress every 2 minutes
"""

import sqlite3
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

class CoverageDashboard:
    """Real-time dashboard for coverage monitoring"""

    def __init__(self, db_path="coverage_monitor.db"):
        self.db_path = Path(db_path)
        self.terminal_width = 80

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_latest_metrics(self):
        """Get the latest coverage metrics from database"""
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get latest results for each component
        cursor.execute('''
        SELECT component, coverage_percent, lines_covered, lines_total,
               tests_passed, tests_failed, tests_total, execution_time, 
               issues, timestamp
        FROM coverage_history
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM coverage_history h2
            WHERE h2.component = coverage_history.component
        )
        ORDER BY component
        ''')

        latest_results = cursor.fetchall()

        # Get recent alerts (last 5 minutes)
        cursor.execute('''
        SELECT timestamp, alert_type, component, message, severity
        FROM alerts
        WHERE timestamp > datetime('now', '-5 minutes')
        ORDER BY timestamp DESC
        LIMIT 5
        ''')

        recent_alerts = cursor.fetchall()

        # Get coverage history for trend
        cursor.execute('''
        SELECT component, coverage_percent, timestamp
        FROM coverage_history
        WHERE timestamp > datetime('now', '-1 hour')
        ORDER BY timestamp
        ''')

        trend_data = cursor.fetchall()

        conn.close()

        return {
            'latest_results': latest_results,
            'recent_alerts': recent_alerts,
            'trend_data': trend_data
        }

    def create_mini_chart(self, data_points, width=20):
        """Create a mini ASCII chart"""
        if not data_points:
            return "No data"

        min_val = min(data_points)
        max_val = max(data_points)
        
        if max_val == min_val:
            return "─" * width

        chart = ""
        for i in range(width):
            if i < len(data_points):
                normalized = (data_points[i] - min_val) / (max_val - min_val)
                height = int(normalized * 3)
                chart += ["▁", "▂", "▃", "▄"][height] if height < 4 else "▄"
            else:
                chart += " "

        return chart

    def render_dashboard(self):
        """Render the real-time dashboard"""
        self.clear_screen()

        data = self.get_latest_metrics()
        if not data:
            print("📊 Coverage Monitor Dashboard")
            print("=" * self.terminal_width)
            print("⚠️  No monitoring data available yet.")
            print("   Run the coverage monitor first: python 20250921-2310_coverage_monitor.py")
            return

        latest_results = data['latest_results']
        recent_alerts = data['recent_alerts']
        trend_data = data['trend_data']

        # Header
        print("📊 LIVE COVERAGE MONITORING DASHBOARD")
        print("=" * self.terminal_width)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Updated every 2 minutes")
        print("=" * self.terminal_width)

        if not latest_results:
            print("⚠️  No coverage data available yet.")
            return

        # Coverage Overview
        total_coverage = 0
        component_count = 0

        print("\n📈 COVERAGE OVERVIEW")
        print("-" * 50)

        for result in latest_results:
            component, coverage_percent, lines_covered, lines_total, tests_passed, tests_failed, tests_total, execution_time, issues, timestamp = result

            # Progress bar
            progress_bar = self.create_progress_bar(coverage_percent, 30)
            
            # Status indicator
            status = "🟢" if coverage_percent >= 100 else "🟡" if coverage_percent >= 80 else "🔴"
            
            print(f"{status} {component.upper():<8} {progress_bar} {coverage_percent:5.1f}%")
            print(f"   Tests: {tests_passed}/{tests_total} passed | Lines: {lines_covered}/{lines_total} | Time: {execution_time:.1f}s")
            
            if issues:
                print(f"   ⚠️  {issues[:60]}...")

            total_coverage += coverage_percent
            component_count += 1

        # Overall Progress
        if component_count > 0:
            avg_coverage = total_coverage / component_count
            overall_status = "🎉" if avg_coverage >= 100 else "🎯" if avg_coverage >= 90 else "📈"
            overall_bar = self.create_progress_bar(avg_coverage, 40)
            
            print("\n" + "─" * 50)
            print(f"{overall_status} OVERALL   {overall_bar} {avg_coverage:5.1f}%")
            
            remaining = 100 - avg_coverage
            if remaining > 0:
                print(f"🎯 Target: {remaining:.1f}% more needed for 100% coverage")
            else:
                print("🏆 100% COVERAGE ACHIEVED ON ALL COMPONENTS!")

        # Trend Analysis
        if trend_data:
            print("\n📊 TREND ANALYSIS (Last Hour)")
            print("-" * 50)
            
            # Group by component
            component_trends = {}
            for component, coverage, timestamp in trend_data:
                if component not in component_trends:
                    component_trends[component] = []
                component_trends[component].append(coverage)
            
            for component, coverages in component_trends.items():
                if len(coverages) > 1:
                    trend_direction = "📈" if coverages[-1] > coverages[0] else "📉" if coverages[-1] < coverages[0] else "➡️"
                    change = coverages[-1] - coverages[0]
                    chart = self.create_mini_chart(coverages[-20:])  # Last 20 points
                    print(f"{trend_direction} {component.upper():<8} {chart} ({change:+.1f}%)")

        # Recent Alerts
        if recent_alerts:
            print("\n🚨 RECENT ALERTS (Last 5 minutes)")
            print("-" * 50)
            
            for alert in recent_alerts[:3]:  # Show top 3
                timestamp, alert_type, component, message, severity = alert
                alert_time = datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
                emoji = "🚨" if severity == "ERROR" else "⚠️" if severity == "WARNING" else "📈"
                print(f"[{alert_time}] {emoji} {message[:60]}")

        # Progress Targets
        print(f"\n🎯 PROGRESS TARGETS")
        print("-" * 50)
        
        for result in latest_results:
            component, coverage_percent, lines_covered, lines_total = result[:4]
            remaining_lines = lines_total - lines_covered
            
            if coverage_percent < 100:
                print(f"{component.upper()}: {remaining_lines} lines to cover ({100-coverage_percent:.1f}% remaining)")
            else:
                print(f"{component.upper()}: ✅ Target achieved!")

        # Bottom status bar
        print("\n" + "=" * self.terminal_width)
        next_update = datetime.now() + timedelta(minutes=2)
        print(f"⏰ Next update: {next_update.strftime('%H:%M:%S')} | Press Ctrl+C to stop")

    def create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """Create a compact progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def live_dashboard(self, update_interval=120):
        """Run live dashboard with automatic updates"""
        print("🚀 Starting Live Coverage Dashboard...")
        print("📊 Monitoring coverage progress every 2 minutes")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            while True:
                self.render_dashboard()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            self.clear_screen()
            print("📊 Live Dashboard stopped")
            print("Thank you for monitoring coverage progress!")

def main():
    """Main dashboard entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Coverage Dashboard")
    parser.add_argument("--db", type=str, default="coverage_monitor.db", help="Database path")
    parser.add_argument("--interval", type=int, default=120, help="Update interval in seconds")
    parser.add_argument("--once", action="store_true", help="Show dashboard once and exit")
    
    args = parser.parse_args()
    
    dashboard = CoverageDashboard(args.db)
    
    if args.once:
        dashboard.render_dashboard()
    else:
        dashboard.live_dashboard(args.interval)

if __name__ == "__main__":
    main()