#!/usr/bin/env python3
"""
Real-time Coverage Dashboard for workspace-qdrant-mcp
Provides live visualization of coverage progression and performance metrics.
"""

import json
import sqlite3
import time
import subprocess
from pathlib import Path
from typing import Dict, List
import datetime


class CoverageDashboard:
    """Real-time dashboard for coverage monitoring"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "20250921-2252_coverage_monitoring.db"

    def get_current_coverage(self) -> Dict:
        """Get current coverage statistics"""
        try:
            # Run coverage analysis
            result = subprocess.run([
                'uv', 'run', 'pytest', '--cov=src', '--cov-report=json',
                '--tb=short', '-q'
            ], capture_output=True, text=True, cwd=self.project_root)

            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    coverage_data = json.load(f)

                totals = coverage_data.get('totals', {})
                files = coverage_data.get('files', {})

                return {
                    'line_coverage': totals.get('percent_covered', 0.0),
                    'covered_lines': totals.get('covered_lines', 0),
                    'missing_lines': totals.get('missing_lines', 0),
                    'total_lines': totals.get('num_statements', 0),
                    'files_covered': len([f for f in files.values() if f.get('summary', {}).get('covered_lines', 0) > 0]),
                    'total_files': len(files),
                    'timestamp': datetime.datetime.now().isoformat()
                }
            else:
                return {'error': 'No coverage data available'}

        except Exception as e:
            return {'error': f'Coverage analysis failed: {e}'}

    def get_detailed_file_analysis(self) -> List[Dict]:
        """Get detailed analysis of files and their coverage"""
        try:
            coverage_json_path = self.project_root / "coverage.json"
            if not coverage_json_path.exists():
                return []

            with open(coverage_json_path) as f:
                coverage_data = json.load(f)

            files = coverage_data.get('files', {})
            analysis = []

            for filepath, file_data in files.items():
                summary = file_data.get('summary', {})
                missing_lines = file_data.get('missing_lines', [])

                analysis.append({
                    'file': filepath,
                    'coverage_percent': summary.get('percent_covered', 0),
                    'covered_lines': summary.get('covered_lines', 0),
                    'missing_lines_count': summary.get('missing_lines', 0),
                    'total_lines': summary.get('num_statements', 0),
                    'missing_line_numbers': missing_lines,
                    'priority_score': self.calculate_priority_score(filepath, summary)
                })

            # Sort by priority (lowest coverage + highest line count)
            analysis.sort(key=lambda x: x['priority_score'], reverse=True)
            return analysis

        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return []

    def calculate_priority_score(self, filepath: str, summary: Dict) -> float:
        """Calculate priority score for testing (higher = more urgent)"""
        coverage_percent = summary.get('percent_covered', 0)
        total_lines = summary.get('num_statements', 0)
        missing_lines = summary.get('missing_lines', 0)

        # High priority: low coverage + many lines + core files
        core_modules = ['client.py', 'server.py', 'memory.py', 'hybrid_search.py']
        is_core = any(module in filepath for module in core_modules)

        priority = (100 - coverage_percent) * 0.4  # Coverage gap weight
        priority += missing_lines * 0.3  # Missing lines weight
        priority += total_lines * 0.2  # File size weight
        priority += 20 if is_core else 0  # Core module bonus

        return priority

    def get_test_recommendations(self) -> List[Dict]:
        """Generate specific test recommendations"""
        file_analysis = self.get_detailed_file_analysis()
        recommendations = []

        for file_info in file_analysis[:10]:  # Top 10 priority files
            filepath = file_info['file']
            missing_lines = file_info['missing_line_numbers']

            if missing_lines:
                # Analyze what type of code is missing coverage
                try:
                    with open(self.project_root / filepath) as f:
                        lines = f.readlines()

                    missing_code = []
                    for line_num in missing_lines[:5]:  # Sample first 5 missing lines
                        if 1 <= line_num <= len(lines):
                            code = lines[line_num - 1].strip()
                            missing_code.append(f"Line {line_num}: {code}")

                    recommendations.append({
                        'file': filepath,
                        'coverage': file_info['coverage_percent'],
                        'missing_lines_count': len(missing_lines),
                        'sample_missing_code': missing_code,
                        'test_suggestions': self.generate_test_suggestions(filepath, missing_code)
                    })

                except Exception as e:
                    recommendations.append({
                        'file': filepath,
                        'coverage': file_info['coverage_percent'],
                        'error': f"Could not analyze: {e}"
                    })

        return recommendations

    def generate_test_suggestions(self, filepath: str, missing_code: List[str]) -> List[str]:
        """Generate specific test suggestions based on missing code"""
        suggestions = []

        # Analyze patterns in missing code
        code_text = ' '.join(missing_code).lower()

        if 'def ' in code_text:
            suggestions.append("Add unit tests for uncovered functions")
        if 'if ' in code_text or 'else' in code_text:
            suggestions.append("Add tests for conditional branches")
        if 'except' in code_text or 'raise' in code_text:
            suggestions.append("Add tests for error handling paths")
        if 'async' in code_text:
            suggestions.append("Add async test cases")
        if 'log' in code_text:
            suggestions.append("Add tests that verify logging behavior")
        if '__init__' in code_text:
            suggestions.append("Add tests for class initialization")

        # File-specific suggestions
        if 'client.py' in filepath:
            suggestions.append("Add integration tests with Qdrant client")
        elif 'server.py' in filepath:
            suggestions.append("Add MCP server endpoint tests")
        elif 'hybrid_search.py' in filepath:
            suggestions.append("Add search algorithm tests")
        elif 'memory.py' in filepath:
            suggestions.append("Add document memory tests")

        return suggestions if suggestions else ["Add comprehensive unit tests"]

    def generate_milestone_report(self) -> str:
        """Generate milestone progress report"""
        current_coverage = self.get_current_coverage()

        if 'error' in current_coverage:
            return f"❌ Error getting coverage: {current_coverage['error']}"

        coverage_percent = current_coverage['line_coverage']

        # Determine current milestone phase
        if coverage_percent < 25:
            phase = "Phase 1: Foundation Building"
            next_target = 25
        elif coverage_percent < 50:
            phase = "Phase 2: Core Coverage"
            next_target = 50
        elif coverage_percent < 75:
            phase = "Phase 3: Advanced Features"
            next_target = 75
        else:
            phase = "Phase 4: Completion Push"
            next_target = 100

        progress_to_next = ((coverage_percent - (next_target - 25)) / 25) * 100 if next_target != 100 else ((coverage_percent - 75) / 25) * 100

        report = f"""
🎯 MILESTONE PROGRESS REPORT
========================

CURRENT STATUS:
📊 Coverage: {coverage_percent:.2f}%
🎯 Current Phase: {phase}
📈 Progress to {next_target}%: {progress_to_next:.1f}%

COVERAGE BREAKDOWN:
✅ Covered Lines: {current_coverage['covered_lines']}
❌ Missing Lines: {current_coverage['missing_lines']}
📊 Total Lines: {current_coverage['total_lines']}
📁 Files with Coverage: {current_coverage['files_covered']}/{current_coverage['total_files']}

MILESTONE TARGETS:
🏆 25%: Basic functionality covered
🏆 50%: Core modules covered
🏆 75%: Advanced features covered
🏆 100%: Complete codebase covered

"""

        # Add phase-specific guidance
        if coverage_percent < 25:
            report += """
PHASE 1 PRIORITIES:
🎯 Focus on main entry points and core classes
🎯 Test basic functionality and happy paths
🎯 Ensure imports and basic operations work
"""
        elif coverage_percent < 50:
            report += """
PHASE 2 PRIORITIES:
🎯 Cover all public API methods
🎯 Test main business logic paths
🎯 Add error handling tests
"""
        elif coverage_percent < 75:
            report += """
PHASE 3 PRIORITIES:
🎯 Test edge cases and corner scenarios
🎯 Add integration tests
🎯 Cover utility and helper functions
"""
        else:
            report += """
PHASE 4 PRIORITIES:
🎯 Achieve 100% line coverage
🎯 Add comprehensive error scenarios
🎯 Test all code paths and branches
"""

        return report

    def display_real_time_dashboard(self):
        """Display real-time coverage dashboard"""
        print("\n" + "="*80)
        print("🎯 WORKSPACE-QDRANT-MCP COVERAGE DASHBOARD")
        print("="*80)

        # Current coverage status
        current_coverage = self.get_current_coverage()
        if 'error' not in current_coverage:
            print(f"📊 Current Coverage: {current_coverage['line_coverage']:.2f}%")
            print(f"✅ Covered Lines: {current_coverage['covered_lines']}")
            print(f"❌ Missing Lines: {current_coverage['missing_lines']}")
            print(f"📁 Files: {current_coverage['files_covered']}/{current_coverage['total_files']} covered")

            # Progress bar
            progress = int(current_coverage['line_coverage'])
            bar = "█" * (progress // 2) + "░" * (50 - progress // 2)
            print(f"Progress: [{bar}] {progress}%")

        # Milestone report
        milestone_report = self.generate_milestone_report()
        print(milestone_report)

        # Top priority files
        print("\n🎯 TOP PRIORITY FILES FOR TESTING:")
        print("-" * 50)
        file_analysis = self.get_detailed_file_analysis()
        for i, file_info in enumerate(file_analysis[:5], 1):
            print(f"{i}. {file_info['file']}")
            print(f"   Coverage: {file_info['coverage_percent']:.1f}%")
            print(f"   Missing: {file_info['missing_lines_count']} lines")
            print(f"   Priority Score: {file_info['priority_score']:.1f}")
            print()

        # Test recommendations
        print("💡 SPECIFIC TEST RECOMMENDATIONS:")
        print("-" * 50)
        recommendations = self.get_test_recommendations()
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. {rec['file']} ({rec['coverage']:.1f}% coverage)")
            for suggestion in rec.get('test_suggestions', [])[:2]:
                print(f"   • {suggestion}")
            print()

        print("="*80)

    def monitor_continuous(self, interval_seconds: int = 300):
        """Run continuous monitoring with dashboard updates"""
        print(f"🚀 Starting continuous coverage monitoring (update every {interval_seconds//60} minutes)")

        try:
            while True:
                self.display_real_time_dashboard()
                print(f"\n⏳ Next update in {interval_seconds//60} minutes...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n🛑 Dashboard monitoring stopped")


def main():
    """Main dashboard execution"""
    dashboard = CoverageDashboard()

    print("🎯 Real-time Coverage Dashboard")
    print("Target: 100% coverage progression monitoring")

    # Display initial dashboard
    dashboard.display_real_time_dashboard()

    # Ask for continuous monitoring
    try:
        response = input("\n🚀 Start continuous monitoring? (y/n): ").lower()
        if response == 'y':
            dashboard.monitor_continuous()
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed")


if __name__ == "__main__":
    main()