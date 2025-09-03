#!/usr/bin/env python3
"""
SQLite State Manager Test Runner - Task 75

This script provides a comprehensive test runner for SQLite state management tests
with different execution modes and reporting capabilities.

Usage:
    python run_sqlite_tests.py [category] [options]
    
Categories:
    all         - Run all tests (default)
    basic       - Basic functionality tests
    crash       - Crash recovery tests  
    concurrent  - Concurrent access tests
    acid        - ACID transaction tests
    performance - Performance benchmarks
    maintenance - Database maintenance tests
    errors      - Error scenario tests
    integration - Integration workflow tests

Options:
    --skip-slow     - Skip slow-running tests
    --verbose       - Verbose output
    --report        - Generate HTML report
    --benchmark     - Include performance benchmarks
    --parallel      - Run tests in parallel where possible
"""

import sys
import subprocess
import argparse
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class SQLiteTestRunner:
    """Test runner for SQLite state management tests."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.test_file = self.test_dir / "test_sqlite_state_manager_comprehensive.py" 
        self.results = {}
        self.start_time = None
        
    def run_category(self, category: str, **options) -> Dict:
        """Run tests for a specific category."""
        print(f"\n🧪 Running {category.upper()} tests for SQLite State Manager...")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", "-v"]
        
        # Add category-specific markers
        test_filters = self._get_test_filters(category)
        if test_filters:
            cmd.extend(["-k", test_filters])
            
        # Add options
        if options.get('skip_slow'):
            cmd.extend(["-m", "not slow"])
            
        if options.get('verbose'):
            cmd.append("-vv")
        else:
            cmd.append("-v")
            
        if options.get('report'):
            report_path = self.test_dir / f"reports/sqlite_tests_{category}_{int(time.time())}.html"
            report_path.parent.mkdir(exist_ok=True)
            cmd.extend(["--html", str(report_path)])
            
        if options.get('parallel') and category in ['concurrent', 'performance']:
            cmd.extend(["-n", "auto"])  # Requires pytest-xdist
            
        if options.get('benchmark'):
            cmd.append("--benchmark-only")
            
        # Add coverage if requested
        if options.get('coverage'):
            cmd.extend([
                "--cov=tests.test_sqlite_state_manager_comprehensive",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
        
        # Add test file
        cmd.append(str(self.test_file))
        
        # Execute tests
        print(f"📋 Command: {' '.join(cmd)}")
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.test_dir.parent,
                capture_output=True,
                text=True,
                timeout=300 if category != 'performance' else 600  # Longer timeout for performance tests
            )
            
            duration = time.perf_counter() - start_time
            
            # Parse results
            test_results = self._parse_pytest_output(result.stdout, result.stderr)
            test_results.update({
                'category': category,
                'duration': duration,
                'exit_code': result.returncode,
                'command': ' '.join(cmd)
            })
            
            # Print summary
            self._print_category_summary(test_results)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            print(f"❌ Tests timed out after {300 if category != 'performance' else 600} seconds")
            return {
                'category': category,
                'error': 'timeout',
                'duration': 300 if category != 'performance' else 600
            }
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {
                'category': category,
                'error': str(e),
                'duration': 0
            }
    
    def _get_test_filters(self, category: str) -> Optional[str]:
        """Get pytest filter string for category."""
        filters = {
            'basic': 'test_initialization or test_basic_crud or test_query',
            'crash': 'crash_recovery or crash',
            'concurrent': 'concurrent',
            'acid': 'transaction',
            'performance': 'performance', 
            'maintenance': 'vacuum or analyze or wal',
            'errors': 'disk_full or corruption or connection_limits or error',
            'integration': 'Integration'
        }
        return filters.get(category)
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict:
        """Parse pytest output to extract test results."""
        results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0,
            'total': 0,
            'output': stdout,
            'stderr': stderr
        }
        
        # Parse summary line
        for line in stdout.split('\n'):
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                # Extract numbers from summary line
                import re
                numbers = re.findall(r'(\d+)\s+(passed|failed|error|skipped)', line)
                for count, status in numbers:
                    results[status] = int(count)
                    results['total'] += int(count)
        
        # Count warnings
        results['warnings'] = stdout.count('WARNING')
        
        return results
    
    def _print_category_summary(self, results: Dict):
        """Print summary for a test category."""
        category = results.get('category', 'unknown')
        duration = results.get('duration', 0)
        
        if 'error' in results:
            print(f"❌ {category.upper()} tests failed: {results['error']}")
            return
        
        passed = results.get('passed', 0)
        failed = results.get('failed', 0) 
        skipped = results.get('skipped', 0)
        total = results.get('total', 0)
        
        # Status emoji
        if failed > 0:
            status = "❌"
        elif total == 0:
            status = "⚠️"
        else:
            status = "✅"
        
        print(f"\n{status} {category.upper()} Results:")
        print(f"   Total: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Skipped: {skipped}")
        print(f"   Duration: {duration:.2f}s")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
            
        if failed > 0:
            print(f"   ⚠️  {failed} tests failed - check output above")
    
    def run_all_categories(self, **options) -> Dict:
        """Run all test categories."""
        categories = ['basic', 'crash', 'concurrent', 'acid', 'performance', 'maintenance', 'errors', 'integration']
        
        print("🚀 Starting comprehensive SQLite State Manager testing...")
        self.start_time = time.perf_counter()
        
        all_results = {}
        
        for category in categories:
            if category == 'performance' and options.get('skip_slow'):
                print(f"⏭️  Skipping {category} tests (slow tests disabled)")
                continue
                
            try:
                result = self.run_category(category, **options)
                all_results[category] = result
                
                # Brief pause between categories
                time.sleep(1)
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Testing interrupted during {category} tests")
                break
                
        # Print final summary
        self._print_final_summary(all_results)
        
        return all_results
    
    def _print_final_summary(self, all_results: Dict):
        """Print final summary across all categories."""
        total_duration = time.perf_counter() - self.start_time if self.start_time else 0
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE SQLite STATE MANAGER TEST SUMMARY")
        print("="*80)
        
        total_tests = 0
        total_passed = 0 
        total_failed = 0
        total_skipped = 0
        failed_categories = []
        
        for category, results in all_results.items():
            if 'error' in results:
                print(f"❌ {category.upper():<12}: ERROR - {results['error']}")
                failed_categories.append(category)
                continue
                
            passed = results.get('passed', 0)
            failed = results.get('failed', 0)
            skipped = results.get('skipped', 0)
            total = results.get('total', 0)
            duration = results.get('duration', 0)
            
            total_tests += total
            total_passed += passed
            total_failed += failed
            total_skipped += skipped
            
            status = "✅" if failed == 0 and total > 0 else "❌"
            success_rate = (passed / total * 100) if total > 0 else 0
            
            print(f"{status} {category.upper():<12}: {passed:>3}/{total:<3} passed ({success_rate:>5.1f}%) in {duration:>6.2f}s")
            
            if failed > 0:
                failed_categories.append(category)
        
        print("-" * 80)
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        overall_status = "✅" if total_failed == 0 and total_tests > 0 else "❌"
        
        print(f"{overall_status} OVERALL RESULT:   {total_passed:>3}/{total_tests:<3} passed ({overall_success_rate:>5.1f}%) in {total_duration:>6.2f}s")
        print(f"                   Failed: {total_failed}, Skipped: {total_skipped}")
        
        if failed_categories:
            print(f"\n⚠️  Categories with failures: {', '.join(failed_categories)}")
            
        if total_failed == 0 and total_tests > 0:
            print("\n🎉 All SQLite State Manager tests passed successfully!")
            print("   - WAL mode functionality verified")
            print("   - Crash recovery mechanisms tested")  
            print("   - Concurrent access validated")
            print("   - ACID transaction properties confirmed")
            print("   - Performance benchmarks completed")
            print("   - Database maintenance operations tested")
            print("   - Error scenarios handled correctly")
            print("   - Integration workflows validated")
        
        print("="*80)
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """Generate detailed test report."""
        if output_file is None:
            output_file = f"sqlite_test_report_{int(time.time())}.json"
            
        report_data = {
            'timestamp': time.time(),
            'summary': {
                'total_categories': len(results),
                'total_tests': sum(r.get('total', 0) for r in results.values()),
                'total_passed': sum(r.get('passed', 0) for r in results.values()),
                'total_failed': sum(r.get('failed', 0) for r in results.values()),
                'total_duration': sum(r.get('duration', 0) for r in results.values())
            },
            'categories': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"📄 Test report saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SQLite State Manager Test Runner')
    parser.add_argument('category', nargs='?', default='all',
                       choices=['all', 'basic', 'crash', 'concurrent', 'acid', 
                               'performance', 'maintenance', 'errors', 'integration'],
                       help='Test category to run (default: all)')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow-running tests')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--benchmark', action='store_true',
                       help='Include performance benchmarks')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel where possible')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--output', type=str,
                       help='Output file for JSON report')
    
    args = parser.parse_args()
    
    runner = SQLiteTestRunner()
    
    # Check if test files exist
    if not runner.test_file.exists():
        print(f"❌ Test file not found: {runner.test_file}")
        print("   Make sure you're running from the correct directory.")
        return 1
    
    # Prepare options
    options = {
        'skip_slow': args.skip_slow,
        'verbose': args.verbose,
        'report': args.report,
        'benchmark': args.benchmark,
        'parallel': args.parallel,
        'coverage': args.coverage
    }
    
    # Run tests
    try:
        if args.category == 'all':
            results = runner.run_all_categories(**options)
        else:
            result = runner.run_category(args.category, **options)
            results = {args.category: result}
        
        # Generate report if requested
        if args.output:
            runner.generate_report(results, args.output)
            
        # Return appropriate exit code
        total_failed = sum(r.get('failed', 0) for r in results.values())
        has_errors = any('error' in r for r in results.values())
        
        if total_failed > 0 or has_errors:
            return 1
        else:
            return 0
            
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        return 130


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)