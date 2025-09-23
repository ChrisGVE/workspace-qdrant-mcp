#!/usr/bin/env python3
"""
Coverage Summary for 5 Python Modules - Scaled File-by-File Approach
"""

import subprocess
import sys
import time

def measure_module_coverage(module_path, test_file):
    """Measure coverage for a specific module."""
    try:
        print(f"\n{'='*60}")
        print(f"Testing module: {module_path}")
        print(f"Using test file: {test_file}")
        print(f"{'='*60}")

        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            f"--cov={module_path}",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--tb=no",  # Suppress traceback for cleaner output
            "-q"  # Quiet mode
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_time = time.time()

        # Extract coverage from output
        coverage_pct = "0.00%"
        for line in result.stdout.split('\n'):
            if module_path in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        coverage_pct = part
                        break

        print(f"Time: {end_time - start_time:.1f}s")
        print(f"Coverage: {coverage_pct}")
        print(f"Status: {'PASSED' if result.returncode == 0 else 'FAILED'}")

        return coverage_pct, end_time - start_time, result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT - Test took longer than 120 seconds")
        return "TIMEOUT", 120, False
    except Exception as e:
        print(f"ERROR: {e}")
        return "ERROR", 0, False

def main():
    """Run coverage measurement on all 5 modules."""

    print("SCALING FILE-BY-FILE APPROACH TO 5 PYTHON MODULES")
    print("Target: 30%+ coverage per module within 10 minutes total")
    print("Proven success: config.py achieved 46.30% in <2 minutes")

    modules_and_tests = [
        ("src.python.common.core.hybrid_search", "20250923-1638_test_hybrid_search_focused.py"),
        ("src.python.common.core.sparse_vectors", "20250923-1640_test_sparse_vectors_focused.py"),
        ("src.python.common.core.metadata_schema", "20250923-1641_test_metadata_schema_focused.py"),
        ("src.python.common.core.advanced_watch_config", "20250923-1642_test_advanced_watch_config.py"),
        ("src.python.common.core.graceful_degradation", "20250923-1643_test_graceful_degradation.py")
    ]

    results = []
    total_start = time.time()

    for module_path, test_file in modules_and_tests:
        coverage, duration, success = measure_module_coverage(module_path, test_file)
        results.append({
            'module': module_path.split('.')[-1],
            'coverage': coverage,
            'duration': duration,
            'success': success
        })

        # Check if we're over time limit
        elapsed = time.time() - total_start
        if elapsed > 600:  # 10 minutes
            print(f"\nTIMEOUT: Exceeded 10-minute limit at {elapsed:.1f} seconds")
            break

    total_time = time.time() - total_start

    print(f"\n{'='*80}")
    print("FINAL RESULTS - SCALED FILE-BY-FILE APPROACH")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f} seconds")
    print()

    successful_modules = 0
    coverage_achieved = []

    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['module']:<25} | {result['coverage']:<8} | {result['duration']:.1f}s")

        if result['success'] and result['coverage'] not in ['TIMEOUT', 'ERROR']:
            successful_modules += 1
            try:
                pct = float(result['coverage'].replace('%', ''))
                coverage_achieved.append(pct)
            except:
                pass

    print(f"\n{'='*80}")
    print("ACHIEVEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Modules successfully tested: {successful_modules}/5")
    print(f"Average coverage achieved: {sum(coverage_achieved)/len(coverage_achieved):.2f}%" if coverage_achieved else "0.00%")
    print(f"Modules meeting 30% target: {len([c for c in coverage_achieved if c >= 30.0])}")
    print(f"Total time efficiency: {total_time/60:.1f} minutes of 10-minute limit")

    if successful_modules >= 3 and total_time <= 600:
        print("\n🎯 SCALING SUCCESS: Proven approach successfully scaled to multiple modules!")
    else:
        print("\n⚠️ SCALING CHALLENGE: Approach needs refinement for better scalability")

if __name__ == "__main__":
    main()