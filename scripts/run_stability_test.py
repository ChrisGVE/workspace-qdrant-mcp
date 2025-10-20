#!/usr/bin/env python3
"""
Standalone stability test runner.

Executes extended stability tests separately from the main test suite.
Supports various test durations and provides real-time monitoring.

Usage:
    # Run 1-hour baseline test
    python scripts/run_stability_test.py --duration 1h

    # Run 6-hour test
    python scripts/run_stability_test.py --duration 6h

    # Run full 24-hour test
    python scripts/run_stability_test.py --duration 24h

    # Run with custom output directory
    python scripts/run_stability_test.py --duration 6h --output ./stability_results
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_stability_test(duration: str, output_dir: Path, verbose: bool = False) -> int:
    """
    Run stability test for specified duration.

    Args:
        duration: Test duration (1h, 6h, 24h)
        output_dir: Directory for test results
        verbose: Enable verbose output

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    # Map duration to pytest markers
    duration_map = {
        "1h": "test_one_hour_stability_baseline",
        "6h": "test_six_hour_stability",
        "24h": "test_24_hour_stability",
    }

    if duration not in duration_map:
        print(f"Error: Invalid duration '{duration}'. Must be one of: 1h, 6h, 24h")
        return 1

    test_name = duration_map[duration]

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"stability_test_{duration}_{timestamp}.log"
    junit_file = output_dir / f"stability_test_{duration}_{timestamp}.xml"

    print(f"Starting {duration} stability test...")
    print(f"Test: {test_name}")
    print(f"Log file: {log_file}")
    print(f"JUnit XML: {junit_file}")
    print()

    # Build pytest command
    cmd = [
        "uv",
        "run",
        "pytest",
        f"tests/e2e/test_stability.py::{test_name}",
        "-v",
        "--tb=short",
        f"--junit-xml={junit_file}",
        "-o",
        f"log_file={log_file}",
        "--log-file-level=INFO",
    ]

    if verbose:
        cmd.append("-s")  # Show print statements

    print(f"Running command: {' '.join(cmd)}")
    print()

    # Run test
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running test: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run extended stability tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 1-hour baseline test
  python scripts/run_stability_test.py --duration 1h

  # Run 6-hour test with custom output
  python scripts/run_stability_test.py --duration 6h --output ./my_results

  # Run 24-hour test with verbose output
  python scripts/run_stability_test.py --duration 24h --verbose

Duration options:
  1h  - 1 hour baseline test (recommended for CI)
  6h  - 6 hour extended test
  24h - Full 24 hour stability test
        """,
    )

    parser.add_argument(
        "--duration",
        required=True,
        choices=["1h", "6h", "24h"],
        help="Test duration (1h, 6h, or 24h)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./stability_test_results"),
        help="Output directory for test results (default: ./stability_test_results)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (show print statements)",
    )

    args = parser.parse_args()

    # Run test
    exit_code = run_stability_test(
        duration=args.duration,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # Print results
    if exit_code == 0:
        print("\n✓ Stability test completed successfully")
    else:
        print(f"\n✗ Stability test failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
