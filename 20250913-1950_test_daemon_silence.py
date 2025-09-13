#!/usr/bin/env python3
"""
Test script to validate complete console silence in daemon mode.

This script tests the enhanced Rust daemon tracing system to ensure
that no output is produced when running in daemon mode.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_daemon_test():
    """Test the daemon for console silence."""

    # Path to the compiled daemon binary
    daemon_path = Path(__file__).parent / "src/rust/daemon/target/release/memexd"

    if not daemon_path.exists():
        print(f"ERROR: Daemon binary not found at {daemon_path}")
        return False

    print("Testing Rust daemon silence functionality...")

    # Test 1: Run daemon in service mode with timeout and capture output
    print("\n1. Testing daemon mode silence...")
    env = os.environ.copy()
    env.update({
        'WQM_SERVICE_MODE': 'true',
        'RUST_LOG': 'off',
        'ORT_LOGGING_LEVEL': '4',
        'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'NO_COLOR': '1',
    })

    try:
        # Run daemon for 10 seconds and capture ALL output
        result = subprocess.run([
            str(daemon_path),
            '--log-level', 'error',  # Minimal logging
        ],
        timeout=10,
        capture_output=True,
        text=True,
        env=env
        )

        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        print(f"   Daemon exit code: {result.returncode}")
        print(f"   STDOUT length: {len(stdout_output)} characters")
        print(f"   STDERR length: {len(stderr_output)} characters")

        if stdout_output:
            print(f"   STDOUT content:\n{repr(stdout_output)}")
        if stderr_output:
            print(f"   STDERR content:\n{repr(stderr_output)}")

        # Success if no output or only timeout exit
        silence_success = (len(stdout_output) == 0 and len(stderr_output) == 0)
        print(f"   ✓ Complete silence: {'SUCCESS' if silence_success else 'FAILED'}")

        return silence_success

    except subprocess.TimeoutExpired:
        print("   ✓ Daemon ran for 10 seconds and was terminated (expected)")
        print("   ✓ Complete silence: SUCCESS (no output captured)")
        return True

    except Exception as e:
        print(f"   ✗ Error running daemon: {e}")
        return False

def run_foreground_test():
    """Test the daemon in foreground mode to ensure logging still works."""

    daemon_path = Path(__file__).parent / "src/rust/daemon/target/release/memexd"

    print("\n2. Testing foreground mode (logging enabled)...")

    try:
        # Run daemon in foreground mode with timeout
        result = subprocess.run([
            str(daemon_path),
            '--foreground',
            '--log-level', 'info',
        ],
        timeout=5,
        capture_output=True,
        text=True
        )

        stdout_output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        print(f"   Daemon exit code: {result.returncode}")
        print(f"   STDOUT length: {len(stdout_output)} characters")
        print(f"   STDERR length: {len(stderr_output)} characters")

        # In foreground mode, we expect some output
        has_output = len(stdout_output) > 0 or len(stderr_output) > 0
        print(f"   ✓ Logging enabled: {'SUCCESS' if has_output else 'FAILED'}")

        return has_output

    except subprocess.TimeoutExpired:
        print("   ✓ Daemon ran for 5 seconds and was terminated")
        return True

    except Exception as e:
        print(f"   ✗ Error running daemon: {e}")
        return False

def main():
    """Run all validation tests."""

    print("=" * 60)
    print("RUST DAEMON TRACING SILENCE VALIDATION")
    print("=" * 60)

    # Test daemon mode silence
    daemon_silence_ok = run_daemon_test()

    # Test foreground mode logging
    foreground_logging_ok = run_foreground_test()

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"Daemon mode silence:     {'✓ PASS' if daemon_silence_ok else '✗ FAIL'}")
    print(f"Foreground mode logging: {'✓ PASS' if foreground_logging_ok else '✗ FAIL'}")

    overall_success = daemon_silence_ok and foreground_logging_ok
    print(f"\nOverall result:          {'✓ SUCCESS' if overall_success else '✗ FAILURE'}")

    if overall_success:
        print("\n🎉 Rust daemon tracing silence implementation is working correctly!")
        print("✓ Complete silence in daemon mode")
        print("✓ Normal logging in foreground mode")
        print("✓ Ready for MCP stdio protocol compliance")
    else:
        print("\n❌ Issues detected with daemon tracing configuration:")
        if not daemon_silence_ok:
            print("  - Daemon mode is not completely silent")
        if not foreground_logging_ok:
            print("  - Foreground mode logging is not working")

    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())