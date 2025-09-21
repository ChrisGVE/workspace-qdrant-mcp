#!/usr/bin/env python3
"""
Continuous Quality Monitor - Zero Tolerance Test Enforcement

This script implements the aggressive quality loop with:
- Continuous test monitoring every 5 minutes
- Immediate failure identification and remediation
- Zero tolerance for any test failures
- Parallel test execution with pytest-xdist
- Performance optimization and reporting
"""

import asyncio
import subprocess
import time
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import signal

class ContinuousQualityMonitor:
    """Implements zero-tolerance continuous quality monitoring."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.monitoring = True
        self.test_interval = 300  # 5 minutes
        self.cycle_count = 0
        self.last_success_time = None
        self.failure_count = 0
        self.consecutive_passes = 0

    async def run_full_test_suite(self) -> Tuple[bool, Dict]:
        """Run complete test suite with parallel execution."""
        cmd = [
            "uv", "run", "pytest",
            "--tb=short",
            "-n", "auto",  # Use all CPU cores
            "--maxfail=10",  # Stop after 10 failures
            "--timeout=300",  # 5 minute timeout per test
            "--strict-markers",  # Enforce proper test markers
            "-v"
        ]

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max for full suite
            )
            execution_time = time.time() - start_time

            return result.returncode == 0, {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "test_count": self._extract_test_count(result.stdout)
            }
        except subprocess.TimeoutExpired:
            return False, {"error": "Test suite timeout - over 30 minutes"}
        except Exception as e:
            return False, {"error": str(e)}

    def _extract_test_count(self, output: str) -> Dict[str, int]:
        """Extract test statistics from pytest output."""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        # Parse the summary line
        for line in output.split('\n'):
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                # Extract numbers from summary line
                import re
                numbers = re.findall(r'(\d+)\s+(passed|failed|skipped|error)', line)
                for count, status in numbers:
                    if status == 'error':
                        status = 'errors'
                    stats[status] = int(count)
                break

        return stats

    async def analyze_and_fix_failures(self, results: Dict) -> int:
        """Analyze failures and attempt automatic fixes."""
        stdout = results.get("stdout", "")
        stderr = results.get("stderr", "")

        # Count collection errors
        collection_errors = stdout.count("ERROR collecting")
        test_failures = stdout.count("FAILED ")

        fixes_applied = 0

        # Try to fix collection errors first (most critical)
        if collection_errors > 0:
            fixes_applied += await self._fix_collection_errors(stdout)

        # Try to fix test failures
        if test_failures > 0:
            fixes_applied += await self._fix_test_failures(stdout)

        return fixes_applied

    async def _fix_collection_errors(self, output: str) -> int:
        """Attempt to fix collection errors automatically."""
        fixes = 0

        # Look for missing dependencies
        if "ModuleNotFoundError:" in output:
            missing_modules = self._extract_missing_modules(output)
            for module in missing_modules:
                if await self._install_missing_module(module):
                    fixes += 1
                    print(f"✅ Installed missing module: {module}")

        # Look for import errors
        if "ImportError:" in output:
            import_errors = self._extract_import_errors(output)
            for error in import_errors:
                if await self._fix_import_error(error):
                    fixes += 1
                    print(f"✅ Fixed import error: {error[:50]}...")

        return fixes

    def _extract_missing_modules(self, output: str) -> List[str]:
        """Extract missing module names from error output."""
        import re
        modules = []

        pattern = r"No module named '([^']+)'"
        matches = re.findall(pattern, output)

        # Filter for likely external dependencies
        external_modules = []
        for module in matches:
            if not module.startswith(('common.', 'workspace_qdrant_mcp.', 'wqm_cli.')):
                external_modules.append(module.split('.')[0])  # Get base module

        return list(set(external_modules))  # Remove duplicates

    def _extract_import_errors(self, output: str) -> List[str]:
        """Extract import error details."""
        import re
        errors = []

        # Find import error patterns
        pattern = r"ImportError:.*cannot import name '([^']+)' from '([^']+)'"
        matches = re.findall(pattern, output)

        for name, module in matches:
            errors.append(f"cannot import {name} from {module}")

        return errors

    async def _install_missing_module(self, module: str) -> bool:
        """Install missing external module."""
        known_modules = {
            'playwright': 'playwright',
            'docker': 'docker',
            'redis': 'redis',
            'psutil': 'psutil',
        }

        if module in known_modules:
            try:
                cmd = ["uv", "add", known_modules[module]]
                result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
                return result.returncode == 0
            except Exception:
                return False

        return False

    async def _fix_import_error(self, error: str) -> bool:
        """Attempt to fix specific import errors."""
        # This is a placeholder for more sophisticated import fixing
        # Could implement class name mapping, module path corrections, etc.
        return False

    async def _fix_test_failures(self, output: str) -> int:
        """Attempt to fix actual test failures."""
        # Placeholder for test failure analysis and fixing
        # Could implement:
        # - Mock missing services
        # - Fix configuration issues
        # - Update test expectations
        return 0

    async def run_quality_monitor(self):
        """Main continuous monitoring loop."""
        print("🚀 CONTINUOUS QUALITY MONITOR - ZERO TOLERANCE MODE")
        print("=" * 60)
        print(f"📁 Project: {self.project_root}")
        print(f"⏰ Test Interval: {self.test_interval} seconds")
        print(f"🎯 Target: 100% test passing rate")
        print("=" * 60)

        # Initial test run
        await self._run_monitoring_cycle()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Main monitoring loop
        while self.monitoring:
            await asyncio.sleep(self.test_interval)
            if self.monitoring:
                await self._run_monitoring_cycle()

    async def _run_monitoring_cycle(self):
        """Run a single monitoring cycle."""
        self.cycle_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'🔄' * 20} CYCLE #{self.cycle_count} {'🔄' * 20}")
        print(f"⏰ Time: {timestamp}")

        # Run tests
        print("🧪 Running comprehensive test suite...")
        success, results = await self.run_full_test_suite()

        execution_time = results.get("execution_time", 0)
        test_stats = results.get("test_count", {})

        if success:
            self.consecutive_passes += 1
            self.last_success_time = datetime.now()
            print(f"✅ ALL TESTS PASSED! ({execution_time:.1f}s)")
            print(f"📊 Stats: {test_stats}")
            print(f"🏆 Consecutive passes: {self.consecutive_passes}")
            self.failure_count = 0  # Reset failure count

        else:
            self.failure_count += 1
            self.consecutive_passes = 0
            print(f"❌ TEST FAILURES DETECTED! (Attempt #{self.failure_count})")
            print(f"⏱️  Execution time: {execution_time:.1f}s")
            print(f"📊 Stats: {test_stats}")

            # Attempt automatic fixes
            print("\n🔧 ATTEMPTING AUTOMATIC REMEDIATION...")
            fixes_applied = await self.analyze_and_fix_failures(results)

            if fixes_applied > 0:
                print(f"✅ Applied {fixes_applied} automatic fixes")
                print("🔄 Re-running tests to validate fixes...")

                # Re-run tests after fixes
                success_after_fix, _ = await self.run_full_test_suite()
                if success_after_fix:
                    print("🎉 FIXES SUCCESSFUL - All tests now passing!")
                    self.consecutive_passes = 1
                    self.failure_count = 0
                else:
                    print("⚠️  Some issues remain - manual intervention required")
            else:
                print("❌ No automatic fixes available - manual intervention required")

        # Quality metrics
        if self.last_success_time:
            time_since_success = datetime.now() - self.last_success_time
            print(f"⏰ Time since last success: {time_since_success}")

        print(f"📈 Quality Score: {self._calculate_quality_score():.1f}%")
        print("=" * 60)

    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score based on recent performance."""
        if self.cycle_count == 0:
            return 0.0

        # Base score from consecutive passes
        base_score = min(100.0, (self.consecutive_passes / max(1, self.cycle_count)) * 100)

        # Penalty for failures
        failure_penalty = min(50.0, self.failure_count * 10)

        return max(0.0, base_score - failure_penalty)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n🛑 Received signal {signum} - shutting down gracefully...")
        self.monitoring = False

async def main():
    """Main entry point."""
    project_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
    monitor = ContinuousQualityMonitor(project_root)

    try:
        await monitor.run_quality_monitor()
    except KeyboardInterrupt:
        print("\n🛑 Quality monitoring stopped by user")
    except Exception as e:
        print(f"💥 Quality monitor failed: {e}")
        sys.exit(1)
    finally:
        print("👋 Quality monitor shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())