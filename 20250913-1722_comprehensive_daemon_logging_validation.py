#!/usr/bin/env python3
"""
Comprehensive End-to-End Validation Suite for Daemon Logging Remediation

This test suite validates that all daemon logging issues have been resolved and
MCP stdio compliance is achieved through systematic testing of:

1. Environment variable suppression (ORT, tokenizers, TTY detection)
2. Fixed embedding println! calls (now use tracing)
3. Fixed CLI error handling (no unwrap panics)
4. Tracing system daemon mode configuration

Test Categories:
- Daemon Mode Silence Test
- MCP Protocol Compliance
- Functionality Preservation
- Regression Testing
- Performance Impact Assessment

Success Criteria:
- Zero console output in daemon mode
- All critical functionality preserved
- MCP stdio protocol ready
- No regressions in interactive mode
- Performance impact < 1%
"""

import subprocess
import sys
import time
import os
import json
import psutil
import tempfile
import signal
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

@dataclass
class TestResult:
    """Structure for individual test results"""
    name: str
    category: str
    passed: bool
    execution_time_ms: float
    details: Dict[str, Any]
    output_stdout: str = ""
    output_stderr: str = ""
    error_message: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance measurement data"""
    startup_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    process_count: int

@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: str
    daemon_binary_path: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[TestResult]
    performance_baseline: Optional[PerformanceMetrics]
    performance_daemon: Optional[PerformanceMetrics]
    performance_impact_percent: float
    overall_passed: bool
    recommendations: List[str]

class DaemonValidationSuite:
    """Comprehensive validation test suite for daemon logging fixes"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.daemon_path = self.find_daemon_binary()
        self.test_results: List[TestResult] = []
        self.temp_files: List[Path] = []

    def find_daemon_binary(self) -> Path:
        """Find the most recent daemon binary"""
        potential_paths = [
            self.project_root / "src/rust/daemon/target/release/memexd",
            self.project_root / "rust-engine/target/release/memexd",
            self.project_root / "target/release/memexd"
        ]

        for path in potential_paths:
            if path.exists():
                return path

        raise FileNotFoundError("No memexd daemon binary found. Please build first with 'cargo build --release'")

    def setup_test_environment(self, daemon_mode: bool = True) -> Dict[str, str]:
        """Setup environment variables for testing"""
        env = os.environ.copy()

        if daemon_mode:
            env.update({
                'WQM_SERVICE_MODE': 'true',
                'ORT_LOGGING_LEVEL': '4',
                'OMP_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false',
                'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                'HF_HUB_DISABLE_TELEMETRY': '1',
                'NO_COLOR': '1',
                'TERM': 'dumb',
                'RUST_BACKTRACE': '0',
            })
        else:
            # Interactive mode - clean environment
            for key in ['WQM_SERVICE_MODE', 'NO_COLOR', 'TERM']:
                env.pop(key, None)

        return env

    def run_daemon_test(self, args: List[str], env: Dict[str, str],
                       timeout: int = 10) -> Tuple[subprocess.CompletedProcess, float]:
        """Run daemon with specified args and environment, measuring execution time"""
        start_time = time.time()

        try:
            result = subprocess.run(
                [str(self.daemon_path)] + args,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=env
            )
            execution_time = (time.time() - start_time) * 1000
            return result, execution_time

        except subprocess.TimeoutExpired as e:
            execution_time = (time.time() - start_time) * 1000
            # Create a mock result for timeout case
            result = subprocess.CompletedProcess(
                args=[str(self.daemon_path)] + args,
                returncode=-signal.SIGTERM,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else ""
            )
            return result, execution_time

    def measure_performance(self, args: List[str], env: Dict[str, str]) -> PerformanceMetrics:
        """Measure performance metrics for daemon execution"""
        # Start daemon process
        process = subprocess.Popen(
            [str(self.daemon_path)] + args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Give it time to initialize
            time.sleep(2)

            # Measure performance
            proc = psutil.Process(process.pid)
            memory_info = proc.memory_info()
            cpu_percent = proc.cpu_percent(interval=1)

            metrics = PerformanceMetrics(
                startup_time_ms=2000,  # Estimated from sleep time
                memory_usage_mb=memory_info.rss / 1024 / 1024,
                cpu_usage_percent=cpu_percent,
                process_count=len(proc.children(recursive=True)) + 1
            )

        finally:
            # Clean up
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        return metrics

    # Test Category 1: Daemon Mode Silence Tests

    def test_pure_daemon_silence(self) -> TestResult:
        """Test daemon runs completely silently in service mode"""
        env = self.setup_test_environment(daemon_mode=True)
        result, exec_time = self.run_daemon_test([], env, timeout=8)

        stdout_len = len(result.stdout)
        stderr_len = len(result.stderr)
        silence_achieved = stdout_len == 0 and stderr_len == 0

        return TestResult(
            name="Pure Daemon Silence",
            category="Silence Validation",
            passed=silence_achieved,
            execution_time_ms=exec_time,
            details={
                "stdout_length": stdout_len,
                "stderr_length": stderr_len,
                "exit_code": result.returncode,
                "expected_timeout": True
            },
            output_stdout=result.stdout,
            output_stderr=result.stderr
        )

    def test_daemon_with_log_levels(self) -> TestResult:
        """Test daemon silence across different log levels"""
        env = self.setup_test_environment(daemon_mode=True)

        all_silent = True
        details = {}
        combined_stdout = ""
        combined_stderr = ""

        for level in ["error", "warn", "info", "debug"]:
            result, _ = self.run_daemon_test(["--log-level", level], env, timeout=5)

            stdout_len = len(result.stdout)
            stderr_len = len(result.stderr)
            level_silent = stdout_len == 0 and stderr_len == 0

            details[f"log_level_{level}"] = {
                "silent": level_silent,
                "stdout_length": stdout_len,
                "stderr_length": stderr_len
            }

            if not level_silent:
                all_silent = False
                combined_stdout += f"[{level}] " + result.stdout
                combined_stderr += f"[{level}] " + result.stderr

        return TestResult(
            name="Daemon Log Level Silence",
            category="Silence Validation",
            passed=all_silent,
            execution_time_ms=sum([details[f"log_level_{level}"].get("execution_time", 0) for level in ["error", "warn", "info", "debug"]]),
            details=details,
            output_stdout=combined_stdout,
            output_stderr=combined_stderr
        )

    def test_third_party_suppression(self) -> TestResult:
        """Test that third-party library output is suppressed"""
        env = self.setup_test_environment(daemon_mode=True)

        # Force conditions that would normally cause output
        env.update({
            'RUST_LOG': 'trace',  # This should still be suppressed in daemon mode
            'TOKENIZERS_PARALLELISM': 'true',  # Should be overridden
        })

        result, exec_time = self.run_daemon_test(["--log-level", "trace"], env, timeout=6)

        # Check for specific third-party output patterns that should be suppressed
        problematic_patterns = [
            "ORT:",  # ONNX Runtime
            "Loading tokenizer",
            "Progress:",
            "Downloading",
            "TTY Detection",
            "Terminal capabilities"
        ]

        found_patterns = []
        for pattern in problematic_patterns:
            if pattern.lower() in (result.stdout + result.stderr).lower():
                found_patterns.append(pattern)

        suppression_successful = len(found_patterns) == 0 and len(result.stdout) == 0 and len(result.stderr) == 0

        return TestResult(
            name="Third-Party Output Suppression",
            category="Silence Validation",
            passed=suppression_successful,
            execution_time_ms=exec_time,
            details={
                "problematic_patterns_found": found_patterns,
                "total_output_length": len(result.stdout + result.stderr),
                "environment_overrides": ["TOKENIZERS_PARALLELISM", "RUST_LOG"]
            },
            output_stdout=result.stdout,
            output_stderr=result.stderr
        )

    # Test Category 2: Functionality Preservation Tests

    def test_cli_argument_parsing(self) -> TestResult:
        """Test CLI argument parsing works without panics"""
        env = self.setup_test_environment(daemon_mode=True)

        test_args = [
            ["--help"],  # Should exit gracefully in daemon mode
            ["--version"],  # Should exit gracefully in daemon mode
            ["--invalid-arg"],  # Should handle errors gracefully
            ["--log-level", "invalid"],  # Invalid log level
            ["--config", "/nonexistent/file"],  # Missing config file
        ]

        all_graceful = True
        details = {}

        for i, args in enumerate(test_args):
            try:
                result, exec_time = self.run_daemon_test(args, env, timeout=3)

                # In daemon mode, should exit silently even on errors
                graceful_exit = len(result.stdout) == 0 and len(result.stderr) == 0

                details[f"test_{i}_{args[0]}"] = {
                    "graceful_exit": graceful_exit,
                    "exit_code": result.returncode,
                    "execution_time_ms": exec_time,
                    "output_lengths": {
                        "stdout": len(result.stdout),
                        "stderr": len(result.stderr)
                    }
                }

                if not graceful_exit:
                    all_graceful = False

            except Exception as e:
                details[f"test_{i}_{args[0]}"] = {
                    "graceful_exit": False,
                    "error": str(e)
                }
                all_graceful = False

        return TestResult(
            name="CLI Argument Parsing",
            category="Functionality Preservation",
            passed=all_graceful,
            execution_time_ms=sum([details[k].get("execution_time_ms", 0) for k in details.keys()]),
            details=details
        )

    def test_configuration_loading(self) -> TestResult:
        """Test configuration loading works properly"""
        env = self.setup_test_environment(daemon_mode=True)

        # Create a temporary valid config file
        config_content = """
[qdrant]
transport = "http"
url = "http://localhost:6333"

[processing]
batch_size = 100
"""

        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
        temp_config.write(config_content)
        temp_config.close()
        self.temp_files.append(Path(temp_config.name))

        result, exec_time = self.run_daemon_test(["--config", temp_config.name], env, timeout=5)

        # Should start without output
        config_loaded = len(result.stdout) == 0 and len(result.stderr) == 0

        return TestResult(
            name="Configuration Loading",
            category="Functionality Preservation",
            passed=config_loaded,
            execution_time_ms=exec_time,
            details={
                "config_file_used": temp_config.name,
                "output_suppressed": config_loaded,
                "exit_code": result.returncode
            },
            output_stdout=result.stdout,
            output_stderr=result.stderr
        )

    # Test Category 3: Regression Tests

    def test_interactive_mode_logging(self) -> TestResult:
        """Test that interactive/foreground mode still shows output"""
        env = self.setup_test_environment(daemon_mode=False)

        result, exec_time = self.run_daemon_test(["--foreground", "--log-level", "info"], env, timeout=5)

        # In interactive mode, we expect some output
        has_output = len(result.stdout) > 0 or len(result.stderr) > 0

        # Look for expected log messages
        expected_patterns = ["Starting", "daemon", "memexd"]
        patterns_found = []

        full_output = result.stdout + result.stderr
        for pattern in expected_patterns:
            if pattern.lower() in full_output.lower():
                patterns_found.append(pattern)

        return TestResult(
            name="Interactive Mode Logging",
            category="Regression Testing",
            passed=has_output,
            execution_time_ms=exec_time,
            details={
                "has_output": has_output,
                "expected_patterns_found": patterns_found,
                "total_output_length": len(full_output),
                "exit_code": result.returncode
            },
            output_stdout=result.stdout,
            output_stderr=result.stderr
        )

    def test_debug_mode_functionality(self) -> TestResult:
        """Test debug mode maintains full logging capabilities"""
        env = self.setup_test_environment(daemon_mode=False)

        result, exec_time = self.run_daemon_test(
            ["--foreground", "--log-level", "debug"], env, timeout=4
        )

        # Debug mode should produce more verbose output
        has_debug_output = len(result.stdout + result.stderr) > 0

        return TestResult(
            name="Debug Mode Functionality",
            category="Regression Testing",
            passed=has_debug_output,
            execution_time_ms=exec_time,
            details={
                "debug_output_present": has_debug_output,
                "output_length": len(result.stdout + result.stderr),
                "exit_code": result.returncode
            },
            output_stdout=result.stdout,
            output_stderr=result.stderr
        )

    # Test Category 4: Performance Tests

    def test_performance_impact(self) -> Tuple[TestResult, PerformanceMetrics, PerformanceMetrics]:
        """Test that silence implementation has < 1% performance impact"""

        # Baseline: daemon without silence (interactive mode)
        env_baseline = self.setup_test_environment(daemon_mode=False)
        baseline_metrics = self.measure_performance(["--foreground"], env_baseline)

        # Daemon mode with silence
        env_daemon = self.setup_test_environment(daemon_mode=True)
        daemon_metrics = self.measure_performance([], env_daemon)

        # Calculate performance impact
        memory_impact = ((daemon_metrics.memory_usage_mb - baseline_metrics.memory_usage_mb)
                        / baseline_metrics.memory_usage_mb) * 100

        startup_impact = ((daemon_metrics.startup_time_ms - baseline_metrics.startup_time_ms)
                         / baseline_metrics.startup_time_ms) * 100

        overall_impact = max(abs(memory_impact), abs(startup_impact))

        performance_acceptable = overall_impact < 1.0

        result = TestResult(
            name="Performance Impact Assessment",
            category="Performance Testing",
            passed=performance_acceptable,
            execution_time_ms=baseline_metrics.startup_time_ms + daemon_metrics.startup_time_ms,
            details={
                "memory_impact_percent": memory_impact,
                "startup_impact_percent": startup_impact,
                "overall_impact_percent": overall_impact,
                "acceptable": performance_acceptable,
                "baseline_memory_mb": baseline_metrics.memory_usage_mb,
                "daemon_memory_mb": daemon_metrics.memory_usage_mb,
                "baseline_startup_ms": baseline_metrics.startup_time_ms,
                "daemon_startup_ms": daemon_metrics.startup_time_ms
            }
        )

        return result, baseline_metrics, daemon_metrics

    # Test Category 5: MCP Protocol Compliance Tests

    def test_stdio_protocol_readiness(self) -> TestResult:
        """Test daemon readiness for MCP stdio protocol"""
        env = self.setup_test_environment(daemon_mode=True)

        # Simulate stdio environment
        env.update({
            'STDIN_TYPE': 'pipe',
            'STDOUT_TYPE': 'pipe',
            'MCP_PROTOCOL': 'stdio'
        })

        result, exec_time = self.run_daemon_test([], env, timeout=6)

        # Should be completely silent for JSON-only communication
        stdio_ready = len(result.stdout) == 0 and len(result.stderr) == 0

        return TestResult(
            name="MCP Stdio Protocol Readiness",
            category="MCP Protocol Compliance",
            passed=stdio_ready,
            execution_time_ms=exec_time,
            details={
                "stdio_silence_achieved": stdio_ready,
                "json_communication_ready": stdio_ready,
                "protocol_interference": not stdio_ready,
                "exit_code": result.returncode
            },
            output_stdout=result.stdout,
            output_stderr=result.stderr
        )

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except:
                pass

    def run_all_tests(self) -> ValidationReport:
        """Execute complete test suite and generate report"""
        print("🔍 Starting Comprehensive Daemon Logging Validation Suite")
        print(f"📁 Daemon binary: {self.daemon_path}")
        print("=" * 80)

        try:
            # Verify binary exists and is recent
            if not self.daemon_path.exists():
                raise FileNotFoundError(f"Daemon binary not found: {self.daemon_path}")

            binary_stat = self.daemon_path.stat()
            binary_age_hours = (time.time() - binary_stat.st_mtime) / 3600

            print(f"✓ Binary found (modified {binary_age_hours:.1f} hours ago)")

            # Execute all test categories
            print("\n🧪 Running Silence Validation Tests...")
            self.test_results.append(self.test_pure_daemon_silence())
            self.test_results.append(self.test_daemon_with_log_levels())
            self.test_results.append(self.test_third_party_suppression())

            print("\n🔧 Running Functionality Preservation Tests...")
            self.test_results.append(self.test_cli_argument_parsing())
            self.test_results.append(self.test_configuration_loading())

            print("\n🔄 Running Regression Tests...")
            self.test_results.append(self.test_interactive_mode_logging())
            self.test_results.append(self.test_debug_mode_functionality())

            print("\n⚡ Running Performance Tests...")
            perf_result, baseline_metrics, daemon_metrics = self.test_performance_impact()
            self.test_results.append(perf_result)

            print("\n📡 Running MCP Protocol Compliance Tests...")
            self.test_results.append(self.test_stdio_protocol_readiness())

            # Generate comprehensive report
            passed_tests = sum(1 for result in self.test_results if result.passed)
            failed_tests = len(self.test_results) - passed_tests
            overall_passed = failed_tests == 0

            # Calculate overall performance impact
            overall_perf_impact = perf_result.details["overall_impact_percent"]

            # Generate recommendations
            recommendations = self.generate_recommendations(overall_passed, self.test_results)

            report = ValidationReport(
                timestamp=datetime.now().isoformat(),
                daemon_binary_path=str(self.daemon_path),
                total_tests=len(self.test_results),
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                test_results=self.test_results,
                performance_baseline=baseline_metrics,
                performance_daemon=daemon_metrics,
                performance_impact_percent=overall_perf_impact,
                overall_passed=overall_passed,
                recommendations=recommendations
            )

            return report

        finally:
            self.cleanup()

    def generate_recommendations(self, overall_passed: bool, results: List[TestResult]) -> List[str]:
        """Generate deployment recommendations based on test results"""
        recommendations = []

        if overall_passed:
            recommendations.extend([
                "✅ All tests passed - daemon is ready for production deployment",
                "✅ MCP stdio protocol compliance achieved",
                "✅ Performance impact is within acceptable limits (<1%)",
                "✅ No regressions detected in interactive mode",
                "🚀 Recommended action: Deploy to production"
            ])
        else:
            recommendations.append("❌ Issues detected - review failures before deployment")

            # Analyze specific failure patterns
            failed_by_category = {}
            for result in results:
                if not result.passed:
                    category = result.category
                    if category not in failed_by_category:
                        failed_by_category[category] = []
                    failed_by_category[category].append(result.name)

            for category, failed_tests in failed_by_category.items():
                recommendations.append(f"🔍 {category} issues: {', '.join(failed_tests)}")

        return recommendations

def print_detailed_report(report: ValidationReport):
    """Print comprehensive validation report"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE DAEMON LOGGING VALIDATION REPORT")
    print("=" * 80)

    print(f"🕒 Timestamp: {report.timestamp}")
    print(f"📁 Binary: {report.daemon_binary_path}")
    print(f"📋 Total Tests: {report.total_tests}")
    print(f"✅ Passed: {report.passed_tests}")
    print(f"❌ Failed: {report.failed_tests}")
    print(f"⚡ Performance Impact: {report.performance_impact_percent:.2f}%")
    print(f"🎯 Overall Status: {'✅ SUCCESS' if report.overall_passed else '❌ FAILURE'}")

    print("\n" + "─" * 80)
    print("📊 TEST RESULTS BY CATEGORY")
    print("─" * 80)

    # Group results by category
    by_category = {}
    for result in report.test_results:
        if result.category not in by_category:
            by_category[result.category] = []
        by_category[result.category].append(result)

    for category, results in by_category.items():
        print(f"\n📂 {category}")
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {status} {result.name} ({result.execution_time_ms:.1f}ms)")

            if not result.passed:
                if result.error_message:
                    print(f"      ❌ Error: {result.error_message}")
                if result.output_stdout:
                    print(f"      📄 STDOUT: {repr(result.output_stdout[:100])}")
                if result.output_stderr:
                    print(f"      📄 STDERR: {repr(result.output_stderr[:100])}")

    # Performance Details
    if report.performance_baseline and report.performance_daemon:
        print(f"\n" + "─" * 80)
        print("⚡ PERFORMANCE METRICS")
        print("─" * 80)
        print(f"Baseline (Interactive Mode):")
        print(f"  Memory: {report.performance_baseline.memory_usage_mb:.1f} MB")
        print(f"  Startup: {report.performance_baseline.startup_time_ms:.1f} ms")
        print(f"  CPU: {report.performance_baseline.cpu_usage_percent:.1f}%")
        print(f"\nDaemon (Silent Mode):")
        print(f"  Memory: {report.performance_daemon.memory_usage_mb:.1f} MB")
        print(f"  Startup: {report.performance_daemon.startup_time_ms:.1f} ms")
        print(f"  CPU: {report.performance_daemon.cpu_usage_percent:.1f}%")
        print(f"\nOverall Impact: {report.performance_impact_percent:.2f}%")

    # Recommendations
    print(f"\n" + "─" * 80)
    print("🎯 RECOMMENDATIONS")
    print("─" * 80)
    for recommendation in report.recommendations:
        print(f"  {recommendation}")

    print("\n" + "=" * 80)

def save_report_to_file(report: ValidationReport, filename: str):
    """Save report to JSON file for analysis"""
    report_dict = asdict(report)

    with open(filename, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)

    print(f"📄 Detailed report saved to: {filename}")

def main():
    """Execute comprehensive validation suite"""
    suite = DaemonValidationSuite()

    try:
        report = suite.run_all_tests()
        print_detailed_report(report)

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        report_filename = f"{timestamp}_daemon_logging_validation_report.json"
        save_report_to_file(report, report_filename)

        # Return appropriate exit code
        return 0 if report.overall_passed else 1

    except Exception as e:
        print(f"\n❌ Critical error during validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())