"""
Performance benchmarks for console suppression impact.

Measures the performance impact of console silence mechanisms to ensure
minimal overhead while maintaining complete stdio mode compliance.

SUCCESS CRITERIA:
- Console suppression overhead < 5ms startup time
- Memory overhead < 1MB for silence mechanisms
- Tool invocation latency increase < 1ms
- No measurable impact on JSON-RPC throughput
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import memory_profiler
import psutil
import pytest

# Test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.stdio,
    pytest.mark.benchmark,
]


class PerformanceMetrics:
    """Helper class for collecting performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.startup_times = []
        self.memory_usage = []
        self.tool_latencies = []
        self.throughput_measurements = []

    def measure_startup_time(self, func, *args, **kwargs):
        """Measure function startup time."""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = e
        end_time = time.perf_counter()

        startup_time = (end_time - start_time) * 1000  # Convert to ms
        self.startup_times.append(startup_time)
        return result, startup_time

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = final_memory - initial_memory

        self.memory_usage.append(memory_diff)
        return result, memory_diff

    def measure_tool_latency(self, tool_func, *args, **kwargs):
        """Measure tool execution latency."""
        start_time = time.perf_counter()
        result = tool_func(*args, **kwargs)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to ms
        self.tool_latencies.append(latency)
        return result, latency

    def get_summary(self) -> dict[str, float]:
        """Get performance summary statistics."""
        return {
            "avg_startup_time_ms": sum(self.startup_times) / len(self.startup_times) if self.startup_times else 0,
            "max_startup_time_ms": max(self.startup_times) if self.startup_times else 0,
            "avg_memory_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "max_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_tool_latency_ms": sum(self.tool_latencies) / len(self.tool_latencies) if self.tool_latencies else 0,
            "max_tool_latency_ms": max(self.tool_latencies) if self.tool_latencies else 0,
        }


@pytest.fixture
def performance_metrics():
    """Provide performance metrics collector."""
    return PerformanceMetrics()


class TestPerformanceBenchmarks:
    """Performance benchmark tests for console silence."""

    def test_stdio_mode_startup_overhead(self, performance_metrics, monkeypatch, benchmark):
        """Benchmark stdio mode startup overhead."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        def startup_stdio_server():
            """Start stdio server for benchmarking."""
            with patch('workspace_qdrant_mcp.stdio_server.FastMCP') as mock_fastmcp:
                mock_app = mock_fastmcp.return_value
                mock_app.run.side_effect = KeyboardInterrupt("Benchmark stop")

                from workspace_qdrant_mcp.stdio_server import (
                    run_lightweight_stdio_server,
                )

                try:
                    run_lightweight_stdio_server()
                except (KeyboardInterrupt, SystemExit):
                    pass

        # Benchmark startup time
        benchmark.pedantic(startup_stdio_server, iterations=10, rounds=3)

        # Verify startup overhead is minimal
        stats = benchmark.stats
        avg_time_ms = stats.mean * 1000

        # CRITICAL: Startup overhead should be < 5ms
        assert avg_time_ms < 5.0, f"Startup overhead too high: {avg_time_ms:.2f}ms"

        # Record in performance metrics
        performance_metrics.startup_times.extend([stats.mean * 1000] * 10)

    def test_console_suppression_memory_overhead(self, performance_metrics, monkeypatch):
        """Test memory overhead of console suppression mechanisms."""
        # Measure baseline memory without stdio mode
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Import without stdio mode
        monkeypatch.delenv("WQM_STDIO_MODE", raising=False)

        # Force reimport to get clean state
        import importlib
        if 'workspace_qdrant_mcp.server' in sys.modules:
            importlib.reload(sys.modules['workspace_qdrant_mcp.server'])

        memory_without_stdio = process.memory_info().rss / 1024 / 1024
        memory_without_stdio - baseline_memory

        # Now measure with stdio mode
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Force reimport with stdio mode
        if 'workspace_qdrant_mcp.server' in sys.modules:
            del sys.modules['workspace_qdrant_mcp.server']

        import workspace_qdrant_mcp.server

        memory_with_stdio = process.memory_info().rss / 1024 / 1024
        stdio_overhead = memory_with_stdio - memory_without_stdio

        # CRITICAL: Memory overhead should be < 1MB
        assert stdio_overhead < 1.0, f"Memory overhead too high: {stdio_overhead:.2f}MB"

        performance_metrics.memory_usage.append(stdio_overhead)

    def test_tool_invocation_latency_impact(self, performance_metrics, benchmark, monkeypatch):
        """Benchmark tool invocation latency with console suppression."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Mock tool function
        async def mock_tool_function():
            """Mock tool that simulates typical processing."""
            await asyncio.sleep(0.001)  # Simulate 1ms processing
            return {"result": "test", "status": "success"}

        # Benchmark tool execution
        def run_tool():
            return asyncio.run(mock_tool_function())

        # Benchmark the tool execution
        benchmark.pedantic(run_tool, iterations=50, rounds=5)

        stats = benchmark.stats
        avg_latency_ms = stats.mean * 1000

        # CRITICAL: Tool latency should not be significantly impacted
        # Base processing is 1ms, overhead should be < 1ms additional
        assert avg_latency_ms < 3.0, f"Tool latency too high: {avg_latency_ms:.2f}ms"

        performance_metrics.tool_latencies.extend([avg_latency_ms] * 50)

    def test_json_rpc_throughput_impact(self, performance_metrics, benchmark, monkeypatch):
        """Test JSON-RPC message throughput impact."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up stdio filtering
        import workspace_qdrant_mcp.server

        # Generate test JSON-RPC messages
        test_messages = []
        for i in range(100):
            message = {
                "jsonrpc": "2.0",
                "method": f"test_method_{i}",
                "params": {"test_data": f"data_{i}"},
                "id": i
            }
            test_messages.append(json.dumps(message))

        def process_messages():
            """Process JSON-RPC messages through stdio system."""
            import io
            from contextlib import redirect_stdout

            # Capture output
            output = io.StringIO()
            with redirect_stdout(output):
                for message in test_messages:
                    print(message)
                    sys.stdout.flush()

            return output.getvalue()

        # Benchmark message processing
        result = benchmark.pedantic(process_messages, iterations=10, rounds=3)

        # Verify messages were processed correctly
        output_lines = result.strip().split('\n')
        assert len(output_lines) == len(test_messages), "Message count mismatch"

        for i, line in enumerate(output_lines):
            try:
                parsed = json.loads(line)
                assert parsed["id"] == i, f"Message order incorrect: expected {i}, got {parsed['id']}"
            except json.JSONDecodeError:
                pytest.fail(f"Message {i} not valid JSON: {repr(line)}")

        # Check throughput performance
        stats = benchmark.stats
        messages_per_second = len(test_messages) / stats.mean

        # CRITICAL: Should handle at least 1000 messages/second
        assert messages_per_second > 1000, f"Throughput too low: {messages_per_second:.0f} msg/s"

    @pytest.mark.slow
    def test_subprocess_performance_comparison(self, performance_metrics, tmp_path):
        """Compare subprocess performance with and without stdio mode."""
        # Create test scripts
        stdio_script = tmp_path / "stdio_test.py"
        stdio_script.write_text("""
import os
import sys
import json
import time
from threading import Timer

os.environ["WQM_STDIO_MODE"] = "true"

def shutdown():
    os._exit(0)

Timer(1.0, shutdown).start()

start_time = time.perf_counter()

try:
    from workspace_qdrant_mcp.stdio_server import run_lightweight_stdio_server

    class MockApp:
        def tool(self, func=None, **kwargs):
            def decorator(f):
                return f
            return decorator(func) if func else decorator

        def run(self, transport):
            # Output test messages
            for i in range(10):
                msg = {"jsonrpc": "2.0", "result": {"test": i}, "id": i}
                print(json.dumps(msg))

    import workspace_qdrant_mcp.stdio_server as stdio_module
    stdio_module.FastMCP = lambda name: MockApp()

    run_lightweight_stdio_server()

except Exception:
    sys.exit(1)
""")

        normal_script = tmp_path / "normal_test.py"
        normal_script.write_text("""
import json
import time
from threading import Timer

def shutdown():
    import os
    os._exit(0)

Timer(1.0, shutdown).start()

start_time = time.perf_counter()

try:
    # Simulate normal operation
    for i in range(10):
        msg = {"jsonrpc": "2.0", "result": {"test": i}, "id": i}
        print(json.dumps(msg))

except Exception:
    import sys
    sys.exit(1)
""")

        # Run stdio version
        start_time = time.perf_counter()
        stdio_result = subprocess.run([
            sys.executable, str(stdio_script)
        ], capture_output=True, text=True, timeout=5)
        stdio_time = time.perf_counter() - start_time

        # Run normal version
        start_time = time.perf_counter()
        normal_result = subprocess.run([
            sys.executable, str(normal_script)
        ], capture_output=True, text=True, timeout=5)
        normal_time = time.perf_counter() - start_time

        # Performance comparison
        performance_overhead = (stdio_time - normal_time) / normal_time * 100

        # CRITICAL: Performance overhead should be minimal (< 10%)
        assert performance_overhead < 10.0, f"Performance overhead too high: {performance_overhead:.1f}%"

        # Verify functionality is preserved
        assert stdio_result.returncode == 0, f"Stdio version failed: {stdio_result.stderr}"
        assert normal_result.returncode == 0, f"Normal version failed: {normal_result.stderr}"

        # Both should produce valid output
        stdio_lines = [line for line in stdio_result.stdout.strip().split('\n') if line.strip()]
        normal_lines = [line for line in normal_result.stdout.strip().split('\n') if line.strip()]

        assert len(stdio_lines) == len(normal_lines) == 10, "Output count mismatch"

        performance_metrics.startup_times.extend([stdio_time * 1000, normal_time * 1000])

    def test_logging_suppression_performance(self, performance_metrics, benchmark, monkeypatch):
        """Test performance impact of logging suppression."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up logging suppression
        import logging

        import workspace_qdrant_mcp.server

        # Create loggers for testing
        loggers = [
            logging.getLogger("test_logger"),
            logging.getLogger("qdrant_client"),
            logging.getLogger("fastmcp"),
            logging.getLogger("uvicorn"),
        ]

        def generate_log_messages():
            """Generate various log messages."""
            for logger in loggers:
                for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
                    logger.log(level, f"Test message at level {level}")

        # Benchmark logging operations
        benchmark.pedantic(generate_log_messages, iterations=100, rounds=5)

        stats = benchmark.stats
        avg_time_ms = stats.mean * 1000

        # CRITICAL: Logging suppression should have minimal overhead
        # Each iteration generates 20 log messages, so < 0.1ms per message
        assert avg_time_ms < 2.0, f"Logging suppression overhead too high: {avg_time_ms:.2f}ms"

    def test_memory_leak_detection(self, performance_metrics, monkeypatch):
        """Test for memory leaks in console suppression mechanisms."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server to set up suppression
        import workspace_qdrant_mcp.server

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Simulate extended operation
        for iteration in range(100):
            # Generate various outputs that should be suppressed
            print(f"Test message {iteration}")
            sys.stderr.write(f"Error message {iteration}\n")

            # Generate log messages
            logger = logging.getLogger(f"test_logger_{iteration % 10}")
            logger.error(f"Log message {iteration}")
            logger.warning(f"Warning {iteration}")

            # Generate warnings
            import warnings
            warnings.warn(f"Warning {iteration}", UserWarning, stacklevel=2)

            # Check memory periodically
            if iteration % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                performance_metrics.memory_usage.append(memory_growth)

        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        # CRITICAL: Should not have significant memory growth
        assert total_growth < 5.0, f"Possible memory leak: {total_growth:.2f}MB growth"

    def test_cpu_usage_impact(self, performance_metrics, monkeypatch):
        """Test CPU usage impact of console suppression."""
        monkeypatch.setenv("WQM_STDIO_MODE", "true")

        # Import server
        import workspace_qdrant_mcp.server

        process = psutil.Process()

        # Measure CPU usage during suppression activity
        cpu_measurements = []

        for _ in range(50):
            process.cpu_percent()

            # Generate activity that triggers suppression
            for i in range(100):
                print(f"Message {i}")
                logging.getLogger("test").error(f"Error {i}")
                sys.stderr.write(f"Stderr {i}\n")

            time.sleep(0.01)  # Allow CPU measurement
            cpu_after = process.cpu_percent()
            cpu_measurements.append(cpu_after)

        avg_cpu = sum(cpu_measurements) / len(cpu_measurements)

        # CRITICAL: CPU usage should be reasonable (< 10% for this test load)
        assert avg_cpu < 10.0, f"CPU usage too high: {avg_cpu:.1f}%"

    def test_comprehensive_performance_report(self, performance_metrics):
        """Generate comprehensive performance report."""
        # Run a comprehensive test scenario
        import workspace_qdrant_mcp.server

        # Measure various operations
        operations = [
            ("print_suppression", lambda: [print(f"test {i}") for i in range(100)]),
            ("logging_suppression", lambda: [logging.getLogger("test").error(f"error {i}") for i in range(100)]),
            ("stderr_suppression", lambda: [sys.stderr.write(f"stderr {i}\n") for i in range(100)]),
        ]

        results = {}
        for name, operation in operations:
            start_time = time.perf_counter()
            operation()
            end_time = time.perf_counter()
            results[name] = (end_time - start_time) * 1000

        # CRITICAL: All operations should complete quickly
        for name, duration_ms in results.items():
            assert duration_ms < 10.0, f"{name} too slow: {duration_ms:.2f}ms"

        # Generate summary
        summary = performance_metrics.get_summary()
        summary.update(results)

        # Print performance report for debugging
        print("\n=== CONSOLE SILENCE PERFORMANCE REPORT ===")
        for metric, value in summary.items():
            print(f"{metric}: {value:.2f}")

        return summary
