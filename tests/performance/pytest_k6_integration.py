"""
Integration between pytest and k6 performance tests.

This module provides pytest fixtures and utilities to run k6 performance tests
as part of the Python test suite, enabling unified testing workflows.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pytest
import psutil
import requests
from contextlib import contextmanager

# Performance test configuration
K6_SCRIPT_DIR = Path(__file__).parent / "k6"
K6_MAIN_SCRIPT = K6_SCRIPT_DIR / "mcp_performance_tests.js"
K6_QUICK_SCRIPT = K6_SCRIPT_DIR / "quick_performance_test.js"


class K6TestRunner:
    """Wrapper for running k6 tests from pytest."""

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        self.server_url = server_url
        self.k6_available = self._check_k6_available()

    def _check_k6_available(self) -> bool:
        """Check if k6 is installed and available."""
        try:
            result = subprocess.run(
                ["k6", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def run_quick_test(self) -> Dict[str, Any]:
        """Run quick performance test suitable for CI."""
        if not self.k6_available:
            pytest.skip("k6 not available")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            env = {"MCP_SERVER_URL": self.server_url}
            result = subprocess.run(
                [
                    "k6", "run",
                    "--out", f"json={output_file}",
                    "--quiet",
                    str(K6_QUICK_SCRIPT)
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            # Parse results
            results = self._parse_k6_results(output_file)
            results["k6_exit_code"] = result.returncode
            results["k6_stdout"] = result.stdout
            results["k6_stderr"] = result.stderr

            return results

        except subprocess.TimeoutExpired:
            return {"error": "k6 test timed out", "success": False}
        finally:
            # Cleanup
            Path(output_file).unlink(missing_ok=True)

    def run_load_test(self, duration: str = "30s", vus: int = 10) -> Dict[str, Any]:
        """Run load test with specified parameters."""
        if not self.k6_available:
            pytest.skip("k6 not available")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            env = {"MCP_SERVER_URL": self.server_url}
            result = subprocess.run(
                [
                    "k6", "run",
                    "--out", f"json={output_file}",
                    "--vus", str(vus),
                    "--duration", duration,
                    "--scenario", "load_test",
                    str(K6_MAIN_SCRIPT)
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            results = self._parse_k6_results(output_file)
            results["k6_exit_code"] = result.returncode
            results["test_config"] = {"duration": duration, "vus": vus}

            return results

        except subprocess.TimeoutExpired:
            return {"error": "k6 load test timed out", "success": False}
        finally:
            Path(output_file).unlink(missing_ok=True)

    def _parse_k6_results(self, output_file: str) -> Dict[str, Any]:
        """Parse k6 JSON output file."""
        try:
            with open(output_file, 'r') as f:
                # Read all lines and parse the last summary
                lines = f.readlines()
                for line in reversed(lines):
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'Point' and 'metrics' in data:
                            continue
                        elif 'metrics' in data:
                            return {
                                "success": True,
                                "metrics": data.get('metrics', {}),
                                "raw_data": data
                            }
                    except json.JSONDecodeError:
                        continue

            return {"success": False, "error": "No valid metrics found in k6 output"}

        except Exception as e:
            return {"success": False, "error": f"Failed to parse k6 results: {e}"}


@contextmanager
def mcp_server_for_testing(port: int = 8000):
    """Context manager to start/stop MCP server for testing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

    server_process = None
    try:
        # Start MCP server
        server_process = subprocess.Popen(
            [
                sys.executable, "-m", "workspace_qdrant_mcp.web.server",
                "--host", "127.0.0.1",
                "--port", str(port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        server_url = f"http://127.0.0.1:{port}"
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get(f"{server_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("MCP server failed to start within 30 seconds")

        yield server_url

    finally:
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


# Pytest fixtures
@pytest.fixture
def k6_runner():
    """Provide a K6TestRunner instance."""
    return K6TestRunner()


@pytest.fixture
def mcp_server():
    """Start MCP server for performance testing."""
    with mcp_server_for_testing() as server_url:
        yield server_url


@pytest.fixture
def performance_test_config():
    """Load performance test configuration."""
    config_file = K6_SCRIPT_DIR / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {
        "target_response_time_ms": 200,
        "max_acceptable_response_time_ms": 500,
        "max_error_rate": 0.01
    }


# Performance test markers
pytestmark = pytest.mark.performance


class TestK6Performance:
    """Performance tests using k6."""

    @pytest.mark.slow
    def test_mcp_tools_response_time_quick(self, k6_runner, mcp_server, performance_test_config):
        """Quick performance test for CI - verify MCP tools meet response time targets."""
        k6_runner.server_url = mcp_server
        results = k6_runner.run_quick_test()

        assert results["success"], f"k6 test failed: {results.get('error', 'Unknown error')}"

        # Check if we have metrics
        metrics = results.get("metrics", {})
        assert metrics, "No performance metrics available"

        # Verify response time targets
        http_req_duration = metrics.get("http_req_duration", {}).get("values", {})
        p95_duration = http_req_duration.get("p95", float('inf'))

        target_ms = performance_test_config["target_response_time_ms"]
        assert p95_duration < target_ms, (
            f"Response time P95 ({p95_duration:.2f}ms) exceeds target ({target_ms}ms)"
        )

        # Verify error rate
        http_req_failed = metrics.get("http_req_failed", {}).get("values", {})
        error_rate = http_req_failed.get("rate", 1.0)
        max_error_rate = performance_test_config["max_error_rate"]

        assert error_rate <= max_error_rate, (
            f"Error rate ({error_rate:.3f}) exceeds maximum ({max_error_rate})"
        )

        # Log performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   Response Time P95: {p95_duration:.2f}ms")
        print(f"   Error Rate: {error_rate:.3f}")
        print(f"   Total Requests: {metrics.get('http_reqs', {}).get('values', {}).get('count', 'N/A')}")

    @pytest.mark.slow
    @pytest.mark.skipif(not K6TestRunner()._check_k6_available(), reason="k6 not available")
    def test_mcp_tools_load_test(self, k6_runner, mcp_server, performance_test_config):
        """Load test - verify MCP tools handle sustained load."""
        k6_runner.server_url = mcp_server
        results = k6_runner.run_load_test(duration="30s", vus=10)

        assert results["success"], f"Load test failed: {results.get('error', 'Unknown error')}"

        # Verify performance under load
        metrics = results.get("metrics", {})
        http_req_duration = metrics.get("http_req_duration", {}).get("values", {})
        p95_duration = http_req_duration.get("p95", float('inf'))
        p99_duration = http_req_duration.get("p99", float('inf'))

        target_ms = performance_test_config["target_response_time_ms"]
        max_ms = performance_test_config["max_acceptable_response_time_ms"]

        # P95 should meet target, P99 should be acceptable
        assert p95_duration < target_ms, (
            f"Load test P95 ({p95_duration:.2f}ms) exceeds target ({target_ms}ms)"
        )
        assert p99_duration < max_ms, (
            f"Load test P99 ({p99_duration:.2f}ms) exceeds maximum ({max_ms}ms)"
        )

        print(f"\n📊 Load Test Results:")
        print(f"   Response Time P95: {p95_duration:.2f}ms")
        print(f"   Response Time P99: {p99_duration:.2f}ms")
        print(f"   Throughput: {metrics.get('http_reqs', {}).get('values', {}).get('rate', 'N/A')} req/s")

    def test_performance_test_infrastructure(self, k6_runner):
        """Verify k6 performance test infrastructure is properly set up."""
        # Check k6 availability
        assert k6_runner.k6_available, "k6 is not installed or not available in PATH"

        # Check test scripts exist
        assert K6_MAIN_SCRIPT.exists(), f"Main k6 script not found: {K6_MAIN_SCRIPT}"
        assert K6_QUICK_SCRIPT.exists(), f"Quick k6 script not found: {K6_QUICK_SCRIPT}"

        # Verify script syntax by checking if file exists and has valid JS
        assert K6_QUICK_SCRIPT.read_text().startswith('/**'), "Quick script should start with comment block"
        assert 'export default function' in K6_QUICK_SCRIPT.read_text(), "Script should have default export function"


# Helper function for manual testing
def run_performance_benchmark(server_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    """Run performance benchmark manually - useful for development."""
    runner = K6TestRunner(server_url)
    if not runner.k6_available:
        return {"error": "k6 not available"}

    print("🚀 Running performance benchmark...")
    results = runner.run_quick_test()

    if results["success"]:
        metrics = results["metrics"]
        http_req_duration = metrics.get("http_req_duration", {}).get("values", {})
        print(f"✅ Performance benchmark completed:")
        print(f"   P95 Response Time: {http_req_duration.get('p95', 'N/A'):.2f}ms")
        print(f"   P99 Response Time: {http_req_duration.get('p99', 'N/A'):.2f}ms")
        print(f"   Total Requests: {metrics.get('http_reqs', {}).get('values', {}).get('count', 'N/A')}")
    else:
        print(f"❌ Performance benchmark failed: {results.get('error', 'Unknown error')}")

    return results


if __name__ == "__main__":
    # Run benchmark if called directly
    run_performance_benchmark()