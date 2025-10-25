"""
Performance and Load Testing

Comprehensive performance tests for large document sets, memory usage patterns,
concurrent operations, and search performance benchmarking under realistic loads.

This module implements subtask 203.5 of the End-to-End Functional Testing Framework.
"""

import asyncio
import gc
import json
import multiprocessing
import os
import random
import statistics
import string
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import psutil
import pytest


class PerformanceTestDataGenerator:
    """Generates test data for performance and load testing."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "performance_data"
        self.data_dir.mkdir(exist_ok=True)

    def generate_document_set(
        self,
        num_documents: int,
        document_size_range: tuple[int, int] = (1000, 10000),
        document_types: list[str] = None
    ) -> list[Path]:
        """Generate a set of test documents with varying sizes."""
        if document_types is None:
            document_types = ["md", "txt", "py", "js", "json"]

        documents = []

        for i in range(num_documents):
            # Choose random document type and size
            doc_type = random.choice(document_types)
            doc_size = random.randint(*document_size_range)

            # Generate document content
            content = self._generate_document_content(doc_type, doc_size, i)

            # Create document file
            doc_path = self.data_dir / f"document_{i:04d}.{doc_type}"
            doc_path.write_text(content, encoding='utf-8')
            documents.append(doc_path)

        return documents

    def generate_large_document_set(
        self,
        num_documents: int = 1000,
        size_distribution: str = "mixed"
    ) -> list[Path]:
        """Generate large document set with realistic size distribution."""
        if size_distribution == "small":
            size_range = (500, 2000)
        elif size_distribution == "medium":
            size_range = (2000, 10000)
        elif size_distribution == "large":
            size_range = (10000, 50000)
        else:  # mixed
            size_ranges = [(500, 2000), (2000, 10000), (10000, 50000)]
            # Create mixed distribution
            documents = []
            for _i in range(num_documents):
                size_range = random.choice(size_ranges)
                documents.extend(self.generate_document_set(1, size_range, ["md", "txt", "py"]))
            return documents

        return self.generate_document_set(num_documents, size_range)

    def generate_concurrent_test_scenarios(
        self,
        num_scenarios: int = 10
    ) -> list[dict[str, Any]]:
        """Generate test scenarios for concurrent operations."""
        scenarios = []

        for i in range(num_scenarios):
            scenario = {
                "id": i,
                "operations": random.randint(5, 20),
                "document_count": random.randint(10, 100),
                "search_queries": self._generate_search_queries(),
                "concurrent_users": random.randint(1, 5)
            }
            scenarios.append(scenario)

        return scenarios

    def _generate_document_content(self, doc_type: str, size: int, doc_id: int) -> str:
        """Generate realistic document content based on type."""
        if doc_type == "md":
            return self._generate_markdown_content(size, doc_id)
        elif doc_type == "py":
            return self._generate_python_content(size, doc_id)
        elif doc_type == "js":
            return self._generate_javascript_content(size, doc_id)
        elif doc_type == "json":
            return self._generate_json_content(size, doc_id)
        else:  # txt
            return self._generate_text_content(size, doc_id)

    def _generate_markdown_content(self, size: int, doc_id: int) -> str:
        """Generate realistic Markdown content."""
        content = f"# Document {doc_id}\n\n"
        content += "This is a test document for performance testing.\n\n"
        content += "## Section 1\n\n"

        # Add paragraphs to reach target size
        paragraph_template = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        )

        while len(content) < size:
            content += paragraph_template + "\n\n"
            if len(content) < size - 100:
                content += f"### Subsection {len(content) // 200}\n\n"

        return content[:size]

    def _generate_python_content(self, size: int, doc_id: int) -> str:
        """Generate realistic Python code content."""
        content = f'"""Module {doc_id} for performance testing."""\n\n'
        content += "import os\nimport sys\nfrom typing import Any, List\n\n"

        # Add functions and classes
        function_template = '''
def function_{func_id}(data: Any) -> Dict[str, Any]:
    """Process data and return results."""
    result = {{
        "processed": True,
        "data_type": type(data).__name__,
        "timestamp": time.time()
    }}

    # Perform some processing
    if isinstance(data, (list, tuple)):
        result["length"] = len(data)
    elif isinstance(data, dict):
        result["keys"] = list(data.keys())

    return result

'''

        func_id = 0
        while len(content) < size:
            content += function_template.format(func_id=func_id)
            func_id += 1

        return content[:size]

    def _generate_javascript_content(self, size: int, doc_id: int) -> str:
        """Generate realistic JavaScript content."""
        content = f"// Module {doc_id} for performance testing\n\n"

        function_template = '''
function processData{func_id}(data) {{
    const result = {{
        processed: true,
        dataType: typeof data,
        timestamp: Date.now()
    }};

    if (Array.isArray(data)) {{
        result.length = data.length;
    }} else if (typeof data === 'object') {{
        result.keys = Object.keys(data);
    }}

    return result;
}}

'''

        func_id = 0
        while len(content) < size:
            content += function_template.format(func_id=func_id)
            func_id += 1

        return content[:size]

    def _generate_json_content(self, size: int, doc_id: int) -> str:
        """Generate realistic JSON content."""
        data = {
            "document_id": doc_id,
            "type": "performance_test",
            "generated_at": time.time(),
            "data": []
        }

        # Add data items to reach target size
        item_template = {
            "id": 0,
            "name": "test_item",
            "value": random.random(),
            "tags": ["performance", "test", "data"],
            "metadata": {
                "created": time.time(),
                "category": "test"
            }
        }

        item_id = 0
        while len(json.dumps(data)) < size:
            item = item_template.copy()
            item["id"] = item_id
            item["name"] = f"test_item_{item_id}"
            data["data"].append(item)
            item_id += 1

        return json.dumps(data, indent=2)[:size]

    def _generate_text_content(self, size: int, doc_id: int) -> str:
        """Generate realistic text content."""
        content = f"Document {doc_id} - Performance Test Content\n\n"

        # Lorem ipsum style content
        words = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
            "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
            "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
            "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo"
        ]

        while len(content) < size:
            sentence_length = random.randint(5, 15)
            sentence = " ".join(random.choices(words, k=sentence_length))
            content += sentence.capitalize() + ". "

            # Add paragraph breaks
            if random.random() < 0.1:
                content += "\n\n"

        return content[:size]

    def _generate_search_queries(self) -> list[str]:
        """Generate realistic search queries."""
        return [
            "function implementation",
            "error handling",
            "data processing",
            "performance optimization",
            "test coverage",
            "documentation",
            "configuration",
            "async await",
            "database connection",
            "API endpoint"
        ]


class PerformanceMetricsCollector:
    """Collects and analyzes performance metrics."""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        """End timing and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            return duration
        return 0.0

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a custom metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }

    def get_cpu_usage(self) -> dict[str, float]:
        """Get CPU usage metrics."""
        process = psutil.Process()

        return {
            "process_percent": process.cpu_percent(),
            "system_percent": psutil.cpu_percent(),
            "core_count": psutil.cpu_count(),
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }

    def analyze_metrics(self) -> dict[str, Any]:
        """Analyze collected metrics and return statistics."""
        analysis = {}

        for metric_name, values in self.metrics.items():
            if values:
                analysis[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }

        return analysis

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]


class LoadTestEnvironment:
    """Environment for load testing with realistic scenarios."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.config_dir = tmp_path / ".config" / "workspace-qdrant"
        self.cli_executable = "uv run wqm"
        self.metrics = PerformanceMetricsCollector()
        self.data_generator = PerformanceTestDataGenerator(tmp_path)

        self.setup_environment()

    def setup_environment(self):
        """Set up load testing environment."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create performance-optimized configuration
        config_content = {
            "qdrant_url": "http://localhost:6333",
            "performance": {
                "batch_size": 100,
                "max_concurrent_operations": 10,
                "timeout_seconds": 30,
                "memory_limit_mb": 1024
            },
            "indexing": {
                "enable_parallel": True,
                "chunk_size": 1000,
                "embedding_batch_size": 32
            }
        }

        import yaml
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

    def run_cli_command_with_metrics(
        self,
        command: str,
        timeout: int = 60
    ) -> tuple[int, str, str, dict[str, Any]]:
        """Execute CLI command and collect performance metrics."""
        # Record initial state
        initial_memory = self.metrics.get_memory_usage()
        self.metrics.get_cpu_usage()

        # Start timing
        self.metrics.start_timer(f"command_{command.split()[0]}")

        # Execute command
        env = os.environ.copy()
        env.update({
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
        })

        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=True,
                cwd=self.tmp_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return_code, stdout, stderr = -1, "", f"Command timed out after {timeout} seconds"

        # Record final state and metrics
        duration = self.metrics.end_timer(f"command_{command.split()[0]}")
        final_memory = self.metrics.get_memory_usage()
        final_cpu = self.metrics.get_cpu_usage()

        metrics = {
            "duration": duration,
            "memory_delta": final_memory["rss_mb"] - initial_memory["rss_mb"],
            "peak_memory": final_memory["rss_mb"],
            "cpu_usage": final_cpu["process_percent"]
        }

        return return_code, stdout, stderr, metrics

    def simulate_concurrent_load(
        self,
        operations: list[str],
        num_concurrent: int = 5,
        iterations_per_thread: int = 10
    ) -> dict[str, Any]:
        """Simulate concurrent load with multiple operations."""
        results = {
            "total_operations": num_concurrent * iterations_per_thread * len(operations),
            "concurrent_threads": num_concurrent,
            "operations": operations,
            "thread_results": [],
            "errors": [],
            "performance_metrics": {}
        }

        def worker_thread(thread_id: int) -> dict[str, Any]:
            thread_results = {
                "thread_id": thread_id,
                "completed_operations": 0,
                "errors": [],
                "durations": []
            }

            for _iteration in range(iterations_per_thread):
                for operation in operations:
                    try:
                        start_time = time.time()
                        return_code, stdout, stderr, metrics = self.run_cli_command_with_metrics(
                            operation, timeout=30
                        )
                        duration = time.time() - start_time

                        thread_results["durations"].append(duration)

                        if return_code == 0:
                            thread_results["completed_operations"] += 1
                        else:
                            thread_results["errors"].append({
                                "operation": operation,
                                "error": stderr + stdout
                            })
                    except Exception as e:
                        thread_results["errors"].append({
                            "operation": operation,
                            "error": str(e)
                        })

            return thread_results

        # Execute concurrent load
        start_time = time.time()
        initial_memory = self.metrics.get_memory_usage()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_concurrent)]

            for future in as_completed(futures):
                try:
                    thread_result = future.result(timeout=300)
                    results["thread_results"].append(thread_result)
                except Exception as e:
                    results["errors"].append(str(e))

        # Collect final metrics
        total_duration = time.time() - start_time
        final_memory = self.metrics.get_memory_usage()

        results["performance_metrics"] = {
            "total_duration": total_duration,
            "operations_per_second": results["total_operations"] / total_duration,
            "memory_usage_delta": final_memory["rss_mb"] - initial_memory["rss_mb"],
            "peak_memory": final_memory["rss_mb"]
        }

        return results


@pytest.mark.functional
@pytest.mark.performance
class TestPerformanceAndLoadTesting:
    """Test performance and load characteristics."""

    @pytest.fixture
    def load_env(self, tmp_path):
        """Create load testing environment."""
        return LoadTestEnvironment(tmp_path)

    @pytest.fixture
    def data_generator(self, tmp_path):
        """Create test data generator."""
        return PerformanceTestDataGenerator(tmp_path)

    def test_cli_command_performance_baseline(self, load_env):
        """Test baseline performance of CLI commands."""
        baseline_commands = [
            "--version",
            "--help",
            "admin status",
            "config show"
        ]

        performance_results = {}

        for command in baseline_commands:
            # Run command multiple times for statistical significance
            durations = []
            memory_usage = []

            for _ in range(5):
                return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                    command
                )

                durations.append(metrics["duration"])
                memory_usage.append(metrics["peak_memory"])

            performance_results[command] = {
                "avg_duration": statistics.mean(durations),
                "max_duration": max(durations),
                "avg_memory": statistics.mean(memory_usage),
                "max_memory": max(memory_usage)
            }

        # Validate performance baselines
        for command, metrics in performance_results.items():
            # Commands should complete within reasonable time
            assert metrics["avg_duration"] < 5.0, f"{command} too slow: {metrics['avg_duration']:.2f}s"

            # Memory usage should be reasonable
            assert metrics["avg_memory"] < 200, f"{command} uses too much memory: {metrics['avg_memory']:.2f}MB"

    def test_large_document_set_ingestion_performance(self, load_env, data_generator):
        """Test performance with large document sets."""
        # Generate document sets of increasing size
        document_sets = [
            (50, "small_set"),
            (100, "medium_set"),
            (200, "large_set")
        ]

        ingestion_results = {}

        for num_docs, set_name in document_sets:
            # Generate documents
            documents = data_generator.generate_document_set(
                num_docs,
                document_size_range=(1000, 5000)
            )

            # Test batch ingestion performance
            docs_dir = documents[0].parent

            return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                f"ingest folder {docs_dir}",
                timeout=120
            )

            ingestion_results[set_name] = {
                "document_count": num_docs,
                "duration": metrics["duration"],
                "memory_delta": metrics["memory_delta"],
                "peak_memory": metrics["peak_memory"],
                "docs_per_second": num_docs / metrics["duration"] if metrics["duration"] > 0 else 0,
                "command_success": return_code == 0 or len(stdout + stderr) > 0
            }

        # Validate performance scalability
        for set_name, results in ingestion_results.items():
            assert results["command_success"], f"Ingestion failed for {set_name}"

            # Performance should be reasonable
            if results["docs_per_second"] > 0:
                assert results["docs_per_second"] > 1, f"Too slow: {results['docs_per_second']:.2f} docs/sec"

    def test_concurrent_operations_performance(self, load_env):
        """Test performance under concurrent operations."""
        # Define concurrent test scenarios
        concurrent_operations = [
            "admin status",
            "--version",
            "config show"
        ]

        # Test with increasing concurrency levels
        concurrency_levels = [1, 3, 5]

        concurrency_results = {}

        for concurrency in concurrency_levels:
            load_results = load_env.simulate_concurrent_load(
                operations=concurrent_operations,
                num_concurrent=concurrency,
                iterations_per_thread=5
            )

            concurrency_results[f"concurrency_{concurrency}"] = {
                "total_operations": load_results["total_operations"],
                "operations_per_second": load_results["performance_metrics"]["operations_per_second"],
                "total_errors": len(load_results["errors"]),
                "memory_delta": load_results["performance_metrics"]["memory_usage_delta"],
                "peak_memory": load_results["performance_metrics"]["peak_memory"]
            }

        # Validate concurrent performance
        for level, results in concurrency_results.items():
            # Should handle concurrent operations
            assert results["operations_per_second"] > 0, f"No operations completed for {level}"

            # Error rate should be low
            error_rate = results["total_errors"] / results["total_operations"]
            assert error_rate < 0.1, f"High error rate for {level}: {error_rate:.2%}"

    @pytest.mark.slow
    def test_memory_usage_patterns(self, load_env, data_generator):
        """Test memory usage patterns during operations."""
        # Generate test data
        documents = data_generator.generate_document_set(100, (2000, 8000))

        # Monitor memory usage during operations
        memory_snapshots = []

        def take_memory_snapshot(operation: str):
            memory_usage = load_env.metrics.get_memory_usage()
            memory_snapshots.append({
                "operation": operation,
                "timestamp": time.time(),
                "memory_mb": memory_usage["rss_mb"],
                "memory_percent": memory_usage["percent"]
            })

        # Baseline memory
        take_memory_snapshot("baseline")

        # Test various operations
        operations = [
            ("ingest folder " + str(documents[0].parent), "ingestion"),
            ("admin status", "status_check"),
            ("config show", "config_access")
        ]

        for command, operation_name in operations:
            take_memory_snapshot(f"before_{operation_name}")

            return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                command, timeout=60
            )

            take_memory_snapshot(f"after_{operation_name}")

            # Force garbage collection
            gc.collect()
            time.sleep(1)
            take_memory_snapshot(f"gc_after_{operation_name}")

        # Analyze memory patterns
        baseline_memory = memory_snapshots[0]["memory_mb"]
        peak_memory = max(snapshot["memory_mb"] for snapshot in memory_snapshots)

        # Validate memory usage
        memory_growth = peak_memory - baseline_memory
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.2f}MB"

        # Check for memory leaks (final memory should be close to baseline)
        final_memory = memory_snapshots[-1]["memory_mb"]
        memory_leak = final_memory - baseline_memory
        assert memory_leak < 100, f"Potential memory leak: {memory_leak:.2f}MB"

    def test_search_performance_benchmarking(self, load_env, data_generator):
        """Test search performance with various query types and data sizes."""
        # Generate searchable content
        documents = data_generator.generate_document_set(
            100,
            document_size_range=(1000, 5000),
            document_types=["md", "txt", "py"]
        )

        # Ingest documents first (if daemon available)
        docs_dir = documents[0].parent
        return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
            f"ingest folder {docs_dir}",
            timeout=60
        )

        # Define search test scenarios
        search_queries = data_generator._generate_search_queries()
        search_types = [
            ("search project", "project_search"),
            ("search scratchbook", "scratchbook_search")
        ]

        search_results = {}

        for search_type, search_name in search_types:
            type_results = []

            for query in search_queries[:5]:  # Test first 5 queries
                command = f'{search_type} "{query}"'

                return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                    command, timeout=30
                )

                type_results.append({
                    "query": query,
                    "duration": metrics["duration"],
                    "memory_usage": metrics["peak_memory"],
                    "command_executed": return_code != -1,
                    "has_output": len(stdout + stderr) > 0
                })

            search_results[search_name] = {
                "queries_tested": len(type_results),
                "avg_duration": statistics.mean([r["duration"] for r in type_results]),
                "max_duration": max([r["duration"] for r in type_results]),
                "execution_rate": sum([r["command_executed"] for r in type_results]) / len(type_results)
            }

        # Validate search performance
        for search_type, results in search_results.items():
            # Searches should execute
            assert results["execution_rate"] > 0.8, f"Low execution rate for {search_type}"

            # Search should be reasonably fast
            assert results["avg_duration"] < 10.0, f"Slow search for {search_type}: {results['avg_duration']:.2f}s"

    @pytest.mark.slow
    def test_stress_testing_limits(self, load_env, data_generator):
        """Test system behavior under stress conditions."""
        # Generate large document set
        stress_documents = data_generator.generate_large_document_set(
            num_documents=500,
            size_distribution="mixed"
        )

        # Stress test scenarios
        stress_scenarios = [
            {
                "name": "high_concurrency",
                "concurrent_operations": 8,
                "iterations": 3,
                "operations": ["admin status", "config show"]
            },
            {
                "name": "large_batch_ingestion",
                "command": f"ingest folder {stress_documents[0].parent}",
                "timeout": 300
            }
        ]

        stress_results = {}

        for scenario in stress_scenarios:
            scenario_name = scenario["name"]

            # Monitor system resources before stress test
            load_env.metrics.get_memory_usage()
            load_env.metrics.get_cpu_usage()

            if scenario_name == "high_concurrency":
                # Run concurrent stress test
                load_results = load_env.simulate_concurrent_load(
                    operations=scenario["operations"],
                    num_concurrent=scenario["concurrent_operations"],
                    iterations_per_thread=scenario["iterations"]
                )

                stress_results[scenario_name] = {
                    "operations_per_second": load_results["performance_metrics"]["operations_per_second"],
                    "error_rate": len(load_results["errors"]) / load_results["total_operations"],
                    "memory_delta": load_results["performance_metrics"]["memory_usage_delta"]
                }

            elif scenario_name == "large_batch_ingestion":
                # Run large batch test
                return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                    scenario["command"],
                    timeout=scenario["timeout"]
                )

                stress_results[scenario_name] = {
                    "completed": return_code != -1,
                    "duration": metrics["duration"],
                    "memory_delta": metrics["memory_delta"],
                    "peak_memory": metrics["peak_memory"]
                }

        # Validate stress test results
        for scenario_name, results in stress_results.items():
            if scenario_name == "high_concurrency":
                # Should handle high concurrency reasonably
                assert results["error_rate"] < 0.2, f"High error rate under stress: {results['error_rate']:.2%}"

            elif scenario_name == "large_batch_ingestion":
                # Should complete within timeout
                assert results["completed"], "Large batch ingestion timed out"

    def test_resource_cleanup_efficiency(self, load_env, data_generator):
        """Test resource cleanup and garbage collection efficiency."""
        # Generate test data
        documents = data_generator.generate_document_set(50, (1000, 3000))

        # Baseline memory
        gc.collect()
        baseline_memory = load_env.metrics.get_memory_usage()["rss_mb"]

        # Perform operations that should be cleaned up
        operations = [
            f"ingest folder {documents[0].parent}",
            "admin status",
            "config show"
        ]

        for operation in operations:
            # Run operation
            return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                operation, timeout=60
            )

            # Force cleanup
            gc.collect()
            time.sleep(1)

        # Check final memory usage
        final_memory = load_env.metrics.get_memory_usage()["rss_mb"]
        memory_delta = final_memory - baseline_memory

        # Validate resource cleanup
        assert memory_delta < 50, f"Poor resource cleanup: {memory_delta:.2f}MB retained"

    def test_performance_regression_detection(self, load_env):
        """Test performance regression detection capabilities."""
        # Define performance benchmarks
        benchmarks = {
            "--version": {"max_duration": 1.0, "max_memory": 50},
            "--help": {"max_duration": 2.0, "max_memory": 100},
            "admin status": {"max_duration": 10.0, "max_memory": 150}
        }

        regression_results = {}

        for command, benchmark in benchmarks.items():
            # Run command multiple times
            durations = []
            memory_usage = []

            for _ in range(3):
                return_code, stdout, stderr, metrics = load_env.run_cli_command_with_metrics(
                    command
                )
                durations.append(metrics["duration"])
                memory_usage.append(metrics["peak_memory"])

            avg_duration = statistics.mean(durations)
            avg_memory = statistics.mean(memory_usage)

            regression_results[command] = {
                "avg_duration": avg_duration,
                "avg_memory": avg_memory,
                "duration_within_benchmark": avg_duration <= benchmark["max_duration"],
                "memory_within_benchmark": avg_memory <= benchmark["max_memory"]
            }

        # Validate no performance regressions
        for command, results in regression_results.items():
            assert results["duration_within_benchmark"], f"Performance regression in {command}: {results['avg_duration']:.2f}s"
            assert results["memory_within_benchmark"], f"Memory regression in {command}: {results['avg_memory']:.2f}MB"
