"""
Parallel Test Execution Engine with Dependency Management

This module provides advanced parallel test execution capabilities with intelligent
resource management, dependency handling, and optimization strategies.

Features:
- Intelligent parallel execution with resource conflict avoidance
- Test dependency resolution and execution ordering
- Dynamic resource allocation and management
- Test isolation and cleanup coordination
- Failure recovery and retry mechanisms
- Performance monitoring and optimization
- Load balancing across available resources
"""

import asyncio
import concurrent.futures
import json
import logging
import multiprocessing
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    Optional,
    Union,
)

import psutil

from .discovery import ResourceRequirement, TestCategory, TestComplexity, TestMetadata


class ExecutionStrategy(Enum):
    """Test execution strategies."""
    SEQUENTIAL = auto()         # Run tests sequentially
    PARALLEL_SIMPLE = auto()    # Basic parallel execution
    PARALLEL_SMART = auto()     # Intelligent parallel with resource management
    PARALLEL_ADAPTIVE = auto()  # Dynamic adaptation based on performance
    HYBRID = auto()            # Mixed sequential/parallel based on test types


class ExecutionStatus(Enum):
    """Test execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class ResourcePool(Enum):
    """Resource pool types for test execution."""
    CPU_INTENSIVE = auto()      # CPU-bound tests
    IO_INTENSIVE = auto()       # I/O-bound tests
    MEMORY_INTENSIVE = auto()   # Memory-intensive tests
    DATABASE = auto()          # Database-dependent tests
    NETWORK = auto()           # Network-dependent tests
    FILESYSTEM = auto()        # Filesystem-dependent tests


@dataclass
class ExecutionResult:
    """Result of a single test execution."""
    test_name: str
    status: ExecutionStatus
    duration: float
    start_time: float
    end_time: float
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error_message: str | None = None
    resource_usage: dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    worker_id: str | None = None


@dataclass
class ExecutionPlan:
    """Test execution plan with dependencies and resource allocation."""
    test_batches: list[list[str]] = field(default_factory=list)
    resource_allocation: dict[str, ResourcePool] = field(default_factory=dict)
    estimated_duration: float = 0.0
    max_parallelism: int = 1
    dependency_graph: dict[str, set[str]] = field(default_factory=dict)


class ResourceManager:
    """Manages system resources for test execution."""

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self._resource_locks = {
            ResourcePool.DATABASE: threading.Semaphore(1),  # Single DB access
            ResourcePool.NETWORK: threading.Semaphore(3),   # Limited network concurrent
            ResourcePool.FILESYSTEM: threading.Semaphore(5), # Limited FS access
            ResourcePool.CPU_INTENSIVE: threading.Semaphore(self.cpu_count),
            ResourcePool.IO_INTENSIVE: threading.Semaphore(self.cpu_count * 2),
            ResourcePool.MEMORY_INTENSIVE: threading.Semaphore(max(1, int(self.memory_gb / 2))),
        }
        self._allocated_resources = defaultdict(int)
        self._resource_lock = threading.RLock()

    def acquire_resources(self, test_name: str, requirements: set[ResourceRequirement]) -> list[ResourcePool]:
        """Acquire required resources for test execution."""
        pools_needed = self._map_requirements_to_pools(requirements)
        acquired_pools = []

        try:
            for pool in pools_needed:
                self._resource_locks[pool].acquire(timeout=30)
                acquired_pools.append(pool)

            with self._resource_lock:
                for pool in acquired_pools:
                    self._allocated_resources[pool] += 1

            return acquired_pools

        except Exception:
            # Release any acquired resources on failure
            for pool in reversed(acquired_pools):
                self._resource_locks[pool].release()
                with self._resource_lock:
                    self._allocated_resources[pool] -= 1
            raise

    def release_resources(self, test_name: str, pools: list[ResourcePool]):
        """Release resources after test completion."""
        with self._resource_lock:
            for pool in reversed(pools):
                if pool in self._resource_locks:
                    self._resource_locks[pool].release()
                    self._allocated_resources[pool] = max(0, self._allocated_resources[pool] - 1)

    def get_resource_usage(self) -> dict[ResourcePool, float]:
        """Get current resource usage percentages."""
        with self._resource_lock:
            usage = {}
            for pool, semaphore in self._resource_locks.items():
                initial_value = semaphore._initial_value
                current_value = semaphore._value
                allocated = initial_value - current_value
                usage[pool] = (allocated / initial_value) * 100 if initial_value > 0 else 0
            return usage

    def _map_requirements_to_pools(self, requirements: set[ResourceRequirement]) -> list[ResourcePool]:
        """Map resource requirements to resource pools."""
        pool_mapping = {
            ResourceRequirement.DATABASE: [ResourcePool.DATABASE, ResourcePool.IO_INTENSIVE],
            ResourceRequirement.NETWORK: [ResourcePool.NETWORK, ResourcePool.IO_INTENSIVE],
            ResourceRequirement.FILESYSTEM: [ResourcePool.FILESYSTEM, ResourcePool.IO_INTENSIVE],
            ResourceRequirement.HIGH_MEMORY: [ResourcePool.MEMORY_INTENSIVE],
            ResourceRequirement.EXTERNAL_SERVICE: [ResourcePool.NETWORK, ResourcePool.IO_INTENSIVE],
            ResourceRequirement.GPU: [ResourcePool.CPU_INTENSIVE],  # Treat GPU as CPU intensive
            ResourceRequirement.LONG_RUNNING: [ResourcePool.CPU_INTENSIVE],
        }

        pools = []
        for req in requirements:
            pools.extend(pool_mapping.get(req, [ResourcePool.CPU_INTENSIVE]))

        # Remove duplicates while preserving order
        seen = set()
        unique_pools = []
        for pool in pools:
            if pool not in seen:
                seen.add(pool)
                unique_pools.append(pool)

        return unique_pools if unique_pools else [ResourcePool.CPU_INTENSIVE]


class DependencyResolver:
    """Resolves test dependencies for execution ordering."""

    def __init__(self):
        self._dependency_graph: dict[str, set[str]] = {}
        self._reverse_graph: dict[str, set[str]] = {}

    def add_dependency(self, test: str, depends_on: str):
        """Add a dependency relationship."""
        if test not in self._dependency_graph:
            self._dependency_graph[test] = set()
        self._dependency_graph[test].add(depends_on)

        if depends_on not in self._reverse_graph:
            self._reverse_graph[depends_on] = set()
        self._reverse_graph[depends_on].add(test)

    def resolve_execution_order(self, tests: list[str]) -> list[list[str]]:
        """
        Resolve execution order using topological sorting.

        Returns batches of tests that can run in parallel.
        """
        # Create subgraph for only the tests we're running
        subgraph = {}
        in_degree = {}

        for test in tests:
            subgraph[test] = set()
            in_degree[test] = 0

        for test in tests:
            if test in self._dependency_graph:
                for dep in self._dependency_graph[test]:
                    if dep in tests:  # Only include dependencies within our test set
                        subgraph[dep].add(test)
                        in_degree[test] += 1

        # Topological sort to create execution batches
        batches = []
        remaining_tests = set(tests)

        while remaining_tests:
            # Find tests with no dependencies in current batch
            current_batch = []
            for test in list(remaining_tests):
                if in_degree[test] == 0:
                    current_batch.append(test)
                    remaining_tests.remove(test)

            if not current_batch:
                # Circular dependency detected - break it by picking arbitrary test
                current_batch = [list(remaining_tests)[0]]
                remaining_tests.remove(current_batch[0])

            batches.append(current_batch)

            # Update in-degrees for next batch
            for completed_test in current_batch:
                for dependent_test in subgraph.get(completed_test, []):
                    if dependent_test in remaining_tests:
                        in_degree[dependent_test] -= 1

        return batches

    def detect_cycles(self, tests: list[str]) -> list[list[str]]:
        """Detect circular dependencies."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(test: str, path: list[str]):
            if test in rec_stack:
                # Found cycle
                cycle_start = path.index(test)
                cycles.append(path[cycle_start:] + [test])
                return

            if test in visited:
                return

            visited.add(test)
            rec_stack.add(test)

            for dep in self._dependency_graph.get(test, []):
                if dep in tests:
                    dfs(dep, path + [test])

            rec_stack.remove(test)

        for test in tests:
            if test not in visited:
                dfs(test, [])

        return cycles


class TestWorker:
    """Individual test worker for parallel execution."""

    def __init__(self, worker_id: str, resource_manager: ResourceManager):
        self.worker_id = worker_id
        self.resource_manager = resource_manager
        self.current_test: str | None = None
        self.start_time: float | None = None

    async def execute_test(self,
                          test_name: str,
                          metadata: TestMetadata,
                          test_command: list[str],
                          timeout: float | None = None) -> ExecutionResult:
        """Execute a single test with resource management."""
        self.current_test = test_name
        self.start_time = time.time()

        # Acquire resources
        acquired_resources = []
        try:
            acquired_resources = self.resource_manager.acquire_resources(
                test_name, metadata.resources
            )

            # Execute test
            result = await self._run_test_subprocess(
                test_name, test_command, timeout or metadata.estimated_duration * 3
            )

            result.worker_id = self.worker_id
            return result

        finally:
            # Always release resources
            if acquired_resources:
                self.resource_manager.release_resources(test_name, acquired_resources)
            self.current_test = None
            self.start_time = None

    async def _run_test_subprocess(self,
                                  test_name: str,
                                  command: list[str],
                                  timeout: float) -> ExecutionResult:
        """Run test in subprocess with monitoring."""
        start_time = time.time()

        try:
            # Create subprocess with proper isolation
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ, PYTEST_CURRENT_TEST=test_name)
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                end_time = time.time()
                duration = end_time - start_time

                return ExecutionResult(
                    test_name=test_name,
                    status=ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.FAILED,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    return_code=process.returncode,
                    error_message=stderr.decode('utf-8', errors='replace') if process.returncode != 0 else None
                )

            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

                return ExecutionResult(
                    test_name=test_name,
                    status=ExecutionStatus.TIMEOUT,
                    duration=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    error_message=f"Test timed out after {timeout} seconds"
                )

        except Exception as e:
            return ExecutionResult(
                test_name=test_name,
                status=ExecutionStatus.FAILED,
                duration=time.time() - start_time,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )


class ParallelTestExecutor:
    """Advanced parallel test execution engine."""

    def __init__(self,
                 max_workers: int | None = None,
                 strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_SMART,
                 retry_failed: bool = True,
                 max_retries: int = 2):
        """
        Initialize parallel test executor.

        Args:
            max_workers: Maximum number of worker processes
            strategy: Execution strategy to use
            retry_failed: Whether to retry failed tests
            max_retries: Maximum number of retries for failed tests
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.strategy = strategy
        self.retry_failed = retry_failed
        self.max_retries = max_retries

        self.resource_manager = ResourceManager()
        self.dependency_resolver = DependencyResolver()

        self._results: dict[str, ExecutionResult] = {}
        self._running_tests: dict[str, TestWorker] = {}
        self._failed_tests: set[str] = set()
        self._execution_lock = asyncio.Lock()

        # Performance tracking
        self._execution_start_time: float | None = None
        self._batch_times: list[float] = []

    def create_execution_plan(self, tests: dict[str, TestMetadata]) -> ExecutionPlan:
        """Create optimized execution plan for tests."""
        # Build dependency graph
        for test_name, metadata in tests.items():
            for dep in metadata.dependencies:
                self.dependency_resolver.add_dependency(test_name, dep)

        # Check for circular dependencies
        test_names = list(tests.keys())
        cycles = self.dependency_resolver.detect_cycles(test_names)
        if cycles:
            logging.warning(f"Detected circular dependencies: {cycles}")

        # Resolve execution order
        execution_batches = self.dependency_resolver.resolve_execution_order(test_names)

        # Optimize batches based on strategy
        if self.strategy in [ExecutionStrategy.PARALLEL_SMART, ExecutionStrategy.PARALLEL_ADAPTIVE]:
            execution_batches = self._optimize_batches(execution_batches, tests)

        # Create resource allocation
        resource_allocation = {}
        for test_name, metadata in tests.items():
            pools = self.resource_manager._map_requirements_to_pools(metadata.resources)
            resource_allocation[test_name] = pools[0] if pools else ResourcePool.CPU_INTENSIVE

        # Estimate total duration
        estimated_duration = self._estimate_execution_duration(execution_batches, tests)

        return ExecutionPlan(
            test_batches=execution_batches,
            resource_allocation=resource_allocation,
            estimated_duration=estimated_duration,
            max_parallelism=min(self.max_workers, max(len(batch) for batch in execution_batches)),
            dependency_graph=dict(self.dependency_resolver._dependency_graph)
        )

    async def execute_tests(self,
                           tests: dict[str, TestMetadata],
                           pytest_args: list[str] = None,
                           progress_callback: Callable | None = None) -> dict[str, ExecutionResult]:
        """
        Execute tests according to the execution plan.

        Args:
            tests: Dictionary of test metadata
            pytest_args: Additional pytest arguments
            progress_callback: Callback for progress updates

        Returns:
            Dictionary of execution results
        """
        self._execution_start_time = time.time()
        pytest_args = pytest_args or []

        # Create execution plan
        plan = self.create_execution_plan(tests)

        # Execute batches
        total_tests = len(tests)
        completed_tests = 0

        for _batch_idx, batch in enumerate(plan.test_batches):
            batch_start_time = time.time()

            if self.strategy == ExecutionStrategy.SEQUENTIAL:
                # Sequential execution
                for test_name in batch:
                    result = await self._execute_single_test(
                        test_name, tests[test_name], pytest_args
                    )
                    self._results[test_name] = result
                    completed_tests += 1

                    if progress_callback:
                        progress_callback(completed_tests, total_tests, result)

            else:
                # Parallel execution within batch
                batch_results = await self._execute_batch_parallel(
                    batch, tests, pytest_args, progress_callback
                )

                self._results.update(batch_results)
                completed_tests += len(batch)

            batch_duration = time.time() - batch_start_time
            self._batch_times.append(batch_duration)

            # Adaptive strategy adjustment
            if self.strategy == ExecutionStrategy.PARALLEL_ADAPTIVE:
                self._adjust_strategy_based_on_performance()

        # Retry failed tests if enabled
        if self.retry_failed:
            await self._retry_failed_tests(tests, pytest_args, progress_callback)

        return self._results.copy()

    async def _execute_batch_parallel(self,
                                     batch: list[str],
                                     tests: dict[str, TestMetadata],
                                     pytest_args: list[str],
                                     progress_callback: Callable | None) -> dict[str, ExecutionResult]:
        """Execute a batch of tests in parallel."""
        # Create workers
        workers = [
            TestWorker(f"worker-{i}", self.resource_manager)
            for i in range(min(self.max_workers, len(batch)))
        ]

        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(len(workers))
        results = {}

        async def execute_with_worker(test_name: str):
            async with semaphore:
                # Find available worker
                worker = min(workers, key=lambda w: w.current_test is None)

                # Execute test
                result = await worker.execute_test(
                    test_name,
                    tests[test_name],
                    self._build_test_command(test_name, pytest_args),
                    timeout=tests[test_name].estimated_duration * 3
                )

                results[test_name] = result

                if progress_callback:
                    progress_callback(len(results), len(batch), result)

                return result

        # Execute all tests in batch concurrently
        await asyncio.gather(*[
            execute_with_worker(test_name) for test_name in batch
        ], return_exceptions=True)

        return results

    async def _execute_single_test(self,
                                  test_name: str,
                                  metadata: TestMetadata,
                                  pytest_args: list[str]) -> ExecutionResult:
        """Execute a single test."""
        worker = TestWorker("single", self.resource_manager)
        command = self._build_test_command(test_name, pytest_args)

        return await worker.execute_test(
            test_name, metadata, command, metadata.estimated_duration * 3
        )

    def _build_test_command(self, test_name: str, pytest_args: list[str]) -> list[str]:
        """Build pytest command for test execution."""
        base_command = ["python", "-m", "pytest"]
        base_command.extend(pytest_args)
        base_command.extend(["-v", "--tb=short"])

        # Add specific test
        if "::" in test_name:
            base_command.append(test_name)
        else:
            base_command.extend(["-k", test_name])

        return base_command

    def _optimize_batches(self,
                         batches: list[list[str]],
                         tests: dict[str, TestMetadata]) -> list[list[str]]:
        """Optimize test batches for better resource utilization."""
        optimized_batches = []

        for batch in batches:
            if len(batch) <= 1:
                optimized_batches.append(batch)
                continue

            # Group by resource requirements
            resource_groups = defaultdict(list)
            for test_name in batch:
                metadata = tests[test_name]
                key = frozenset(metadata.resources)
                resource_groups[key].append(test_name)

            # Create sub-batches to avoid resource conflicts
            sub_batches = []
            for resource_key, test_group in resource_groups.items():
                # Split large groups to balance load
                max_concurrent = self._get_max_concurrent_for_resources(resource_key)
                for i in range(0, len(test_group), max_concurrent):
                    sub_batch = test_group[i:i + max_concurrent]
                    sub_batches.append(sub_batch)

            optimized_batches.extend(sub_batches)

        return optimized_batches

    def _get_max_concurrent_for_resources(self, resources: frozenset) -> int:
        """Get maximum concurrent tests for resource set."""
        if ResourceRequirement.DATABASE in resources:
            return 1  # Database tests should be serialized
        elif ResourceRequirement.EXTERNAL_SERVICE in resources:
            return 2  # Limited external service tests
        elif ResourceRequirement.HIGH_MEMORY in resources:
            return max(1, int(self.resource_manager.memory_gb / 2))
        else:
            return self.max_workers

    def _estimate_execution_duration(self,
                                   batches: list[list[str]],
                                   tests: dict[str, TestMetadata]) -> float:
        """Estimate total execution duration."""
        total_duration = 0.0

        for batch in batches:
            if self.strategy == ExecutionStrategy.SEQUENTIAL:
                # Sequential: sum all durations
                batch_duration = sum(tests[test_name].estimated_duration for test_name in batch)
            else:
                # Parallel: maximum duration in batch
                batch_duration = max(tests[test_name].estimated_duration for test_name in batch) if batch else 0

            total_duration += batch_duration

        # Add overhead for setup/teardown
        return total_duration * 1.2

    def _adjust_strategy_based_on_performance(self):
        """Adjust strategy based on performance metrics."""
        if len(self._batch_times) < 2:
            return

        # Calculate efficiency metrics
        recent_times = self._batch_times[-3:]
        avg_time = sum(recent_times) / len(recent_times)

        # Get system resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # Adjust max_workers based on performance
        if cpu_percent > 90 and avg_time > 10:
            # High CPU usage and slow - reduce workers
            self.max_workers = max(1, int(self.max_workers * 0.8))
        elif cpu_percent < 60 and memory_percent < 70:
            # Low resource usage - potentially increase workers
            self.max_workers = min(multiprocessing.cpu_count(), int(self.max_workers * 1.2))

    async def _retry_failed_tests(self,
                                 tests: dict[str, TestMetadata],
                                 pytest_args: list[str],
                                 progress_callback: Callable | None):
        """Retry failed tests up to max_retries."""
        for retry_count in range(1, self.max_retries + 1):
            failed_tests = [
                name for name, result in self._results.items()
                if result.status == ExecutionStatus.FAILED and result.retry_count < retry_count
            ]

            if not failed_tests:
                break

            logging.info(f"Retrying {len(failed_tests)} failed tests (attempt {retry_count})")

            retry_results = {}
            for test_name in failed_tests:
                result = await self._execute_single_test(
                    test_name, tests[test_name], pytest_args + ["--lf"]  # Last failed
                )
                result.retry_count = retry_count
                retry_results[test_name] = result

                if progress_callback:
                    progress_callback(0, len(failed_tests), result)

            # Update results with retry results
            self._results.update(retry_results)

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get comprehensive execution statistics."""
        if not self._results:
            return {}

        total_duration = time.time() - self._execution_start_time if self._execution_start_time else 0

        # Status breakdown
        status_counts = defaultdict(int)
        for result in self._results.values():
            status_counts[result.status] += 1

        # Performance metrics
        successful_results = [r for r in self._results.values() if r.status == ExecutionStatus.COMPLETED]
        durations = [r.duration for r in successful_results]

        return {
            'total_tests': len(self._results),
            'total_duration': total_duration,
            'status_breakdown': dict(status_counts),
            'avg_test_duration': sum(durations) / len(durations) if durations else 0,
            'min_test_duration': min(durations) if durations else 0,
            'max_test_duration': max(durations) if durations else 0,
            'batch_count': len(self._batch_times),
            'avg_batch_duration': sum(self._batch_times) / len(self._batch_times) if self._batch_times else 0,
            'resource_usage': self.resource_manager.get_resource_usage(),
            'retry_stats': {
                'retried_tests': len([r for r in self._results.values() if r.retry_count > 0]),
                'max_retries_used': max((r.retry_count for r in self._results.values()), default=0)
            }
        }

    def cancel_execution(self):
        """Cancel ongoing test execution."""
        # This would be implemented with proper signal handling
        # and worker process termination in a production system
        pass
