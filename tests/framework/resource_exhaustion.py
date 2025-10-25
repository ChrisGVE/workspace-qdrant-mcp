"""
Resource Exhaustion Simulation for Stress Testing

This module provides controlled simulation of resource exhaustion scenarios
to test system behavior under extreme resource pressure including memory,
CPU, disk I/O, and network bandwidth constraints.

Features:
- Memory pressure simulation with safe limits
- CPU saturation with configurable utilization
- Disk I/O saturation through file operations
- Network bandwidth simulation (token bucket)
- Automatic cleanup and safety checks
- Integration with stress test orchestration
"""

import asyncio
import logging
import multiprocessing
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import psutil

# Safety limits to prevent system crashes
SAFETY_MAX_MEMORY_PERCENT = 80  # Max 80% of total system RAM
SAFETY_MAX_CPU_PERCENT = 90     # Max 90% CPU utilization
SAFETY_MAX_DISK_PERCENT = 80    # Max 80% of available disk space


class ResourceType(Enum):
    """Types of resources that can be exhausted."""
    MEMORY = "memory"
    CPU = "cpu"
    DISK_IO = "disk_io"
    NETWORK = "network"


@dataclass
class ResourceExhaustionScenario:
    """Configuration for a resource exhaustion scenario."""
    memory_target_mb: int | None = None
    cpu_target_percent: int | None = None
    disk_io_target_mb_per_sec: int | None = None
    network_target_mb_per_sec: int | None = None
    duration_seconds: float = 60.0
    ramp_up_seconds: float = 10.0

    def __post_init__(self):
        """Validate scenario configuration."""
        # Validate memory target
        if self.memory_target_mb is not None:
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            max_allowed_mb = total_memory_mb * (SAFETY_MAX_MEMORY_PERCENT / 100.0)
            if self.memory_target_mb > max_allowed_mb:
                raise ValueError(
                    f"Memory target {self.memory_target_mb}MB exceeds safety limit "
                    f"{max_allowed_mb:.0f}MB ({SAFETY_MAX_MEMORY_PERCENT}% of {total_memory_mb:.0f}MB)"
                )

        # Validate CPU target
        if self.cpu_target_percent is not None:
            if self.cpu_target_percent > SAFETY_MAX_CPU_PERCENT:
                raise ValueError(
                    f"CPU target {self.cpu_target_percent}% exceeds safety limit "
                    f"{SAFETY_MAX_CPU_PERCENT}%"
                )


class MemoryPressureSimulator:
    """
    Simulates memory pressure through controlled allocation.

    Gradually allocates memory to achieve target usage, maintaining
    references to prevent garbage collection.
    """

    def __init__(self):
        """Initialize memory pressure simulator."""
        self.logger = logging.getLogger(__name__)
        self._allocated_blocks: list[bytearray] = []
        self._target_mb: int = 0
        self._running = False

    def start(self, target_mb: int) -> None:
        """Start allocating memory to reach target.

        Args:
            target_mb: Target memory allocation in MB
        """
        self._target_mb = target_mb
        self._running = True

        # Check safety limit
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        max_allowed_mb = total_memory_mb * (SAFETY_MAX_MEMORY_PERCENT / 100.0)

        if target_mb > max_allowed_mb:
            self.logger.warning(
                f"Memory target {target_mb}MB exceeds safety limit "
                f"{max_allowed_mb:.0f}MB, clamping to limit"
            )
            self._target_mb = int(max_allowed_mb)

        # Allocate in 100MB chunks
        chunk_size_mb = 100
        num_chunks = self._target_mb // chunk_size_mb

        self.logger.info(f"Allocating {self._target_mb}MB in {num_chunks} chunks")

        for i in range(num_chunks):
            if not self._running:
                break

            try:
                # Allocate chunk
                chunk = bytearray(chunk_size_mb * 1024 * 1024)
                self._allocated_blocks.append(chunk)

                # Measure actual usage
                process = psutil.Process()
                current_mb = process.memory_info().rss / (1024 * 1024)
                self.logger.debug(f"Allocated chunk {i+1}/{num_chunks}, current memory: {current_mb:.0f}MB")

                # Small delay to allow system to stabilize
                time.sleep(0.1)

            except MemoryError:
                self.logger.error(f"MemoryError after allocating {i} chunks")
                break

        final_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        self.logger.info(f"Memory pressure simulation started: {final_mb:.0f}MB allocated")

    def release(self) -> None:
        """Release all allocated memory."""
        self._running = False
        self._allocated_blocks.clear()
        self._target_mb = 0
        self.logger.info("Memory pressure simulation stopped")


class CPUSaturationSimulator:
    """
    Simulates CPU saturation through busy loops.

    Spawns worker threads that consume CPU cycles to achieve
    target utilization percentage.
    """

    def __init__(self):
        """Initialize CPU saturation simulator."""
        self.logger = logging.getLogger(__name__)
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._target_percent: int = 0

    def start(self, target_percent: int, num_cores: int = -1) -> None:
        """Start CPU saturation.

        Args:
            target_percent: Target CPU utilization percentage
            num_cores: Number of cores to use (-1 = all cores)
        """
        if target_percent > SAFETY_MAX_CPU_PERCENT:
            self.logger.warning(
                f"CPU target {target_percent}% exceeds safety limit "
                f"{SAFETY_MAX_CPU_PERCENT}%, clamping to limit"
            )
            target_percent = SAFETY_MAX_CPU_PERCENT

        self._target_percent = target_percent
        self._stop_event.clear()

        # Determine number of cores
        if num_cores <= 0:
            num_cores = multiprocessing.cpu_count()

        self.logger.info(
            f"Starting CPU saturation: {target_percent}% utilization across {num_cores} cores"
        )

        # Spawn worker threads
        for _i in range(num_cores):
            worker = threading.Thread(
                target=self._cpu_worker,
                args=(target_percent,),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

        self.logger.info(f"Started {len(self._workers)} CPU saturation workers")

    def stop(self) -> None:
        """Stop CPU saturation."""
        self._stop_event.set()

        # Wait for workers to stop
        for worker in self._workers:
            worker.join(timeout=1.0)

        self._workers.clear()
        self.logger.info("CPU saturation simulation stopped")

    def _cpu_worker(self, target_percent: int) -> None:
        """Worker thread that consumes CPU cycles.

        Args:
            target_percent: Target CPU utilization percentage
        """
        # Calculate sleep time to achieve target
        # Run busy loop for target_percent% of each interval
        interval = 0.1  # 100ms intervals
        busy_time = interval * (target_percent / 100.0)
        sleep_time = interval - busy_time

        while not self._stop_event.is_set():
            # Busy loop
            start = time.time()
            while time.time() - start < busy_time:
                # Perform CPU-intensive work
                _ = sum(i * i for i in range(1000))

            # Sleep to throttle
            if sleep_time > 0:
                time.sleep(sleep_time)


class DiskIOSaturationSimulator:
    """
    Simulates disk I/O saturation through file operations.

    Creates large temporary files and performs continuous read/write
    operations to achieve target throughput.
    """

    def __init__(self):
        """Initialize disk I/O saturation simulator."""
        self.logger = logging.getLogger(__name__)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._temp_dir: Path | None = None
        self._temp_files: list[Path] = []

    def start(self, target_mb_per_sec: int, temp_dir: Path) -> None:
        """Start disk I/O saturation.

        Args:
            target_mb_per_sec: Target disk throughput in MB/s
            temp_dir: Directory for temporary files
        """
        self._temp_dir = temp_dir
        self._stop_event.clear()

        # Check available disk space
        disk_usage = psutil.disk_usage(str(temp_dir))
        available_mb = disk_usage.free / (1024 * 1024)
        max_file_size_mb = available_mb * (SAFETY_MAX_DISK_PERCENT / 100.0)

        # Create temp file (1GB max)
        file_size_mb = min(1024, max_file_size_mb)

        self.logger.info(
            f"Starting disk I/O saturation: {target_mb_per_sec}MB/s target, "
            f"using {file_size_mb}MB temp file"
        )

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._disk_io_worker,
            args=(target_mb_per_sec, file_size_mb),
            daemon=True
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop disk I/O saturation."""
        self._stop_event.set()

        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        # Clean up temp files
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                self.logger.error(f"Error removing temp file {temp_file}: {e}")

        self._temp_files.clear()
        self.logger.info("Disk I/O saturation simulation stopped")

    def _disk_io_worker(self, target_mb_per_sec: int, file_size_mb: int) -> None:
        """Worker thread that performs disk I/O.

        Args:
            target_mb_per_sec: Target throughput in MB/s
            file_size_mb: Size of temp file in MB
        """
        # Create temp file
        temp_file = Path(self._temp_dir) / f"stress_test_{time.time()}.tmp"
        self._temp_files.append(temp_file)

        try:
            # Create file with random data
            chunk_size = 1024 * 1024  # 1MB chunks
            data = os.urandom(chunk_size)

            with open(temp_file, 'wb') as f:
                for _ in range(file_size_mb):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())

            self.logger.info(f"Created {file_size_mb}MB temp file: {temp_file}")

            # Continuous read/write loop
            while not self._stop_event.is_set():
                start_time = time.time()
                bytes_transferred = 0

                # Write cycle
                with open(temp_file, 'rb+') as f:
                    # Read and write chunks to achieve target throughput
                    while bytes_transferred < target_mb_per_sec * 1024 * 1024:
                        if self._stop_event.is_set():
                            break

                        # Read chunk
                        chunk = f.read(chunk_size)
                        if not chunk:
                            f.seek(0)
                            continue

                        # Write back
                        f.seek(f.tell() - len(chunk))
                        f.write(chunk)
                        bytes_transferred += len(chunk)

                    f.flush()
                    os.fsync(f.fileno())

                # Calculate sleep time to maintain target rate
                elapsed = time.time() - start_time
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)

        except Exception as e:
            self.logger.error(f"Disk I/O worker error: {e}")


class NetworkBandwidthLimiter:
    """
    Simulates network bandwidth constraints using token bucket algorithm.

    For testing purposes, tracks theoretical bandwidth usage.
    Real network limiting requires OS-level tools (tc, netem).
    """

    def __init__(self):
        """Initialize network bandwidth limiter."""
        self.logger = logging.getLogger(__name__)
        self._target_mb_per_sec: int = 0
        self._tokens: float = 0.0
        self._last_refill: float = 0.0
        self._running = False

    def start(self, target_mb_per_sec: int) -> None:
        """Start bandwidth limiting.

        Args:
            target_mb_per_sec: Target bandwidth in MB/s
        """
        self._target_mb_per_sec = target_mb_per_sec
        self._tokens = target_mb_per_sec * 1024 * 1024  # Convert to bytes
        self._last_refill = time.time()
        self._running = True

        self.logger.info(f"Network bandwidth limiter started: {target_mb_per_sec}MB/s")
        self.logger.warning(
            "Note: Network bandwidth limiting is simulated via token bucket. "
            "Real network limiting requires OS-level tools (tc, netem)"
        )

    def stop(self) -> None:
        """Stop bandwidth limiting."""
        self._running = False
        self.logger.info("Network bandwidth limiter stopped")

    def consume_bandwidth(self, bytes_count: int) -> float:
        """Consume bandwidth tokens and return delay needed.

        Args:
            bytes_count: Number of bytes to transmit

        Returns:
            Delay in seconds needed to comply with bandwidth limit
        """
        if not self._running:
            return 0.0

        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self._last_refill
        refill_amount = elapsed * self._target_mb_per_sec * 1024 * 1024
        self._tokens = min(
            self._tokens + refill_amount,
            self._target_mb_per_sec * 1024 * 1024
        )
        self._last_refill = now

        # Check if we have enough tokens
        if bytes_count <= self._tokens:
            self._tokens -= bytes_count
            return 0.0
        else:
            # Calculate delay needed
            deficit = bytes_count - self._tokens
            delay = deficit / (self._target_mb_per_sec * 1024 * 1024)
            self._tokens = 0
            return delay


class ResourceExhaustionSimulator:
    """
    Main simulator coordinating all resource exhaustion scenarios.

    Manages memory, CPU, disk I/O, and network bandwidth simulation
    with automatic cleanup and safety checks.
    """

    def __init__(self):
        """Initialize resource exhaustion simulator."""
        self.logger = logging.getLogger(__name__)

        # Individual simulators
        self.memory_simulator = MemoryPressureSimulator()
        self.cpu_simulator = CPUSaturationSimulator()
        self.disk_io_simulator = DiskIOSaturationSimulator()
        self.network_limiter = NetworkBandwidthLimiter()

        # Active simulations tracking
        self._active_simulations: dict[ResourceType, bool] = {}
        self._temp_dir: Path | None = None

    async def simulate_memory_pressure(
        self,
        target_mb: int,
        duration_seconds: float
    ) -> None:
        """Simulate memory pressure for specified duration.

        Args:
            target_mb: Target memory allocation in MB
            duration_seconds: How long to maintain pressure
        """
        self.logger.info(f"Simulating memory pressure: {target_mb}MB for {duration_seconds}s")

        try:
            self.memory_simulator.start(target_mb)
            self._active_simulations[ResourceType.MEMORY] = True

            # Run for duration
            await asyncio.sleep(duration_seconds)

        finally:
            self.memory_simulator.release()
            del self._active_simulations[ResourceType.MEMORY]

    async def simulate_cpu_saturation(
        self,
        target_percent: int,
        duration_seconds: float
    ) -> None:
        """Simulate CPU saturation for specified duration.

        Args:
            target_percent: Target CPU utilization percentage
            duration_seconds: How long to maintain saturation
        """
        self.logger.info(f"Simulating CPU saturation: {target_percent}% for {duration_seconds}s")

        try:
            self.cpu_simulator.start(target_percent)
            self._active_simulations[ResourceType.CPU] = True

            # Run for duration
            await asyncio.sleep(duration_seconds)

        finally:
            self.cpu_simulator.stop()
            del self._active_simulations[ResourceType.CPU]

    async def simulate_disk_io_saturation(
        self,
        target_mb_per_sec: int,
        duration_seconds: float
    ) -> None:
        """Simulate disk I/O saturation for specified duration.

        Args:
            target_mb_per_sec: Target disk throughput in MB/s
            duration_seconds: How long to maintain saturation
        """
        self.logger.info(
            f"Simulating disk I/O saturation: {target_mb_per_sec}MB/s for {duration_seconds}s"
        )

        # Create temp directory if needed
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="stress_test_"))

        try:
            self.disk_io_simulator.start(target_mb_per_sec, self._temp_dir)
            self._active_simulations[ResourceType.DISK_IO] = True

            # Run for duration
            await asyncio.sleep(duration_seconds)

        finally:
            self.disk_io_simulator.stop()
            del self._active_simulations[ResourceType.DISK_IO]

    async def simulate_network_bandwidth_limit(
        self,
        target_mb_per_sec: int,
        duration_seconds: float
    ) -> None:
        """Simulate network bandwidth constraint for specified duration.

        Args:
            target_mb_per_sec: Target bandwidth in MB/s
            duration_seconds: How long to maintain limit
        """
        self.logger.info(
            f"Simulating network bandwidth limit: {target_mb_per_sec}MB/s for {duration_seconds}s"
        )

        try:
            self.network_limiter.start(target_mb_per_sec)
            self._active_simulations[ResourceType.NETWORK] = True

            # Run for duration
            await asyncio.sleep(duration_seconds)

        finally:
            self.network_limiter.stop()
            del self._active_simulations[ResourceType.NETWORK]

    def stop_all_simulations(self) -> None:
        """Stop all active resource exhaustion simulations."""
        self.logger.info("Stopping all resource exhaustion simulations")

        # Stop each simulator
        if self._active_simulations.get(ResourceType.MEMORY):
            self.memory_simulator.release()

        if self._active_simulations.get(ResourceType.CPU):
            self.cpu_simulator.stop()

        if self._active_simulations.get(ResourceType.DISK_IO):
            self.disk_io_simulator.stop()

        if self._active_simulations.get(ResourceType.NETWORK):
            self.network_limiter.stop()

        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
                self.logger.info(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                self.logger.error(f"Error cleaning up temp directory: {e}")
            finally:
                self._temp_dir = None

        self._active_simulations.clear()

    async def execute_scenario(self, scenario: ResourceExhaustionScenario) -> None:
        """Execute a complete resource exhaustion scenario.

        Args:
            scenario: Resource exhaustion scenario configuration
        """
        self.logger.info(f"Executing resource exhaustion scenario: {scenario}")

        # Launch simulations concurrently
        tasks = []

        if scenario.memory_target_mb is not None:
            tasks.append(
                self.simulate_memory_pressure(
                    scenario.memory_target_mb,
                    scenario.duration_seconds
                )
            )

        if scenario.cpu_target_percent is not None:
            tasks.append(
                self.simulate_cpu_saturation(
                    scenario.cpu_target_percent,
                    scenario.duration_seconds
                )
            )

        if scenario.disk_io_target_mb_per_sec is not None:
            tasks.append(
                self.simulate_disk_io_saturation(
                    scenario.disk_io_target_mb_per_sec,
                    scenario.duration_seconds
                )
            )

        if scenario.network_target_mb_per_sec is not None:
            tasks.append(
                self.simulate_network_bandwidth_limit(
                    scenario.network_target_mb_per_sec,
                    scenario.duration_seconds
                )
            )

        # Run all simulations
        try:
            await asyncio.gather(*tasks)
        finally:
            # Ensure cleanup
            self.stop_all_simulations()

        self.logger.info("Resource exhaustion scenario completed")
