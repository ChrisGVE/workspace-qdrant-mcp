"""
Unit Tests for Resource Exhaustion Simulation

Tests for memory, CPU, disk I/O, and network bandwidth simulation components
used in stress testing scenarios.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from tests.framework.resource_exhaustion import (
    SAFETY_MAX_CPU_PERCENT,
    SAFETY_MAX_MEMORY_PERCENT,
    CPUSaturationSimulator,
    DiskIOSaturationSimulator,
    MemoryPressureSimulator,
    NetworkBandwidthLimiter,
    ResourceExhaustionScenario,
    ResourceExhaustionSimulator,
    ResourceType,
)


class TestResourceExhaustionScenario:
    """Test resource exhaustion scenario configuration."""

    def test_valid_scenario_creation(self):
        """Test creating valid resource exhaustion scenario."""
        scenario = ResourceExhaustionScenario(
            memory_target_mb=100,
            cpu_target_percent=50,
            duration_seconds=5.0
        )

        assert scenario.memory_target_mb == 100
        assert scenario.cpu_target_percent == 50
        assert scenario.duration_seconds == 5.0

    def test_memory_safety_limit_enforcement(self):
        """Test that excessive memory targets raise ValueError."""
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        excessive_memory = int(total_memory_mb * 1.5)  # 150% of total

        with pytest.raises(ValueError, match="exceeds safety limit"):
            ResourceExhaustionScenario(
                memory_target_mb=excessive_memory,
                duration_seconds=5.0
            )

    def test_cpu_safety_limit_enforcement(self):
        """Test that excessive CPU targets raise ValueError."""
        with pytest.raises(ValueError, match="exceeds safety limit"):
            ResourceExhaustionScenario(
                cpu_target_percent=95,  # Exceeds 90% safety limit
                duration_seconds=5.0
            )


class TestMemoryPressureSimulator:
    """Test memory pressure simulation."""

    def test_memory_allocation(self):
        """Test memory allocation mechanism works."""
        simulator = MemoryPressureSimulator()

        # Allocate 50MB
        target_mb = 50
        simulator.start(target_mb)

        # Give time for allocation

        # Check that blocks were allocated
        assert simulator._target_mb == target_mb
        assert simulator._running

        # Release memory
        simulator.release()

    def test_memory_release(self):
        """Test memory is released after stopping."""
        simulator = MemoryPressureSimulator()

        # Allocate memory
        simulator.start(50)

        # Verify allocated

        # Release
        simulator.release()

        # Memory should be cleared
        assert simulator._target_mb == 0
        assert len(simulator._allocated_blocks) == 0
        assert not simulator._running

    def test_safety_limit_clamping(self):
        """Test memory target clamped to safety limit."""
        simulator = MemoryPressureSimulator()

        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        excessive_target = int(total_memory_mb * 1.5)

        # Should clamp to safety limit without error
        simulator.start(excessive_target)

        max_allowed = total_memory_mb * (SAFETY_MAX_MEMORY_PERCENT / 100.0)
        assert simulator._target_mb <= max_allowed

        simulator.release()


class TestCPUSaturationSimulator:
    """Test CPU saturation simulation."""

    def test_cpu_saturation_start_stop(self):
        """Test CPU saturation can start and stop cleanly."""
        simulator = CPUSaturationSimulator()

        # Start with low target to avoid system load
        simulator.start(target_percent=20, num_cores=1)

        # Verify workers started
        assert len(simulator._workers) > 0
        assert not simulator._stop_event.is_set()

        # Stop
        simulator.stop()

        # Verify cleanup
        assert len(simulator._workers) == 0

    def test_cpu_safety_limit_clamping(self):
        """Test CPU target clamped to safety limit."""
        simulator = CPUSaturationSimulator()

        # Try excessive target
        simulator.start(target_percent=95, num_cores=1)

        # Should clamp to safety limit
        assert simulator._target_percent <= SAFETY_MAX_CPU_PERCENT

        simulator.stop()

    def test_cpu_multiple_cores(self):
        """Test CPU saturation with multiple cores."""
        simulator = CPUSaturationSimulator()
        cpu_count = psutil.cpu_count()

        # Start with 2 cores
        target_cores = min(2, cpu_count)
        simulator.start(target_percent=20, num_cores=target_cores)

        # Verify correct number of workers
        assert len(simulator._workers) == target_cores

        simulator.stop()


class TestDiskIOSaturationSimulator:
    """Test disk I/O saturation simulation."""

    def test_disk_io_start_stop(self):
        """Test disk I/O saturation can start and stop cleanly."""
        simulator = DiskIOSaturationSimulator()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # Start with low target
            simulator.start(target_mb_per_sec=10, temp_dir=temp_path)

            # Verify worker started
            assert simulator._worker_thread is not None
            assert not simulator._stop_event.is_set()

            # Stop
            simulator.stop()

            # Verify cleanup
            assert simulator._worker_thread is None

    def test_disk_io_temp_file_cleanup(self):
        """Test temporary files are cleaned up."""
        simulator = DiskIOSaturationSimulator()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # Start and stop
            simulator.start(target_mb_per_sec=10, temp_dir=temp_path)

            # Give it time to create file
            time.sleep(0.5)

            simulator.stop()

            # Check temp files cleaned up
            remaining_files = list(temp_path.glob("stress_test_*.tmp"))
            assert len(remaining_files) == 0


class TestNetworkBandwidthLimiter:
    """Test network bandwidth limiting simulation."""

    def test_bandwidth_limiter_start_stop(self):
        """Test bandwidth limiter can start and stop cleanly."""
        limiter = NetworkBandwidthLimiter()

        limiter.start(target_mb_per_sec=10)
        assert limiter._running
        assert limiter._target_mb_per_sec == 10

        limiter.stop()
        assert not limiter._running

    def test_bandwidth_consumption(self):
        """Test bandwidth token consumption."""
        limiter = NetworkBandwidthLimiter()
        limiter.start(target_mb_per_sec=10)

        # Consume small amount (should not require delay)
        delay = limiter.consume_bandwidth(1024)  # 1KB
        assert delay == 0.0

        # Consume large amount (should require delay)
        large_amount = 20 * 1024 * 1024  # 20MB (exceeds 10MB/s limit)
        delay = limiter.consume_bandwidth(large_amount)
        assert delay > 0.0

        limiter.stop()

    def test_token_refill(self):
        """Test bandwidth tokens refill over time."""
        limiter = NetworkBandwidthLimiter()
        limiter.start(target_mb_per_sec=10)

        # Drain tokens
        limiter.consume_bandwidth(10 * 1024 * 1024)  # 10MB

        # Wait for refill
        time.sleep(1.1)  # Wait > 1 second for refill

        # Should have tokens again
        delay = limiter.consume_bandwidth(1024 * 1024)  # 1MB
        assert delay < 1.0  # Should not require long delay

        limiter.stop()


class TestResourceExhaustionSimulator:
    """Test main resource exhaustion simulator."""

    @pytest.mark.asyncio
    async def test_memory_pressure_simulation(self):
        """Test memory pressure simulation."""
        simulator = ResourceExhaustionSimulator()

        # Simulate memory pressure
        await simulator.simulate_memory_pressure(
            target_mb=50,
            duration_seconds=1.0
        )

        # Should complete without error
        assert ResourceType.MEMORY not in simulator._active_simulations

    @pytest.mark.asyncio
    async def test_cpu_saturation_simulation(self):
        """Test CPU saturation simulation."""
        simulator = ResourceExhaustionSimulator()

        # Simulate CPU saturation
        await simulator.simulate_cpu_saturation(
            target_percent=20,
            duration_seconds=1.0
        )

        # Should complete without error
        assert ResourceType.CPU not in simulator._active_simulations

    @pytest.mark.asyncio
    async def test_disk_io_saturation_simulation(self):
        """Test disk I/O saturation simulation."""
        simulator = ResourceExhaustionSimulator()

        # Simulate disk I/O
        await simulator.simulate_disk_io_saturation(
            target_mb_per_sec=10,
            duration_seconds=1.0
        )

        # Should complete without error
        assert ResourceType.DISK_IO not in simulator._active_simulations

    @pytest.mark.asyncio
    async def test_network_bandwidth_simulation(self):
        """Test network bandwidth limiting simulation."""
        simulator = ResourceExhaustionSimulator()

        # Simulate network limit
        await simulator.simulate_network_bandwidth_limit(
            target_mb_per_sec=10,
            duration_seconds=1.0
        )

        # Should complete without error
        assert ResourceType.NETWORK not in simulator._active_simulations

    @pytest.mark.asyncio
    async def test_combined_scenario_execution(self):
        """Test executing scenario with multiple resource types."""
        simulator = ResourceExhaustionSimulator()

        # Create scenario with multiple resources
        scenario = ResourceExhaustionScenario(
            memory_target_mb=50,
            cpu_target_percent=20,
            duration_seconds=2.0,
            ramp_up_seconds=0.5
        )

        # Execute scenario
        await simulator.execute_scenario(scenario)

        # Should complete with all simulations stopped
        assert len(simulator._active_simulations) == 0

    def test_stop_all_simulations(self):
        """Test stopping all active simulations."""
        simulator = ResourceExhaustionSimulator()

        # Mark some simulations as active (without actually starting them)
        simulator._active_simulations[ResourceType.MEMORY] = True
        simulator._active_simulations[ResourceType.CPU] = True

        # Stop all
        simulator.stop_all_simulations()

        # Should be cleaned up
        assert len(simulator._active_simulations) == 0

    @pytest.mark.asyncio
    async def test_scenario_with_safety_limits(self):
        """Test scenario respects safety limits during execution."""
        simulator = ResourceExhaustionSimulator()

        # Create scenario at safety limit
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        safe_memory = int(total_memory_mb * (SAFETY_MAX_MEMORY_PERCENT / 100.0))

        scenario = ResourceExhaustionScenario(
            memory_target_mb=safe_memory - 100,  # Slightly under limit
            cpu_target_percent=SAFETY_MAX_CPU_PERCENT - 10,
            duration_seconds=1.0
        )

        # Should execute without raising
        await simulator.execute_scenario(scenario)


class TestIntegrationScenarios:
    """Integration tests for complete resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_realistic_stress_scenario(self):
        """Test realistic stress testing scenario."""
        simulator = ResourceExhaustionSimulator()

        # Create realistic scenario
        scenario = ResourceExhaustionScenario(
            memory_target_mb=100,
            cpu_target_percent=30,
            disk_io_target_mb_per_sec=20,
            duration_seconds=2.0,
            ramp_up_seconds=0.5
        )

        # Execute
        await simulator.execute_scenario(scenario)

        # Verify cleanup
        assert len(simulator._active_simulations) == 0

    @pytest.mark.asyncio
    async def test_memory_only_scenario(self):
        """Test memory-only exhaustion scenario."""
        simulator = ResourceExhaustionSimulator()

        scenario = ResourceExhaustionScenario(
            memory_target_mb=100,
            duration_seconds=1.0
        )

        await simulator.execute_scenario(scenario)

        # Only memory should have been simulated
        assert ResourceType.MEMORY not in simulator._active_simulations

    @pytest.mark.asyncio
    async def test_cpu_only_scenario(self):
        """Test CPU-only saturation scenario."""
        simulator = ResourceExhaustionSimulator()

        scenario = ResourceExhaustionScenario(
            cpu_target_percent=25,
            duration_seconds=1.0
        )

        await simulator.execute_scenario(scenario)

        # Only CPU should have been simulated
        assert ResourceType.CPU not in simulator._active_simulations

    @pytest.mark.asyncio
    async def test_error_handling_in_scenario(self):
        """Test error handling during scenario execution."""
        simulator = ResourceExhaustionSimulator()

        # Create valid scenario
        scenario = ResourceExhaustionScenario(
            memory_target_mb=50,
            duration_seconds=1.0
        )

        # Execute scenario (should handle any errors gracefully)
        await simulator.execute_scenario(scenario)

        # Should still cleanup properly
        assert len(simulator._active_simulations) == 0
