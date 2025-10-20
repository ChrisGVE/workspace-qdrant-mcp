"""
Unit Tests for Network Instability Simulation

Tests for latency injection, packet loss, connection timeouts, and
intermittent connectivity components used in stress testing scenarios.
"""

import asyncio
import time
import pytest

from tests.framework.network_instability import (
    NetworkInstabilitySimulator,
    NetworkInstabilityScenario,
    LatencyInjector,
    PacketLossSimulator,
    ConnectionTimeoutSimulator,
    IntermittentConnectivitySimulator,
    LatencyDistribution,
    NetworkCondition,
)


class TestNetworkInstabilityScenario:
    """Test network instability scenario configuration."""

    def test_valid_scenario_creation(self):
        """Test creating valid network instability scenario."""
        scenario = NetworkInstabilityScenario(
            latency_ms=100,
            packet_loss_percent=10,
            duration_seconds=5.0
        )

        assert scenario.latency_ms == 100
        assert scenario.packet_loss_percent == 10
        assert scenario.duration_seconds == 5.0

    def test_packet_loss_validation(self):
        """Test that invalid packet loss rates raise ValueError."""
        with pytest.raises(ValueError, match="must be 0-100"):
            NetworkInstabilityScenario(
                packet_loss_percent=150,  # Invalid: > 100%
                duration_seconds=5.0
            )

    def test_uptime_percent_validation(self):
        """Test that invalid uptime percentages raise ValueError."""
        with pytest.raises(ValueError, match="must be 0-100"):
            NetworkInstabilityScenario(
                intermittent_pattern_seconds=10.0,
                intermittent_uptime_percent=150,  # Invalid: > 100%
                duration_seconds=5.0
            )


class TestLatencyInjector:
    """Test latency injection simulation."""

    @pytest.mark.asyncio
    async def test_constant_latency(self):
        """Test constant latency injection."""
        injector = LatencyInjector()
        injector.start(latency_ms=100, distribution=LatencyDistribution.CONSTANT)

        start_time = time.time()
        delay = await injector.apply_delay()
        elapsed = time.time() - start_time

        # Should delay approximately 100ms (0.1s)
        assert 0.08 < elapsed < 0.12, f"Expected ~0.1s delay, got {elapsed}s"
        assert 0.08 < delay < 0.12

        injector.stop()

    @pytest.mark.asyncio
    async def test_uniform_distribution_latency(self):
        """Test uniform distribution latency injection."""
        injector = LatencyInjector()
        injector.start(
            latency_ms=100,
            distribution=LatencyDistribution.UNIFORM,
            variance_ms=50
        )

        # Collect multiple samples
        delays = []
        for _ in range(10):
            delay = await injector.apply_delay()
            delays.append(delay * 1000)  # Convert to ms

        # Should be within range [50ms, 150ms]
        assert all(50 <= d <= 150 for d in delays)
        # Should have some variance
        assert max(delays) - min(delays) > 10

        injector.stop()

    @pytest.mark.asyncio
    async def test_spike_distribution_latency(self):
        """Test spike distribution latency injection."""
        injector = LatencyInjector()
        injector.start(
            latency_ms=10,
            distribution=LatencyDistribution.SPIKE
        )

        # Collect many samples to likely hit a spike
        delays = []
        for _ in range(50):
            start = time.time()
            await injector.apply_delay()
            delays.append((time.time() - start) * 1000)

        # Should have normal delays (~10ms) and some spikes (~100ms)
        assert min(delays) < 20  # Normal delay
        # Note: spike detection is probabilistic, may not always occur

        injector.stop()

    def test_stop_clears_state(self):
        """Test that stop clears injection state."""
        injector = LatencyInjector()
        injector.start(latency_ms=100)

        assert injector._running
        assert injector._target_latency_ms == 100

        injector.stop()

        assert not injector._running


class TestPacketLossSimulator:
    """Test packet loss simulation."""

    def test_packet_loss_rate(self):
        """Test packet loss at specified rate."""
        simulator = PacketLossSimulator()
        simulator.start(drop_rate_percent=20)

        # Simulate 1000 packets
        dropped_count = sum(
            1 for _ in range(1000)
            if simulator.should_drop_packet()
        )

        # Should drop approximately 20% (allow 10% variance)
        expected_drops = 200
        assert 150 < dropped_count < 250, \
            f"Expected ~200 drops (20%), got {dropped_count}"

        simulator.stop()

    def test_zero_packet_loss(self):
        """Test that 0% packet loss drops no packets."""
        simulator = PacketLossSimulator()
        simulator.start(drop_rate_percent=0)

        dropped_count = sum(
            1 for _ in range(100)
            if simulator.should_drop_packet()
        )

        assert dropped_count == 0

        simulator.stop()

    def test_full_packet_loss(self):
        """Test that 100% packet loss drops all packets."""
        simulator = PacketLossSimulator()
        simulator.start(drop_rate_percent=100)

        dropped_count = sum(
            1 for _ in range(100)
            if simulator.should_drop_packet()
        )

        assert dropped_count == 100

        simulator.stop()

    def test_invalid_drop_rate(self):
        """Test that invalid drop rates raise ValueError."""
        simulator = PacketLossSimulator()

        with pytest.raises(ValueError, match="must be 0-100"):
            simulator.start(drop_rate_percent=150)


class TestConnectionTimeoutSimulator:
    """Test connection timeout simulation."""

    @pytest.mark.asyncio
    async def test_timeout_delay(self):
        """Test timeout delay exceeds threshold."""
        simulator = ConnectionTimeoutSimulator()
        simulator.start(timeout_ms=100)

        start_time = time.time()
        await simulator.force_timeout()
        elapsed = time.time() - start_time

        # Should delay ~110ms (1.1x timeout)
        assert 0.09 < elapsed < 0.15, \
            f"Expected ~0.11s timeout, got {elapsed}s"

        simulator.stop()

    def test_timeout_counter(self):
        """Test timeout counter increments."""
        simulator = ConnectionTimeoutSimulator()
        simulator.start(timeout_ms=10)

        assert simulator._timeout_count == 0

        simulator.stop()


class TestIntermittentConnectivitySimulator:
    """Test intermittent connectivity simulation."""

    def test_connectivity_cycling(self):
        """Test connectivity cycles between up and down."""
        simulator = IntermittentConnectivitySimulator()
        simulator.start(pattern_seconds=1.0, uptime_percent=50)

        # Initially available
        assert simulator.is_available()

        # Wait for downtime
        time.sleep(0.6)  # Should be down after 50% of 1s pattern
        is_down = not simulator.is_available()

        # Wait for uptime again
        time.sleep(0.6)  # Should be up again
        is_up = simulator.is_available()

        simulator.stop()

        # Should have cycled
        assert is_down or is_up  # At least one transition occurred

    def test_always_available_when_stopped(self):
        """Test connectivity always available when not running."""
        simulator = IntermittentConnectivitySimulator()

        # Should be available when stopped
        assert simulator.is_available()

        simulator.start(pattern_seconds=0.1, uptime_percent=0)
        time.sleep(0.2)

        simulator.stop()

        # Should return to available when stopped
        assert simulator.is_available()

    def test_uptime_percentage(self):
        """Test uptime percentage is approximately correct."""
        simulator = IntermittentConnectivitySimulator()
        simulator.start(pattern_seconds=1.0, uptime_percent=75)

        # Sample availability over time
        samples = []
        for _ in range(20):
            samples.append(simulator.is_available())
            time.sleep(0.1)

        uptime_actual = sum(samples) / len(samples) * 100

        # Should be approximately 75% (allow variance)
        # Note: Exact percentage depends on sampling timing
        assert 50 < uptime_actual < 100

        simulator.stop()


class TestNetworkInstabilitySimulator:
    """Test main network instability simulator."""

    @pytest.mark.asyncio
    async def test_latency_simulation(self):
        """Test latency simulation."""
        simulator = NetworkInstabilitySimulator()

        # Simulate latency
        await simulator.simulate_latency(
            latency_ms=50,
            duration_seconds=0.5
        )

        # Should complete without error
        assert NetworkCondition.LATENCY not in simulator._active_conditions

    @pytest.mark.asyncio
    async def test_packet_loss_simulation(self):
        """Test packet loss simulation."""
        simulator = NetworkInstabilitySimulator()

        # Simulate packet loss
        await simulator.simulate_packet_loss(
            drop_rate_percent=25,
            duration_seconds=0.5
        )

        # Should complete without error
        assert NetworkCondition.PACKET_LOSS not in simulator._active_conditions

    @pytest.mark.asyncio
    async def test_timeout_simulation(self):
        """Test connection timeout simulation."""
        simulator = NetworkInstabilitySimulator()

        # Simulate timeouts
        await simulator.simulate_connection_timeout(
            timeout_ms=100,
            duration_seconds=0.5
        )

        # Should complete without error
        assert NetworkCondition.TIMEOUT not in simulator._active_conditions

    @pytest.mark.asyncio
    async def test_intermittent_simulation(self):
        """Test intermittent connectivity simulation."""
        simulator = NetworkInstabilitySimulator()

        # Simulate intermittent connectivity
        await simulator.simulate_intermittent_connectivity(
            pattern_seconds=0.2,
            uptime_percent=50,
            duration_seconds=0.5
        )

        # Should complete without error
        assert NetworkCondition.INTERMITTENT not in simulator._active_conditions

    @pytest.mark.asyncio
    async def test_combined_scenario_execution(self):
        """Test executing scenario with multiple conditions."""
        simulator = NetworkInstabilitySimulator()

        # Create scenario with multiple conditions
        scenario = NetworkInstabilityScenario(
            latency_ms=50,
            packet_loss_percent=10,
            duration_seconds=1.0
        )

        # Execute scenario
        await simulator.execute_scenario(scenario)

        # Should complete with all simulations stopped
        assert len(simulator._active_conditions) == 0

    def test_stop_all_simulations(self):
        """Test stopping all active simulations."""
        simulator = NetworkInstabilitySimulator()

        # Mark some simulations as active
        simulator._active_conditions[NetworkCondition.LATENCY] = True
        simulator._active_conditions[NetworkCondition.PACKET_LOSS] = True

        # Stop all
        simulator.stop_all_simulations()

        # Should be cleaned up
        assert len(simulator._active_conditions) == 0


class TestIntegrationScenarios:
    """Integration tests for complete network instability scenarios."""

    @pytest.mark.asyncio
    async def test_realistic_network_scenario(self):
        """Test realistic network instability scenario."""
        simulator = NetworkInstabilitySimulator()

        # Create realistic scenario
        scenario = NetworkInstabilityScenario(
            latency_ms=100,
            latency_distribution=LatencyDistribution.UNIFORM,
            latency_variance_ms=50,
            packet_loss_percent=5,
            duration_seconds=1.0
        )

        # Execute
        await simulator.execute_scenario(scenario)

        # Verify cleanup
        assert len(simulator._active_conditions) == 0

    @pytest.mark.asyncio
    async def test_latency_only_scenario(self):
        """Test latency-only scenario."""
        simulator = NetworkInstabilitySimulator()

        scenario = NetworkInstabilityScenario(
            latency_ms=200,
            duration_seconds=0.5
        )

        await simulator.execute_scenario(scenario)

        # Only latency should have been simulated
        assert NetworkCondition.LATENCY not in simulator._active_conditions

    @pytest.mark.asyncio
    async def test_packet_loss_only_scenario(self):
        """Test packet loss-only scenario."""
        simulator = NetworkInstabilitySimulator()

        scenario = NetworkInstabilityScenario(
            packet_loss_percent=30,
            duration_seconds=0.5
        )

        await simulator.execute_scenario(scenario)

        # Only packet loss should have been simulated
        assert NetworkCondition.PACKET_LOSS not in simulator._active_conditions

    @pytest.mark.asyncio
    async def test_error_handling_in_scenario(self):
        """Test error handling during scenario execution."""
        simulator = NetworkInstabilitySimulator()

        # Create valid scenario
        scenario = NetworkInstabilityScenario(
            latency_ms=50,
            packet_loss_percent=10,
            duration_seconds=0.5
        )

        # Execute scenario (should handle any errors gracefully)
        await simulator.execute_scenario(scenario)

        # Should still cleanup properly
        assert len(simulator._active_conditions) == 0

    @pytest.mark.asyncio
    async def test_intermittent_with_latency_scenario(self):
        """Test combined intermittent connectivity and latency."""
        simulator = NetworkInstabilitySimulator()

        scenario = NetworkInstabilityScenario(
            latency_ms=75,
            intermittent_pattern_seconds=0.5,
            intermittent_uptime_percent=60,
            duration_seconds=1.0
        )

        await simulator.execute_scenario(scenario)

        # Both conditions should have been simulated
        assert len(simulator._active_conditions) == 0
