"""
Network Instability Simulation for Stress Testing

This module provides controlled simulation of network instability scenarios
to test system behavior under adverse network conditions including latency,
packet loss, connection timeouts, and intermittent connectivity.

Features:
- Network latency injection with configurable delay distributions
- Packet loss simulation with configurable drop rates
- Connection timeout scenarios
- Intermittent connectivity patterns
- Network partition simulation
- Integration with stress test orchestration
"""

import asyncio
import logging
import random
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from collections import defaultdict


class NetworkCondition(Enum):
    """Types of network conditions that can be simulated."""
    LATENCY = "latency"
    PACKET_LOSS = "packet_loss"
    TIMEOUT = "timeout"
    INTERMITTENT = "intermittent"
    PARTITION = "partition"


class LatencyDistribution(Enum):
    """Distribution patterns for latency injection."""
    CONSTANT = "constant"  # Fixed delay
    UNIFORM = "uniform"    # Random delay within range
    NORMAL = "normal"      # Normal distribution
    SPIKE = "spike"        # Occasional large spikes


@dataclass
class NetworkInstabilityScenario:
    """Configuration for a network instability scenario."""
    latency_ms: Optional[int] = None
    latency_distribution: LatencyDistribution = LatencyDistribution.CONSTANT
    latency_variance_ms: Optional[int] = None
    packet_loss_percent: Optional[float] = None
    connection_timeout_ms: Optional[int] = None
    intermittent_pattern_seconds: Optional[float] = None
    intermittent_uptime_percent: float = 50.0
    duration_seconds: float = 60.0

    def __post_init__(self):
        """Validate scenario configuration."""
        if self.packet_loss_percent is not None:
            if not 0 <= self.packet_loss_percent <= 100:
                raise ValueError(
                    f"Packet loss percent must be 0-100, got {self.packet_loss_percent}"
                )

        if self.intermittent_uptime_percent < 0 or self.intermittent_uptime_percent > 100:
            raise ValueError(
                f"Intermittent uptime percent must be 0-100, got {self.intermittent_uptime_percent}"
            )


class LatencyInjector:
    """
    Simulates network latency through controlled delays.

    Supports multiple delay distributions and can be applied
    to async operations to simulate network round-trip time.
    """

    def __init__(self):
        """Initialize latency injector."""
        self.logger = logging.getLogger(__name__)
        self._target_latency_ms: int = 0
        self._variance_ms: int = 0
        self._distribution: LatencyDistribution = LatencyDistribution.CONSTANT
        self._running = False
        self._delay_count = 0

    def start(
        self,
        latency_ms: int,
        distribution: LatencyDistribution = LatencyDistribution.CONSTANT,
        variance_ms: int = 0
    ) -> None:
        """Start latency injection.

        Args:
            latency_ms: Base latency in milliseconds
            distribution: Delay distribution pattern
            variance_ms: Variance for distributions (for UNIFORM, NORMAL)
        """
        self._target_latency_ms = latency_ms
        self._variance_ms = variance_ms
        self._distribution = distribution
        self._running = True
        self._delay_count = 0

        self.logger.info(
            f"Latency injector started: {latency_ms}ms "
            f"({distribution.value}, variance={variance_ms}ms)"
        )

    def stop(self) -> None:
        """Stop latency injection."""
        self._running = False
        self.logger.info(
            f"Latency injector stopped (applied {self._delay_count} delays)"
        )

    async def apply_delay(self) -> float:
        """Apply latency delay based on configuration.

        Returns:
            Actual delay applied in seconds
        """
        if not self._running:
            return 0.0

        delay_ms = self._calculate_delay()
        delay_seconds = delay_ms / 1000.0

        await asyncio.sleep(delay_seconds)
        self._delay_count += 1

        return delay_seconds

    def _calculate_delay(self) -> float:
        """Calculate delay based on distribution.

        Returns:
            Delay in milliseconds
        """
        if self._distribution == LatencyDistribution.CONSTANT:
            return self._target_latency_ms

        elif self._distribution == LatencyDistribution.UNIFORM:
            min_delay = max(0, self._target_latency_ms - self._variance_ms)
            max_delay = self._target_latency_ms + self._variance_ms
            return random.uniform(min_delay, max_delay)

        elif self._distribution == LatencyDistribution.NORMAL:
            # Normal distribution with target as mean, variance as std dev
            delay = random.gauss(self._target_latency_ms, self._variance_ms)
            return max(0, delay)  # Ensure non-negative

        elif self._distribution == LatencyDistribution.SPIKE:
            # 95% normal latency, 5% spike to 10x latency
            if random.random() < 0.05:
                return self._target_latency_ms * 10
            return self._target_latency_ms

        return self._target_latency_ms


class PacketLossSimulator:
    """
    Simulates packet loss by randomly dropping network operations.

    Uses configurable drop rate to simulate unreliable network conditions.
    """

    def __init__(self):
        """Initialize packet loss simulator."""
        self.logger = logging.getLogger(__name__)
        self._drop_rate: float = 0.0
        self._running = False
        self._packets_sent = 0
        self._packets_dropped = 0

    def start(self, drop_rate_percent: float) -> None:
        """Start packet loss simulation.

        Args:
            drop_rate_percent: Percentage of packets to drop (0-100)
        """
        if not 0 <= drop_rate_percent <= 100:
            raise ValueError(f"Drop rate must be 0-100, got {drop_rate_percent}")

        self._drop_rate = drop_rate_percent / 100.0
        self._running = True
        self._packets_sent = 0
        self._packets_dropped = 0

        self.logger.info(f"Packet loss simulator started: {drop_rate_percent}% drop rate")

    def stop(self) -> None:
        """Stop packet loss simulation."""
        self._running = False
        actual_drop_rate = (
            (self._packets_dropped / self._packets_sent * 100)
            if self._packets_sent > 0 else 0
        )
        self.logger.info(
            f"Packet loss simulator stopped "
            f"(dropped {self._packets_dropped}/{self._packets_sent} "
            f"= {actual_drop_rate:.1f}%)"
        )

    def should_drop_packet(self) -> bool:
        """Determine if packet should be dropped.

        Returns:
            True if packet should be dropped, False otherwise
        """
        if not self._running:
            return False

        self._packets_sent += 1
        should_drop = random.random() < self._drop_rate

        if should_drop:
            self._packets_dropped += 1

        return should_drop


class ConnectionTimeoutSimulator:
    """
    Simulates connection timeout scenarios.

    Forces operations to wait beyond timeout thresholds to test
    timeout handling and recovery mechanisms.
    """

    def __init__(self):
        """Initialize connection timeout simulator."""
        self.logger = logging.getLogger(__name__)
        self._timeout_ms: int = 0
        self._running = False
        self._timeout_count = 0

    def start(self, timeout_ms: int) -> None:
        """Start connection timeout simulation.

        Args:
            timeout_ms: Timeout delay in milliseconds
        """
        self._timeout_ms = timeout_ms
        self._running = True
        self._timeout_count = 0

        self.logger.info(f"Connection timeout simulator started: {timeout_ms}ms")

    def stop(self) -> None:
        """Stop connection timeout simulation."""
        self._running = False
        self.logger.info(
            f"Connection timeout simulator stopped "
            f"({self._timeout_count} timeouts triggered)"
        )

    async def force_timeout(self) -> None:
        """Force operation to exceed timeout threshold."""
        if not self._running:
            return

        # Wait slightly longer than timeout to trigger timeout handling
        delay_seconds = (self._timeout_ms * 1.1) / 1000.0
        await asyncio.sleep(delay_seconds)
        self._timeout_count += 1


class IntermittentConnectivitySimulator:
    """
    Simulates intermittent network connectivity.

    Cycles between available and unavailable states based on
    configurable pattern and uptime percentage.
    """

    def __init__(self):
        """Initialize intermittent connectivity simulator."""
        self.logger = logging.getLogger(__name__)
        self._pattern_seconds: float = 10.0
        self._uptime_percent: float = 50.0
        self._running = False
        self._available = True
        self._last_state_change = time.time()
        self._state_lock = threading.Lock()

    def start(
        self,
        pattern_seconds: float = 10.0,
        uptime_percent: float = 50.0
    ) -> None:
        """Start intermittent connectivity simulation.

        Args:
            pattern_seconds: Duration of one up/down cycle
            uptime_percent: Percentage of time network is available
        """
        self._pattern_seconds = pattern_seconds
        self._uptime_percent = uptime_percent
        self._running = True
        self._available = True
        self._last_state_change = time.time()

        self.logger.info(
            f"Intermittent connectivity simulator started: "
            f"{pattern_seconds}s pattern, {uptime_percent}% uptime"
        )

    def stop(self) -> None:
        """Stop intermittent connectivity simulation."""
        self._running = False
        self.logger.info("Intermittent connectivity simulator stopped")

    def is_available(self) -> bool:
        """Check if network is currently available.

        Returns:
            True if network is available, False otherwise
        """
        if not self._running:
            return True

        with self._state_lock:
            current_time = time.time()
            elapsed = current_time - self._last_state_change

            # Calculate state durations
            uptime_duration = self._pattern_seconds * (self._uptime_percent / 100.0)
            downtime_duration = self._pattern_seconds - uptime_duration

            # Check if state should change
            if self._available and elapsed >= uptime_duration:
                self._available = False
                self._last_state_change = current_time
                self.logger.debug("Network became unavailable")
            elif not self._available and elapsed >= downtime_duration:
                self._available = True
                self._last_state_change = current_time
                self.logger.debug("Network became available")

            return self._available


class NetworkInstabilitySimulator:
    """
    Main simulator coordinating all network instability scenarios.

    Manages latency injection, packet loss, timeouts, and intermittent
    connectivity with automatic cleanup and safety checks.
    """

    def __init__(self):
        """Initialize network instability simulator."""
        self.logger = logging.getLogger(__name__)

        # Individual simulators
        self.latency_injector = LatencyInjector()
        self.packet_loss_simulator = PacketLossSimulator()
        self.timeout_simulator = ConnectionTimeoutSimulator()
        self.intermittent_simulator = IntermittentConnectivitySimulator()

        # Active conditions tracking
        self._active_conditions: Dict[NetworkCondition, bool] = {}

    async def simulate_latency(
        self,
        latency_ms: int,
        duration_seconds: float,
        distribution: LatencyDistribution = LatencyDistribution.CONSTANT,
        variance_ms: int = 0
    ) -> None:
        """Simulate network latency for specified duration.

        Args:
            latency_ms: Base latency in milliseconds
            duration_seconds: How long to maintain latency
            distribution: Delay distribution pattern
            variance_ms: Variance for distributions
        """
        self.logger.info(
            f"Simulating latency: {latency_ms}ms for {duration_seconds}s "
            f"({distribution.value})"
        )

        try:
            self.latency_injector.start(latency_ms, distribution, variance_ms)
            self._active_conditions[NetworkCondition.LATENCY] = True

            await asyncio.sleep(duration_seconds)

        finally:
            self.latency_injector.stop()
            del self._active_conditions[NetworkCondition.LATENCY]

    async def simulate_packet_loss(
        self,
        drop_rate_percent: float,
        duration_seconds: float
    ) -> None:
        """Simulate packet loss for specified duration.

        Args:
            drop_rate_percent: Percentage of packets to drop (0-100)
            duration_seconds: How long to maintain packet loss
        """
        self.logger.info(
            f"Simulating packet loss: {drop_rate_percent}% for {duration_seconds}s"
        )

        try:
            self.packet_loss_simulator.start(drop_rate_percent)
            self._active_conditions[NetworkCondition.PACKET_LOSS] = True

            await asyncio.sleep(duration_seconds)

        finally:
            self.packet_loss_simulator.stop()
            del self._active_conditions[NetworkCondition.PACKET_LOSS]

    async def simulate_connection_timeout(
        self,
        timeout_ms: int,
        duration_seconds: float
    ) -> None:
        """Simulate connection timeouts for specified duration.

        Args:
            timeout_ms: Timeout delay in milliseconds
            duration_seconds: How long to maintain timeouts
        """
        self.logger.info(
            f"Simulating connection timeouts: {timeout_ms}ms for {duration_seconds}s"
        )

        try:
            self.timeout_simulator.start(timeout_ms)
            self._active_conditions[NetworkCondition.TIMEOUT] = True

            await asyncio.sleep(duration_seconds)

        finally:
            self.timeout_simulator.stop()
            del self._active_conditions[NetworkCondition.TIMEOUT]

    async def simulate_intermittent_connectivity(
        self,
        pattern_seconds: float,
        uptime_percent: float,
        duration_seconds: float
    ) -> None:
        """Simulate intermittent connectivity for specified duration.

        Args:
            pattern_seconds: Duration of one up/down cycle
            uptime_percent: Percentage of time network is available
            duration_seconds: How long to maintain intermittent pattern
        """
        self.logger.info(
            f"Simulating intermittent connectivity: {pattern_seconds}s pattern, "
            f"{uptime_percent}% uptime for {duration_seconds}s"
        )

        try:
            self.intermittent_simulator.start(pattern_seconds, uptime_percent)
            self._active_conditions[NetworkCondition.INTERMITTENT] = True

            await asyncio.sleep(duration_seconds)

        finally:
            self.intermittent_simulator.stop()
            del self._active_conditions[NetworkCondition.INTERMITTENT]

    def stop_all_simulations(self) -> None:
        """Stop all active network instability simulations."""
        self.logger.info("Stopping all network instability simulations")

        if self._active_conditions.get(NetworkCondition.LATENCY):
            self.latency_injector.stop()

        if self._active_conditions.get(NetworkCondition.PACKET_LOSS):
            self.packet_loss_simulator.stop()

        if self._active_conditions.get(NetworkCondition.TIMEOUT):
            self.timeout_simulator.stop()

        if self._active_conditions.get(NetworkCondition.INTERMITTENT):
            self.intermittent_simulator.stop()

        self._active_conditions.clear()

    async def execute_scenario(self, scenario: NetworkInstabilityScenario) -> None:
        """Execute a complete network instability scenario.

        Args:
            scenario: Network instability scenario configuration
        """
        self.logger.info(f"Executing network instability scenario: {scenario}")

        # Launch simulations concurrently
        tasks = []

        if scenario.latency_ms is not None:
            tasks.append(
                self.simulate_latency(
                    scenario.latency_ms,
                    scenario.duration_seconds,
                    scenario.latency_distribution,
                    scenario.latency_variance_ms or 0
                )
            )

        if scenario.packet_loss_percent is not None:
            tasks.append(
                self.simulate_packet_loss(
                    scenario.packet_loss_percent,
                    scenario.duration_seconds
                )
            )

        if scenario.connection_timeout_ms is not None:
            tasks.append(
                self.simulate_connection_timeout(
                    scenario.connection_timeout_ms,
                    scenario.duration_seconds
                )
            )

        if scenario.intermittent_pattern_seconds is not None:
            tasks.append(
                self.simulate_intermittent_connectivity(
                    scenario.intermittent_pattern_seconds,
                    scenario.intermittent_uptime_percent,
                    scenario.duration_seconds
                )
            )

        # Run all simulations
        try:
            await asyncio.gather(*tasks)
        finally:
            # Ensure cleanup
            self.stop_all_simulations()

        self.logger.info("Network instability scenario completed")
