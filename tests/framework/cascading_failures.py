"""
Cascading Failure Testing Framework

This module provides simulation of cascading failures to test system resilience
under conditions where one failure triggers additional failures through
dependencies, resource constraints, or network issues.

Features:
- Failure chain simulation with configurable propagation delays
- Dependency-based failure propagation
- Circuit breaker behavior testing
- Graceful degradation validation
- Recovery sequence testing
- Integration with resource exhaustion and network instability
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Set
from collections import defaultdict

from tests.framework.resource_exhaustion import (
    ResourceExhaustionSimulator,
    ResourceExhaustionScenario,
)
from tests.framework.network_instability import (
    NetworkInstabilitySimulator,
    NetworkInstabilityScenario,
)


class FailureType(Enum):
    """Types of failures that can occur."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    COMPONENT_CRASH = "component_crash"
    TIMEOUT = "timeout"
    OVERLOAD = "overload"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class FailureNode:
    """Represents a single failure point in the cascade."""
    node_id: str
    failure_type: FailureType
    trigger_delay_seconds: float = 0.0
    recovery_delay_seconds: float = 5.0
    dependencies: List[str] = field(default_factory=list)
    propagates_to: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeScenario:
    """Configuration for a cascading failure scenario."""
    failure_nodes: List[FailureNode]
    initial_trigger: str
    max_cascade_depth: int = 5
    propagation_delay_seconds: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout_seconds: float = 10.0
    duration_seconds: float = 30.0

    def __post_init__(self):
        """Validate scenario configuration."""
        # Validate all node IDs are unique
        node_ids = [node.node_id for node in self.failure_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Failure node IDs must be unique")

        # Validate initial trigger exists
        if self.initial_trigger not in node_ids:
            raise ValueError(
                f"Initial trigger '{self.initial_trigger}' not found in failure nodes"
            )

        # Validate dependencies and propagation targets exist
        for node in self.failure_nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    raise ValueError(
                        f"Node '{node.node_id}' depends on non-existent node '{dep}'"
                    )
            for target in node.propagates_to:
                if target not in node_ids:
                    raise ValueError(
                        f"Node '{node.node_id}' propagates to non-existent node '{target}'"
                    )


class CircuitBreaker:
    """
    Implements circuit breaker pattern for failure detection and recovery.

    Tracks failures and opens circuit when threshold is exceeded,
    preventing cascading failures by rejecting requests.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        timeout_seconds: float = 10.0
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: How long to wait before attempting recovery
        """
        self.logger = logging.getLogger(__name__)
        self._failure_threshold = failure_threshold
        self._timeout_seconds = timeout_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state_change_callbacks: List[Callable] = []

    def record_success(self) -> None:
        """Record successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            # Successful operation in half-open state, close circuit
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self._failure_threshold:
                self._transition_to(CircuitState.OPEN)

        elif self._state == CircuitState.HALF_OPEN:
            # Failure in half-open state, reopen circuit
            self._transition_to(CircuitState.OPEN)

    def can_attempt_request(self) -> bool:
        """Check if request can be attempted.

        Returns:
            True if request should be attempted, False if circuit is open
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            return True

        return False

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new circuit state.

        Args:
            new_state: Target circuit state
        """
        old_state = self._state
        self._state = new_state

        self.logger.info(f"Circuit breaker: {old_state.value} → {new_state.value}")

        # Notify callbacks
        for callback in self._state_change_callbacks:
            callback(old_state, new_state)

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes.

        Args:
            callback: Function called on state transitions
        """
        self._state_change_callbacks.append(callback)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count


class FailureChainSimulator:
    """
    Simulates cascading failure chains with dependency propagation.

    Tracks failure nodes, dependencies, and propagation paths to simulate
    realistic cascading failure scenarios.
    """

    def __init__(self):
        """Initialize failure chain simulator."""
        self.logger = logging.getLogger(__name__)
        self._nodes: Dict[str, FailureNode] = {}
        self._failed_nodes: Set[str] = set()
        self._recovering_nodes: Set[str] = set()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def add_node(self, node: FailureNode) -> None:
        """Add failure node to simulation.

        Args:
            node: Failure node configuration
        """
        self._nodes[node.node_id] = node
        self.logger.debug(f"Added failure node: {node.node_id}")

    def add_circuit_breaker(
        self,
        node_id: str,
        failure_threshold: int = 3,
        timeout_seconds: float = 10.0
    ) -> None:
        """Add circuit breaker for node.

        Args:
            node_id: Node to protect with circuit breaker
            failure_threshold: Failures before opening circuit
            timeout_seconds: Recovery timeout
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not found")

        breaker = CircuitBreaker(failure_threshold, timeout_seconds)
        self._circuit_breakers[node_id] = breaker

        self.logger.info(
            f"Added circuit breaker for {node_id} "
            f"(threshold={failure_threshold}, timeout={timeout_seconds}s)"
        )

    async def trigger_failure(self, node_id: str) -> None:
        """Trigger failure for specific node.

        Args:
            node_id: Node to fail
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not found")

        node = self._nodes[node_id]

        # Check circuit breaker
        if node_id in self._circuit_breakers:
            breaker = self._circuit_breakers[node_id]
            if not breaker.can_attempt_request():
                self.logger.warning(
                    f"Circuit breaker OPEN for {node_id}, rejecting failure trigger"
                )
                return

        self.logger.info(f"Triggering failure: {node_id} ({node.failure_type.value})")

        # Mark as failed
        self._failed_nodes.add(node_id)

        # Record failure in circuit breaker
        if node_id in self._circuit_breakers:
            self._circuit_breakers[node_id].record_failure()

        # Wait for trigger delay
        if node.trigger_delay_seconds > 0:
            await asyncio.sleep(node.trigger_delay_seconds)

        # Propagate to dependent nodes
        for target_id in node.propagates_to:
            if target_id not in self._failed_nodes:
                self.logger.info(f"Failure propagating: {node_id} → {target_id}")
                await self.trigger_failure(target_id)

    async def recover_node(self, node_id: str) -> None:
        """Recover failed node.

        Args:
            node_id: Node to recover
        """
        if node_id not in self._failed_nodes:
            self.logger.warning(f"Node {node_id} not failed, skipping recovery")
            return

        node = self._nodes[node_id]
        self.logger.info(f"Recovering node: {node_id}")

        self._recovering_nodes.add(node_id)

        # Wait for recovery delay
        if node.recovery_delay_seconds > 0:
            await asyncio.sleep(node.recovery_delay_seconds)

        # Mark as recovered
        self._failed_nodes.discard(node_id)
        self._recovering_nodes.discard(node_id)

        # Record success in circuit breaker
        if node_id in self._circuit_breakers:
            self._circuit_breakers[node_id].record_success()

        self.logger.info(f"Node recovered: {node_id}")

    def is_node_failed(self, node_id: str) -> bool:
        """Check if node is currently failed.

        Args:
            node_id: Node to check

        Returns:
            True if node is failed, False otherwise
        """
        return node_id in self._failed_nodes

    def get_failed_nodes(self) -> List[str]:
        """Get list of currently failed nodes.

        Returns:
            List of failed node IDs
        """
        return list(self._failed_nodes)

    def get_circuit_breaker_states(self) -> Dict[str, CircuitState]:
        """Get circuit breaker states for all nodes.

        Returns:
            Dictionary mapping node IDs to circuit states
        """
        return {
            node_id: breaker.state
            for node_id, breaker in self._circuit_breakers.items()
        }

    def reset(self) -> None:
        """Reset all failure states."""
        self._failed_nodes.clear()
        self._recovering_nodes.clear()
        self.logger.info("Failure chain simulator reset")


class CascadingFailureSimulator:
    """
    Main simulator coordinating cascading failure scenarios.

    Integrates failure chain simulation, resource exhaustion, network
    instability, and circuit breakers for comprehensive cascade testing.
    """

    def __init__(self):
        """Initialize cascading failure simulator."""
        self.logger = logging.getLogger(__name__)
        self.chain_simulator = FailureChainSimulator()
        self.resource_simulator = ResourceExhaustionSimulator()
        self.network_simulator = NetworkInstabilitySimulator()

    async def execute_scenario(self, scenario: CascadeScenario) -> Dict[str, Any]:
        """Execute cascading failure scenario.

        Args:
            scenario: Cascading failure scenario configuration

        Returns:
            Dictionary with test results and metrics
        """
        self.logger.info(f"Executing cascading failure scenario")

        # Setup failure nodes
        for node in scenario.failure_nodes:
            self.chain_simulator.add_node(node)

            # Add circuit breakers if enabled
            if scenario.enable_circuit_breaker:
                self.chain_simulator.add_circuit_breaker(
                    node.node_id,
                    scenario.circuit_breaker_threshold,
                    scenario.circuit_breaker_timeout_seconds
                )

        # Track metrics
        start_time = time.time()
        failure_sequence = []
        recovery_times = {}

        try:
            # Trigger initial failure
            await self.chain_simulator.trigger_failure(scenario.initial_trigger)

            # Record failure sequence
            failed_nodes = self.chain_simulator.get_failed_nodes()
            failure_sequence = failed_nodes.copy()

            # Wait for scenario duration
            await asyncio.sleep(scenario.duration_seconds)

            # Attempt recovery for all failed nodes
            recovery_start = time.time()
            for node_id in failed_nodes:
                node_recovery_start = time.time()
                await self.chain_simulator.recover_node(node_id)
                recovery_times[node_id] = time.time() - node_recovery_start

        finally:
            # Cleanup
            self.chain_simulator.reset()
            self.resource_simulator.stop_all_simulations()
            self.network_simulator.stop_all_simulations()

        # Calculate metrics
        total_duration = time.time() - start_time
        failed_node_count = len(failure_sequence)
        circuit_breaker_states = self.chain_simulator.get_circuit_breaker_states()

        results = {
            "total_duration_seconds": total_duration,
            "failed_node_count": failed_node_count,
            "failure_sequence": failure_sequence,
            "recovery_times": recovery_times,
            "circuit_breaker_states": {
                node_id: state.value
                for node_id, state in circuit_breaker_states.items()
            },
            "max_cascade_depth": len(failure_sequence),
            "scenario_completed": True
        }

        self.logger.info(
            f"Cascading failure scenario completed: "
            f"{failed_node_count} nodes failed, "
            f"{total_duration:.2f}s total duration"
        )

        return results

    async def simulate_graceful_degradation(
        self,
        primary_node: str,
        fallback_nodes: List[str],
        test_duration_seconds: float = 10.0
    ) -> Dict[str, Any]:
        """Simulate graceful degradation with fallback nodes.

        Args:
            primary_node: Primary service node
            fallback_nodes: List of fallback nodes
            test_duration_seconds: How long to run test

        Returns:
            Dictionary with degradation test results
        """
        self.logger.info(
            f"Testing graceful degradation: {primary_node} → {fallback_nodes}"
        )

        # Create nodes
        primary = FailureNode(
            node_id=primary_node,
            failure_type=FailureType.COMPONENT_CRASH,
            trigger_delay_seconds=0.0
        )
        self.chain_simulator.add_node(primary)

        fallback_sequence = []
        for i, fallback_id in enumerate(fallback_nodes):
            fallback = FailureNode(
                node_id=fallback_id,
                failure_type=FailureType.OVERLOAD,
                trigger_delay_seconds=1.0
            )
            self.chain_simulator.add_node(fallback)

        # Trigger primary failure
        start_time = time.time()
        await self.chain_simulator.trigger_failure(primary_node)

        # Wait for test duration
        await asyncio.sleep(test_duration_seconds)

        # Check which fallbacks activated
        failed_fallbacks = [
            node_id for node_id in fallback_nodes
            if self.chain_simulator.is_node_failed(node_id)
        ]

        # Cleanup
        self.chain_simulator.reset()

        results = {
            "primary_node": primary_node,
            "fallback_nodes": fallback_nodes,
            "failed_fallbacks": failed_fallbacks,
            "successful_fallbacks": [
                node_id for node_id in fallback_nodes
                if node_id not in failed_fallbacks
            ],
            "degradation_handled": len(failed_fallbacks) < len(fallback_nodes),
            "test_duration_seconds": time.time() - start_time
        }

        self.logger.info(
            f"Graceful degradation test completed: "
            f"{len(results['successful_fallbacks'])}/{len(fallback_nodes)} "
            f"fallbacks successful"
        )

        return results
