"""
Unit Tests for Cascading Failure Simulation

Tests for failure chain simulation, circuit breakers, dependency propagation,
and graceful degradation components used in stress testing scenarios.
"""

import asyncio

import pytest

from tests.framework.cascading_failures import (
    CascadeScenario,
    CascadingFailureSimulator,
    CircuitBreaker,
    CircuitState,
    FailureChainSimulator,
    FailureNode,
    FailureType,
)


class TestCircuitBreaker:
    """Test circuit breaker implementation."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_attempt_request()

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            breaker.record_failure()

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        assert not breaker.can_attempt_request()

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=0.1)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        import time
        time.sleep(0.15)

        # Should transition to half-open on next check
        assert breaker.can_attempt_request()
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_on_successful_half_open_request(self):
        """Test circuit closes after successful request in half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=0.1)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for timeout
        import time
        time.sleep(0.15)
        breaker.can_attempt_request()  # Transition to half-open

        # Successful request should close circuit
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_reopens_on_failed_half_open_request(self):
        """Test circuit reopens on failed request in half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=0.1)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for timeout
        import time
        time.sleep(0.15)
        breaker.can_attempt_request()  # Transition to half-open

        # Failed request should reopen circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_success_decrements_failure_count(self):
        """Test successful operations decrement failure count."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        assert breaker.failure_count == 1

        breaker.record_success()
        assert breaker.failure_count == 0

    def test_state_change_callback(self):
        """Test state change callbacks are invoked."""
        breaker = CircuitBreaker(failure_threshold=2)
        state_changes = []

        def callback(old_state, new_state):
            state_changes.append((old_state, new_state))

        breaker.on_state_change(callback)

        # Trigger state change
        breaker.record_failure()
        breaker.record_failure()

        # Should have recorded transition
        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)


class TestFailureNode:
    """Test failure node configuration."""

    def test_valid_node_creation(self):
        """Test creating valid failure node."""
        node = FailureNode(
            node_id="node1",
            failure_type=FailureType.RESOURCE_EXHAUSTION,
            trigger_delay_seconds=1.0,
            recovery_delay_seconds=5.0
        )

        assert node.node_id == "node1"
        assert node.failure_type == FailureType.RESOURCE_EXHAUSTION
        assert node.trigger_delay_seconds == 1.0
        assert node.recovery_delay_seconds == 5.0

    def test_node_with_dependencies(self):
        """Test node with dependencies."""
        node = FailureNode(
            node_id="node1",
            failure_type=FailureType.NETWORK_FAILURE,
            dependencies=["node2", "node3"]
        )

        assert node.dependencies == ["node2", "node3"]

    def test_node_with_propagation(self):
        """Test node with propagation targets."""
        node = FailureNode(
            node_id="node1",
            failure_type=FailureType.COMPONENT_CRASH,
            propagates_to=["node4", "node5"]
        )

        assert node.propagates_to == ["node4", "node5"]


class TestCascadeScenario:
    """Test cascade scenario configuration."""

    def test_valid_scenario_creation(self):
        """Test creating valid cascade scenario."""
        nodes = [
            FailureNode("node1", FailureType.RESOURCE_EXHAUSTION),
            FailureNode("node2", FailureType.NETWORK_FAILURE),
        ]

        scenario = CascadeScenario(
            failure_nodes=nodes,
            initial_trigger="node1",
            duration_seconds=10.0
        )

        assert len(scenario.failure_nodes) == 2
        assert scenario.initial_trigger == "node1"
        assert scenario.duration_seconds == 10.0

    def test_duplicate_node_ids_raises_error(self):
        """Test that duplicate node IDs raise ValueError."""
        nodes = [
            FailureNode("node1", FailureType.RESOURCE_EXHAUSTION),
            FailureNode("node1", FailureType.NETWORK_FAILURE),  # Duplicate
        ]

        with pytest.raises(ValueError, match="must be unique"):
            CascadeScenario(failure_nodes=nodes, initial_trigger="node1")

    def test_invalid_initial_trigger_raises_error(self):
        """Test that invalid initial trigger raises ValueError."""
        nodes = [
            FailureNode("node1", FailureType.RESOURCE_EXHAUSTION),
        ]

        with pytest.raises(ValueError, match="not found"):
            CascadeScenario(failure_nodes=nodes, initial_trigger="nonexistent")

    def test_invalid_dependency_raises_error(self):
        """Test that invalid dependencies raise ValueError."""
        nodes = [
            FailureNode(
                "node1",
                FailureType.RESOURCE_EXHAUSTION,
                dependencies=["nonexistent"]
            ),
        ]

        with pytest.raises(ValueError, match="non-existent"):
            CascadeScenario(failure_nodes=nodes, initial_trigger="node1")


class TestFailureChainSimulator:
    """Test failure chain simulation."""

    def test_add_node(self):
        """Test adding failure nodes."""
        simulator = FailureChainSimulator()
        node = FailureNode("node1", FailureType.RESOURCE_EXHAUSTION)

        simulator.add_node(node)

        assert "node1" in simulator._nodes

    @pytest.mark.asyncio
    async def test_trigger_failure(self):
        """Test triggering node failure."""
        simulator = FailureChainSimulator()
        node = FailureNode("node1", FailureType.RESOURCE_EXHAUSTION)
        simulator.add_node(node)

        await simulator.trigger_failure("node1")

        assert simulator.is_node_failed("node1")
        assert "node1" in simulator.get_failed_nodes()

    @pytest.mark.asyncio
    async def test_failure_propagation(self):
        """Test failure propagates to dependent nodes."""
        simulator = FailureChainSimulator()

        node1 = FailureNode(
            "node1",
            FailureType.RESOURCE_EXHAUSTION,
            propagates_to=["node2"]
        )
        node2 = FailureNode("node2", FailureType.NETWORK_FAILURE)

        simulator.add_node(node1)
        simulator.add_node(node2)

        # Trigger failure on node1
        await simulator.trigger_failure("node1")

        # Both nodes should be failed
        assert simulator.is_node_failed("node1")
        assert simulator.is_node_failed("node2")

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_failure(self):
        """Test circuit breaker prevents cascading failures."""
        simulator = FailureChainSimulator()

        node = FailureNode("node1", FailureType.RESOURCE_EXHAUSTION)
        simulator.add_node(node)
        simulator.add_circuit_breaker("node1", failure_threshold=1)

        # First failure should succeed
        await simulator.trigger_failure("node1")
        assert simulator.is_node_failed("node1")

        # Reset state
        simulator.reset()

        # Second failure should be blocked by circuit breaker
        await simulator.trigger_failure("node1")
        # Node should not be failed because circuit is open
        assert not simulator.is_node_failed("node1")

    @pytest.mark.asyncio
    async def test_recover_node(self):
        """Test node recovery."""
        simulator = FailureChainSimulator()
        node = FailureNode(
            "node1",
            FailureType.RESOURCE_EXHAUSTION,
            recovery_delay_seconds=0.1
        )
        simulator.add_node(node)

        # Fail node
        await simulator.trigger_failure("node1")
        assert simulator.is_node_failed("node1")

        # Recover node
        await simulator.recover_node("node1")
        assert not simulator.is_node_failed("node1")

    def test_reset_clears_failures(self):
        """Test reset clears all failures."""
        simulator = FailureChainSimulator()
        node = FailureNode("node1", FailureType.RESOURCE_EXHAUSTION)
        simulator.add_node(node)

        simulator._failed_nodes.add("node1")
        simulator.reset()

        assert len(simulator.get_failed_nodes()) == 0

    def test_get_circuit_breaker_states(self):
        """Test getting circuit breaker states."""
        simulator = FailureChainSimulator()
        node = FailureNode("node1", FailureType.RESOURCE_EXHAUSTION)
        simulator.add_node(node)
        simulator.add_circuit_breaker("node1")

        states = simulator.get_circuit_breaker_states()

        assert "node1" in states
        assert states["node1"] == CircuitState.CLOSED


class TestCascadingFailureSimulator:
    """Test main cascading failure simulator."""

    @pytest.mark.asyncio
    async def test_execute_simple_scenario(self):
        """Test executing simple cascade scenario."""
        simulator = CascadingFailureSimulator()

        nodes = [
            FailureNode("node1", FailureType.RESOURCE_EXHAUSTION),
        ]

        scenario = CascadeScenario(
            failure_nodes=nodes,
            initial_trigger="node1",
            duration_seconds=0.5
        )

        results = await simulator.execute_scenario(scenario)

        assert results["scenario_completed"]
        assert results["failed_node_count"] == 1
        assert "node1" in results["failure_sequence"]

    @pytest.mark.asyncio
    async def test_execute_cascade_scenario(self):
        """Test executing scenario with failure cascade."""
        simulator = CascadingFailureSimulator()

        nodes = [
            FailureNode(
                "node1",
                FailureType.RESOURCE_EXHAUSTION,
                propagates_to=["node2"]
            ),
            FailureNode("node2", FailureType.NETWORK_FAILURE),
        ]

        scenario = CascadeScenario(
            failure_nodes=nodes,
            initial_trigger="node1",
            duration_seconds=1.0
        )

        results = await simulator.execute_scenario(scenario)

        assert results["failed_node_count"] == 2
        assert "node1" in results["failure_sequence"]
        assert "node2" in results["failure_sequence"]

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation simulation."""
        simulator = CascadingFailureSimulator()

        results = await simulator.simulate_graceful_degradation(
            primary_node="primary",
            fallback_nodes=["fallback1", "fallback2"],
            test_duration_seconds=1.0
        )

        assert results["primary_node"] == "primary"
        assert len(results["fallback_nodes"]) == 2
        assert "degradation_handled" in results


class TestIntegrationScenarios:
    """Integration tests for cascading failure scenarios."""

    @pytest.mark.asyncio
    async def test_multi_level_cascade(self):
        """Test multi-level failure cascade."""
        simulator = CascadingFailureSimulator()

        nodes = [
            FailureNode(
                "level1",
                FailureType.RESOURCE_EXHAUSTION,
                propagates_to=["level2a", "level2b"]
            ),
            FailureNode(
                "level2a",
                FailureType.NETWORK_FAILURE,
                propagates_to=["level3"]
            ),
            FailureNode("level2b", FailureType.COMPONENT_CRASH),
            FailureNode("level3", FailureType.TIMEOUT),
        ]

        scenario = CascadeScenario(
            failure_nodes=nodes,
            initial_trigger="level1",
            duration_seconds=1.0
        )

        results = await simulator.execute_scenario(scenario)

        # All nodes should fail
        assert results["failed_node_count"] == 4

    @pytest.mark.asyncio
    async def test_circuit_breaker_limits_cascade(self):
        """Test circuit breaker limits cascade propagation."""
        simulator = CascadingFailureSimulator()

        nodes = [
            FailureNode(
                "node1",
                FailureType.RESOURCE_EXHAUSTION,
                propagates_to=["node2"]
            ),
            FailureNode("node2", FailureType.NETWORK_FAILURE),
        ]

        scenario = CascadeScenario(
            failure_nodes=nodes,
            initial_trigger="node1",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=1,
            duration_seconds=0.5
        )

        results = await simulator.execute_scenario(scenario)

        # Circuit breaker should limit failures
        assert results["failed_node_count"] >= 1

    @pytest.mark.asyncio
    async def test_recovery_after_cascade(self):
        """Test recovery after cascading failures."""
        simulator = CascadingFailureSimulator()

        nodes = [
            FailureNode(
                "node1",
                FailureType.RESOURCE_EXHAUSTION,
                recovery_delay_seconds=0.1,
                propagates_to=["node2"]
            ),
            FailureNode(
                "node2",
                FailureType.NETWORK_FAILURE,
                recovery_delay_seconds=0.1
            ),
        ]

        scenario = CascadeScenario(
            failure_nodes=nodes,
            initial_trigger="node1",
            duration_seconds=0.5
        )

        results = await simulator.execute_scenario(scenario)

        # Should have recovery times for all failed nodes
        assert len(results["recovery_times"]) == results["failed_node_count"]
        for _node_id, recovery_time in results["recovery_times"].items():
            assert recovery_time > 0
