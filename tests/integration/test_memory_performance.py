"""
Integration tests for memory system performance and scalability.

Tests large rule collections, memory usage validation, data structure performance,
and end-to-end workflows with realistic workloads.

These tests focus on in-memory operations and don't require a live Qdrant instance.
"""

import asyncio
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from common.core.memory import (
    AgentDefinition,
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryRule,
    MemoryStats,
)


class TestRuleCollectionScaling:
    """Test memory rule collection scaling characteristics."""

    def create_test_rules(self, count: int) -> list[MemoryRule]:
        """Create test rules for performance testing."""
        rules = []
        for i in range(count):
            rule = MemoryRule(
                id=f"perf-rule-{i}",
                category=MemoryCategory.BEHAVIOR if i % 2 == 0 else MemoryCategory.PREFERENCE,
                name=f"Performance Test Rule {i}",
                rule=f"This is a test rule number {i} with some content to make it realistic",
                authority=AuthorityLevel.ABSOLUTE if i % 5 == 0 else AuthorityLevel.DEFAULT,
                scope=["python", "testing"] if i % 3 == 0 else [],
                source="test_generator",
                metadata={"test_id": i, "batch": "performance"},
            )
            rules.append(rule)
        return rules

    def test_small_collection_creation_performance(self):
        """Test performance of creating small rule collection (50 rules)."""
        start = time.perf_counter()
        rules = self.create_test_rules(50)
        elapsed = time.perf_counter() - start

        assert len(rules) == 50
        assert elapsed < 0.5  # Should be nearly instant

    def test_medium_collection_creation_performance(self):
        """Test performance of creating medium rule collection (200 rules)."""
        start = time.perf_counter()
        rules = self.create_test_rules(200)
        elapsed = time.perf_counter() - start

        assert len(rules) == 200
        assert elapsed < 1.0

    @pytest.mark.slow
    def test_large_collection_creation_performance(self):
        """Test performance of creating large rule collection (1000 rules)."""
        start = time.perf_counter()
        rules = self.create_test_rules(1000)
        elapsed = time.perf_counter() - start

        assert len(rules) == 1000
        assert elapsed < 3.0

    def test_rule_serialization_batch_performance(self):
        """Test batch serialization performance."""
        rules = self.create_test_rules(500)

        start = time.perf_counter()
        serialized = [asdict(rule) for rule in rules]
        elapsed = time.perf_counter() - start

        assert len(serialized) == 500
        assert elapsed < 1.0  # Should be very fast

    def test_rule_filtering_performance(self):
        """Test filtering operations performance."""
        rules = self.create_test_rules(1000)

        # Filter by authority
        start = time.perf_counter()
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        elapsed = time.perf_counter() - start

        assert len(absolute_rules) == 200  # Every 5th rule
        assert elapsed < 0.1

        # Filter by category
        start = time.perf_counter()
        behavior_rules = [r for r in rules if r.category == MemoryCategory.BEHAVIOR]
        elapsed = time.perf_counter() - start

        assert len(behavior_rules) == 500  # Every other rule
        assert elapsed < 0.1

        # Filter by scope
        start = time.perf_counter()
        scoped_rules = [r for r in rules if "python" in r.scope]
        elapsed = time.perf_counter() - start

        assert len(scoped_rules) > 0
        assert elapsed < 0.1


class TestMemoryStatsCalculation:
    """Test memory statistics calculation performance and accuracy."""

    def create_diverse_ruleset(self, count: int) -> list[MemoryRule]:
        """Create diverse ruleset for statistics testing."""
        rules = []
        for i in range(count):
            rule = MemoryRule(
                id=f"stats-rule-{i}",
                category=MemoryCategory.BEHAVIOR if i % 3 == 0 else (
                    MemoryCategory.PREFERENCE if i % 3 == 1 else MemoryCategory.AGENT
                ),
                name=f"Stats Rule {i}",
                rule=f"Rule text {i}" * (i % 10 + 1),  # Varying lengths
                authority=AuthorityLevel.ABSOLUTE if i % 4 == 0 else AuthorityLevel.DEFAULT,
                scope=[] if i % 2 == 0 else [f"scope-{i % 5}"],
            )
            rules.append(rule)
        return rules

    def calculate_stats(self, rules: list[MemoryRule]) -> MemoryStats:
        """Calculate memory statistics from rule list."""
        rules_by_category = dict.fromkeys(MemoryCategory, 0)
        rules_by_authority = dict.fromkeys(AuthorityLevel, 0)

        for rule in rules:
            rules_by_category[rule.category] += 1
            rules_by_authority[rule.authority] += 1

        # Rough token estimation (4 chars per token average)
        estimated_tokens = sum(len(rule.rule) for rule in rules) // 4

        return MemoryStats(
            total_rules=len(rules),
            rules_by_category=rules_by_category,
            rules_by_authority=rules_by_authority,
            estimated_tokens=estimated_tokens,
            last_optimization=datetime.now(timezone.utc),
        )

    def test_stats_calculation_performance(self):
        """Test statistics calculation performance."""
        rules = self.create_diverse_ruleset(500)

        start = time.perf_counter()
        stats = self.calculate_stats(rules)
        elapsed = time.perf_counter() - start

        assert stats.total_rules == 500
        assert elapsed < 0.5

    def test_stats_accuracy(self):
        """Test statistics accuracy with known distribution."""
        rules = self.create_diverse_ruleset(300)

        stats = self.calculate_stats(rules)

        # Verify total
        assert stats.total_rules == 300

        # Verify category distribution
        assert sum(stats.rules_by_category.values()) == 300

        # Verify authority distribution
        assert sum(stats.rules_by_authority.values()) == 300

        # Verify token estimation is reasonable
        assert stats.estimated_tokens > 0


class TestDataStructurePerformance:
    """Test performance of data structures and operations."""

    def test_rule_lookup_by_id_performance(self):
        """Test dictionary-based lookup performance."""
        # Create large rule set
        rules = [
            MemoryRule(
                id=f"lookup-{i}",
                category=MemoryCategory.BEHAVIOR,
                name=f"Rule {i}",
                rule=f"Content {i}",
                authority=AuthorityLevel.DEFAULT,
                scope=[],
            )
            for i in range(1000)
        ]

        # Build lookup dict
        start = time.perf_counter()
        lookup_dict = {rule.id: rule for rule in rules}
        build_elapsed = time.perf_counter() - start

        assert len(lookup_dict) == 1000
        assert build_elapsed < 0.1

        # Test lookup performance
        start = time.perf_counter()
        for i in range(100):
            rule = lookup_dict.get(f"lookup-{i * 10}")
            assert rule is not None
        lookup_elapsed = time.perf_counter() - start

        assert lookup_elapsed < 0.01  # Should be very fast

    def test_rule_modification_performance(self):
        """Test rule update operations."""
        rule = MemoryRule(
            id="modify-test-1",
            category=MemoryCategory.BEHAVIOR,
            name="Modify Test",
            rule="Original text",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        # Test multiple modifications
        start = time.perf_counter()
        for i in range(100):
            rule.rule = f"Modified text version {i}"
            rule.updated_at = datetime.now(timezone.utc)
        elapsed = time.perf_counter() - start

        assert "version 99" in rule.rule
        assert elapsed < 0.1


class TestMemorySchemaCompliance:
    """Test compliance with memory schema and data integrity."""

    def test_rule_field_completeness(self):
        """Test that all required fields are present."""
        rule = MemoryRule(
            id="field-test-1",
            category=MemoryCategory.BEHAVIOR,
            name="Field Test",
            rule="Test all fields",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        # Verify all required fields
        assert rule.id is not None
        assert rule.category is not None
        assert rule.name is not None
        assert rule.rule is not None
        assert rule.authority is not None
        assert rule.scope is not None
        assert rule.created_at is not None
        assert rule.updated_at is not None

    def test_rule_validation_logic(self):
        """Test rule validation logic."""
        # Valid rule
        valid_rule = MemoryRule(
            id="valid-1",
            category=MemoryCategory.BEHAVIOR,
            name="Valid Rule",
            rule="This is valid",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        assert isinstance(valid_rule.category, MemoryCategory)
        assert isinstance(valid_rule.authority, AuthorityLevel)
        assert isinstance(valid_rule.scope, list)
        assert isinstance(valid_rule.created_at, datetime)

    def test_timestamp_initialization(self):
        """Test automatic timestamp initialization."""
        before = datetime.now(timezone.utc)
        rule = MemoryRule(
            id="timestamp-test-1",
            category=MemoryCategory.BEHAVIOR,
            name="Timestamp Test",
            rule="Test timestamps",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )
        after = datetime.now(timezone.utc)

        assert before <= rule.created_at <= after
        assert rule.updated_at == rule.created_at


class TestEndToEndWorkflows:
    """Test complete end-to-end memory workflows."""

    def test_rule_lifecycle_workflow(self):
        """Test complete rule lifecycle from creation to deletion."""
        # Create
        rule = MemoryRule(
            id="lifecycle-1",
            category=MemoryCategory.BEHAVIOR,
            name="Lifecycle Test",
            rule="Version 1",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
            metadata={"version": 1},
        )

        assert rule.metadata["version"] == 1

        # Update multiple times
        for version in range(2, 6):
            rule.rule = f"Version {version}"
            rule.metadata = {"version": version}
            rule.updated_at = datetime.now(timezone.utc)

        assert rule.metadata["version"] == 5
        assert "Version 5" in rule.rule
        assert rule.updated_at > rule.created_at

    def test_batch_operations_workflow(self):
        """Test batch rule operations."""
        # Create batch of rules
        rules = [
            MemoryRule(
                id=f"batch-{i}",
                category=MemoryCategory.BEHAVIOR,
                name=f"Batch Rule {i}",
                rule=f"Batch content {i}",
                authority=AuthorityLevel.DEFAULT,
                scope=[],
            )
            for i in range(50)
        ]

        # Serialize batch
        start = time.perf_counter()
        serialized = [asdict(rule) for rule in rules]
        serialize_elapsed = time.perf_counter() - start

        assert len(serialized) == 50
        assert serialize_elapsed < 0.5

        # Filter batch
        start = time.perf_counter()
        filtered = [r for r in rules if int(r.id.split("-")[1]) % 2 == 0]
        filter_elapsed = time.perf_counter() - start

        assert len(filtered) == 25
        assert filter_elapsed < 0.1

    def test_authority_precedence_workflow(self):
        """Test authority precedence in rule application."""
        # Create conflicting rules
        absolute_rule = MemoryRule(
            id="precedence-abs-1",
            category=MemoryCategory.BEHAVIOR,
            name="Absolute Rule",
            rule="Must do X",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
        )

        default_rule = MemoryRule(
            id="precedence-def-1",
            category=MemoryCategory.BEHAVIOR,
            name="Default Rule",
            rule="Prefer Y",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
        )

        rules = [absolute_rule, default_rule]

        # Sort by authority (absolute first)
        sorted_rules = sorted(
            rules, key=lambda r: 0 if r.authority == AuthorityLevel.ABSOLUTE else 1
        )

        assert sorted_rules[0].authority == AuthorityLevel.ABSOLUTE
        assert sorted_rules[1].authority == AuthorityLevel.DEFAULT

    def test_scope_matching_workflow(self):
        """Test scope matching logic."""
        # Create rules with various scopes
        rules = [
            MemoryRule(
                id="scope-global",
                category=MemoryCategory.BEHAVIOR,
                name="Global",
                rule="Global rule",
                authority=AuthorityLevel.DEFAULT,
                scope=[],  # Empty = global
            ),
            MemoryRule(
                id="scope-python",
                category=MemoryCategory.BEHAVIOR,
                name="Python",
                rule="Python rule",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
            ),
            MemoryRule(
                id="scope-multi",
                category=MemoryCategory.BEHAVIOR,
                name="Multi",
                rule="Multi-scope rule",
                authority=AuthorityLevel.DEFAULT,
                scope=["python", "testing"],
            ),
        ]

        # Match active scope: python
        active_scopes = ["python"]

        def matches_scope(rule: MemoryRule, active: list[str]) -> bool:
            """Check if rule scope matches active scopes."""
            if not rule.scope:  # Empty scope = global, always matches
                return True
            return any(s in active for s in rule.scope)

        matching = [r for r in rules if matches_scope(r, active_scopes)]

        assert len(matching) == 3  # All should match


class TestAgentDefinitionIntegration:
    """Test AgentDefinition integration with memory system."""

    def test_agent_serialization(self):
        """Test agent definition serialization."""
        agent = AgentDefinition(
            id="agent-1",
            name="test-agent",
            description="Test agent",
            capabilities=["testing", "validation"],
            deploy_cost="medium",
            metadata={"version": "1.0"},
        )

        # Serialize
        agent_dict = asdict(agent)

        assert agent_dict["id"] == "agent-1"
        assert agent_dict["name"] == "test-agent"
        assert agent_dict["capabilities"] == ["testing", "validation"]
        assert agent_dict["metadata"]["version"] == "1.0"

    def test_agent_batch_operations(self):
        """Test batch operations with agent definitions."""
        agents = [
            AgentDefinition(
                id=f"agent-{i}",
                name=f"agent-{i}",
                description=f"Agent {i}",
                capabilities=[f"skill-{i}"],
                deploy_cost="medium" if i % 2 == 0 else "high",
            )
            for i in range(20)
        ]

        # Filter by cost
        medium_cost = [a for a in agents if a.deploy_cost == "medium"]
        high_cost = [a for a in agents if a.deploy_cost == "high"]

        assert len(medium_cost) == 10
        assert len(high_cost) == 10

    def test_agent_last_used_tracking(self):
        """Test agent last_used timestamp tracking."""
        agent = AgentDefinition(
            id="tracking-agent",
            name="tracking-agent",
            description="Track usage",
            capabilities=["tracking"],
        )

        assert agent.last_used is None

        # Simulate usage
        agent.last_used = datetime.now(timezone.utc)

        assert agent.last_used is not None
        assert isinstance(agent.last_used, datetime)
