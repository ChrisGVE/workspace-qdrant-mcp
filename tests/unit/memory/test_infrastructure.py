"""
Test infrastructure verification for memory rules testing.

Validates that all fixtures, utilities, and base classes work correctly.
"""

from datetime import datetime, timezone

import pytest
from common.core.memory import (
    AgentDefinition,
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryRule,
)

from tests.unit.memory.rule_test_utils import (
    ConflictSimulator,
    MemoryRuleGenerator,
    MemoryRuleValidator,
    assert_rule_has_field,
    assert_rule_replaces,
    assert_rules_have_same_scope,
)
from tests.unit.memory.test_rules_base import BaseMemoryRuleTest


@pytest.mark.memory_rules
class TestInfrastructure(BaseMemoryRuleTest):
    """Test that memory testing infrastructure is working correctly."""

    def test_base_class_setup(self):
        """Test that base class setup is working."""
        assert hasattr(self, "test_rules")
        assert hasattr(self, "test_agents")
        assert isinstance(self.test_rules, list)
        assert isinstance(self.test_agents, list)

    def test_create_test_rule(self):
        """Test creating a test rule using base class helper."""
        rule = self.create_test_rule(
            rule_id="test-1",
            name="Test Rule",
            rule_text="Always test",
        )

        assert rule.id == "test-1"
        assert rule.name == "Test Rule"
        assert rule.rule == "Always test"
        assert rule.category == MemoryCategory.BEHAVIOR
        assert rule.authority == AuthorityLevel.DEFAULT
        assert rule.scope == ["global"]
        assert rule in self.test_rules

    def test_create_test_agent(self):
        """Test creating a test agent using base class helper."""
        agent = self.create_test_agent(
            agent_id="agent-1",
            name="test-agent",
            description="Test agent",
        )

        assert agent.id == "agent-1"
        assert agent.name == "test-agent"
        assert agent.description == "Test agent"
        assert len(agent.capabilities) > 0
        assert agent in self.test_agents

    def test_assert_rule_valid(self):
        """Test rule validation assertion."""
        rule = self.create_test_rule()

        # Should not raise
        self.assert_rule_valid(rule)

    def test_assert_agent_valid(self):
        """Test agent validation assertion."""
        agent = self.create_test_agent()

        # Should not raise
        self.assert_agent_valid(agent)

    def test_assert_rules_equal(self):
        """Test rules equality assertion."""
        rule1 = self.create_test_rule(
            rule_id="same-id",
            name="Same Rule",
            rule_text="Same text",
        )

        rule2 = MemoryRule(
            id="same-id",
            category=MemoryCategory.BEHAVIOR,
            name="Same Rule",
            rule="Same text",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        # Should not raise
        self.assert_rules_equal(rule1, rule2)

    def test_create_mock_qdrant_client(self):
        """Test creating mock Qdrant client."""
        client = self.create_mock_qdrant_client()

        assert client is not None
        assert hasattr(client, "create_collection")
        assert hasattr(client, "upsert")
        assert hasattr(client, "search")
        assert self.mock_client is client


@pytest.mark.memory_rules
class TestFixtures:
    """Test that pytest fixtures are working correctly."""

    def test_mock_memory_client(self, mock_memory_client):
        """Test mock memory client fixture."""
        assert mock_memory_client is not None
        assert hasattr(mock_memory_client, "upsert")
        assert hasattr(mock_memory_client, "search")
        assert hasattr(mock_memory_client, "upserted_points")
        assert hasattr(mock_memory_client, "search_queries")

    def test_sample_memory_rules(self, sample_memory_rules):
        """Test sample memory rules fixture."""
        assert len(sample_memory_rules) > 0

        for rule in sample_memory_rules:
            assert isinstance(rule, MemoryRule)
            assert rule.id
            assert rule.category
            assert rule.name
            assert rule.rule
            assert rule.authority
            assert rule.scope

    def test_sample_agent_definitions(self, sample_agent_definitions):
        """Test sample agent definitions fixture."""
        assert len(sample_agent_definitions) > 0

        for agent in sample_agent_definitions:
            assert isinstance(agent, AgentDefinition)
            assert agent.id
            assert agent.name
            assert agent.description
            assert len(agent.capabilities) > 0

    def test_mock_bm25_encoder(self, mock_bm25_encoder):
        """Test mock BM25 encoder fixture."""
        assert mock_bm25_encoder is not None
        assert hasattr(mock_bm25_encoder, "encode_documents")
        assert hasattr(mock_bm25_encoder, "encode_queries")

        # Test encoding
        result = mock_bm25_encoder.encode_documents(["test"])
        assert len(result) > 0
        assert "indices" in result[0]
        assert "values" in result[0]

    def test_memory_collection_names(self, memory_collection_names):
        """Test memory collection names fixture."""
        assert isinstance(memory_collection_names, dict)
        assert "memory" in memory_collection_names
        assert "agent_memory" in memory_collection_names
        assert memory_collection_names["memory"] == "memory"

    def test_rule_conflict_pairs(self, rule_conflict_pairs):
        """Test rule conflict pairs fixture."""
        assert len(rule_conflict_pairs) > 0

        for rule1, rule2 in rule_conflict_pairs:
            assert isinstance(rule1, MemoryRule)
            assert isinstance(rule2, MemoryRule)
            assert rule1.id != rule2.id

    def test_sample_memory_conflicts(self, sample_memory_conflicts):
        """Test sample memory conflicts fixture."""
        assert len(sample_memory_conflicts) > 0

        for conflict in sample_memory_conflicts:
            assert isinstance(conflict, MemoryConflict)
            assert conflict.conflict_type
            assert conflict.rule1
            assert conflict.rule2
            assert 0.0 <= conflict.confidence <= 1.0

    def test_rule_validator(self, rule_validator):
        """Test rule validator fixture."""
        assert isinstance(rule_validator, dict)
        assert "structure" in rule_validator
        assert "category" in rule_validator
        assert "authority" in rule_validator
        assert "scope" in rule_validator
        assert "timestamps" in rule_validator

        # Test validator functions
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        assert rule_validator["structure"](rule)
        assert rule_validator["category"](rule)
        assert rule_validator["authority"](rule)
        assert rule_validator["scope"](rule)
        assert rule_validator["timestamps"](rule)

    def test_mock_embedding_model(self, mock_embedding_model):
        """Test mock embedding model fixture."""
        assert mock_embedding_model is not None
        assert hasattr(mock_embedding_model, "embed")

        # Test embedding
        result = mock_embedding_model.embed(["test"])
        assert len(result) > 0
        assert len(result[0]) == 384  # Standard embedding dimension


@pytest.mark.memory_rules
class TestUtilities:
    """Test utility classes and functions."""

    def test_memory_rule_generator_create_rule(self):
        """Test MemoryRuleGenerator.generate_rule()."""
        generator = MemoryRuleGenerator(seed=42)

        rule = generator.generate_rule()

        assert isinstance(rule, MemoryRule)
        assert rule.id
        assert rule.category
        assert rule.name
        assert rule.rule
        assert rule.authority
        assert rule.scope

    def test_memory_rule_generator_create_rules(self):
        """Test MemoryRuleGenerator.generate_rules()."""
        generator = MemoryRuleGenerator(seed=42)

        rules = generator.generate_rules(5)

        assert len(rules) == 5
        for rule in rules:
            assert isinstance(rule, MemoryRule)

        # Check uniqueness
        rule_ids = [r.id for r in rules]
        assert len(rule_ids) == len(set(rule_ids))

    def test_memory_rule_generator_behavior_rule(self):
        """Test MemoryRuleGenerator.generate_behavior_rule()."""
        generator = MemoryRuleGenerator()

        rule = generator.generate_behavior_rule()

        assert rule.category == MemoryCategory.BEHAVIOR

    def test_memory_rule_generator_preference_rule(self):
        """Test MemoryRuleGenerator.generate_preference_rule()."""
        generator = MemoryRuleGenerator()

        rule = generator.generate_preference_rule(scope=["python"])

        assert rule.category == MemoryCategory.PREFERENCE
        assert "python" in rule.scope

    def test_memory_rule_generator_agent_definition(self):
        """Test MemoryRuleGenerator.generate_agent_definition()."""
        generator = MemoryRuleGenerator()

        agent = generator.generate_agent_definition(name="test-agent")

        assert isinstance(agent, AgentDefinition)
        assert agent.name == "test-agent"
        assert len(agent.capabilities) > 0

    def test_memory_rule_validator_structure(self):
        """Test MemoryRuleValidator.validate_structure()."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        is_valid, errors = MemoryRuleValidator.validate_structure(rule)

        assert is_valid
        assert len(errors) == 0

    def test_memory_rule_validator_structure_invalid(self):
        """Test MemoryRuleValidator.validate_structure() with invalid rule."""
        # Create rule with missing fields
        rule = MemoryRule(
            id="",  # Empty ID
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="",  # Empty rule text
            authority=AuthorityLevel.DEFAULT,
            scope=[],  # Empty scope
        )

        is_valid, errors = MemoryRuleValidator.validate_structure(rule)

        assert not is_valid
        assert len(errors) > 0

    def test_memory_rule_validator_semantics(self):
        """Test MemoryRuleValidator.validate_semantics()."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="This is a reasonable length rule text for testing",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        is_valid, warnings = MemoryRuleValidator.validate_semantics(rule)

        assert is_valid
        assert len(warnings) == 0

    def test_conflict_simulator_contradictory_pair(self):
        """Test ConflictSimulator.generate_contradictory_pair()."""
        simulator = ConflictSimulator()

        rule1, rule2 = simulator.generate_contradictory_pair()

        assert isinstance(rule1, MemoryRule)
        assert isinstance(rule2, MemoryRule)
        assert rule1.id != rule2.id
        assert rule1.scope == rule2.scope  # Same scope for conflict

    def test_conflict_simulator_authority_conflict(self):
        """Test ConflictSimulator.generate_authority_conflict()."""
        simulator = ConflictSimulator()

        rule1, rule2 = simulator.generate_authority_conflict()

        assert isinstance(rule1, MemoryRule)
        assert isinstance(rule2, MemoryRule)
        assert rule1.authority != rule2.authority
        # One should be ABSOLUTE
        assert rule1.authority == AuthorityLevel.ABSOLUTE or rule2.authority == AuthorityLevel.ABSOLUTE

    def test_assert_rule_has_field(self):
        """Test assert_rule_has_field() helper."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        # Should not raise
        assert_rule_has_field(rule, "id")
        assert_rule_has_field(rule, "category")
        assert_rule_has_field(rule, "name")

    def test_assert_rules_have_same_scope(self):
        """Test assert_rules_have_same_scope() helper."""
        rule1 = MemoryRule(
            id="test1",
            category=MemoryCategory.BEHAVIOR,
            name="Test1",
            rule="Test rule 1",
            authority=AuthorityLevel.DEFAULT,
            scope=["global", "python"],
        )

        rule2 = MemoryRule(
            id="test2",
            category=MemoryCategory.BEHAVIOR,
            name="Test2",
            rule="Test rule 2",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "global"],  # Different order but same set
        )

        # Should not raise
        assert_rules_have_same_scope(rule1, rule2)

    def test_assert_rule_replaces(self):
        """Test assert_rule_replaces() helper."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            replaces=["old-rule-1", "old-rule-2"],
        )

        # Should not raise
        assert_rule_replaces(rule, "old-rule-1")
        assert_rule_replaces(rule, "old-rule-2")


@pytest.mark.memory_rules
def test_markers_are_applied():
    """Test that pytest markers are being applied correctly."""
    # This test itself has the memory_rules marker
    # If this passes, markers are working
    assert True
