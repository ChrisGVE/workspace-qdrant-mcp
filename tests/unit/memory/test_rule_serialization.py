"""
Test suite for memory rule serialization and deserialization.

Tests rule persistence, JSON serialization formats, data integrity validation,
schema compliance, and round-trip consistency for various rule types.
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from common.core.memory import (
    AgentDefinition,
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)

from .test_rules_base import BaseMemoryRuleTest


class TestRuleSerialization(BaseMemoryRuleTest):
    """Test JSON serialization of memory rules."""

    @pytest.mark.asyncio
    async def test_basic_rule_serialization(self):
        """Test serialization of a basic rule to JSON."""
        rule = self.create_test_rule(
            rule_id="serialize-test-1",
            name="Basic Serialization Test",
            rule_text="Always write tests",
        )

        # Serialize to dict
        rule_dict = asdict(rule)

        # Verify all required fields are present
        assert "id" in rule_dict
        assert "category" in rule_dict
        assert "name" in rule_dict
        assert "rule" in rule_dict
        assert "authority" in rule_dict
        assert "scope" in rule_dict
        assert "created_at" in rule_dict
        assert "updated_at" in rule_dict

        # Verify field values
        assert rule_dict["id"] == "serialize-test-1"
        assert rule_dict["name"] == "Basic Serialization Test"
        assert rule_dict["rule"] == "Always write tests"

    @pytest.mark.asyncio
    async def test_enum_serialization(self):
        """Test that enums are properly serialized."""
        rule = self.create_test_rule(
            rule_id="serialize-test-2",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        rule_dict = asdict(rule)

        # Enums should be serialized to their values
        assert isinstance(rule_dict["category"], MemoryCategory)
        assert isinstance(rule_dict["authority"], AuthorityLevel)
        assert rule_dict["category"] == MemoryCategory.BEHAVIOR
        assert rule_dict["authority"] == AuthorityLevel.ABSOLUTE

    @pytest.mark.asyncio
    async def test_datetime_serialization(self):
        """Test that datetime fields are properly serialized."""
        now = datetime.now(timezone.utc)
        rule = self.create_test_rule(
            rule_id="serialize-test-3",
            metadata={"created_at": now}
        )

        rule_dict = asdict(rule)

        # Verify datetime objects are preserved
        assert isinstance(rule_dict["created_at"], datetime)
        assert isinstance(rule_dict["updated_at"], datetime)
        assert rule_dict["created_at"].tzinfo is not None
        assert rule_dict["updated_at"].tzinfo is not None

    @pytest.mark.asyncio
    async def test_metadata_serialization(self):
        """Test serialization of rule metadata."""
        rule = self.create_test_rule(
            rule_id="serialize-test-4",
            metadata={
                "version": 1,
                "tags": ["important", "testing"],
                "custom_field": "custom_value",
                "nested": {"key": "value"}
            }
        )

        rule_dict = asdict(rule)

        # Verify metadata is preserved
        assert "metadata" in rule_dict
        assert rule_dict["metadata"]["version"] == 1
        assert rule_dict["metadata"]["tags"] == ["important", "testing"]
        assert rule_dict["metadata"]["custom_field"] == "custom_value"
        assert rule_dict["metadata"]["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_optional_fields_serialization(self):
        """Test serialization of optional fields."""
        # Rule with all optional fields populated
        rule = self.create_test_rule(
            rule_id="serialize-test-5",
            conditions={"when": "editing", "language": "python"},
            replaces=["old-rule-1", "old-rule-2"],
            metadata={"note": "test"}
        )

        rule_dict = asdict(rule)

        assert rule_dict["conditions"] == {"when": "editing", "language": "python"}
        assert rule_dict["replaces"] == ["old-rule-1", "old-rule-2"]
        assert rule_dict["metadata"]["note"] == "test"

    @pytest.mark.asyncio
    async def test_none_optional_fields_serialization(self):
        """Test serialization when optional fields are None."""
        rule = self.create_test_rule(
            rule_id="serialize-test-6",
        )

        # Explicitly set optional fields to None
        rule.conditions = None
        rule.replaces = None
        rule.metadata = None

        rule_dict = asdict(rule)

        # None values should be preserved
        assert rule_dict["conditions"] is None
        assert rule_dict["replaces"] is None
        assert rule_dict["metadata"] is None

    @pytest.mark.asyncio
    async def test_scope_list_serialization(self):
        """Test serialization of scope list."""
        rule = self.create_test_rule(
            rule_id="serialize-test-7",
            scope=["project1", "project2", "module:auth"]
        )

        rule_dict = asdict(rule)

        assert isinstance(rule_dict["scope"], list)
        assert len(rule_dict["scope"]) == 3
        assert "project1" in rule_dict["scope"]
        assert "project2" in rule_dict["scope"]
        assert "module:auth" in rule_dict["scope"]

    @pytest.mark.asyncio
    async def test_empty_scope_serialization(self):
        """Test serialization of empty scope (global scope)."""
        rule = self.create_test_rule(
            rule_id="serialize-test-8",
            scope=[]
        )

        rule_dict = asdict(rule)

        # Empty scope should serialize as empty list
        assert isinstance(rule_dict["scope"], list)
        assert len(rule_dict["scope"]) == 0


class TestRuleDeserialization(BaseMemoryRuleTest):
    """Test JSON deserialization to memory rules."""

    @pytest.mark.asyncio
    async def test_basic_rule_deserialization(self):
        """Test deserialization of a basic rule from dict."""
        rule_dict = {
            "id": "deserialize-test-1",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Deserialization Test",
            "rule": "Always validate inputs",
            "authority": AuthorityLevel.DEFAULT,
            "scope": ["global"],
            "source": "user_explicit",
            "conditions": None,
            "replaces": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "metadata": None,
        }

        # Deserialize to MemoryRule
        rule = MemoryRule(**rule_dict)

        # Verify all fields are correctly restored
        assert rule.id == "deserialize-test-1"
        assert rule.category == MemoryCategory.BEHAVIOR
        assert rule.name == "Deserialization Test"
        assert rule.rule == "Always validate inputs"
        assert rule.authority == AuthorityLevel.DEFAULT
        assert rule.scope == ["global"]

    @pytest.mark.asyncio
    async def test_enum_deserialization(self):
        """Test that enum values are properly deserialized."""
        rule_dict = {
            "id": "deserialize-test-2",
            "category": MemoryCategory.PREFERENCE,
            "name": "Enum Test",
            "rule": "Use dark mode",
            "authority": AuthorityLevel.ABSOLUTE,
            "scope": [],
        }

        rule = MemoryRule(**rule_dict)

        # Verify enums are correctly restored
        assert isinstance(rule.category, MemoryCategory)
        assert isinstance(rule.authority, AuthorityLevel)
        assert rule.category == MemoryCategory.PREFERENCE
        assert rule.authority == AuthorityLevel.ABSOLUTE

    @pytest.mark.asyncio
    async def test_metadata_deserialization(self):
        """Test deserialization of rule metadata."""
        rule_dict = {
            "id": "deserialize-test-3",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Metadata Test",
            "rule": "Follow conventions",
            "authority": AuthorityLevel.DEFAULT,
            "scope": ["project1"],
            "metadata": {
                "version": 2,
                "history": ["v1 created", "v2 updated"],
                "tags": ["important"],
            }
        }

        rule = MemoryRule(**rule_dict)

        # Verify metadata is correctly restored
        assert rule.metadata is not None
        assert rule.metadata["version"] == 2
        assert rule.metadata["history"] == ["v1 created", "v2 updated"]
        assert rule.metadata["tags"] == ["important"]

    @pytest.mark.asyncio
    async def test_optional_fields_deserialization(self):
        """Test deserialization of optional fields."""
        rule_dict = {
            "id": "deserialize-test-4",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Optional Fields Test",
            "rule": "Context-aware rule",
            "authority": AuthorityLevel.DEFAULT,
            "scope": ["module:auth"],
            "conditions": {"when": "testing", "env": "dev"},
            "replaces": ["old-rule-1"],
        }

        rule = MemoryRule(**rule_dict)

        # Verify optional fields are correctly restored
        assert rule.conditions == {"when": "testing", "env": "dev"}
        assert rule.replaces == ["old-rule-1"]

    @pytest.mark.asyncio
    async def test_timestamp_auto_initialization(self):
        """Test that timestamps are auto-initialized if not provided."""
        rule_dict = {
            "id": "deserialize-test-5",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Timestamp Test",
            "rule": "Test timestamps",
            "authority": AuthorityLevel.DEFAULT,
            "scope": [],
        }

        before = datetime.now(timezone.utc)
        rule = MemoryRule(**rule_dict)
        after = datetime.now(timezone.utc)

        # Timestamps should be auto-initialized by __post_init__
        assert rule.created_at is not None
        assert rule.updated_at is not None
        assert before <= rule.created_at <= after
        assert rule.updated_at == rule.created_at


class TestRoundTripConsistency(BaseMemoryRuleTest):
    """Test round-trip serialization/deserialization consistency."""

    @pytest.mark.asyncio
    async def test_basic_round_trip(self):
        """Test that a basic rule survives serialization round-trip."""
        original = self.create_test_rule(
            rule_id="roundtrip-test-1",
            name="Round Trip Test",
            rule_text="Test round-trip consistency",
        )

        # Serialize to dict
        rule_dict = asdict(original)

        # Deserialize back to MemoryRule
        restored = MemoryRule(**rule_dict)

        # Verify equality
        self.assert_rules_equal(original, restored, ignore_timestamps=False)

    @pytest.mark.asyncio
    async def test_complex_rule_round_trip(self):
        """Test round-trip with all fields populated."""
        original = self.create_test_rule(
            rule_id="roundtrip-test-2",
            category=MemoryCategory.BEHAVIOR,
            name="Complex Round Trip",
            rule_text="Complex rule with all fields",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["project1", "module:auth"],
            conditions={"when": "editing", "lang": "python"},
            replaces=["old-1", "old-2"],
            metadata={
                "version": 3,
                "tags": ["critical", "security"],
                "history": ["created", "updated", "updated"],
            }
        )

        # Round-trip
        rule_dict = asdict(original)
        restored = MemoryRule(**rule_dict)

        # Verify all fields are preserved
        self.assert_rules_equal(original, restored, ignore_timestamps=False)
        assert restored.conditions == original.conditions
        assert restored.replaces == original.replaces
        assert restored.metadata == original.metadata

    @pytest.mark.asyncio
    async def test_round_trip_with_none_fields(self):
        """Test round-trip when optional fields are None."""
        original = self.create_test_rule(
            rule_id="roundtrip-test-3",
            name="None Fields Test",
        )
        original.conditions = None
        original.replaces = None
        original.metadata = None

        # Round-trip
        rule_dict = asdict(original)
        restored = MemoryRule(**rule_dict)

        # Verify None values are preserved
        assert restored.conditions is None
        assert restored.replaces is None
        assert restored.metadata is None

    @pytest.mark.asyncio
    async def test_multiple_rules_round_trip(self):
        """Test round-trip consistency for multiple rules."""
        originals = [
            self.create_test_rule(rule_id=f"roundtrip-batch-{i}", name=f"Rule {i}")
            for i in range(5)
        ]

        # Serialize all
        rule_dicts = [asdict(rule) for rule in originals]

        # Deserialize all
        restored = [MemoryRule(**rule_dict) for rule_dict in rule_dicts]

        # Verify all rules are preserved
        for original, restored_rule in zip(originals, restored, strict=False):
            self.assert_rules_equal(original, restored_rule, ignore_timestamps=False)


class TestDataIntegrityValidation(BaseMemoryRuleTest):
    """Test data integrity validation during serialization."""

    @pytest.mark.asyncio
    async def test_required_fields_validation(self):
        """Test that required fields are validated."""
        rule = self.create_test_rule(
            rule_id="integrity-test-1",
            name="Required Fields Test",
        )

        # Verify all required fields are present
        self.assert_rule_valid(rule)

    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        """Test that missing required fields raise errors."""
        # Missing 'rule' field should raise TypeError
        with pytest.raises(TypeError):
            MemoryRule(
                id="integrity-test-2",
                category=MemoryCategory.BEHAVIOR,
                name="Missing Rule Field",
                # 'rule' field missing
                authority=AuthorityLevel.DEFAULT,
                scope=[],
            )

    @pytest.mark.asyncio
    async def test_invalid_enum_value(self):
        """Test that invalid enum values can be detected."""
        # Dataclass accepts string values, but they can be detected via validation
        rule = MemoryRule(
            id="integrity-test-3",
            category="invalid_category",  # Invalid enum value accepted by dataclass
            name="Invalid Enum Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        # Validation should detect the invalid enum
        with pytest.raises(AssertionError):
            self.assert_rule_valid(rule)

    @pytest.mark.asyncio
    async def test_scope_type_validation(self):
        """Test that scope must be a list."""
        rule = self.create_test_rule(
            rule_id="integrity-test-4",
            scope=["valid", "scope"]
        )

        rule_dict = asdict(rule)
        assert isinstance(rule_dict["scope"], list)

    @pytest.mark.asyncio
    async def test_metadata_structure_preservation(self):
        """Test that complex metadata structures are preserved."""
        rule = self.create_test_rule(
            rule_id="integrity-test-5",
            metadata={
                "nested": {
                    "level1": {
                        "level2": {
                            "value": "deep"
                        }
                    }
                },
                "list": [1, 2, {"key": "value"}],
                "mixed": [{"a": 1}, {"b": 2}],
            }
        )

        rule_dict = asdict(rule)
        restored = MemoryRule(**rule_dict)

        # Verify deep equality
        assert restored.metadata["nested"]["level1"]["level2"]["value"] == "deep"
        assert restored.metadata["list"] == [1, 2, {"key": "value"}]
        assert restored.metadata["mixed"] == [{"a": 1}, {"b": 2}]


class TestSchemaCompliance(BaseMemoryRuleTest):
    """Test compliance with MemoryRule schema."""

    @pytest.mark.asyncio
    async def test_all_fields_present(self):
        """Test that serialized rule contains all schema fields."""
        rule = self.create_test_rule(
            rule_id="schema-test-1",
            name="Schema Compliance Test",
        )

        rule_dict = asdict(rule)

        # Verify all MemoryRule fields are present
        expected_fields = {
            "id", "category", "name", "rule", "authority", "scope",
            "source", "conditions", "replaces", "created_at",
            "updated_at", "metadata"
        }

        assert set(rule_dict.keys()) == expected_fields

    @pytest.mark.asyncio
    async def test_field_types_compliance(self):
        """Test that field types match schema."""
        rule = self.create_test_rule(
            rule_id="schema-test-2",
            name="Field Types Test",
        )

        rule_dict = asdict(rule)

        # Verify field types
        assert isinstance(rule_dict["id"], str)
        assert isinstance(rule_dict["category"], MemoryCategory)
        assert isinstance(rule_dict["name"], str)
        assert isinstance(rule_dict["rule"], str)
        assert isinstance(rule_dict["authority"], AuthorityLevel)
        assert isinstance(rule_dict["scope"], list)
        assert isinstance(rule_dict["source"], str)
        assert isinstance(rule_dict["created_at"], datetime)
        assert isinstance(rule_dict["updated_at"], datetime)

    @pytest.mark.asyncio
    async def test_default_values_compliance(self):
        """Test that default values match schema."""
        rule_dict = {
            "id": "schema-test-3",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Default Values Test",
            "rule": "Test defaults",
            "authority": AuthorityLevel.DEFAULT,
            "scope": [],
        }

        rule = MemoryRule(**rule_dict)

        # Verify default values
        assert rule.source == "user_explicit"  # Default source value
        assert rule.conditions is None
        assert rule.replaces is None
        assert rule.metadata is None
        assert rule.created_at is not None  # Auto-initialized
        assert rule.updated_at is not None  # Auto-initialized


class TestErrorHandling(BaseMemoryRuleTest):
    """Test error handling for corrupted or invalid data."""

    @pytest.mark.asyncio
    async def test_corrupted_enum_handling(self):
        """Test handling of corrupted enum values."""
        # Simulate corrupted data with invalid enum
        corrupted_dict = {
            "id": "error-test-1",
            "category": "corrupted_category",  # Invalid
            "name": "Error Test",
            "rule": "Test error handling",
            "authority": AuthorityLevel.DEFAULT,
            "scope": [],
        }

        # Dataclass accepts invalid enum but validation can detect it
        rule = MemoryRule(**corrupted_dict)
        with pytest.raises(AssertionError):
            self.assert_rule_valid(rule)

    @pytest.mark.asyncio
    async def test_corrupted_timestamp_handling(self):
        """Test handling of corrupted timestamp values."""
        corrupted_dict = {
            "id": "error-test-2",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Timestamp Error Test",
            "rule": "Test timestamp error",
            "authority": AuthorityLevel.DEFAULT,
            "scope": [],
            "created_at": "not-a-datetime",  # Invalid type
        }

        # Dataclass accepts invalid datetime - it gets assigned as string
        rule = MemoryRule(**corrupted_dict)

        # Verify that datetime validation would catch this
        assert isinstance(rule.created_at, str)  # Not a datetime
        assert rule.created_at == "not-a-datetime"

    @pytest.mark.asyncio
    async def test_invalid_scope_type(self):
        """Test handling of invalid scope type."""
        corrupted_dict = {
            "id": "error-test-3",
            "category": MemoryCategory.BEHAVIOR,
            "name": "Scope Error Test",
            "rule": "Test scope error",
            "authority": AuthorityLevel.DEFAULT,
            "scope": "not-a-list",  # Invalid type
        }

        # Dataclass accepts invalid scope type
        rule = MemoryRule(**corrupted_dict)

        # Verify that scope is wrong type
        assert isinstance(rule.scope, str)  # Not a list
        assert rule.scope == "not-a-list"

    @pytest.mark.asyncio
    async def test_corrupted_metadata_handling(self):
        """Test that corrupted metadata is handled gracefully."""
        # Metadata can contain arbitrary JSON, so it shouldn't raise errors
        # unless it's fundamentally invalid
        rule = self.create_test_rule(
            rule_id="error-test-4",
            metadata={
                "unexpected_field": "value",
                "another_field": 123,
            }
        )

        rule_dict = asdict(rule)
        restored = MemoryRule(**rule_dict)

        # Metadata should be preserved even with unexpected fields
        assert restored.metadata["unexpected_field"] == "value"
        assert restored.metadata["another_field"] == 123


class TestAgentSerialization(BaseMemoryRuleTest):
    """Test serialization of AgentDefinition objects."""

    @pytest.mark.asyncio
    async def test_basic_agent_serialization(self):
        """Test serialization of agent definitions."""
        agent = self.create_test_agent(
            agent_id="agent-serialize-1",
            name="test-agent",
            description="Test agent for serialization",
        )

        agent_dict = asdict(agent)

        # Verify all fields are present
        assert "id" in agent_dict
        assert "name" in agent_dict
        assert "description" in agent_dict
        assert "capabilities" in agent_dict
        assert "deploy_cost" in agent_dict

        # Verify values
        assert agent_dict["id"] == "agent-serialize-1"
        assert agent_dict["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_agent_round_trip(self):
        """Test agent definition round-trip consistency."""
        original = self.create_test_agent(
            agent_id="agent-roundtrip-1",
            name="roundtrip-agent",
            description="Round-trip test agent",
            capabilities=["coding", "testing", "review"],
            deploy_cost="high",
            metadata={"version": "1.0", "tags": ["python", "testing"]},
        )

        # Round-trip
        agent_dict = asdict(original)
        restored = AgentDefinition(**agent_dict)

        # Verify all fields are preserved
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.capabilities == original.capabilities
        assert restored.deploy_cost == original.deploy_cost
        assert restored.metadata == original.metadata

    @pytest.mark.asyncio
    async def test_agent_with_last_used_timestamp(self):
        """Test agent serialization with last_used timestamp."""
        now = datetime.now(timezone.utc)
        agent = self.create_test_agent(
            agent_id="agent-timestamp-1",
            name="timestamp-agent",
            description="Agent with timestamp",
        )
        agent.last_used = now

        # Round-trip
        agent_dict = asdict(agent)
        restored = AgentDefinition(**agent_dict)

        # Verify timestamp is preserved
        assert restored.last_used is not None
        assert restored.last_used == now
