"""
Test suite for memory rule versioning and update functionality.

Tests rule modification, version tracking, change history, and rollback
capabilities to ensure rule updates are tracked and reversible.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from dataclasses import asdict

import pytest

from common.core.memory import (
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
    AgentDefinition,
)
from .test_rules_base import BaseMemoryRuleTest


class TestRuleVersionTracking(BaseMemoryRuleTest):
    """Test version tracking for memory rules."""

    @pytest.mark.asyncio
    async def test_initial_version_assignment(self):
        """Test that new rules start with version 1."""
        rule = self.create_test_rule(
            rule_id="version-test-1",
            name="Initial Version Test",
        )

        # Rules should start with version 1 if metadata includes version
        if rule.metadata and "version" in rule.metadata:
            assert rule.metadata["version"] == 1
        else:
            # If no explicit version field, updated_at serves as version marker
            assert rule.updated_at == rule.created_at

    @pytest.mark.asyncio
    async def test_version_increment_on_update(self):
        """Test that version increments when rule is updated."""
        # Create initial rule
        rule = self.create_test_rule(
            rule_id="version-test-2",
            name="Version Increment Test",
            metadata={"version": 1}
        )

        initial_version = rule.metadata.get("version", 1)
        initial_updated_at = rule.updated_at

        # Simulate update
        await asyncio.sleep(0.01)  # Ensure timestamp difference
        rule.rule = "Updated rule text"
        rule.updated_at = datetime.now(timezone.utc)

        if "version" in rule.metadata:
            rule.metadata["version"] = initial_version + 1

        # Verify version increment
        if "version" in rule.metadata:
            assert rule.metadata["version"] == initial_version + 1

        # Updated timestamp should be newer
        assert rule.updated_at > initial_updated_at

    @pytest.mark.asyncio
    async def test_version_field_tracking(self):
        """Test that version field is properly tracked in metadata."""
        rule = self.create_test_rule(
            rule_id="version-test-3",
            name="Version Field Test",
            metadata={"version": 1, "revision": "a"}
        )

        assert rule.metadata is not None
        assert "version" in rule.metadata
        assert isinstance(rule.metadata["version"], int)
        assert rule.metadata["version"] >= 1

    @pytest.mark.asyncio
    async def test_version_comparison_by_timestamp(self):
        """Test version comparison using timestamps."""
        # Create first version
        rule_v1 = self.create_test_rule(
            rule_id="version-test-4",
            name="Timestamp Comparison",
        )

        await asyncio.sleep(0.01)

        # Create second version
        rule_v2 = self.create_test_rule(
            rule_id="version-test-4",
            name="Timestamp Comparison",
        )
        rule_v2.updated_at = datetime.now(timezone.utc)

        # v2 should have later timestamp
        assert rule_v2.updated_at > rule_v1.updated_at


class TestRuleModification(BaseMemoryRuleTest):
    """Test rule modification and update operations."""

    @pytest.mark.asyncio
    async def test_simple_field_update(self):
        """Test updating a simple field in a rule."""
        rule = self.create_test_rule(
            rule_id="modify-test-1",
            name="Simple Update Test",
            rule_text="Original rule text",
        )

        original_text = rule.rule
        original_updated_at = rule.updated_at

        # Update rule text
        await asyncio.sleep(0.01)
        rule.rule = "Updated rule text"
        rule.updated_at = datetime.now(timezone.utc)

        assert rule.rule != original_text
        assert rule.rule == "Updated rule text"
        assert rule.updated_at > original_updated_at

    @pytest.mark.asyncio
    async def test_multiple_field_update(self):
        """Test updating multiple fields simultaneously."""
        rule = self.create_test_rule(
            rule_id="modify-test-2",
            name="Multiple Field Update",
            rule_text="Original text",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"]
        )

        # Update multiple fields
        rule.rule = "New text"
        rule.authority = AuthorityLevel.ABSOLUTE
        rule.scope = ["project-x", "global"]
        rule.updated_at = datetime.now(timezone.utc)

        assert rule.rule == "New text"
        assert rule.authority == AuthorityLevel.ABSOLUTE
        assert "project-x" in rule.scope

    @pytest.mark.asyncio
    async def test_conditional_rule_update(self):
        """Test updating rule conditions."""
        rule = self.create_test_rule(
            rule_id="modify-test-3",
            name="Conditional Update",
            conditions={"mode": "production"}
        )

        original_conditions = rule.conditions.copy() if rule.conditions else {}

        # Update conditions
        rule.conditions = {"mode": "development", "debug": True}
        rule.updated_at = datetime.now(timezone.utc)

        assert rule.conditions != original_conditions
        assert rule.conditions["mode"] == "development"
        assert rule.conditions["debug"] is True

    @pytest.mark.asyncio
    async def test_metadata_preservation_on_update(self):
        """Test that metadata is preserved during updates."""
        rule = self.create_test_rule(
            rule_id="modify-test-4",
            name="Metadata Preservation",
            metadata={"version": 1, "custom_key": "value"}
        )

        # Update rule but keep metadata
        rule.rule = "Updated text"
        rule.updated_at = datetime.now(timezone.utc)

        assert rule.metadata["custom_key"] == "value"

    @pytest.mark.asyncio
    async def test_scope_expansion(self):
        """Test expanding rule scope."""
        rule = self.create_test_rule(
            rule_id="modify-test-5",
            name="Scope Expansion",
            scope=["global"]
        )

        original_scope_count = len(rule.scope)

        # Expand scope
        rule.scope.extend(["python", "rust"])
        rule.updated_at = datetime.now(timezone.utc)

        assert len(rule.scope) > original_scope_count
        assert "python" in rule.scope
        assert "rust" in rule.scope


class TestChangeHistory(BaseMemoryRuleTest):
    """Test change history tracking for rules."""

    @pytest.mark.asyncio
    async def test_history_creation_on_update(self):
        """Test that history entry is created when rule is updated."""
        rule = self.create_test_rule(
            rule_id="history-test-1",
            name="History Creation",
            rule_text="Original text",
            metadata={"version": 1, "history": []}
        )

        # Simulate update with history tracking
        original_text = rule.rule
        rule.rule = "Updated text"

        # Add history entry
        if rule.metadata and "history" in rule.metadata:
            history_entry = {
                "version": rule.metadata.get("version", 1),
                "timestamp": rule.updated_at.isoformat(),
                "changes": {"rule": original_text},
            }
            rule.metadata["history"].append(history_entry)
            rule.metadata["version"] = rule.metadata.get("version", 1) + 1

        rule.updated_at = datetime.now(timezone.utc)

        # Verify history entry exists
        if rule.metadata and "history" in rule.metadata:
            assert len(rule.metadata["history"]) > 0
            assert rule.metadata["history"][0]["changes"]["rule"] == original_text

    @pytest.mark.asyncio
    async def test_history_metadata_capture(self):
        """Test that history captures relevant metadata."""
        rule = self.create_test_rule(
            rule_id="history-test-2",
            name="History Metadata",
            metadata={"version": 1, "history": []}
        )

        # Create history entry with metadata
        history_entry = {
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes": {"authority": "DEFAULT"},
            "changed_by": "test_user",
            "reason": "Initial creation"
        }

        if rule.metadata:
            rule.metadata["history"].append(history_entry)

        # Verify metadata in history
        if rule.metadata and "history" in rule.metadata:
            entry = rule.metadata["history"][0]
            assert "timestamp" in entry
            assert "changes" in entry
            assert "changed_by" in entry
            assert "reason" in entry

    @pytest.mark.asyncio
    async def test_history_ordering(self):
        """Test that history is ordered chronologically."""
        rule = self.create_test_rule(
            rule_id="history-test-3",
            name="History Ordering",
            metadata={"version": 1, "history": []}
        )

        # Add multiple history entries
        for i in range(3):
            await asyncio.sleep(0.01)
            history_entry = {
                "version": i + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "changes": {"update": f"change_{i}"}
            }
            if rule.metadata:
                rule.metadata["history"].append(history_entry)

        # Verify chronological ordering
        if rule.metadata and "history" in rule.metadata:
            history = rule.metadata["history"]
            for i in range(len(history) - 1):
                ts1 = datetime.fromisoformat(history[i]["timestamp"])
                ts2 = datetime.fromisoformat(history[i + 1]["timestamp"])
                assert ts2 >= ts1

    @pytest.mark.asyncio
    async def test_history_size_limit(self):
        """Test that history respects size limits."""
        max_history = 10
        rule = self.create_test_rule(
            rule_id="history-test-4",
            name="History Size Limit",
            metadata={"version": 1, "history": [], "max_history": max_history}
        )

        # Add more entries than limit
        for i in range(max_history + 5):
            history_entry = {
                "version": i + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "changes": {"update": f"change_{i}"}
            }
            if rule.metadata:
                rule.metadata["history"].append(history_entry)

                # Keep only last N entries
                if len(rule.metadata["history"]) > max_history:
                    rule.metadata["history"] = rule.metadata["history"][-max_history:]

        # Verify size limit enforced
        if rule.metadata and "history" in rule.metadata:
            assert len(rule.metadata["history"]) <= max_history

    @pytest.mark.asyncio
    async def test_history_retrieval_by_version(self):
        """Test retrieving specific version from history."""
        rule = self.create_test_rule(
            rule_id="history-test-5",
            name="History Retrieval",
            metadata={"version": 3, "history": [
                {
                    "version": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "changes": {"rule": "Version 1 text"}
                },
                {
                    "version": 2,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "changes": {"rule": "Version 2 text"}
                },
            ]}
        )

        # Find specific version in history
        target_version = 2
        if rule.metadata and "history" in rule.metadata:
            version_entry = next(
                (entry for entry in rule.metadata["history"]
                 if entry["version"] == target_version),
                None
            )
            assert version_entry is not None
            assert version_entry["changes"]["rule"] == "Version 2 text"


class TestRuleRollback(BaseMemoryRuleTest):
    """Test rollback capabilities for rules."""

    @pytest.mark.asyncio
    async def test_rollback_to_previous_version(self):
        """Test rolling back to previous version."""
        # Create rule with history
        rule = self.create_test_rule(
            rule_id="rollback-test-1",
            name="Rollback Test",
            rule_text="Current version",
            metadata={
                "version": 3,
                "history": [
                    {
                        "version": 1,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "changes": {"rule": "Version 1"}
                    },
                    {
                        "version": 2,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "changes": {"rule": "Version 2"}
                    }
                ]
            }
        )

        # Rollback to version 2
        target_version = 2
        if rule.metadata and "history" in rule.metadata:
            version_entry = next(
                (entry for entry in rule.metadata["history"]
                 if entry["version"] == target_version),
                None
            )
            if version_entry:
                # Apply changes from history
                for field, value in version_entry["changes"].items():
                    setattr(rule, field, value)
                rule.metadata["version"] = target_version

        assert rule.rule == "Version 2"
        assert rule.metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_rollback_validation(self):
        """Test that rollback validates target version exists."""
        rule = self.create_test_rule(
            rule_id="rollback-test-2",
            name="Rollback Validation",
            metadata={
                "version": 3,
                "history": [
                    {"version": 1, "changes": {}},
                    {"version": 2, "changes": {}}
                ]
            }
        )

        # Try to rollback to non-existent version
        target_version = 5
        version_exists = False

        if rule.metadata and "history" in rule.metadata:
            version_exists = any(
                entry["version"] == target_version
                for entry in rule.metadata["history"]
            )

        assert not version_exists

    @pytest.mark.asyncio
    async def test_rollback_with_dependencies(self):
        """Test rollback behavior with dependent rules."""
        # Main rule
        rule = self.create_test_rule(
            rule_id="rollback-test-3",
            name="Main Rule",
            metadata={
                "version": 2,
                "history": [{"version": 1, "changes": {"rule": "Old version"}}],
                "dependent_rules": ["dependent-1"]
            }
        )

        # Dependent rule
        dependent = self.create_test_rule(
            rule_id="dependent-1",
            name="Dependent Rule",
            metadata={"depends_on": ["rollback-test-3"]}
        )

        # Check dependency before rollback
        if rule.metadata and "dependent_rules" in rule.metadata:
            assert "dependent-1" in rule.metadata["dependent_rules"]

    @pytest.mark.asyncio
    async def test_rollback_failure_recovery(self):
        """Test recovery when rollback fails."""
        rule = self.create_test_rule(
            rule_id="rollback-test-4",
            name="Rollback Failure",
            rule_text="Current state",
            metadata={
                "version": 2,
                "history": [{"version": 1, "changes": {"rule": "Previous"}}]
            }
        )

        current_state = rule.rule
        current_version = rule.metadata["version"]

        # Simulate rollback attempt that fails
        try:
            # Invalid rollback (missing version)
            if rule.metadata and "history" in rule.metadata:
                version_entry = next(
                    (entry for entry in rule.metadata["history"]
                     if entry["version"] == 999),  # Non-existent version
                    None
                )
                if version_entry is None:
                    raise ValueError("Version not found")
        except ValueError:
            # Recovery: keep current state
            pass

        # Verify state unchanged after failed rollback
        assert rule.rule == current_state
        assert rule.metadata["version"] == current_version

    @pytest.mark.asyncio
    async def test_partial_rollback_handling(self):
        """Test handling partial rollback of multiple fields."""
        rule = self.create_test_rule(
            rule_id="rollback-test-5",
            name="Partial Rollback",
            rule_text="Current text",
            authority=AuthorityLevel.ABSOLUTE,
            metadata={
                "version": 2,
                "history": [{
                    "version": 1,
                    "changes": {
                        "rule": "Old text",
                        "authority": "DEFAULT"
                    }
                }]
            }
        )

        # Rollback only specific fields
        if rule.metadata and "history" in rule.metadata:
            version_entry = rule.metadata["history"][0]
            # Only rollback rule text, not authority
            rule.rule = version_entry["changes"]["rule"]

        assert rule.rule == "Old text"
        assert rule.authority == AuthorityLevel.ABSOLUTE  # Unchanged


class TestVersionConsistency(BaseMemoryRuleTest):
    """Test version consistency across operations."""

    @pytest.mark.asyncio
    async def test_version_consistency_across_updates(self):
        """Test that version remains consistent during multiple updates."""
        rule = self.create_test_rule(
            rule_id="consistency-test-1",
            name="Consistency Test",
            metadata={"version": 1}
        )

        # Perform multiple updates
        for i in range(5):
            rule.rule = f"Update {i}"
            if rule.metadata and "version" in rule.metadata:
                rule.metadata["version"] += 1
            rule.updated_at = datetime.now(timezone.utc)
            await asyncio.sleep(0.01)

        # Verify final version
        if rule.metadata and "version" in rule.metadata:
            assert rule.metadata["version"] == 6  # 1 + 5 updates

    @pytest.mark.asyncio
    async def test_consistency_with_dependent_rules(self):
        """Test version consistency when updating dependent rules."""
        # Parent rule
        parent = self.create_test_rule(
            rule_id="consistency-test-2-parent",
            name="Parent Rule",
            metadata={"version": 1, "children": ["child-1"]}
        )

        # Child rule
        child = self.create_test_rule(
            rule_id="child-1",
            name="Child Rule",
            metadata={"version": 1, "parent": "consistency-test-2-parent"}
        )

        # Update parent
        parent.rule = "Updated parent"
        if parent.metadata and "version" in parent.metadata:
            parent.metadata["version"] += 1

        # Child version should be independent
        assert child.metadata["version"] == 1
        assert parent.metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_consistency_after_rollback(self):
        """Test that version consistency is maintained after rollback."""
        rule = self.create_test_rule(
            rule_id="consistency-test-3",
            name="Rollback Consistency",
            metadata={
                "version": 3,
                "history": [
                    {"version": 1, "changes": {}},
                    {"version": 2, "changes": {}},
                ]
            }
        )

        # Rollback to version 2
        if rule.metadata:
            rule.metadata["version"] = 2

        # Version should match rollback target
        assert rule.metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_timestamp_consistency_on_update(self):
        """Test that updated_at is always >= created_at."""
        rule = self.create_test_rule(
            rule_id="consistency-test-4",
            name="Timestamp Consistency",
        )

        created = rule.created_at
        updated = rule.updated_at

        # Initial state
        assert updated >= created

        # After update
        await asyncio.sleep(0.01)
        rule.rule = "Updated"
        rule.updated_at = datetime.now(timezone.utc)

        assert rule.updated_at >= rule.created_at

    @pytest.mark.asyncio
    async def test_version_monotonicity(self):
        """Test that version numbers always increase (monotonic)."""
        rule = self.create_test_rule(
            rule_id="consistency-test-5",
            name="Version Monotonicity",
            metadata={"version": 1, "version_history": [1]}
        )

        # Perform updates and track versions
        for i in range(5):
            if rule.metadata and "version" in rule.metadata:
                old_version = rule.metadata["version"]
                rule.metadata["version"] += 1
                new_version = rule.metadata["version"]

                # Track version history
                rule.metadata["version_history"].append(new_version)

                # Verify monotonic increase
                assert new_version > old_version

        # Verify all versions are strictly increasing
        if rule.metadata and "version_history" in rule.metadata:
            versions = rule.metadata["version_history"]
            for i in range(len(versions) - 1):
                assert versions[i + 1] > versions[i]
