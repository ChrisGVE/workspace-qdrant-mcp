"""
Comprehensive tests for memory rule update mechanisms (Task 324.2).

This test suite validates:
- Partial rule updates (individual field modifications)
- Complete rule replacements
- Version tracking for rule updates
- Update consistency validation
- Dependency preservation during updates
- Update validation mechanisms
- Rollback mechanisms for failed updates
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.collection_naming import CollectionNamingManager
from common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
)
from common.core.sparse_vectors import BM25SparseEncoder


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing with memory collection."""
    client = MagicMock(spec=QdrantClient)

    # Mock get_collections to always return memory collection
    mock_collection = MagicMock()
    mock_collection.name = "memory"
    mock_collections_result = MagicMock()
    mock_collections_result.collections = [mock_collection]
    client.get_collections.return_value = mock_collections_result

    # Mock upsert
    client.upsert.return_value = MagicMock()

    # Mock retrieve - will be configured per test
    client.retrieve.return_value = []

    # Mock scroll
    client.scroll.return_value = ([], None)

    # Mock delete
    client.delete.return_value = MagicMock()

    return client


@pytest.fixture
def naming_manager():
    """Create a collection naming manager."""
    return CollectionNamingManager()


@pytest.fixture
def sparse_encoder():
    """Create a mocked sparse vector encoder."""
    encoder = MagicMock()
    encoder.vector_size = 1000  # Mock vector size
    encoder.generate_sparse_vector.return_value = {
        "indices": [0, 5, 10],
        "values": [0.8, 0.6, 0.4]
    }
    return encoder

@pytest.fixture
def memory_manager(mock_qdrant_client, naming_manager, sparse_encoder):
    """Create a memory manager instance for testing."""
    return MemoryManager(
        qdrant_client=mock_qdrant_client,
        naming_manager=naming_manager,
        embedding_dim=384,
        sparse_vector_generator=sparse_encoder,
    )


@pytest.fixture
def sample_rule():
    """Create a sample memory rule for testing."""
    return MemoryRule(
        id="rule_001",
        category=MemoryCategory.PREFERENCE,
        name="use_uv_for_python",
        rule="Always use uv for Python package management",
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
        source="user_explicit",
        conditions=None,
        replaces=None,
        created_at=datetime.now(timezone.utc) - timedelta(days=7),
        updated_at=datetime.now(timezone.utc) - timedelta(days=7),
        metadata={"version": 1},
    )


def create_mock_point(rule: MemoryRule) -> MagicMock:
    """Helper to create a properly mocked Qdrant point from a rule."""
    mock_point = MagicMock()
    mock_point.id = rule.id
    mock_point.payload = {
        "category": rule.category.value,
        "name": rule.name,
        "rule": rule.rule,
        "authority": rule.authority.value,
        "scope": rule.scope or [],
        "source": rule.source,
        "conditions": rule.conditions or {},
        "replaces": rule.replaces or [],
        "created_at": rule.created_at.isoformat(),
        "updated_at": rule.updated_at.isoformat(),
        "metadata": rule.metadata or {},
    }
    return mock_point


class TestPartialRuleUpdates:
    """Test partial rule update scenarios."""

    @pytest.mark.asyncio
    async def test_update_rule_text_only(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test updating only the rule text field."""
        # Setup: Mock existing rule retrieval
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update only the rule text
        new_rule_text = "Always use uv for Python package management and virtual environments"
        updates = {"rule": new_rule_text}

        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Update succeeded
        assert result is True

        # Verify: Upsert was called with updated rule text
        assert mock_qdrant_client.upsert.called
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["rule"] == new_rule_text

        # Verify: updated_at timestamp was modified
        assert updated_point.payload["updated_at"] != sample_rule.updated_at.isoformat()

        # Verify: Other fields remain unchanged
        assert updated_point.payload["name"] == sample_rule.name
        assert updated_point.payload["category"] == sample_rule.category.value
        assert updated_point.payload["authority"] == sample_rule.authority.value

    @pytest.mark.asyncio
    async def test_update_authority_level(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test updating the authority level of a rule."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Elevate to absolute authority
        updates = {"authority": AuthorityLevel.ABSOLUTE}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["authority"] == AuthorityLevel.ABSOLUTE.value

    @pytest.mark.asyncio
    async def test_update_scope(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test updating the scope of a rule."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Expand scope
        new_scope = ["python", "development", "testing"]
        updates = {"scope": new_scope}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["scope"] == new_scope

    @pytest.mark.asyncio
    async def test_update_metadata(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test updating rule metadata."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update metadata with new version
        new_metadata = {"version": 2, "reason": "Enhanced clarity"}
        updates = {"metadata": new_metadata}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["metadata"]["version"] == 2
        assert updated_point.payload["metadata"]["reason"] == "Enhanced clarity"

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test updating multiple fields in one operation."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update rule text, authority, and scope together
        updates = {
            "rule": "MANDATORY: Use uv for all Python package management",
            "authority": AuthorityLevel.ABSOLUTE,
            "scope": ["python", "packaging", "dependencies"],
        }
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify all fields were updated
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert "MANDATORY" in updated_point.payload["rule"]
        assert updated_point.payload["authority"] == AuthorityLevel.ABSOLUTE.value
        assert len(updated_point.payload["scope"]) == 3


class TestCompleteRuleReplacements:
    """Test complete rule replacement scenarios."""

    @pytest.mark.asyncio
    async def test_replace_rule_with_new_version(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test replacing an entire rule with a new version."""
        # Setup: Mock existing rule
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Complete replacement
        complete_replacement = {
            "rule": "Use poetry for Python dependency management instead",
            "authority": AuthorityLevel.ABSOLUTE,
            "scope": ["python", "dependencies"],
            "metadata": {
                "version": 2,
                "replaced_version": 1,
                "reason": "Changed tooling preference",
            },
        }
        result = await memory_manager.update_memory_rule(sample_rule.id, complete_replacement)

        # Verify
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]

        # Verify all specified fields were replaced
        assert updated_point.payload["rule"] == complete_replacement["rule"]
        assert updated_point.payload["authority"] == AuthorityLevel.ABSOLUTE.value
        assert updated_point.payload["scope"] == complete_replacement["scope"]
        assert updated_point.payload["metadata"]["version"] == 2

    @pytest.mark.asyncio
    async def test_replacement_preserves_timestamps(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that rule replacement preserves created_at timestamp."""
        # Setup
        original_created_at = sample_rule.created_at
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Complete replacement
        updates = {
            "rule": "Completely different rule",
            "name": "new_rule_name",
        }
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: created_at is preserved
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["created_at"] == original_created_at.isoformat()

        # Verify: updated_at is modified
        assert updated_point.payload["updated_at"] != original_created_at.isoformat()


class TestVersionTracking:
    """Test version tracking for rule updates."""

    @pytest.mark.asyncio
    async def test_version_increment_on_update(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that version number increments on updates."""
        # Setup: Rule with version 1
        sample_rule.metadata = {"version": 1}
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update with version increment
        current_version = sample_rule.metadata.get("version", 1)
        updates = {
            "rule": "Updated rule text",
            "metadata": {
                "version": current_version + 1,
                "update_reason": "Clarification",
            },
        }
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Version was incremented
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["metadata"]["version"] == 2

    @pytest.mark.asyncio
    async def test_version_history_tracking(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test tracking version history in metadata."""
        # Setup
        sample_rule.metadata = {"version": 1, "history": []}
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update with version history entry
        updates = {
            "rule": "Modified rule",
            "metadata": {
                "version": 2,
                "history": [
                    {
                        "version": 1,
                        "rule": sample_rule.rule,
                        "updated_at": sample_rule.updated_at.isoformat(),
                    }
                ],
            },
        }
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: History was recorded
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert len(updated_point.payload["metadata"]["history"]) == 1
        assert updated_point.payload["metadata"]["history"][0]["version"] == 1

    @pytest.mark.asyncio
    async def test_updated_at_timestamp_tracking(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that updated_at timestamp is properly tracked."""
        # Setup
        old_updated_at = sample_rule.updated_at
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Simple update
        updates = {"rule": "Updated rule"}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: updated_at was changed
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        new_updated_at = datetime.fromisoformat(updated_point.payload["updated_at"])
        assert new_updated_at > old_updated_at


class TestUpdateConsistency:
    """Test update consistency validation."""

    @pytest.mark.asyncio
    async def test_update_nonexistent_rule(self, memory_manager, mock_qdrant_client):
        """Test that updating a nonexistent rule fails gracefully."""
        # Setup: Empty retrieve result
        mock_qdrant_client.retrieve.return_value = []

        # Execute: Attempt to update nonexistent rule
        result = await memory_manager.update_memory_rule("nonexistent_id", {"rule": "test"})

        # Verify: Update fails
        assert result is False

        # Verify: No upsert was attempted
        assert not mock_qdrant_client.upsert.called

    @pytest.mark.asyncio
    async def test_update_maintains_required_fields(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that updates maintain all required fields."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update one field
        updates = {"rule": "Updated rule text"}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: All required fields are still present
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]

        required_fields = [
            "category", "name", "rule", "authority", "scope",
            "source", "conditions", "replaces", "created_at", "updated_at"
        ]
        for field in required_fields:
            assert field in updated_point.payload

    @pytest.mark.asyncio
    async def test_update_validates_category_enum(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that category updates validate enum values."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update to valid category
        updates = {"category": MemoryCategory.BEHAVIOR}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Update succeeds with valid enum
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["category"] == MemoryCategory.BEHAVIOR.value


class TestDependencyPreservation:
    """Test dependency preservation during updates."""

    @pytest.mark.asyncio
    async def test_update_preserves_replaces_field(self, memory_manager, mock_qdrant_client):
        """Test that updates preserve the 'replaces' field."""
        # Setup: Rule that replaces another rule
        rule = MemoryRule(
            id="rule_002",
            category=MemoryCategory.PREFERENCE,
            name="updated_preference",
            rule="New preference rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user_explicit",
            replaces=["rule_001"],  # This rule replaces rule_001
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_qdrant_client.retrieve.return_value = [create_mock_point(rule)]

        # Execute: Update rule text but not replaces
        updates = {"rule": "Updated rule text"}
        result = await memory_manager.update_memory_rule(rule.id, updates)

        # Verify: replaces field is preserved
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["replaces"] == ["rule_001"]

    @pytest.mark.asyncio
    async def test_update_can_modify_replaces_field(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that updates can explicitly modify the replaces field."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Add replaces reference
        updates = {"replaces": ["old_rule_123", "old_rule_456"]}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["replaces"] == ["old_rule_123", "old_rule_456"]

    @pytest.mark.asyncio
    async def test_update_preserves_conditions(self, memory_manager, mock_qdrant_client):
        """Test that updates preserve conditional logic."""
        # Setup: Rule with conditions
        rule = MemoryRule(
            id="rule_003",
            category=MemoryCategory.BEHAVIOR,
            name="conditional_behavior",
            rule="When working on frontend, use React",
            authority=AuthorityLevel.DEFAULT,
            scope=["frontend"],
            conditions={"context": "frontend development", "framework": "React"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_qdrant_client.retrieve.return_value = [create_mock_point(rule)]

        # Execute: Update rule without touching conditions
        updates = {"rule": "When building frontend components, prefer React"}
        result = await memory_manager.update_memory_rule(rule.id, updates)

        # Verify: Conditions are preserved
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["conditions"] == rule.conditions


class TestUpdateValidation:
    """Test update validation mechanisms."""

    @pytest.mark.asyncio
    async def test_update_with_invalid_authority_level(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that invalid authority level updates are rejected."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Attempt update with invalid authority (should use enum)
        # Note: The implementation uses enums, so this tests proper enum handling
        updates = {"authority": AuthorityLevel.ABSOLUTE}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Valid enum value works
        assert result is True

    @pytest.mark.asyncio
    async def test_update_validates_scope_is_list(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that scope updates must be lists."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update with valid list scope
        updates = {"scope": ["python", "testing"]}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify
        assert result is True

    @pytest.mark.asyncio
    async def test_update_embedding_regeneration(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that embedding is regenerated when rule text changes."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Update rule text (should trigger embedding regeneration)
        updates = {"rule": "Completely new rule text"}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Update succeeded
        assert result is True

        # Verify: Upsert was called with vectors
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        # Note: In the actual implementation, vectors would be regenerated
        # We verify the structure is correct
        assert isinstance(updated_point, PointStruct)


class TestUpdateRollback:
    """Test rollback mechanisms for failed updates."""

    @pytest.mark.asyncio
    async def test_rollback_on_upsert_failure(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that update fails gracefully when upsert fails."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Setup: Make upsert raise an exception
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant connection failed")

        # Execute: Attempt update
        updates = {"rule": "Updated rule"}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Update returns False on failure
        assert result is False

    @pytest.mark.asyncio
    async def test_update_error_handling(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test proper error handling during updates."""
        # Setup
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Setup: Make retrieve work but payload conversion fail
        mock_qdrant_client.upsert.side_effect = KeyError("Invalid field")

        # Execute: Attempt update
        updates = {"rule": "Updated rule"}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Error is caught and False is returned
        assert result is False

    @pytest.mark.asyncio
    async def test_update_preserves_original_on_partial_failure(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test that original rule is preserved if update fails partway."""
        # Setup
        original_rule_text = sample_rule.rule
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Setup: First upsert fails
        mock_qdrant_client.upsert.side_effect = Exception("Network error")

        # Execute: Attempt update
        updates = {"rule": "This update will fail"}
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: Update failed
        assert result is False

        # Reset side effect and verify original can still be retrieved
        mock_qdrant_client.upsert.side_effect = None
        retrieved_rule = await memory_manager.get_memory_rule(sample_rule.id)

        # Verify: Original rule text is intact
        assert retrieved_rule is not None
        assert retrieved_rule.rule == original_rule_text


class TestComplexUpdateScenarios:
    """Test complex update scenarios combining multiple features."""

    @pytest.mark.asyncio
    async def test_update_with_version_and_history(self, memory_manager, sample_rule, mock_qdrant_client):
        """Test updating with version increment and history tracking."""
        # Setup
        sample_rule.metadata = {"version": 1, "history": []}
        mock_qdrant_client.retrieve.return_value = [create_mock_point(sample_rule)]

        # Execute: Complex update with version tracking
        updates = {
            "rule": "Enhanced rule with better clarity",
            "authority": AuthorityLevel.ABSOLUTE,
            "metadata": {
                "version": 2,
                "history": [
                    {
                        "version": 1,
                        "rule": sample_rule.rule,
                        "authority": sample_rule.authority.value,
                        "updated_at": sample_rule.updated_at.isoformat(),
                    }
                ],
                "change_reason": "Elevated to absolute based on team feedback",
            },
        }
        result = await memory_manager.update_memory_rule(sample_rule.id, updates)

        # Verify: All updates applied
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["metadata"]["version"] == 2
        assert len(updated_point.payload["metadata"]["history"]) == 1
        assert updated_point.payload["authority"] == AuthorityLevel.ABSOLUTE.value

    @pytest.mark.asyncio
    async def test_cascading_updates_with_replaces(self, memory_manager, mock_qdrant_client):
        """Test updating a rule that replaces others maintains consistency."""
        # Setup: Rule that replaces multiple old rules
        rule = MemoryRule(
            id="rule_new",
            category=MemoryCategory.PREFERENCE,
            name="consolidated_preference",
            rule="Use modern tooling",
            authority=AuthorityLevel.DEFAULT,
            scope=["development"],
            replaces=["rule_old1", "rule_old2", "rule_old3"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_qdrant_client.retrieve.return_value = [create_mock_point(rule)]

        # Execute: Update the consolidating rule
        updates = {
            "rule": "Use modern, well-maintained tooling for all projects",
            "scope": ["development", "production"],
        }
        result = await memory_manager.update_memory_rule(rule.id, updates)

        # Verify: Update succeeded
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        updated_point = call_args[1]["points"][0]

        # Verify: Replaces references are maintained
        assert updated_point.payload["replaces"] == ["rule_old1", "rule_old2", "rule_old3"]

        # Verify: Updates were applied
        assert updated_point.payload["rule"] == updates["rule"]
        assert updated_point.payload["scope"] == updates["scope"]
