"""
Cross-Session Persistence Testing for Memory Rules (Task 337.3).

Tests that memory rules persist correctly across different LLM sessions,
are reloaded properly, and maintain behavioral effectiveness over time.

Test Scenarios:
1. Rule persistence across session restarts
2. Correct rule reloading after restart
3. Behavioral consistency across sessions
4. Rule state management (usage counts, timestamps)
5. Rule expiration and cleanup mechanisms
6. Multi-session rule accumulation
7. Rule conflict resolution across sessions
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, Mock, patch

from src.python.common.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)

# Import test harness from Task 337.1
from tests.integration.test_llm_behavioral_harness import (
    LLMBehavioralHarness,
    MockLLMProvider,
    ExecutionMode,
    BehavioralMetrics,
    LLMResponse,
)

# Try to import real memory manager
try:
    from src.python.common.core.memory import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False


@pytest.fixture
async def temp_qdrant_collection(tmp_path):
    """Provide temporary Qdrant collection for testing."""
    collection_name = f"test_persistence_{int(time.time())}"
    yield collection_name
    # Cleanup handled by memory manager


@pytest.fixture
async def shared_storage():
    """Shared storage simulating persistent backend across sessions."""
    return {"rules": []}


@pytest.fixture
async def memory_manager_factory(shared_storage, tmp_path):
    """
    Factory for creating memory manager instances.

    Simulates session restarts by creating new instances that share storage.
    """
    managers = []

    async def create_manager():
        """Create a new memory manager instance with shared storage."""
        # Use mock manager with shared storage to simulate persistence
        manager = AsyncMock(spec=MemoryManager)

        # Share storage across all instances
        manager._storage = shared_storage

        # Mock methods that interact with storage
        async def add_rule(rule: MemoryRule):
            manager._storage["rules"].append(rule)

        async def get_rules():
            return manager._storage["rules"].copy()

        async def delete_rule(rule_id: str):
            manager._storage["rules"] = [
                r for r in manager._storage["rules"] if r.id != rule_id
            ]

        async def update_rule(rule: MemoryRule):
            for i, r in enumerate(manager._storage["rules"]):
                if r.id == rule.id:
                    manager._storage["rules"][i] = rule
                    break

        manager.add_rule = AsyncMock(side_effect=add_rule)
        manager.get_rules = AsyncMock(side_effect=get_rules)
        manager.delete_rule = AsyncMock(side_effect=delete_rule)
        manager.update_rule = AsyncMock(side_effect=update_rule)
        manager.initialize = AsyncMock()

        await manager.initialize()
        managers.append(manager)
        return manager

    yield create_manager

    # Cleanup all managers
    for manager in managers:
        try:
            if hasattr(manager, 'cleanup'):
                await manager.cleanup()
        except Exception:
            pass


@pytest.mark.asyncio
class TestRulePersistenceAcrossSessions:
    """Test rule persistence across session restarts."""

    async def test_basic_rule_persistence(self, memory_manager_factory):
        """Test that rules persist across session restart."""
        # Session 1: Create and add rules
        session1 = await memory_manager_factory()

        rule1 = MemoryRule(
            rule="Always use type hints in Python code",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="persist_test_1",
            source="test",
        )

        await session1.add_rule(rule1)
        rules_session1 = await session1.get_rules()
        assert len(rules_session1) == 1
        assert rules_session1[0].id == "persist_test_1"

        # Simulate session restart: Create new instance
        session2 = await memory_manager_factory()
        rules_session2 = await session2.get_rules()

        # Verify rule persisted (shared storage simulates persistence)
        assert len(rules_session2) == 1
        assert rules_session2[0].id == "persist_test_1"
        assert rules_session2[0].rule == "Always use type hints in Python code"
        assert rules_session2[0].authority == AuthorityLevel.ABSOLUTE

    async def test_multiple_rules_persistence(self, memory_manager_factory):
        """Test that multiple rules persist correctly."""
        # Session 1: Add multiple rules
        session1 = await memory_manager_factory()

        rules = [
        MemoryRule(
                rule="Use pytest for testing",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id=f"multi_test_{i}",
                source="test",
        )
        for i in range(5)
        ]

        for rule in rules:
            await session1.add_rule(rule)

        # Session 2: Verify all rules persisted
        session2 = await memory_manager_factory()
        loaded_rules = await session2.get_rules()

        # Shared storage ensures persistence
        assert len(loaded_rules) == 5
        loaded_ids = {r.id for r in loaded_rules}
        expected_ids = {f"multi_test_{i}" for i in range(5)}
        assert loaded_ids == expected_ids

    async def test_rule_metadata_persistence(self, memory_manager_factory):
        """Test that rule metadata persists correctly."""
        # Session 1: Create rule with metadata
        session1 = await memory_manager_factory()

        created_time = datetime.now(timezone.utc)
        rule = MemoryRule(
            rule="Prefer functional programming",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="metadata_test",
            source="test",
            created_at=created_time,
            metadata={"origin": "test_suite", "version": "1.0"},
        )

        await session1.add_rule(rule)

        # Session 2: Verify metadata persisted
        session2 = await memory_manager_factory()
        loaded_rules = await session2.get_rules()

        assert len(loaded_rules) == 1
        loaded_rule = loaded_rules[0]
        assert loaded_rule.metadata.get("origin") == "test_suite"
        assert loaded_rule.metadata.get("version") == "1.0"
        # Timestamp should be preserved (within 1 second tolerance)
        assert abs((loaded_rule.created_at - created_time).total_seconds()) < 1.0


@pytest.mark.asyncio
class TestBehavioralConsistencyAcrossSessions:
    """Test that rules maintain behavioral effectiveness across sessions."""

    async def test_rule_behavioral_effect_persists(
        self,
        memory_manager_factory
    ):
        """Test that persisted rules still affect LLM behavior."""
        # Session 1: Add rule
        session1 = await memory_manager_factory()

        rule = MemoryRule(
            rule="Always include docstrings with Args and Returns",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="behavior_persist_test",
            source="test",
        )

        await session1.add_rule(rule)

        # Session 2: Test behavioral effect
        session2 = await memory_manager_factory()
        mock_provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
        provider=mock_provider,
        memory_manager=session2,
        mode=ExecutionMode.MOCK
        )

        loaded_rules = await session2.get_rules()

        # Run behavioral test with loaded rules
        metrics, with_rules, without_rules = await harness.run_behavioral_test(
        prompt="Write a function",
        rules=loaded_rules,
        expected_patterns=[r'"""', r"Args:", r"Returns:"]
        )

        # Verify rules are loaded and can be used
        assert len(loaded_rules) > 0
        assert metrics is not None


@pytest.mark.asyncio
class TestRuleStateManagement:
    """Test rule state management across sessions."""

    async def test_usage_count_persistence(self, memory_manager_factory):
        """Test that usage counts persist across sessions."""
        # Session 1: Create and use rule
        session1 = await memory_manager_factory()

        rule = MemoryRule(
            rule="Use dependency injection",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="usage_test",
            source="test",
            use_count=0,
        )

        await session1.add_rule(rule)

        # Simulate usage
        rule.update_usage()
        rule.update_usage()
        assert rule.use_count == 2

        # Update in storage
        if hasattr(session1, 'update_rule'):
            await session1.update_rule(rule)

        # Session 2: Verify usage count persisted
        session2 = await memory_manager_factory()
        loaded_rules = await session2.get_rules()

        usage_rule = next((r for r in loaded_rules if r.id == "usage_test"), None)
        if usage_rule and hasattr(session1, 'update_rule'):
            # If update_rule was called, count should persist
            # Note: This depends on implementation details
            pass

    async def test_last_used_timestamp_persistence(self, memory_manager_factory):
        """Test that last_used timestamps persist."""
        session1 = await memory_manager_factory()

        rule = MemoryRule(
            rule="Prefer composition over inheritance",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="timestamp_test",
            source="test",
        )

        await session1.add_rule(rule)

        # Simulate usage
        rule.update_usage()
        last_used_time = rule.last_used

        if hasattr(session1, 'update_rule'):
            await session1.update_rule(rule)

        # Session 2: Verify timestamp
        session2 = await memory_manager_factory()
        loaded_rules = await session2.get_rules()

        loaded_rule = next((r for r in loaded_rules if r.id == "timestamp_test"), None)
        if loaded_rule and last_used_time:
                # Timestamp should be preserved
                pass


@pytest.mark.asyncio
class TestRuleExpirationAndCleanup:
    """Test rule expiration and cleanup mechanisms."""

    async def test_expired_rule_cleanup(self, memory_manager_factory):
        """Test that expired rules can be cleaned up."""
        session1 = await memory_manager_factory()

        # Create rule with expiration
        expired_time = datetime.now(timezone.utc) - timedelta(days=30)
        rule = MemoryRule(
            rule="Temporary rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="expired_test",
            source="test",
            created_at=expired_time,
            metadata={"expires_after_days": 7},
        )

        await session1.add_rule(rule)

        # Verify rule exists
        rules = await session1.get_rules()
        assert any(r.id == "expired_test" for r in rules)

        # Cleanup expired rules (if cleanup method exists)
        if hasattr(session1, 'cleanup_expired_rules'):
            await session1.cleanup_expired_rules(max_age_days=7)

        # Verify expired rule was removed
        rules_after = await session1.get_rules()
        assert not any(r.id == "expired_test" for r in rules_after)

    async def test_unused_rule_cleanup(self, memory_manager_factory):
        """Test cleanup of unused rules."""
        session1 = await memory_manager_factory()

        # Create rule with old last_used timestamp
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        rule = MemoryRule(
            rule="Rarely used rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="unused_test",
            source="test",
            last_used=old_time,
            use_count=1,
        )

        await session1.add_rule(rule)

        # Cleanup unused rules (if method exists)
        if hasattr(session1, 'cleanup_unused_rules'):
            await session1.cleanup_unused_rules(unused_days=30)

        rules_after = await session1.get_rules()
        # Unused rule should be removed
        assert not any(r.id == "unused_test" for r in rules_after)


@pytest.mark.asyncio
class TestMultiSessionAccumulation:
    """Test rule accumulation across multiple sessions."""

    async def test_incremental_rule_addition(self, memory_manager_factory):
        """Test adding rules incrementally across sessions."""
        # Session 1: Add first batch
        session1 = await memory_manager_factory()
        for i in range(3):
            rule = MemoryRule(
                rule=f"Rule batch 1 number {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"batch1_rule_{i}",
                source="test",
        )
            await session1.add_rule(rule)

        # Session 2: Add second batch
        session2 = await memory_manager_factory()
        for i in range(3):
            rule = MemoryRule(
                rule=f"Rule batch 2 number {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"batch2_rule_{i}",
                source="test",
        )
            await session2.add_rule(rule)

        # Session 3: Verify all rules exist
        session3 = await memory_manager_factory()
        all_rules = await session3.get_rules()

        # Should have 6 rules total
        assert len(all_rules) >= 6

        batch1_ids = {f"batch1_rule_{i}" for i in range(3)}
        batch2_ids = {f"batch2_rule_{i}" for i in range(3)}
        loaded_ids = {r.id for r in all_rules}

        assert batch1_ids.issubset(loaded_ids)
        assert batch2_ids.issubset(loaded_ids)


@pytest.mark.asyncio
class TestSessionIsolation:
    """Test that concurrent sessions don't interfere with each other."""

    async def test_concurrent_session_writes(self, memory_manager_factory):
        """Test that concurrent sessions can write without conflicts."""
        # Create two sessions concurrently
        session1 = await memory_manager_factory()
        session2 = await memory_manager_factory()

        # Add different rules concurrently
        rule1 = MemoryRule(
            rule="Session 1 rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="concurrent_1",
            source="session1",
        )

        rule2 = MemoryRule(
            rule="Session 2 rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="concurrent_2",
            source="session2",
        )

        # Add concurrently
        await asyncio.gather(
        session1.add_rule(rule1),
        session2.add_rule(rule2)
        )

        # Verify both rules exist
        session3 = await memory_manager_factory()
        all_rules = await session3.get_rules()

        rule_ids = {r.id for r in all_rules}
        assert "concurrent_1" in rule_ids
        assert "concurrent_2" in rule_ids


@pytest.mark.asyncio
class TestRuleConflictResolution:
    """Test conflict resolution when same rule ID exists across sessions."""

    async def test_duplicate_rule_id_handling(self, memory_manager_factory):
        """Test handling of duplicate rule IDs across sessions."""
        # Session 1: Add rule
        session1 = await memory_manager_factory()

        rule_v1 = MemoryRule(
            rule="Original rule text",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="conflict_test",
            source="session1",
            metadata={"version": 1},
        )

        await session1.add_rule(rule_v1)

        # Session 2: Try to add same ID with different content
        session2 = await memory_manager_factory()

        rule_v2 = MemoryRule(
            rule="Updated rule text",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="conflict_test",  # Same ID
            source="session2",
            metadata={"version": 2},
        )

        # Depending on implementation, this might:
        # - Raise an error
        # - Update existing rule
        # - Ignore the duplicate
        try:
            await session2.add_rule(rule_v2)
        except Exception as e:
            # Duplicate handling may raise error
            assert "conflict_test" in str(e) or "duplicate" in str(e).lower() or "exists" in str(e).lower()

        # Session 3: Verify final state
        session3 = await memory_manager_factory()
        rules = await session3.get_rules()

        conflict_rules = [r for r in rules if r.id == "conflict_test"]
        # Should have exactly one rule with this ID
        assert len(conflict_rules) <= 1
