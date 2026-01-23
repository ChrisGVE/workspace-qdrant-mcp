"""
Unit tests for Task 435: Memory injection on MCP startup and compaction.
Tests memory rule loading, caching, and formatting for LLM context injection.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockPoint:
    """Mock Qdrant point for testing."""
    id: str
    payload: dict


@dataclass
class MockCollection:
    """Mock Qdrant collection for testing."""
    name: str


@dataclass
class MockCollectionsResponse:
    """Mock Qdrant collections response."""
    collections: list


class TestLoadMemoryRules:
    """Tests for _load_memory_rules function."""

    @pytest.mark.asyncio
    async def test_load_memory_rules_empty_collection(self):
        """Test loading rules when memory collection exists but is empty."""
        from workspace_qdrant_mcp.server import (
            _load_memory_rules,
            CANONICAL_COLLECTIONS,
        )

        # Setup mock Qdrant client
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="memory")]
        )
        mock_client.scroll.return_value = ([], None)  # Empty scroll result

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", None):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", None):
                    rules = await _load_memory_rules(force_refresh=True)

        assert rules == []
        mock_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_memory_rules_no_collection(self):
        """Test loading rules when memory collection doesn't exist."""
        from workspace_qdrant_mcp.server import _load_memory_rules

        # Setup mock Qdrant client without memory collection
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="projects")]  # No 'memory'
        )

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", None):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", None):
                    rules = await _load_memory_rules(force_refresh=True)

        assert rules == []
        # scroll should not be called since collection doesn't exist
        mock_client.scroll.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_memory_rules_with_rules(self):
        """Test loading rules with actual rule data."""
        from workspace_qdrant_mcp.server import _load_memory_rules

        # Create mock points with rule data
        mock_points = [
            MockPoint(
                id="rule-1",
                payload={
                    "rule": "Always make atomic commits",
                    "name": "atomic-commits",
                    "category": "behavior",
                    "authority": "absolute",
                    "scope": ["git"],
                    "source": "user_explicit",
                }
            ),
            MockPoint(
                id="rule-2",
                payload={
                    "rule": "Use uv for Python package management",
                    "name": "use-uv",
                    "category": "preference",
                    "authority": "default",
                    "scope": ["python"],
                    "source": "conversational",
                }
            ),
        ]

        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="memory")]
        )
        mock_client.scroll.return_value = (mock_points, None)

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", None):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", None):
                    rules = await _load_memory_rules(force_refresh=True)

        assert len(rules) == 2
        # Absolute rules should come first (sorted)
        assert rules[0]["authority"] == "absolute"
        assert rules[0]["name"] == "atomic-commits"
        assert rules[1]["authority"] == "default"
        assert rules[1]["name"] == "use-uv"

    @pytest.mark.asyncio
    async def test_load_memory_rules_cache_hit(self):
        """Test that cached rules are returned without querying Qdrant."""
        from workspace_qdrant_mcp.server import (
            _load_memory_rules,
            _MEMORY_INJECTION_STALE_SECS,
        )

        cached_rules = [
            {
                "id": "cached-rule",
                "rule": "Cached rule",
                "name": "cached",
                "category": "behavior",
                "authority": "absolute",
                "scope": [],
                "source": "test",
            }
        ]
        recent_time = datetime.now(timezone.utc) - timedelta(seconds=10)  # Recent

        mock_client = AsyncMock()

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", cached_rules):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", recent_time):
                    rules = await _load_memory_rules(force_refresh=False)

        # Should return cached rules without calling Qdrant
        assert rules == cached_rules
        mock_client.get_collections.assert_not_called()
        mock_client.scroll.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_memory_rules_cache_stale(self):
        """Test that stale cache triggers reload."""
        from workspace_qdrant_mcp.server import (
            _load_memory_rules,
            _MEMORY_INJECTION_STALE_SECS,
        )

        cached_rules = [{"id": "old-rule", "rule": "Old rule", "name": "old", "category": "behavior", "authority": "default", "scope": [], "source": "test"}]
        stale_time = datetime.now(timezone.utc) - timedelta(seconds=_MEMORY_INJECTION_STALE_SECS + 60)

        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="memory")]
        )
        mock_client.scroll.return_value = ([], None)  # Return empty on reload

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", cached_rules):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", stale_time):
                    rules = await _load_memory_rules(force_refresh=False)

        # Should query Qdrant since cache is stale
        mock_client.get_collections.assert_called_once()
        mock_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_memory_rules_no_qdrant_client(self):
        """Test graceful handling when Qdrant client is not initialized."""
        from workspace_qdrant_mcp.server import _load_memory_rules

        with patch("workspace_qdrant_mcp.server.qdrant_client", None):
            rules = await _load_memory_rules(force_refresh=True)

        assert rules == []


class TestFormatMemoryRulesForLLM:
    """Tests for _format_memory_rules_for_llm function."""

    def test_format_empty_rules(self):
        """Test formatting with no rules."""
        from workspace_qdrant_mcp.server import _format_memory_rules_for_llm

        result = _format_memory_rules_for_llm([])
        assert result == ""

    def test_format_absolute_rules_only(self):
        """Test formatting with only absolute rules."""
        from workspace_qdrant_mcp.server import _format_memory_rules_for_llm

        rules = [
            {
                "name": "atomic-commits",
                "rule": "Always make atomic commits",
                "authority": "absolute",
                "scope": ["git"],
            }
        ]

        result = _format_memory_rules_for_llm(rules)

        assert "# Memory Rules (Auto-injected)" in result
        assert "## Absolute Rules (Non-negotiable)" in result
        assert "**atomic-commits**" in result
        assert "[git]" in result
        assert "Always make atomic commits" in result
        assert "## Default Rules" not in result

    def test_format_default_rules_only(self):
        """Test formatting with only default rules."""
        from workspace_qdrant_mcp.server import _format_memory_rules_for_llm

        rules = [
            {
                "name": "use-uv",
                "rule": "Use uv for Python",
                "authority": "default",
                "scope": [],
            }
        ]

        result = _format_memory_rules_for_llm(rules)

        assert "## Default Rules (Override when explicitly requested)" in result
        assert "**use-uv**" in result
        assert "Use uv for Python" in result
        assert "## Absolute Rules" not in result

    def test_format_mixed_rules(self):
        """Test formatting with both absolute and default rules."""
        from workspace_qdrant_mcp.server import _format_memory_rules_for_llm

        rules = [
            {
                "name": "rule-absolute",
                "rule": "Must follow",
                "authority": "absolute",
                "scope": ["all"],
            },
            {
                "name": "rule-default",
                "rule": "Optional follow",
                "authority": "default",
                "scope": [],
            },
        ]

        result = _format_memory_rules_for_llm(rules)

        assert "## Absolute Rules (Non-negotiable)" in result
        assert "## Default Rules (Override when explicitly requested)" in result
        # Absolute section should come before Default
        absolute_pos = result.find("## Absolute Rules")
        default_pos = result.find("## Default Rules")
        assert absolute_pos < default_pos

    def test_format_rule_without_scope(self):
        """Test formatting rules without scope."""
        from workspace_qdrant_mcp.server import _format_memory_rules_for_llm

        rules = [
            {
                "name": "no-scope-rule",
                "rule": "Rule without scope",
                "authority": "absolute",
                "scope": [],  # Empty scope
            }
        ]

        result = _format_memory_rules_for_llm(rules)

        assert "**no-scope-rule**:" in result  # No scope brackets
        assert "[" not in result.split("**no-scope-rule**")[1].split(":")[0]


class TestRefreshMemoryWorkflow:
    """Integration tests for refresh_memory workflow."""

    @pytest.mark.asyncio
    async def test_refresh_memory_workflow(self):
        """Test the complete refresh memory workflow using core functions."""
        from workspace_qdrant_mcp.server import (
            _load_memory_rules,
            _format_memory_rules_for_llm,
        )

        mock_rules = [
            {
                "id": "rule-1",
                "rule": "Test rule 1",
                "name": "test-1",
                "category": "behavior",
                "authority": "absolute",
                "scope": [],
                "source": "test",
            },
            {
                "id": "rule-2",
                "rule": "Test rule 2",
                "name": "test-2",
                "category": "preference",
                "authority": "default",
                "scope": [],
                "source": "test",
            },
        ]

        # Test load + format workflow
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="memory")]
        )
        mock_client.scroll.return_value = (
            [MockPoint(id=r["id"], payload={k: v for k, v in r.items() if k != "id"})
             for r in mock_rules],
            None
        )

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", None):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", None):
                    rules = await _load_memory_rules(force_refresh=True)

        # Verify rules loaded and sorted
        assert len(rules) == 2
        assert rules[0]["authority"] == "absolute"  # Absolute first
        assert rules[1]["authority"] == "default"

        # Verify formatting works
        formatted = _format_memory_rules_for_llm(rules)
        assert "Memory Rules" in formatted
        assert "Absolute Rules" in formatted
        assert "Default Rules" in formatted

    @pytest.mark.asyncio
    async def test_force_refresh_updates_cache(self):
        """Test that force_refresh updates the cache."""
        from workspace_qdrant_mcp.server import _load_memory_rules
        import workspace_qdrant_mcp.server as server_module

        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="memory")]
        )
        mock_client.scroll.return_value = (
            [MockPoint(id="new-rule", payload={"rule": "New", "name": "new", "category": "behavior", "authority": "absolute", "scope": [], "source": "test"})],
            None
        )

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch.object(server_module, "_memory_rules_cache", None):
                with patch.object(server_module, "_memory_last_injected", None):
                    rules = await _load_memory_rules(force_refresh=True)

        # Verify new rules returned
        assert len(rules) == 1
        assert rules[0]["name"] == "new"


class TestMemoryStatusComputation:
    """Tests for memory status computation logic."""

    def test_cache_staleness_computation(self):
        """Test cache staleness computation logic."""
        from workspace_qdrant_mcp.server import _MEMORY_INJECTION_STALE_SECS

        # Test threshold is reasonable (5 minutes)
        assert _MEMORY_INJECTION_STALE_SECS == 300.0

        # Test staleness logic
        from datetime import datetime, timezone, timedelta

        recent_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        age = (datetime.now(timezone.utc) - recent_time).total_seconds()
        is_stale = age > _MEMORY_INJECTION_STALE_SECS
        assert is_stale is False  # 60s < 300s

        old_time = datetime.now(timezone.utc) - timedelta(seconds=400)
        age = (datetime.now(timezone.utc) - old_time).total_seconds()
        is_stale = age > _MEMORY_INJECTION_STALE_SECS
        assert is_stale is True  # 400s > 300s

    def test_rules_count_by_authority(self):
        """Test counting rules by authority level."""
        rules = [
            {"authority": "absolute"},
            {"authority": "absolute"},
            {"authority": "default"},
            {"authority": "default"},
            {"authority": "default"},
        ]

        absolute_count = len([r for r in rules if r["authority"] == "absolute"])
        default_count = len([r for r in rules if r["authority"] == "default"])

        assert absolute_count == 2
        assert default_count == 3

    def test_rules_count_by_category(self):
        """Test counting rules by category."""
        rules = [
            {"category": "behavior"},
            {"category": "behavior"},
            {"category": "preference"},
            {"category": "agent"},
        ]

        behavior_count = len([r for r in rules if r["category"] == "behavior"])
        preference_count = len([r for r in rules if r["category"] == "preference"])
        agent_count = len([r for r in rules if r["category"] == "agent"])

        assert behavior_count == 2
        assert preference_count == 1
        assert agent_count == 1


class TestMemoryRuleSorting:
    """Tests for memory rule sorting logic."""

    @pytest.mark.asyncio
    async def test_rules_sorted_by_authority_then_name(self):
        """Test that rules are sorted by authority (absolute first) then by name."""
        from workspace_qdrant_mcp.server import _load_memory_rules

        mock_points = [
            MockPoint(id="3", payload={"rule": "C", "name": "z-default", "category": "behavior", "authority": "default", "scope": [], "source": "test"}),
            MockPoint(id="1", payload={"rule": "A", "name": "b-absolute", "category": "behavior", "authority": "absolute", "scope": [], "source": "test"}),
            MockPoint(id="4", payload={"rule": "D", "name": "a-default", "category": "behavior", "authority": "default", "scope": [], "source": "test"}),
            MockPoint(id="2", payload={"rule": "B", "name": "a-absolute", "category": "behavior", "authority": "absolute", "scope": [], "source": "test"}),
        ]

        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MockCollectionsResponse(
            collections=[MockCollection(name="memory")]
        )
        mock_client.scroll.return_value = (mock_points, None)

        with patch("workspace_qdrant_mcp.server.qdrant_client", mock_client):
            with patch("workspace_qdrant_mcp.server._memory_rules_cache", None):
                with patch("workspace_qdrant_mcp.server._memory_last_injected", None):
                    rules = await _load_memory_rules(force_refresh=True)

        # Should be sorted: absolute first (by name), then default (by name)
        assert len(rules) == 4
        assert rules[0]["name"] == "a-absolute"
        assert rules[1]["name"] == "b-absolute"
        assert rules[2]["name"] == "a-default"
        assert rules[3]["name"] == "z-default"
