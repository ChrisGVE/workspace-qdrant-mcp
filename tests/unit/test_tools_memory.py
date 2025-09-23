"""
Unit tests for workspace_qdrant_mcp.tools.memory.

Focused testing for memory tools MCP functionality to achieve 90%+ coverage.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Any, Dict, List
import pytest

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

# Import the actual module
from python.workspace_qdrant_mcp.tools.memory import register_memory_tools

# Mock FastMCP server for testing
class MockFastMCP:
    def __init__(self, name):
        self.name = name
        self._tools = {}

    def tool(self):
        def decorator(func):
            tool_mock = Mock()
            tool_mock.name = func.__name__
            tool_mock.fn = func
            self._tools[func.__name__] = tool_mock
            return func
        return decorator


@pytest.fixture
def mock_server():
    """Mock FastMCP server with registered tools."""
    server = MockFastMCP("test-server")
    register_memory_tools(server)
    return server


@pytest.fixture
def mock_memory_rule():
    """Create a mock memory rule."""
    from python.common.core.memory import MemoryRule, MemoryCategory, AuthorityLevel
    return MemoryRule(
        id="test-rule-1",
        category=MemoryCategory.PREFERENCE,
        name="test-rule",
        rule="Use pytest for testing",
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
        source="test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={}
    )


class TestMemoryToolsBasic:
    """Test basic memory tools functionality."""

    def test_register_memory_tools(self, mock_server):
        """Test that memory tools are properly registered."""
        expected_tools = [
            "initialize_memory_session",
            "add_memory_rule",
            "update_memory_from_conversation",
            "search_memory_rules",
            "get_memory_stats",
            "detect_memory_conflicts",
            "list_memory_rules",
            "apply_memory_context",
            "optimize_memory_tokens",
            "export_memory_profile"
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_server._tools, f"Tool {tool_name} not registered"

    @patch('python.common.core.config.Config')
    async def test_initialize_memory_session_error(self, mock_config_class, mock_server):
        """Test memory session initialization error handling."""
        mock_config_class.side_effect = Exception("Config error")

        tool_func = mock_server._tools["initialize_memory_session"].fn
        result = await tool_func()

        assert result["status"] == "error"
        assert "error" in result

    async def test_add_memory_rule_invalid_category(self, mock_server):
        """Test memory rule addition with invalid category."""
        tool_func = mock_server._tools["add_memory_rule"].fn

        result = await tool_func(
            category="invalid_category",
            name="test-rule",
            rule="Test rule",
            authority="default"
        )

        assert result["success"] is False
        assert "Invalid parameter" in result["error"]

    @patch('python.common.core.memory.parse_conversational_memory_update')
    async def test_update_memory_from_conversation_no_pattern(
        self, mock_parse_function, mock_server
    ):
        """Test conversational update with no pattern detected."""
        mock_parse_function.return_value = None

        tool_func = mock_server._tools["update_memory_from_conversation"].fn
        result = await tool_func("Just a regular message")

        assert result["detected"] is False
        assert "No memory update pattern detected" in result["message"]

    async def test_search_memory_rules_invalid_category(self, mock_server):
        """Test search with invalid category."""
        tool_func = mock_server._tools["search_memory_rules"].fn
        result = await tool_func(
            query="test query",
            category="invalid_category"
        )

        assert result["success"] is False
        assert "Invalid category" in result["error"]

    @patch('python.common.core.config.Config')
    async def test_get_memory_stats_error(self, mock_config_class, mock_server):
        """Test memory stats error handling."""
        mock_config_class.side_effect = Exception("Stats error")

        tool_func = mock_server._tools["get_memory_stats"].fn
        result = await tool_func()

        assert "error" in result
        assert "Stats error" in result["error"]

    async def test_search_memory_rules_invalid_authority(self, mock_server):
        """Test search with invalid authority parameter."""
        tool_func = mock_server._tools["search_memory_rules"].fn
        result = await tool_func(
            query="test query",
            authority="invalid_authority"
        )

        assert result["success"] is False
        assert "Invalid authority" in result["error"]

    async def test_list_memory_rules_invalid_category(self, mock_server):
        """Test list rules with invalid category."""
        tool_func = mock_server._tools["list_memory_rules"].fn
        result = await tool_func(category="invalid_category")

        assert result["success"] is False
        assert "Invalid category" in result["error"]

    async def test_list_memory_rules_invalid_authority(self, mock_server):
        """Test list rules with invalid authority."""
        tool_func = mock_server._tools["list_memory_rules"].fn
        result = await tool_func(authority="invalid_authority")

        assert result["success"] is False
        assert "Invalid authority" in result["error"]

    async def test_add_memory_rule_invalid_authority(self, mock_server):
        """Test add rule with invalid authority."""
        tool_func = mock_server._tools["add_memory_rule"].fn
        result = await tool_func(
            category="preference",
            name="test-rule",
            rule="Test rule",
            authority="invalid_authority"
        )

        assert result["success"] is False
        assert "Invalid parameter" in result["error"]

    @patch('python.common.core.config.Config')
    async def test_tool_generic_error_handling(self, mock_config_class, mock_server):
        """Test generic error handling across tools."""
        mock_config_class.side_effect = Exception("Database connection failed")

        # Test multiple tools for error handling
        tool_names = ["get_memory_stats", "detect_memory_conflicts", "export_memory_profile"]

        for tool_name in tool_names:
            tool_func = mock_server._tools[tool_name].fn
            result = await tool_func()
            assert "error" in result
            assert "Database connection failed" in result["error"]

    async def test_export_memory_profile_error(self, mock_server):
        """Test memory profile export with error."""
        with patch('python.common.core.config.Config') as mock_config_class:
            mock_config_class.side_effect = Exception("Export error")

            tool_func = mock_server._tools["export_memory_profile"].fn
            result = await tool_func()

            assert result["success"] is False
            assert "error" in result
            assert "Export error" in result["error"]


class TestMemoryToolsComprehensive:
    """Test comprehensive memory tools functionality with working mocks."""

    @patch('python.common.core.config.Config')
    @patch('python.common.core.client.create_qdrant_client')
    @patch('python.common.core.collection_naming.create_naming_manager')
    @patch('python.common.core.memory.MemoryManager')
    async def test_working_initialize_memory_session(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class, mock_server, mock_memory_rule
    ):
        """Test successful memory session initialization."""
        from python.common.core.memory import MemoryCategory, AuthorityLevel

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Create mock stats
        mock_stats = Mock()
        mock_stats.total_rules = 1
        mock_stats.rules_by_category = {MemoryCategory.PREFERENCE: 1}
        mock_stats.rules_by_authority = {AuthorityLevel.DEFAULT: 1}
        mock_stats.estimated_tokens = 500
        mock_stats.last_optimization = datetime.now(timezone.utc)

        # Create mock conflict
        mock_conflict = Mock()
        mock_conflict.conflict_type = "semantic_conflict"
        mock_conflict.description = "Conflicting rules"
        mock_conflict.rule1 = Mock(name="rule1")
        mock_conflict.rule2 = Mock(name="rule2")
        mock_conflict.confidence = 0.85
        mock_conflict.resolution_options = ["merge", "prioritize"]

        mock_memory_manager = AsyncMock()
        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager.list_memory_rules.return_value = [mock_memory_rule]
        mock_memory_manager.detect_conflicts.return_value = [mock_conflict]  # Test conflicts path
        mock_memory_manager.get_memory_stats.return_value = mock_stats
        mock_memory_manager_class.return_value = mock_memory_manager

        # Get tool function
        tool_func = mock_server._tools["initialize_memory_session"].fn

        # Execute
        result = await tool_func()

        # Verify - this should trigger the conflicts branch (lines 107-118)
        assert result["status"] == "conflicts_require_resolution"
        assert result["conflicts_detected"] == 1
        assert "conflicts" in result
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["type"] == "semantic_conflict"

    @patch('python.common.core.config.Config')
    @patch('python.common.core.client.create_qdrant_client')
    @patch('python.common.core.collection_naming.create_naming_manager')
    @patch('python.common.core.memory.MemoryManager')
    @patch('python.common.core.memory.parse_conversational_memory_update')
    async def test_working_conversational_update(
        self, mock_parse_function, mock_memory_manager_class,
        mock_naming_manager, mock_qdrant_client, mock_config_class, mock_server
    ):
        """Test successful conversational memory update."""
        from python.common.core.memory import MemoryCategory, AuthorityLevel

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        mock_parse_function.return_value = {
            "category": MemoryCategory.PREFERENCE,
            "rule": "Call me Alex",
            "authority": AuthorityLevel.DEFAULT,
            "source": "conversational"
        }

        mock_memory_manager = AsyncMock()
        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager.add_memory_rule.return_value = "conv-rule-123"
        mock_memory_manager_class.return_value = mock_memory_manager

        tool_func = mock_server._tools["update_memory_from_conversation"].fn
        result = await tool_func("Note: call me Alex")

        # Should trigger lines 258-259 (logger.info and return)
        assert result["detected"] is True
        assert result["rule_added"] is True
        assert result["rule_id"] == "conv-rule-123"
        assert result["category"] == "preference"

    @patch('python.common.core.config.Config')
    @patch('python.common.core.client.create_qdrant_client')
    @patch('python.common.core.collection_naming.create_naming_manager')
    @patch('python.common.core.memory.MemoryManager')
    async def test_working_memory_operations(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class, mock_server, mock_memory_rule
    ):
        """Test various memory operations to improve coverage."""
        from python.common.core.memory import MemoryCategory, AuthorityLevel

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Mock stats
        mock_stats = Mock()
        mock_stats.total_rules = 5
        mock_stats.rules_by_category = {MemoryCategory.PREFERENCE: 2}
        mock_stats.rules_by_authority = {AuthorityLevel.DEFAULT: 4}
        mock_stats.estimated_tokens = 500
        mock_stats.last_optimization = datetime.now(timezone.utc)

        mock_memory_manager = AsyncMock()
        mock_memory_manager.search_memory_rules.return_value = [(mock_memory_rule, 0.95)]
        mock_memory_manager.get_memory_stats.return_value = mock_stats
        mock_memory_manager.list_memory_rules.return_value = [mock_memory_rule]
        mock_memory_manager.add_memory_rule.return_value = "rule-123"
        mock_memory_manager.detect_conflicts.return_value = []
        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager_class.return_value = mock_memory_manager

        # Test multiple operations
        search_func = mock_server._tools["search_memory_rules"].fn
        search_result = await search_func(query="python", category="preference")
        assert search_result["success"] is True

        stats_func = mock_server._tools["get_memory_stats"].fn
        stats_result = await stats_func()
        assert stats_result["total_rules"] == 5

        list_func = mock_server._tools["list_memory_rules"].fn
        list_result = await list_func(limit=10)
        assert list_result["success"] is True

        add_func = mock_server._tools["add_memory_rule"].fn
        add_result = await add_func(
            category="preference",
            name="test-rule",
            rule="Use pytest",
            authority="default"
        )
        assert add_result["success"] is True

    @patch('python.common.core.config.Config')
    @patch('python.common.core.client.create_qdrant_client')
    @patch('python.common.core.collection_naming.create_naming_manager')
    @patch('python.common.core.memory.MemoryManager')
    async def test_memory_optimization_paths(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class, mock_server
    ):
        """Test memory optimization to hit different paths."""
        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Test no optimization needed path
        low_stats = Mock()
        low_stats.estimated_tokens = 500

        mock_memory_manager = AsyncMock()
        mock_memory_manager.get_memory_stats.return_value = low_stats
        mock_memory_manager_class.return_value = mock_memory_manager

        optimize_func = mock_server._tools["optimize_memory_tokens"].fn
        result = await optimize_func(max_tokens=2000)

        assert result["optimization_needed"] is False
        assert result["current_tokens"] == 500

        # Test optimization needed path
        high_stats = Mock()
        high_stats.estimated_tokens = 3000
        low_stats_after = Mock()
        low_stats_after.estimated_tokens = 1500

        mock_memory_manager.get_memory_stats.side_effect = [high_stats, low_stats_after]
        mock_memory_manager.optimize_memory.return_value = (1500, ["removed_duplicates"])

        result2 = await optimize_func(max_tokens=2000)

        assert result2["optimization_needed"] is True
        assert result2["optimization_completed"] is True
        assert result2["tokens_saved"] == 1500

    @patch('python.common.core.config.Config')
    @patch('python.common.core.client.create_qdrant_client')
    @patch('python.common.core.collection_naming.create_naming_manager')
    @patch('python.common.core.memory.MemoryManager')
    async def test_apply_memory_context_comprehensive(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class, mock_server
    ):
        """Test apply memory context to hit different rule paths."""
        from python.common.core.memory import MemoryCategory, AuthorityLevel

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Create different types of rules
        absolute_rule = Mock()
        absolute_rule.id = "abs-1"
        absolute_rule.name = "absolute"
        absolute_rule.rule = "Always commit"
        absolute_rule.category = MemoryCategory.BEHAVIOR
        absolute_rule.authority = AuthorityLevel.ABSOLUTE
        absolute_rule.scope = ["git"]

        preference_rule = Mock()
        preference_rule.id = "pref-1"
        preference_rule.name = "preference"
        preference_rule.rule = "Use TypeScript"
        preference_rule.category = MemoryCategory.PREFERENCE
        preference_rule.authority = AuthorityLevel.DEFAULT
        preference_rule.scope = ["frontend"]

        behavioral_rule = Mock()
        behavioral_rule.id = "behavior-1"
        behavioral_rule.name = "behavior"
        behavioral_rule.rule = "Test first"
        behavioral_rule.category = MemoryCategory.BEHAVIOR
        behavioral_rule.authority = AuthorityLevel.DEFAULT
        behavioral_rule.scope = ["testing"]

        mock_memory_manager = AsyncMock()
        # Return high relevance rules to trigger different categorization paths
        mock_memory_manager.search_memory_rules.return_value = [
            (preference_rule, 0.8),  # High enough to trigger preference path
            (behavioral_rule, 0.75)   # High enough to trigger contextual path
        ]
        mock_memory_manager.list_memory_rules.return_value = [absolute_rule]
        mock_memory_manager_class.return_value = mock_memory_manager

        apply_func = mock_server._tools["apply_memory_context"].fn
        result = await apply_func(
            task_description="Implement feature",
            project_context="Frontend project"
        )

        assert result["success"] is True
        assert result["memory_applied"] is True
        # Should have triggered all rule categorization paths (lines 604-617)
        assert "applicable_rules" in result
        assert "absolute" in result["applicable_rules"]
        assert "contextual" in result["applicable_rules"]
        assert "preferences" in result["applicable_rules"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])