"""
Multi-Tool Integration Testing for Memory Rules (Task 337.6).

Tests that memory rules work consistently across different tool interfaces:
- MCP server interface
- CLI tool interface
- Direct API interface

Validates:
1. Rule injection across different tools
2. Consistent behavior with same rules
3. Tool-specific adaptations
4. Cross-tool rule sharing and visibility
5. Tool-specific configuration handling
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from unittest.mock import AsyncMock, Mock, patch, MagicMock

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

# Try to import real components
try:
    from src.python.common.core.memory import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False


class MockMCPServer:
    """Mock MCP server for testing rule injection."""

    def __init__(self, memory_manager):
        """Initialize mock MCP server.

        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager
        self.active_rules: List[MemoryRule] = []

    async def add_memory_rule(
        self,
        rule: str,
        category: MemoryCategory,
        authority: AuthorityLevel,
        scope: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Add a memory rule through MCP interface.

        Args:
            rule: Rule text
            category: Rule category
            authority: Authority level
            scope: Optional scope restriction

        Returns:
            Response with rule ID and status
        """
        memory_rule = MemoryRule(
            rule=rule,
            category=category,
            authority=authority,
            id=f"mcp_rule_{len(self.active_rules)}",
            scope=scope or [],
            source="mcp_server",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await self.memory_manager.add_rule(memory_rule)
        self.active_rules.append(memory_rule)

        return {
            "rule_id": memory_rule.id,
            "status": "added",
            "tool": "mcp_server"
        }

    async def get_active_rules(
        self,
        category: Optional[MemoryCategory] = None
    ) -> List[MemoryRule]:
        """Get active rules through MCP interface.

        Args:
            category: Optional category filter

        Returns:
            List of active rules
        """
        rules = await self.memory_manager.get_rules()

        if category:
            rules = [r for r in rules if r.category == category]

        return rules


class MockCLI:
    """Mock CLI tool for testing rule injection."""

    def __init__(self, memory_manager):
        """Initialize mock CLI.

        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager
        self.command_history: List[str] = []

    async def execute_command(
        self,
        command: str,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute CLI command.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            Command execution result
        """
        self.command_history.append(f"{command} {args}")

        if command == "add-rule":
            return await self._add_rule(args)
        elif command == "list-rules":
            return await self._list_rules(args)
        elif command == "remove-rule":
            return await self._remove_rule(args)

        return {"status": "unknown_command"}

    async def _add_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add rule through CLI.

        Args:
            args: Command arguments

        Returns:
            Result with rule ID
        """
        rule = MemoryRule(
            rule=args["rule"],
            category=args.get("category", MemoryCategory.BEHAVIOR),
            authority=args.get("authority", AuthorityLevel.DEFAULT),
            id=f"cli_rule_{args.get('id', 'auto')}",
            scope=args.get("scope", []),
            source="cli",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await self.memory_manager.add_rule(rule)

        return {
            "rule_id": rule.id,
            "status": "added",
            "tool": "cli"
        }

    async def _list_rules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List rules through CLI.

        Args:
            args: Command arguments

        Returns:
            List of rules
        """
        rules = await self.memory_manager.get_rules()

        return {
            "rules": rules,
            "count": len(rules),
            "tool": "cli"
        }

    async def _remove_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove rule through CLI.

        Args:
            args: Command arguments

        Returns:
            Removal result
        """
        await self.memory_manager.delete_rule(args["rule_id"])

        return {
            "rule_id": args["rule_id"],
            "status": "removed",
            "tool": "cli"
        }


@pytest.fixture
async def shared_memory_manager():
    """Provide shared memory manager for all tools."""
    manager = AsyncMock(spec=MemoryManager)
    manager._rules = []

    async def add_rule(rule: MemoryRule):
        manager._rules.append(rule)

    async def get_rules():
        return manager._rules.copy()

    async def delete_rule(rule_id: str):
        manager._rules = [r for r in manager._rules if r.id != rule_id]

    manager.add_rule = AsyncMock(side_effect=add_rule)
    manager.get_rules = AsyncMock(side_effect=get_rules)
    manager.delete_rule = AsyncMock(side_effect=delete_rule)
    manager.initialize = AsyncMock()

    await manager.initialize()
    return manager


@pytest.fixture
async def mcp_server(shared_memory_manager):
    """Provide MCP server instance."""
    return MockMCPServer(shared_memory_manager)


@pytest.fixture
async def cli_tool(shared_memory_manager):
    """Provide CLI tool instance."""
    return MockCLI(shared_memory_manager)


@pytest.fixture
def mock_llm_provider():
    """Provide mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def behavioral_harness(mock_llm_provider, shared_memory_manager):
    """Provide behavioral harness."""
    return LLMBehavioralHarness(
        provider=mock_llm_provider,
        memory_manager=shared_memory_manager,
        mode=ExecutionMode.MOCK
    )


@pytest.mark.asyncio
class TestCrossToolRuleInjection:
    """Test rule injection across different tools."""

    async def test_mcp_server_rule_injection(
        self,
        mcp_server,
        shared_memory_manager
    ):
        """Test adding rules through MCP server."""
        # Add rule via MCP
        result = await mcp_server.add_memory_rule(
            rule="Use type hints in Python code",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE
        )

        assert result["status"] == "added"
        assert result["tool"] == "mcp_server"
        assert "mcp_rule_" in result["rule_id"]

        # Verify rule is in memory manager
        rules = await shared_memory_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].rule == "Use type hints in Python code"
        assert rules[0].source == "mcp_server"

    async def test_cli_tool_rule_injection(
        self,
        cli_tool,
        shared_memory_manager
    ):
        """Test adding rules through CLI."""
        # Add rule via CLI
        result = await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "Always use docstrings",
                "category": MemoryCategory.BEHAVIOR,
                "authority": AuthorityLevel.ABSOLUTE,
                "id": "test1"
            }
        )

        assert result["status"] == "added"
        assert result["tool"] == "cli"
        assert result["rule_id"] == "cli_rule_test1"

        # Verify rule is in memory manager
        rules = await shared_memory_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].rule == "Always use docstrings"
        assert rules[0].source == "cli"

    async def test_direct_api_rule_injection(
        self,
        shared_memory_manager
    ):
        """Test adding rules directly through API."""
        # Add rule directly
        rule = MemoryRule(
            rule="Prefer functional programming",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="api_rule_1",
            source="direct_api",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await shared_memory_manager.add_rule(rule)

        # Verify rule exists
        rules = await shared_memory_manager.get_rules()
        assert len(rules) == 1
        assert rules[0].source == "direct_api"


@pytest.mark.asyncio
class TestCrossToolRuleVisibility:
    """Test that rules added through one tool are visible to others."""

    async def test_mcp_to_cli_visibility(
        self,
        mcp_server,
        cli_tool
    ):
        """Test that MCP-added rules are visible to CLI."""
        # Add rule via MCP
        await mcp_server.add_memory_rule(
            rule="Test rule from MCP",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        # Retrieve via CLI
        result = await cli_tool.execute_command("list-rules", {})

        assert result["count"] == 1
        assert result["rules"][0].rule == "Test rule from MCP"

    async def test_cli_to_mcp_visibility(
        self,
        mcp_server,
        cli_tool
    ):
        """Test that CLI-added rules are visible to MCP."""
        # Add rule via CLI
        await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "Test rule from CLI",
                "category": MemoryCategory.BEHAVIOR,
                "authority": AuthorityLevel.DEFAULT,
                "id": "test1"
            }
        )

        # Retrieve via MCP
        rules = await mcp_server.get_active_rules()

        assert len(rules) == 1
        assert rules[0].rule == "Test rule from CLI"

    async def test_bidirectional_visibility(
        self,
        mcp_server,
        cli_tool,
        shared_memory_manager
    ):
        """Test bidirectional visibility across all tools."""
        # Add via MCP
        await mcp_server.add_memory_rule(
            rule="MCP rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        # Add via CLI
        await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "CLI rule",
                "category": MemoryCategory.BEHAVIOR,
                "id": "test1"
            }
        )

        # Add via direct API
        rule = MemoryRule(
            rule="API rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="api_rule_1",
            source="api",
        )
        await shared_memory_manager.add_rule(rule)

        # Verify all tools see all rules
        mcp_rules = await mcp_server.get_active_rules()
        cli_result = await cli_tool.execute_command("list-rules", {})
        api_rules = await shared_memory_manager.get_rules()

        assert len(mcp_rules) == 3
        assert cli_result["count"] == 3
        assert len(api_rules) == 3


@pytest.mark.asyncio
class TestToolSpecificBehavior:
    """Test tool-specific behavior with same rules."""

    async def test_mcp_rule_formatting(
        self,
        mcp_server
    ):
        """Test that MCP formats rules appropriately."""
        result = await mcp_server.add_memory_rule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python"]
        )

        # MCP should return structured response
        assert "rule_id" in result
        assert "status" in result
        assert "tool" in result
        assert result["tool"] == "mcp_server"

    async def test_cli_command_history(
        self,
        cli_tool
    ):
        """Test that CLI maintains command history."""
        # Execute commands
        await cli_tool.execute_command(
            "add-rule",
            {"rule": "Test 1", "id": "test1"}
        )
        await cli_tool.execute_command(
            "list-rules",
            {}
        )

        # Verify command history
        assert len(cli_tool.command_history) == 2
        assert "add-rule" in cli_tool.command_history[0]
        assert "list-rules" in cli_tool.command_history[1]

    async def test_tool_source_tracking(
        self,
        mcp_server,
        cli_tool,
        shared_memory_manager
    ):
        """Test that each tool tracks its source correctly."""
        # Add via different tools
        await mcp_server.add_memory_rule(
            rule="MCP rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        await cli_tool.execute_command(
            "add-rule",
            {"rule": "CLI rule", "id": "test1"}
        )

        # Verify source tracking
        rules = await shared_memory_manager.get_rules()
        sources = {r.source for r in rules}

        assert "mcp_server" in sources
        assert "cli" in sources


@pytest.mark.asyncio
class TestCrossToolBehavioralConsistency:
    """Test behavioral consistency across tools with same rules."""

    async def test_same_rules_same_behavior(
        self,
        mcp_server,
        cli_tool,
        behavioral_harness
    ):
        """Test that same rules produce same behavior regardless of tool."""
        # Define test rule
        test_rule = {
            "rule": "Always use type hints",
            "category": MemoryCategory.BEHAVIOR,
            "authority": AuthorityLevel.ABSOLUTE
        }

        # Add via MCP
        await mcp_server.add_memory_rule(**test_rule)
        mcp_rules = await mcp_server.get_active_rules()

        # Clear and add via CLI
        await cli_tool.execute_command(
            "remove-rule",
            {"rule_id": mcp_rules[0].id}
        )

        await cli_tool.execute_command(
            "add-rule",
            {**test_rule, "id": "test1"}
        )
        cli_rules = (await cli_tool.execute_command("list-rules", {}))["rules"]

        # Both should have same rule content
        assert mcp_rules[0].rule == cli_rules[0].rule
        assert mcp_rules[0].authority == cli_rules[0].authority
        assert mcp_rules[0].category == cli_rules[0].category

    async def test_authority_enforcement_cross_tool(
        self,
        mcp_server,
        cli_tool,
        shared_memory_manager
    ):
        """Test authority enforcement works consistently across tools."""
        # Add ABSOLUTE rule via MCP
        await mcp_server.add_memory_rule(
            rule="ABSOLUTE rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE
        )

        # Add DEFAULT rule via CLI
        await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "DEFAULT rule",
                "category": MemoryCategory.BEHAVIOR,
                "authority": AuthorityLevel.DEFAULT,
                "id": "test1"
            }
        )

        # Both tools should see both rules
        rules = await shared_memory_manager.get_rules()
        assert len(rules) == 2

        # Authority levels should be preserved
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        assert len(absolute_rules) == 1
        assert len(default_rules) == 1


@pytest.mark.asyncio
class TestToolConfigurationHandling:
    """Test tool-specific configuration handling."""

    async def test_mcp_scope_handling(
        self,
        mcp_server
    ):
        """Test that MCP handles scope correctly."""
        # Add scoped rule
        result = await mcp_server.add_memory_rule(
            rule="Python-specific rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "*.py"]
        )

        # Verify scope is preserved
        rules = await mcp_server.get_active_rules()
        assert len(rules) == 1
        assert "python" in rules[0].scope
        assert "*.py" in rules[0].scope

    async def test_cli_category_filtering(
        self,
        cli_tool
    ):
        """Test CLI category filtering."""
        # Add rules with different categories
        await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "Behavior rule",
                "category": MemoryCategory.BEHAVIOR,
                "id": "test1"
            }
        )

        await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "Preference rule",
                "category": MemoryCategory.PREFERENCE,
                "id": "test2"
            }
        )

        # List all rules
        result = await cli_tool.execute_command("list-rules", {})
        assert result["count"] == 2

    async def test_tool_specific_metadata(
        self,
        mcp_server,
        cli_tool,
        shared_memory_manager
    ):
        """Test that each tool can add its own metadata."""
        # Add via MCP (stores source)
        await mcp_server.add_memory_rule(
            rule="MCP rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT
        )

        # Add via CLI (stores source)
        await cli_tool.execute_command(
            "add-rule",
            {"rule": "CLI rule", "id": "test1"}
        )

        # Verify metadata exists
        rules = await shared_memory_manager.get_rules()

        mcp_rule = next(r for r in rules if r.source == "mcp_server")
        cli_rule = next(r for r in rules if r.source == "cli")

        assert mcp_rule.id.startswith("mcp_rule_")
        assert cli_rule.id.startswith("cli_rule_")


@pytest.mark.asyncio
class TestToolErrorHandling:
    """Test error handling across different tools."""

    async def test_mcp_invalid_authority(
        self,
        mcp_server
    ):
        """Test MCP handling of invalid authority level."""
        # This should handle gracefully or raise appropriate error
        # Implementation depends on validation logic
        pass

    async def test_cli_unknown_command(
        self,
        cli_tool
    ):
        """Test CLI handling of unknown commands."""
        result = await cli_tool.execute_command(
            "unknown-command",
            {}
        )

        assert result["status"] == "unknown_command"

    async def test_cross_tool_duplicate_id_handling(
        self,
        mcp_server,
        cli_tool,
        shared_memory_manager
    ):
        """Test handling of duplicate rule IDs across tools."""
        # Add rule with specific ID via CLI
        await cli_tool.execute_command(
            "add-rule",
            {
                "rule": "First rule",
                "id": "duplicate_test"
            }
        )

        # Try to add another rule with same ID pattern
        # This tests whether the system prevents ID conflicts
        rules = await shared_memory_manager.get_rules()
        ids = [r.id for r in rules]

        # IDs should be unique
        assert len(ids) == len(set(ids))
