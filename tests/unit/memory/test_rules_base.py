"""
Base test class for memory rules testing.

Provides common setup, teardown, and helper methods for memory rules tests.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock
from dataclasses import asdict

import pytest

from common.core.memory import (
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
    AgentDefinition,
)


class BaseMemoryRuleTest:
    """
    Base class for memory rule tests.

    Provides common setup, teardown, and helper methods for testing
    memory rules functionality.
    """

    @pytest.fixture(autouse=True)
    async def setup_base(self):
        """
        Automatic setup for all memory rule tests.

        Initializes common test data and mocks.
        """
        self.test_rules: List[MemoryRule] = []
        self.test_agents: List[AgentDefinition] = []
        self.mock_client = None

        yield

        # Cleanup
        self.test_rules.clear()
        self.test_agents.clear()

    def create_test_rule(
        self,
        rule_id: str = "test-rule-1",
        category: MemoryCategory = MemoryCategory.BEHAVIOR,
        name: str = "Test Rule",
        rule_text: str = "Always test thoroughly",
        authority: AuthorityLevel = AuthorityLevel.DEFAULT,
        scope: Optional[List[str]] = None,
        **kwargs: Any
    ) -> MemoryRule:
        """
        Create a test memory rule with sensible defaults.

        Args:
            rule_id: Unique identifier for the rule
            category: Memory category
            name: Rule name
            rule_text: The actual rule text
            authority: Authority level
            scope: List of scope contexts
            **kwargs: Additional MemoryRule fields

        Returns:
            MemoryRule instance
        """
        if scope is None:
            scope = ["global"]

        rule = MemoryRule(
            id=rule_id,
            category=category,
            name=name,
            rule=rule_text,
            authority=authority,
            scope=scope,
            **kwargs
        )

        self.test_rules.append(rule)
        return rule

    def create_test_agent(
        self,
        agent_id: str = "test-agent-1",
        name: str = "test-agent",
        description: str = "Test agent for testing",
        capabilities: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AgentDefinition:
        """
        Create a test agent definition with sensible defaults.

        Args:
            agent_id: Unique identifier for the agent
            name: Agent name
            description: Agent description
            capabilities: List of capabilities
            **kwargs: Additional AgentDefinition fields

        Returns:
            AgentDefinition instance
        """
        if capabilities is None:
            capabilities = ["testing", "validation"]

        agent = AgentDefinition(
            id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
            **kwargs
        )

        self.test_agents.append(agent)
        return agent

    def assert_rule_valid(self, rule: MemoryRule) -> None:
        """
        Assert that a memory rule has all required fields and valid values.

        Args:
            rule: MemoryRule to validate

        Raises:
            AssertionError: If rule is invalid
        """
        assert rule.id, "Rule must have an ID"
        assert rule.category, "Rule must have a category"
        assert isinstance(rule.category, MemoryCategory), "Category must be MemoryCategory enum"
        assert rule.name, "Rule must have a name"
        assert rule.rule, "Rule must have rule text"
        assert rule.authority, "Rule must have authority level"
        assert isinstance(rule.authority, AuthorityLevel), "Authority must be AuthorityLevel enum"
        assert rule.scope, "Rule must have scope"
        assert isinstance(rule.scope, list), "Scope must be a list"
        assert rule.created_at, "Rule must have created_at timestamp"
        assert rule.updated_at, "Rule must have updated_at timestamp"

    def assert_agent_valid(self, agent: AgentDefinition) -> None:
        """
        Assert that an agent definition has all required fields and valid values.

        Args:
            agent: AgentDefinition to validate

        Raises:
            AssertionError: If agent is invalid
        """
        assert agent.id, "Agent must have an ID"
        assert agent.name, "Agent must have a name"
        assert agent.description, "Agent must have a description"
        assert agent.capabilities, "Agent must have capabilities"
        assert isinstance(agent.capabilities, list), "Capabilities must be a list"
        assert agent.deploy_cost, "Agent must have deploy_cost"

    def assert_rules_equal(self, rule1: MemoryRule, rule2: MemoryRule, ignore_timestamps: bool = True) -> None:
        """
        Assert that two rules are equal (optionally ignoring timestamps).

        Args:
            rule1: First rule
            rule2: Second rule
            ignore_timestamps: If True, ignore created_at and updated_at differences

        Raises:
            AssertionError: If rules are not equal
        """
        assert rule1.id == rule2.id, "Rule IDs must match"
        assert rule1.category == rule2.category, "Rule categories must match"
        assert rule1.name == rule2.name, "Rule names must match"
        assert rule1.rule == rule2.rule, "Rule texts must match"
        assert rule1.authority == rule2.authority, "Rule authority levels must match"
        assert rule1.scope == rule2.scope, "Rule scopes must match"
        assert rule1.source == rule2.source, "Rule sources must match"
        assert rule1.conditions == rule2.conditions, "Rule conditions must match"
        assert rule1.replaces == rule2.replaces, "Rule replaces must match"
        assert rule1.metadata == rule2.metadata, "Rule metadata must match"

        if not ignore_timestamps:
            assert rule1.created_at == rule2.created_at, "Rule created_at must match"
            assert rule1.updated_at == rule2.updated_at, "Rule updated_at must match"

    def create_mock_qdrant_client(self) -> Mock:
        """
        Create a mock Qdrant client for testing.

        Returns:
            Mock Qdrant client with common methods
        """
        mock_client = Mock()

        # Mock async methods
        mock_client.create_collection = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.get_collection = AsyncMock()
        mock_client.upsert = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        mock_client.scroll = AsyncMock(return_value=([], None))
        mock_client.delete = AsyncMock()
        mock_client.retrieve = AsyncMock()

        self.mock_client = mock_client
        return mock_client

    def get_rule_as_dict(self, rule: MemoryRule) -> Dict[str, Any]:
        """
        Convert MemoryRule to dictionary representation.

        Args:
            rule: MemoryRule to convert

        Returns:
            Dictionary representation of the rule
        """
        return asdict(rule)

    def get_agent_as_dict(self, agent: AgentDefinition) -> Dict[str, Any]:
        """
        Convert AgentDefinition to dictionary representation.

        Args:
            agent: AgentDefinition to convert

        Returns:
            Dictionary representation of the agent
        """
        return asdict(agent)
