"""
Pytest fixtures for memory rules testing.

Provides reusable fixtures for mock memory collections, sample rules,
and testing utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from common.core.memory import (
    AgentDefinition,
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryRule,
)


@pytest.fixture
def mock_memory_client():
    """
    Provide mock Qdrant client for memory operations.

    Returns:
        Mock Qdrant client with async methods
    """
    mock_client = Mock()

    # Mock collection operations
    mock_client.create_collection = AsyncMock()
    mock_client.collection_exists = AsyncMock(return_value=True)
    mock_client.get_collection = AsyncMock()
    mock_client.delete_collection = AsyncMock()

    # Mock point operations
    mock_client.upsert = AsyncMock()
    mock_client.search = AsyncMock(return_value=[])
    mock_client.scroll = AsyncMock(return_value=([], None))
    mock_client.delete = AsyncMock()
    mock_client.retrieve = AsyncMock()
    mock_client.count = AsyncMock(return_value=0)

    # Storage for tracking calls
    mock_client.upserted_points = []
    mock_client.search_queries = []
    mock_client.deleted_points = []

    def track_upsert(collection_name, points, **kwargs):
        """Track upserted points."""
        mock_client.upserted_points.extend(points)
        return AsyncMock()

    def track_search(collection_name, query_vector, **kwargs):
        """Track search queries."""
        mock_client.search_queries.append({
            "collection": collection_name,
            "query": query_vector,
            "kwargs": kwargs
        })
        return AsyncMock(return_value=[])

    def track_delete(collection_name, points_selector, **kwargs):
        """Track deleted points."""
        mock_client.deleted_points.append({
            "collection": collection_name,
            "selector": points_selector,
            "kwargs": kwargs
        })
        return AsyncMock()

    mock_client.upsert.side_effect = track_upsert
    mock_client.search.side_effect = track_search
    mock_client.delete.side_effect = track_delete

    return mock_client


@pytest.fixture
def sample_memory_rules() -> list[MemoryRule]:
    """
    Provide sample memory rules for testing.

    Returns:
        List of sample MemoryRule instances
    """
    rules = [
        # Absolute behavior rule
        MemoryRule(
            id="rule-behavior-1",
            category=MemoryCategory.BEHAVIOR,
            name="Atomic Commits",
            rule="Always make atomic commits after each change",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
        ),
        # Default behavior rule
        MemoryRule(
            id="rule-behavior-2",
            category=MemoryCategory.BEHAVIOR,
            name="Test Coverage",
            rule="Aim for 90% test coverage unless specified otherwise",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        ),
        # User preference
        MemoryRule(
            id="rule-preference-1",
            category=MemoryCategory.PREFERENCE,
            name="Package Manager",
            rule="Use uv for Python package management",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user_explicit",
        ),
        # Project-scoped preference
        MemoryRule(
            id="rule-preference-2",
            category=MemoryCategory.PREFERENCE,
            name="Code Style",
            rule="Use black for code formatting with line length 88",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "project-x"],
            source="conversational",
        ),
        # Conditional rule
        MemoryRule(
            id="rule-behavior-3",
            category=MemoryCategory.BEHAVIOR,
            name="Sequential Agents",
            rule="Execute agents sequentially to conserve API usage",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
            conditions={"api_mode": "rate_limited"},
        ),
    ]

    return rules


@pytest.fixture
def sample_agent_definitions() -> list[AgentDefinition]:
    """
    Provide sample agent definitions for testing.

    Returns:
        List of sample AgentDefinition instances
    """
    agents = [
        AgentDefinition(
            id="agent-python-pro",
            name="python-pro",
            description="Expert Python developer with mastery of modern best practices",
            capabilities=[
                "python_development",
                "async_programming",
                "testing",
                "type_hints",
                "performance_optimization",
            ],
            deploy_cost="medium",
            metadata={
                "specialization": "Python 3.11+",
                "testing_framework": "pytest",
            },
        ),
        AgentDefinition(
            id="agent-rust-engineer",
            name="rust-engineer",
            description="Systems programming expert specializing in Rust",
            capabilities=[
                "rust_development",
                "systems_programming",
                "performance_optimization",
                "memory_safety",
                "concurrent_programming",
            ],
            deploy_cost="high",
            metadata={
                "specialization": "High-performance systems",
                "focus": "Memory safety and concurrency",
            },
        ),
        AgentDefinition(
            id="agent-frontend-developer",
            name="frontend-developer",
            description="Frontend specialist with React and TypeScript expertise",
            capabilities=[
                "react_development",
                "typescript",
                "ui_ux_design",
                "responsive_design",
                "state_management",
            ],
            deploy_cost="medium",
            metadata={
                "frameworks": ["React", "Next.js"],
                "styling": "TailwindCSS",
            },
        ),
    ]

    return agents


@pytest.fixture
def mock_bm25_encoder():
    """
    Provide mock BM25 sparse vector encoder.

    Returns:
        Mock BM25SparseEncoder instance
    """
    mock_encoder = Mock()
    mock_encoder.encode_documents = Mock(return_value=[
        {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]}
    ])
    mock_encoder.encode_queries = Mock(return_value=[
        {"indices": [1, 5], "values": [0.6, 0.4]}
    ])
    return mock_encoder


@pytest.fixture
def memory_collection_names() -> dict[str, str]:
    """
    Provide standard memory collection names.

    Returns:
        Dict mapping collection types to names
    """
    return {
        "memory": "memory",
        "agent_memory": "_agent_memory",
        "user_preferences": "_user_preferences",
        "behavioral_rules": "_behavioral_rules",
        "agent_library": "_agent_library",
    }


@pytest.fixture
def rule_conflict_pairs() -> list[tuple[MemoryRule, MemoryRule]]:
    """
    Provide pairs of conflicting rules for testing conflict detection.

    Returns:
        List of tuples containing conflicting rule pairs
    """
    conflicts = [
        # Direct contradiction
        (
            MemoryRule(
                id="conflict-1a",
                category=MemoryCategory.BEHAVIOR,
                name="Parallel Execution",
                rule="Always execute agents in parallel for speed",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="user_explicit",
            ),
            MemoryRule(
                id="conflict-1b",
                category=MemoryCategory.BEHAVIOR,
                name="Sequential Execution",
                rule="Always execute agents sequentially to conserve resources",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="user_explicit",
            ),
        ),
        # Authority level conflict
        (
            MemoryRule(
                id="conflict-2a",
                category=MemoryCategory.PREFERENCE,
                name="Python Version Absolute",
                rule="Must use Python 3.11",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                source="user_explicit",
            ),
            MemoryRule(
                id="conflict-2b",
                category=MemoryCategory.PREFERENCE,
                name="Python Version Default",
                rule="Prefer Python 3.12",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="conversational",
            ),
        ),
    ]

    return conflicts


@pytest.fixture
def sample_memory_conflicts(rule_conflict_pairs) -> list[MemoryConflict]:
    """
    Provide sample memory conflicts for testing.

    Args:
        rule_conflict_pairs: Fixture providing conflicting rule pairs

    Returns:
        List of MemoryConflict instances
    """
    conflicts = []

    for idx, (rule1, rule2) in enumerate(rule_conflict_pairs):
        conflict = MemoryConflict(
            conflict_type="direct_contradiction" if idx == 0 else "authority_mismatch",
            rule1=rule1,
            rule2=rule2,
            confidence=0.95 if idx == 0 else 0.85,
            description=f"Conflict between {rule1.name} and {rule2.name}",
            resolution_options=[
                "keep_newer",
                "keep_higher_authority",
                "merge_conditions",
                "user_decision",
            ],
        )
        conflicts.append(conflict)

    return conflicts


@pytest.fixture
def rule_validator():
    """
    Provide rule validation utility functions.

    Returns:
        Dict of validation functions
    """
    def validate_rule_structure(rule: MemoryRule) -> bool:
        """Validate that rule has all required fields."""
        required_fields = ["id", "category", "name", "rule", "authority", "scope"]
        return all(hasattr(rule, field) and getattr(rule, field) for field in required_fields)

    def validate_rule_category(rule: MemoryRule) -> bool:
        """Validate that rule category is valid."""
        return isinstance(rule.category, MemoryCategory)

    def validate_rule_authority(rule: MemoryRule) -> bool:
        """Validate that rule authority is valid."""
        return isinstance(rule.authority, AuthorityLevel)

    def validate_rule_scope(rule: MemoryRule) -> bool:
        """Validate that rule scope is non-empty list."""
        return isinstance(rule.scope, list) and len(rule.scope) > 0

    def validate_rule_timestamps(rule: MemoryRule) -> bool:
        """Validate that rule has valid timestamps."""
        return (
            rule.created_at is not None
            and rule.updated_at is not None
            and rule.updated_at >= rule.created_at
        )

    return {
        "structure": validate_rule_structure,
        "category": validate_rule_category,
        "authority": validate_rule_authority,
        "scope": validate_rule_scope,
        "timestamps": validate_rule_timestamps,
    }


@pytest.fixture
def mock_embedding_model():
    """
    Provide mock embedding model for testing.

    Returns:
        Mock embedding model
    """
    mock_model = Mock()
    mock_model.embed = Mock(return_value=[[0.1] * 384])  # 384-dim embeddings
    mock_model.passage_embed = Mock(return_value=[[0.1] * 384])
    mock_model.query_embed = Mock(return_value=[[0.1] * 384])
    return mock_model


@pytest.fixture
async def mock_memory_manager(mock_memory_client, mock_bm25_encoder, mock_embedding_model):
    """
    Provide mock MemoryManager instance for testing.

    Args:
        mock_memory_client: Mock Qdrant client
        mock_bm25_encoder: Mock BM25 encoder
        mock_embedding_model: Mock embedding model

    Returns:
        Mock MemoryManager instance
    """
    from unittest.mock import patch

    with patch("common.core.memory.QdrantClient", return_value=mock_memory_client):
        with patch("common.core.memory.BM25SparseEncoder", return_value=mock_bm25_encoder):
            # Import here to use patched dependencies
            from common.core.memory import MemoryManager

            # Create instance with mocked dependencies
            manager = Mock(spec=MemoryManager)
            manager.client = mock_memory_client
            manager.sparse_encoder = mock_bm25_encoder
            manager.embedding_model = mock_embedding_model

            # Mock async methods
            manager.initialize = AsyncMock()
            manager.add_rule = AsyncMock()
            manager.get_rule = AsyncMock()
            manager.search_rules = AsyncMock(return_value=[])
            manager.update_rule = AsyncMock()
            manager.delete_rule = AsyncMock()
            manager.detect_conflicts = AsyncMock(return_value=[])

            yield manager
