"""
Unit tests for rule retrieval module.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.rule_retrieval import (
    RuleFilter,
    RuleRetrieval,
    RuleRetrievalResult,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
)


class TestRuleFilter:
    """Test RuleFilter dataclass."""

    def test_default_filter(self):
        """Test default filter values."""
        filter = RuleFilter()

        assert filter.scope is None
        assert filter.project_id is None
        assert filter.category is None
        assert filter.authority is None
        assert filter.tags is None
        assert filter.limit == 100

    def test_custom_filter(self):
        """Test custom filter values."""
        filter = RuleFilter(
            scope=["test"],
            project_id="proj123",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            tags=["python"],
            limit=50,
        )

        assert filter.scope == ["test"]
        assert filter.project_id == "proj123"
        assert filter.category == MemoryCategory.BEHAVIOR
        assert filter.authority == AuthorityLevel.ABSOLUTE
        assert filter.tags == ["python"]
        assert filter.limit == 50


class TestRuleRetrievalResult:
    """Test RuleRetrievalResult dataclass."""

    def test_result_structure(self):
        """Test result structure."""
        result = RuleRetrievalResult(
            rules=[], total_count=10, filtered_count=5, cache_hit=True
        )

        assert result.rules == []
        assert result.total_count == 10
        assert result.filtered_count == 5
        assert result.cache_hit is True


class TestRuleRetrieval:
    """Test RuleRetrieval class."""

    @pytest.fixture
    def mock_memory_manager(self, mocker):
        """Create mock memory manager."""
        manager = mocker.Mock(spec=MemoryManager)
        return manager

    @pytest.fixture
    def rule_retrieval(self, mock_memory_manager):
        """Create RuleRetrieval instance with mock."""
        return RuleRetrieval(mock_memory_manager)

    @pytest.fixture
    def sample_rules(self):
        """Create sample memory rules for testing."""
        now = datetime.now(timezone.utc)

        return [
            MemoryRule(
                id="rule1",
                category=MemoryCategory.BEHAVIOR,
                name="Rule 1",
                rule="Always use Python",
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"],
                created_at=now,
                updated_at=now,
                metadata={"priority": 90, "tags": ["python"]},
            ),
            MemoryRule(
                id="rule2",
                category=MemoryCategory.PREFERENCE,
                name="Rule 2",
                rule="Prefer pytest for testing",
                authority=AuthorityLevel.DEFAULT,
                scope=["testing"],
                created_at=now,
                updated_at=now,
                metadata={"priority": 70, "tags": ["testing", "python"]},
            ),
            MemoryRule(
                id="rule3",
                category=MemoryCategory.BEHAVIOR,
                name="Rule 3",
                rule="Use type hints",
                authority=AuthorityLevel.ABSOLUTE,
                scope=[],  # Global rule
                created_at=now,
                updated_at=now,
                metadata={"priority": 95},
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_rules_basic(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test basic rule retrieval."""
        # Setup mock
        mock_memory_manager.list_memory_rules.return_value = sample_rules

        # Test
        filter = RuleFilter()
        result = await rule_retrieval.get_rules(filter)

        # Verify
        assert isinstance(result, RuleRetrievalResult)
        assert len(result.rules) == 3
        assert result.total_count == 3
        assert result.filtered_count == 3
        assert result.cache_hit is False

        # Verify rules are sorted (absolute first, then by priority)
        assert result.rules[0].id == "rule3"  # Absolute, priority 95
        assert result.rules[1].id == "rule1"  # Absolute, priority 90
        assert result.rules[2].id == "rule2"  # Default, priority 70

    @pytest.mark.asyncio
    async def test_get_rules_with_category_filter(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test rule retrieval with category filter."""
        # Filter to only behavior rules
        behavior_rules = [
            r for r in sample_rules if r.category == MemoryCategory.BEHAVIOR
        ]
        mock_memory_manager.list_memory_rules.return_value = behavior_rules

        # Test
        filter = RuleFilter(category=MemoryCategory.BEHAVIOR)
        result = await rule_retrieval.get_rules(filter)

        # Verify
        assert len(result.rules) == 2
        assert all(r.category == MemoryCategory.BEHAVIOR for r in result.rules)

    @pytest.mark.asyncio
    async def test_get_rules_with_authority_filter(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test rule retrieval with authority filter."""
        # Filter to only absolute rules
        absolute_rules = [
            r for r in sample_rules if r.authority == AuthorityLevel.ABSOLUTE
        ]
        mock_memory_manager.list_memory_rules.return_value = absolute_rules

        # Test
        filter = RuleFilter(authority=AuthorityLevel.ABSOLUTE)
        result = await rule_retrieval.get_rules(filter)

        # Verify
        assert len(result.rules) == 2
        assert all(r.authority == AuthorityLevel.ABSOLUTE for r in result.rules)

    @pytest.mark.asyncio
    async def test_get_rules_with_scope_filter(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test rule retrieval with scope filter."""
        mock_memory_manager.list_memory_rules.return_value = sample_rules

        # Test with python scope
        filter = RuleFilter(scope=["python"])
        result = await rule_retrieval.get_rules(filter)

        # Verify - should include python-scoped rules + global rules
        assert len(result.rules) == 2  # rule1 (python scope) + rule3 (global)

    @pytest.mark.asyncio
    async def test_get_rules_with_limit(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test rule retrieval with limit."""
        mock_memory_manager.list_memory_rules.return_value = sample_rules

        # Test with limit of 2
        filter = RuleFilter(limit=2)
        result = await rule_retrieval.get_rules(filter)

        # Verify
        assert len(result.rules) == 2
        assert result.total_count == 3
        assert result.filtered_count == 3

    @pytest.mark.asyncio
    async def test_get_rules_by_scope(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test get_rules_by_scope helper method."""
        mock_memory_manager.list_memory_rules.return_value = sample_rules

        # Test
        rules = await rule_retrieval.get_rules_by_scope(["python"])

        # Verify
        assert len(rules) == 2  # python-scoped + global

    @pytest.mark.asyncio
    async def test_get_absolute_rules(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test get_absolute_rules helper method."""
        absolute_rules = [
            r for r in sample_rules if r.authority == AuthorityLevel.ABSOLUTE
        ]
        mock_memory_manager.list_memory_rules.return_value = absolute_rules

        # Test
        rules = await rule_retrieval.get_absolute_rules(["python"])

        # Verify
        assert len(rules) == 2
        assert all(r.authority == AuthorityLevel.ABSOLUTE for r in rules)

    @pytest.mark.asyncio
    async def test_get_rules_by_category(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test get_rules_by_category helper method."""
        behavior_rules = [
            r for r in sample_rules if r.category == MemoryCategory.BEHAVIOR
        ]
        mock_memory_manager.list_memory_rules.return_value = behavior_rules

        # Test
        rules = await rule_retrieval.get_rules_by_category(MemoryCategory.BEHAVIOR)

        # Verify
        assert len(rules) == 2
        assert all(r.category == MemoryCategory.BEHAVIOR for r in rules)

    @pytest.mark.asyncio
    async def test_search_rules(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test semantic search for rules."""
        # Setup mock to return rules with scores
        search_results = [(sample_rules[0], 0.9), (sample_rules[1], 0.7)]
        mock_memory_manager.search_memory_rules.return_value = search_results

        # Test
        results = await rule_retrieval.search_rules("python testing", limit=10)

        # Verify
        assert len(results) == 2
        assert results[0][1] == 0.9  # First result has higher score
        assert results[1][1] == 0.7

    @pytest.mark.asyncio
    async def test_search_rules_with_filter(
        self, rule_retrieval, mock_memory_manager, sample_rules
    ):
        """Test semantic search with additional filtering."""
        # Setup mock - return only behavior rules since category filter is passed to search
        behavior_results = [(sample_rules[0], 0.9)]  # rule1 is BEHAVIOR
        mock_memory_manager.search_memory_rules.return_value = behavior_results

        # Test with category filter
        filter = RuleFilter(category=MemoryCategory.BEHAVIOR)
        results = await rule_retrieval.search_rules(
            "python testing", limit=10, filter=filter
        )

        # Verify - should only return behavior rules
        assert len(results) == 1
        assert results[0][0].category == MemoryCategory.BEHAVIOR

    @pytest.mark.asyncio
    async def test_get_rules_error_handling(
        self, rule_retrieval, mock_memory_manager
    ):
        """Test error handling in get_rules."""
        # Setup mock to raise exception
        mock_memory_manager.list_memory_rules.side_effect = Exception("Test error")

        # Test
        filter = RuleFilter()
        result = await rule_retrieval.get_rules(filter)

        # Verify - should return empty result
        assert len(result.rules) == 0
        assert result.total_count == 0
        assert result.filtered_count == 0

    def test_sort_by_priority(self, rule_retrieval, sample_rules):
        """Test sorting by authority and priority."""
        # Unsort the rules first
        unsorted = [sample_rules[1], sample_rules[0], sample_rules[2]]

        # Sort
        sorted_rules = rule_retrieval._sort_by_priority(unsorted)

        # Verify order: absolute rules first (by priority), then default rules
        assert sorted_rules[0].id == "rule3"  # Absolute, priority 95
        assert sorted_rules[1].id == "rule1"  # Absolute, priority 90
        assert sorted_rules[2].id == "rule2"  # Default, priority 70

    def test_matches_filter_project_id(self, rule_retrieval):
        """Test filter matching for project_id."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
            metadata={"project_id": "proj123"},
        )

        # Should match
        filter = RuleFilter(project_id="proj123")
        assert rule_retrieval._matches_filter(rule, filter) is True

        # Should not match
        filter = RuleFilter(project_id="proj456")
        assert rule_retrieval._matches_filter(rule, filter) is False

    def test_matches_filter_tags(self, rule_retrieval):
        """Test filter matching for tags."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
            metadata={"tags": ["python", "testing"]},
        )

        # Should match (any overlap)
        filter = RuleFilter(tags=["python"])
        assert rule_retrieval._matches_filter(rule, filter) is True

        filter = RuleFilter(tags=["python", "java"])
        assert rule_retrieval._matches_filter(rule, filter) is True

        # Should not match (no overlap)
        filter = RuleFilter(tags=["java", "rust"])
        assert rule_retrieval._matches_filter(rule, filter) is False

    def test_matches_filter_scope_global(self, rule_retrieval):
        """Test filter matching for global scope rules."""
        # Global rule (empty scope)
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="Test",
            rule="Test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        # Global rules should always match any scope
        filter = RuleFilter(scope=["python"])
        assert rule_retrieval._matches_filter(rule, filter) is True
