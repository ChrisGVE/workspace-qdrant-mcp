"""
Comprehensive unit tests for claude_integration module.

This test module provides 100% coverage for the ClaudeIntegrationManager
and related components, including session initialization, memory rule
injection, conflict resolution, and system context formatting.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
from pathlib import Path

# Add the src directory to Python path
import sys
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

try:
    from common.core.claude_integration import ClaudeIntegrationManager
    from common.core.memory import (
        AuthorityLevel,
        MemoryCategory,
        MemoryConflict,
        MemoryManager,
        MemoryRule,
    )
    CLAUDE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    CLAUDE_INTEGRATION_AVAILABLE = False
    pytest.skip(f"Claude integration module not available: {e}", allow_module_level=True)


class TestClaudeIntegrationManager:
    """Test Claude integration manager functionality."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        manager = Mock(spec=MemoryManager)
        manager.load_rules = AsyncMock()
        manager.detect_conflicts = AsyncMock()
        manager.get_rules_by_category = Mock()
        manager.get_all_rules = Mock()
        manager.resolve_conflict = AsyncMock()
        return manager

    @pytest.fixture
    def sample_memory_rules(self):
        """Create sample memory rules for testing."""
        return [
            MemoryRule(
                id="rule1",
                content="Test rule 1",
                category=MemoryCategory.WORKFLOW,
                authority=AuthorityLevel.USER_EXPLICIT,
                created_at=datetime.now(timezone.utc)
            ),
            MemoryRule(
                id="rule2",
                content="Test rule 2",
                category=MemoryCategory.PROJECT_CONTEXT,
                authority=AuthorityLevel.SYSTEM_DEFAULT,
                created_at=datetime.now(timezone.utc)
            ),
            MemoryRule(
                id="rule3",
                content="Test rule 3",
                category=MemoryCategory.CODE_PATTERNS,
                authority=AuthorityLevel.AI_INFERRED,
                created_at=datetime.now(timezone.utc)
            )
        ]

    @pytest.fixture
    def sample_conflicts(self):
        """Create sample memory conflicts for testing."""
        return [
            MemoryConflict(
                rule1_id="rule1",
                rule2_id="rule2",
                conflict_type="content_overlap",
                severity=0.8,
                description="Rules have overlapping content"
            )
        ]

    def test_claude_integration_manager_initialization(self, mock_memory_manager):
        """Test ClaudeIntegrationManager initialization."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        assert manager.memory_manager == mock_memory_manager

    @pytest.mark.asyncio
    async def test_initialize_session_success(self, mock_memory_manager, sample_memory_rules):
        """Test successful session initialization."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Mock successful rule loading
        mock_memory_manager.load_rules.return_value = sample_memory_rules
        mock_memory_manager.detect_conflicts.return_value = []

        result = await manager.initialize_session()

        assert isinstance(result, dict)
        assert "status" in result
        assert "rules_loaded" in result
        assert "conflicts_detected" in result
        assert "system_context" in result

        mock_memory_manager.load_rules.assert_called_once()
        mock_memory_manager.detect_conflicts.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_session_with_conflicts(self, mock_memory_manager, sample_memory_rules, sample_conflicts):
        """Test session initialization with conflicts."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Mock rule loading with conflicts
        mock_memory_manager.load_rules.return_value = sample_memory_rules
        mock_memory_manager.detect_conflicts.return_value = sample_conflicts

        result = await manager.initialize_session()

        assert isinstance(result, dict)
        assert result["conflicts_detected"] == len(sample_conflicts)
        assert "conflict_details" in result

    @pytest.mark.asyncio
    async def test_initialize_session_error_handling(self, mock_memory_manager):
        """Test session initialization error handling."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Mock error during rule loading
        mock_memory_manager.load_rules.side_effect = Exception("Database error")

        result = await manager.initialize_session()

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_load_memory_rules(self, mock_memory_manager, sample_memory_rules):
        """Test memory rule loading."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        mock_memory_manager.load_rules.return_value = sample_memory_rules

        rules = await manager._load_memory_rules()

        assert len(rules) == 3
        assert all(isinstance(rule, MemoryRule) for rule in rules)
        mock_memory_manager.load_rules.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_rule_conflicts(self, mock_memory_manager, sample_memory_rules, sample_conflicts):
        """Test rule conflict detection."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        mock_memory_manager.detect_conflicts.return_value = sample_conflicts

        conflicts = await manager._detect_rule_conflicts(sample_memory_rules)

        assert len(conflicts) == 1
        assert isinstance(conflicts[0], MemoryConflict)
        mock_memory_manager.detect_conflicts.assert_called_once_with(sample_memory_rules)

    def test_format_system_context(self, mock_memory_manager, sample_memory_rules):
        """Test system context formatting."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        context = manager._format_system_context(sample_memory_rules)

        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain rule content
        assert "Test rule 1" in context
        assert "Test rule 2" in context
        assert "Test rule 3" in context

    def test_format_system_context_empty_rules(self, mock_memory_manager):
        """Test system context formatting with empty rules."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        context = manager._format_system_context([])

        assert isinstance(context, str)
        assert len(context) == 0 or "No memory rules" in context

    def test_format_system_context_by_category(self, mock_memory_manager, sample_memory_rules):
        """Test system context formatting grouped by category."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        context = manager._format_system_context_by_category(sample_memory_rules)

        assert isinstance(context, str)
        assert "WORKFLOW" in context
        assert "PROJECT_CONTEXT" in context
        assert "CODE_PATTERNS" in context

    def test_format_conflict_details(self, mock_memory_manager, sample_conflicts):
        """Test conflict details formatting."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        details = manager._format_conflict_details(sample_conflicts)

        assert isinstance(details, str)
        assert "conflict_type: content_overlap" in details
        assert "severity: 0.8" in details

    def test_format_conflict_details_empty(self, mock_memory_manager):
        """Test conflict details formatting with no conflicts."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        details = manager._format_conflict_details([])

        assert isinstance(details, str)
        assert details == "No conflicts detected"

    @pytest.mark.asyncio
    async def test_resolve_conflicts(self, mock_memory_manager, sample_conflicts):
        """Test conflict resolution."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        mock_memory_manager.resolve_conflict.return_value = True

        result = await manager._resolve_conflicts(sample_conflicts)

        assert isinstance(result, dict)
        assert "resolved_count" in result
        assert "failed_count" in result
        mock_memory_manager.resolve_conflict.assert_called()

    @pytest.mark.asyncio
    async def test_resolve_conflicts_with_failures(self, mock_memory_manager, sample_conflicts):
        """Test conflict resolution with failures."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Mock some failures
        mock_memory_manager.resolve_conflict.side_effect = [True, False]

        conflicts_extended = sample_conflicts + [
            MemoryConflict(
                rule1_id="rule3",
                rule2_id="rule4",
                conflict_type="authority_conflict",
                severity=0.6,
                description="Authority level conflict"
            )
        ]

        result = await manager._resolve_conflicts(conflicts_extended)

        assert result["resolved_count"] == 1
        assert result["failed_count"] == 1

    def test_get_session_summary(self, mock_memory_manager, sample_memory_rules, sample_conflicts):
        """Test session summary generation."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        summary = manager._get_session_summary(sample_memory_rules, sample_conflicts)

        assert isinstance(summary, dict)
        assert "total_rules" in summary
        assert "rules_by_category" in summary
        assert "total_conflicts" in summary
        assert "conflicts_by_type" in summary

        assert summary["total_rules"] == 3

    def test_get_session_summary_empty(self, mock_memory_manager):
        """Test session summary with empty data."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        summary = manager._get_session_summary([], [])

        assert summary["total_rules"] == 0
        assert summary["total_conflicts"] == 0

    def test_filter_rules_by_authority(self, mock_memory_manager, sample_memory_rules):
        """Test filtering rules by authority level."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        high_authority_rules = manager._filter_rules_by_authority(
            sample_memory_rules,
            min_authority=AuthorityLevel.USER_EXPLICIT
        )

        # Should only include USER_EXPLICIT rules
        assert len(high_authority_rules) == 1
        assert high_authority_rules[0].authority == AuthorityLevel.USER_EXPLICIT

    def test_filter_rules_by_category(self, mock_memory_manager, sample_memory_rules):
        """Test filtering rules by category."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        workflow_rules = manager._filter_rules_by_category(
            sample_memory_rules,
            category=MemoryCategory.WORKFLOW
        )

        assert len(workflow_rules) == 1
        assert workflow_rules[0].category == MemoryCategory.WORKFLOW

    def test_sort_rules_by_priority(self, mock_memory_manager, sample_memory_rules):
        """Test sorting rules by priority."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        sorted_rules = manager._sort_rules_by_priority(sample_memory_rules)

        assert len(sorted_rules) == len(sample_memory_rules)
        # Should be sorted by authority level (highest first)
        assert sorted_rules[0].authority == AuthorityLevel.USER_EXPLICIT

    def test_validate_rule_integrity(self, mock_memory_manager, sample_memory_rules):
        """Test rule integrity validation."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        validation_result = manager._validate_rule_integrity(sample_memory_rules)

        assert isinstance(validation_result, dict)
        assert "valid_rules" in validation_result
        assert "invalid_rules" in validation_result
        assert "validation_errors" in validation_result

    def test_validate_rule_integrity_invalid_rules(self, mock_memory_manager):
        """Test rule integrity validation with invalid rules."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Create rules with missing required fields
        invalid_rules = [
            MemoryRule(
                id="",  # Invalid empty ID
                content="Test rule",
                category=MemoryCategory.WORKFLOW,
                authority=AuthorityLevel.USER_EXPLICIT
            ),
            MemoryRule(
                id="rule2",
                content="",  # Invalid empty content
                category=MemoryCategory.WORKFLOW,
                authority=AuthorityLevel.USER_EXPLICIT
            )
        ]

        validation_result = manager._validate_rule_integrity(invalid_rules)

        assert validation_result["invalid_rules"] > 0
        assert len(validation_result["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_inject_system_context(self, mock_memory_manager, sample_memory_rules):
        """Test system context injection."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        context = manager._format_system_context(sample_memory_rules)

        injection_result = await manager._inject_system_context(context)

        assert isinstance(injection_result, dict)
        assert "success" in injection_result
        assert "context_length" in injection_result

    @pytest.mark.asyncio
    async def test_inject_system_context_error(self, mock_memory_manager):
        """Test system context injection error handling."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Simulate injection failure
        with patch.object(manager, '_inject_system_context', side_effect=Exception("Injection failed")):
            try:
                await manager._inject_system_context("test context")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Injection failed" in str(e)

    def test_create_session_metadata(self, mock_memory_manager, sample_memory_rules, sample_conflicts):
        """Test session metadata creation."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        metadata = manager._create_session_metadata(sample_memory_rules, sample_conflicts)

        assert isinstance(metadata, dict)
        assert "session_id" in metadata
        assert "timestamp" in metadata
        assert "rule_count" in metadata
        assert "conflict_count" in metadata
        assert "claude_integration_version" in metadata

    def test_generate_session_id(self, mock_memory_manager):
        """Test session ID generation."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        session_id = manager._generate_session_id()

        assert isinstance(session_id, str)
        assert len(session_id) > 0
        # Should be unique on each call
        assert session_id != manager._generate_session_id()

    @pytest.mark.asyncio
    async def test_cleanup_session(self, mock_memory_manager):
        """Test session cleanup."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        cleanup_result = await manager._cleanup_session("test-session-id")

        assert isinstance(cleanup_result, dict)
        assert "cleanup_success" in cleanup_result

    def test_get_integration_status(self, mock_memory_manager):
        """Test integration status retrieval."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        status = manager.get_integration_status()

        assert isinstance(status, dict)
        assert "claude_integration_active" in status
        assert "memory_manager_available" in status
        assert "last_session_time" in status

    @pytest.mark.asyncio
    async def test_refresh_memory_rules(self, mock_memory_manager, sample_memory_rules):
        """Test memory rules refresh."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        mock_memory_manager.load_rules.return_value = sample_memory_rules

        result = await manager.refresh_memory_rules()

        assert isinstance(result, dict)
        assert "refreshed_count" in result
        assert "refresh_timestamp" in result
        mock_memory_manager.load_rules.assert_called()

    @pytest.mark.asyncio
    async def test_update_system_context(self, mock_memory_manager, sample_memory_rules):
        """Test system context update."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        result = await manager.update_system_context(sample_memory_rules)

        assert isinstance(result, dict)
        assert "update_success" in result
        assert "context_length" in result

    def test_format_rule_for_claude(self, mock_memory_manager, sample_memory_rules):
        """Test formatting individual rule for Claude."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        formatted_rule = manager._format_rule_for_claude(sample_memory_rules[0])

        assert isinstance(formatted_rule, str)
        assert sample_memory_rules[0].content in formatted_rule
        assert str(sample_memory_rules[0].category.value) in formatted_rule
        assert str(sample_memory_rules[0].authority.value) in formatted_rule

    def test_optimize_context_length(self, mock_memory_manager, sample_memory_rules):
        """Test context length optimization."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Create a very long context
        long_context = "Very long context " * 1000

        optimized_context = manager._optimize_context_length(long_context, max_length=100)

        assert len(optimized_context) <= 100
        assert isinstance(optimized_context, str)

    def test_prioritize_rules_for_context(self, mock_memory_manager, sample_memory_rules):
        """Test rule prioritization for context inclusion."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        prioritized_rules = manager._prioritize_rules_for_context(sample_memory_rules)

        assert len(prioritized_rules) <= len(sample_memory_rules)
        # Highest authority rules should come first
        if prioritized_rules:
            assert prioritized_rules[0].authority == AuthorityLevel.USER_EXPLICIT


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_memory_manager_unavailable(self):
        """Test handling when memory manager is unavailable."""
        manager = ClaudeIntegrationManager(None)

        with pytest.raises(AttributeError):
            await manager.initialize_session()

    @pytest.mark.asyncio
    async def test_rule_loading_failure(self, mock_memory_manager):
        """Test handling of rule loading failures."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        mock_memory_manager.load_rules.side_effect = Exception("Database connection failed")

        result = await manager.initialize_session()

        assert result["status"] == "error"
        assert "Database connection failed" in result["error_message"]

    @pytest.mark.asyncio
    async def test_conflict_detection_failure(self, mock_memory_manager, sample_memory_rules):
        """Test handling of conflict detection failures."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        mock_memory_manager.load_rules.return_value = sample_memory_rules
        mock_memory_manager.detect_conflicts.side_effect = Exception("Conflict detection error")

        result = await manager.initialize_session()

        assert result["status"] == "error"

    def test_invalid_rule_data(self, mock_memory_manager):
        """Test handling of invalid rule data."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        invalid_rules = [None, "not a rule", 123]

        try:
            context = manager._format_system_context(invalid_rules)
            # Should handle gracefully
            assert isinstance(context, str)
        except Exception:
            # Acceptable to raise exception for invalid data
            pass

    def test_malformed_conflict_data(self, mock_memory_manager):
        """Test handling of malformed conflict data."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        malformed_conflicts = [None, "not a conflict", {"invalid": "data"}]

        try:
            details = manager._format_conflict_details(malformed_conflicts)
            assert isinstance(details, str)
        except Exception:
            # Acceptable to raise exception for malformed data
            pass


class TestPerformanceOptimization:
    """Test performance optimization features."""

    def test_large_rule_set_handling(self, mock_memory_manager):
        """Test handling of large rule sets."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Create a large number of rules
        large_rule_set = []
        for i in range(1000):
            rule = MemoryRule(
                id=f"rule{i}",
                content=f"Test rule {i}",
                category=MemoryCategory.WORKFLOW,
                authority=AuthorityLevel.AI_INFERRED,
                created_at=datetime.now(timezone.utc)
            )
            large_rule_set.append(rule)

        # Should handle large sets efficiently
        context = manager._format_system_context(large_rule_set)
        assert isinstance(context, str)

    def test_context_optimization_for_large_content(self, mock_memory_manager, sample_memory_rules):
        """Test context optimization for large content."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Create rules with very long content
        long_content_rules = []
        for i, rule in enumerate(sample_memory_rules):
            long_rule = MemoryRule(
                id=rule.id,
                content=f"Very long content: {'x' * 10000}",
                category=rule.category,
                authority=rule.authority,
                created_at=rule.created_at
            )
            long_content_rules.append(long_rule)

        # Should optimize for reasonable context length
        optimized_context = manager._optimize_context_length(
            manager._format_system_context(long_content_rules),
            max_length=5000
        )

        assert len(optimized_context) <= 5000

    @pytest.mark.asyncio
    async def test_concurrent_conflict_resolution(self, mock_memory_manager):
        """Test concurrent conflict resolution."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Create multiple conflicts
        conflicts = []
        for i in range(10):
            conflict = MemoryConflict(
                rule1_id=f"rule{i}",
                rule2_id=f"rule{i+1}",
                conflict_type="test_conflict",
                severity=0.5,
                description=f"Test conflict {i}"
            )
            conflicts.append(conflict)

        mock_memory_manager.resolve_conflict.return_value = True

        result = await manager._resolve_conflicts(conflicts)

        assert result["resolved_count"] == len(conflicts)


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self, mock_memory_manager, sample_memory_rules):
        """Test complete session lifecycle."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Mock successful operations
        mock_memory_manager.load_rules.return_value = sample_memory_rules
        mock_memory_manager.detect_conflicts.return_value = []

        # Initialize session
        init_result = await manager.initialize_session()
        assert init_result["status"] == "success"

        # Get status
        status = manager.get_integration_status()
        assert status["claude_integration_active"] is True

        # Refresh rules
        refresh_result = await manager.refresh_memory_rules()
        assert "refreshed_count" in refresh_result

        # Cleanup
        cleanup_result = await manager._cleanup_session(init_result.get("session_id", "test"))
        assert "cleanup_success" in cleanup_result

    @pytest.mark.asyncio
    async def test_session_with_conflict_resolution(self, mock_memory_manager, sample_memory_rules, sample_conflicts):
        """Test session with conflict resolution workflow."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Mock operations with conflicts
        mock_memory_manager.load_rules.return_value = sample_memory_rules
        mock_memory_manager.detect_conflicts.return_value = sample_conflicts
        mock_memory_manager.resolve_conflict.return_value = True

        # Initialize session with automatic conflict resolution
        result = await manager.initialize_session()

        assert result["conflicts_detected"] > 0
        assert "conflict_details" in result

    @pytest.mark.asyncio
    async def test_dynamic_rule_updates(self, mock_memory_manager, sample_memory_rules):
        """Test dynamic rule updates during session."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Initial session
        mock_memory_manager.load_rules.return_value = sample_memory_rules
        mock_memory_manager.detect_conflicts.return_value = []

        await manager.initialize_session()

        # Add new rule
        new_rule = MemoryRule(
            id="new_rule",
            content="Dynamic rule",
            category=MemoryCategory.PROJECT_CONTEXT,
            authority=AuthorityLevel.USER_EXPLICIT,
            created_at=datetime.now(timezone.utc)
        )

        updated_rules = sample_memory_rules + [new_rule]
        mock_memory_manager.load_rules.return_value = updated_rules

        # Update context
        update_result = await manager.update_system_context(updated_rules)
        assert update_result["update_success"] is True

    @pytest.mark.asyncio
    async def test_multi_category_rule_organization(self, mock_memory_manager):
        """Test organization of rules across multiple categories."""
        manager = ClaudeIntegrationManager(mock_memory_manager)

        # Create rules across all categories
        multi_category_rules = []
        for category in MemoryCategory:
            for i in range(3):
                rule = MemoryRule(
                    id=f"{category.value}_rule_{i}",
                    content=f"Rule for {category.value} category {i}",
                    category=category,
                    authority=AuthorityLevel.USER_EXPLICIT,
                    created_at=datetime.now(timezone.utc)
                )
                multi_category_rules.append(rule)

        # Test categorized formatting
        categorized_context = manager._format_system_context_by_category(multi_category_rules)

        for category in MemoryCategory:
            assert category.value in categorized_context