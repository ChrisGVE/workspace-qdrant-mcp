"""
Unit tests for SystemPromptInjector.

Tests system prompt generation, token budget enforcement, injection modes,
and integration with ClaudeCodeAdapter.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import pytest

from src.python.common.core.context_injection.system_prompt_injector import (
    SystemPromptInjector,
    SystemPromptConfig,
    InjectionMode,
    generate_mcp_context,
    generate_api_system_prompt,
)
from src.python.common.core.context_injection.rule_retrieval import (
    RuleFilter,
    RuleRetrievalResult,
)
from src.python.common.core.memory import (
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
)


@pytest.fixture
def mock_memory_manager():
    """Create mock MemoryManager."""
    return Mock()


@pytest.fixture
def mock_rule_retrieval():
    """Create mock RuleRetrieval."""
    mock = Mock()
    mock.get_rules = AsyncMock()
    return mock


@pytest.fixture
def sample_rules():
    """Create sample memory rules for testing."""
    now = datetime.now(timezone.utc)

    return [
        MemoryRule(
            id="rule-1",
            category=MemoryCategory.BEHAVIOR,
            name="Atomic Commits",
            rule="Always make atomic commits with clear messages",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["git", "development"],
            created_at=now,
            updated_at=now,
        ),
        MemoryRule(
            id="rule-2",
            category=MemoryCategory.PREFERENCE,
            name="Use UV",
            rule="Use uv for Python package management",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            created_at=now,
            updated_at=now,
        ),
        MemoryRule(
            id="rule-3",
            category=MemoryCategory.BEHAVIOR,
            name="Test Coverage",
            rule="Maintain 80% test coverage minimum",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["testing"],
            created_at=now,
            updated_at=now,
        ),
        MemoryRule(
            id="rule-4",
            category=MemoryCategory.PREFERENCE,
            name="Code Style",
            rule="Follow PEP 8 style guidelines",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "style"],
            created_at=now,
            updated_at=now,
        ),
    ]


@pytest.mark.asyncio
class TestSystemPromptInjector:
    """Test suite for SystemPromptInjector class."""

    async def test_initialization(self, mock_memory_manager):
        """Test SystemPromptInjector initialization."""
        injector = SystemPromptInjector(mock_memory_manager)

        assert injector.memory_manager == mock_memory_manager
        assert injector.rule_retrieval is not None
        assert injector.adapter is not None
        assert injector.token_budget_manager is not None

    async def test_initialization_with_dependencies(
        self, mock_memory_manager, mock_rule_retrieval
    ):
        """Test initialization with provided dependencies."""
        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        assert injector.rule_retrieval == mock_rule_retrieval

    async def test_generate_system_prompt_basic(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test basic system prompt generation."""
        # Setup mock
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        config = SystemPromptConfig(token_budget=15000, injection_mode=InjectionMode.MCP)

        prompt = await injector.generate_system_prompt(config)

        # Verify prompt was generated
        assert prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Verify content structure
        assert "# Memory System Context" in prompt or "# Project Memory Rules" in prompt
        assert "CRITICAL RULES" in prompt
        assert "GUIDELINES" in prompt or "DEFAULT" in prompt

        # Verify rules are included
        assert "Atomic Commits" in prompt
        assert "Use UV" in prompt

    async def test_generate_system_prompt_with_filter(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test system prompt generation with rule filter."""
        # Filter for only absolute rules
        absolute_rules = [r for r in sample_rules if r.authority == AuthorityLevel.ABSOLUTE]

        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=absolute_rules,
            total_count=len(absolute_rules),
            filtered_count=len(absolute_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        filter = RuleFilter(authority=AuthorityLevel.ABSOLUTE)
        prompt = await injector.generate_system_prompt(filter=filter)

        # Verify only absolute rules included
        assert "Atomic Commits" in prompt
        assert "Test Coverage" in prompt
        # Default rules should not be present
        assert "Use UV" not in prompt or "GUIDELINES" not in prompt

    async def test_generate_system_prompt_empty_rules(
        self, mock_memory_manager, mock_rule_retrieval
    ):
        """Test system prompt generation with no rules."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=[], total_count=0, filtered_count=0
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        prompt = await injector.generate_system_prompt()

        # Should return empty string when no rules
        assert prompt == ""

    async def test_compact_formatting(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test compact formatting option."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        # Generate with compact formatting
        config_compact = SystemPromptConfig(
            token_budget=15000, compact_format=True, include_metadata=False
        )
        prompt_compact = await injector.generate_system_prompt(config_compact)

        # Generate without compact formatting
        config_normal = SystemPromptConfig(
            token_budget=15000, compact_format=False, include_metadata=False
        )
        prompt_normal = await injector.generate_system_prompt(config_normal)

        # Compact should be shorter
        assert len(prompt_compact) < len(prompt_normal)

        # Compact should have shorter headers
        assert "## CRITICAL RULES" in prompt_compact
        assert "## GUIDELINES" in prompt_compact

        # Normal should have full headers
        assert (
            "## CRITICAL RULES (Always Follow)" in prompt_normal
            or "## DEFAULT GUIDELINES" in prompt_normal
        )

    async def test_metadata_header_inclusion(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test metadata header inclusion/exclusion."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        # With metadata
        config_with = SystemPromptConfig(include_metadata=True)
        prompt_with = await injector.generate_system_prompt(config_with)

        # Without metadata
        config_without = SystemPromptConfig(include_metadata=False)
        prompt_without = await injector.generate_system_prompt(config_without)

        # Verify metadata present/absent
        assert "<!-- Generated for" in prompt_with
        assert "<!-- Token budget:" in prompt_with
        assert "<!-- Rules:" in prompt_with

        assert "<!--" not in prompt_without

    async def test_token_budget_enforcement(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test token budget enforcement."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        # Use small token budget
        config = SystemPromptConfig(token_budget=500, include_metadata=False)
        prompt = await injector.generate_system_prompt(config)

        # Verify prompt generated (even if truncated)
        assert prompt

        # Token count should be reasonable (allowing for estimation variance)
        estimated_tokens = injector.adapter.estimate_token_count(prompt)
        assert estimated_tokens <= 600  # Allow 20% variance

    async def test_injection_mode_budgets(self, mock_memory_manager):
        """Test recommended budgets for different injection modes."""
        injector = SystemPromptInjector(mock_memory_manager)

        mcp_budget = injector.get_recommended_budget(InjectionMode.MCP)
        api_budget = injector.get_recommended_budget(InjectionMode.API)
        custom_budget = injector.get_recommended_budget(InjectionMode.CUSTOM)

        # Verify budget ordering
        assert api_budget < mcp_budget  # API most conservative
        assert mcp_budget < custom_budget  # Custom most flexible

        # Verify specific values
        assert api_budget == 10000
        assert mcp_budget == 15000
        assert custom_budget == 20000

    async def test_inject_to_file(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test writing system prompt to file."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "system_prompt.md"
            config = SystemPromptConfig(injection_mode=InjectionMode.MCP)

            success = await injector.inject_to_file(output_path, config)

            # Verify success
            assert success
            assert output_path.exists()

            # Verify content
            content = output_path.read_text()
            assert content
            assert "CRITICAL RULES" in content
            assert "Atomic Commits" in content

    async def test_inject_to_file_empty_rules(
        self, mock_memory_manager, mock_rule_retrieval
    ):
        """Test inject_to_file with no rules."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=[], total_count=0, filtered_count=0
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "system_prompt.md"

            success = await injector.inject_to_file(output_path)

            # Should fail when no content
            assert not success
            assert not output_path.exists()

    async def test_inject_to_file_directory_creation(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test that output directory is created if needed."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "system_prompt.md"

            success = await injector.inject_to_file(output_path)

            # Verify directory created
            assert success
            assert output_path.parent.exists()
            assert output_path.exists()

    async def test_apply_compact_formatting(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test compact formatting transformation."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        # Generate formatted context
        formatted = injector.adapter.format_rules(sample_rules, token_budget=15000)

        # Apply compact formatting
        compact = injector._apply_compact_formatting(formatted)

        # Verify transformations
        assert compact.token_count < formatted.token_count
        assert compact.metadata.get("compact") is True

        # Verify header shortening
        assert "## CRITICAL RULES" in compact.content or "## GUIDELINES" in compact.content
        assert "Always Follow" not in compact.content

    async def test_rules_sorting_by_authority(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test that absolute rules appear before default rules."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        prompt = await injector.generate_system_prompt()

        # Find positions of absolute and default rules
        critical_pos = prompt.find("## CRITICAL")
        guidelines_pos = max(prompt.find("## GUIDELINES"), prompt.find("## DEFAULT"))

        # Critical rules should come before guidelines
        assert critical_pos < guidelines_pos


@pytest.mark.asyncio
class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    async def test_generate_mcp_context(
        self, mock_memory_manager, sample_rules
    ):
        """Test generate_mcp_context convenience function."""
        with patch(
            "src.python.common.core.context_injection.system_prompt_injector.RuleRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = Mock()
            mock_retrieval.get_rules = AsyncMock(
                return_value=RuleRetrievalResult(
                    rules=sample_rules,
                    total_count=len(sample_rules),
                    filtered_count=len(sample_rules)
                )
            )
            mock_retrieval_class.return_value = mock_retrieval

            context = await generate_mcp_context(mock_memory_manager)

            # Verify context generated
            assert context
            assert isinstance(context, str)
            assert "CRITICAL" in context

            # Verify MCP-specific configuration used
            assert "<!--" in context  # Metadata included

    async def test_generate_mcp_context_with_budget(
        self, mock_memory_manager, sample_rules
    ):
        """Test generate_mcp_context with custom token budget."""
        with patch(
            "src.python.common.core.context_injection.system_prompt_injector.RuleRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = Mock()
            mock_retrieval.get_rules = AsyncMock(
                return_value=RuleRetrievalResult(
                    rules=sample_rules,
                    total_count=len(sample_rules),
                    filtered_count=len(sample_rules)
                )
            )
            mock_retrieval_class.return_value = mock_retrieval

            context = await generate_mcp_context(mock_memory_manager, token_budget=8000)

            assert context
            # Should respect smaller budget
            assert len(context) < 10000  # Rough check

    async def test_generate_api_system_prompt(
        self, mock_memory_manager, sample_rules
    ):
        """Test generate_api_system_prompt convenience function."""
        with patch(
            "src.python.common.core.context_injection.system_prompt_injector.RuleRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = Mock()
            mock_retrieval.get_rules = AsyncMock(
                return_value=RuleRetrievalResult(
                    rules=sample_rules,
                    total_count=len(sample_rules),
                    filtered_count=len(sample_rules)
                )
            )
            mock_retrieval_class.return_value = mock_retrieval

            prompt = await generate_api_system_prompt(mock_memory_manager)

            # Verify prompt generated
            assert prompt
            assert isinstance(prompt, str)
            assert "CRITICAL" in prompt

            # Verify API-specific configuration (compact, no metadata)
            assert "<!--" not in prompt  # No metadata

    async def test_generate_api_system_prompt_compact(
        self, mock_memory_manager, sample_rules
    ):
        """Test API system prompt is compact by default."""
        with patch(
            "src.python.common.core.context_injection.system_prompt_injector.RuleRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = Mock()
            mock_retrieval.get_rules = AsyncMock(
                return_value=RuleRetrievalResult(
                    rules=sample_rules,
                    total_count=len(sample_rules),
                    filtered_count=len(sample_rules)
                )
            )
            mock_retrieval_class.return_value = mock_retrieval

            # Compact version (default)
            prompt_compact = await generate_api_system_prompt(
                mock_memory_manager, compact=True
            )

            # Non-compact version
            prompt_normal = await generate_api_system_prompt(
                mock_memory_manager, compact=False
            )

            # Compact should be shorter
            assert len(prompt_compact) < len(prompt_normal)

    async def test_generate_api_system_prompt_with_filter(
        self, mock_memory_manager, sample_rules
    ):
        """Test API system prompt with rule filter."""
        absolute_rules = [r for r in sample_rules if r.authority == AuthorityLevel.ABSOLUTE]

        with patch(
            "src.python.common.core.context_injection.system_prompt_injector.RuleRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = Mock()
            mock_retrieval.get_rules = AsyncMock(
                return_value=RuleRetrievalResult(
                    rules=absolute_rules,
                    total_count=len(absolute_rules),
                    filtered_count=len(absolute_rules)
                )
            )
            mock_retrieval_class.return_value = mock_retrieval

            filter = RuleFilter(authority=AuthorityLevel.ABSOLUTE)
            prompt = await generate_api_system_prompt(
                mock_memory_manager, filter=filter
            )

            # Should only contain absolute rules
            assert "Atomic Commits" in prompt
            assert "Test Coverage" in prompt


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_very_small_token_budget(
        self, mock_memory_manager, mock_rule_retrieval, sample_rules
    ):
        """Test behavior with extremely small token budget."""
        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=sample_rules,
            total_count=len(sample_rules),
            filtered_count=len(sample_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        # Extremely small budget
        config = SystemPromptConfig(token_budget=100, include_metadata=False)
        prompt = await injector.generate_system_prompt(config)

        # Should still generate something (even if minimal)
        assert prompt
        # Should contain at least header
        assert "#" in prompt

    async def test_large_number_of_rules(
        self, mock_memory_manager, mock_rule_retrieval
    ):
        """Test with large number of rules."""
        now = datetime.now(timezone.utc)

        # Create 100 rules
        many_rules = [
            MemoryRule(
                id=f"rule-{i}",
                category=MemoryCategory.BEHAVIOR,
                name=f"Rule {i}",
                rule=f"This is rule number {i}",
                authority=AuthorityLevel.DEFAULT,
                scope=["test"],
                created_at=now,
                updated_at=now,
            )
            for i in range(100)
        ]

        mock_rule_retrieval.get_rules.return_value = RuleRetrievalResult(
            rules=many_rules,
            total_count=len(many_rules),
            filtered_count=len(many_rules)
        )

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        config = SystemPromptConfig(token_budget=15000)
        prompt = await injector.generate_system_prompt(config)

        # Verify prompt generated
        assert prompt
        # Not all rules will fit in budget
        assert "Rule 0" in prompt or "Rule 1" in prompt

    async def test_exception_handling_in_inject_to_file(
        self, mock_memory_manager, mock_rule_retrieval
    ):
        """Test exception handling in inject_to_file."""
        # Mock to raise exception
        mock_rule_retrieval.get_rules.side_effect = Exception("Test error")

        injector = SystemPromptInjector(
            mock_memory_manager, rule_retrieval=mock_rule_retrieval
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "system_prompt.md"

            # Should handle exception gracefully
            success = await injector.inject_to_file(output_path)

            assert not success
            # File should not be created on error
            assert not output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
