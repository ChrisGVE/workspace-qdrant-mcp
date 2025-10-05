"""
Unit tests for LLM-specific formatters.

Tests all formatter adapters (Claude Code, GitHub Codex, Google Gemini)
and the format manager.
"""

from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.formatters import (
    ClaudeCodeAdapter,
    FormatManager,
    FormatType,
    FormattedContext,
    GitHubCodexAdapter,
    GoogleGeminiAdapter,
    LLMToolAdapter,
    ToolCapabilities,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


# Test fixtures
@pytest.fixture
def sample_absolute_rule():
    """Create a sample absolute authority rule."""
    return MemoryRule(
        id="rule_absolute_1",
        name="atomic_commits",
        rule="Always make atomic commits with clear, descriptive messages",
        category=MemoryCategory.BEHAVIOR,
        authority=AuthorityLevel.ABSOLUTE,
        scope=["git"],
        source="user_explicit",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions=None,
        replaces=[],
        metadata={"priority": 100},
    )


@pytest.fixture
def sample_default_rule():
    """Create a sample default authority rule."""
    return MemoryRule(
        id="rule_default_1",
        name="code_review",
        rule="Submit PRs for review before merging to main",
        category=MemoryCategory.BEHAVIOR,
        authority=AuthorityLevel.DEFAULT,
        scope=["git", "collaboration"],
        source="conversational_future",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions={"condition": "when working with team"},
        replaces=[],
        metadata={"priority": 50},
    )


@pytest.fixture
def sample_preference_rule():
    """Create a sample preference rule."""
    return MemoryRule(
        id="rule_absolute_2",
        name="python_package_manager",
        rule="Use uv for all Python package management operations",
        category=MemoryCategory.PREFERENCE,
        authority=AuthorityLevel.ABSOLUTE,
        scope=["python"],
        source="user_explicit",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions=None,
        replaces=[],
        metadata={"priority": 90},
    )


# Base Adapter Tests
class TestLLMToolAdapter:
    """Test base LLMToolAdapter functionality."""

    def test_estimate_token_count(self):
        """Test token estimation."""
        adapter = ClaudeCodeAdapter()  # Use concrete implementation
        text = "This is a simple test. It has some punctuation!"
        tokens = adapter.estimate_token_count(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_separate_by_authority(
        self, sample_absolute_rule, sample_default_rule, sample_preference_rule
    ):
        """Test authority separation."""
        adapter = ClaudeCodeAdapter()
        rules = [sample_absolute_rule, sample_default_rule, sample_preference_rule]
        absolute, default = adapter.separate_by_authority(rules)

        assert len(absolute) == 2  # absolute_rule and preference_rule
        assert len(default) == 1  # default_rule
        assert all(r.authority == AuthorityLevel.ABSOLUTE for r in absolute)
        assert all(r.authority == AuthorityLevel.DEFAULT for r in default)

    def test_sort_by_priority(
        self, sample_absolute_rule, sample_default_rule, sample_preference_rule
    ):
        """Test priority sorting."""
        adapter = ClaudeCodeAdapter()
        rules = [sample_default_rule, sample_preference_rule, sample_absolute_rule]
        sorted_rules = adapter.sort_by_priority(rules)

        # Absolute rules should come first
        assert sorted_rules[0].authority == AuthorityLevel.ABSOLUTE
        assert sorted_rules[1].authority == AuthorityLevel.ABSOLUTE
        assert sorted_rules[2].authority == AuthorityLevel.DEFAULT


# Claude Code Adapter Tests
class TestClaudeCodeAdapter:
    """Test Claude Code adapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ClaudeCodeAdapter()
        assert adapter.capabilities.tool_name == "claude"
        assert adapter.capabilities.format_type == FormatType.MARKDOWN
        assert adapter.capabilities.supports_markdown is True

    def test_format_rules_basic(self, sample_absolute_rule, sample_default_rule):
        """Test basic rule formatting."""
        adapter = ClaudeCodeAdapter()
        rules = [sample_absolute_rule, sample_default_rule]
        formatted = adapter.format_rules(rules, token_budget=10000)

        assert isinstance(formatted, FormattedContext)
        assert formatted.tool_name == "claude"
        assert formatted.format_type == FormatType.MARKDOWN
        assert formatted.content.startswith("# Project Memory Rules")
        assert "CRITICAL RULES" in formatted.content
        assert "DEFAULT GUIDELINES" in formatted.content
        assert formatted.rules_included == 2
        assert formatted.rules_skipped == 0

    def test_format_rules_token_budget(
        self, sample_absolute_rule, sample_default_rule
    ):
        """Test token budget enforcement."""
        adapter = ClaudeCodeAdapter()
        rules = [sample_absolute_rule, sample_default_rule]

        # Very small budget - should skip default rules
        formatted = adapter.format_rules(rules, token_budget=100)

        # Absolute rules should always be included
        assert formatted.rules_included >= 1
        assert "atomic_commits" in formatted.content

    def test_format_single_rule(self, sample_absolute_rule):
        """Test single rule formatting."""
        adapter = ClaudeCodeAdapter()
        formatted_text = adapter._format_single_rule(sample_absolute_rule)

        assert "**atomic_commits**" in formatted_text
        assert "Always make atomic commits" in formatted_text
        assert "Scope: git" in formatted_text

    def test_validate_format(self, sample_absolute_rule):
        """Test format validation."""
        adapter = ClaudeCodeAdapter()
        rules = [sample_absolute_rule]
        formatted = adapter.format_rules(rules, token_budget=10000)

        assert adapter.validate_format(formatted) is True

        # Test invalid formats
        invalid_formatted = FormattedContext(
            tool_name="claude",
            format_type=FormatType.MARKDOWN,
            content="",  # Empty content
            token_count=0,
            rules_included=0,
            rules_skipped=0,
            metadata={},
        )
        assert adapter.validate_format(invalid_formatted) is False


# GitHub Codex Adapter Tests
class TestGitHubCodexAdapter:
    """Test GitHub Codex adapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = GitHubCodexAdapter()
        assert adapter.capabilities.tool_name == "codex"
        assert adapter.capabilities.format_type == FormatType.PLAIN_TEXT
        assert adapter.capabilities.supports_markdown is False
        assert adapter.capabilities.max_context_tokens == 4096

    def test_format_rules_basic(self, sample_absolute_rule, sample_default_rule):
        """Test basic rule formatting."""
        adapter = GitHubCodexAdapter()
        rules = [sample_absolute_rule, sample_default_rule]
        formatted = adapter.format_rules(rules, token_budget=5000)

        assert isinstance(formatted, FormattedContext)
        assert formatted.tool_name == "codex"
        assert formatted.format_type == FormatType.PLAIN_TEXT
        assert "PROJECT CONTEXT RULES" in formatted.content
        assert "=== CRITICAL RULES" in formatted.content
        assert "=== DEFAULT GUIDELINES ===" in formatted.content
        assert formatted.rules_included == 2

    def test_format_single_rule(self, sample_absolute_rule):
        """Test single rule formatting."""
        adapter = GitHubCodexAdapter()
        formatted_text = adapter._format_single_rule(sample_absolute_rule, number=1)

        assert "[1] atomic_commits" in formatted_text
        assert "Always make atomic commits" in formatted_text
        assert "Scope: git" in formatted_text

    def test_validate_format(self, sample_absolute_rule):
        """Test format validation."""
        adapter = GitHubCodexAdapter()
        rules = [sample_absolute_rule]
        formatted = adapter.format_rules(rules, token_budget=5000)

        assert adapter.validate_format(formatted) is True


# Google Gemini Adapter Tests
class TestGoogleGeminiAdapter:
    """Test Google Gemini adapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = GoogleGeminiAdapter()
        assert adapter.capabilities.tool_name == "gemini"
        assert adapter.capabilities.format_type == FormatType.SYSTEM_INSTRUCTION
        assert adapter.capabilities.max_context_tokens == 32000

    def test_format_rules_basic(self, sample_absolute_rule, sample_default_rule):
        """Test basic rule formatting."""
        adapter = GoogleGeminiAdapter()
        rules = [sample_absolute_rule, sample_default_rule]
        formatted = adapter.format_rules(rules, token_budget=10000)

        assert isinstance(formatted, FormattedContext)
        assert formatted.tool_name == "gemini"
        assert formatted.format_type == FormatType.SYSTEM_INSTRUCTION
        assert formatted.content.startswith("You are an AI assistant")
        assert "MANDATORY RULES" in formatted.content
        assert "RECOMMENDED GUIDELINES" in formatted.content
        assert formatted.rules_included == 2

    def test_format_single_rule(self, sample_absolute_rule):
        """Test single rule formatting."""
        adapter = GoogleGeminiAdapter()
        formatted_text = adapter._format_single_rule(sample_absolute_rule, number=1)

        assert "1. atomic_commits:" in formatted_text
        assert "Always make atomic commits" in formatted_text
        assert "[Scope: git]" in formatted_text

    def test_validate_format(self, sample_absolute_rule):
        """Test format validation."""
        adapter = GoogleGeminiAdapter()
        rules = [sample_absolute_rule]
        formatted = adapter.format_rules(rules, token_budget=10000)

        assert adapter.validate_format(formatted) is True


# Format Manager Tests
class TestFormatManager:
    """Test FormatManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = FormatManager()
        supported = manager.list_supported_tools()

        assert "claude" in supported
        assert "codex" in supported
        assert "gemini" in supported

    def test_get_adapter(self):
        """Test adapter retrieval."""
        manager = FormatManager()

        claude_adapter = manager.get_adapter("claude")
        assert isinstance(claude_adapter, ClaudeCodeAdapter)

        codex_adapter = manager.get_adapter("codex")
        assert isinstance(codex_adapter, GitHubCodexAdapter)

        gemini_adapter = manager.get_adapter("gemini")
        assert isinstance(gemini_adapter, GoogleGeminiAdapter)

        # Test unknown adapter
        unknown = manager.get_adapter("unknown")
        assert unknown is None

    def test_format_for_tool(self, sample_absolute_rule):
        """Test formatting via manager."""
        manager = FormatManager()
        rules = [sample_absolute_rule]

        # Claude formatting
        claude_formatted = manager.format_for_tool("claude", rules, 10000)
        assert claude_formatted.tool_name == "claude"
        assert "# Project Memory Rules" in claude_formatted.content

        # Codex formatting
        codex_formatted = manager.format_for_tool("codex", rules, 5000)
        assert codex_formatted.tool_name == "codex"
        assert "PROJECT CONTEXT RULES" in codex_formatted.content

        # Gemini formatting
        gemini_formatted = manager.format_for_tool("gemini", rules, 10000)
        assert gemini_formatted.tool_name == "gemini"
        assert "You are an AI assistant" in gemini_formatted.content

    def test_format_for_tool_invalid(self, sample_absolute_rule):
        """Test formatting for invalid tool."""
        manager = FormatManager()
        rules = [sample_absolute_rule]

        with pytest.raises(ValueError, match="No adapter registered"):
            manager.format_for_tool("invalid_tool", rules, 10000)

    def test_register_custom_adapter(self):
        """Test custom adapter registration."""

        class CustomAdapter(LLMToolAdapter):
            """Custom test adapter."""

            def __init__(self):
                capabilities = ToolCapabilities(
                    tool_name="custom",
                    format_type=FormatType.JSON,
                    max_context_tokens=8000,
                    supports_sections=False,
                    supports_markdown=False,
                    supports_priorities=False,
                    injection_method="api",
                )
                super().__init__(capabilities)

            def format_rules(self, rules, token_budget, options=None):
                return FormattedContext(
                    tool_name="custom",
                    format_type=FormatType.JSON,
                    content='{"rules": []}',
                    token_count=10,
                    rules_included=0,
                    rules_skipped=0,
                    metadata={},
                )

            def get_capabilities(self):
                return self.capabilities

            def validate_format(self, formatted_context):
                return True

        manager = FormatManager()
        custom_adapter = CustomAdapter()
        manager.register_adapter("custom", custom_adapter)

        assert "custom" in manager.list_supported_tools()
        retrieved = manager.get_adapter("custom")
        assert retrieved is custom_adapter


# Integration Tests
class TestFormatterIntegration:
    """Integration tests for complete formatting workflow."""

    def test_multi_rule_formatting(
        self, sample_absolute_rule, sample_default_rule, sample_preference_rule
    ):
        """Test formatting with multiple rules."""
        manager = FormatManager()
        rules = [sample_absolute_rule, sample_default_rule, sample_preference_rule]

        # Test all adapters
        for tool_name in ["claude", "codex", "gemini"]:
            formatted = manager.format_for_tool(tool_name, rules, 10000)
            assert formatted.rules_included == 3
            assert formatted.rules_skipped == 0
            assert len(formatted.content) > 0

    def test_token_budget_scenarios(
        self, sample_absolute_rule, sample_default_rule, sample_preference_rule
    ):
        """Test various token budget scenarios."""
        manager = FormatManager()
        rules = [sample_absolute_rule, sample_default_rule, sample_preference_rule]

        # Large budget - all rules included
        formatted = manager.format_for_tool("claude", rules, 100000)
        assert formatted.rules_included == 3

        # Medium budget - some rules may be skipped
        formatted = manager.format_for_tool("codex", rules, 1000)
        # Absolute rules should be prioritized
        assert formatted.rules_included >= 2  # At least absolute rules

        # Very small budget
        formatted = manager.format_for_tool("claude", rules, 50)
        # May skip some rules, but should not crash
        assert formatted.rules_included >= 0
