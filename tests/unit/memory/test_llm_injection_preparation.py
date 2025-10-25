"""
Comprehensive LLM injection preparation tests for memory rules system (Task 324.6).

This module tests:
- Rule formatting for LLM context injection
- Proper serialization and context structuring
- Token optimization for LLM injection
- Semantic meaning preservation and authority level retention
- Different LLM context formats (system prompt, markdown, JSON)
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from common.memory.claude_integration import ClaudeCodeIntegration
from common.memory.token_counter import TokenCounter
from common.memory.types import (
    AuthorityLevel,
    ClaudeCodeSession,
    MemoryCategory,
    MemoryContext,
    MemoryRule,
)


@pytest.fixture
def token_counter():
    """Create a token counter for testing."""
    return TokenCounter()


@pytest.fixture
def integration(token_counter):
    """Create a ClaudeCodeIntegration instance for testing."""
    return ClaudeCodeIntegration(
        token_counter=token_counter,
        max_memory_tokens=5000
    )


@pytest.fixture
def sample_rules() -> list[MemoryRule]:
    """Create sample rules for testing."""
    return [
        MemoryRule(
            rule="Always write unit tests before implementation",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "testing"],
            tags=["tdd", "quality"],
        ),
        MemoryRule(
            rule="Prefer async/await for IO operations",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            tags=["async", "performance"],
        ),
        MemoryRule(
            rule="Use uv for package management",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "package-management"],
            tags=["tools", "python"],
        ),
        MemoryRule(
            rule="User name is Chris",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
            tags=["identity"],
        ),
    ]


@pytest.fixture
def memory_context() -> MemoryContext:
    """Create a memory context for testing."""
    return MemoryContext(
        session_id="test-session-001",
        project_name="workspace-qdrant-mcp",
        project_path="/path/to/workspace-qdrant-mcp",
        user_name="Chris",
        active_scopes=["python", "testing"],
    )


class TestRuleFormattingForLLMContext:
    """Test rule formatting for LLM context injection."""

    def test_system_prompt_format(self, integration, sample_rules, memory_context):
        """Test formatting rules as system prompt."""
        system_prompt = integration.create_system_prompt_injection(
            sample_rules, memory_context
        )

        # Verify structure
        assert "You are Claude Code with memory-driven behavior" in system_prompt
        assert "ABSOLUTE RULES (Always follow):" in system_prompt
        assert "DEFAULT RULES (Follow unless overridden):" in system_prompt
        assert "CONTEXT:" in system_prompt

        # Verify absolute rules are present
        assert "Always write unit tests before implementation" in system_prompt
        assert "User name is Chris" in system_prompt

        # Verify default rules are present
        assert "Prefer async/await for IO operations" in system_prompt
        assert "Use uv for package management" in system_prompt

        # Verify context information
        assert "User: Chris" in system_prompt
        assert "Project: workspace-qdrant-mcp" in system_prompt

    def test_markdown_format(self, integration, sample_rules, memory_context):
        """Test formatting rules as markdown."""
        markdown_content = integration._generate_injection_content(
            sample_rules, memory_context
        )

        # Verify markdown structure
        assert "# Memory-Driven Behavior Rules" in markdown_content
        assert "## Absolute Rules (Non-negotiable)" in markdown_content
        assert "## Default Rules (Override if needed)" in markdown_content

        # Verify bullet points
        assert markdown_content.count("- ") >= len(sample_rules)

        # Verify scope information is included
        assert "(Scope:" in markdown_content or "Scope:" in markdown_content

    def test_format_preserves_rule_order(self, integration, memory_context):
        """Test that formatting preserves rule ordering."""
        rules = [
            MemoryRule(
                rule="First absolute rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Second absolute rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="First default rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        formatted = integration._generate_injection_content(rules, memory_context)

        # Find positions
        first_abs_pos = formatted.find("First absolute rule")
        second_abs_pos = formatted.find("Second absolute rule")
        first_def_pos = formatted.find("First default rule")

        # Verify absolute rules appear before default rules
        assert first_abs_pos < first_def_pos
        assert second_abs_pos < first_def_pos

    def test_format_handles_empty_rules(self, integration, memory_context):
        """Test formatting with no rules."""
        formatted = integration._generate_injection_content([], memory_context)

        # Should still have header and context
        assert "# Memory-Driven Behavior Rules" in formatted
        assert "User: Chris" in formatted
        assert "Project: workspace-qdrant-mcp" in formatted

        # Should not have rule sections
        assert "## Absolute Rules" not in formatted
        assert "## Default Rules" not in formatted

    def test_format_with_special_characters(self, integration, memory_context):
        """Test formatting rules with special characters."""
        rules = [
            MemoryRule(
                rule="Use `backticks` for code references",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule='Use "quotes" for string literals',
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Escape special chars: $, *, [, ]",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        formatted = integration._generate_injection_content(rules, memory_context)

        # Verify special characters are preserved
        assert "`backticks`" in formatted
        assert '"quotes"' in formatted
        assert "$, *, [, ]" in formatted


class TestSerializationAndContextStructuring:
    """Test proper serialization and context structuring."""

    def test_rule_to_dict_serialization(self, sample_rules):
        """Test that rules serialize to dict correctly."""
        for rule in sample_rules:
            rule_dict = rule.to_dict()

            # Verify all required fields
            assert "id" in rule_dict
            assert "rule" in rule_dict
            assert "category" in rule_dict
            assert "authority" in rule_dict
            assert "scope" in rule_dict
            assert "tags" in rule_dict
            assert "created_at" in rule_dict

            # Verify enums are serialized as strings
            assert isinstance(rule_dict["category"], str)
            assert isinstance(rule_dict["authority"], str)

            # Verify lists are preserved
            assert isinstance(rule_dict["scope"], list)
            assert isinstance(rule_dict["tags"], list)

    def test_rule_dict_roundtrip(self, sample_rules):
        """Test that rules can be serialized and deserialized."""
        for rule in sample_rules:
            rule_dict = rule.to_dict()
            restored_rule = MemoryRule.from_dict(rule_dict)

            # Verify all fields match
            assert restored_rule.rule == rule.rule
            assert restored_rule.category == rule.category
            assert restored_rule.authority == rule.authority
            assert restored_rule.scope == rule.scope
            assert restored_rule.tags == rule.tags

    def test_context_to_scope_list(self, memory_context):
        """Test context conversion to scope list."""
        scope_list = memory_context.to_scope_list()

        # Verify project scope
        assert "project:workspace-qdrant-mcp" in scope_list

        # Verify user scope
        assert "user:Chris" in scope_list

        # Verify active scopes
        assert "python" in scope_list
        assert "testing" in scope_list

    def test_context_structuring_with_files(self):
        """Test context structuring with active files."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/path/to/project",
            user_name="Chris",
            project_name="test-project",
            active_files=[
                "/path/to/project/main.py",
                "/path/to/project/test_main.py",
                "/path/to/project/README.md",
            ],
        )

        integration = ClaudeCodeIntegration(
            token_counter=TokenCounter(),
            max_memory_tokens=5000
        )

        import asyncio
        context = asyncio.run(integration._create_memory_context(session))
        scope_list = context.to_scope_list()

        # Verify file type scopes
        assert "filetype:py" in scope_list
        assert "filetype:md" in scope_list

    def test_serialization_preserves_metadata(self):
        """Test that serialization preserves metadata."""
        metadata = {
            "source_file": "CLAUDE.md",
            "line_number": 42,
            "confidence": 0.95,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }

        rule = MemoryRule(
            rule="Test rule with metadata",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            metadata=metadata,
        )

        rule_dict = rule.to_dict()
        assert rule_dict["metadata"] == metadata

        # Roundtrip
        restored = MemoryRule.from_dict(rule_dict)
        assert restored.metadata == metadata


class TestTokenOptimization:
    """Test token optimization for LLM injection."""

    def test_optimize_rules_respects_token_limit(self, token_counter):
        """Test that optimization respects token limits."""
        # Create many rules to exceed limit
        rules = [
            MemoryRule(
                rule=f"Rule number {i} with some text content to increase token count",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
            for i in range(100)
        ]

        selected, usage = token_counter.optimize_rules_for_context(
            rules, max_tokens=1000, preserve_absolute=False
        )

        # Should respect token limit
        assert usage.total_tokens <= 1000
        assert len(selected) < len(rules)

    def test_optimize_preserves_absolute_rules(self, token_counter):
        """Test that absolute rules are preserved during optimization."""
        absolute_rules = [
            MemoryRule(
                rule=f"Absolute rule {i} " * 20,  # Make them long
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            )
            for i in range(5)
        ]

        default_rules = [
            MemoryRule(
                rule=f"Default rule {i} " * 20,
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
            for i in range(50)
        ]

        all_rules = absolute_rules + default_rules

        selected, usage = token_counter.optimize_rules_for_context(
            all_rules, max_tokens=2000, preserve_absolute=True
        )

        # All absolute rules should be present
        selected_absolute = [
            r for r in selected if r.authority == AuthorityLevel.ABSOLUTE
        ]
        assert len(selected_absolute) == 5

    def test_optimization_prioritizes_recent_rules(self, token_counter):
        """Test that optimization prioritizes recently used rules."""
        now = datetime.now(timezone.utc)

        rules = [
            MemoryRule(
                rule="Recent rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                created_at=now - timedelta(days=1),
                last_used=now,
            ),
            MemoryRule(
                rule="Old unused rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                created_at=now - timedelta(days=365),
                last_used=now - timedelta(days=365),
            ),
        ]

        selected, usage = token_counter.optimize_rules_for_context(
            rules, max_tokens=100, preserve_absolute=False
        )

        # If only one rule fits, it should be the recent one
        if len(selected) == 1:
            assert selected[0].rule == "Recent rule"

    def test_token_counting_accuracy(self, token_counter):
        """Test token counting accuracy for different text patterns."""
        test_cases = [
            ("Simple text", 2),  # Approximate
            ("Text with punctuation!", 3),
            ("Code example: `function()`", 5),
            ("Multi-word sentence here", 3),
        ]

        for text, expected_min_tokens in test_cases:
            rule = MemoryRule(
                rule=text,
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
            token_count = token_counter.count_rule_tokens(rule)
            assert token_count >= expected_min_tokens


class TestSemanticMeaningPreservation:
    """Test semantic meaning preservation in formatted output."""

    def test_rule_text_preserved_exactly(self, integration, memory_context):
        """Test that rule text is preserved exactly in formatted output."""
        original_text = "Always commit after changes with descriptive messages"
        rule = MemoryRule(
            rule=original_text,
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        formatted = integration._generate_injection_content([rule], memory_context)

        # Rule text should appear exactly as written
        assert original_text in formatted

    def test_scope_context_preserved(self, integration, memory_context):
        """Test that scope context is preserved in formatted output."""
        rule = MemoryRule(
            rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing", "pytest"],
        )

        formatted = integration._generate_injection_content([rule], memory_context)

        # Verify scope information is preserved
        for scope_item in rule.scope:
            assert scope_item in formatted

    def test_multiline_rules_preserved(self, integration, memory_context):
        """Test that multiline rules are preserved correctly."""
        multiline_rule = """When writing code:
1. Write tests first
2. Implement functionality
3. Refactor if needed"""

        rule = MemoryRule(
            rule=multiline_rule,
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        formatted = integration._generate_injection_content([rule], memory_context)

        # All lines should be present
        assert "When writing code:" in formatted
        assert "1. Write tests first" in formatted
        assert "2. Implement functionality" in formatted
        assert "3. Refactor if needed" in formatted

    def test_semantic_grouping_preserved(self, integration, memory_context):
        """Test that semantic grouping by category is preserved."""
        rules = [
            MemoryRule(
                rule="Behavior rule 1",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Preference rule 1",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Behavior rule 2",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        formatted = integration._generate_injection_content(rules, memory_context)

        # All rules should be present regardless of category
        assert "Behavior rule 1" in formatted
        assert "Preference rule 1" in formatted
        assert "Behavior rule 2" in formatted


class TestAuthorityLevelRetention:
    """Test authority level retention in formatted output."""

    def test_absolute_authority_clearly_marked(self, integration, memory_context):
        """Test that absolute authority is clearly marked."""
        rule = MemoryRule(
            rule="Never skip error handling",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        # System prompt format
        system_prompt = integration.create_system_prompt_injection([rule], memory_context)
        assert "ABSOLUTE RULES (Always follow):" in system_prompt
        assert "Never skip error handling" in system_prompt

        # Markdown format
        markdown = integration._generate_injection_content([rule], memory_context)
        assert "## Absolute Rules (Non-negotiable)" in markdown
        assert "Never skip error handling" in markdown

    def test_default_authority_clearly_marked(self, integration, memory_context):
        """Test that default authority is clearly marked."""
        rule = MemoryRule(
            rule="Prefer descriptive variable names",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # System prompt format
        system_prompt = integration.create_system_prompt_injection([rule], memory_context)
        assert "DEFAULT RULES (Follow unless overridden):" in system_prompt
        assert "Prefer descriptive variable names" in system_prompt

        # Markdown format
        markdown = integration._generate_injection_content([rule], memory_context)
        assert "## Default Rules (Override if needed)" in markdown
        assert "Prefer descriptive variable names" in markdown

    def test_authority_level_separation(self, integration, memory_context):
        """Test that authority levels are clearly separated."""
        rules = [
            MemoryRule(
                rule="Absolute rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Default rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        formatted = integration._generate_injection_content(rules, memory_context)

        # Find section positions
        abs_section_pos = formatted.find("## Absolute Rules")
        def_section_pos = formatted.find("## Default Rules")

        # Absolute section should come before default
        assert abs_section_pos < def_section_pos

        # Each rule should be in its correct section
        abs_rule_pos = formatted.find("Absolute rule")
        def_rule_pos = formatted.find("Default rule")

        assert abs_section_pos < abs_rule_pos < def_section_pos
        assert def_section_pos < def_rule_pos

    def test_authority_retained_in_serialization(self):
        """Test that authority level is retained through serialization."""
        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        # Serialize and deserialize
        rule_dict = rule.to_dict()
        assert rule_dict["authority"] == "absolute"

        restored = MemoryRule.from_dict(rule_dict)
        assert restored.authority == AuthorityLevel.ABSOLUTE


class TestDifferentLLMContextFormats:
    """Test with different LLM context formats."""

    def test_json_format_structure(self, sample_rules):
        """Test JSON format structure for API-based injection."""
        import json

        # Convert rules to JSON-serializable format
        json_rules = [rule.to_dict() for rule in sample_rules]

        # Verify JSON serialization works
        json_str = json.dumps(json_rules, indent=2)
        parsed = json.loads(json_str)

        assert len(parsed) == len(sample_rules)

        # Verify each rule has required fields
        for rule_data in parsed:
            assert "rule" in rule_data
            assert "category" in rule_data
            assert "authority" in rule_data

    def test_xml_like_format(self, integration, sample_rules, memory_context):
        """Test XML-like format for legacy systems."""
        formatted = integration._generate_injection_content(sample_rules, memory_context)

        # While not XML, markdown structure should be parseable
        lines = formatted.split("\n")

        # Should have clear section markers
        assert any("# Memory-Driven Behavior Rules" in line for line in lines)
        assert any("## Absolute Rules" in line for line in lines)
        assert any("## Default Rules" in line for line in lines)

    def test_plain_text_format(self, integration, sample_rules, memory_context):
        """Test plain text format for simple LLMs."""
        system_prompt = integration.create_system_prompt_injection(
            sample_rules, memory_context
        )

        # Should be readable plain text
        lines = system_prompt.split("\n")

        # Should not have complex markdown
        assert all("<" not in line or ">" not in line for line in lines)

        # Should have clear structure
        assert any("ABSOLUTE RULES" in line for line in lines)
        assert any("DEFAULT RULES" in line for line in lines)

    def test_compact_format_for_small_context(self, integration, memory_context):
        """Test compact format for small context windows."""
        # Create minimal rules
        rules = [
            MemoryRule(
                rule="Use Python 3.11+",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        formatted = integration._generate_injection_content(rules, memory_context)

        # Should be concise but complete
        assert "Use Python 3.11+" in formatted
        assert len(formatted) < 500  # Should be compact

    def test_structured_format_with_metadata(self, integration):
        """Test structured format including metadata."""
        rule = MemoryRule(
            rule="Use type hints",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            metadata={
                "rationale": "Improves code quality",
                "applies_to": ["python"],
            },
        )

        rule_dict = rule.to_dict()

        # Verify metadata is preserved in structured format
        assert "metadata" in rule_dict
        assert rule_dict["metadata"]["rationale"] == "Improves code quality"

    @pytest.mark.asyncio
    async def test_session_injection_format(self, integration, sample_rules):
        """Test format used for actual session injection."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            session = ClaudeCodeSession(
                session_id="format-test",
                workspace_path=tmpdir,
                user_name="Chris",
                project_name="test-project",
            )

            result = await integration.initialize_session(session, sample_rules)

            # Should successfully inject
            assert result.success is True
            assert result.rules_injected > 0

            # Check if injection file was created
            import os
            injection_file = os.path.join(tmpdir, ".claude", "memory.md")

            # If file exists, verify format
            if os.path.exists(injection_file):
                with open(injection_file) as f:
                    content = f.read()

                # Should have proper markdown structure
                assert "# Memory-Driven Behavior Rules" in content


class TestEdgeCases:
    """Test edge cases in LLM injection preparation."""

    def test_empty_rule_list(self, integration, memory_context):
        """Test handling of empty rule list."""
        formatted = integration._generate_injection_content([], memory_context)

        # Should handle gracefully
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "# Memory-Driven Behavior Rules" in formatted

    def test_very_long_rules(self, integration, memory_context):
        """Test handling of very long rules."""
        long_rule = "This is a very long rule. " * 100

        rule = MemoryRule(
            rule=long_rule,
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        formatted = integration._generate_injection_content([rule], memory_context)

        # Should include the full rule
        assert long_rule in formatted

    def test_unicode_characters(self, integration, memory_context):
        """Test handling of Unicode characters."""
        rule = MemoryRule(
            rule="Use emojis sparingly: 🚀 ✅ ❌",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        formatted = integration._generate_injection_content([rule], memory_context)

        # Should preserve Unicode
        assert "🚀" in formatted
        assert "✅" in formatted
        assert "❌" in formatted

    def test_rules_without_scope(self, integration, memory_context):
        """Test handling of rules without scope (global rules)."""
        rule = MemoryRule(
            rule="Global rule applies everywhere",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=[],  # Empty scope = global
        )

        formatted = integration._generate_injection_content([rule], memory_context)

        # Should include the rule
        assert "Global rule applies everywhere" in formatted

    def test_mixed_authority_and_categories(self, integration, memory_context):
        """Test complex mix of authority levels and categories."""
        rules = [
            MemoryRule(
                rule="Absolute behavior",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Default behavior",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Absolute preference",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Default preference",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        formatted = integration._generate_injection_content(rules, memory_context)

        # All rules should be present
        for rule in rules:
            assert rule.rule in formatted

        # Should be properly grouped by authority
        abs_section = formatted.find("## Absolute Rules")
        def_section = formatted.find("## Default Rules")
        assert abs_section < def_section
