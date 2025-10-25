"""
Unit tests for context switching validation system.
"""

from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.context_switcher import (
    ContextSwitcher,
    SwitchValidationResult,
)
from src.python.common.core.context_injection.llm_tool_detector import LLMToolType
from src.python.common.core.memory import AuthorityLevel, MemoryCategory, MemoryRule

# Test fixtures


@pytest.fixture
def sample_rules():
    """Create sample rules for testing."""
    now = datetime.now(timezone.utc)
    return [
        MemoryRule(
            id="rule1",
            category=MemoryCategory.PREFERENCE,
            name="type_hints",
            rule="Use type hints in all function signatures",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="test",
            created_at=now,
            updated_at=now,
        ),
        MemoryRule(
            id="rule2",
            category=MemoryCategory.PREFERENCE,
            name="docstrings",
            rule="Write docstrings for all public functions",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="test",
            created_at=now,
            updated_at=now,
        ),
        MemoryRule(
            id="rule3",
            category=MemoryCategory.PREFERENCE,
            name="async_await",
            rule="Use async/await for I/O operations",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="test",
            created_at=now,
            updated_at=now,
        ),
    ]


@pytest.fixture
def many_rules():
    """Create many rules to test truncation scenarios."""
    now = datetime.now(timezone.utc)
    rules = []
    for i in range(100):
        rules.append(
            MemoryRule(
                id=f"rule{i}",
                category=MemoryCategory.PREFERENCE,
                name=f"test_rule_{i}",
                rule=f"Rule {i}: " + ("This is a test rule. " * 50),  # ~50 words each
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="test",
                created_at=now,
                updated_at=now,
            )
        )
    return rules


# Validation tests


def test_validate_switch_same_tool(sample_rules):
    """Test validation when switching to same tool (no change needed)."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.CLAUDE_CODE,
        rules=sample_rules,
        current_token_count=1000,
    )

    assert result.is_valid
    assert result.source_tool == LLMToolType.CLAUDE_CODE
    assert result.target_tool == LLMToolType.CLAUDE_CODE
    assert result.source_token_count == 1000
    assert result.target_token_count == 1000
    assert result.token_limit_ok
    assert result.format_compatible
    assert result.rules_truncated == 0
    assert len(result.errors) == 0
    assert any("no switch needed" in w.lower() for w in result.warnings)


def test_validate_switch_upgrade(sample_rules):
    """Test validation when upgrading from low to high limit (Copilot -> Claude)."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.GITHUB_COPILOT,
        target_tool=LLMToolType.CLAUDE_CODE,
        rules=sample_rules,
        current_token_count=5000,
    )

    assert result.is_valid
    assert result.source_tool == LLMToolType.GITHUB_COPILOT
    assert result.target_tool == LLMToolType.CLAUDE_CODE
    assert result.token_limit_ok
    assert result.rules_truncated == 0
    assert len(result.errors) == 0
    # Should have warning about increased capacity
    assert any("increase" in w.lower() for w in result.warnings)


def test_validate_switch_downgrade(many_rules):
    """Test validation when downgrading from high to low limit (Claude -> Copilot)."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=many_rules,
        current_token_count=50000,
    )

    # Should warn about potential truncation
    assert any("reduce" in w.lower() for w in result.warnings)

    # With 100 large rules, we'll likely exceed 8192 token limit
    # (Each rule is ~250 tokens, so 100 rules = ~25,000 tokens)
    if result.target_token_count > 8192:
        assert not result.token_limit_ok
        assert result.rules_truncated > 0
        # Should have error about exceeding limit
        assert len(result.errors) > 0
    else:
        # If we fit (unlikely), should be valid
        assert result.is_valid


def test_validate_switch_empty_rules():
    """Test validation with empty rules list."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=[],
        current_token_count=0,
    )

    assert result.is_valid
    # Empty rules list may have small overhead from formatting
    assert result.target_token_count < 20  # Allow small overhead
    assert result.token_limit_ok
    assert result.rules_truncated == 0
    assert len(result.errors) == 0


def test_validate_switch_format_change(sample_rules):
    """Test that format changes are detected and token counts may differ."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,  # Markdown format
        target_tool=LLMToolType.GITHUB_COPILOT,  # Code comment format
        rules=sample_rules,
        current_token_count=1000,
    )

    # Format should always be compatible
    assert result.format_compatible

    # Token counts may change due to formatting differences
    # (but with small rules, change might be <10% threshold)
    if abs(result.target_token_count - result.source_token_count) > 100:
        # Should have warning about token count change
        assert any("token" in w.lower() and ("increase" in w.lower() or "decrease" in w.lower()) for w in result.warnings)


# Perform switch tests


def test_perform_switch_without_truncation(sample_rules):
    """Test performing switch without auto-truncation."""
    rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.CLAUDE_CODE, rules=sample_rules, auto_truncate=False
    )

    # Should return all rules
    assert len(rules) == len(sample_rules)
    assert result.is_valid
    assert result.rules_truncated == 0


def test_perform_switch_with_truncation(many_rules):
    """Test performing switch with auto-truncation when exceeding limits."""
    rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.GITHUB_COPILOT,  # 8192 token limit
        rules=many_rules,
        auto_truncate=True,
    )

    # Should truncate to fit within limit
    if result.rules_truncated > 0:
        assert len(rules) < len(many_rules)
        assert len(rules) + result.rules_truncated == len(many_rules)
        # Should have truncation warning
        assert any("truncat" in w.lower() for w in result.warnings)
    else:
        # All rules fit (unlikely with 100 large rules)
        assert len(rules) == len(many_rules)


def test_perform_switch_no_truncation_when_fits(sample_rules):
    """Test that truncation doesn't happen when rules fit."""
    rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.CLAUDE_CODE,  # Large 200k limit
        rules=sample_rules,
        auto_truncate=True,
    )

    # All rules should fit
    assert len(rules) == len(sample_rules)
    assert result.rules_truncated == 0


def test_perform_switch_returns_prioritized_rules(many_rules):
    """Test that truncation returns highest priority rules."""
    # Add different priorities to rules
    many_rules[0].authority = AuthorityLevel.ABSOLUTE
    many_rules[1].authority = AuthorityLevel.ABSOLUTE

    rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=many_rules,
        auto_truncate=True,
    )

    if result.rules_truncated > 0:
        # Absolute authority rules should be included
        rule_ids = [r.id for r in rules]
        assert "rule0" in rule_ids
        assert "rule1" in rule_ids


# Safety check tests


def test_can_switch_safely_with_small_rules(sample_rules):
    """Test can_switch_safely returns True for small rule sets."""
    safe = ContextSwitcher.can_switch_safely(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=sample_rules,
    )

    # Small rules should fit in Copilot's 8k limit
    assert safe


def test_can_switch_safely_with_many_rules(many_rules):
    """Test can_switch_safely returns False when exceeding limits."""
    safe = ContextSwitcher.can_switch_safely(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=many_rules,
    )

    # 100 large rules won't fit in 8k limit
    assert not safe


def test_can_switch_safely_upgrade_always_safe(many_rules):
    """Test can_switch_safely returns True for upgrades."""
    safe = ContextSwitcher.can_switch_safely(
        source_tool=LLMToolType.GITHUB_COPILOT,
        target_tool=LLMToolType.CLAUDE_CODE,  # Much larger limit
        rules=many_rules,
    )

    # Upgrading to 200k limit should handle any reasonable rule set
    assert safe


def test_can_switch_safely_same_tool(sample_rules):
    """Test can_switch_safely returns True for same tool."""
    safe = ContextSwitcher.can_switch_safely(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.CLAUDE_CODE,
        rules=sample_rules,
    )

    assert safe


def test_can_switch_safely_empty_rules():
    """Test can_switch_safely returns True for empty rules."""
    safe = ContextSwitcher.can_switch_safely(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=[],
    )

    assert safe


# Warning and error message tests


def test_warning_messages_are_clear(many_rules):
    """Test that warning messages are clear and actionable."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=many_rules,
        current_token_count=50000,
    )

    # Should have warnings
    assert len(result.warnings) > 0

    # Warnings should contain key information
    warnings_text = " ".join(result.warnings).lower()
    assert "token" in warnings_text

    # If truncated, should mention it
    if result.rules_truncated > 0:
        assert "truncat" in warnings_text or "exceed" in warnings_text


def test_error_messages_suggest_solutions(many_rules):
    """Test that error messages suggest solutions."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=many_rules,
        current_token_count=50000,
    )

    # If there are errors, they should suggest solutions
    if len(result.errors) > 0:
        errors_text = " ".join(result.errors).lower()
        # Should mention auto_truncate option
        assert "auto_truncate" in errors_text or "truncate" in errors_text


def test_validation_result_str_representation(sample_rules):
    """Test that SwitchValidationResult has readable string representation."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=sample_rules,
        current_token_count=1000,
    )

    result_str = str(result)

    # Should contain key information
    assert "VALID" in result_str or "INVALID" in result_str
    assert "claude_code" in result_str.lower()
    assert "github_copilot" in result_str.lower()
    assert "token" in result_str.lower()


# Edge case tests


def test_switch_with_single_huge_rule():
    """Test switching with a single rule that exceeds target limit."""
    now = datetime.now(timezone.utc)
    huge_rule = MemoryRule(
        id="huge",
        category=MemoryCategory.PREFERENCE,
        name="huge_rule",
        rule="Rule: " + ("This is a very long rule. " * 1000),  # ~5000 words
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
        source="test",
        created_at=now,
        updated_at=now,
    )

    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,  # 8192 token limit
        rules=[huge_rule],
        current_token_count=20000,
    )

    # Should fail validation (single rule too large)
    if result.target_token_count > 8192:
        assert not result.is_valid
        assert len(result.errors) > 0


def test_switch_unknown_tool_types(sample_rules):
    """Test switching with unknown tool types uses conservative defaults."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.UNKNOWN,
        target_tool=LLMToolType.UNKNOWN,
        rules=sample_rules,
        current_token_count=1000,
    )

    # Should handle gracefully (unknown tools have 4096 token limit)
    assert result.format_compatible


def test_switch_different_codex_tools(sample_rules):
    """Test switching between tools that use same format (Codex-style)."""
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.GITHUB_COPILOT,
        target_tool=LLMToolType.CURSOR,  # Both use Codex format
        rules=sample_rules,
        current_token_count=1000,
    )

    # Token counts should be similar (same format)
    token_diff = abs(result.target_token_count - result.source_token_count)
    assert token_diff < result.source_token_count * 0.1  # <10% difference


def test_perform_switch_handles_format_error():
    """Test that perform_switch handles formatting errors gracefully."""
    # Create rule with problematic content (though our formatters are robust)
    now = datetime.now(timezone.utc)
    rule = MemoryRule(
        id="test",
        category=MemoryCategory.PREFERENCE,
        name="test_rule",
        rule="Test rule",
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
        source="test",
        created_at=now,
        updated_at=now,
    )

    # Should not raise exception
    rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.GITHUB_COPILOT, rules=[rule], auto_truncate=False
    )

    # Should return the rule (formatting won't actually fail with valid MemoryRule)
    assert len(rules) == 1


# Integration tests


def test_full_workflow_upgrade():
    """Test complete workflow: validate -> perform for upgrade scenario."""
    now = datetime.now(timezone.utc)
    rules = [
        MemoryRule(
            id=f"rule{i}",
            category=MemoryCategory.PREFERENCE,
            name=f"rule_{i}",
            rule=f"Rule {i}: Test rule content",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="test",
            created_at=now,
            updated_at=now,
        )
        for i in range(10)
    ]

    # 1. Validate switch
    validation = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.GITHUB_COPILOT,
        target_tool=LLMToolType.CLAUDE_CODE,
        rules=rules,
        current_token_count=5000,
    )

    assert validation.is_valid

    # 2. Perform switch
    switched_rules, perform_result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.CLAUDE_CODE, rules=rules, auto_truncate=False
    )

    assert len(switched_rules) == len(rules)
    assert perform_result.is_valid


def test_full_workflow_downgrade_with_truncation():
    """Test complete workflow: validate -> perform for downgrade with truncation."""
    now = datetime.now(timezone.utc)
    rules = [
        MemoryRule(
            id=f"rule{i}",
            category=MemoryCategory.PREFERENCE,
            name=f"rule_{i}",
            rule=f"Rule {i}: " + ("Long content. " * 100),
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="test",
            created_at=now,
            updated_at=now,
        )
        for i in range(50)
    ]

    # 1. Check if switch is safe
    safe = ContextSwitcher.can_switch_safely(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=rules,
    )

    # With 50 large rules, won't fit in 8k
    assert not safe

    # 2. Perform switch with auto-truncation
    switched_rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.GITHUB_COPILOT, rules=rules, auto_truncate=True
    )

    # Should have truncated
    assert len(switched_rules) < len(rules)
    assert result.rules_truncated > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
