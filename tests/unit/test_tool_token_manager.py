"""
Unit tests for tool-specific token limit management.

Tests the ToolTokenManager class and ToolTokenLimits dataclass for per-tool
token limit validation and enforcement.
"""

import pytest

from src.python.common.core.context_injection.llm_tool_detector import LLMToolType
from src.python.common.core.context_injection.tool_token_manager import (
    ToolTokenLimits,
    ToolTokenManager,
)


class TestToolTokenLimits:
    """Tests for ToolTokenLimits dataclass."""

    def test_create_valid_limits(self):
        """Test creating valid ToolTokenLimits."""
        limits = ToolTokenLimits(
            tool_type=LLMToolType.CLAUDE_CODE,
            max_context_tokens=200_000,
            recommended_budget=160_000,
            warning_threshold=0.8,
            critical_threshold=0.95,
        )

        assert limits.tool_type == LLMToolType.CLAUDE_CODE
        assert limits.max_context_tokens == 200_000
        assert limits.recommended_budget == 160_000
        assert limits.warning_threshold == 0.8
        assert limits.critical_threshold == 0.95

    def test_default_thresholds(self):
        """Test default threshold values."""
        limits = ToolTokenLimits(
            tool_type=LLMToolType.CODEX_API,
            max_context_tokens=8_192,
            recommended_budget=6_553,
        )

        assert limits.warning_threshold == 0.8  # 80%
        assert limits.critical_threshold == 0.95  # 95%

    def test_invalid_warning_threshold(self):
        """Test validation rejects invalid warning threshold."""
        with pytest.raises(ValueError, match="warning_threshold must be 0-1"):
            ToolTokenLimits(
                tool_type=LLMToolType.CLAUDE_CODE,
                max_context_tokens=200_000,
                recommended_budget=160_000,
                warning_threshold=1.5,  # Invalid: > 1
            )

    def test_invalid_critical_threshold(self):
        """Test validation rejects invalid critical threshold."""
        with pytest.raises(ValueError, match="critical_threshold must be 0-1"):
            ToolTokenLimits(
                tool_type=LLMToolType.CLAUDE_CODE,
                max_context_tokens=200_000,
                recommended_budget=160_000,
                critical_threshold=-0.1,  # Invalid: < 0
            )

    def test_warning_exceeds_critical(self):
        """Test validation rejects warning > critical."""
        with pytest.raises(ValueError, match="warning_threshold cannot exceed"):
            ToolTokenLimits(
                tool_type=LLMToolType.CLAUDE_CODE,
                max_context_tokens=200_000,
                recommended_budget=160_000,
                warning_threshold=0.95,
                critical_threshold=0.8,  # Lower than warning
            )


class TestToolTokenManager:
    """Tests for ToolTokenManager class."""

    def test_get_limits_claude(self):
        """Test getting limits for Claude Code."""
        limits = ToolTokenManager.get_limits(LLMToolType.CLAUDE_CODE)

        assert limits.tool_type == LLMToolType.CLAUDE_CODE
        assert limits.max_context_tokens == 200_000
        assert limits.recommended_budget == 160_000  # 80% of max
        assert limits.warning_threshold == 0.8
        assert limits.critical_threshold == 0.95

    def test_get_limits_copilot(self):
        """Test getting limits for GitHub Copilot."""
        limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)

        assert limits.tool_type == LLMToolType.GITHUB_COPILOT
        assert limits.max_context_tokens == 8_192
        assert limits.recommended_budget == 6_553  # 80% of max

    def test_get_limits_gemini(self):
        """Test getting limits for Google Gemini."""
        limits = ToolTokenManager.get_limits(LLMToolType.GOOGLE_GEMINI)

        assert limits.tool_type == LLMToolType.GOOGLE_GEMINI
        assert limits.max_context_tokens == 1_000_000
        assert limits.recommended_budget == 800_000  # 80% of max

    def test_get_limits_cursor(self):
        """Test getting limits for Cursor."""
        limits = ToolTokenManager.get_limits(LLMToolType.CURSOR)

        assert limits.tool_type == LLMToolType.CURSOR
        assert limits.max_context_tokens == 8_192  # Codex-style limit

    def test_get_limits_unknown_tool(self):
        """Test getting limits for unknown tool returns conservative default."""
        limits = ToolTokenManager.get_limits(LLMToolType.UNKNOWN)

        assert limits.tool_type == LLMToolType.UNKNOWN
        assert limits.max_context_tokens == 4_096  # Conservative default
        assert limits.recommended_budget == 3_276  # 80% of 4096

    def test_validate_token_count_valid(self):
        """Test validation passes for valid token count."""
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 100_000
        )

        assert is_valid is True
        assert message is None  # No warnings under 80%

    def test_validate_token_count_zero(self):
        """Test validation handles zero tokens."""
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 0
        )

        assert is_valid is True
        assert message is None

    def test_validate_token_count_negative(self):
        """Test validation rejects negative tokens."""
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, -1000
        )

        assert is_valid is False
        assert "ERROR" in message
        assert "negative" in message.lower()

    def test_validate_token_count_warning_threshold(self):
        """Test validation warns at 80% threshold."""
        # 80% of 200,000 = 160,000
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 165_000
        )

        assert is_valid is True
        assert message is not None
        assert "WARNING" in message
        assert "approaching limit" in message.lower()
        assert "165,000 / 200,000" in message

    def test_validate_token_count_critical_threshold(self):
        """Test validation warns critically at 95% threshold."""
        # 95% of 200,000 = 190,000
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 195_000
        )

        assert is_valid is True
        assert message is not None
        assert "CRITICAL" in message
        assert "near limit" in message.lower()
        assert "195,000 / 200,000" in message

    def test_validate_token_count_exceeds_limit(self):
        """Test validation fails when exceeding limit."""
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 250_000
        )

        assert is_valid is False
        assert message is not None
        assert "ERROR" in message
        assert "exceeds" in message.lower()
        assert "250,000 / 200,000" in message

    def test_validate_token_count_codex_limit(self):
        """Test validation against Codex's smaller limit."""
        # Codex has 8,192 token limit
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.GITHUB_COPILOT, 7_000
        )

        # 7000 / 8192 = 85.4% (above 80% warning threshold)
        assert is_valid is True
        assert message is not None
        assert "WARNING" in message

        # Test exceeding limit
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.GITHUB_COPILOT, 10_000
        )

        assert is_valid is False
        assert "ERROR" in message

    def test_get_recommended_budget_claude(self):
        """Test getting recommended budget for Claude."""
        budget = ToolTokenManager.get_recommended_budget(LLMToolType.CLAUDE_CODE)

        assert budget == 160_000  # 80% of 200,000

    def test_get_recommended_budget_copilot(self):
        """Test getting recommended budget for Copilot."""
        budget = ToolTokenManager.get_recommended_budget(LLMToolType.GITHUB_COPILOT)

        assert budget == 6_553  # 80% of 8,192

    def test_get_recommended_budget_gemini(self):
        """Test getting recommended budget for Gemini."""
        budget = ToolTokenManager.get_recommended_budget(LLMToolType.GOOGLE_GEMINI)

        assert budget == 800_000  # 80% of 1,000,000

    def test_get_recommended_budget_auto_detect_no_session(self):
        """Test auto-detect with or without active session."""
        # When tool_type=None, auto-detects current tool
        budget = ToolTokenManager.get_recommended_budget(None)

        # Budget should be valid for some tool (either detected or default)
        # Claude Code: 160,000
        # Copilot: 6,553
        # Gemini: 800,000
        # Default: 3,276
        assert budget in [3_276, 6_553, 160_000, 800_000], (
            f"Budget {budget:,} not in expected set of tool budgets"
        )

        # Budget should be positive
        assert budget > 0

    def test_calculate_usage_percentage(self):
        """Test calculating usage percentage."""
        # Claude: 100,000 / 200,000 = 50%
        pct = ToolTokenManager.calculate_usage_percentage(
            LLMToolType.CLAUDE_CODE, 100_000
        )
        assert pct == 50.0

        # Codex: 4,096 / 8,192 = 50%
        pct = ToolTokenManager.calculate_usage_percentage(
            LLMToolType.GITHUB_COPILOT, 4_096
        )
        assert pct == 50.0

        # Over 100%
        pct = ToolTokenManager.calculate_usage_percentage(
            LLMToolType.GITHUB_COPILOT, 10_000
        )
        assert pct > 100.0

    def test_calculate_usage_percentage_zero_tokens(self):
        """Test usage percentage with zero tokens."""
        pct = ToolTokenManager.calculate_usage_percentage(LLMToolType.CLAUDE_CODE, 0)

        assert pct == 0.0

    def test_calculate_usage_percentage_exact_limit(self):
        """Test usage percentage at exact limit."""
        pct = ToolTokenManager.calculate_usage_percentage(
            LLMToolType.CLAUDE_CODE, 200_000
        )

        assert pct == 100.0

    def test_get_all_limits(self):
        """Test getting all tool limits."""
        all_limits = ToolTokenManager.get_all_limits()

        # Should have entries for all known tools
        assert LLMToolType.CLAUDE_CODE in all_limits
        assert LLMToolType.GITHUB_COPILOT in all_limits
        assert LLMToolType.GOOGLE_GEMINI in all_limits
        assert LLMToolType.CURSOR in all_limits
        assert LLMToolType.JETBRAINS_AI in all_limits
        assert LLMToolType.CODEX_API in all_limits
        assert LLMToolType.TABNINE in all_limits

        # Verify it's a copy (not reference to internal dict)
        original_size = len(all_limits)
        all_limits[LLMToolType.UNKNOWN] = ToolTokenLimits(
            tool_type=LLMToolType.UNKNOWN,
            max_context_tokens=1000,
            recommended_budget=800,
        )
        new_limits = ToolTokenManager.get_all_limits()
        assert len(new_limits) == original_size  # Should not be modified

    def test_recommended_budget_is_80_percent(self):
        """Test that all tools have recommended budget = 80% of max."""
        all_limits = ToolTokenManager.get_all_limits()

        for tool_type, limits in all_limits.items():
            expected_budget = int(limits.max_context_tokens * 0.8)
            assert limits.recommended_budget == expected_budget, (
                f"{tool_type.value} should have recommended_budget = "
                f"{expected_budget} (80% of {limits.max_context_tokens})"
            )


class TestToolTokenManagerIntegration:
    """Integration tests for ToolTokenManager with other components."""

    def test_validation_thresholds_match_budgetwarningsystem(self):
        """Test that thresholds align with BudgetWarningSystem defaults."""
        # BudgetWarningSystem uses 75%, 85%, 95%
        # We use 80% warning, 95% critical
        # Our 80% warning should be between their 75% and 85%
        # Our 95% critical should match their 95%

        limits = ToolTokenManager.get_limits(LLMToolType.CLAUDE_CODE)

        assert limits.warning_threshold == 0.8  # Between 75% and 85%
        assert limits.critical_threshold == 0.95  # Matches critical

    def test_token_limits_match_adapter_capabilities(self):
        """Test that token limits match adapter ToolCapabilities."""
        # These values should match what's in the adapter classes
        # Claude: 100,000 → updated to 200,000 (newer models)
        claude_limits = ToolTokenManager.get_limits(LLMToolType.CLAUDE_CODE)
        assert claude_limits.max_context_tokens == 200_000

        # Codex: 4,096 → updated to 8,192 (newer models)
        codex_limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)
        assert codex_limits.max_context_tokens == 8_192

        # Gemini: 32,000 → updated to 1,000,000 (Gemini 1.5 Pro)
        gemini_limits = ToolTokenManager.get_limits(LLMToolType.GOOGLE_GEMINI)
        assert gemini_limits.max_context_tokens == 1_000_000

    def test_validation_message_format(self):
        """Test that validation messages are clear and actionable."""
        # Warning message
        _, warning_msg = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 165_000
        )
        assert warning_msg is not None
        assert "WARNING" in warning_msg
        assert "165,000" in warning_msg  # Current count
        assert "200,000" in warning_msg  # Max limit
        assert "160,000" in warning_msg  # Recommended budget

        # Critical message
        _, critical_msg = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 195_000
        )
        assert critical_msg is not None
        assert "CRITICAL" in critical_msg
        assert "5,000" in critical_msg or "5000" in critical_msg  # Remaining tokens

        # Error message
        _, error_msg = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, 250_000
        )
        assert error_msg is not None
        assert "ERROR" in error_msg
        assert "exceeds" in error_msg.lower()
        assert "prioritization" in error_msg.lower()  # Suggests solution
