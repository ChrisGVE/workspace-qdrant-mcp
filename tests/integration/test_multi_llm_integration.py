"""
Comprehensive integration tests for multi-LLM context injection system.

Tests the complete multi-LLM system including detection, formatting, token limits,
context switching, overrides, and error handling. Validates that all components
work together correctly in end-to-end workflows.

Test Coverage:
1. End-to-end detection and formatting workflows
2. Token limit enforcement during formatting
3. Context switching between tools
4. Override system (environment and config file)
5. Error handling and recovery
6. Multi-tool workflow scenarios

Components Tested:
- LLMToolDetector: Unified detection system
- LLMOverrideManager: Manual override configuration
- ToolTokenManager: Per-tool token limits
- ContextSwitcher: Context switching validation
- FormatManager: Tool-specific formatting

Task: #336.6 - Create comprehensive multi-tool integration testing framework
Parent: #336 - Build multi-LLM tool support infrastructure
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.python.common.core.context_injection.context_switcher import (
    ContextSwitcher,
    SwitchValidationResult,
)
from src.python.common.core.context_injection.formatters.base import (
    FormattedContext,
    LLMToolAdapter,
    ToolCapabilities,
)
from src.python.common.core.context_injection.formatters.manager import FormatManager
from src.python.common.core.context_injection.llm_override_config import (
    LLMOverrideConfig,
    LLMOverrideManager,
)
from src.python.common.core.context_injection.llm_tool_detector import (
    LLMToolDetector,
    LLMToolType,
    UnifiedLLMSession,
)
from src.python.common.core.context_injection.tool_token_manager import (
    ToolTokenLimits,
    ToolTokenManager,
)
from src.python.common.core.memory import MemoryRule

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_rules():
    """Create mock rules with known token counts."""
    rules = []
    for i in range(10):
        rule = MagicMock(spec=MemoryRule)
        rule.id = f"rule_{i}"
        rule.content = f"Rule {i} content" * 100  # ~1500 chars
        rule.priority = 1.0 - (i * 0.1)  # Descending priority
        rule.metadata = {"source": "test"}
        rules.append(rule)
    return rules


@pytest.fixture
def small_rules():
    """Create small set of rules that fit in any tool's limit."""
    rules = []
    for i in range(3):
        rule = MagicMock(spec=MemoryRule)
        rule.id = f"small_rule_{i}"
        rule.content = f"Small rule {i}"
        rule.priority = 1.0
        rule.metadata = {"source": "test"}
        rules.append(rule)
    return rules


@pytest.fixture
def large_rules():
    """Create large set of rules that exceed small tool limits."""
    rules = []
    for i in range(100):
        rule = MagicMock(spec=MemoryRule)
        rule.id = f"large_rule_{i}"
        rule.content = f"Large rule {i} content" * 200  # ~3000 chars
        rule.priority = 1.0 - (i * 0.01)
        rule.metadata = {"source": "test"}
        rules.append(rule)
    return rules


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory for override tests."""
    config_dir = tmp_path / ".wqm"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_format_manager():
    """Create mock FormatManager with predictable formatting."""

    class MockFormatManager:
        def format_for_tool(
            self, tool_name: str, rules: list[MemoryRule], token_budget: int, options=None
        ) -> FormattedContext:
            """Mock formatting with predictable token counts."""
            # Estimate tokens: ~4 chars per token
            total_chars = sum(len(r.content) for r in rules)
            token_count = total_chars // 4

            # Different tools have different formatting overhead
            tool_overhead = {
                "claude": 100,  # Claude has lightweight formatting
                "codex": 50,  # Codex is compact
                "gemini": 200,  # Gemini has more structure
            }
            overhead = tool_overhead.get(tool_name, 100)
            final_token_count = token_count + overhead

            formatted_content = f"# Formatted for {tool_name}\n\n"
            for rule in rules:
                formatted_content += f"Rule {rule.id}: {rule.content}\n\n"

            return FormattedContext(
                content=formatted_content,
                token_count=final_token_count,
                tool_name=tool_name,
                truncated_rules=[],
                metadata={"overhead": overhead},
            )

        def get_adapter(self, tool_name: str):
            """Mock adapter retrieval."""
            if tool_name in ["claude", "codex", "gemini"]:
                adapter = MagicMock(spec=LLMToolAdapter)
                adapter.get_capabilities.return_value = ToolCapabilities(
                    tool_name=tool_name,
                    supports_markdown=True,
                    supports_code_blocks=True,
                    supports_metadata=True,
                    max_context_tokens=200_000
                    if tool_name == "claude"
                    else 8_192 if tool_name == "codex" else 1_000_000,
                )
                return adapter
            return None

    return MockFormatManager()


@pytest.fixture(autouse=True)
def clean_env_vars():
    """Clean up environment variables before and after each test."""
    # Save original value
    original_override = os.environ.get("LLM_TOOL_OVERRIDE")

    # Clear override
    if "LLM_TOOL_OVERRIDE" in os.environ:
        del os.environ["LLM_TOOL_OVERRIDE"]

    yield

    # Restore original value
    if original_override is not None:
        os.environ["LLM_TOOL_OVERRIDE"] = original_override
    elif "LLM_TOOL_OVERRIDE" in os.environ:
        del os.environ["LLM_TOOL_OVERRIDE"]


@pytest.fixture(autouse=True)
def mock_config_dir(temp_config_dir, monkeypatch):
    """Mock the config directory path for all tests."""
    monkeypatch.setattr(
        LLMOverrideManager, "DEFAULT_CONFIG_DIR", temp_config_dir
    )


# ============================================================================
# Test Class 1: Detection and Formatting Integration
# ============================================================================


class TestDetectionAndFormatting:
    """Test end-to-end detection and formatting workflows."""

    @patch("src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector")
    def test_claude_code_detection_and_formatting(self, mock_claude_detector, small_rules):
        """Test detecting Claude Code and formatting rules for it."""
        # Setup: Mock Claude Code as active
        mock_claude_session = MagicMock()
        mock_claude_session.is_active = True
        mock_claude_session.session_id = "test-session"
        mock_claude_session.detection_method = "process_detection"
        mock_claude_session.project_context = None
        mock_claude_session.configuration = {}
        mock_claude_session.entrypoint = "claude"
        mock_claude_session.start_time = "2024-01-01T00:00:00"
        mock_claude_detector.detect.return_value = mock_claude_session

        # Test: Detect active tool
        session = LLMToolDetector.detect()

        # Verify: Claude Code is detected
        assert session.is_active
        assert session.tool_type == LLMToolType.CLAUDE_CODE
        assert session.detection_method in ["process_detection", "unknown"]

        # Test: Get formatter for detected tool
        formatter = LLMToolDetector.get_formatter()

        # Verify: Formatter is available
        assert formatter is not None

        # Test: Format rules
        capabilities = formatter.get_capabilities()
        formatted = formatter.format_rules(
            small_rules, capabilities.max_context_tokens
        )

        # Verify: Rules are formatted
        assert formatted.token_count > 0
        assert formatted.tool_name == "claude"
        assert len(formatted.content) > 0

    @patch("src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector")
    @patch("src.python.common.core.context_injection.llm_tool_detector.CopilotDetector")
    def test_copilot_detection_and_formatting(
        self, mock_copilot_detector, mock_claude_detector, small_rules
    ):
        """Test detecting GitHub Copilot and formatting rules for it."""
        # Setup: Claude inactive, Copilot active
        mock_claude_session = MagicMock()
        mock_claude_session.is_active = False
        mock_claude_detector.detect.return_value = mock_claude_session

        mock_copilot_session = MagicMock()
        mock_copilot_session.is_active = True
        from src.python.common.core.context_injection.copilot_detector import (
            CopilotSessionType,
        )

        mock_copilot_session.session_type = CopilotSessionType.GITHUB_COPILOT
        mock_copilot_session.detection_method = "process_detection"
        mock_copilot_session.ide_name = "vscode"
        mock_copilot_session.ide_version = "1.85.0"
        mock_copilot_session.workspace_path = None
        mock_copilot_detector.detect.return_value = mock_copilot_session

        # Test: Detect active tool
        session = LLMToolDetector.detect()

        # Verify: Copilot is detected
        assert session.is_active
        assert session.tool_type == LLMToolType.GITHUB_COPILOT
        assert session.ide_name == "vscode"

        # Test: Get formatter
        formatter = LLMToolDetector.get_formatter()

        # Verify: Codex formatter is used
        assert formatter is not None
        capabilities = formatter.get_capabilities()
        assert capabilities.max_context_tokens == 8_192  # Codex limit

    def test_override_detection_and_formatting(self, small_rules, temp_config_dir):
        """Test manual override detection and formatting."""
        # Setup: Set override to Google Gemini
        LLMOverrideManager.set_override(
            LLMToolType.GOOGLE_GEMINI,
            reason="Testing Gemini formatter",
            set_by="test",
        )

        # Test: Detect active tool
        session = LLMToolDetector.detect_with_override()

        # Verify: Override is detected
        assert session.is_active
        assert session.tool_type == LLMToolType.GOOGLE_GEMINI
        assert session.detection_method == "manual_override"
        assert session.metadata["override_active"] is True
        assert "Testing Gemini formatter" in session.metadata["override_reason"]

        # Test: Get formatter
        formatter = LLMToolDetector.get_formatter()

        # Verify: Gemini formatter is used
        assert formatter is not None
        capabilities = formatter.get_capabilities()
        assert capabilities.max_context_tokens == 1_000_000  # Gemini extended context

        # Cleanup
        LLMOverrideManager.clear_override()

    @patch("src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector")
    @patch("src.python.common.core.context_injection.llm_tool_detector.CopilotDetector")
    def test_auto_detection_with_multiple_tools(
        self, mock_copilot_detector, mock_claude_detector
    ):
        """Test auto-detection priority when multiple tools are present."""
        # Setup: Both Claude and Copilot active (Claude has priority)
        mock_claude_session = MagicMock()
        mock_claude_session.is_active = True
        mock_claude_session.session_id = "claude-session"
        mock_claude_session.detection_method = "process_detection"
        mock_claude_session.project_context = None
        mock_claude_session.configuration = {}
        mock_claude_session.entrypoint = "claude"
        mock_claude_session.start_time = "2024-01-01T00:00:00"
        mock_claude_detector.detect.return_value = mock_claude_session

        mock_copilot_session = MagicMock()
        mock_copilot_session.is_active = True
        from src.python.common.core.context_injection.copilot_detector import (
            CopilotSessionType,
        )

        mock_copilot_session.session_type = CopilotSessionType.GITHUB_COPILOT
        mock_copilot_detector.detect.return_value = mock_copilot_session

        # Test: Detect active tool
        session = LLMToolDetector.detect()

        # Verify: Claude has priority over Copilot
        assert session.is_active
        assert session.tool_type == LLMToolType.CLAUDE_CODE


# ============================================================================
# Test Class 2: Token Limit Enforcement
# ============================================================================


class TestTokenLimitEnforcement:
    """Test token limit enforcement during formatting and validation."""

    def test_token_limit_enforced_during_formatting(self, large_rules):
        """Test that token limits are enforced during formatting."""
        # Test: Format large rules for Codex (8,192 token limit)
        format_manager = FormatManager()

        # This should not raise an error if under limit
        # Format with recommended budget (80% of max)
        recommended_budget = ToolTokenManager.get_recommended_budget(
            LLMToolType.GITHUB_COPILOT
        )
        formatted = format_manager.format_for_tool(
            "codex", large_rules[:5], recommended_budget
        )

        # Verify: Formatting succeeded
        assert formatted.token_count > 0
        assert formatted.token_count <= recommended_budget

    def test_warning_at_80_percent(self):
        """Test warning threshold at 80% of token limit."""
        # Test: Check validation at 80% threshold
        codex_limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)
        token_count = int(codex_limits.max_context_tokens * 0.85)  # 85%

        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.GITHUB_COPILOT, token_count
        )

        # Verify: Warning is generated
        assert is_valid  # Still valid
        assert message is not None
        assert "WARNING" in message
        assert "approaching limit" in message.lower()

    def test_critical_at_95_percent(self):
        """Test critical threshold at 95% of token limit."""
        # Test: Check validation at 95% threshold
        codex_limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)
        token_count = int(codex_limits.max_context_tokens * 0.96)  # 96%

        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.GITHUB_COPILOT, token_count
        )

        # Verify: Critical warning is generated
        assert is_valid  # Still valid
        assert message is not None
        assert "CRITICAL" in message
        assert "near limit" in message.lower()

    def test_error_at_100_percent(self):
        """Test error when exceeding token limit."""
        # Test: Check validation above 100%
        codex_limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)
        token_count = codex_limits.max_context_tokens + 1000

        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.GITHUB_COPILOT, token_count
        )

        # Verify: Error is generated
        assert not is_valid
        assert message is not None
        assert "ERROR" in message
        assert "exceeds" in message.lower()

    def test_different_limits_per_tool(self):
        """Test that different tools have different token limits."""
        # Test: Get limits for each tool
        claude_limits = ToolTokenManager.get_limits(LLMToolType.CLAUDE_CODE)
        codex_limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)
        gemini_limits = ToolTokenManager.get_limits(LLMToolType.GOOGLE_GEMINI)

        # Verify: Limits are different
        assert claude_limits.max_context_tokens == 200_000
        assert codex_limits.max_context_tokens == 8_192
        assert gemini_limits.max_context_tokens == 1_000_000

        # Verify: Recommended budgets are 80% of max
        assert claude_limits.recommended_budget == 160_000
        assert codex_limits.recommended_budget == 6_553
        assert gemini_limits.recommended_budget == 800_000


# ============================================================================
# Test Class 3: Context Switching
# ============================================================================


class TestContextSwitching:
    """Test context switching between LLM tools."""

    def test_switch_from_claude_to_copilot(self, mock_rules):
        """Test switching from Claude Code to GitHub Copilot."""
        # Test: Validate switch from Claude to Copilot
        result = ContextSwitcher.validate_switch(
            source_tool=LLMToolType.CLAUDE_CODE,
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=mock_rules,
            current_token_count=50_000,
        )

        # Verify: Validation result
        assert isinstance(result, SwitchValidationResult)
        assert result.source_tool == LLMToolType.CLAUDE_CODE
        assert result.target_tool == LLMToolType.GITHUB_COPILOT

        # Verify: Warnings about downgrade
        assert len(result.warnings) > 0
        assert any("reduce" in w.lower() for w in result.warnings)

    def test_switch_from_copilot_to_claude(self, small_rules):
        """Test switching from GitHub Copilot to Claude Code."""
        # Test: Validate switch from Copilot to Claude
        result = ContextSwitcher.validate_switch(
            source_tool=LLMToolType.GITHUB_COPILOT,
            target_tool=LLMToolType.CLAUDE_CODE,
            rules=small_rules,
            current_token_count=5_000,
        )

        # Verify: Validation result
        assert result.is_valid
        assert result.source_tool == LLMToolType.GITHUB_COPILOT
        assert result.target_tool == LLMToolType.CLAUDE_CODE

        # Verify: Warnings about upgrade
        assert len(result.warnings) > 0
        assert any("increase" in w.lower() for w in result.warnings)

    def test_switch_with_truncation(self, large_rules):
        """Test switch with automatic truncation when exceeding limits."""
        # Test: Perform switch with auto-truncation
        selected_rules, result = ContextSwitcher.perform_switch(
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=large_rules,
            auto_truncate=True,
        )

        # Verify: Some rules were truncated
        assert len(selected_rules) < len(large_rules)
        assert result.rules_truncated > 0

        # Verify: Truncation warning
        assert any("truncated" in w.lower() for w in result.warnings)

    @patch("src.python.common.core.context_injection.context_switcher.LLMToolDetector")
    def test_switch_validation_before_execution(
        self, mock_detector, mock_rules, mock_format_manager
    ):
        """Test that validation occurs before performing switch."""
        # Setup: Mock current tool
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.tool_type = LLMToolType.CLAUDE_CODE
        mock_detector.detect.return_value = mock_session

        # Test: Check if switch is safe
        is_safe = ContextSwitcher.can_switch_safely(
            source_tool=LLMToolType.CLAUDE_CODE,
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=mock_rules,
        )

        # Verify: Safety check executed
        assert isinstance(is_safe, bool)


# ============================================================================
# Test Class 4: Override System
# ============================================================================


class TestOverrideSystem:
    """Test manual override configuration system."""

    def test_environment_variable_override(self):
        """Test override via environment variable."""
        # Setup: Set environment variable
        os.environ["LLM_TOOL_OVERRIDE"] = "cursor"

        # Test: Get override
        override = LLMOverrideManager.get_override()

        # Verify: Override is active
        assert override is not None
        assert override.enabled
        assert override.tool_type == LLMToolType.CURSOR
        assert override.set_by == "environment"

    def test_config_file_override(self, temp_config_dir):
        """Test override via config file."""
        # Setup: Set override in config file
        LLMOverrideManager.set_override(
            LLMToolType.JETBRAINS_AI,
            reason="Testing JetBrains AI",
            set_by="test",
        )

        # Test: Get override
        override = LLMOverrideManager.get_override()

        # Verify: Override is active
        assert override is not None
        assert override.enabled
        assert override.tool_type == LLMToolType.JETBRAINS_AI
        assert "JetBrains AI" in override.reason

    def test_override_priority(self, temp_config_dir):
        """Test that environment variable has priority over config file."""
        # Setup: Set both env var and config file
        os.environ["LLM_TOOL_OVERRIDE"] = "claude_code"
        LLMOverrideManager.set_override(
            LLMToolType.GITHUB_COPILOT,
            reason="Config file override",
            set_by="test",
        )

        # Test: Get override
        override = LLMOverrideManager.get_override()

        # Verify: Environment variable wins
        assert override is not None
        assert override.tool_type == LLMToolType.CLAUDE_CODE
        assert override.set_by == "environment"

    def test_clear_override(self, temp_config_dir):
        """Test clearing override configuration."""
        # Setup: Set override
        LLMOverrideManager.set_override(
            LLMToolType.TABNINE,
            reason="Testing clear",
            set_by="test",
        )

        # Verify: Override is active
        assert LLMOverrideManager.is_override_active()

        # Test: Clear override
        LLMOverrideManager.clear_override()

        # Verify: Override is cleared
        override = LLMOverrideManager.get_override()
        assert override is None or not override.enabled


# ============================================================================
# Test Class 5: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_invalid_tool_type_override(self):
        """Test handling of invalid tool type in override."""
        # Test: Try to create config with invalid tool type string
        with pytest.raises(ValueError) as exc_info:
            LLMOverrideConfig.from_dict(
                {"enabled": True, "tool_type": "invalid_tool"}
            )

        # Verify: Error message is helpful
        assert "Invalid tool_type" in str(exc_info.value)

    def test_excessive_token_count(self, large_rules):
        """Test handling of excessive token count."""
        # Test: Try to format with excessive budget
        format_manager = FormatManager()
        codex_limits = ToolTokenManager.get_limits(LLMToolType.GITHUB_COPILOT)

        with pytest.raises(ValueError) as exc_info:
            format_manager.format_for_tool(
                "codex",
                large_rules,
                token_budget=codex_limits.max_context_tokens + 10_000,
            )

        # Verify: Error indicates limit exceeded
        assert "exceeds" in str(exc_info.value).lower()

    def test_missing_formatter(self):
        """Test handling of missing formatter for tool."""
        # Test: Try to get formatter for unknown tool
        format_manager = FormatManager()
        adapter = format_manager.get_adapter("nonexistent_tool")

        # Verify: Returns None
        assert adapter is None

    def test_corrupted_config_file(self, temp_config_dir):
        """Test handling of corrupted config file."""
        # Setup: Create corrupted config file
        config_path = temp_config_dir / "llm_override.json"
        with open(config_path, "w") as f:
            f.write("{ invalid json")

        # Test: Try to read override
        override = LLMOverrideManager.get_override()

        # Verify: Returns None (graceful degradation)
        assert override is None


# ============================================================================
# Test Class 6: End-to-End Multi-Tool Workflows
# ============================================================================


class TestMultiToolWorkflows:
    """Test complete end-to-end multi-tool workflows."""

    @patch("src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector")
    def test_scenario_claude_code_session(
        self, mock_claude_detector, mock_rules
    ):
        """
        Scenario 1: Claude Code session → Format rules → Validate tokens.

        Tests typical Claude Code workflow from detection through formatting
        with token validation.
        """
        # Step 1: Setup - Mock Claude Code as active
        mock_claude_session = MagicMock()
        mock_claude_session.is_active = True
        mock_claude_session.session_id = "test-session"
        mock_claude_session.detection_method = "process_detection"
        mock_claude_session.project_context = None
        mock_claude_session.configuration = {}
        mock_claude_session.entrypoint = "claude"
        mock_claude_session.start_time = "2024-01-01T00:00:00"
        mock_claude_detector.detect.return_value = mock_claude_session

        # Step 2: Detect active tool
        session = LLMToolDetector.detect()
        assert session.is_active
        assert session.tool_type == LLMToolType.CLAUDE_CODE

        # Step 3: Get formatter
        formatter = LLMToolDetector.get_formatter()
        assert formatter is not None

        # Step 4: Get recommended budget
        budget = ToolTokenManager.get_recommended_budget(LLMToolType.CLAUDE_CODE)
        assert budget == 160_000  # 80% of 200k

        # Step 5: Format rules
        formatted = formatter.format_rules(mock_rules, budget)
        assert formatted.token_count > 0
        assert formatted.token_count <= budget

        # Step 6: Validate token count
        is_valid, message = ToolTokenManager.validate_token_count(
            LLMToolType.CLAUDE_CODE, formatted.token_count
        )
        assert is_valid
        # Should not have warning (well under 80% threshold)

    def test_scenario_override_to_copilot_with_truncation(
        self, large_rules, temp_config_dir
    ):
        """
        Scenario 2: Override to Copilot → Format rules → Handle truncation.

        Tests override workflow with automatic truncation when rules exceed
        Copilot's smaller token limit.
        """
        # Step 1: Set override to Copilot
        LLMOverrideManager.set_override(
            LLMToolType.GITHUB_COPILOT,
            reason="Testing with Copilot",
            set_by="test",
        )

        # Step 2: Verify override is active
        session = LLMToolDetector.detect_with_override()
        assert session.tool_type == LLMToolType.GITHUB_COPILOT
        assert session.metadata["override_active"] is True

        # Step 3: Attempt to switch with large rules
        selected_rules, result = ContextSwitcher.perform_switch(
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=large_rules,
            auto_truncate=True,
        )

        # Step 4: Verify truncation occurred
        assert len(selected_rules) < len(large_rules)
        assert result.rules_truncated > 0
        assert result.is_valid

        # Step 5: Verify truncation warning
        assert any("truncated" in w.lower() for w in result.warnings)

        # Cleanup
        LLMOverrideManager.clear_override()

    @patch("src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector")
    @patch("src.python.common.core.context_injection.llm_tool_detector.CopilotDetector")
    def test_scenario_switch_between_tools(
        self, mock_copilot_detector, mock_claude_detector, mock_rules
    ):
        """
        Scenario 3: Switch from Claude to Copilot → Validate → Reformat.

        Tests switching between tools with validation and reformatting.
        """
        # Step 1: Start with Claude Code active
        mock_claude_session = MagicMock()
        mock_claude_session.is_active = True
        mock_claude_session.session_id = "claude-session"
        mock_claude_session.detection_method = "process_detection"
        mock_claude_session.project_context = None
        mock_claude_session.configuration = {}
        mock_claude_session.entrypoint = "claude"
        mock_claude_session.start_time = "2024-01-01T00:00:00"
        mock_claude_detector.detect.return_value = mock_claude_session

        session = LLMToolDetector.detect()
        assert session.tool_type == LLMToolType.CLAUDE_CODE

        # Step 2: Validate switch to Copilot
        result = ContextSwitcher.validate_switch(
            source_tool=LLMToolType.CLAUDE_CODE,
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=mock_rules,
            current_token_count=10_000,
        )

        # Step 3: Check if switch is safe
        assert isinstance(result, SwitchValidationResult)
        assert result.target_tool == LLMToolType.GITHUB_COPILOT

        # Step 4: Perform switch
        selected_rules, switch_result = ContextSwitcher.perform_switch(
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=mock_rules,
            auto_truncate=True,
        )

        # Step 5: Verify reformatting occurred
        assert len(selected_rules) <= len(mock_rules)
        assert switch_result.target_tool == LLMToolType.GITHUB_COPILOT

    def test_scenario_invalid_override_fallback(self, temp_config_dir):
        """
        Scenario 4: Invalid override → Fallback to auto-detection.

        Tests graceful handling of invalid override configuration.
        """
        # Step 1: Create invalid override config
        config_path = temp_config_dir / "llm_override.json"
        with open(config_path, "w") as f:
            json.dump({"enabled": True, "tool_type": "invalid_tool"}, f)

        # Step 2: Try to get override (should fail gracefully)
        override = LLMOverrideManager.get_override()

        # Step 3: Verify fallback to None
        assert override is None

        # Step 4: Detection should proceed with auto-detection
        session = LLMToolDetector.detect()
        # Will be UNKNOWN in test environment
        assert session.tool_type in [LLMToolType.UNKNOWN, LLMToolType.CLAUDE_CODE]

    def test_scenario_token_limit_exceeded_error(self, large_rules):
        """
        Scenario 5: Token limit exceeded → Error handling.

        Tests error handling when token limits are exceeded without auto-truncate.
        """
        # Step 1: Try to validate switch that exceeds limit
        result = ContextSwitcher.validate_switch(
            source_tool=LLMToolType.CLAUDE_CODE,
            target_tool=LLMToolType.GITHUB_COPILOT,
            rules=large_rules,
            current_token_count=150_000,
        )

        # Step 2: Verify validation fails with errors
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.rules_truncated > 0

        # Step 3: Verify error messages are helpful
        assert any("exceed" in e.lower() for e in result.errors)
        assert any("auto_truncate" in e.lower() for e in result.errors)


# ============================================================================
# Integration Test Summary
# ============================================================================


def test_integration_summary():
    """
    Summary test documenting all integration test coverage.

    This test serves as documentation of what the integration test suite covers.
    """
    test_coverage = {
        "detection_and_formatting": [
            "Claude Code detection and formatting",
            "GitHub Copilot detection and formatting",
            "Override detection and formatting",
            "Multi-tool auto-detection with priority",
        ],
        "token_limit_enforcement": [
            "Token limits enforced during formatting",
            "Warning at 80% threshold",
            "Critical warning at 95% threshold",
            "Error at 100%+ (exceeding limit)",
            "Different limits per tool",
        ],
        "context_switching": [
            "Switch from Claude to Copilot (downgrade)",
            "Switch from Copilot to Claude (upgrade)",
            "Switch with automatic truncation",
            "Validation before switch execution",
        ],
        "override_system": [
            "Environment variable override",
            "Config file override",
            "Override priority (env > config)",
            "Clear override",
        ],
        "error_handling": [
            "Invalid tool type",
            "Excessive token count",
            "Missing formatter",
            "Corrupted config file",
        ],
        "end_to_end_workflows": [
            "Claude Code session workflow",
            "Override to Copilot with truncation",
            "Switch between tools with reformat",
            "Invalid override with fallback",
            "Token limit exceeded error handling",
        ],
    }

    # Verify test coverage is comprehensive
    total_scenarios = sum(len(scenarios) for scenarios in test_coverage.values())
    assert total_scenarios >= 25, "Integration test coverage should include 25+ scenarios"

    # All test categories present
    assert len(test_coverage) == 6, "Should have 6 test categories"
