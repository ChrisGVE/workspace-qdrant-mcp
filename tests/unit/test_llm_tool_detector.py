"""
Unit tests for unified LLM tool detection system.

Tests cover:
- Detection of Claude Code sessions
- Detection of Copilot/IDE sessions
- Priority logic when multiple tools detected
- Mapping from CopilotSessionType to LLMToolType
- Integration with FormatManager
- Convenience functions
- Unknown sessions
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.python.common.core.context_injection import (
    ClaudeCodeSession,
    CopilotSession,
    CopilotSessionType,
    LLMToolDetector,
    LLMToolType,
    ProjectContextMetadata,
    UnifiedLLMSession,
    get_active_llm_tool,
    get_llm_formatter,
    is_llm_tool_active,
)
from src.python.common.core.context_injection.formatters import (
    FormatManager,
    FormatType,
    ToolCapabilities,
)


@pytest.fixture
def mock_claude_session():
    """Create a mock Claude Code session."""
    return ClaudeCodeSession(
        is_active=True,
        entrypoint="cli",
        detection_method="environment_variable_claudecode",
        session_id="claude-session-abc123",
        start_time=1234567890.0,
        project_context=ProjectContextMetadata(
            project_id="test-project",
            project_root=Path("/test/project"),
            current_path=Path("/test/project/src"),
            scope=["python", "testing"],
            is_submodule=False,
        ),
        configuration={"python_version": "3.11.0"},
    )


@pytest.fixture
def mock_copilot_session():
    """Create a mock Copilot session."""
    return CopilotSession(
        is_active=True,
        session_type=CopilotSessionType.GITHUB_COPILOT,
        ide_name="vscode",
        ide_version="1.80.0",
        detection_method="environment_variable_vscode",
        workspace_path=Path("/test/workspace"),
    )


@pytest.fixture
def mock_cursor_session():
    """Create a mock Cursor session."""
    return CopilotSession(
        is_active=True,
        session_type=CopilotSessionType.CURSOR,
        ide_name="cursor",
        detection_method="environment_variable_cursor",
        workspace_path=Path("/test/workspace"),
    )


@pytest.fixture
def mock_jetbrains_session():
    """Create a mock JetBrains session."""
    return CopilotSession(
        is_active=True,
        session_type=CopilotSessionType.JETBRAINS_AI,
        ide_name="pycharm",
        detection_method="environment_variable_jetbrains",
        workspace_path=Path("/test/workspace"),
    )


class TestLLMToolType:
    """Test LLMToolType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert LLMToolType.CLAUDE_CODE.value == "claude_code"
        assert LLMToolType.GITHUB_COPILOT.value == "github_copilot"
        assert LLMToolType.CODEX_API.value == "codex_api"
        assert LLMToolType.CURSOR.value == "cursor"
        assert LLMToolType.JETBRAINS_AI.value == "jetbrains_ai"
        assert LLMToolType.GOOGLE_GEMINI.value == "google_gemini"
        assert LLMToolType.TABNINE.value == "tabnine"
        assert LLMToolType.UNKNOWN.value == "unknown"


class TestUnifiedLLMSession:
    """Test UnifiedLLMSession dataclass."""

    def test_minimal_session(self):
        """Test creating a minimal session."""
        session = UnifiedLLMSession(
            tool_type=LLMToolType.UNKNOWN,
            is_active=False,
            detection_method="none",
        )

        assert session.tool_type == LLMToolType.UNKNOWN
        assert session.is_active is False
        assert session.detection_method == "none"
        assert session.session_id is None
        assert session.ide_name is None
        assert session.workspace_path is None
        assert session.capabilities is None
        assert session.metadata == {}

    def test_full_session(self):
        """Test creating a session with all fields."""
        capabilities = ToolCapabilities(
            tool_name="claude",
            format_type=FormatType.MARKDOWN,
            max_context_tokens=200000,
            supports_sections=True,
            supports_markdown=True,
            supports_priorities=True,
            injection_method="file",
        )

        session = UnifiedLLMSession(
            tool_type=LLMToolType.CLAUDE_CODE,
            is_active=True,
            detection_method="environment_variable",
            session_id="session-123",
            ide_name="claude_code_cli",
            workspace_path=Path("/workspace"),
            capabilities=capabilities,
            metadata={"key": "value"},
        )

        assert session.tool_type == LLMToolType.CLAUDE_CODE
        assert session.is_active is True
        assert session.session_id == "session-123"
        assert session.ide_name == "claude_code_cli"
        assert session.workspace_path == Path("/workspace")
        assert session.capabilities == capabilities
        assert session.metadata == {"key": "value"}


class TestLLMToolDetector:
    """Test LLMToolDetector class."""

    def test_detect_no_session(self):
        """Test detection when no LLM tool is active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                # Mock both detectors returning inactive sessions
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = CopilotSession(
                    is_active=False, session_type=None, detection_method=None
                )

                session = LLMToolDetector.detect()

                assert session.tool_type == LLMToolType.UNKNOWN
                assert session.is_active is False
                assert session.detection_method == "none"

    def test_detect_claude_code_session(self, mock_claude_session):
        """Test detection of Claude Code session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            session = LLMToolDetector.detect()

            assert session.tool_type == LLMToolType.CLAUDE_CODE
            assert session.is_active is True
            assert session.detection_method == "environment_variable_claudecode"
            assert session.session_id == "claude-session-abc123"
            assert session.ide_name == "claude_code_cli"
            assert session.workspace_path == Path("/test/project")
            assert session.metadata["entrypoint"] == "cli"
            assert session.metadata["project_id"] == "test-project"
            assert session.metadata["scope"] == ["python", "testing"]

    def test_detect_copilot_session(self, mock_copilot_session):
        """Test detection of Copilot session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                # No Claude Code session
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                # Active Copilot session
                mock_copilot.return_value = mock_copilot_session

                session = LLMToolDetector.detect()

                assert session.tool_type == LLMToolType.GITHUB_COPILOT
                assert session.is_active is True
                assert session.detection_method == "environment_variable_vscode"
                assert session.ide_name == "vscode"
                assert session.workspace_path == Path("/test/workspace")
                assert session.metadata["ide_version"] == "1.80.0"

    def test_priority_claude_over_copilot(self, mock_claude_session, mock_copilot_session):
        """Test that Claude Code takes priority over Copilot."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                # Both active
                mock_claude.return_value = mock_claude_session
                mock_copilot.return_value = mock_copilot_session

                session = LLMToolDetector.detect()

                # Claude Code should win
                assert session.tool_type == LLMToolType.CLAUDE_CODE
                assert session.ide_name == "claude_code_cli"

    def test_detect_cursor_session(self, mock_cursor_session):
        """Test detection of Cursor session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = mock_cursor_session

                session = LLMToolDetector.detect()

                assert session.tool_type == LLMToolType.CURSOR
                assert session.is_active is True
                assert session.ide_name == "cursor"

    def test_detect_jetbrains_session(self, mock_jetbrains_session):
        """Test detection of JetBrains session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = mock_jetbrains_session

                session = LLMToolDetector.detect()

                assert session.tool_type == LLMToolType.JETBRAINS_AI
                assert session.is_active is True
                assert session.ide_name == "pycharm"

    def test_copilot_session_type_mapping(self):
        """Test all CopilotSessionType mappings to LLMToolType."""
        mappings = {
            CopilotSessionType.GITHUB_COPILOT: LLMToolType.GITHUB_COPILOT,
            CopilotSessionType.CODEX_API: LLMToolType.CODEX_API,
            CopilotSessionType.CURSOR: LLMToolType.CURSOR,
            CopilotSessionType.JETBRAINS_AI: LLMToolType.JETBRAINS_AI,
            CopilotSessionType.TABNINE: LLMToolType.TABNINE,
            CopilotSessionType.UNKNOWN: LLMToolType.UNKNOWN,
        }

        for copilot_type, expected_llm_type in mappings.items():
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
            ) as mock_claude:
                with patch(
                    "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
                ) as mock_copilot:
                    mock_claude.return_value = ClaudeCodeSession(
                        is_active=False, entrypoint=None, detection_method=None
                    )
                    mock_copilot.return_value = CopilotSession(
                        is_active=True,
                        session_type=copilot_type,
                        ide_name="test_ide",
                        detection_method="test",
                    )

                    session = LLMToolDetector.detect()
                    assert session.tool_type == expected_llm_type

    def test_is_active_true(self, mock_claude_session):
        """Test is_active when session is active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            assert LLMToolDetector.is_active() is True

    def test_is_active_false(self):
        """Test is_active when no session is active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = CopilotSession(
                    is_active=False, session_type=None, detection_method=None
                )

                assert LLMToolDetector.is_active() is False

    def test_get_active_tool_type_active(self, mock_claude_session):
        """Test get_active_tool_type when session is active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            tool_type = LLMToolDetector.get_active_tool_type()
            assert tool_type == LLMToolType.CLAUDE_CODE

    def test_get_active_tool_type_inactive(self):
        """Test get_active_tool_type when no session is active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = CopilotSession(
                    is_active=False, session_type=None, detection_method=None
                )

                tool_type = LLMToolDetector.get_active_tool_type()
                assert tool_type is None

    def test_get_formatter_no_session(self):
        """Test get_formatter when no session is active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = CopilotSession(
                    is_active=False, session_type=None, detection_method=None
                )

                formatter = LLMToolDetector.get_formatter()
                assert formatter is None

    def test_get_formatter_claude_code(self, mock_claude_session):
        """Test get_formatter for Claude Code session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            formatter = LLMToolDetector.get_formatter()

            # Should return ClaudeCodeAdapter
            assert formatter is not None
            assert formatter.capabilities.tool_name == "claude"

    def test_get_formatter_copilot(self, mock_copilot_session):
        """Test get_formatter for Copilot session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = mock_copilot_session

                formatter = LLMToolDetector.get_formatter()

                # Should return GitHubCodexAdapter
                assert formatter is not None
                assert formatter.capabilities.tool_name == "codex"

    def test_get_formatter_cursor(self, mock_cursor_session):
        """Test get_formatter for Cursor session (uses Codex adapter)."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = mock_cursor_session

                formatter = LLMToolDetector.get_formatter()

                # Cursor uses Codex adapter
                assert formatter is not None
                assert formatter.capabilities.tool_name == "codex"

    def test_capabilities_included_in_session(self, mock_claude_session):
        """Test that capabilities are included in unified session."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            session = LLMToolDetector.detect()

            # Check capabilities are present
            assert session.capabilities is not None
            assert session.capabilities.tool_name == "claude"
            assert session.capabilities.format_type == FormatType.MARKDOWN
            assert session.capabilities.supports_markdown is True


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_is_llm_tool_active_true(self, mock_claude_session):
        """Test is_llm_tool_active when active."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            assert is_llm_tool_active() is True

    def test_is_llm_tool_active_false(self):
        """Test is_llm_tool_active when inactive."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            with patch(
                "src.python.common.core.context_injection.llm_tool_detector.CopilotDetector.detect"
            ) as mock_copilot:
                mock_claude.return_value = ClaudeCodeSession(
                    is_active=False, entrypoint=None, detection_method=None
                )
                mock_copilot.return_value = CopilotSession(
                    is_active=False, session_type=None, detection_method=None
                )

                assert is_llm_tool_active() is False

    def test_get_active_llm_tool(self, mock_claude_session):
        """Test get_active_llm_tool function."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            session = get_active_llm_tool()

            assert isinstance(session, UnifiedLLMSession)
            assert session.tool_type == LLMToolType.CLAUDE_CODE
            assert session.is_active is True

    def test_get_llm_formatter_function(self, mock_claude_session):
        """Test get_llm_formatter function."""
        with patch(
            "src.python.common.core.context_injection.llm_tool_detector.ClaudeCodeDetector.detect"
        ) as mock_claude:
            mock_claude.return_value = mock_claude_session

            formatter = get_llm_formatter()

            assert formatter is not None
            assert formatter.capabilities.tool_name == "claude"
