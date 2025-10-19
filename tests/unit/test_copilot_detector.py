"""
Unit tests for Copilot/Codex session detection.

Tests the CopilotDetector's ability to detect various AI coding assistant
sessions through environment variables, process inspection, and LSP detection.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.python.common.core.context_injection.copilot_detector import (
    CopilotDetector,
    CopilotSession,
    CopilotSessionType,
    is_copilot_session,
    get_copilot_session,
    get_code_comment_prefix,
)


class TestCopilotDetector:
    """Test suite for CopilotDetector class."""

    def test_no_session_detected_by_default(self):
        """Test that no session is detected in normal Python environment."""
        with patch.dict(os.environ, {}, clear=True):
            # Mock glob to return no Copilot LSP sockets
            with patch("glob.glob", return_value=[]):
                # Mock psutil to return no matching processes
                mock_proc = Mock()
                mock_proc.info = {"name": "python", "cmdline": ["python"]}
                with patch("psutil.process_iter", return_value=[mock_proc]):
                    session = CopilotDetector.detect()

                    assert session.is_active is False
                    assert session.session_type is None
                    assert session.ide_name is None
                    assert session.detection_method is None

    def test_vscode_detection_via_vscode_pid(self):
        """Test VSCode detection via VSCODE_PID environment variable."""
        with patch.dict(os.environ, {"VSCODE_PID": "12345"}, clear=True):
            session = CopilotDetector.detect()

            assert session.is_active is True
            assert session.session_type == CopilotSessionType.GITHUB_COPILOT
            assert session.ide_name == "vscode"
            assert session.detection_method == "environment_variable_vscode"

    def test_vscode_detection_via_ipc_hook(self):
        """Test VSCode detection via VSCODE_IPC_HOOK environment variable."""
        with patch.dict(
            os.environ,
            {"VSCODE_IPC_HOOK": "/tmp/vscode-ipc-socket"},
            clear=True,
        ):
            session = CopilotDetector.detect()

            assert session.is_active is True
            assert session.session_type == CopilotSessionType.GITHUB_COPILOT
            assert session.ide_name == "vscode"

    def test_vscode_insiders_detection(self):
        """Test VSCode Insiders detection via IPC hook path."""
        with patch.dict(
            os.environ,
            {"VSCODE_IPC_HOOK": "/tmp/vscode-insiders-ipc"},
            clear=True,
        ):
            session = CopilotDetector.detect()

            assert session.is_active is True
            assert session.ide_name == "vscode-insiders"

    def test_vscode_detection_via_term_program(self):
        """Test VSCode detection via TERM_PROGRAM environment variable."""
        with patch.dict(
            os.environ,
            {"TERM_PROGRAM": "vscode"},
            clear=True,
        ):
            session = CopilotDetector.detect()

            assert session.is_active is True
            assert session.session_type == CopilotSessionType.GITHUB_COPILOT
            assert session.ide_name == "vscode"
            assert session.detection_method == "environment_variable_term"

    def test_cursor_detection(self):
        """Test Cursor editor detection."""
        with patch.dict(
            os.environ,
            {"CURSOR_SESSION": "active"},
            clear=True,
        ):
            session = CopilotDetector.detect()

            assert session.is_active is True
            assert session.session_type == CopilotSessionType.CURSOR
            assert session.ide_name == "cursor"
            assert session.detection_method == "environment_variable_cursor"

    def test_jetbrains_detection(self):
        """Test JetBrains IDE detection."""
        with patch.dict(
            os.environ,
            {"JETBRAINS_IDE": "pycharm"},
            clear=True,
        ):
            session = CopilotDetector.detect()

            assert session.is_active is True
            assert session.session_type == CopilotSessionType.JETBRAINS_AI
            assert session.ide_name == "pycharm"
            assert session.detection_method == "environment_variable_jetbrains"

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not available"),
        reason="psutil required for process detection",
    )
    def test_process_detection_vscode(self):
        """Test IDE detection via parent process inspection."""
        # Mock psutil process hierarchy
        mock_grandparent = Mock()
        mock_grandparent.name.return_value = "code"
        mock_grandparent.parent.return_value = None

        mock_parent = Mock()
        mock_parent.name.return_value = "bash"
        mock_parent.parent.return_value = mock_grandparent

        mock_process = Mock()
        mock_process.parent.return_value = mock_parent

        with patch.dict(os.environ, {}, clear=True):
            with patch("psutil.Process", return_value=mock_process):
                session = CopilotDetector.detect()

                assert session.is_active is True
                assert session.session_type == CopilotSessionType.GITHUB_COPILOT
                assert session.ide_name == "vscode"
                assert session.detection_method == "parent_process"

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not available"),
        reason="psutil required for process detection",
    )
    def test_process_detection_cursor(self):
        """Test Cursor detection via parent process."""
        mock_parent = Mock()
        mock_parent.name.return_value = "cursor"
        mock_parent.parent.return_value = None

        mock_process = Mock()
        mock_process.parent.return_value = mock_parent

        with patch.dict(os.environ, {}, clear=True):
            with patch("psutil.Process", return_value=mock_process):
                session = CopilotDetector.detect()

                assert session.is_active is True
                assert session.session_type == CopilotSessionType.CURSOR
                assert session.ide_name == "cursor"

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not available"),
        reason="psutil required for process detection",
    )
    def test_process_detection_jetbrains(self):
        """Test JetBrains IDE detection via parent process."""
        mock_parent = Mock()
        mock_parent.name.return_value = "pycharm"
        mock_parent.parent.return_value = None

        mock_process = Mock()
        mock_process.parent.return_value = mock_parent

        with patch.dict(os.environ, {}, clear=True):
            with patch("psutil.Process", return_value=mock_process):
                session = CopilotDetector.detect()

                assert session.is_active is True
                assert session.session_type == CopilotSessionType.JETBRAINS_AI
                assert session.ide_name == "pycharm"

    def test_lsp_socket_detection(self):
        """Test Copilot LSP detection via socket files."""
        with patch("glob.glob", return_value=["/tmp/copilot-lsp-abc123.sock"]):
            with patch.dict(os.environ, {}, clear=True):
                session = CopilotDetector.detect()

                assert session.is_active is True
                assert session.session_type == CopilotSessionType.GITHUB_COPILOT
                assert session.detection_method == "lsp_socket"

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not available"),
        reason="psutil required for LSP process detection",
    )
    def test_lsp_process_detection(self):
        """Test Copilot LSP detection via running process."""
        # Mock a Copilot LSP process
        mock_proc = Mock()
        mock_proc.info = {
            "name": "copilot-lsp",
            "cmdline": ["node", "/path/to/copilot-lsp"],
        }

        with patch("glob.glob", return_value=[]):  # No socket files
            with patch("psutil.process_iter", return_value=[mock_proc]):
                with patch.dict(os.environ, {}, clear=True):
                    session = CopilotDetector.detect()

                    assert session.is_active is True
                    assert session.session_type == CopilotSessionType.GITHUB_COPILOT
                    assert session.detection_method == "lsp_process"

    def test_workspace_detection_from_vscode(self):
        """Test workspace path detection from VSCode environment."""
        workspace_path = "/home/user/project"
        with patch.dict(
            os.environ,
            {
                "VSCODE_PID": "12345",
                "VSCODE_WORKSPACE": workspace_path,
            },
            clear=True,
        ):
            with patch("os.path.isdir", return_value=True):
                session = CopilotDetector.detect()

                assert session.workspace_path == Path(workspace_path)

    def test_workspace_detection_from_pwd(self):
        """Test workspace path detection from PWD."""
        workspace_path = "/home/user/project"
        with patch.dict(
            os.environ,
            {
                "VSCODE_PID": "12345",
                "PWD": workspace_path,
            },
            clear=True,
        ):
            with patch("os.path.isdir", return_value=True):
                session = CopilotDetector.detect()

                assert session.workspace_path == Path(workspace_path)

    def test_is_active_convenience_function(self):
        """Test is_active() convenience method."""
        with patch.dict(os.environ, {"VSCODE_PID": "12345"}, clear=True):
            assert CopilotDetector.is_active() is True

        with patch.dict(os.environ, {}, clear=True):
            # Mock glob to return no Copilot LSP sockets
            with patch("glob.glob", return_value=[]):
                # Mock psutil to return no matching processes
                mock_proc = Mock()
                mock_proc.info = {"name": "python", "cmdline": ["python"]}
                with patch("psutil.process_iter", return_value=[mock_proc]):
                    assert CopilotDetector.is_active() is False

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        with patch.dict(os.environ, {"VSCODE_PID": "12345"}, clear=True):
            # Test is_copilot_session()
            assert is_copilot_session() is True

            # Test get_copilot_session()
            session = get_copilot_session()
            assert isinstance(session, CopilotSession)
            assert session.is_active is True
            assert session.session_type == CopilotSessionType.GITHUB_COPILOT


class TestCommentStyleDetection:
    """Test suite for programming language comment style detection."""

    def test_python_comment_style(self):
        """Test Python comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("python")

        assert line_prefix == "#"
        assert block_style == ('"""', '"""')

    def test_javascript_comment_style(self):
        """Test JavaScript comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("javascript")

        assert line_prefix == "//"
        assert block_style == ("/*", "*/")

    def test_rust_comment_style(self):
        """Test Rust comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("rust")

        assert line_prefix == "//"
        assert block_style == ("/*", "*/")

    def test_go_comment_style(self):
        """Test Go comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("go")

        assert line_prefix == "//"
        assert block_style == ("/*", "*/")

    def test_ruby_comment_style(self):
        """Test Ruby comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("ruby")

        assert line_prefix == "#"
        assert block_style == ("=begin", "=end")

    def test_shell_comment_style(self):
        """Test shell script comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("shell")

        assert line_prefix == "#"
        assert block_style is None

    def test_haskell_comment_style(self):
        """Test Haskell comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("haskell")

        assert line_prefix == "--"
        assert block_style == ("{-", "-}")

    def test_lua_comment_style(self):
        """Test Lua comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("lua")

        assert line_prefix == "--"
        assert block_style == ("--[[", "]]")

    def test_sql_comment_style(self):
        """Test SQL comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("sql")

        assert line_prefix == "--"
        assert block_style == ("/*", "*/")

    def test_html_comment_style(self):
        """Test HTML comment style."""
        line_prefix, block_style = CopilotDetector.get_comment_style("html")

        assert line_prefix == "<!--"
        assert block_style == ("<!--", "-->")

    def test_case_insensitive_language_detection(self):
        """Test that language detection is case-insensitive."""
        styles = [
            CopilotDetector.get_comment_style("Python"),
            CopilotDetector.get_comment_style("PYTHON"),
            CopilotDetector.get_comment_style("python"),
        ]

        # All should return the same style
        assert len(set(styles)) == 1
        assert styles[0] == ("#", ('"""', '"""'))

    def test_unknown_language_defaults(self):
        """Test that unknown languages get sensible defaults."""
        line_prefix, block_style = CopilotDetector.get_comment_style("unknownlang")

        assert line_prefix == "#"  # Default to hash comment
        assert block_style is None

    def test_get_code_comment_prefix_convenience(self):
        """Test get_code_comment_prefix() convenience function."""
        assert get_code_comment_prefix("python") == "#"
        assert get_code_comment_prefix("javascript") == "//"
        assert get_code_comment_prefix("rust") == "//"
        assert get_code_comment_prefix("shell") == "#"


class TestCopilotSessionDataclass:
    """Test the CopilotSession dataclass."""

    def test_minimal_session_creation(self):
        """Test creating session with minimal fields."""
        session = CopilotSession(is_active=False)

        assert session.is_active is False
        assert session.session_type is None
        assert session.ide_name is None
        assert session.ide_version is None
        assert session.detection_method is None
        assert session.workspace_path is None

    def test_full_session_creation(self):
        """Test creating session with all fields."""
        workspace = Path("/home/user/project")
        session = CopilotSession(
            is_active=True,
            session_type=CopilotSessionType.GITHUB_COPILOT,
            ide_name="vscode",
            ide_version="1.75.0",
            detection_method="environment_variable",
            workspace_path=workspace,
        )

        assert session.is_active is True
        assert session.session_type == CopilotSessionType.GITHUB_COPILOT
        assert session.ide_name == "vscode"
        assert session.ide_version == "1.75.0"
        assert session.detection_method == "environment_variable"
        assert session.workspace_path == workspace


class TestCopilotSessionType:
    """Test the CopilotSessionType enum."""

    def test_all_session_types_have_values(self):
        """Test that all session types have string values."""
        assert CopilotSessionType.GITHUB_COPILOT.value == "github_copilot"
        assert CopilotSessionType.CODEX_API.value == "codex_api"
        assert CopilotSessionType.CURSOR.value == "cursor"
        assert CopilotSessionType.JETBRAINS_AI.value == "jetbrains_ai"
        assert CopilotSessionType.TABNINE.value == "tabnine"
        assert CopilotSessionType.UNKNOWN.value == "unknown"

    def test_enum_comparison(self):
        """Test enum value comparison."""
        session1 = CopilotSession(
            is_active=True,
            session_type=CopilotSessionType.GITHUB_COPILOT,
        )
        session2 = CopilotSession(
            is_active=True,
            session_type=CopilotSessionType.GITHUB_COPILOT,
        )

        assert session1.session_type == session2.session_type
