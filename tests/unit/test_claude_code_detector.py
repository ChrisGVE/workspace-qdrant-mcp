"""
Unit tests for Claude Code session detection.

Tests ClaudeCodeDetector, ClaudeCodeSession, and detection methods.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.python.common.core.context_injection.claude_code_detector import (
    ClaudeCodeDetector,
    ClaudeCodeSession,
    get_claude_code_session,
    is_claude_code_session,
)


class TestClaudeCodeSession:
    """Test ClaudeCodeSession dataclass."""

    def test_session_active_with_entrypoint(self):
        """Test session with active status and entrypoint."""
        session = ClaudeCodeSession(
            is_active=True, entrypoint="cli", detection_method="environment_variable"
        )

        assert session.is_active is True
        assert session.entrypoint == "cli"
        assert session.detection_method == "environment_variable"

    def test_session_inactive(self):
        """Test inactive session."""
        session = ClaudeCodeSession(
            is_active=False, entrypoint=None, detection_method=None
        )

        assert session.is_active is False
        assert session.entrypoint is None
        assert session.detection_method is None

    def test_session_active_without_entrypoint(self):
        """Test session active but without entrypoint information."""
        session = ClaudeCodeSession(
            is_active=True, entrypoint=None, detection_method="parent_process"
        )

        assert session.is_active is True
        assert session.entrypoint is None
        assert session.detection_method == "parent_process"


class TestClaudeCodeDetector:
    """Test ClaudeCodeDetector detection methods."""

    def test_detect_via_claudecode_env_variable(self, monkeypatch):
        """Test detection via CLAUDECODE environment variable."""
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")

        session = ClaudeCodeDetector.detect()

        assert session.is_active is True
        assert session.entrypoint == "cli"
        assert session.detection_method == "environment_variable_claudecode"

    def test_detect_via_claudecode_without_entrypoint(self, monkeypatch):
        """Test detection via CLAUDECODE without entrypoint."""
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
        monkeypatch.setenv("CLAUDECODE", "1")

        session = ClaudeCodeDetector.detect()

        assert session.is_active is True
        assert session.entrypoint is None
        assert session.detection_method == "environment_variable_claudecode"

    def test_detect_via_entrypoint_env_variable(self, monkeypatch):
        """Test detection via CLAUDE_CODE_ENTRYPOINT environment variable."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "api")

        session = ClaudeCodeDetector.detect()

        assert session.is_active is True
        assert session.entrypoint == "api"
        assert session.detection_method == "environment_variable_entrypoint"

    def test_detect_via_parent_process_name(self, monkeypatch):
        """Test detection via parent process name."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock psutil to simulate claude parent process
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "claude"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            assert session.is_active is True
            assert session.entrypoint is None
            assert session.detection_method == "parent_process"

    def test_detect_via_parent_process_cmdline(self, monkeypatch):
        """Test detection via parent process command line."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock psutil to simulate claude in cmdline but not in name
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "node"
        mock_parent.cmdline.return_value = ["/usr/bin/claude", "--verbose"]
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            assert session.is_active is True
            assert session.entrypoint is None
            assert session.detection_method == "parent_process"

    def test_detect_no_claude_session(self, monkeypatch):
        """Test detection when no Claude Code session exists."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock psutil to simulate non-claude parent process
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "bash"
        mock_parent.cmdline.return_value = ["/bin/bash"]
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            assert session.is_active is False
            assert session.entrypoint is None
            assert session.detection_method is None

    def test_detect_psutil_not_available(self, monkeypatch):
        """Test detection when psutil is not available."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Temporarily hide psutil from sys.modules
        original_psutil = sys.modules.get("psutil")
        if "psutil" in sys.modules:
            del sys.modules["psutil"]

        try:
            # Mock psutil import to raise ImportError
            with patch.dict("sys.modules", {"psutil": None}):

                def mock_import(name, *args, **kwargs):
                    if name == "psutil":
                        raise ImportError("psutil not installed")
                    return __import__(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    session = ClaudeCodeDetector.detect()

                    # Should return inactive session since no env vars and no psutil
                    assert session.is_active is False
                    assert session.entrypoint is None
                    assert session.detection_method is None
        finally:
            # Restore original psutil
            if original_psutil:
                sys.modules["psutil"] = original_psutil

    def test_detect_parent_process_none(self, monkeypatch):
        """Test detection when parent process is None."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock psutil to return None parent (orphaned process)
        mock_process = MagicMock()
        mock_process.parent.return_value = None

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            assert session.is_active is False
            assert session.entrypoint is None
            assert session.detection_method is None

    def test_detect_parent_process_access_denied(self, monkeypatch):
        """Test detection when parent process access is denied."""
        import psutil

        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock psutil to raise AccessDenied on cmdline access
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "some_process"
        mock_parent.cmdline.side_effect = psutil.AccessDenied()
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            # Should return inactive since can't verify parent and no env vars
            assert session.is_active is False

    def test_is_active_convenience_method(self, monkeypatch):
        """Test is_active() convenience method."""
        monkeypatch.setenv("CLAUDECODE", "1")
        assert ClaudeCodeDetector.is_active() is True

        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock non-claude parent
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "bash"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            assert ClaudeCodeDetector.is_active() is False


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_is_claude_code_session(self, monkeypatch):
        """Test is_claude_code_session() convenience function."""
        monkeypatch.setenv("CLAUDECODE", "1")
        assert is_claude_code_session() is True

        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock non-claude parent
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "bash"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            assert is_claude_code_session() is False

    def test_get_claude_code_session(self, monkeypatch):
        """Test get_claude_code_session() convenience function."""
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")

        session = get_claude_code_session()

        assert isinstance(session, ClaudeCodeSession)
        assert session.is_active is True
        assert session.entrypoint == "cli"
        assert session.detection_method == "environment_variable_claudecode"


class TestDetectionPriority:
    """Test detection method priority."""

    def test_env_var_takes_priority_over_parent_process(self, monkeypatch):
        """Test that environment variable detection takes priority."""
        monkeypatch.setenv("CLAUDECODE", "1")

        # Even if parent process is not claude, env var should take priority
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "bash"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            assert session.is_active is True
            assert session.detection_method == "environment_variable_claudecode"

    def test_claudecode_takes_priority_over_entrypoint(self, monkeypatch):
        """Test that CLAUDECODE takes priority over CLAUDE_CODE_ENTRYPOINT."""
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")

        session = ClaudeCodeDetector.detect()

        # Should use CLAUDECODE method even though both are set
        assert session.detection_method == "environment_variable_claudecode"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_claudecode_not_equal_to_one(self, monkeypatch):
        """Test CLAUDECODE set to value other than '1'."""
        monkeypatch.setenv("CLAUDECODE", "0")
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock non-claude parent
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "bash"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            # Should be inactive since CLAUDECODE is not "1"
            assert session.is_active is False

    def test_empty_entrypoint_string(self, monkeypatch):
        """Test empty CLAUDE_CODE_ENTRYPOINT string."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "")

        # Mock non-claude parent
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "bash"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            # Empty string is falsy, should be inactive
            assert session.is_active is False

    def test_case_insensitive_parent_process_detection(self, monkeypatch):
        """Test that parent process detection is case-insensitive."""
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

        # Mock psutil with uppercase CLAUDE
        mock_process = MagicMock()
        mock_parent = MagicMock()
        mock_parent.name.return_value = "CLAUDE"
        mock_process.parent.return_value = mock_parent

        with patch("psutil.Process", return_value=mock_process):
            session = ClaudeCodeDetector.detect()

            # Should detect regardless of case
            assert session.is_active is True
            assert session.detection_method == "parent_process"
