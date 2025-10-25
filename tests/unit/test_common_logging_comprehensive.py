"""
Comprehensive unit tests for python.common.logging.loguru_config module.

Tests cover loguru configuration, MCP stdio mode compatibility,
and OS-standard log directory usage with 100% coverage.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import modules under test
from src.python.common.logging.loguru_config import _is_mcp_stdio_mode, setup_logging


class TestSetupLogging:
    """Test setup_logging function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        with patch('src.python.common.logging.loguru_config.logger') as mock_logger:
            yield mock_logger

    @pytest.fixture
    def mock_os_directories(self):
        """Create a mock OSDirectories."""
        with patch('src.python.common.logging.loguru_config.OSDirectories') as mock_class:
            mock_instance = Mock()
            mock_instance.ensure_directories.return_value = None
            mock_instance.get_log_file.return_value = Path("/test/logs/workspace.log")
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_setup_logging_default_os_standard(self, mock_logger, mock_os_directories):
        """Test setup_logging with OS-standard log directory."""
        setup_logging()

        # Should remove default handler
        mock_logger.remove.assert_called_once()

        # Should ensure directories exist
        mock_os_directories.ensure_directories.assert_called_once()

        # Should get log file path
        mock_os_directories.get_log_file.assert_called_once_with("workspace.log")

        # Should add file handler
        mock_logger.add.assert_called()

        # Check file handler was added with correct parameters
        file_add_call = mock_logger.add.call_args_list[0]
        assert file_add_call[0][0] == Path("/test/logs/workspace.log")
        assert "rotation" in file_add_call[1]
        assert "retention" in file_add_call[1]
        assert "compression" in file_add_call[1]
        assert "format" in file_add_call[1]

    def test_setup_logging_custom_log_file(self, mock_logger):
        """Test setup_logging with custom log file."""
        custom_path = "/custom/path/test.log"

        with patch('pathlib.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            setup_logging(log_file=custom_path)

            # Should create Path from custom path
            mock_path_class.assert_called_with(custom_path)

            # Should create parent directories
            mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Should add file handler with custom path
            mock_logger.add.assert_called()
            file_add_call = mock_logger.add.call_args_list[0]
            assert file_add_call[0][0] == mock_path

    def test_setup_logging_verbose_not_mcp_mode(self, mock_logger):
        """Test setup_logging with verbose=True and not in MCP mode."""
        with patch('src.python.common.logging.loguru_config._is_mcp_stdio_mode', return_value=False):
            with patch('src.python.common.logging.loguru_config.OSDirectories'):
                setup_logging(verbose=True)

                # Should add both file and console handlers
                assert mock_logger.add.call_count == 2

                # Check console handler (second call)
                console_add_call = mock_logger.add.call_args_list[1]
                assert console_add_call[0][0] == sys.stderr
                assert "format" in console_add_call[1]

    def test_setup_logging_verbose_in_mcp_mode(self, mock_logger):
        """Test setup_logging with verbose=True but in MCP mode."""
        with patch('src.python.common.logging.loguru_config._is_mcp_stdio_mode', return_value=True):
            with patch('src.python.common.logging.loguru_config.OSDirectories'):
                setup_logging(verbose=True)

                # Should only add file handler, not console
                assert mock_logger.add.call_count == 1

    def test_setup_logging_not_verbose(self, mock_logger):
        """Test setup_logging with verbose=False."""
        with patch('src.python.common.logging.loguru_config._is_mcp_stdio_mode', return_value=False):
            with patch('src.python.common.logging.loguru_config.OSDirectories'):
                setup_logging(verbose=False)

                # Should only add file handler
                assert mock_logger.add.call_count == 1

    def test_setup_logging_log_path_creation(self, mock_logger, mock_os_directories):
        """Test that log path parent directories are created."""
        # Mock Path to track mkdir calls
        mock_path = Mock()
        mock_path.parent.mkdir = Mock()
        mock_os_directories.get_log_file.return_value = mock_path

        setup_logging()

        # Should create parent directories
        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_setup_logging_file_handler_parameters(self, mock_logger, mock_os_directories):
        """Test file handler is configured with correct parameters."""
        setup_logging()

        # Get file handler call
        file_add_call = mock_logger.add.call_args_list[0]
        kwargs = file_add_call[1]

        assert kwargs["rotation"] == "10 MB"
        assert kwargs["retention"] == "7 days"
        assert kwargs["compression"] == "gz"
        assert "format" in kwargs
        assert "{time:YYYY-MM-DD HH:mm:ss.SSS}" in kwargs["format"]
        assert "{level: <8}" in kwargs["format"]
        assert "{name}:{function}:{line}" in kwargs["format"]
        assert "{message}" in kwargs["format"]

    def test_setup_logging_console_handler_format(self, mock_logger):
        """Test console handler is configured with correct format."""
        with patch('src.python.common.logging.loguru_config._is_mcp_stdio_mode', return_value=False):
            with patch('src.python.common.logging.loguru_config.OSDirectories'):
                setup_logging(verbose=True)

                # Get console handler call (second call)
                console_add_call = mock_logger.add.call_args_list[1]
                kwargs = console_add_call[1]

                assert "format" in kwargs
                format_str = kwargs["format"]
                assert "<green>{time:HH:mm:ss.SSS}</green>" in format_str
                assert "<level>{level: <8}</level>" in format_str
                assert "<cyan>{name}</cyan>" in format_str
                assert "<level>{message}</level>" in format_str

    def test_setup_logging_logger_info_call(self, mock_logger, mock_os_directories):
        """Test that logger.info is called with log path information."""
        setup_logging()

        # Should call logger.info with log path information
        mock_logger.info.assert_called_once()
        info_call_args = mock_logger.info.call_args[0][0]
        assert "Using OS-standard log directory" in info_call_args
        assert "/test/logs/workspace.log" in info_call_args

    def test_setup_logging_legacy_warning(self, mock_logger):
        """Test that logger.warning is called for legacy log path."""
        custom_path = "/custom/legacy.log"

        with patch('pathlib.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            setup_logging(log_file=custom_path)

            # Should call logger.warning for legacy path
            mock_logger.warning.assert_called_once()
            warning_call_args = mock_logger.warning.call_args[0][0]
            assert "Using legacy log path" in warning_call_args
            assert custom_path in warning_call_args
            assert "Consider migrating to OS-standard location" in warning_call_args


class TestIsMcpStdioMode:
    """Test _is_mcp_stdio_mode function."""

    def test_wqm_stdio_mode_true(self):
        """Test MCP stdio mode detection with WQM_STDIO_MODE=true."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            assert _is_mcp_stdio_mode()

    def test_wqm_stdio_mode_true_uppercase(self):
        """Test MCP stdio mode detection with WQM_STDIO_MODE=TRUE."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "TRUE"}):
            assert _is_mcp_stdio_mode()

    def test_wqm_stdio_mode_false(self):
        """Test MCP stdio mode detection with WQM_STDIO_MODE=false."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "false"}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=True):
                    assert not _is_mcp_stdio_mode()

    def test_mcp_quiet_mode_true(self):
        """Test MCP stdio mode detection with MCP_QUIET_MODE=true."""
        with patch.dict(os.environ, {"MCP_QUIET_MODE": "true"}):
            assert _is_mcp_stdio_mode()

    def test_mcp_quiet_mode_false(self):
        """Test MCP stdio mode detection with MCP_QUIET_MODE=false."""
        with patch.dict(os.environ, {"MCP_QUIET_MODE": "false"}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=True):
                    assert not _is_mcp_stdio_mode()

    def test_no_tty_no_term(self):
        """Test MCP stdio mode detection with no TTY and no TERM."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=False):
                with patch('sys.stdin.isatty', return_value=False):
                    assert _is_mcp_stdio_mode()

    def test_has_tty(self):
        """Test MCP stdio mode detection with TTY available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=True):
                    assert not _is_mcp_stdio_mode()

    def test_has_term_env(self):
        """Test MCP stdio mode detection with TERM environment variable."""
        with patch.dict(os.environ, {"TERM": "xterm"}, clear=True):
            with patch('sys.stdout.isatty', return_value=False):
                with patch('sys.stdin.isatty', return_value=False):
                    assert not _is_mcp_stdio_mode()

    def test_stdout_tty_stdin_not_tty(self):
        """Test MCP stdio mode detection with mixed TTY availability."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=False):
                    assert not _is_mcp_stdio_mode()

    def test_stdout_not_tty_stdin_tty(self):
        """Test MCP stdio mode detection with mixed TTY availability."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=False):
                with patch('sys.stdin.isatty', return_value=True):
                    assert not _is_mcp_stdio_mode()

    def test_priority_order_wqm_over_tty(self):
        """Test that WQM_STDIO_MODE takes priority over TTY detection."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=True):
                    assert _is_mcp_stdio_mode()

    def test_priority_order_mcp_quiet_over_tty(self):
        """Test that MCP_QUIET_MODE takes priority over TTY detection."""
        with patch.dict(os.environ, {"MCP_QUIET_MODE": "true"}):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=True):
                    assert _is_mcp_stdio_mode()

    def test_empty_env_vars(self):
        """Test behavior with empty environment variables."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "", "MCP_QUIET_MODE": ""}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                with patch('sys.stdin.isatty', return_value=True):
                    assert not _is_mcp_stdio_mode()

    def test_case_insensitive_env_vars(self):
        """Test case insensitive handling of environment variables."""
        test_cases = [
            ("True", True),
            ("TRUE", True),
            ("true", True),
            ("tRuE", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
            ("anything_else", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"WQM_STDIO_MODE": env_value}, clear=True):
                with patch('sys.stdout.isatty', return_value=True):
                    with patch('sys.stdin.isatty', return_value=True):
                        result = _is_mcp_stdio_mode()
                        assert result == expected, f"Expected {expected} for '{env_value}', got {result}"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_complete_logging_setup_mcp_mode(self):
        """Test complete logging setup in MCP mode."""
        with patch('src.python.common.logging.loguru_config.logger') as mock_logger:
            with patch('src.python.common.logging.loguru_config.OSDirectories') as mock_os_dirs_class:
                mock_os_dirs = Mock()
                mock_os_dirs.ensure_directories.return_value = None
                mock_os_dirs.get_log_file.return_value = Path("/app/logs/workspace.log")
                mock_os_dirs_class.return_value = mock_os_dirs

                with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
                    setup_logging(verbose=True)

                    # Should remove default handler
                    mock_logger.remove.assert_called_once()

                    # Should only add file handler (no console in MCP mode)
                    assert mock_logger.add.call_count == 1

                    # Should use OS directories
                    mock_os_dirs.ensure_directories.assert_called_once()
                    mock_os_dirs.get_log_file.assert_called_once_with("workspace.log")

    def test_complete_logging_setup_interactive_mode(self):
        """Test complete logging setup in interactive mode."""
        with patch('src.python.common.logging.loguru_config.logger') as mock_logger:
            with patch('src.python.common.logging.loguru_config.OSDirectories') as mock_os_dirs_class:
                mock_os_dirs = Mock()
                mock_os_dirs.ensure_directories.return_value = None
                mock_os_dirs.get_log_file.return_value = Path("/user/logs/workspace.log")
                mock_os_dirs_class.return_value = mock_os_dirs

                with patch.dict(os.environ, {}, clear=True):
                    with patch('sys.stdout.isatty', return_value=True):
                        with patch('sys.stdin.isatty', return_value=True):
                            setup_logging(verbose=True)

                            # Should remove default handler
                            mock_logger.remove.assert_called_once()

                            # Should add both file and console handlers
                            assert mock_logger.add.call_count == 2

                            # Check file handler (first call)
                            file_call = mock_logger.add.call_args_list[0]
                            assert file_call[0][0] == Path("/user/logs/workspace.log")

                            # Check console handler (second call)
                            console_call = mock_logger.add.call_args_list[1]
                            assert console_call[0][0] == sys.stderr

    def test_legacy_path_with_directory_creation(self):
        """Test legacy path usage with automatic directory creation."""
        custom_path = "/custom/nested/deep/log.txt"

        with patch('src.python.common.logging.loguru_config.logger') as mock_logger:
            with patch('pathlib.Path') as mock_path_class:
                mock_path = Mock()
                mock_parent = Mock()
                mock_path.parent = mock_parent
                mock_path_class.return_value = mock_path

                setup_logging(log_file=custom_path)

                # Should create Path object
                mock_path_class.assert_called_with(custom_path)

                # Should create parent directories
                mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

                # Should add file handler with custom path
                file_call = mock_logger.add.call_args_list[0]
                assert file_call[0][0] == mock_path

                # Should log warning about legacy path
                mock_logger.warning.assert_called_once()

    def test_error_handling_with_os_directories(self):
        """Test error handling when OS directories operations fail."""
        with patch('src.python.common.logging.loguru_config.logger'):
            with patch('src.python.common.logging.loguru_config.OSDirectories') as mock_os_dirs_class:
                mock_os_dirs = Mock()
                mock_os_dirs.ensure_directories.side_effect = Exception("Directory creation failed")
                mock_os_dirs_class.return_value = mock_os_dirs

                # Should still attempt setup even if directory creation fails
                with pytest.raises(Exception, match="Directory creation failed"):
                    setup_logging()

    def test_environment_variable_combinations(self):
        """Test various combinations of environment variables."""
        test_scenarios = [
            # (env_vars, expected_mcp_mode)
            ({"WQM_STDIO_MODE": "true"}, True),
            ({"MCP_QUIET_MODE": "true"}, True),
            ({"WQM_STDIO_MODE": "true", "MCP_QUIET_MODE": "false"}, True),
            ({"WQM_STDIO_MODE": "false", "MCP_QUIET_MODE": "true"}, True),
            ({"WQM_STDIO_MODE": "false", "MCP_QUIET_MODE": "false"}, False),
            ({}, False),  # No env vars, with TTY
        ]

        for env_vars, expected_mcp_mode in test_scenarios:
            with patch.dict(os.environ, env_vars, clear=True):
                with patch('sys.stdout.isatty', return_value=True):
                    with patch('sys.stdin.isatty', return_value=True):
                        result = _is_mcp_stdio_mode()
                        assert result == expected_mcp_mode, f"Failed for env_vars={env_vars}"


if __name__ == "__main__":
    pytest.main([__file__])
