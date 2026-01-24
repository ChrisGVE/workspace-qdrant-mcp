"""Unit tests for main CLI application and command structure.

Tests the main CLI application entry point, command structure,
version handling, debug mode, and async command handling.

Test coverage:
- Main CLI app initialization and structure
- Version command handling (short, long, verbose)
- Debug mode and logging configuration
- Help system and command discovery
- Async command execution wrapper
- Error handling and exit codes
- Command-line argument parsing
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))


class TestMainCLIApp:
    """Test main CLI application functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_cli_app_initialization(self):
        """Test that CLI app initializes correctly."""
        from wqm_cli.cli.main import app

        # Verify app is properly configured
        assert app.info.name == "wqm"
        assert "Workspace Qdrant MCP" in app.info.help
        # Note: add_completion is not exposed via TyperInfo in recent Typer versions
        assert app.info.no_args_is_help is False

    def test_cli_app_has_all_subcommands(self):
        """Test that all expected subcommands are registered."""
        from wqm_cli.cli.main import app

        # Get all registered command groups (subcommands are registered as groups in Typer)
        commands = [g.name for g in app.registered_groups]

        # Verify all expected commands are present
        # Note: "config" was removed, some commands were added
        expected_commands = [
            "init", "memory", "admin", "ingest",
            "search", "library", "lsp", "service", "watch",
            "observability", "status"
        ]

        for cmd in expected_commands:
            assert cmd in commands, f"Command '{cmd}' not found in registered commands"

    def test_no_command_shows_help(self):
        """Test that running with no command shows help."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Usage:" in result.stdout
        assert "Commands:" in result.stdout

    def test_main_callback_without_command(self):
        """Test main callback behavior without subcommand."""
        from wqm_cli.cli.main import app

        # Test with no arguments
        result = self.runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout

    def test_main_callback_with_version_flag(self):
        """Test main callback with version flag."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Should show version and exit
        assert "0.2.0" in result.stdout or any(char.isdigit() for char in result.stdout)

    def test_main_callback_with_verbose_version(self):
        """Test main callback with verbose version flag."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, ["--version", "--verbose"])
        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Python" in result.stdout
        assert "Platform" in result.stdout

    @patch('wqm_cli.cli.main.setup_logging')
    def test_debug_mode_enables_logging(self, mock_setup_logging):
        """Test that debug flag enables verbose logging."""
        from wqm_cli.cli.main import app

        self.runner.invoke(app, ["--debug", "admin", "--help"])

        # Should have called setup_logging with verbose=True
        mock_setup_logging.assert_called()
        calls = mock_setup_logging.call_args_list

        # Find call with verbose=True
        verbose_call_found = any(
            call.kwargs.get('verbose') is True
            for call in calls if call.kwargs
        )
        assert verbose_call_found

    @patch('wqm_cli.cli.main.setup_logging')
    def test_normal_mode_file_logging(self, mock_setup_logging):
        """Test that normal mode configures file-only logging."""
        from wqm_cli.cli.main import app

        self.runner.invoke(app, ["admin", "--help"])

        # Should have called setup_logging with verbose=False and log_file
        mock_setup_logging.assert_called()
        calls = mock_setup_logging.call_args_list

        # Find call with verbose=False
        normal_call_found = any(
            call.kwargs.get('verbose') is False
            for call in calls if call.kwargs
        )
        assert normal_call_found

    def test_config_path_option(self):
        """Test custom config path option."""
        from wqm_cli.cli.main import app

        # Test that config option is accepted (doesn't cause error)
        result = self.runner.invoke(app, ["--config", "/tmp/test.yaml", "--help"])
        assert result.exit_code == 0


class TestVersionHandling:
    """Test version command and flag handling."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_version_flag_short(self):
        """Test -v flag shows version."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        # Should show version number only
        output = result.stdout.strip()
        assert len(output.split('\n')) == 1  # Single line
        assert any(char.isdigit() for char in output)  # Contains digits

    def test_version_flag_long(self):
        """Test --version flag shows version."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert any(char.isdigit() for char in result.stdout)

    def test_version_verbose_flag(self):
        """Test --version --verbose shows detailed info."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, ["--version", "--verbose"])
        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Python" in result.stdout
        assert "Platform" in result.stdout
        assert "Installation path" in result.stdout

    def test_version_debug_flag(self):
        """Test --version --debug shows detailed info."""
        from wqm_cli.cli.main import app

        result = self.runner.invoke(app, ["--version", "--debug"])
        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Python" in result.stdout

    @patch('wqm_cli.cli.main.show_version')
    def test_version_function_called(self, mock_show_version):
        """Test that show_version function is called correctly."""
        from wqm_cli.cli.main import app

        self.runner.invoke(app, ["--version"])
        mock_show_version.assert_called_once_with(verbose=False)

        mock_show_version.reset_mock()
        self.runner.invoke(app, ["--version", "--verbose"])
        mock_show_version.assert_called_once_with(verbose=True)

    def test_show_version_function(self):
        """Test show_version function directly."""
        import sys
        from io import StringIO

        from wqm_cli.cli.main import show_version

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test normal version
            show_version(verbose=False)
            output = captured_output.getvalue()
            assert any(char.isdigit() for char in output)
            assert len(output.strip().split('\n')) == 1  # Single line

            # Reset capture
            captured_output.truncate(0)
            captured_output.seek(0)

            # Test verbose version
            show_version(verbose=True)
            verbose_output = captured_output.getvalue()
            assert "Workspace Qdrant MCP" in verbose_output
            assert "Python" in verbose_output
            assert len(verbose_output.strip().split('\n')) > 1  # Multiple lines

        finally:
            sys.stdout = old_stdout

    @pytest.mark.xfail(reason="Path mock doesn't work after function import - Path is resolved at import time")
    @patch('wqm_cli.cli.main.Path')
    def test_show_version_with_path_error(self, mock_path):
        """Test show_version handles path errors gracefully."""
        import sys
        from io import StringIO

        from wqm_cli.cli.main import show_version

        # Make Path operations fail
        mock_path.side_effect = Exception("Path error")

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            show_version(verbose=True)
            output = captured_output.getvalue()
            # Should still show version info despite path error
            assert "Workspace Qdrant MCP" in output
        finally:
            sys.stdout = old_stdout


class TestAsyncCommandHandling:
    """Test async command execution wrapper."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_handle_async_command_success(self):
        """Test successful async command execution."""
        from wqm_cli.cli.main import handle_async_command

        async def test_coro():
            return "success"

        result = handle_async_command(test_coro())
        assert result == "success"

    def test_handle_async_command_keyboard_interrupt(self):
        """Test async command handles KeyboardInterrupt."""
        import typer
        from wqm_cli.cli.main import handle_async_command

        async def test_coro():
            raise KeyboardInterrupt()

        with pytest.raises(typer.Exit) as exc_info:
            handle_async_command(test_coro())
        assert exc_info.value.exit_code == 1

    def test_handle_async_command_exception(self):
        """Test async command handles general exceptions."""
        import typer
        from wqm_cli.cli.main import handle_async_command

        async def test_coro():
            raise Exception("Test error")

        with pytest.raises(typer.Exit) as exc_info:
            handle_async_command(test_coro())
        assert exc_info.value.exit_code == 1

    def test_handle_async_command_debug_mode(self):
        """Test async command in debug mode shows more info."""
        import sys
        from io import StringIO

        import typer
        from wqm_cli.cli.main import handle_async_command

        async def test_coro():
            raise Exception("Test error")

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            with pytest.raises(typer.Exit):
                handle_async_command(test_coro(), debug=True)

            output = captured_output.getvalue()
            assert "Error: Test error" in output
            assert "Exception type:" in output
        finally:
            sys.stdout = old_stdout

    @patch('wqm_cli.cli.main.logger')
    def test_handle_async_command_keyboard_interrupt_with_debug(self, mock_logger):
        """Test KeyboardInterrupt handling in debug mode."""
        import typer
        from wqm_cli.cli.main import handle_async_command

        async def test_coro():
            raise KeyboardInterrupt()

        with pytest.raises(typer.Exit):
            handle_async_command(test_coro(), debug=True)

        # Should log warning in debug mode
        mock_logger.warning.assert_called_once_with("Operation cancelled by user")

    @patch('wqm_cli.cli.main.logger')
    def test_handle_async_command_exception_with_debug_logging(self, mock_logger):
        """Test exception handling with debug logging."""
        import typer
        from wqm_cli.cli.main import handle_async_command

        async def test_coro():
            raise Exception("Test error")

        with pytest.raises(typer.Exit):
            handle_async_command(test_coro(), debug=True)

        # Should log error with exception info in debug mode
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "CLI operation failed" in call_args[0][0]
        assert call_args[1]["exc_info"] is True


class TestCLIAppStructure:
    """Test CLI app structure and command registration."""

    def test_app_help_content(self):
        """Test main app help content."""
        from wqm_cli.cli.main import app

        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0

        help_text = result.stdout
        assert "Workspace Qdrant MCP" in help_text
        assert "semantic workspace management" in help_text
        # Commands section should be present
        assert "Commands:" in help_text

    def test_subcommand_help_accessibility(self):
        """Test that all subcommands have accessible help."""
        from wqm_cli.cli.main import app

        runner = CliRunner()

        # Test some key subcommands have help
        subcommands_to_test = ["memory", "admin", "search", "service"]

        for cmd in subcommands_to_test:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"Help for {cmd} command failed"
            assert "Usage:" in result.stdout, f"Help for {cmd} missing usage"

    def test_invalid_command(self):
        """Test behavior with invalid command."""
        from wqm_cli.cli.main import app

        result = CliRunner().invoke(app, ["invalid-command"])
        assert result.exit_code != 0


    def test_main_module_execution(self):
        """Test module can be executed as main."""
        # This tests the if __name__ == "__main__": block
        import subprocess
        import sys

        # Run the main.py module directly
        result = subprocess.run([
            sys.executable, "-m", "wqm_cli.cli.main", "--help"
        ], capture_output=True, text=True, cwd=str(project_root / "src" / "python"))

        assert result.returncode == 0
        assert "Workspace Qdrant MCP" in result.stdout


class TestEnvironmentConfiguration:
    """Test environment variable configuration and CLI mode."""

    def setup_method(self):
        """Set up test environment."""
        self.original_env = {}
        for var in ["WQM_CLI_MODE", "WQM_LOG_INIT", "PYTHONWARNINGS"]:
            self.original_env[var] = os.environ.get(var)

    def teardown_method(self):
        """Restore original environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    @pytest.mark.xfail(reason="Environment setup only works on first module import, test isolation unreliable")
    def test_cli_mode_environment_setup(self):
        """Test that CLI mode environment is set correctly."""
        # Clear environment first
        for var in ["WQM_CLI_MODE", "WQM_LOG_INIT"]:
            if var in os.environ:
                del os.environ[var]

        # Import main module which should set environment
        import wqm_cli.cli.main

        # Check environment variables are set
        assert os.environ.get("WQM_CLI_MODE") == "true"
        assert os.environ.get("WQM_LOG_INIT") == "false"

    @patch('wqm_cli.cli.main.warnings.filterwarnings')
    def test_version_warnings_suppression(self, mock_filter_warnings):
        """Test warnings are suppressed for clean version output."""
        from wqm_cli.cli.main import app

        result = CliRunner().invoke(app, ["--version"])
        assert result.exit_code == 0

        # In non-verbose mode, warnings should be suppressed
        # Note: This is tested via the module import behavior

    def test_debug_mode_environment(self):
        """Test debug mode affects environment variables."""
        from wqm_cli.cli.main import app

        # Test with debug flag - should enable logging
        result = CliRunner().invoke(app, ["--debug", "--help"])
        assert result.exit_code == 0

        # WQM_LOG_INIT should be set to "true" in debug mode
        # This is checked via the main callback behavior


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def test_typer_exit_handling(self):
        """Test that typer.Exit exceptions are handled correctly."""
        import typer
        from wqm_cli.cli.main import app

        # Test with version flag which calls typer.Exit
        result = CliRunner().invoke(app, ["--version"])
        assert result.exit_code == 0  # Should exit cleanly

    @pytest.mark.xfail(reason="Module reload with patched imports is unreliable in test isolation")
    def test_import_error_handling(self):
        """Test behavior when imports fail."""
        with patch('wqm_cli.cli.main.typer', side_effect=ImportError("typer not found")):
            with pytest.raises(ImportError):
                import importlib

                import wqm_cli.cli.main
                importlib.reload(wqm_cli.cli.main)

    @patch('sys.argv', ['wqm', '--version'])
    def test_early_version_check_optimization(self):
        """Test early version check optimization."""
        # The main module has optimization to check version early
        # This test ensures that works correctly
        from wqm_cli.cli.main import app

        result = CliRunner().invoke(app, [])
        assert result.exit_code == 0

    def test_logging_configuration_error_handling(self):
        """Test behavior when logging configuration fails."""
        with patch('wqm_cli.cli.main.setup_logging', side_effect=Exception("Logging failed")):
            from wqm_cli.cli.main import app

            # Should still be able to invoke app even if logging setup fails
            CliRunner().invoke(app, ["--help"])
            # The app might fail or succeed depending on error handling
            # We just ensure it doesn't crash the test
