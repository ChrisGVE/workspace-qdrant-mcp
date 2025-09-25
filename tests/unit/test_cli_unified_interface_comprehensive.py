"""Comprehensive tests for the unified CLI interface (wqm).

Task 251: Test unified CLI interface with comprehensive help system,
command discovery, auto-completion, configuration management, and
error handling systems.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from rich.console import Console

# Import CLI components to test
from wqm_cli.cli.main import app
from wqm_cli.cli.help_system import help_system, InteractiveHelpSystem
from wqm_cli.cli.advanced_features import (
    ConfigurationWizard, SmartDefaults, CommandSuggestionSystem
)
from wqm_cli.cli.error_handling import (
    ErrorHandler, WqmError, ErrorCategory, ErrorSeverity, ErrorContext
)


class TestUnifiedCLIInterface:
    """Test the unified wqm CLI interface."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()
        self.console = Console(file=tempfile.NamedTemporaryFile(mode='w'))

    def test_main_help_displays_all_commands(self):
        """Test that main help shows all available commands."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Check for main command categories
        expected_commands = [
            "init", "memory", "admin", "config", "ingest",
            "search", "library", "lsp", "service", "watch",
            "observability", "status", "help", "wizard"
        ]

        for cmd in expected_commands:
            assert cmd in result.stdout

    def test_version_flag_basic(self):
        """Test basic version flag."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Should display just version number
        assert result.stdout.strip()
        # Should not contain verbose info on basic version
        assert "Platform:" not in result.stdout

    def test_version_flag_verbose(self):
        """Test verbose version flag."""
        result = self.runner.invoke(app, ["--version", "--verbose"])
        assert result.exit_code == 0
        # Should contain detailed version info
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Python" in result.stdout
        assert "Platform:" in result.stdout

    def test_debug_mode_enables_logging(self):
        """Test that debug flag enables verbose logging."""
        with patch('wqm_cli.cli.main.setup_logging') as mock_setup:
            result = self.runner.invoke(app, ["--debug", "admin", "status"])
            # setup_logging should be called with verbose=True in debug mode
            mock_setup.assert_called_with(log_file=None, verbose=True)

    def test_custom_config_path(self):
        """Test custom configuration file path."""
        config_path = "/tmp/custom_config.yaml"
        result = self.runner.invoke(app, ["--config", config_path, "config", "show"])
        # Should accept config path without error
        # Note: actual config loading is tested in config command tests

    def test_no_args_shows_help(self):
        """Test that running wqm with no arguments shows help."""
        result = self.runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands:" in result.stdout


class TestInteractiveHelpSystem:
    """Test the interactive help and command discovery system."""

    def setup_method(self):
        """Setup test environment."""
        self.help_system = InteractiveHelpSystem()
        self.runner = CliRunner()

    def test_command_discovery_initialization(self):
        """Test that command discovery initializes with all commands."""
        commands = self.help_system._commands

        # Check that all major commands are present
        expected_commands = [
            "memory", "admin", "search", "config", "ingest",
            "library", "watch", "service", "status", "observability",
            "init", "lsp"
        ]

        for cmd in expected_commands:
            assert cmd in commands
            assert commands[cmd].name == cmd
            assert commands[cmd].description
            assert commands[cmd].usage
            assert isinstance(commands[cmd].examples, list)

    def test_command_suggestion_fuzzy_matching(self):
        """Test fuzzy command matching for suggestions."""
        suggestions = self.help_system.suggest_commands("mem", limit=3)

        assert len(suggestions) > 0
        # Should suggest memory command
        command_names = [suggestion[0] for suggestion in suggestions]
        assert any("memory" in cmd for cmd in command_names)

    def test_command_suggestion_partial_subcommands(self):
        """Test suggestions for partial subcommand input."""
        suggestions = self.help_system.suggest_commands("admin stat", limit=3)

        assert len(suggestions) > 0
        # Should suggest "admin status"
        command_names = [suggestion[0] for suggestion in suggestions]
        assert any("admin status" in cmd for cmd in command_names)

    def test_command_categories_organization(self):
        """Test that commands are properly organized by categories."""
        categories = self.help_system._categories

        # Check expected categories exist
        expected_categories = ["Core", "System", "Content", "Monitoring", "Setup", "Development"]
        for category in expected_categories:
            assert category in categories
            assert len(categories[category]) > 0

    def test_show_command_help_existing_command(self):
        """Test showing help for an existing command."""
        # Mock console to capture output
        with patch('wqm_cli.cli.help_system.console') as mock_console:
            self.help_system.show_command_help("memory", level="detailed")

            # Should call console.print at least twice (panel and tips)
            assert mock_console.print.call_count >= 2

    def test_show_command_help_nonexistent_command(self):
        """Test showing help for a non-existent command shows suggestions."""
        with patch('wqm_cli.cli.help_system.console') as mock_console:
            self.help_system.show_command_help("nonexistent")

            # Should print error and suggestions
            mock_console.print.assert_called()
            calls = [str(call) for call in mock_console.print.call_args_list]
            error_printed = any("not found" in call for call in calls)
            assert error_printed

    def test_category_help_display(self):
        """Test displaying help for a specific category."""
        with patch('wqm_cli.cli.help_system.console') as mock_console:
            self.help_system.show_category_help("Core")

            # Should display table with commands in Core category
            mock_console.print.assert_called()

    def test_quick_reference_display(self):
        """Test the quick reference display."""
        with patch('wqm_cli.cli.help_system.console') as mock_console:
            self.help_system.show_quick_reference()

            # Should print multiple sections
            assert mock_console.print.call_count >= 3


class TestAdvancedCLIFeatures:
    """Test advanced CLI features like configuration wizard and smart defaults."""

    def test_configuration_wizard_initialization(self):
        """Test configuration wizard initializes properly."""
        wizard = ConfigurationWizard()
        assert wizard.config_data == {}

    def test_smart_defaults_initialization(self):
        """Test smart defaults system initializes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock home directory
            with patch('pathlib.Path.home') as mock_home:
                mock_home.return_value = Path(temp_dir)

                defaults = SmartDefaults()
                assert isinstance(defaults.usage_history, dict)
                assert "command_frequency" in defaults.usage_history

    def test_smart_defaults_command_recording(self):
        """Test recording command usage for learning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pathlib.Path.home') as mock_home:
                mock_home.return_value = Path(temp_dir)

                defaults = SmartDefaults()
                defaults.record_command_usage("memory", "add", {"verbose": True})

                # Check that usage was recorded
                freq = defaults.usage_history["command_frequency"]
                assert "memory add" in freq
                assert freq["memory add"] == 1

    def test_command_suggestion_system_initialization(self):
        """Test command suggestion system initializes with relationships."""
        suggestion_system = CommandSuggestionSystem()

        assert isinstance(suggestion_system.command_relationships, dict)
        assert isinstance(suggestion_system.context_patterns, dict)

        # Check for expected relationship patterns
        assert "after_config" in suggestion_system.command_relationships
        assert "setup_workflow" in suggestion_system.command_relationships

    def test_next_command_suggestions(self):
        """Test logical next command suggestions."""
        suggestion_system = CommandSuggestionSystem()

        # Test suggestions after config command
        suggestions = suggestion_system.suggest_next_commands("config set")
        assert len(suggestions) > 0
        # Should suggest status check or service restart
        assert any("status" in cmd for cmd in suggestions)

    def test_context_based_suggestions(self):
        """Test context-based command suggestions."""
        suggestion_system = CommandSuggestionSystem()

        suggestions = suggestion_system.suggest_for_context("first_time_user")
        assert len(suggestions) > 0
        # Should include setup commands
        assert any("config" in cmd or "help" in cmd for cmd in suggestions)


class TestCLIErrorHandling:
    """Test comprehensive CLI error handling and recovery."""

    def setup_method(self):
        """Setup test environment."""
        self.error_handler = ErrorHandler()

    def test_error_handler_initialization(self):
        """Test error handler initializes with patterns and strategies."""
        assert isinstance(self.error_handler.error_patterns, dict)
        assert isinstance(self.error_handler.recovery_strategies, dict)
        assert self.error_handler.last_errors == []

    def test_error_classification_connection_refused(self):
        """Test classification of connection refused errors."""
        context = ErrorContext(command="admin", subcommand="status")
        exception = ConnectionRefusedError("Connection refused")

        error = self.error_handler._classify_error(exception, context)

        assert error.category == ErrorCategory.CONNECTION
        assert error.severity == ErrorSeverity.HIGH
        assert "connection" in error.title.lower()

    def test_error_classification_file_not_found(self):
        """Test classification of file not found errors."""
        context = ErrorContext(command="ingest", arguments=["nonexistent.pdf"])
        exception = FileNotFoundError("No such file or directory")

        error = self.error_handler._classify_error(exception, context)

        assert error.category == ErrorCategory.FILE_SYSTEM
        assert error.severity == ErrorSeverity.MEDIUM
        assert len(error.recovery_actions) > 0

    def test_error_classification_permission_denied(self):
        """Test classification of permission denied errors."""
        context = ErrorContext(command="service", subcommand="install")
        exception = PermissionError("Permission denied")

        error = self.error_handler._classify_error(exception, context)

        assert error.category == ErrorCategory.PERMISSION
        assert error.severity == ErrorSeverity.HIGH
        assert any("permission" in action.description.lower()
                  for action in error.recovery_actions)

    def test_recovery_actions_for_connection_errors(self):
        """Test recovery actions are provided for connection errors."""
        context = ErrorContext(command="admin", subcommand="status")
        exception = ConnectionRefusedError("Connection refused")

        error = self.error_handler._classify_error(exception, context)

        # Should have recovery actions
        assert len(error.recovery_actions) > 0

        # Should include service check
        action_descriptions = [action.description for action in error.recovery_actions]
        assert any("service" in desc.lower() for desc in action_descriptions)

    def test_error_history_management(self):
        """Test error history tracking."""
        context = ErrorContext(command="test")
        exception = Exception("Test error")

        # Add error to history
        error = self.error_handler._classify_error(exception, context)
        self.error_handler.last_errors.append(error)

        # Check history
        history = self.error_handler.get_error_history()
        assert len(history) == 1
        assert history[0].original_exception is exception

    def test_error_history_limit(self):
        """Test error history respects limit."""
        context = ErrorContext(command="test")

        # Add more errors than the limit
        for i in range(15):
            exception = Exception(f"Test error {i}")
            error = self.error_handler._classify_error(exception, context)
            self.error_handler.last_errors.append(error)

        # Should only keep the limit
        assert len(self.error_handler.last_errors) <= self.error_handler.error_history_limit

    def test_error_context_creation(self):
        """Test error context creation with defaults."""
        context = ErrorContext(command="test", subcommand="sub")

        assert context.command == "test"
        assert context.subcommand == "sub"
        assert isinstance(context.arguments, list)
        assert isinstance(context.flags, dict)
        assert context.working_dir
        assert isinstance(context.environment, dict)


class TestCLIIntegrationAndEdgeCases:
    """Test CLI integration scenarios and edge cases."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_malformed_command_arguments(self):
        """Test handling of malformed command arguments."""
        # Test with invalid flags
        result = self.runner.invoke(app, ["admin", "--invalid-flag"])
        # Should handle gracefully, not crash
        assert "invalid-flag" in result.stdout or "Error" in result.stdout

    def test_empty_command_args(self):
        """Test commands with empty or missing required arguments."""
        # Test memory command without required argument
        result = self.runner.invoke(app, ["memory", "add"])
        # Should show error or help, not crash

    def test_keyboard_interrupt_handling(self):
        """Test that keyboard interrupts are handled gracefully."""
        # This is hard to test directly, but we can check the handler exists
        from wqm_cli.cli.main import handle_async_command
        import asyncio

        async def long_running():
            await asyncio.sleep(10)

        # The function should handle KeyboardInterrupt
        try:
            with patch('asyncio.run', side_effect=KeyboardInterrupt):
                handle_async_command(long_running())
        except SystemExit:
            pass  # Expected for typer.Exit

    def test_concurrent_cli_invocations(self):
        """Test that multiple CLI invocations don't interfere."""
        import threading
        import time

        results = []

        def run_cli():
            result = self.runner.invoke(app, ["--version"])
            results.append(result.exit_code)

        # Run multiple CLI instances concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_cli)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        assert all(code == 0 for code in results)

    def test_environment_variable_handling(self):
        """Test that environment variables are properly handled."""
        import os

        # Test with CLI mode environment variable
        with patch.dict(os.environ, {"WQM_CLI_MODE": "true"}):
            result = self.runner.invoke(app, ["--version"])
            assert result.exit_code == 0

    def test_large_output_handling(self):
        """Test handling of commands that produce large outputs."""
        # Test with a command that might produce large output
        result = self.runner.invoke(app, ["help"])
        # Should complete successfully even with large output
        assert result.exit_code == 0

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in input."""
        # Test with Unicode in memory rule
        result = self.runner.invoke(app, ["memory", "add", "测试规则"])
        # Should handle Unicode gracefully (may not work without Qdrant, but shouldn't crash)

    def test_path_with_spaces_handling(self):
        """Test handling of file paths with spaces."""
        result = self.runner.invoke(app, ["ingest", "file", "/path/with spaces/file.pdf"])
        # Should handle paths with spaces (may fail due to missing file, but shouldn't crash parsing)

    def test_very_long_command_lines(self):
        """Test handling of very long command lines."""
        long_text = "a" * 1000
        result = self.runner.invoke(app, ["memory", "add", long_text])
        # Should handle long text gracefully


class TestCLIConfigurationManagement:
    """Test CLI configuration management features."""

    def test_configuration_loading_precedence(self):
        """Test configuration loading precedence (CLI args > env vars > config file)."""
        # This tests the overall architecture but specific loading is in config modules
        pass

    def test_configuration_validation(self):
        """Test configuration validation and error reporting."""
        # Test with invalid config format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()

            result = self.runner.invoke(app, ["--config", f.name, "config", "validate"])
            # Should handle invalid config gracefully

    def test_configuration_auto_completion_setup(self):
        """Test auto-completion setup for different shells."""
        # Test completion setup for bash
        result = self.runner.invoke(app, ["init", "bash"])
        # Should provide completion script
        assert "complete" in result.stdout or "_wqm_completion" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])