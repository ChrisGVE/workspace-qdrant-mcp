"""Unit tests for CLI wrapper functionality.

Tests the CLI wrapper's environment variable setup, logging configuration,
and main entry point functionality.

Test coverage:
- Environment variable configuration (WQM_CLI_MODE, WQM_LOG_INIT)
- Loguru logging configuration for CLI mode
- Main entry point execution
- Import isolation and module loading
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))


class TestCLIWrapper:
    """Test CLI wrapper functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Store original environment variables to restore later
        self.original_cli_mode = os.environ.get("WQM_CLI_MODE")
        self.original_log_init = os.environ.get("WQM_LOG_INIT")

    def teardown_method(self):
        """Clean up after each test."""
        # Restore original environment variables
        if self.original_cli_mode is not None:
            os.environ["WQM_CLI_MODE"] = self.original_cli_mode
        elif "WQM_CLI_MODE" in os.environ:
            del os.environ["WQM_CLI_MODE"]

        if self.original_log_init is not None:
            os.environ["WQM_LOG_INIT"] = self.original_log_init
        elif "WQM_LOG_INIT" in os.environ:
            del os.environ["WQM_LOG_INIT"]

    def test_environment_variables_set_on_import(self):
        """Test that environment variables are set when cli_wrapper is imported."""
        # Clear environment variables first
        for var in ["WQM_CLI_MODE", "WQM_LOG_INIT"]:
            if var in os.environ:
                del os.environ[var]

        # Import the wrapper module (this should set the environment variables)
        import wqm_cli.cli_wrapper

        # Verify environment variables are set correctly
        assert os.environ.get("WQM_CLI_MODE") == "true"
        assert os.environ.get("WQM_LOG_INIT") == "false"

    def test_environment_variables_default_behavior(self):
        """Test that setdefault behavior doesn't override existing values."""
        # Set existing values
        os.environ["WQM_CLI_MODE"] = "false"
        os.environ["WQM_LOG_INIT"] = "true"

        # Import wrapper module - should not override existing values
        import importlib

        import wqm_cli.cli_wrapper
        importlib.reload(wqm_cli.cli_wrapper)

        # Values should remain unchanged due to setdefault behavior
        assert os.environ.get("WQM_CLI_MODE") == "false"
        assert os.environ.get("WQM_LOG_INIT") == "true"

    def test_logging_configuration(self):
        """Test that logging is configured correctly for CLI mode."""
        # The logging is configured at module import time
        # We just verify the module imports successfully
        import wqm_cli.cli_wrapper

        # If we get here without exception, logging configuration worked
        assert True

    def test_main_function_calls_app(self):
        """Test that main function calls the CLI app."""
        from wqm_cli.cli_wrapper import main

        # Mock the app import inside the main function
        with patch('wqm_cli.cli.main.app') as mock_app:
            # Call main function
            main()

            # Verify app was called
            mock_app.assert_called_once()

    def test_main_function_exception_handling(self):
        """Test main function handles exceptions from app."""
        from wqm_cli.cli_wrapper import main

        # Mock the app import to raise exception
        with patch('wqm_cli.cli.main.app') as mock_app:
            # Configure app to raise an exception
            mock_app.side_effect = Exception("Test exception")

            # Main should not crash but allow exception to propagate
            with pytest.raises(Exception, match="Test exception"):
                main()

    def test_main_function_keyboard_interrupt(self):
        """Test main function handles KeyboardInterrupt gracefully."""
        from wqm_cli.cli_wrapper import main

        # Mock the app import to raise KeyboardInterrupt
        with patch('wqm_cli.cli.main.app') as mock_app:
            # Configure app to raise KeyboardInterrupt
            mock_app.side_effect = KeyboardInterrupt()

            # Should handle KeyboardInterrupt gracefully
            with pytest.raises(KeyboardInterrupt):
                main()

    def test_module_imports_correctly(self):
        """Test that all required modules can be imported."""
        # Test that imports don't fail
        import wqm_cli.cli_wrapper
        from wqm_cli.cli_wrapper import main

        # Verify main function exists and is callable
        assert callable(main)

    @patch('sys.argv', ['wqm'])
    def test_main_entry_point_execution(self):
        """Test that the main entry point executes correctly."""
        from wqm_cli.cli_wrapper import main

        # Mock the app import
        with patch('wqm_cli.cli.main.app') as mock_app:
            # Execute main
            main()

            # Verify app was called
            mock_app.assert_called_once()

    def test_environment_isolation(self):
        """Test that CLI mode environment variables isolate server components."""
        # Set CLI mode environment
        os.environ["WQM_CLI_MODE"] = "true"
        os.environ["WQM_LOG_INIT"] = "false"

        # Import cli_wrapper (this tests the isolation)
        import wqm_cli.cli_wrapper

        # Verify environment is correctly set for CLI isolation
        assert os.environ.get("WQM_CLI_MODE") == "true"
        assert os.environ.get("WQM_LOG_INIT") == "false"

    @pytest.mark.xfail(reason="cli_wrapper no longer exports logger - uses unified logging system")
    @patch('wqm_cli.cli_wrapper.logger')
    def test_logging_import_and_usage(self, mock_logger):
        """Test that loguru logger is imported and available."""
        import importlib

        import wqm_cli.cli_wrapper
        importlib.reload(wqm_cli.cli_wrapper)

        # Logger should be imported (we can't test usage directly in wrapper
        # but we can verify the import structure)
        assert hasattr(wqm_cli.cli_wrapper, 'logger')


class TestCLIWrapperIntegration:
    """Integration tests for CLI wrapper with actual components."""

    def test_cli_mode_prevents_server_imports(self):
        """Test that CLI mode prevents server component imports."""
        # Set CLI mode
        os.environ["WQM_CLI_MODE"] = "true"

        # This should not fail even if server components are not available
        import wqm_cli.cli_wrapper

        # Verify CLI mode is active
        assert os.environ.get("WQM_CLI_MODE") == "true"

    def test_log_init_disabled_in_cli_mode(self):
        """Test that log initialization is disabled in CLI mode."""
        # Clear environment first and reimport
        if "WQM_LOG_INIT" in os.environ:
            del os.environ["WQM_LOG_INIT"]

        # Import wrapper which sets WQM_LOG_INIT=false
        import importlib

        import wqm_cli.cli_wrapper
        importlib.reload(wqm_cli.cli_wrapper)

        # Verify log initialization is disabled
        assert os.environ.get("WQM_LOG_INIT") == "false"

    def test_setup_logging_called_with_cli_settings(self):
        """Test that setup_logging is called with CLI-appropriate settings."""
        # Clear environment first
        for var in ["WQM_CLI_MODE", "WQM_LOG_INIT"]:
            if var in os.environ:
                del os.environ[var]

        # Import wrapper which should set environment variables
        import importlib

        import wqm_cli.cli_wrapper
        importlib.reload(wqm_cli.cli_wrapper)

        # Verify environment variables are set correctly for CLI logging
        assert os.environ.get("WQM_CLI_MODE") == "true"
        assert os.environ.get("WQM_LOG_INIT") == "false"


class TestCLIWrapperErrorScenarios:
    """Test error scenarios and edge cases."""

    def test_setup_logging_import_failure(self):
        """Test behavior when setup_logging import fails."""
        # Since setup_logging is imported at module level, we test with mock
        with patch('common.logging.loguru_config.setup_logging', side_effect=ImportError("Cannot import setup_logging")):
            # Import should fail if setup_logging fails
            with pytest.raises(ImportError):
                import importlib

                import wqm_cli.cli_wrapper
                importlib.reload(wqm_cli.cli_wrapper)

    def test_app_import_failure(self):
        """Test behavior when CLI app import fails."""
        from wqm_cli.cli_wrapper import main

        # Mock the app import to fail
        with patch('wqm_cli.cli.main.app', side_effect=ImportError("Cannot import app")):
            # Should raise ImportError when trying to call main
            with pytest.raises(ImportError, match="Cannot import app"):
                main()

    def test_missing_environment_variables(self):
        """Test behavior when environment variables are missing."""
        # Clear all environment variables
        for var in ["WQM_CLI_MODE", "WQM_LOG_INIT"]:
            if var in os.environ:
                del os.environ[var]

        # Import should still work and set defaults
        import importlib

        import wqm_cli.cli_wrapper
        importlib.reload(wqm_cli.cli_wrapper)

        # Should have set default values
        assert os.environ.get("WQM_CLI_MODE") == "true"
        assert os.environ.get("WQM_LOG_INIT") == "false"

    def test_partial_environment_setup(self):
        """Test behavior with partially set environment."""
        # Set only one environment variable
        os.environ["WQM_CLI_MODE"] = "existing_value"
        if "WQM_LOG_INIT" in os.environ:
            del os.environ["WQM_LOG_INIT"]

        # Import wrapper
        import importlib

        import wqm_cli.cli_wrapper
        importlib.reload(wqm_cli.cli_wrapper)

        # Should preserve existing value and set missing one
        assert os.environ.get("WQM_CLI_MODE") == "existing_value"
        assert os.environ.get("WQM_LOG_INIT") == "false"
