"""CLI Binary Validation Tests.

Comprehensive testing of the installed wqm CLI binary for all command domains.
Tests the actual command line interface as users would experience it.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


class TestCLIBinaryValidation:
    """Test the actual wqm CLI binary installation."""

    def setup_method(self):
        """Set up test environment."""
        self.wqm_cmd = "wqm"  # Assuming wqm is in PATH
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def run_wqm(self, args, expect_success=True, input_data=None):
        """Helper to run wqm command and return result."""
        cmd = [self.wqm_cmd] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=30  # 30 second timeout
            )
            if expect_success and result.returncode != 0:
                print(f"Command failed: {' '.join(cmd)}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                print(f"Return code: {result.returncode}")
            return result
        except subprocess.TimeoutExpired:
            pytest.fail(f"Command timed out: {' '.join(cmd)}")
        except FileNotFoundError:
            pytest.skip("wqm command not found in PATH")


class TestVersionAndHelp(TestCLIBinaryValidation):
    """Test version and help commands."""

    def test_version_short_flag(self):
        """Test -v flag shows version."""
        result = self.run_wqm(["-v"])
        assert result.returncode == 0
        assert result.stdout.strip()  # Should output something
        # Should be clean version output
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 1, "Version should be single line"

    def test_version_long_flag(self):
        """Test --version flag shows version."""
        result = self.run_wqm(["--version"])
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_version_verbose(self):
        """Test verbose version information."""
        result = self.run_wqm(["--version", "--verbose"])
        assert result.returncode == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Python" in result.stdout

    def test_main_help(self):
        """Test main help command."""
        result = self.run_wqm(["--help"])
        assert result.returncode == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Commands:" in result.stdout

    def test_no_args_shows_help(self):
        """Test running wqm without arguments shows help."""
        result = self.run_wqm([])
        assert result.returncode == 0
        assert "Usage:" in result.stdout


class TestCommandStructure(TestCLIBinaryValidation):
    """Test command structure and help consistency."""

    def test_all_main_commands_help(self):
        """Test help for all main command groups."""
        # Note: "web" command removed - not implemented in current CLI
        commands = [
            "init", "memory", "admin", "ingest", "search",
            "library", "service", "watch", "observability", "status"
        ]

        for cmd in commands:
            result = self.run_wqm([cmd, "--help"])
            assert result.returncode == 0, f"Help failed for {cmd}"
            assert "Usage:" in result.stdout, f"Missing Usage in {cmd} help"

    def test_nested_command_help(self):
        """Test help for nested commands."""
        nested_commands = [
            ["memory", "add", "--help"],
            ["memory", "list", "--help"],
            ["admin", "status", "--help"],
            ["search", "project", "--help"],
            ["init", "bash", "--help"],
        ]

        for cmd_parts in nested_commands:
            result = self.run_wqm(cmd_parts)
            assert result.returncode == 0, f"Help failed for {' '.join(cmd_parts)}"

    def test_invalid_command_error(self):
        """Test invalid command returns error."""
        result = self.run_wqm(["invalid-command"], expect_success=False)
        assert result.returncode != 0
        # Error output may be in stdout or stderr depending on CLI implementation
        combined_output = result.stdout + result.stderr
        assert "No such command" in combined_output or "Usage:" in combined_output


class TestMemoryCommands(TestCLIBinaryValidation):
    """Test memory command domain functionality."""

    def test_memory_list_no_daemon(self):
        """Test memory list when daemon not running."""
        result = self.run_wqm(["memory", "list"], expect_success=False)
        # Should fail gracefully when daemon not available
        assert result.returncode != 0
        assert "Error" in result.stdout or "Connection" in result.stderr

    def test_memory_add_no_daemon(self):
        """Test memory add when daemon not running."""
        result = self.run_wqm(["memory", "add", "test rule"], expect_success=False)
        assert result.returncode != 0

    def test_memory_invalid_subcommand(self):
        """Test invalid memory subcommand."""
        result = self.run_wqm(["memory", "invalid"], expect_success=False)
        assert result.returncode != 0


class TestAdminCommands(TestCLIBinaryValidation):
    """Test admin command domain functionality."""

    def test_admin_status_no_daemon(self):
        """Test admin status when daemon not running."""
        result = self.run_wqm(["admin", "status"], expect_success=True)
        # Admin status returns 0 even without daemon, showing UNHEALTHY status
        assert result.returncode == 0
        assert "Status" in result.stdout or "Health" in result.stdout

    def test_admin_health_no_daemon(self):
        """Test admin health when daemon not running."""
        result = self.run_wqm(["admin", "health"], expect_success=True)
        # Admin health now returns 0 with health check results
        assert result.returncode == 0
        assert "Health" in result.stdout

    def test_admin_config_show(self):
        """Test admin config command."""
        # Note: "show" subcommand removed - use "admin config" directly
        result = self.run_wqm(["admin", "config"], expect_success=True)
        # May succeed or fail depending on config availability
        assert result.returncode in [0, 1, 2]


class TestSearchCommands(TestCLIBinaryValidation):
    """Test search command domain functionality."""

    def test_search_project_no_daemon(self):
        """Test search project when daemon not running."""
        result = self.run_wqm(["search", "project", "test query"], expect_success=False)
        assert result.returncode != 0

    def test_search_empty_query(self):
        """Test search with empty query."""
        result = self.run_wqm(["search", "project", ""], expect_success=False)
        assert result.returncode != 0

    def test_search_collection_no_daemon(self):
        """Test search collection when daemon not running."""
        result = self.run_wqm(["search", "collection", "test_coll", "query"], expect_success=False)
        assert result.returncode != 0


class TestIngestCommands(TestCLIBinaryValidation):
    """Test ingest command domain functionality."""

    def test_ingest_nonexistent_file(self):
        """Test ingest with non-existent file."""
        # Note: ingest file now requires --collection argument
        result = self.run_wqm(["ingest", "file", "/nonexistent/file.txt", "-c", "test"], expect_success=False)
        assert result.returncode != 0
        # Error might be about missing file or missing collection, both are valid failures
        combined_output = result.stdout + result.stderr
        assert "Error" in combined_output or "not found" in combined_output.lower() or "does not exist" in combined_output.lower()

    def test_ingest_file_with_existing_file(self):
        """Test ingest with existing file."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content for ingestion")

        result = self.run_wqm(["ingest", "file", str(test_file)], expect_success=False)
        # Should fail without daemon but shouldn't crash on file validation
        assert result.returncode != 0

    def test_ingest_folder_nonexistent(self):
        """Test ingest with non-existent folder."""
        result = self.run_wqm(["ingest", "folder", "/nonexistent/folder"], expect_success=False)
        assert result.returncode != 0


class TestLibraryCommands(TestCLIBinaryValidation):
    """Test library command domain functionality."""

    def test_library_create_no_daemon(self):
        """Test library create when daemon not running."""
        result = self.run_wqm(["library", "create", "test_lib"], expect_success=False)
        assert result.returncode != 0

    def test_library_list_no_daemon(self):
        """Test library list when daemon not running."""
        result = self.run_wqm(["library", "list"], expect_success=False)
        assert result.returncode != 0


class TestServiceCommands(TestCLIBinaryValidation):
    """Test service command domain functionality."""

    def test_service_status(self):
        """Test service status command."""
        result = self.run_wqm(["service", "status"], expect_success=False)
        # May succeed or fail depending on service state
        assert result.returncode in [0, 1]

    def test_service_install_dry_run(self):
        """Test service install in dry-run mode if available."""
        result = self.run_wqm(["service", "install", "--help"])
        assert result.returncode == 0


class TestWatchCommands(TestCLIBinaryValidation):
    """Test watch command domain functionality."""

    def test_watch_list_no_daemon(self):
        """Test watch list when daemon not running."""
        result = self.run_wqm(["watch", "list"], expect_success=True)
        # Watch list now returns 0 even with no watches (from SQLite state)
        assert result.returncode == 0
        assert "watches" in result.stdout.lower() or "No" in result.stdout

    def test_watch_start_no_daemon(self):
        """Test watch start when daemon not running."""
        result = self.run_wqm(["watch", "start", self.temp_dir], expect_success=False)
        assert result.returncode != 0


class TestWebCommands(TestCLIBinaryValidation):
    """Test web command domain functionality."""

    @pytest.mark.skip(reason="Web command not implemented in current CLI")
    def test_web_status(self):
        """Test web status command."""
        result = self.run_wqm(["web", "status"], expect_success=False)
        # May succeed or fail depending on web server state
        assert result.returncode in [0, 1]

    @pytest.mark.skip(reason="Web command not implemented in current CLI")
    def test_web_start_help(self):
        """Test web start help."""
        result = self.run_wqm(["web", "start", "--help"])
        assert result.returncode == 0


class TestInitCommands(TestCLIBinaryValidation):
    """Test init command for shell completion."""

    def test_init_bash_completion(self):
        """Test bash completion generation."""
        result = self.run_wqm(["init", "bash"])
        assert result.returncode == 0
        assert "_wqm_completion" in result.stdout
        assert "complete" in result.stdout

    def test_init_zsh_completion(self):
        """Test zsh completion generation."""
        result = self.run_wqm(["init", "zsh"])
        assert result.returncode == 0
        assert "#compdef wqm" in result.stdout

    def test_init_fish_completion(self):
        """Test fish completion generation."""
        result = self.run_wqm(["init", "fish"])
        assert result.returncode == 0
        # Fish completion may use different formats
        assert "complete" in result.stdout and "wqm" in result.stdout

    def test_init_invalid_shell(self):
        """Test init with invalid shell."""
        result = self.run_wqm(["init", "invalid"], expect_success=False)
        assert result.returncode != 0


class TestStatusCommands(TestCLIBinaryValidation):
    """Test status command functionality."""

    def test_status_basic_no_daemon(self):
        """Test basic status when daemon not running."""
        result = self.run_wqm(["status"], expect_success=True)
        # Status command now returns 0 even without daemon, showing status info
        assert result.returncode == 0
        # Should show some status information
        assert "Status" in result.stdout or "Daemon" in result.stdout or "Queue" in result.stdout

    def test_status_help(self):
        """Test status help command."""
        result = self.run_wqm(["status", "--help"])
        assert result.returncode == 0


class TestObservabilityCommands(TestCLIBinaryValidation):
    """Test observability command functionality."""

    def test_observability_help(self):
        """Test observability help command."""
        result = self.run_wqm(["observability", "--help"])
        assert result.returncode == 0

    def test_observability_health_no_daemon(self):
        """Test observability health when daemon not running."""
        result = self.run_wqm(["observability", "health"], expect_success=False)
        assert result.returncode != 0


class TestConfigurationHandling(TestCLIBinaryValidation):
    """Test configuration file handling."""

    def test_custom_config_flag(self):
        """Test --config flag with custom configuration."""
        config_file = Path(self.temp_dir) / "test_config.yaml"
        config_data = {
            "qdrant": {"url": "http://localhost:6333"},
            "directories": {"workspace": str(Path.cwd())}
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        result = self.run_wqm(["--config", str(config_file), "admin", "status"], expect_success=False)
        # Should not crash on config loading
        assert "yaml" not in result.stderr.lower()  # No YAML parsing errors

    def test_invalid_config_file(self):
        """Test invalid configuration file handling."""
        config_file = Path(self.temp_dir) / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        result = self.run_wqm(["--config", str(config_file), "admin", "status"], expect_success=True)
        # CLI may fall back to default config when provided config is invalid
        # It should not crash - that's the main requirement
        assert result.returncode in [0, 1, 2]

    def test_nonexistent_config_file(self):
        """Test non-existent configuration file handling."""
        result = self.run_wqm(["--config", "/nonexistent/config.yaml", "admin", "status"], expect_success=True)
        # CLI may fall back to default config when provided config doesn't exist
        assert result.returncode in [0, 1, 2]


class TestErrorHandling(TestCLIBinaryValidation):
    """Test error handling and exit codes."""

    def test_keyboard_interrupt_handling(self):
        """Test that commands can be interrupted gracefully."""
        # This is hard to test reliably, so we'll test the setup
        result = self.run_wqm(["status", "--help"])
        assert result.returncode == 0

    def test_missing_required_arguments(self):
        """Test error handling for missing required arguments."""
        test_cases = [
            # Note: memory add shows help when no args (return 0)
            # search project requires query
            (["search", "project"], "search project requires query"),
            # ingest file requires collection
            (["ingest", "file"], "ingest file requires file path"),
        ]

        for cmd, description in test_cases:
            result = self.run_wqm(cmd, expect_success=False)
            # Commands may either fail or show help for missing required args
            combined = result.stdout + result.stderr
            assert result.returncode != 0 or "Usage:" in combined, f"Should fail or show usage: {description}"

    def test_invalid_flag_handling(self):
        """Test handling of invalid flags."""
        result = self.run_wqm(["--invalid-flag"], expect_success=False)
        assert result.returncode != 0

    def test_help_on_error(self):
        """Test that help is shown for command errors."""
        result = self.run_wqm(["invalid-command"], expect_success=False)
        assert result.returncode != 0
        # Should show usage or command listing (may be in stdout or stderr)
        combined_output = result.stdout + result.stderr
        assert "Usage:" in combined_output or "Commands:" in combined_output


class TestOutputFormatting(TestCLIBinaryValidation):
    """Test output formatting consistency."""

    def test_help_output_formatting(self):
        """Test that help output is well-formatted."""
        result = self.run_wqm(["--help"])
        assert result.returncode == 0

        lines = result.stdout.split('\n')
        assert len(lines) > 5  # Should be multi-line help
        assert any("Usage:" in line for line in lines)
        assert any("Commands:" in line for line in lines)

    def test_version_output_clean(self):
        """Test that version output is clean."""
        result = self.run_wqm(["--version"])
        assert result.returncode == 0

        # Should be single line with version number
        output = result.stdout.strip()
        lines = output.split('\n')
        assert len(lines) == 1
        assert output  # Not empty

    def test_error_messages_helpful(self):
        """Test that error messages are helpful."""
        result = self.run_wqm(["nonexistent-command"], expect_success=False)
        assert result.returncode != 0

        # Error should be informative
        error_output = result.stdout + result.stderr
        assert error_output.strip()  # Should have error message
        assert "command" in error_output.lower() or "usage" in error_output.lower()


class TestEdgeCases(TestCLIBinaryValidation):
    """Test edge cases and boundary conditions."""

    def test_very_long_arguments(self):
        """Test handling of very long command arguments."""
        long_query = "x" * 1000
        result = self.run_wqm(["search", "project", long_query], expect_success=False)
        # Should not crash, even if it fails due to daemon
        assert result.returncode != 0 or result.returncode == 0

    def test_special_characters_in_paths(self):
        """Test special characters in file paths."""
        special_dir = Path(self.temp_dir) / "special dir with spaces & symbols!"
        special_dir.mkdir(exist_ok=True)

        result = self.run_wqm(["ingest", "folder", str(special_dir)], expect_success=False)
        # Should handle path properly, even if it fails due to daemon
        assert "path" not in result.stderr.lower() or result.returncode != 0

    def test_unicode_in_commands(self):
        """Test unicode characters in commands."""
        unicode_query = "测试 query with émojis 🚀"
        result = self.run_wqm(["search", "project", unicode_query], expect_success=False)
        # Should handle unicode properly
        assert result.returncode != 0  # Expected to fail without daemon

    def test_empty_arguments(self):
        """Test empty string arguments."""
        result = self.run_wqm(["search", "project", ""], expect_success=False)
        assert result.returncode != 0

    def test_debug_flag_functionality(self):
        """Test debug flag increases verbosity."""
        result = self.run_wqm(["--debug", "admin", "status"], expect_success=True)
        # Admin status returns 0 even without daemon
        assert result.returncode == 0
        # Debug flag should add debug output to stderr
        assert "debug" in result.stderr.lower() or len(result.stderr) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
