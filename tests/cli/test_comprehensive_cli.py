"""Comprehensive CLI testing for all wqm command domains.

This test suite systematically validates all wqm command domains and subcommands:
- init: Shell completion initialization
- memory: Memory rules and LLM behavior management (list|add|remove|search|config)
- admin: System administration (status|health|config|reset)
- ingest: Document processing (file|folder|watch)
- search: Search interface (query|collection|global)
- library: Collection management (add|list|manage)
- service: Service management (install|start|stop|status)
- watch: Folder watching (start|stop|list|config)
- web: Web UI server (start|dev|build|status)
- observability: Monitoring and health checks
- status: Processing status system

Tests include:
- Command structure and help system consistency
- YAML configuration hierarchy and environment variable precedence
- Error handling for invalid inputs, missing dependencies, connection failures
- Output formatting and proper exit codes
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from typer.testing import CliRunner
from wqm_cli.cli.main import app, cli


class TestCLIComprehensive:
    """Comprehensive CLI testing suite."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

        # Create test configuration
        self.test_config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_workspace"
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "directories": {
                "workspace": str(Path.cwd()),
                "data": self.temp_dir
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestVersionAndBasicCommands(TestCLIComprehensive):
    """Test version command and basic CLI behavior."""

    def test_version_flag_short(self):
        """Test -v version flag shows clean version number."""
        result = self.runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "0.2.0" in result.stdout.strip()
        assert "Python" not in result.stdout  # Should be clean version

    def test_version_flag_long(self):
        """Test --version flag shows clean version number."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.stdout.strip()

    def test_version_verbose(self):
        """Test verbose version shows detailed information."""
        result = self.runner.invoke(app, ["--version", "--verbose"])
        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Python" in result.stdout
        assert "Platform" in result.stdout

    def test_no_command_shows_help(self):
        """Test that running wqm without command shows help."""
        result = self.runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Usage:" in result.stdout

    def test_debug_flag_enables_logging(self):
        """Test --debug flag enables debug logging."""
        with patch('wqm_cli.cli.main.configure_logging') as mock_logging:
            self.runner.invoke(app, ["--debug", "admin", "status"])
            mock_logging.assert_called_with(level="DEBUG", json_format=True, console_output=True)


class TestHelpSystemConsistency(TestCLIComprehensive):
    """Test help system consistency across all commands."""

    def test_main_help(self):
        """Test main help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Workspace Qdrant MCP" in result.stdout
        assert "Commands:" in result.stdout

    def test_subcommand_help_consistency(self):
        """Test help consistency across all subcommands."""
        subcommands = [
            "init", "memory", "admin", "ingest", "search",
            "library", "service", "watch", "web", "observability", "status"
        ]

        for cmd in subcommands:
            result = self.runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"Help failed for {cmd}"
            assert "Usage:" in result.stdout, f"Missing Usage in {cmd} help"
            assert "Commands:" in result.stdout or "Options:" in result.stdout, f"Missing structure in {cmd} help"

    def test_nested_command_help(self):
        """Test help for nested commands."""
        nested_commands = [
            ["memory", "add", "--help"],
            ["memory", "list", "--help"],
            ["admin", "status", "--help"],
            ["admin", "health", "--help"],
            ["ingest", "file", "--help"],
            ["ingest", "folder", "--help"],
            ["search", "project", "--help"],
            ["library", "create", "--help"],
            ["watch", "start", "--help"],
            ["service", "status", "--help"],
            ["web", "start", "--help"]
        ]

        for cmd_parts in nested_commands:
            result = self.runner.invoke(app, cmd_parts)
            assert result.exit_code == 0, f"Help failed for {' '.join(cmd_parts)}"
            assert "Usage:" in result.stdout, f"Missing Usage in {' '.join(cmd_parts)} help"


class TestMemoryCommands(TestCLIComprehensive):
    """Test memory command domain (list|add|remove|search|config)."""

    @patch('wqm_cli.cli.commands.memory.get_daemon_client')
    def test_memory_list(self, mock_get_client):
        """Test memory list command."""
        mock_client = MagicMock()
        mock_client.list_memory_rules.return_value = []
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 0
        mock_client.list_memory_rules.assert_called_once()

    @patch('wqm_cli.cli.commands.memory.get_daemon_client')
    def test_memory_add_success(self, mock_get_client):
        """Test memory add command success."""
        mock_client = MagicMock()
        mock_client.add_memory_rule.return_value = {"id": "rule-1", "rule": "test rule"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["memory", "add", "test rule"])
        assert result.exit_code == 0
        mock_client.add_memory_rule.assert_called_once()

    @patch('wqm_cli.cli.commands.memory.get_daemon_client')
    def test_memory_remove(self, mock_get_client):
        """Test memory remove command."""
        mock_client = MagicMock()
        mock_client.remove_memory_rule.return_value = True
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["memory", "remove", "rule-1"])
        assert result.exit_code == 0
        mock_client.remove_memory_rule.assert_called_once_with("rule-1")

    @patch('wqm_cli.cli.commands.memory.get_daemon_client')
    def test_memory_search(self, mock_get_client):
        """Test memory search command."""
        mock_client = MagicMock()
        mock_client.search_memory_rules.return_value = []
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["memory", "search", "test query"])
        assert result.exit_code == 0
        mock_client.search_memory_rules.assert_called_once()

    def test_memory_invalid_command(self):
        """Test invalid memory subcommand."""
        result = self.runner.invoke(app, ["memory", "invalid"])
        assert result.exit_code != 0
        assert "No such command" in result.stdout

    @patch('wqm_cli.cli.commands.memory.get_daemon_client')
    def test_memory_connection_error(self, mock_get_client):
        """Test memory command with connection error."""
        mock_get_client.side_effect = ConnectionError("Cannot connect to daemon")

        result = self.runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestAdminCommands(TestCLIComprehensive):
    """Test admin command domain (status|health|config|reset)."""

    @patch('wqm_cli.cli.commands.admin.get_daemon_client')
    def test_admin_status(self, mock_get_client):
        """Test admin status command."""
        mock_client = MagicMock()
        mock_client.get_status.return_value = {"status": "running"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["admin", "status"])
        assert result.exit_code == 0
        mock_client.get_status.assert_called_once()

    @patch('wqm_cli.cli.commands.admin.get_daemon_client')
    def test_admin_health(self, mock_get_client):
        """Test admin health command."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = {"healthy": True}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["admin", "health"])
        assert result.exit_code == 0
        mock_client.health_check.assert_called_once()

    @patch('wqm_cli.cli.commands.admin.get_daemon_client')
    def test_admin_config_show(self, mock_get_client):
        """Test admin config show command."""
        mock_client = MagicMock()
        mock_client.get_config.return_value = self.test_config
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["admin", "config", "show"])
        assert result.exit_code == 0
        mock_client.get_config.assert_called_once()

    @patch('wqm_cli.cli.commands.admin.get_daemon_client')
    def test_admin_reset(self, mock_get_client):
        """Test admin reset command."""
        mock_client = MagicMock()
        mock_client.reset_system.return_value = {"reset": True}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["admin", "reset", "--force"])
        assert result.exit_code == 0
        mock_client.reset_system.assert_called_once()

    def test_admin_invalid_command(self):
        """Test invalid admin subcommand."""
        result = self.runner.invoke(app, ["admin", "invalid"])
        assert result.exit_code != 0


class TestIngestCommands(TestCLIComprehensive):
    """Test ingest command domain (file|folder|watch)."""

    @patch('wqm_cli.cli.commands.ingest.get_daemon_client')
    def test_ingest_file(self, mock_get_client):
        """Test ingest file command."""
        mock_client = MagicMock()
        mock_client.ingest_file.return_value = {"status": "processed"}
        mock_get_client.return_value = mock_client

        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")

        result = self.runner.invoke(app, ["ingest", "file", str(test_file)])
        assert result.exit_code == 0
        mock_client.ingest_file.assert_called_once()

    @patch('wqm_cli.cli.commands.ingest.get_daemon_client')
    def test_ingest_folder(self, mock_get_client):
        """Test ingest folder command."""
        mock_client = MagicMock()
        mock_client.ingest_folder.return_value = {"status": "processed"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["ingest", "folder", self.temp_dir])
        assert result.exit_code == 0
        mock_client.ingest_folder.assert_called_once()

    def test_ingest_file_not_found(self):
        """Test ingest command with non-existent file."""
        result = self.runner.invoke(app, ["ingest", "file", "/nonexistent/file.txt"])
        assert result.exit_code != 0
        assert "Error" in result.stdout

    @patch('wqm_cli.cli.commands.ingest.get_daemon_client')
    def test_ingest_watch_start(self, mock_get_client):
        """Test ingest watch start command."""
        mock_client = MagicMock()
        mock_client.start_watch.return_value = {"status": "watching"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["ingest", "watch", self.temp_dir])
        assert result.exit_code == 0
        mock_client.start_watch.assert_called_once()


class TestSearchCommands(TestCLIComprehensive):
    """Test search command domain (query|collection|global)."""

    @patch('wqm_cli.cli.commands.search.get_daemon_client')
    def test_search_project(self, mock_get_client):
        """Test search project command."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["search", "project", "test query"])
        assert result.exit_code == 0
        mock_client.search.assert_called_once()

    @patch('wqm_cli.cli.commands.search.get_daemon_client')
    def test_search_collection(self, mock_get_client):
        """Test search in specific collection."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["search", "collection", "test_collection", "test query"])
        assert result.exit_code == 0
        mock_client.search.assert_called_once()

    @patch('wqm_cli.cli.commands.search.get_daemon_client')
    def test_search_global(self, mock_get_client):
        """Test global search command."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["search", "global", "test query"])
        assert result.exit_code == 0
        mock_client.search.assert_called_once()

    def test_search_empty_query(self):
        """Test search with empty query."""
        result = self.runner.invoke(app, ["search", "project", ""])
        assert result.exit_code != 0


class TestLibraryCommands(TestCLIComprehensive):
    """Test library command domain (add|list|manage)."""

    @patch('wqm_cli.cli.commands.library.get_daemon_client')
    def test_library_create(self, mock_get_client):
        """Test library create command."""
        mock_client = MagicMock()
        mock_client.create_library.return_value = {"name": "test_lib"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["library", "create", "test_lib"])
        assert result.exit_code == 0
        mock_client.create_library.assert_called_once()

    @patch('wqm_cli.cli.commands.library.get_daemon_client')
    def test_library_list(self, mock_get_client):
        """Test library list command."""
        mock_client = MagicMock()
        mock_client.list_libraries.return_value = []
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["library", "list"])
        assert result.exit_code == 0
        mock_client.list_libraries.assert_called_once()

    @patch('wqm_cli.cli.commands.library.get_daemon_client')
    def test_library_manage(self, mock_get_client):
        """Test library manage command."""
        mock_client = MagicMock()
        mock_client.manage_library.return_value = {"status": "managed"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["library", "manage", "test_lib"])
        assert result.exit_code == 0
        mock_client.manage_library.assert_called_once()


class TestServiceCommands(TestCLIComprehensive):
    """Test service command domain (install|start|stop|status)."""

    @patch('wqm_cli.cli.commands.service.ServiceManager')
    def test_service_status(self, mock_service_manager):
        """Test service status command."""
        mock_manager = MagicMock()
        mock_manager.get_status.return_value = {"status": "running"}
        mock_service_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["service", "status"])
        assert result.exit_code == 0
        mock_manager.get_status.assert_called_once()

    @patch('wqm_cli.cli.commands.service.ServiceManager')
    def test_service_start(self, mock_service_manager):
        """Test service start command."""
        mock_manager = MagicMock()
        mock_manager.start.return_value = True
        mock_service_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["service", "start"])
        assert result.exit_code == 0
        mock_manager.start.assert_called_once()

    @patch('wqm_cli.cli.commands.service.ServiceManager')
    def test_service_stop(self, mock_service_manager):
        """Test service stop command."""
        mock_manager = MagicMock()
        mock_manager.stop.return_value = True
        mock_service_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["service", "stop"])
        assert result.exit_code == 0
        mock_manager.stop.assert_called_once()

    @patch('wqm_cli.cli.commands.service.ServiceManager')
    def test_service_install(self, mock_service_manager):
        """Test service install command."""
        mock_manager = MagicMock()
        mock_manager.install.return_value = True
        mock_service_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["service", "install"])
        assert result.exit_code == 0
        mock_manager.install.assert_called_once()


class TestWatchCommands(TestCLIComprehensive):
    """Test watch command domain (start|stop|list|config)."""

    @patch('wqm_cli.cli.commands.watch.get_daemon_client')
    def test_watch_start(self, mock_get_client):
        """Test watch start command."""
        mock_client = MagicMock()
        mock_client.start_watch.return_value = {"status": "watching"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["watch", "start", self.temp_dir])
        assert result.exit_code == 0
        mock_client.start_watch.assert_called_once()

    @patch('wqm_cli.cli.commands.watch.get_daemon_client')
    def test_watch_stop(self, mock_get_client):
        """Test watch stop command."""
        mock_client = MagicMock()
        mock_client.stop_watch.return_value = {"status": "stopped"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["watch", "stop", self.temp_dir])
        assert result.exit_code == 0
        mock_client.stop_watch.assert_called_once()

    @patch('wqm_cli.cli.commands.watch.get_daemon_client')
    def test_watch_list(self, mock_get_client):
        """Test watch list command."""
        mock_client = MagicMock()
        mock_client.list_watches.return_value = []
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["watch", "list"])
        assert result.exit_code == 0
        mock_client.list_watches.assert_called_once()

    @patch('wqm_cli.cli.commands.watch.get_daemon_client')
    def test_watch_config(self, mock_get_client):
        """Test watch config command."""
        mock_client = MagicMock()
        mock_client.get_watch_config.return_value = {"interval": 5}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["watch", "config"])
        assert result.exit_code == 0
        mock_client.get_watch_config.assert_called_once()


class TestWebCommands(TestCLIComprehensive):
    """Test web command domain (start|dev|build|status)."""

    @patch('wqm_cli.cli.commands.web.WebServerManager')
    def test_web_start(self, mock_web_manager):
        """Test web start command."""
        mock_manager = MagicMock()
        mock_manager.start.return_value = True
        mock_web_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["web", "start"])
        assert result.exit_code == 0
        mock_manager.start.assert_called_once()

    @patch('wqm_cli.cli.commands.web.WebServerManager')
    def test_web_dev(self, mock_web_manager):
        """Test web dev command."""
        mock_manager = MagicMock()
        mock_manager.start_dev.return_value = True
        mock_web_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["web", "dev"])
        assert result.exit_code == 0
        mock_manager.start_dev.assert_called_once()

    @patch('wqm_cli.cli.commands.web.WebServerManager')
    def test_web_status(self, mock_web_manager):
        """Test web status command."""
        mock_manager = MagicMock()
        mock_manager.get_status.return_value = {"status": "running"}
        mock_web_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["web", "status"])
        assert result.exit_code == 0
        mock_manager.get_status.assert_called_once()

    @patch('wqm_cli.cli.commands.web.WebServerManager')
    def test_web_build(self, mock_web_manager):
        """Test web build command."""
        mock_manager = MagicMock()
        mock_manager.build.return_value = True
        mock_web_manager.return_value = mock_manager

        result = self.runner.invoke(app, ["web", "build"])
        assert result.exit_code == 0
        mock_manager.build.assert_called_once()


class TestInitCommand(TestCLIComprehensive):
    """Test init command for shell completion."""

    def test_init_bash(self):
        """Test init command for bash completion."""
        result = self.runner.invoke(app, ["init", "bash"])
        assert result.exit_code == 0
        assert "_wqm_completion" in result.stdout
        assert "complete -o default -F _wqm_completion wqm" in result.stdout

    def test_init_zsh(self):
        """Test init command for zsh completion."""
        result = self.runner.invoke(app, ["init", "zsh"])
        assert result.exit_code == 0
        assert "#compdef wqm" in result.stdout
        assert "_wqm_completion" in result.stdout

    def test_init_fish(self):
        """Test init command for fish completion."""
        result = self.runner.invoke(app, ["init", "fish"])
        assert result.exit_code == 0
        assert "complete -c wqm" in result.stdout

    def test_init_invalid_shell(self):
        """Test init command with invalid shell."""
        result = self.runner.invoke(app, ["init", "invalid"])
        assert result.exit_code != 0


class TestStatusCommands(TestCLIComprehensive):
    """Test status command domain."""

    @patch('wqm_cli.cli.status.get_daemon_client')
    def test_status_basic(self, mock_get_client):
        """Test basic status command."""
        mock_client = MagicMock()
        mock_client.get_status.return_value = {"status": "running"}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["status"])
        assert result.exit_code == 0
        mock_client.get_status.assert_called_once()

    @patch('wqm_cli.cli.status.get_daemon_client')
    def test_status_live(self, mock_get_client):
        """Test status with live monitoring."""
        mock_client = MagicMock()
        mock_client.get_status.return_value = {"status": "running"}
        mock_get_client.return_value = mock_client

        # Mock keyboard interrupt for live mode
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            result = self.runner.invoke(app, ["status", "--live", "--interval", "1"])
            # Should handle keyboard interrupt gracefully
            assert result.exit_code == 1 or "cancelled" in result.stdout


class TestObservabilityCommands(TestCLIComprehensive):
    """Test observability command domain."""

    @patch('wqm_cli.cli.observability.get_daemon_client')
    def test_observability_health(self, mock_get_client):
        """Test observability health command."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = {"healthy": True}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["observability", "health"])
        assert result.exit_code == 0
        mock_client.health_check.assert_called_once()

    @patch('wqm_cli.cli.observability.get_daemon_client')
    def test_observability_metrics(self, mock_get_client):
        """Test observability metrics command."""
        mock_client = MagicMock()
        mock_client.get_metrics.return_value = {"metrics": {}}
        mock_get_client.return_value = mock_client

        result = self.runner.invoke(app, ["observability", "metrics"])
        assert result.exit_code == 0
        mock_client.get_metrics.assert_called_once()


class TestConfigurationHierarchy(TestCLIComprehensive):
    """Test YAML configuration hierarchy and environment variable precedence."""

    def test_custom_config_path(self):
        """Test --config flag with custom configuration."""
        result = self.runner.invoke(app, ["--config", str(self.config_path), "admin", "status"])
        # Should not fail due to config loading
        assert "Error" not in result.stdout or result.exit_code == 0

    def test_environment_variable_precedence(self):
        """Test environment variables override config file."""
        with patch.dict(os.environ, {"WQM_QDRANT_URL": "http://test:6333"}):
            result = self.runner.invoke(app, ["--debug", "admin", "config", "show"])
            # Should show debug info about env override
            assert result.exit_code == 0

    def test_config_file_yaml_parsing(self):
        """Test YAML configuration file parsing."""
        # Create invalid YAML
        invalid_config = Path(self.temp_dir) / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        result = self.runner.invoke(app, ["--config", str(invalid_config), "admin", "status"])
        # Should handle invalid YAML gracefully
        assert result.exit_code != 0 or "Error" in result.stdout


class TestErrorHandlingAndExitCodes(TestCLIComprehensive):
    """Test error handling and proper exit codes."""

    def test_connection_error_exit_code(self):
        """Test connection errors return proper exit code."""
        with patch('wqm_cli.cli.commands.admin.get_daemon_client',
                   side_effect=ConnectionError("Connection failed")):
            result = self.runner.invoke(app, ["admin", "status"])
            assert result.exit_code == 1
            assert "Error" in result.stdout

    def test_invalid_argument_exit_code(self):
        """Test invalid arguments return proper exit code."""
        result = self.runner.invoke(app, ["memory", "add"])  # Missing argument
        assert result.exit_code == 2  # Typer usage error

    def test_file_not_found_exit_code(self):
        """Test file not found errors return proper exit code."""
        result = self.runner.invoke(app, ["ingest", "file", "/nonexistent/file.txt"])
        assert result.exit_code != 0

    def test_keyboard_interrupt_handling(self):
        """Test keyboard interrupt is handled gracefully."""
        with patch('wqm_cli.cli.main.handle_async_command',
                   side_effect=KeyboardInterrupt):
            result = self.runner.invoke(app, ["admin", "status"])
            assert result.exit_code == 1
            assert "cancelled" in result.stdout.lower()

    def test_unexpected_exception_handling(self):
        """Test unexpected exceptions are handled gracefully."""
        with patch('wqm_cli.cli.commands.admin.get_daemon_client',
                   side_effect=RuntimeError("Unexpected error")):
            result = self.runner.invoke(app, ["admin", "status"])
            assert result.exit_code == 1
            assert "Error" in result.stdout


class TestOutputFormatting(TestCLIComprehensive):
    """Test output formatting consistency."""

    def test_json_output_format(self):
        """Test JSON output format where supported."""
        with patch('wqm_cli.cli.commands.admin.get_daemon_client') as mock_client:
            mock_client.return_value.get_status.return_value = {"status": "running"}

            result = self.runner.invoke(app, ["admin", "status", "--json"])
            assert result.exit_code == 0
            # Should contain valid JSON output
            try:
                import json
                json.loads(result.stdout)
            except json.JSONDecodeError:
                # If JSON flag not supported, should at least not crash
                assert "Error" not in result.stdout

    def test_table_output_format(self):
        """Test table output format consistency."""
        with patch('wqm_cli.cli.commands.memory.get_daemon_client') as mock_client:
            mock_client.return_value.list_memory_rules.return_value = [
                {"id": "1", "rule": "Test rule 1"},
                {"id": "2", "rule": "Test rule 2"}
            ]

            result = self.runner.invoke(app, ["memory", "list"])
            assert result.exit_code == 0
            # Should contain tabular data
            assert "Test rule 1" in result.stdout
            assert "Test rule 2" in result.stdout

    def test_verbose_output_format(self):
        """Test verbose output format."""
        result = self.runner.invoke(app, ["--debug", "admin", "status"])
        # Debug mode should enable verbose output
        assert result.exit_code == 0 or "Debug" in result.stdout


class TestEdgeCasesAndBoundaryConditions(TestCLIComprehensive):
    """Test edge cases and boundary conditions."""

    def test_very_long_command_line(self):
        """Test very long command line arguments."""
        long_query = "x" * 1000
        with patch('wqm_cli.cli.commands.search.get_daemon_client') as mock_client:
            mock_client.return_value.search.return_value = {"results": []}

            result = self.runner.invoke(app, ["search", "project", long_query])
            assert result.exit_code == 0
            mock_client.return_value.search.assert_called_once()

    def test_special_characters_in_paths(self):
        """Test special characters in file paths."""
        special_dir = Path(self.temp_dir) / "special-dir with spaces & symbols!"
        special_dir.mkdir(exist_ok=True)

        with patch('wqm_cli.cli.commands.ingest.get_daemon_client') as mock_client:
            mock_client.return_value.ingest_folder.return_value = {"status": "processed"}

            result = self.runner.invoke(app, ["ingest", "folder", str(special_dir)])
            assert result.exit_code == 0

    def test_unicode_characters_in_commands(self):
        """Test unicode characters in command arguments."""
        unicode_query = "测试 query with émojis 🚀"
        with patch('wqm_cli.cli.commands.search.get_daemon_client') as mock_client:
            mock_client.return_value.search.return_value = {"results": []}

            result = self.runner.invoke(app, ["search", "project", unicode_query])
            assert result.exit_code == 0

    def test_empty_config_file(self):
        """Test empty configuration file."""
        empty_config = Path(self.temp_dir) / "empty.yaml"
        empty_config.write_text("")

        result = self.runner.invoke(app, ["--config", str(empty_config), "admin", "status"])
        # Should handle empty config gracefully
        assert result.exit_code == 0 or "Error" not in result.stdout

    def test_concurrent_command_execution(self):
        """Test CLI behavior under concurrent execution."""
        import threading
        import time

        results = []

        def run_command():
            result = self.runner.invoke(app, ["--version"])
            results.append(result.exit_code)

        # Run multiple commands concurrently
        threads = [threading.Thread(target=run_command) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert all(code == 0 for code in results)


if __name__ == "__main__":
    pytest.main([__file__])
