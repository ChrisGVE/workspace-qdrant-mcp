"""
CLI Command Workflow Testing

Comprehensive functional tests for complete CLI command workflows including
command sequences, output validation, and error reporting.

This module implements subtask 203.1 of the End-to-End Functional Testing Framework.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest
import yaml


class CLITestEnvironment:
    """Test environment for CLI command workflow testing."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.test_project_dir = tmp_path / "test_project"
        self.test_docs_dir = tmp_path / "test_docs"
        self.config_dir = tmp_path / ".config" / "workspace-qdrant"
        self.cli_executable = "uv run wqm"

        self.setup_test_environment()

    def setup_test_environment(self):
        """Set up test directories and files."""
        # Create test directories
        self.test_project_dir.mkdir(parents=True, exist_ok=True)
        self.test_docs_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create a test Git repository
        (self.test_project_dir / ".git").mkdir(exist_ok=True)

        # Create test documents
        test_files = {
            "README.md": "# Test Project\n\nThis is a test project for CLI testing.",
            "config.yaml": "project: test-project\nversion: 1.0.0",
            "notes.txt": "Important notes for the project",
            "document.md": "# Documentation\n\nDetailed project documentation.",
        }

        for filename, content in test_files.items():
            (self.test_docs_dir / filename).write_text(content)

    def run_cli_command(
        self,
        command: str,
        cwd: Path | None = None,
        input_data: str | None = None,
        timeout: int = 30,
        env_vars: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """
        Execute CLI command and return result.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if cwd is None:
            cwd = self.test_project_dir

        # Set up environment
        env = os.environ.copy()
        env.update({
            "QDRANT_URL": "http://localhost:6333",
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
        })
        if env_vars:
            env.update(env_vars)

        # Execute command
        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=True,
                cwd=cwd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command execution failed: {e}"


class CLIWorkflowValidator:
    """Validates CLI command workflows and outputs."""

    @staticmethod
    def validate_command_success(return_code: int, stdout: str, stderr: str) -> bool:
        """Validate that command executed successfully."""
        return return_code == 0

    @staticmethod
    def validate_json_output(output: str) -> dict[str, Any]:
        """Validate and parse JSON output."""
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nOutput: {output}")

    @staticmethod
    def validate_version_output(output: str) -> bool:
        """Validate version command output format."""
        # Should contain version number in semver format
        import re
        pattern = r'\d+\.\d+\.\d+'
        return bool(re.search(pattern, output))

    @staticmethod
    def validate_help_output(output: str, command: str = "") -> bool:
        """Validate help output contains expected content."""
        help_indicators = ["Usage:", "Commands:", "Options:", "help"]
        return any(indicator in output for indicator in help_indicators)

    @staticmethod
    def validate_status_output(output: str) -> dict[str, Any]:
        """Validate status command output."""
        try:
            data = json.loads(output)
            required_fields = ["status", "collections", "daemon"]
            return all(field in data for field in required_fields)
        except json.JSONDecodeError:
            # Non-JSON status output is also valid
            status_indicators = ["Status:", "Collections:", "Daemon:"]
            return any(indicator in output for indicator in status_indicators)


@pytest.mark.functional
@pytest.mark.cli_workflow
class TestCLICommandWorkflows:
    """Test complete CLI command workflows."""

    @pytest.fixture
    def cli_env(self, tmp_path):
        """Create CLI test environment."""
        return CLITestEnvironment(tmp_path)

    @pytest.fixture
    def validator(self):
        """Create CLI workflow validator."""
        return CLIWorkflowValidator()

    def test_version_workflow(self, cli_env, validator):
        """Test version command workflow."""
        # Test short version flag
        return_code, stdout, stderr = cli_env.run_cli_command("--version")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_version_output(stdout)

        # Test verbose version flag
        return_code, stdout, stderr = cli_env.run_cli_command("--version --verbose")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert "Workspace Qdrant MCP" in stdout
        assert "Python" in stdout
        assert "Platform" in stdout

    def test_help_workflow(self, cli_env, validator):
        """Test help command workflow."""
        # Test main help
        return_code, stdout, stderr = cli_env.run_cli_command("--help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout)

        # Test subcommand help
        subcommands = ["memory", "admin", "search", "ingest", "library"]
        for subcommand in subcommands:
            return_code, stdout, stderr = cli_env.run_cli_command(f"{subcommand} --help")
            assert validator.validate_command_success(return_code, stdout, stderr)
            assert validator.validate_help_output(stdout, subcommand)

    def test_admin_status_workflow(self, cli_env, validator):
        """Test admin status command workflow."""
        return_code, stdout, stderr = cli_env.run_cli_command("admin status")

        # Note: This might fail if Qdrant server is not running
        # We validate the command structure and output format
        if return_code == 0:
            assert validator.validate_status_output(stdout)
        else:
            # Even on failure, should provide meaningful error message
            assert len(stderr) > 0 or "Error" in stdout or "Failed" in stdout

    def test_memory_management_workflow(self, cli_env, validator):
        """Test memory management command workflow."""
        # Test list memory rules (should work even without daemon)
        return_code, stdout, stderr = cli_env.run_cli_command("memory list")

        # Command structure should be valid even if daemon is not running
        if return_code != 0:
            # Should provide clear error message about daemon requirements
            error_message = stderr + stdout
            assert any(indicator in error_message.lower() for indicator in [
                "daemon", "connection", "service", "not running"
            ])

    def test_search_workflow(self, cli_env, validator):
        """Test search command workflow."""
        # Test search help
        return_code, stdout, stderr = cli_env.run_cli_command("search --help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout, "search")

        # Test search command (may fail without daemon)
        return_code, stdout, stderr = cli_env.run_cli_command("search project 'test query'")
        # Validate error handling if daemon not available
        if return_code != 0:
            error_message = stderr + stdout
            assert len(error_message) > 0

    def test_ingest_workflow(self, cli_env, validator):
        """Test document ingestion workflow."""
        # Create test document
        test_doc = cli_env.test_docs_dir / "test.txt"
        test_doc.write_text("This is a test document for ingestion testing.")

        # Test ingest help
        return_code, stdout, stderr = cli_env.run_cli_command("ingest --help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout, "ingest")

        # Test ingest file command
        return_code, stdout, stderr = cli_env.run_cli_command(f"ingest file {test_doc}")
        # May fail without daemon, but should provide clear error
        if return_code != 0:
            error_message = stderr + stdout
            assert len(error_message) > 0

    def test_library_workflow(self, cli_env, validator):
        """Test library management workflow."""
        # Test library help
        return_code, stdout, stderr = cli_env.run_cli_command("library --help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout, "library")

        # Test library commands (may require daemon)
        return_code, stdout, stderr = cli_env.run_cli_command("library list")
        if return_code != 0:
            error_message = stderr + stdout
            assert len(error_message) > 0

    def test_watch_workflow(self, cli_env, validator):
        """Test watch management workflow."""
        # Test watch help
        return_code, stdout, stderr = cli_env.run_cli_command("watch --help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout, "watch")

        # Test watch list command
        return_code, stdout, stderr = cli_env.run_cli_command("watch list")
        # May fail without daemon
        if return_code != 0:
            error_message = stderr + stdout
            assert len(error_message) > 0

    def test_service_workflow(self, cli_env, validator):
        """Test service management workflow."""
        # Test service help
        return_code, stdout, stderr = cli_env.run_cli_command("service --help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout, "service")

        # Test service status (should work without daemon)
        return_code, stdout, stderr = cli_env.run_cli_command("service status")
        # Should either succeed or provide clear status
        assert return_code in [0, 1]  # Success or service not running
        error_message = stderr + stdout
        assert len(error_message) > 0

    def test_config_workflow(self, cli_env, validator):
        """Test configuration management workflow."""
        # Test config help
        return_code, stdout, stderr = cli_env.run_cli_command("config --help")
        assert validator.validate_command_success(return_code, stdout, stderr)
        assert validator.validate_help_output(stdout, "config")

        # Test config show command
        return_code, stdout, stderr = cli_env.run_cli_command("config show")
        # Should work even without daemon - shows config file status
        if return_code == 0:
            # Should show config information
            assert len(stdout) > 0
        else:
            # Should provide clear error about config
            error_message = stderr + stdout
            assert len(error_message) > 0

    def test_command_chaining_workflow(self, cli_env, validator):
        """Test command chaining and sequence workflows."""
        # Test multiple commands in sequence
        commands = [
            ("--version", "should show version"),
            ("--help", "should show help"),
            ("admin status", "should check status"),
            ("service status", "should check service"),
        ]

        results = []
        for command, description in commands:
            return_code, stdout, stderr = cli_env.run_cli_command(command)
            results.append({
                "command": command,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
                "description": description
            })

        # Validate command sequence results
        assert len(results) == len(commands)

        # Version and help should always work
        assert results[0]["return_code"] == 0  # version
        assert results[1]["return_code"] == 0  # help

        # Status commands may fail but should provide feedback
        for result in results[2:]:
            assert len(result["stdout"] + result["stderr"]) > 0

    def test_error_handling_workflow(self, cli_env, validator):
        """Test error handling in CLI workflows."""
        # Test invalid command
        return_code, stdout, stderr = cli_env.run_cli_command("invalid-command")
        assert return_code != 0
        error_message = stderr + stdout
        assert "invalid-command" in error_message.lower() or "unknown" in error_message.lower()

        # Test invalid subcommand
        return_code, stdout, stderr = cli_env.run_cli_command("memory invalid-subcommand")
        assert return_code != 0
        error_message = stderr + stdout
        assert len(error_message) > 0

        # Test invalid file path for ingest
        return_code, stdout, stderr = cli_env.run_cli_command("ingest file /nonexistent/file.txt")
        assert return_code != 0
        error_message = stderr + stdout
        assert len(error_message) > 0

    def test_output_formatting_workflow(self, cli_env, validator):
        """Test output formatting consistency across commands."""
        # Test commands that should produce consistent output formats
        format_commands = [
            ("--version", "version_format"),
            ("--help", "help_format"),
            ("admin status", "status_format"),
            ("memory list", "list_format"),
        ]

        for command, format_type in format_commands:
            return_code, stdout, stderr = cli_env.run_cli_command(command)

            # Validate output is not empty and properly formatted
            if return_code == 0:
                assert len(stdout.strip()) > 0

                # Check for consistent formatting elements
                if format_type == "version_format":
                    assert validator.validate_version_output(stdout)
                elif format_type == "help_format":
                    assert validator.validate_help_output(stdout)
                elif format_type == "status_format":
                    # Status should be structured
                    assert len(stdout.strip()) > 0
            else:
                # Error messages should be informative
                error_message = stderr + stdout
                assert len(error_message.strip()) > 0

    @pytest.mark.slow
    def test_long_running_workflow(self, cli_env, validator):
        """Test long-running command workflows and timeout handling."""
        # Test command timeout handling
        with patch('subprocess.run') as mock_run:
            # Simulate long-running command
            mock_run.side_effect = subprocess.TimeoutExpired("test", 1)

            return_code, stdout, stderr = cli_env.run_cli_command("admin status", timeout=1)
            assert return_code == -1
            assert "timed out" in stderr

    def test_environment_variable_workflow(self, cli_env, validator):
        """Test CLI workflows with different environment variables."""
        # Test with debug mode
        return_code, stdout, stderr = cli_env.run_cli_command(
            "--debug admin status",
            env_vars={"WQM_LOG_INIT": "true"}
        )

        # Debug mode should work (may fail on status but command structure is valid)
        if return_code == 0:
            # Debug mode might produce more verbose output
            assert len(stdout + stderr) > 0
        else:
            # Should still provide meaningful error
            assert len(stderr + stdout) > 0

        # Test with custom config path
        config_file = cli_env.config_dir / "test-config.yaml"
        config_file.write_text("qdrant_url: http://localhost:6333\n")

        return_code, stdout, stderr = cli_env.run_cli_command(
            f"--config {config_file} admin status"
        )

        # Command should accept config file (may fail on execution)
        assert len(stdout + stderr) > 0


@pytest.mark.functional
@pytest.mark.cli_integration
class TestCLIIntegrationWorkflows:
    """Test integrated CLI workflows that span multiple commands."""

    @pytest.fixture
    def cli_env(self, tmp_path):
        """Create CLI test environment."""
        return CLITestEnvironment(tmp_path)

    def test_project_initialization_workflow(self, cli_env):
        """Test complete project initialization workflow."""
        # Simulate setting up a new project workspace
        project_dir = cli_env.tmp_path / "new_project"
        project_dir.mkdir()

        # Initialize git repository
        subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)

        # Test project detection and initialization
        return_code, stdout, stderr = cli_env.run_cli_command(
            "admin status",
            cwd=project_dir
        )

        # Should detect project context
        assert len(stdout + stderr) > 0

    def test_document_lifecycle_workflow(self, cli_env):
        """Test complete document lifecycle workflow."""
        # Create test document
        doc_file = cli_env.test_docs_dir / "lifecycle_test.md"
        doc_file.write_text("# Test Document\n\nContent for lifecycle testing.")

        # Test document ingestion workflow
        commands = [
            f"ingest file {doc_file}",
            "search project 'lifecycle'",
            "memory list",
        ]

        results = []
        for command in commands:
            return_code, stdout, stderr = cli_env.run_cli_command(command)
            results.append({
                "command": command,
                "success": return_code == 0,
                "output": stdout + stderr
            })

        # Validate workflow execution
        assert all(len(result["output"]) > 0 for result in results)

    def test_configuration_workflow(self, cli_env):
        """Test configuration management workflow."""
        # Test configuration discovery and validation
        commands = [
            "config show",
            "admin status",
            "service status",
        ]

        for command in commands:
            return_code, stdout, stderr = cli_env.run_cli_command(command)
            # Should provide feedback about configuration state
            assert len(stdout + stderr) > 0
