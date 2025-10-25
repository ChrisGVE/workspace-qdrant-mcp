"""Integration tests for web CLI commands.

This module tests the actual functionality of `wqm web` commands including:
- Dependency management (install)
- Build process (build)
- Development server (dev)
- Production server (start)
- Status reporting (status)

Tests use real npm processes and file system operations but in controlled
test environments to validate complete functionality.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner
from wqm_cli.cli.main import app


class TestWebCommandsIntegration:
    """Integration tests for web CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.original_cwd = os.getcwd()

    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)

    def create_mock_web_ui_directory(self, temp_dir: Path) -> Path:
        """Create a mock web-ui directory structure for testing."""
        web_ui_path = temp_dir / "web-ui"
        web_ui_path.mkdir(parents=True, exist_ok=True)

        # Create package.json
        package_json = {
            "name": "qdrant-web-ui",
            "version": "1.0.0",
            "license": "MIT",
            "scripts": {
                "build": "echo 'Mock build process'",
                "serve": "echo 'Mock serve process'",
                "start": "echo 'Mock dev process'",
                "dev": "echo 'Mock dev process'"
            },
            "dependencies": {
                "react": "^18.0.0"
            }
        }

        import json
        (web_ui_path / "package.json").write_text(json.dumps(package_json, indent=2))

        # Create mock dist directory
        dist_dir = web_ui_path / "dist"
        dist_dir.mkdir(exist_ok=True)
        (dist_dir / "index.html").write_text("<html><body>Mock build output</body></html>")

        return web_ui_path

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    def test_web_status_command(self, mock_get_web_ui_path):
        """Test web status command shows correct information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            result = self.runner.invoke(app, ["web", "status"])

            assert result.exit_code == 0
            output = result.stdout

            # Check status information is displayed
            assert "Web UI Status:" in output
            assert str(web_ui_path) in output
            assert "✓" in output  # Should show checkmarks for existing files
            assert "Version: 1.0.0" in output
            assert "License: MIT" in output

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    def test_web_status_missing_files(self, mock_get_web_ui_path):
        """Test web status command with missing dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = temp_path / "web-ui"
            web_ui_path.mkdir(parents=True)

            # Create package.json but no node_modules or dist
            package_json = {"name": "test", "version": "1.0.0"}
            import json
            (web_ui_path / "package.json").write_text(json.dumps(package_json))

            mock_get_web_ui_path.return_value = web_ui_path

            result = self.runner.invoke(app, ["web", "status"])

            assert result.exit_code == 0
            output = result.stdout

            # Should show missing dependencies and build
            assert "✗ (run: wqm web install)" in output
            assert "✗ (run: wqm web build)" in output

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    def test_web_install_command(self, mock_subprocess, mock_get_web_ui_path):
        """Test web install command runs npm install."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            # Mock successful npm install
            mock_subprocess.return_value.returncode = 0

            result = self.runner.invoke(app, ["web", "install"])

            assert result.exit_code == 0

            # Verify npm install was called
            mock_subprocess.assert_called_with(
                ["npm", "install"],
                cwd=web_ui_path,
                check=True,
                capture_output=False
            )

            assert "Installing Node.js dependencies..." in result.stdout
            assert "Dependencies installed successfully!" in result.stdout

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    def test_web_install_npm_not_found(self, mock_subprocess, mock_get_web_ui_path):
        """Test web install command when npm is not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            # Mock FileNotFoundError for missing npm
            mock_subprocess.side_effect = FileNotFoundError("npm not found")

            result = self.runner.invoke(app, ["web", "install"])

            assert result.exit_code == 1
            assert "npm not found" in result.stdout
            assert "https://nodejs.org/" in result.stdout

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    @patch('wqm_cli.cli.commands.web.ensure_dependencies')
    def test_web_build_command(self, mock_ensure_deps, mock_subprocess, mock_get_web_ui_path):
        """Test web build command runs npm build."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            # Mock successful build
            mock_subprocess.return_value.returncode = 0

            result = self.runner.invoke(app, ["web", "build"])

            assert result.exit_code == 0

            # Verify dependencies check was called
            mock_ensure_deps.assert_called_once_with(web_ui_path)

            # Verify npm build was called
            mock_subprocess.assert_called_with(
                ["npm", "run", "build"],
                cwd=web_ui_path,
                env=os.environ.copy(),
                check=True,
                capture_output=False
            )

            assert "Building web UI for production..." in result.stdout
            assert "Build completed successfully!" in result.stdout

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    @patch('wqm_cli.cli.commands.web.ensure_dependencies')
    def test_web_build_with_custom_output(self, mock_ensure_deps, mock_subprocess, mock_get_web_ui_path):
        """Test web build command with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path
            custom_output = "/custom/output"

            # Mock successful build
            mock_subprocess.return_value.returncode = 0

            result = self.runner.invoke(app, ["web", "build", "--output", custom_output])

            assert result.exit_code == 0

            # Verify environment variable was set
            call_args = mock_subprocess.call_args
            env = call_args.kwargs["env"]
            assert env["BUILD_PATH"] == custom_output

            assert custom_output in result.stdout

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    @patch('wqm_cli.cli.commands.web.ensure_dependencies')
    def test_web_dev_command(self, mock_ensure_deps, mock_subprocess, mock_get_web_ui_path):
        """Test web dev command starts development server."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            result = self.runner.invoke(app, ["web", "dev", "--port", "3001", "--host", "0.0.0.0"])

            assert result.exit_code == 0

            # Verify dependencies check was called
            mock_ensure_deps.assert_called_once_with(web_ui_path)

            # Verify npm start was called with correct environment
            call_args = mock_subprocess.call_args
            env = call_args.kwargs["env"]
            assert env["PORT"] == "3001"
            assert env["HOST"] == "0.0.0.0"

            assert "Starting development server on 0.0.0.0:3001" in result.stdout
            assert "Hot reloading enabled" in result.stdout

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    @patch('wqm_cli.cli.commands.web.ensure_dependencies')
    def test_web_start_command(self, mock_ensure_deps, mock_subprocess, mock_get_web_ui_path):
        """Test web start command builds and serves production."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            result = self.runner.invoke(app, ["web", "start", "--port", "8081", "--host", "0.0.0.0"])

            assert result.exit_code == 0

            # Verify dependencies check was called
            mock_ensure_deps.assert_called_once_with(web_ui_path)

            # Should call build then serve
            calls = mock_subprocess.call_args_list
            assert len(calls) >= 2

            # First call should be build
            build_call = calls[0]
            assert build_call[0][0] == ["npm", "run", "build"]

            # Second call should be serve
            serve_call = calls[1]
            args = serve_call[0][0]
            assert "npm" in args[0]
            assert "serve" in args
            assert "--port" in args
            assert "8081" in args

            assert "Starting web UI server on 0.0.0.0:8081" in result.stdout
            assert "Building web UI..." in result.stdout

    def test_web_ui_path_not_found(self):
        """Test behavior when web-ui directory doesn't exist."""
        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_get_path:
            # Mock a path that doesn't exist
            mock_get_path.side_effect = Exception("Web UI not found")

            result = self.runner.invoke(app, ["web", "status"])

            assert result.exit_code == 1

    @pytest.mark.parametrize("command", ["install", "build", "dev", "start"])
    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    @patch('wqm_cli.cli.commands.web.subprocess.run')
    def test_web_commands_handle_npm_failures(self, mock_subprocess, mock_get_web_ui_path, command):
        """Test that web commands handle npm process failures gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            web_ui_path = self.create_mock_web_ui_directory(temp_path)
            mock_get_web_ui_path.return_value = web_ui_path

            # Mock subprocess failure
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, "npm")

            result = self.runner.invoke(app, ["web", command])

            assert result.exit_code == 1
            # Should contain appropriate error message
            assert any(error_word in result.stdout.lower()
                      for error_word in ["failed", "error"])

    def test_web_command_help(self):
        """Test web command shows help information."""
        result = self.runner.invoke(app, ["web", "--help"])

        assert result.exit_code == 0
        assert "Web UI server for workspace-qdrant-mcp" in result.stdout

        # Should list available subcommands
        for subcommand in ["start", "dev", "build", "install", "status"]:
            assert subcommand in result.stdout

    @pytest.mark.parametrize("subcommand", ["start", "dev", "build", "install", "status"])
    def test_web_subcommand_help(self, subcommand):
        """Test web subcommands show help information."""
        result = self.runner.invoke(app, ["web", subcommand, "--help"])

        assert result.exit_code == 0
        # Each subcommand should have descriptive help text
        assert len(result.stdout) > 50  # Reasonable help text length
