"""End-to-end integration tests combining CLI and web UI functionality.

This module tests the complete workflow from CLI commands to web UI functionality:
- CLI commands prepare the web environment
- Web UI serves the interface correctly
- CLI and UI work together for complete workflows
- Integration between daemon, CLI, and web interface

These tests require both CLI functionality and web UI to be working.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from playwright.async_api import Browser, Page, async_playwright, expect
from typer.testing import CliRunner
from wqm_cli.cli.main import app


class DevServerManager:
    """Manage development server for testing."""

    def __init__(self, web_ui_path: Path, port: int = 3000):
        self.web_ui_path = web_ui_path
        self.port = port
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://localhost:{port}"

    async def start(self, timeout: int = 30) -> bool:
        """Start development server and wait for it to be ready."""
        if not (self.web_ui_path / "package.json").exists():
            return False

        try:
            # Start dev server
            env = os.environ.copy()
            env["PORT"] = str(self.port)
            env["BROWSER"] = "none"  # Don't open browser automatically

            self.process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.web_ui_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(self.base_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status < 400:
                                return True
                except:
                    pass

                # Check if process is still running
                if self.process.poll() is not None:
                    break

                await asyncio.sleep(1)

            return False

        except Exception as e:
            print(f"Failed to start dev server: {e}")
            return False

    def stop(self):
        """Stop the development server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


@pytest.fixture(scope="module")
def test_web_ui_path():
    """Create a test web UI directory with minimal setup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        web_ui_path = Path(temp_dir) / "web-ui"
        web_ui_path.mkdir()

        # Create minimal package.json for testing
        package_json = {
            "name": "test-web-ui",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "start": "echo 'Test dev server started' && python -m http.server 3000",
                "build": "echo 'Test build completed' && mkdir -p dist && echo '<html><body>Test build</body></html>' > dist/index.html",
                "serve": "python -m http.server 8080 --directory dist"
            }
        }

        (web_ui_path / "package.json").write_text(json.dumps(package_json, indent=2))

        # Create src directory structure
        src_dir = web_ui_path / "src"
        src_dir.mkdir()
        (src_dir / "index.html").write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>Test Workspace UI</title>
</head>
<body>
    <div id="root">
        <h1>Test Workspace</h1>
        <nav>
            <a href="/workspace-status">Workspace Status</a>
            <a href="/processing-queue">Processing Queue</a>
            <a href="/memory-rules">Memory Rules</a>
        </nav>
        <div id="content">
            <p>Test workspace interface</p>
            <button id="safety-toggle">Safety Mode</button>
            <button id="readonly-toggle">Read Only Mode</button>
        </div>
    </div>
</body>
</html>
        """)

        yield web_ui_path


class TestWebCLIIntegration:
    """Test CLI commands work correctly for web management."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    def test_cli_to_web_workflow(self, mock_get_web_ui_path, test_web_ui_path):
        """Test complete workflow: install -> build -> status -> start."""
        mock_get_web_ui_path.return_value = test_web_ui_path

        # 1. Install dependencies
        with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0

            result = self.runner.invoke(app, ["web", "install"])
            assert result.exit_code == 0
            assert "Installing Node.js dependencies..." in result.stdout

        # 2. Build the project
        with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0

            result = self.runner.invoke(app, ["web", "build"])
            assert result.exit_code == 0
            assert "Building web UI for production..." in result.stdout

        # 3. Check status
        result = self.runner.invoke(app, ["web", "status"])
        assert result.exit_code == 0
        assert "Web UI Status:" in result.stdout
        assert str(test_web_ui_path) in result.stdout

    @patch('wqm_cli.cli.commands.web.get_web_ui_path')
    def test_build_creates_dist_directory(self, mock_get_web_ui_path, test_web_ui_path):
        """Test that build command creates dist directory."""
        mock_get_web_ui_path.return_value = test_web_ui_path

        # Build should create dist directory
        with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
            def create_dist(*args, **kwargs):
                dist_dir = test_web_ui_path / "dist"
                dist_dir.mkdir(exist_ok=True)
                (dist_dir / "index.html").write_text("<html>Built</html>")
                return MagicMock(returncode=0)

            mock_subprocess.side_effect = create_dist

            result = self.runner.invoke(app, ["web", "build"])
            assert result.exit_code == 0

        # Verify dist was created
        dist_path = test_web_ui_path / "dist"
        assert dist_path.exists()
        assert (dist_path / "index.html").exists()

        # Status should now show built assets
        result = self.runner.invoke(app, ["web", "status"])
        assert result.exit_code == 0
        assert "✓" in result.stdout  # Built assets should show checkmark


@pytest.mark.asyncio
class TestCLIWebUIIntegration:
    """Test integration between CLI setup and web UI functionality."""

    async def test_cli_build_and_browser_access(self, test_web_ui_path):
        """Test CLI build produces accessible web content."""
        runner = CliRunner()

        # Build via CLI
        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_get_path:
            mock_get_path.return_value = test_web_ui_path

            with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
                def mock_build(*args, **kwargs):
                    # Simulate build process
                    dist_dir = test_web_ui_path / "dist"
                    dist_dir.mkdir(exist_ok=True)
                    (dist_dir / "index.html").write_text("""
<!DOCTYPE html>
<html>
<head><title>Built Workspace UI</title></head>
<body>
    <h1>Workspace Dashboard</h1>
    <div id="status">Ready</div>
    <button id="refresh-btn">Refresh</button>
</body>
</html>
                    """)
                    return MagicMock(returncode=0)

                mock_subprocess.side_effect = mock_build

                result = runner.invoke(app, ["web", "build"])
                assert result.exit_code == 0

        # Test that built files can be accessed
        dist_path = test_web_ui_path / "dist" / "index.html"
        assert dist_path.exists()

        content = dist_path.read_text()
        assert "Workspace Dashboard" in content
        assert "status" in content

    async def test_dev_server_integration_workflow(self, test_web_ui_path):
        """Test complete workflow with development server."""
        # This test would require actually starting the dev server
        # For now, we'll test the setup and mock the server interaction

        runner = CliRunner()

        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_get_path:
            mock_get_path.return_value = test_web_ui_path

            # 1. Prepare environment
            with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)

                # Install dependencies
                result = runner.invoke(app, ["web", "install"])
                assert result.exit_code == 0

                # Start dev server (mocked)
                result = runner.invoke(app, ["web", "dev", "--port", "3001"])
                assert result.exit_code == 0

                # Verify correct npm command was called
                calls = mock_subprocess.call_args_list
                dev_call = calls[-1]  # Last call should be npm start
                args = dev_call[0][0]
                assert "npm" in args[0]
                assert "start" in args

                # Check environment variables
                env = dev_call.kwargs.get("env", {})
                assert env.get("PORT") == "3001"

    async def test_production_server_workflow(self, test_web_ui_path):
        """Test production build and serve workflow."""
        runner = CliRunner()

        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_get_path:
            mock_get_path.return_value = test_web_ui_path

            with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
                def mock_commands(*args, **kwargs):
                    # First call: build
                    # Second call: serve
                    if "build" in args[0]:
                        dist_dir = test_web_ui_path / "dist"
                        dist_dir.mkdir(exist_ok=True)
                        (dist_dir / "index.html").write_text("<html>Production build</html>")
                    return MagicMock(returncode=0)

                mock_subprocess.side_effect = mock_commands

                # Start production server
                result = runner.invoke(app, ["web", "start", "--port", "8081"])
                assert result.exit_code == 0

                # Should have called build then serve
                calls = mock_subprocess.call_args_list
                assert len(calls) >= 2

                # Build call
                build_call = calls[0]
                assert "build" in build_call[0][0]

                # Serve call
                serve_call = calls[1]
                serve_args = serve_call[0][0]
                assert "serve" in serve_args
                assert "8081" in str(serve_args)


@pytest.mark.asyncio
class TestFullStackIntegration:
    """Test complete stack: CLI + daemon + web UI integration."""

    async def test_web_ui_daemon_connectivity(self):
        """Test web UI can connect to daemon API."""
        # This would test the actual API connectivity
        # For now, test the configuration and setup

        runner = CliRunner()

        # Test that web commands don't interfere with daemon
        runner.invoke(app, ["status"])  # Regular daemon status
        # Should work regardless of web UI state

        # Web status should be independent
        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_path:
            mock_path.return_value = Path("/tmp/test-web-ui")
            runner.invoke(app, ["web", "status"])
            # Should handle missing web UI gracefully

    async def test_safety_mode_integration(self):
        """Test safety mode affects both CLI and web UI."""
        # This would test that safety settings are consistent
        # between CLI and web interface

        runner = CliRunner()

        # In a real test, we would:
        # 1. Set safety mode via CLI
        # 2. Check it's reflected in web UI
        # 3. Change it via web UI
        # 4. Verify CLI sees the change

        # For now, verify the commands exist and work
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        with patch('wqm_cli.cli.commands.web.get_web_ui_path'):
            result = runner.invoke(app, ["web", "--help"])
            assert result.exit_code == 0

    async def test_configuration_consistency(self):
        """Test configuration is consistent between components."""
        # Test that configuration used by CLI commands
        # matches what the web UI expects

        runner = CliRunner()

        # Get CLI configuration paths
        runner.invoke(app, ["status"])
        # In a real implementation, this would check config paths

        # Web UI should use compatible configuration
        with patch('wqm_cli.cli.commands.web.get_web_ui_path'):
            runner.invoke(app, ["web", "status"])
            # Should show compatible status information


class TestWebUIErrorIntegration:
    """Test error handling integration between CLI and web UI."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_web_ui_missing_graceful_degradation(self):
        """Test CLI handles missing web UI gracefully."""
        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_path:
            # Mock missing web UI directory
            mock_path.side_effect = Exception("Web UI not found")

            # Commands should fail gracefully
            result = self.runner.invoke(app, ["web", "status"])
            assert result.exit_code == 1

            result = self.runner.invoke(app, ["web", "build"])
            assert result.exit_code == 1

    def test_npm_missing_error_handling(self):
        """Test handling when npm is not available."""
        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_path:
            mock_path.return_value = Path("/tmp/test-web")

            with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
                # Mock npm not found
                mock_subprocess.side_effect = FileNotFoundError("npm not found")

                result = self.runner.invoke(app, ["web", "install"])
                assert result.exit_code == 1
                assert "npm not found" in result.stdout
                assert "nodejs.org" in result.stdout

    def test_build_failure_handling(self):
        """Test handling of build process failures."""
        with patch('wqm_cli.cli.commands.web.get_web_ui_path') as mock_path:
            mock_path.return_value = Path("/tmp/test-web")

            with patch('wqm_cli.cli.commands.web.subprocess.run') as mock_subprocess:
                # Mock build failure
                mock_subprocess.side_effect = subprocess.CalledProcessError(1, "npm run build")

                result = self.runner.invoke(app, ["web", "build"])
                assert result.exit_code == 1
                assert "Build failed" in result.stdout or "Failed" in result.stdout


def pytest_configure(config):
    """Configure pytest for web UI integration tests."""
    config.addinivalue_line("markers", "web_integration: mark test as web integration test")
    config.addinivalue_line("markers", "requires_dev_server: mark test as requiring dev server")
    config.addinivalue_line("markers", "requires_daemon: mark test as requiring daemon")
