"""
Integration tests for MCP Inspector debugging tool.

This module tests the integration between workspace-qdrant-mcp and the MCP Inspector
React-based debugging tool to ensure real-time protocol monitoring and debugging
capabilities work correctly.
"""

import json
import os
import subprocess
import tempfile
import pytest
from pathlib import Path


class TestMCPInspectorIntegration:
    """Test MCP Inspector integration with workspace-qdrant-mcp server."""

    @pytest.fixture
    def debug_dir(self):
        """Get the debug directory path."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "debug"

    @pytest.fixture
    def mcp_config(self, debug_dir):
        """Load MCP Inspector configuration."""
        config_path = debug_dir / "mcp-inspector-config.json"
        with open(config_path) as f:
            return json.load(f)

    def test_debug_directory_structure(self, debug_dir):
        """Test that debug directory has all required files."""
        required_files = [
            "mcp-inspector-config.json",
            "package.json",
            "debug-stdio.sh",
            "debug-http.sh",
            "test-tools.sh",
            "README.md"
        ]

        for file_name in required_files:
            file_path = debug_dir / file_name
            assert file_path.exists(), f"Required file {file_name} not found in debug directory"

    def test_mcp_inspector_config_structure(self, mcp_config):
        """Test MCP Inspector configuration structure."""
        assert "mcpServers" in mcp_config

        servers = mcp_config["mcpServers"]
        expected_servers = ["workspace-qdrant-stdio", "workspace-qdrant-http", "workspace-qdrant-sse"]

        for server_name in expected_servers:
            assert server_name in servers, f"Server {server_name} not found in configuration"

        # Test STDIO server configuration
        stdio_config = servers["workspace-qdrant-stdio"]
        assert stdio_config["type"] == "stdio"
        assert stdio_config["command"] == "uv"
        assert "workspace_qdrant_mcp.server" in " ".join(stdio_config["args"])
        assert "WQM_STDIO_MODE" in stdio_config["env"]

    def test_package_json_structure(self, debug_dir):
        """Test package.json has required scripts."""
        package_json_path = debug_dir / "package.json"
        with open(package_json_path) as f:
            package_config = json.load(f)

        required_scripts = [
            "debug-stdio",
            "debug-http",
            "debug-ui",
            "list-tools",
            "list-resources",
            "test-connection"
        ]

        for script in required_scripts:
            assert script in package_config["scripts"], f"Script {script} not found in package.json"

        assert "@modelcontextprotocol/inspector" in package_config["devDependencies"]

    def test_debug_scripts_executable(self, debug_dir):
        """Test that debug scripts are executable."""
        script_files = ["debug-stdio.sh", "debug-http.sh", "test-tools.sh"]

        for script_name in script_files:
            script_path = debug_dir / script_name
            assert os.access(script_path, os.X_OK), f"Script {script_name} is not executable"

    @pytest.mark.integration
    def test_mcp_inspector_tool_listing(self, debug_dir):
        """Test MCP Inspector can list tools from workspace-qdrant server."""
        # Run the npm command to list tools
        result = subprocess.run(
            ["npm", "run", "list-tools"],
            cwd=debug_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Tool listing failed: {result.stderr}"

        # Parse the JSON output to verify tools are listed
        # Skip npm output header and find JSON content
        output_lines = result.stdout.strip().split('\n')
        json_start = -1

        for i, line in enumerate(output_lines):
            if line.strip().startswith('{'):
                json_start = i
                break

        assert json_start >= 0, f"No JSON output found in: {result.stdout}"

        # Join lines from JSON start to end to handle multiline JSON
        json_text = '\n'.join(output_lines[json_start:])
        json_output = json.loads(json_text)
        assert "tools" in json_output, "Tools not found in output"

        # Verify expected tools are present
        tools = json_output["tools"]
        tool_names = [tool["name"] for tool in tools]

        expected_tools = ["workspace_status", "echo_test", "search_workspace", "get_server_info"]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Expected tool {expected_tool} not found"

    @pytest.mark.integration
    def test_mcp_inspector_echo_test(self, debug_dir):
        """Test MCP Inspector can call tools on workspace-qdrant server."""
        # Test the echo_test tool
        result = subprocess.run([
            "npx", "@modelcontextprotocol/inspector", "--cli",
            "--config", "mcp-inspector-config.json",
            "--server", "workspace-qdrant-stdio",
            "--method", "tools/call",
            "--tool-name", "echo_test",
            "--tool-arg", "message=MCP Inspector Integration Test"
        ], cwd=debug_dir, capture_output=True, text=True, timeout=30)

        assert result.returncode == 0, f"Echo test failed: {result.stderr}"

        # Parse output to verify response
        # Skip npm output header and find JSON content
        output_lines = result.stdout.strip().split('\n')
        json_start = -1

        for i, line in enumerate(output_lines):
            if line.strip().startswith('{'):
                json_start = i
                break

        assert json_start >= 0, f"No JSON response found in: {result.stdout}"

        # Join lines from JSON start to end to handle multiline JSON
        json_text = '\n'.join(output_lines[json_start:])
        json_output = json.loads(json_text)
        # Check for result in either direct result or structured content
        if "result" in json_output:
            result_text = json_output["result"]
        elif "structuredContent" in json_output and "result" in json_output["structuredContent"]:
            result_text = json_output["structuredContent"]["result"]
        elif "content" in json_output and len(json_output["content"]) > 0:
            result_text = json_output["content"][0].get("text", "")
        else:
            result_text = ""

        assert "MCP Inspector Integration Test" in result_text, f"Expected message not found in: {result_text}"

    def test_npm_dependencies_installed(self, debug_dir):
        """Test that npm dependencies are properly installed."""
        node_modules = debug_dir / "node_modules"
        assert node_modules.exists(), "node_modules directory not found"

        inspector_module = node_modules / "@modelcontextprotocol" / "inspector"
        assert inspector_module.exists(), "MCP Inspector module not installed"

    def test_debug_documentation_comprehensive(self, debug_dir):
        """Test that debug documentation is comprehensive."""
        readme_path = debug_dir / "README.md"
        with open(readme_path) as f:
            readme_content = f.read()

        # Check for key sections
        required_sections = [
            "# MCP Inspector Debugging Tools",
            "## Quick Start",
            "## Transport Modes",
            "## Configuration Files",
            "## Available Scripts",
            "## MCP Tools Available for Debugging",
            "## Debugging Workflow",
            "## Troubleshooting"
        ]

        for section in required_sections:
            assert section in readme_content, f"Required section '{section}' not found in README"

        # Check for practical examples
        assert "debug-stdio.sh" in readme_content
        assert "localhost:6274" in readme_content
        assert "workspace-qdrant-mcp" in readme_content

    @pytest.mark.integration
    def test_server_configuration_paths(self, debug_dir, mcp_config):
        """Test that server configuration paths are correct."""
        stdio_config = mcp_config["mcpServers"]["workspace-qdrant-stdio"]

        # Verify working directory is set correctly
        expected_cwd = str(debug_dir.parent)
        assert stdio_config["cwd"] == expected_cwd, f"Working directory mismatch: expected {expected_cwd}, got {stdio_config['cwd']}"

        # Verify command exists
        result = subprocess.run(["which", "uv"], capture_output=True, text=True)
        assert result.returncode == 0, "uv command not found in PATH"

    def test_environment_variables_configuration(self, mcp_config):
        """Test environment variables are properly configured."""
        stdio_config = mcp_config["mcpServers"]["workspace-qdrant-stdio"]
        env_vars = stdio_config["env"]

        required_env_vars = ["WQM_STDIO_MODE", "WQM_LOG_LEVEL", "QDRANT_URL"]
        for var in required_env_vars:
            assert var in env_vars, f"Required environment variable {var} not found"

        assert env_vars["WQM_STDIO_MODE"] == "true"
        assert env_vars["WQM_LOG_LEVEL"] == "DEBUG"
        assert "localhost:6333" in env_vars["QDRANT_URL"]

    def test_transport_configurations(self, mcp_config):
        """Test different transport configurations."""
        servers = mcp_config["mcpServers"]

        # Test STDIO configuration
        stdio = servers["workspace-qdrant-stdio"]
        assert stdio["type"] == "stdio"
        assert "command" in stdio
        assert "args" in stdio

        # Test HTTP configuration
        http = servers["workspace-qdrant-http"]
        assert http["type"] == "streamable-http"
        assert "url" in http
        assert "localhost:8000" in http["url"]

        # Test SSE configuration
        sse = servers["workspace-qdrant-sse"]
        assert sse["type"] == "sse"
        assert "url" in sse
        assert "localhost:8000" in sse["url"]