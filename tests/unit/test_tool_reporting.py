"""Unit tests for tool reporting system."""

import platform
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.tool_reporting import MissingTool, ToolReporter


@pytest.fixture
def mock_discovery():
    """Create mock ToolDiscovery instance."""
    discovery = MagicMock()
    return discovery


@pytest.fixture
def mock_state_manager():
    """Create mock SQLiteStateManager instance."""
    state_manager = MagicMock()
    state_manager._lock = MagicMock()
    state_manager.connection = MagicMock()
    return state_manager


@pytest.fixture
def tool_reporter(mock_discovery, mock_state_manager):
    """Create ToolReporter instance with mocks."""
    return ToolReporter(mock_discovery, mock_state_manager)


class TestMissingTool:
    """Test MissingTool dataclass."""

    def test_missing_tool_creation(self):
        """Test creating a MissingTool instance."""
        tool = MissingTool(
            name="pyright",
            tool_type="lsp_server",
            language="python",
            severity="critical",
            install_command="npm install -g pyright",
            install_url="https://github.com/microsoft/pyright",
        )

        assert tool.name == "pyright"
        assert tool.tool_type == "lsp_server"
        assert tool.language == "python"
        assert tool.severity == "critical"
        assert tool.install_command == "npm install -g pyright"
        assert tool.install_url == "https://github.com/microsoft/pyright"

    def test_missing_tool_defaults(self):
        """Test MissingTool with default values."""
        tool = MissingTool(name="test-tool", tool_type="lsp_server")

        assert tool.name == "test-tool"
        assert tool.tool_type == "lsp_server"
        assert tool.language is None
        assert tool.severity == "recommended"
        assert tool.install_command is None
        assert tool.install_url is None


class TestToolReporter:
    """Test ToolReporter class."""

    def test_initialization(self, mock_discovery, mock_state_manager):
        """Test ToolReporter initialization."""
        reporter = ToolReporter(mock_discovery, mock_state_manager)

        assert reporter.discovery == mock_discovery
        assert reporter.state_manager == mock_state_manager
        assert reporter._current_platform == platform.system().lower()

    @pytest.mark.asyncio
    async def test_get_missing_tools_with_lsp_servers(
        self, tool_reporter, mock_state_manager
    ):
        """Test getting missing LSP servers."""
        # Mock database query for missing LSP servers
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "language_name": "python",
                "lsp_executable": "pyright",
                "lsp_name": "Pyright",
            }
        ]
        mock_cursor.fetchone.return_value = {"count": 0}  # No missing tree-sitter

        mock_state_manager.connection.execute.return_value = mock_cursor

        missing_tools = await tool_reporter.get_missing_tools()

        assert len(missing_tools) == 1
        assert missing_tools[0].name == "Pyright"
        assert missing_tools[0].tool_type == "lsp_server"
        assert missing_tools[0].language == "python"
        assert missing_tools[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_get_missing_tools_with_tree_sitter(
        self, tool_reporter, mock_state_manager
    ):
        """Test getting missing tree-sitter CLI."""
        # Mock database query
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []  # No missing LSP servers
        mock_cursor.fetchone.return_value = {"count": 5}  # 5 languages missing tree-sitter

        mock_state_manager.connection.execute.return_value = mock_cursor

        missing_tools = await tool_reporter.get_missing_tools()

        assert len(missing_tools) == 1
        assert missing_tools[0].name == "tree-sitter"
        assert missing_tools[0].tool_type == "tree_sitter"
        assert missing_tools[0].language is None
        assert missing_tools[0].severity == "recommended"

    @pytest.mark.asyncio
    async def test_get_missing_tools_multiple(
        self, tool_reporter, mock_state_manager
    ):
        """Test getting multiple missing tools."""
        # Mock database query
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "language_name": "python",
                "lsp_executable": "pyright",
                "lsp_name": "Pyright",
            },
            {
                "language_name": "rust",
                "lsp_executable": "rust-analyzer",
                "lsp_name": "rust-analyzer",
            },
        ]
        mock_cursor.fetchone.return_value = {"count": 3}  # Tree-sitter missing

        mock_state_manager.connection.execute.return_value = mock_cursor

        missing_tools = await tool_reporter.get_missing_tools()

        assert len(missing_tools) == 3
        tool_names = {tool.name for tool in missing_tools}
        assert "Pyright" in tool_names
        assert "rust-analyzer" in tool_names
        assert "tree-sitter" in tool_names

    @pytest.mark.asyncio
    async def test_get_missing_tools_empty(self, tool_reporter, mock_state_manager):
        """Test getting missing tools when none are missing."""
        # Mock database query - no missing tools
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []  # No missing LSP servers
        mock_cursor.fetchone.return_value = {"count": 0}  # No missing tree-sitter

        mock_state_manager.connection.execute.return_value = mock_cursor

        missing_tools = await tool_reporter.get_missing_tools()

        assert len(missing_tools) == 0

    @pytest.mark.asyncio
    async def test_get_missing_tools_error_handling(
        self, tool_reporter, mock_state_manager
    ):
        """Test error handling in get_missing_tools."""
        # Mock database error
        mock_state_manager.connection.execute.side_effect = Exception(
            "Database error"
        )

        missing_tools = await tool_reporter.get_missing_tools()

        # Should return empty list on error
        assert len(missing_tools) == 0

    def test_get_language_severity_critical(self, tool_reporter):
        """Test severity classification for critical languages."""
        assert tool_reporter._get_language_severity("python") == "critical"
        assert tool_reporter._get_language_severity("javascript") == "critical"
        assert tool_reporter._get_language_severity("typescript") == "critical"
        assert tool_reporter._get_language_severity("java") == "critical"

    def test_get_language_severity_recommended(self, tool_reporter):
        """Test severity classification for recommended languages."""
        assert tool_reporter._get_language_severity("csharp") == "recommended"
        assert tool_reporter._get_language_severity("ruby") == "recommended"
        assert tool_reporter._get_language_severity("swift") == "recommended"

    def test_get_language_severity_optional(self, tool_reporter):
        """Test severity classification for optional languages."""
        assert tool_reporter._get_language_severity("scala") == "optional"
        assert tool_reporter._get_language_severity("unknown_lang") == "optional"

    @patch("platform.system", return_value="Darwin")
    def test_get_install_command_macos(self, mock_platform):
        """Test getting installation command for macOS."""
        from src.python.common.core.tool_reporting import ToolReporter

        reporter = ToolReporter(MagicMock(), MagicMock())

        # Test known tool
        command = reporter.get_install_command("pyright", "lsp_server")
        assert command == "npm install -g pyright"

        # Test rust-analyzer
        command = reporter.get_install_command("rust-analyzer", "lsp_server")
        assert command == "rustup component add rust-analyzer"

        # Test tree-sitter
        command = reporter.get_install_command("tree-sitter", "tree_sitter")
        assert "npm install -g tree-sitter-cli" in command

    @patch("platform.system", return_value="Linux")
    def test_get_install_command_linux(self, mock_platform):
        """Test getting installation command for Linux."""
        from src.python.common.core.tool_reporting import ToolReporter

        reporter = ToolReporter(MagicMock(), MagicMock())

        # Test known tool
        command = reporter.get_install_command("pyright", "lsp_server")
        assert command == "npm install -g pyright"

        # Test clangd
        command = reporter.get_install_command("clangd", "lsp_server")
        assert "apt-get install clangd" in command

    @patch("platform.system", return_value="Windows")
    def test_get_install_command_windows(self, mock_platform):
        """Test getting installation command for Windows."""
        from src.python.common.core.tool_reporting import ToolReporter

        reporter = ToolReporter(MagicMock(), MagicMock())

        # Test known tool
        command = reporter.get_install_command("pyright", "lsp_server")
        assert command == "npm install -g pyright"

        # Test clangd
        command = reporter.get_install_command("clangd", "lsp_server")
        assert command == "choco install llvm"

    def test_get_install_command_unknown_tool(self, tool_reporter):
        """Test getting installation command for unknown tool."""
        command = tool_reporter.get_install_command("unknown-tool", "lsp_server")
        assert command is None

    def test_get_install_url(self, tool_reporter):
        """Test getting documentation URL for tools."""
        # Test known tools
        assert (
            tool_reporter.get_install_url("pyright")
            == "https://github.com/microsoft/pyright"
        )
        assert (
            tool_reporter.get_install_url("rust-analyzer")
            == "https://rust-analyzer.github.io/"
        )
        assert (
            tool_reporter.get_install_url("tree-sitter")
            == "https://tree-sitter.github.io/tree-sitter/"
        )

        # Test unknown tool
        assert tool_reporter.get_install_url("unknown-tool") is None

    def test_generate_installation_guide_empty(self, tool_reporter):
        """Test generating installation guide with no missing tools."""
        guide = tool_reporter.generate_installation_guide([])

        assert "All development tools are installed and available" in guide

    def test_generate_installation_guide_critical_only(self, tool_reporter):
        """Test generating installation guide with critical tools only."""
        missing_tools = [
            MissingTool(
                name="pyright",
                tool_type="lsp_server",
                language="python",
                severity="critical",
                install_command="npm install -g pyright",
                install_url="https://github.com/microsoft/pyright",
            )
        ]

        guide = tool_reporter.generate_installation_guide(missing_tools)

        assert "Missing Development Tools" in guide
        assert "CRITICAL (required for core functionality)" in guide
        assert "pyright" in guide
        assert "npm install -g pyright" in guide
        assert "https://github.com/microsoft/pyright" in guide
        assert "Total missing tools: 1" in guide
        assert "Critical: 1" in guide

    def test_generate_installation_guide_all_severities(self, tool_reporter):
        """Test generating installation guide with all severity levels."""
        missing_tools = [
            MissingTool(
                name="pyright",
                tool_type="lsp_server",
                language="python",
                severity="critical",
                install_command="npm install -g pyright",
                install_url="https://github.com/microsoft/pyright",
            ),
            MissingTool(
                name="rust-analyzer",
                tool_type="lsp_server",
                language="rust",
                severity="recommended",
                install_command="rustup component add rust-analyzer",
                install_url="https://rust-analyzer.github.io/",
            ),
            MissingTool(
                name="scala-lsp",
                tool_type="lsp_server",
                language="scala",
                severity="optional",
                install_command="coursier install metals",
                install_url="https://scalameta.org/metals/",
            ),
        ]

        guide = tool_reporter.generate_installation_guide(missing_tools)

        # Check all sections present
        assert "CRITICAL (required for core functionality)" in guide
        assert "RECOMMENDED (improves development experience)" in guide
        assert "OPTIONAL (nice to have)" in guide

        # Check all tools present
        assert "pyright" in guide
        assert "rust-analyzer" in guide
        assert "scala-lsp" in guide

        # Check summary
        assert "Total missing tools: 3" in guide
        assert "Critical: 1" in guide
        assert "Recommended: 1" in guide
        assert "Optional: 1" in guide

    def test_generate_installation_guide_no_install_command(self, tool_reporter):
        """Test generating guide for tool without install command."""
        missing_tools = [
            MissingTool(
                name="unknown-tool",
                tool_type="lsp_server",
                language="unknown",
                severity="recommended",
                install_command=None,
                install_url="https://example.com/unknown-tool",
            )
        ]

        guide = tool_reporter.generate_installation_guide(missing_tools)

        assert "unknown-tool" in guide
        assert "(installation command not available)" in guide
        assert "https://example.com/unknown-tool" in guide

    def test_format_tool_entry_with_language(self, tool_reporter):
        """Test formatting tool entry with language."""
        tool = MissingTool(
            name="pyright",
            tool_type="lsp_server",
            language="python",
            severity="critical",
            install_command="npm install -g pyright",
            install_url="https://github.com/microsoft/pyright",
        )

        lines = tool_reporter._format_tool_entry(tool)

        assert any("pyright" in line for line in lines)
        assert any("python" in line for line in lines)
        assert any("lsp server" in line for line in lines)
        assert any("npm install -g pyright" in line for line in lines)
        assert any("https://github.com/microsoft/pyright" in line for line in lines)

    def test_format_tool_entry_without_language(self, tool_reporter):
        """Test formatting tool entry without language."""
        tool = MissingTool(
            name="tree-sitter",
            tool_type="tree_sitter",
            language=None,
            severity="recommended",
            install_command="npm install -g tree-sitter-cli",
            install_url="https://tree-sitter.github.io/tree-sitter/",
        )

        lines = tool_reporter._format_tool_entry(tool)

        assert any("tree-sitter" in line for line in lines)
        assert any("tree sitter" in line for line in lines)
        assert not any("python" in line for line in lines)
        assert any("npm install -g tree-sitter-cli" in line for line in lines)
