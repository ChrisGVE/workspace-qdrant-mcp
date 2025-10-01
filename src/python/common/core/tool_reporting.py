"""Tool reporting system for missing development tools.

This module provides user-facing reports for missing LSP servers, Tree-sitter CLI,
and other development tools with installation guidance. It queries the language
support database and generates actionable reports with platform-specific
installation commands.

Architecture:
    - Missing tool detection via database queries
    - Installation command lookup by platform (macOS, Linux, Windows)
    - Severity classification (critical, recommended, optional)
    - Formatted terminal output with installation guidance
    - Official documentation links

Example:
    ```python
    from workspace_qdrant_mcp.core.tool_reporting import ToolReporter
    from workspace_qdrant_mcp.core.tool_discovery import ToolDiscovery
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager

    # Initialize components
    discovery = ToolDiscovery()
    state_manager = SQLiteStateManager()
    await state_manager.initialize()

    # Generate missing tools report
    reporter = ToolReporter(discovery, state_manager)
    missing_tools = await reporter.get_missing_tools()
    report = reporter.generate_installation_guide(missing_tools)
    print(report)
    ```
"""

import platform
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from .sqlite_state_manager import SQLiteStateManager
from .tool_discovery import ToolDiscovery


@dataclass
class MissingTool:
    """Record for a missing development tool.

    Attributes:
        name: Tool name (e.g., "pyright", "rust-analyzer", "tree-sitter")
        tool_type: Type of tool (lsp_server, tree_sitter, compiler, build_tool)
        language: Language this tool supports (None for language-agnostic tools)
        severity: Impact severity (critical, recommended, optional)
        install_command: Platform-specific installation command
        install_url: Official documentation URL
    """

    name: str
    tool_type: str  # 'lsp_server', 'tree_sitter', 'compiler', 'build_tool'
    language: Optional[str] = None  # For LSP servers
    severity: str = "recommended"  # 'critical', 'recommended', 'optional'
    install_command: Optional[str] = None
    install_url: Optional[str] = None


class ToolReporter:
    """Tool reporter for missing development tools.

    Generates user-facing reports for missing LSP servers, Tree-sitter CLI,
    and other development tools with platform-specific installation commands
    and official documentation links.

    Attributes:
        discovery: ToolDiscovery instance for tool lookup
        state_manager: SQLiteStateManager for database queries
    """

    # Installation commands by tool name and platform
    INSTALL_COMMANDS: Dict[str, Dict[str, str]] = {
        # LSP Servers
        "pyright": {
            "darwin": "npm install -g pyright",
            "linux": "npm install -g pyright",
            "windows": "npm install -g pyright",
        },
        "pyright-langserver": {
            "darwin": "pip install pyright",
            "linux": "pip install pyright",
            "windows": "pip install pyright",
        },
        "rust-analyzer": {
            "darwin": "rustup component add rust-analyzer",
            "linux": "rustup component add rust-analyzer",
            "windows": "rustup component add rust-analyzer",
        },
        "typescript-language-server": {
            "darwin": "npm install -g typescript-language-server typescript",
            "linux": "npm install -g typescript-language-server typescript",
            "windows": "npm install -g typescript-language-server typescript",
        },
        "gopls": {
            "darwin": "go install golang.org/x/tools/gopls@latest",
            "linux": "go install golang.org/x/tools/gopls@latest",
            "windows": "go install golang.org/x/tools/gopls@latest",
        },
        "clangd": {
            "darwin": "brew install llvm",
            "linux": "sudo apt-get install clangd  # or: sudo dnf install clang-tools-extra",
            "windows": "choco install llvm",
        },
        "jdtls": {
            "darwin": "brew install jdtls",
            "linux": "Download from https://download.eclipse.org/jdtls/snapshots/",
            "windows": "Download from https://download.eclipse.org/jdtls/snapshots/",
        },
        # Tree-sitter
        "tree-sitter": {
            "darwin": "npm install -g tree-sitter-cli  # or: cargo install tree-sitter-cli",
            "linux": "npm install -g tree-sitter-cli  # or: cargo install tree-sitter-cli",
            "windows": "npm install -g tree-sitter-cli  # or: cargo install tree-sitter-cli",
        },
        # Compilers
        "gcc": {
            "darwin": "xcode-select --install  # or: brew install gcc",
            "linux": "sudo apt-get install gcc  # or: sudo dnf install gcc",
            "windows": "Download MinGW from https://www.mingw-w64.org/",
        },
        "clang": {
            "darwin": "xcode-select --install  # or: brew install llvm",
            "linux": "sudo apt-get install clang  # or: sudo dnf install clang",
            "windows": "choco install llvm",
        },
        # Build Tools
        "cmake": {
            "darwin": "brew install cmake",
            "linux": "sudo apt-get install cmake  # or: sudo dnf install cmake",
            "windows": "choco install cmake  # or download from https://cmake.org/",
        },
        "cargo": {
            "darwin": "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
            "linux": "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
            "windows": "Download from https://rustup.rs/",
        },
        "npm": {
            "darwin": "brew install node",
            "linux": "sudo apt-get install nodejs npm  # or: sudo dnf install nodejs npm",
            "windows": "choco install nodejs  # or download from https://nodejs.org/",
        },
    }

    # Official documentation URLs
    DOCUMENTATION_URLS: Dict[str, str] = {
        # LSP Servers
        "pyright": "https://github.com/microsoft/pyright",
        "pyright-langserver": "https://github.com/microsoft/pyright",
        "rust-analyzer": "https://rust-analyzer.github.io/",
        "typescript-language-server": "https://github.com/typescript-language-server/typescript-language-server",
        "gopls": "https://pkg.go.dev/golang.org/x/tools/gopls",
        "clangd": "https://clangd.llvm.org/",
        "jdtls": "https://github.com/eclipse/eclipse.jdt.ls",
        # Tree-sitter
        "tree-sitter": "https://tree-sitter.github.io/tree-sitter/",
        # Compilers
        "gcc": "https://gcc.gnu.org/",
        "clang": "https://clang.llvm.org/",
        # Build Tools
        "cmake": "https://cmake.org/",
        "cargo": "https://doc.rust-lang.org/cargo/",
        "npm": "https://www.npmjs.com/",
    }

    # Language popularity for severity classification
    # Higher values indicate more critical languages
    LANGUAGE_POPULARITY: Dict[str, int] = {
        "python": 10,
        "javascript": 10,
        "typescript": 9,
        "java": 9,
        "cpp": 8,
        "c": 8,
        "rust": 8,
        "go": 8,
        "csharp": 7,
        "ruby": 6,
        "php": 6,
        "swift": 6,
        "kotlin": 6,
        "scala": 5,
    }

    def __init__(
        self, discovery: ToolDiscovery, state_manager: SQLiteStateManager
    ):
        """Initialize tool reporter.

        Args:
            discovery: ToolDiscovery instance for tool lookup
            state_manager: SQLiteStateManager for database queries
        """
        self.discovery = discovery
        self.state_manager = state_manager
        self._current_platform = platform.system().lower()
        logger.debug(f"ToolReporter initialized for platform: {self._current_platform}")

    async def get_missing_tools(self) -> List[MissingTool]:
        """Get list of missing development tools.

        Queries the database for languages with missing LSP servers or
        Tree-sitter parsers and returns a list of MissingTool objects
        with installation guidance.

        Returns:
            List of MissingTool objects for missing tools
        """
        missing_tools: List[MissingTool] = []

        try:
            # Query database for missing LSP servers
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT language_name, lsp_executable, lsp_name
                    FROM languages
                    WHERE lsp_missing = 1 AND lsp_executable IS NOT NULL
                    ORDER BY language_name
                    """
                )
                rows = cursor.fetchall()

                for row in rows:
                    language_name = row["language_name"]
                    lsp_executable = row["lsp_executable"]
                    lsp_name = row["lsp_name"] or lsp_executable

                    # Determine severity based on language popularity
                    severity = self._get_language_severity(language_name)

                    # Get installation command and URL
                    install_command = self.get_install_command(
                        lsp_executable, "lsp_server"
                    )
                    install_url = self.get_install_url(lsp_executable)

                    missing_tool = MissingTool(
                        name=lsp_name,
                        tool_type="lsp_server",
                        language=language_name,
                        severity=severity,
                        install_command=install_command,
                        install_url=install_url,
                    )
                    missing_tools.append(missing_tool)

                # Query database for missing Tree-sitter CLI
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM languages
                    WHERE ts_missing = 1 AND ts_grammar IS NOT NULL
                    """
                )
                row = cursor.fetchone()
                ts_missing_count = row["count"] if row else 0

                if ts_missing_count > 0:
                    # Tree-sitter CLI is missing
                    install_command = self.get_install_command(
                        "tree-sitter", "tree_sitter"
                    )
                    install_url = self.get_install_url("tree-sitter")

                    missing_tool = MissingTool(
                        name="tree-sitter",
                        tool_type="tree_sitter",
                        language=None,  # Language-agnostic
                        severity="recommended",
                        install_command=install_command,
                        install_url=install_url,
                    )
                    missing_tools.append(missing_tool)

            logger.info(f"Found {len(missing_tools)} missing development tools")
            return missing_tools

        except Exception as e:
            logger.error(f"Failed to get missing tools: {e}")
            return []

    def _get_language_severity(self, language_name: str) -> str:
        """Determine severity level for a language based on popularity.

        Args:
            language_name: Name of the programming language

        Returns:
            Severity level: "critical", "recommended", or "optional"
        """
        language_lower = language_name.lower()
        popularity = self.LANGUAGE_POPULARITY.get(language_lower, 0)

        if popularity >= 9:
            return "critical"
        elif popularity >= 6:
            return "recommended"
        else:
            return "optional"

    def get_install_command(self, tool_name: str, tool_type: str) -> Optional[str]:
        """Get platform-specific installation command for a tool.

        Args:
            tool_name: Name of the tool (e.g., "pyright", "tree-sitter")
            tool_type: Type of tool (lsp_server, tree_sitter, compiler, build_tool)

        Returns:
            Platform-specific installation command, or None if not found
        """
        # Check if we have a command for this tool
        tool_commands = self.INSTALL_COMMANDS.get(tool_name)

        if not tool_commands:
            logger.debug(f"No installation command found for tool: {tool_name}")
            return None

        # Get command for current platform
        command = tool_commands.get(self._current_platform)

        if not command:
            logger.debug(
                f"No installation command for {tool_name} on platform: {self._current_platform}"
            )
            return None

        return command

    def get_install_url(self, tool_name: str) -> Optional[str]:
        """Get official documentation URL for a tool.

        Args:
            tool_name: Name of the tool (e.g., "pyright", "tree-sitter")

        Returns:
            Official documentation URL, or None if not found
        """
        url = self.DOCUMENTATION_URLS.get(tool_name)

        if not url:
            logger.debug(f"No documentation URL found for tool: {tool_name}")

        return url

    def generate_installation_guide(
        self, missing_tools: List[MissingTool]
    ) -> str:
        """Generate formatted installation guide for missing tools.

        Creates a formatted report grouped by severity level with
        installation commands and documentation links.

        Args:
            missing_tools: List of MissingTool objects

        Returns:
            Formatted installation guide string
        """
        if not missing_tools:
            return "All development tools are installed and available."

        # Group tools by severity
        critical_tools: List[MissingTool] = []
        recommended_tools: List[MissingTool] = []
        optional_tools: List[MissingTool] = []

        for tool in missing_tools:
            if tool.severity == "critical":
                critical_tools.append(tool)
            elif tool.severity == "recommended":
                recommended_tools.append(tool)
            else:
                optional_tools.append(tool)

        # Build report
        lines = [
            "Missing Development Tools",
            "=" * 60,
            "",
        ]

        # Critical tools
        if critical_tools:
            lines.append("CRITICAL (required for core functionality):")
            lines.append("")
            for tool in critical_tools:
                lines.extend(self._format_tool_entry(tool))
                lines.append("")

        # Recommended tools
        if recommended_tools:
            lines.append("RECOMMENDED (improves development experience):")
            lines.append("")
            for tool in recommended_tools:
                lines.extend(self._format_tool_entry(tool))
                lines.append("")

        # Optional tools
        if optional_tools:
            lines.append("OPTIONAL (nice to have):")
            lines.append("")
            for tool in optional_tools:
                lines.extend(self._format_tool_entry(tool))
                lines.append("")

        # Summary
        total = len(missing_tools)
        lines.extend(
            [
                "=" * 60,
                f"Total missing tools: {total}",
                f"  Critical: {len(critical_tools)}",
                f"  Recommended: {len(recommended_tools)}",
                f"  Optional: {len(optional_tools)}",
            ]
        )

        return "\n".join(lines)

    def _format_tool_entry(self, tool: MissingTool) -> List[str]:
        """Format a single tool entry for the installation guide.

        Args:
            tool: MissingTool object to format

        Returns:
            List of formatted lines for this tool
        """
        lines = []

        # Tool header
        if tool.language:
            header = f"- {tool.name} ({tool.language} {tool.tool_type.replace('_', ' ')})"
        else:
            header = f"- {tool.name} ({tool.tool_type.replace('_', ' ')})"
        lines.append(header)

        # Installation command
        if tool.install_command:
            lines.append(f"  Install: {tool.install_command}")
        else:
            lines.append("  Install: (installation command not available)")

        # Documentation URL
        if tool.install_url:
            lines.append(f"  Docs: {tool.install_url}")

        return lines
