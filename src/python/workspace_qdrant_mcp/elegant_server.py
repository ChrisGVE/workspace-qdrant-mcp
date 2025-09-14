#!/usr/bin/env python3
"""
Elegant FastMCP-inspired server entry point with "quiet by default" behavior.

This implements the clean, protocol-compliant approach:
- Default to file-only logging (no console noise)
- Optional informative banner to stderr (not stdout)
- Clean MCP protocol compliance (stdout reserved for JSON-RPC)
- --verbose flag enables console logging
- Simple, maintainable architecture
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Set up quiet environment before any other imports that might log
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["MCP_QUIET_MODE"] = "true"
os.environ["WQM_STDIO_MODE"] = "true"

import warnings
warnings.filterwarnings("ignore")

import typer

from loguru import logger
from common.logging.loguru_config import # TODO: Replace with setup_logging from common.logging.loguru_config


def setup_quiet_environment():
    """Set up quiet environment immediately to prevent startup noise."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""

    # Set quiet mode early to prevent logging during imports
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["WQM_STDIO_MODE"] = "true"

    import warnings
    warnings.filterwarnings("ignore")


def show_elegant_banner(transport: str, verbose: bool, config_file: Optional[str] = None):
    """Display elegant startup banner to stderr (FastMCP style)."""
    if transport == "stdio" and not verbose:
        # In stdio mode without verbose, don't show banner to avoid protocol interference
        return

    banner_lines = [
        "╭─────────────────────────────────────────╮",
        "│  🚀 workspace-qdrant-mcp                │",
        f"│  📦 Transport: {transport.upper():<24} │",
        "│  🔧 Config: XDG compliant              │",
        "│  📊 Collections: Ready                  │",
        "╰─────────────────────────────────────────╯"
    ]

    if config_file:
        # Truncate long config paths for banner display
        config_display = config_file if len(config_file) <= 19 else f"...{config_file[-16:]}"
        banner_lines.insert(-2, f"│  ⚙️  Config: {config_display:<25} │")

    # Print to stderr so it doesn't interfere with stdout MCP protocol
    for line in banner_lines:
        print(line, file=sys.stderr)
    print(file=sys.stderr)  # Extra newline for spacing


def run_elegant_server(
    transport: str = typer.Option("stdio", help="Transport protocol (stdio, http, sse)"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to (HTTP transports only)"),
    port: int = typer.Option(8000, help="Port to bind to (HTTP transports only)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable console logging and banner"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress banner completely"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to YAML configuration file"),
) -> None:
    """Start workspace-qdrant-mcp server with elegant FastMCP-inspired behavior.

    Features:
    - Quiet by default: File logging only, no console noise
    - Optional banner: Informative startup display (stderr only)
    - Protocol compliant: stdout reserved for MCP JSON-RPC
    - Verbose mode: Enable console logging when needed
    - Simple architecture: Clean, maintainable codebase

    Examples:
        # Start for Claude Desktop (silent, protocol-compliant)
        workspace-qdrant-mcp

        # Start with verbose console logging and banner
        workspace-qdrant-mcp --verbose

        # Start HTTP server with banner
        workspace-qdrant-mcp --transport http --verbose

        # Start with custom config and verbose logging
        workspace-qdrant-mcp --config config.yaml --verbose

        # Completely quiet mode (no banner)
        workspace-qdrant-mcp --quiet
    """

    # Set up quiet environment immediately
    setup_quiet_environment()

    # Show banner unless explicitly quiet
    if not quiet:
        show_elegant_banner(transport, verbose, config)

    # Configure environment for existing server implementation
    if verbose:
        # Enable console logging in the existing server
        os.environ.pop("MCP_QUIET_MODE", None)
        if transport != "stdio":
            os.environ.pop("WQM_STDIO_MODE", None)
    else:
        # Enable quiet mode (default behavior)
        os.environ["MCP_QUIET_MODE"] = "true"
        if transport == "stdio":
            os.environ["WQM_STDIO_MODE"] = "true"

    # Set up file logging for troubleshooting
    log_dir = Path.home() / ".workspace-qdrant-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOG_FILE"] = str(log_dir / "server.log")

    # Set appropriate log level
    os.environ["LOG_LEVEL"] = "INFO" if verbose else "WARNING"

    # Import and delegate to the existing server implementation
    try:
        from .server import run_server as existing_run_server

        # Call the existing server implementation
        existing_run_server(transport=transport, host=host, port=port, config=config)

    except Exception as e:
        if verbose:
            print(f"❌ Server error: {e}", file=sys.stderr)
        # Always log errors to file for debugging with loguru
        # TODO: Replace with setup_logging from common.logging.loguru_config(level="ERROR", console_output=False)
        logger.error(f"Server error in elegant server: {e}")
        sys.exit(1)


def main() -> None:
    """Console script entry point for elegant server."""
    typer.run(run_elegant_server)


if __name__ == "__main__":
    main()