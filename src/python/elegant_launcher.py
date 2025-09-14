#!/usr/bin/env python3
"""
Elegant launcher for workspace-qdrant-mcp with FastMCP-inspired behavior.

This replaces the complex stdio launcher with a simple, elegant approach:
- Quiet by default (file logging only)
- Optional beautiful banner with --verbose
- Clean MCP protocol compliance
- Simple argument parsing
- Maintains compatibility with existing server
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from common.logging.loguru_config import setup_logging


def show_elegant_banner(transport: str, config_file: Optional[str] = None):
    """Display elegant startup banner to stderr (FastMCP style)."""
    banner_lines = [
        "╭─────────────────────────────────────────╮",
        "│  🚀 workspace-qdrant-mcp                │",
        f"│  📦 Transport: {transport.upper():<24} │",
        "│  🔧 Config: XDG compliant              │",
        "│  📊 Collections: Ready                  │",
        "╰─────────────────────────────────────────╯"
    ]

    if config_file:
        config_display = config_file if len(config_file) <= 19 else f"...{config_file[-16:]}"
        banner_lines.insert(-2, f"│  ⚙️  Config: {config_display:<25} │")

    for line in banner_lines:
        print(line, file=sys.stderr)
    print(file=sys.stderr)


def parse_arguments(args):
    """Simple argument parsing."""
    config = {
        "transport": "stdio",
        "host": "127.0.0.1",
        "port": 8000,
        "verbose": False,
        "quiet": False,
        "config_file": None,
        "help": False
    }

    i = 0
    while i < len(args):
        if args[i] == "--transport" and i + 1 < len(args):
            config["transport"] = args[i + 1]
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            config["host"] = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            config["port"] = int(args[i + 1])
            i += 2
        elif args[i] == "--config" and i + 1 < len(args):
            config["config_file"] = args[i + 1]
            i += 2
        elif args[i] in ["--verbose", "-v"]:
            config["verbose"] = True
            i += 1
        elif args[i] in ["--quiet", "-q"]:
            config["quiet"] = True
            i += 1
        elif args[i] in ["--help", "-h"]:
            config["help"] = True
            i += 1
        else:
            i += 1

    return config


def print_help():
    """Print elegant help message."""
    help_text = """
workspace-qdrant-mcp - Elegant FastMCP-inspired MCP server

USAGE:
    workspace-qdrant-mcp [OPTIONS]

OPTIONS:
    --transport <TRANSPORT>    Transport protocol (stdio, http, sse) [default: stdio]
    --host <HOST>              Host to bind to (HTTP only) [default: 127.0.0.1]
    --port <PORT>              Port to bind to (HTTP only) [default: 8000]
    --config <CONFIG>          Path to YAML configuration file
    --verbose, -v              Enable console logging and show banner
    --quiet, -q                Suppress banner completely
    --help, -h                 Show this help message

EXAMPLES:
    # Start for Claude Desktop (silent, protocol-compliant)
    workspace-qdrant-mcp

    # Start with verbose console logging and banner
    workspace-qdrant-mcp --verbose

    # Start HTTP server with banner
    workspace-qdrant-mcp --transport http --verbose

    # Start with custom config
    workspace-qdrant-mcp --config my-config.yaml --verbose

FEATURES:
    🔇 Quiet by default - File logging only, no console noise
    🎨 Optional banner - Beautiful startup display (stderr only)
    🎯 Protocol compliant - stdout reserved for MCP JSON-RPC
    🚀 FastMCP inspired - Clean, maintainable architecture
"""
    print(help_text.strip())


def setup_environment(config):
    """Set up environment variables before server import."""
    # Always set these for third-party library quieting
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""

    # Configure quiet behavior
    if config["verbose"]:
        # Enable console output in verbose mode
        os.environ.pop("MCP_QUIET_MODE", None)
        if config["transport"] != "stdio":
            os.environ.pop("WQM_STDIO_MODE", None)
    else:
        # Default quiet behavior
        os.environ["MCP_QUIET_MODE"] = "true"
        if config["transport"] == "stdio":
            os.environ["WQM_STDIO_MODE"] = "true"

    # Set up file logging
    log_dir = Path.home() / ".workspace-qdrant-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOG_FILE"] = str(log_dir / "server.log")
    os.environ["LOG_LEVEL"] = "INFO" if config["verbose"] else "WARNING"


def run_server(config):
    """Run the server with the given configuration."""
    # Determine the correct path to source code
    script_path = Path(__file__).absolute()
    project_root = script_path.parent
    src_dir = project_root / "src" / "python"

    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    try:
        # Import the server after environment setup
        from workspace_qdrant_mcp.server import run_server as existing_run_server

        # Call the existing server implementation
        existing_run_server(
            transport=config["transport"],
            host=config["host"],
            port=config["port"],
            config=config["config_file"]
        )

    except ImportError as e:
        if config["verbose"]:
            print(f"❌ Import error: {e}", file=sys.stderr)
            print("Make sure you're running from the correct directory", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if config["verbose"]:
            print(f"❌ Server error: {e}", file=sys.stderr)
        # Always log errors to file with loguru
        # TODO: Replace with setup_logging from common.logging.loguru_config(level="ERROR", console_output=False)
        logger.error(f"Server error in elegant launcher: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse arguments
    config = parse_arguments(sys.argv[1:])

    # Show help if requested
    if config["help"]:
        print_help()
        return

    # Set up environment before any imports
    setup_environment(config)

    # Show banner if appropriate
    if config["verbose"] and not config["quiet"]:
        show_elegant_banner(config["transport"], config["config_file"])

    # Run the server
    run_server(config)


if __name__ == "__main__":
    main()