"""
Rust CLI wrapper for workspace-qdrant-mcp.

This module provides a Python entry point that delegates to the Rust wqm binary
for improved performance (<100ms startup vs 1-2s for Python CLI).

The wrapper searches for the Rust binary in the following locations:
1. Development build: src/rust/cli/target/release/wqm (relative to project root)
2. System installation: /usr/local/bin/wqm (or platform equivalent)
3. User local bin: ~/.local/bin/wqm
4. System PATH

When the Rust binary is not found, a clear error message with installation
instructions is displayed.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_rust_binary() -> Optional[Path]:
    """
    Search for the Rust wqm binary in standard locations.

    Returns:
        Path to the binary if found, None otherwise.
    """
    # Binary name varies by platform
    binary_name = "wqm.exe" if sys.platform == "win32" else "wqm"

    # 1. Check development build location (relative to this file)
    # This file is at: src/python/workspace_qdrant_mcp/_rust_cli_wrapper.py
    # Rust binary is at: src/rust/cli/target/release/wqm
    module_path = Path(__file__).resolve()

    # Navigate from workspace_qdrant_mcp/ up to project root
    # workspace_qdrant_mcp -> python -> src -> project_root
    project_root = module_path.parent.parent.parent.parent

    dev_binary = project_root / "src" / "rust" / "cli" / "target" / "release" / binary_name
    if dev_binary.exists() and dev_binary.is_file():
        return dev_binary

    # Also check debug build for development
    debug_binary = project_root / "src" / "rust" / "cli" / "target" / "debug" / binary_name
    if debug_binary.exists() and debug_binary.is_file():
        return debug_binary

    # 2. Check platform-specific system locations
    system_paths = []

    if sys.platform == "darwin":  # macOS
        system_paths = [
            Path("/usr/local/bin") / binary_name,
            Path("/opt/homebrew/bin") / binary_name,  # Apple Silicon homebrew
            Path.home() / ".local" / "bin" / binary_name,
        ]
    elif sys.platform == "linux":
        system_paths = [
            Path("/usr/local/bin") / binary_name,
            Path("/usr/bin") / binary_name,
            Path.home() / ".local" / "bin" / binary_name,
            Path.home() / ".cargo" / "bin" / binary_name,  # If installed via cargo
        ]
    elif sys.platform == "win32":
        system_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "wqm" / binary_name,
            Path.home() / ".cargo" / "bin" / binary_name,
        ]

    for path in system_paths:
        if path.exists() and path.is_file():
            return path

    # 3. Check system PATH using shutil.which
    path_binary = shutil.which(binary_name)
    if path_binary:
        return Path(path_binary)

    return None


def print_installation_instructions() -> None:
    """Print instructions for installing the Rust binary."""
    print("Error: wqm Rust binary not found.", file=sys.stderr)
    print("", file=sys.stderr)
    print("The Rust CLI provides faster startup times (<100ms vs 1-2s for Python).", file=sys.stderr)
    print("", file=sys.stderr)
    print("Installation options:", file=sys.stderr)
    print("", file=sys.stderr)
    print("  1. Build from source (development):", file=sys.stderr)
    print("     cd src/rust/cli && cargo build --release", file=sys.stderr)
    print("", file=sys.stderr)
    print("  2. Install to system (after building):", file=sys.stderr)
    if sys.platform == "win32":
        print("     copy target\\release\\wqm.exe %LOCALAPPDATA%\\Programs\\wqm\\", file=sys.stderr)
    else:
        print("     sudo cp target/release/wqm /usr/local/bin/", file=sys.stderr)
        print("     # Or for user-only install:", file=sys.stderr)
        print("     cp target/release/wqm ~/.local/bin/", file=sys.stderr)
    print("", file=sys.stderr)
    print("  3. Use Python CLI directly (slower but always available):", file=sys.stderr)
    print("     wqm-py <command>", file=sys.stderr)
    print("", file=sys.stderr)


def main() -> int:
    """
    Main entry point that delegates to the Rust binary.

    Returns:
        Exit code from the Rust binary, or 1 if binary not found.
    """
    binary_path = find_rust_binary()

    if binary_path is None:
        print_installation_instructions()
        return 1

    # Pass all arguments to the Rust binary
    # sys.argv[0] is the script name, rest are arguments
    try:
        result = subprocess.run(
            [str(binary_path)] + sys.argv[1:],
            # Let the subprocess handle stdin/stdout/stderr directly
            # This allows interactive features like --follow to work
        )
        return result.returncode
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        return 130  # Standard exit code for SIGINT
    except OSError as e:
        print(f"Error: Failed to execute wqm binary: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
