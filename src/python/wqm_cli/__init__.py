"""
wqm-cli: Rust binary installer for workspace-qdrant-mcp.

This package provides the wqm-install command to compile and install
the Rust-based CLI (wqm) and daemon (memexd) binaries.

The wqm CLI and memexd daemon are high-performance Rust implementations
that replace the previous Python CLI.

Installation:
    wqm-install          # Install/update both binaries
    wqm-install --check  # Check if binaries need updating
    wqm-install --force  # Force rebuild even if up-to-date

After installation:
    wqm --help           # CLI commands
    memexd --help        # Daemon options
"""

__version__ = "0.3.0"
__author__ = "Christian C. Berclaz"
__email__ = "christian.berclaz@mac.com"
__description__ = "Rust binary installer for workspace-qdrant-mcp"
