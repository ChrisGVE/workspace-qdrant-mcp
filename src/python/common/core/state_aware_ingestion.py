"""State-aware ingestion manager module."""

from __future__ import annotations

import os
import sys


# Check for stdio mode to prevent problematic imports
def _is_stdio_mode() -> bool:
    """Check if running in MCP stdio mode."""
    return (
        os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_QUIET_MODE", "").lower() == "true" or
        os.getenv("MCP_TRANSPORT") == "stdio" or
        (hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty())
    )

# Conditional imports to prevent hanging in stdio mode
if not _is_stdio_mode():
    from .auto_ingestion import AutoIngestionConfig, AutoIngestionManager
else:
    # Provide dummy classes for stdio mode
    AutoIngestionConfig = None
    AutoIngestionManager = None

from .client import QdrantWorkspaceClient

# Conditional import for watch management
if not _is_stdio_mode():
    from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager
else:
    WatchToolsManager = None

# Global state-aware ingestion manager instance
_ingestion_manager: AutoIngestionManager | None = None


async def get_ingestion_manager(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager | None,
    config: AutoIngestionConfig | None = None,
    state_db_path: str = "workspace_ingestion_state.db"
) -> AutoIngestionManager | None:
    """Get or create global state-aware ingestion manager."""
    global _ingestion_manager

    # Return None in stdio mode where auto-ingestion is disabled
    if AutoIngestionManager is None or watch_manager is None:
        return None

    if _ingestion_manager is None:
        _ingestion_manager = AutoIngestionManager(
            workspace_client=workspace_client,
            watch_manager=watch_manager,
            config=config
        )

        # AutoIngestionManager doesn't need explicit initialization
        # It's ready to use after construction

    return _ingestion_manager


async def shutdown_ingestion_manager():
    """Shutdown global ingestion manager."""
    global _ingestion_manager

    # AutoIngestionManager doesn't need explicit shutdown
    # Just clear the reference
    if _ingestion_manager:
        _ingestion_manager = None
