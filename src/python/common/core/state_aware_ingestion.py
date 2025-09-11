"""State-aware ingestion manager module."""

from typing import Optional

from .auto_ingestion import AutoIngestionConfig, AutoIngestionManager
from .client import QdrantWorkspaceClient
from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager

# Global state-aware ingestion manager instance
_ingestion_manager: Optional[AutoIngestionManager] = None


async def get_ingestion_manager(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    config: Optional[AutoIngestionConfig] = None,
    state_db_path: str = "workspace_ingestion_state.db"
) -> AutoIngestionManager:
    """Get or create global state-aware ingestion manager."""
    global _ingestion_manager
    
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