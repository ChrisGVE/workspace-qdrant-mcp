

# Global state-aware ingestion manager instance
_ingestion_manager: Optional[StateAwareIngestionManager] = None


async def get_ingestion_manager(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    config: Optional[AutoIngestionConfig] = None,
    state_db_path: str = "workspace_ingestion_state.db"
) -> StateAwareIngestionManager:
    """Get or create global state-aware ingestion manager."""
    global _ingestion_manager
    
    if _ingestion_manager is None:
        _ingestion_manager = StateAwareIngestionManager(
            workspace_client=workspace_client,
            watch_manager=watch_manager,
            config=config,
            state_db_path=state_db_path
        )
        
        if not await _ingestion_manager.initialize():
            raise RuntimeError("Failed to initialize state-aware ingestion manager")
    
    return _ingestion_manager


async def shutdown_ingestion_manager():
    """Shutdown global ingestion manager."""
    global _ingestion_manager
    
    if _ingestion_manager:
        await _ingestion_manager.shutdown()
        _ingestion_manager = None