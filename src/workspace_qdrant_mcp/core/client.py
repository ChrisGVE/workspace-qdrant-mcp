"""
Qdrant workspace client for project-scoped collections.

Main client class for managing Qdrant collections and operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import Config
from .collections import WorkspaceCollectionManager

logger = logging.getLogger(__name__)


class QdrantWorkspaceClient:
    """
    Main client for workspace-scoped Qdrant operations.
    
    Manages project-scoped collections with scratchbook functionality.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection_manager: Optional[WorkspaceCollectionManager] = None
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the Qdrant client and workspace collections."""
        if self.initialized:
            return
            
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(**self.config.qdrant_client_config)
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            
            logger.info("Connected to Qdrant at %s", self.config.qdrant.url)
            
            # Initialize collection manager
            self.collection_manager = WorkspaceCollectionManager(self.client, self.config)
            
            # Initialize workspace collections
            # Note: Project detection will be added in Task 3
            await self.collection_manager.initialize_workspace_collections(
                project_name="default",  # Placeholder until Task 3 is complete
                subprojects=[]
            )
            
            self.initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize Qdrant client: %s", e)
            raise
    
    async def get_status(self) -> Dict:
        """Get workspace and collection status information."""
        if not self.initialized:
            return {"error": "Client not initialized"}
            
        try:
            # Get basic Qdrant info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            
            workspace_collections = await self.collection_manager.list_workspace_collections()
            collection_info = await self.collection_manager.get_collection_info()
            
            return {
                "connected": True,
                "qdrant_url": self.config.qdrant.url,
                "collections_count": len(info.collections),
                "workspace_collections": workspace_collections,
                "current_project": "default",  # TODO: Implement in Task 3
                "collection_info": collection_info,
                "config": {
                    "embedding_model": self.config.embedding.model,
                    "sparse_vectors_enabled": self.config.embedding.enable_sparse_vectors,
                    "global_collections": self.config.workspace.global_collections,
                }
            }
            
        except Exception as e:
            logger.error("Failed to get status: %s", e)
            return {"error": f"Failed to get status: {e}"}
    
    async def list_collections(self) -> List[str]:
        """List all available workspace collections."""
        if not self.initialized:
            return []
            
        try:
            return await self.collection_manager.list_workspace_collections()
            
        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []
    
    async def close(self) -> None:
        """Clean up client connections."""
        if self.client:
            self.client.close()
            self.client = None
        self.initialized = False