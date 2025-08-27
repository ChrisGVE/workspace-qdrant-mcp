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
from .embeddings import EmbeddingService
from ..utils.project_detection import ProjectDetector

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
        self.embedding_service = EmbeddingService(config)
        self.project_detector = ProjectDetector(github_user=self.config.workspace.github_user)
        self.project_info: Optional[Dict] = None
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
            
            # Detect current project and subprojects
            self.project_info = self.project_detector.get_project_info()
            logger.info("Detected project: %s with subprojects: %s", 
                       self.project_info["main_project"], 
                       self.project_info["subprojects"])
            
            # Initialize collection manager
            self.collection_manager = WorkspaceCollectionManager(self.client, self.config)
            
            # Initialize embedding service
            await self.embedding_service.initialize()
            logger.info("Embedding service initialized")
            
            # Initialize workspace collections with detected project info
            await self.collection_manager.initialize_workspace_collections(
                project_name=self.project_info["main_project"],
                subprojects=self.project_info["subprojects"]
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
                "current_project": self.project_info["main_project"] if self.project_info else None,
                "project_info": self.project_info,
                "collection_info": collection_info,
                "embedding_info": self.embedding_service.get_model_info(),
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
    
    def get_project_info(self) -> Optional[Dict]:
        """Get current project information."""
        return self.project_info
    
    def refresh_project_detection(self) -> Dict:
        """Refresh project detection from current directory."""
        self.project_info = self.project_detector.get_project_info()
        return self.project_info
    
    def get_embedding_service(self) -> EmbeddingService:
        """Get the embedding service instance."""
        return self.embedding_service
    
    async def close(self) -> None:
        """Clean up client connections."""
        if self.embedding_service:
            await self.embedding_service.close()
        if self.client:
            self.client.close()
            self.client = None
        self.initialized = False