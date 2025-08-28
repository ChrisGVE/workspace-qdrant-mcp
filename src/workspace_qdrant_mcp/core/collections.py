"""
Collection management for workspace-scoped Qdrant collections.

Handles creation, configuration, and management of project-specific collections.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for a workspace collection."""
    
    name: str
    description: str
    collection_type: str  # 'scratchbook', 'docs', 'global'
    project_name: Optional[str] = None
    vector_size: int = 384  # all-MiniLM-L6-v2 dimension
    distance_metric: str = "Cosine"
    enable_sparse_vectors: bool = True


class WorkspaceCollectionManager:
    """
    Manages project-scoped collections for the workspace.
    
    Handles collection creation, configuration, and lifecycle management.
    """
    
    def __init__(self, client: QdrantClient, config: Config):
        self.client = client
        self.config = config
        self._collections_cache: Optional[Dict[str, CollectionConfig]] = None
        
    async def initialize_workspace_collections(
        self, 
        project_name: str,
        subprojects: List[str] = None
    ) -> None:
        """
        Initialize collections for the current workspace.
        
        Args:
            project_name: Main project name
            subprojects: List of subproject names (optional)
        """
        collections_to_create = []
        
        # Main project collections
        collections_to_create.extend([
            CollectionConfig(
                name=f"{project_name}-scratchbook",
                description=f"Scratchbook notes for {project_name}",
                collection_type="scratchbook",
                project_name=project_name,
                vector_size=self._get_vector_size(),
                enable_sparse_vectors=self.config.embedding.enable_sparse_vectors
            ),
            CollectionConfig(
                name=f"{project_name}-docs",
                description=f"Documentation for {project_name}",
                collection_type="docs", 
                project_name=project_name,
                vector_size=self._get_vector_size(),
                enable_sparse_vectors=self.config.embedding.enable_sparse_vectors
            )
        ])
        
        # Subproject collections
        if subprojects:
            for subproject in subprojects:
                collections_to_create.extend([
                    CollectionConfig(
                        name=f"{subproject}-scratchbook",
                        description=f"Scratchbook notes for {subproject}",
                        collection_type="scratchbook",
                        project_name=subproject,
                        vector_size=self._get_vector_size(),
                        enable_sparse_vectors=self.config.embedding.enable_sparse_vectors
                    ),
                    CollectionConfig(
                        name=f"{subproject}-docs", 
                        description=f"Documentation for {subproject}",
                        collection_type="docs",
                        project_name=subproject,
                        vector_size=self._get_vector_size(),
                        enable_sparse_vectors=self.config.embedding.enable_sparse_vectors
                    )
                ])
        
        # Global collections
        for global_collection in self.config.workspace.global_collections:
            collections_to_create.append(
                CollectionConfig(
                    name=global_collection,
                    description=f"Global {global_collection} collection",
                    collection_type="global",
                    vector_size=self._get_vector_size(),
                    enable_sparse_vectors=self.config.embedding.enable_sparse_vectors
                )
            )
        
        # Create collections in parallel for better performance
        if collections_to_create:
            await asyncio.gather(*[
                self._ensure_collection_exists(config) 
                for config in collections_to_create
            ])
    
    async def _ensure_collection_exists(self, collection_config: CollectionConfig) -> None:
        """
        Ensure a collection exists, creating it if necessary.
        
        Args:
            collection_config: Collection configuration
        """
        try:
            # Check if collection already exists
            existing_collections = self.client.get_collections()
            collection_names = {col.name for col in existing_collections.collections}
            
            if collection_config.name in collection_names:
                logger.info("Collection %s already exists", collection_config.name)
                return
                
            # Create vector configuration
            vectors_config = {
                "dense": models.VectorParams(
                    size=collection_config.vector_size,
                    distance=getattr(models.Distance, collection_config.distance_metric.upper())
                )
            }
            
            # Add sparse vectors if enabled
            if collection_config.enable_sparse_vectors:
                vectors_config["sparse"] = models.SparseVectorParams()
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_config.name,
                vectors_config=vectors_config
            )
            
            # Set collection metadata
            self.client.update_collection(
                collection_name=collection_config.name,
                optimizer_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    memmap_threshold=20000,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                )
            )
            
            logger.info("Created collection: %s", collection_config.name)
            
        except ResponseHandlingException as e:
            logger.error("Failed to create collection %s: %s", collection_config.name, e)
            raise
        except Exception as e:
            logger.error("Unexpected error creating collection %s: %s", collection_config.name, e)
            raise
    
    async def list_workspace_collections(self) -> List[str]:
        """
        List all workspace-related collections.
        
        Returns:
            List of workspace collection names
        """
        try:
            all_collections = self.client.get_collections()
            workspace_collections = []
            
            for collection in all_collections.collections:
                # Filter for workspace collections (exclude memexd -code collections)
                if self._is_workspace_collection(collection.name):
                    workspace_collections.append(collection.name)
                    
            return sorted(workspace_collections)
            
        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []
    
    async def get_collection_info(self) -> Dict:
        """
        Get information about all workspace collections.
        
        Returns:
            Dictionary with collection information
        """
        try:
            workspace_collections = await self.list_workspace_collections()
            collection_info = {}
            
            for collection_name in workspace_collections:
                try:
                    info = self.client.get_collection(collection_name)
                    collection_info[collection_name] = {
                        "vectors_count": info.vectors_count,
                        "points_count": info.points_count,
                        "status": info.status,
                        "optimizer_status": info.optimizer_status,
                        "config": {
                            "distance": info.config.params.vectors.distance,
                            "vector_size": info.config.params.vectors.size,
                        }
                    }
                except Exception as e:
                    logger.warning("Failed to get info for collection %s: %s", collection_name, e)
                    collection_info[collection_name] = {"error": str(e)}
            
            return {
                "collections": collection_info,
                "total_collections": len(workspace_collections)
            }
            
        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
            return {"error": str(e)}
    
    def _is_workspace_collection(self, collection_name: str) -> bool:
        """
        Check if a collection belongs to the workspace.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if it's a workspace collection
        """
        # Exclude memexd daemon collections (those ending with -code)
        if collection_name.endswith("-code"):
            return False
            
        # Include global collections
        if collection_name in self.config.workspace.global_collections:
            return True
            
        # Include project collections (ending with -scratchbook or -docs)
        if collection_name.endswith(("-scratchbook", "-docs")):
            return True
            
        return False
    
    def _get_vector_size(self) -> int:
        """Get the vector size for the current embedding model."""
        # This will be updated when FastEmbed is integrated
        model_sizes = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        
        return model_sizes.get(self.config.embedding.model, 384)