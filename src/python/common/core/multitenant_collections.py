"""
Multi-tenant workspace collections with metadata-based project isolation.

This module extends the existing collection management system to support multi-tenancy
through metadata-based filtering while maintaining backward compatibility with the
existing collection architecture.

Key Features:
    - Metadata-based project isolation for workspace collections
    - Support for workspace collection types (notes, docs, scratchbook)
    - Automatic metadata injection for project context
    - Efficient metadata filtering for search operations
    - Backward compatibility with existing collection naming

Collection Types Supported:
    - notes: Project notes and documentation
    - docs: Formal project documentation
    - scratchbook: Cross-project scratchbook and ideas
    - knowledge: Knowledge base and reference materials
    - context: Contextual information and state
    - memory: Persistent memory and learned patterns
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..utils.project_detection import ProjectDetector

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .collections import CollectionConfig, WorkspaceCollectionManager
from .config import ConfigManager


@dataclass
class ProjectMetadata:
    """Metadata schema for multi-tenant project isolation."""

    # Core tenant isolation fields
    project_id: str           # Unique project identifier
    project_name: str         # Human-readable project name
    tenant_namespace: str     # Namespace for tenant isolation

    # Collection categorization
    collection_type: str      # notes, docs, scratchbook, etc.
    workspace_scope: str      # project, global, shared

    # Access control metadata
    created_by: str = "system"
    access_level: str = "private"  # public, private, shared
    team_access: list[str] = field(default_factory=list)

    # Temporal metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1

    # Organizational metadata
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    priority: int = 3

    def to_dict(self) -> dict:
        """Convert to dictionary for Qdrant metadata storage."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "tenant_namespace": self.tenant_namespace,
            "collection_type": self.collection_type,
            "workspace_scope": self.workspace_scope,
            "created_by": self.created_by,
            "access_level": self.access_level,
            "team_access": self.team_access,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "tags": self.tags,
            "category": self.category,
            "priority": self.priority
        }

    @staticmethod
    def create_project_metadata(
        project_name: str,
        collection_type: str,
        creator: str = "system",
        access_level: str = "private",
        workspace_scope: str = "project"
    ) -> "ProjectMetadata":
        """Factory method for creating project metadata."""
        # Generate stable project ID from name
        project_id = hashlib.sha256(project_name.encode()).hexdigest()[:12]

        # Create tenant namespace for isolation
        tenant_namespace = f"{project_name}.{collection_type}"

        return ProjectMetadata(
            project_id=project_id,
            project_name=project_name,
            tenant_namespace=tenant_namespace,
            collection_type=collection_type,
            workspace_scope=workspace_scope,
            created_by=creator,
            access_level=access_level
        )


@dataclass
class MultiTenantCollectionConfig:
    """Configuration for multi-tenant workspace collections."""

    # Collection specification
    name: str                        # Physical collection name
    display_name: str                # User-friendly name
    collection_type: str             # Workspace collection type

    # Multi-tenancy settings
    isolation_strategy: str = "metadata"  # metadata, collection, hybrid
    enable_cross_tenant_search: bool = False
    default_project_metadata: ProjectMetadata | None = None

    # Performance settings
    enable_metadata_indexing: bool = True
    indexed_metadata_fields: list[str] = field(default_factory=lambda: [
        "project_name", "tenant_namespace", "collection_type",
        "workspace_scope", "created_by", "access_level"
    ])


class WorkspaceCollectionRegistry:
    """Registry for managing workspace collection types and their schemas."""

    def __init__(self):
        self.collection_schemas = {
            "notes": {
                "description": "Project notes and documentation",
                "default_metadata": {"category": "notes", "priority": 3},
                "searchable": True,
                "supports_tags": True,
                "workspace_scope": "project"
            },
            "docs": {
                "description": "Formal project documentation",
                "default_metadata": {"category": "documentation", "priority": 4},
                "searchable": True,
                "supports_versioning": True,
                "workspace_scope": "project"
            },
            "scratchbook": {
                "description": "Cross-project scratchbook and ideas",
                "default_metadata": {"category": "scratchbook", "priority": 2},
                "searchable": True,
                "workspace_scope": "shared"  # Can be shared across projects
            },
            "knowledge": {
                "description": "Knowledge base and reference materials",
                "default_metadata": {"category": "knowledge", "priority": 5},
                "searchable": True,
                "supports_cross_reference": True,
                "workspace_scope": "project"
            },
            "context": {
                "description": "Contextual information and state",
                "default_metadata": {"category": "context", "priority": 3},
                "searchable": True,
                "temporal": True,
                "workspace_scope": "project"
            },
            "memory": {
                "description": "Persistent memory and learned patterns",
                "default_metadata": {"category": "memory", "priority": 4},
                "searchable": False,  # Memory collections have special search logic
                "system_managed": True,
                "workspace_scope": "project"
            }
        }

    def get_workspace_types(self) -> set[str]:
        """Get all supported workspace collection types."""
        return set(self.collection_schemas.keys())

    def is_multi_tenant_type(self, collection_type: str) -> bool:
        """Check if collection type supports multi-tenancy."""
        return collection_type in self.collection_schemas

    def get_default_metadata(self, collection_type: str, project_name: str) -> dict:
        """Get default metadata for a workspace collection type."""
        if collection_type not in self.collection_schemas:
            return {}

        schema = self.collection_schemas[collection_type]
        base_metadata = schema.get("default_metadata", {}).copy()

        # Add project isolation metadata
        project_metadata = ProjectMetadata.create_project_metadata(
            project_name=project_name,
            collection_type=collection_type,
            workspace_scope=schema.get("workspace_scope", "project")
        )

        base_metadata.update(project_metadata.to_dict())
        return base_metadata

    def is_searchable(self, collection_type: str) -> bool:
        """Check if collection type is searchable."""
        return self.collection_schemas.get(collection_type, {}).get("searchable", True)


class ProjectIsolationManager:
    """Manager for project-based tenant isolation through metadata filtering."""

    def __init__(self, project_detector: Optional["ProjectDetector"] = None):
        self.project_detector = project_detector
        self._tenant_metadata_cache = {}

    def create_project_filter(self, project_name: str) -> models.Filter:
        """Create Qdrant filter for project isolation."""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="project_name",
                    match=models.MatchValue(value=project_name)
                )
            ]
        )

    def create_workspace_filter(
        self,
        project_name: str,
        collection_type: str | None = None,
        include_shared: bool = True
    ) -> models.Filter:
        """Create filter for specific workspace type within project."""
        conditions = [
            models.FieldCondition(
                key="project_name",
                match=models.MatchValue(value=project_name)
            )
        ]

        # Add collection type filter if specified
        if collection_type:
            conditions.append(
                models.FieldCondition(
                    key="collection_type",
                    match=models.MatchValue(value=collection_type)
                )
            )

        # Include shared workspace collections if requested
        if include_shared:
            workspace_scope_condition = models.Filter(
                should=[
                    models.FieldCondition(
                        key="workspace_scope",
                        match=models.MatchValue(value="project")
                    ),
                    models.FieldCondition(
                        key="workspace_scope",
                        match=models.MatchValue(value="shared")
                    )
                ]
            )
            conditions.append(workspace_scope_condition)

        return models.Filter(must=conditions)

    def create_tenant_namespace_filter(self, tenant_namespace: str) -> models.Filter:
        """Create filter for specific tenant namespace."""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_namespace",
                    match=models.MatchValue(value=tenant_namespace)
                )
            ]
        )

    def get_tenant_metadata(
        self,
        project_name: str,
        collection_type: str,
        creator: str = "system"
    ) -> ProjectMetadata:
        """Get or create tenant metadata for project/collection combination."""
        cache_key = f"{project_name}:{collection_type}"

        if cache_key not in self._tenant_metadata_cache:
            metadata = ProjectMetadata.create_project_metadata(
                project_name=project_name,
                collection_type=collection_type,
                creator=creator
            )
            self._tenant_metadata_cache[cache_key] = metadata

        return self._tenant_metadata_cache[cache_key]

    def enrich_document_metadata(
        self,
        base_metadata: dict,
        project_name: str,
        collection_type: str
    ) -> dict:
        """Enrich document metadata with project isolation fields."""
        enriched_metadata = base_metadata.copy()

        # Get project metadata
        project_metadata = self.get_tenant_metadata(project_name, collection_type)

        # Merge project isolation metadata
        enriched_metadata.update(project_metadata.to_dict())

        # Update timestamps
        enriched_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        return enriched_metadata


class MultiTenantWorkspaceCollectionManager(WorkspaceCollectionManager):
    """
    Extended collection manager with multi-tenant workspace support.

    This class extends the existing WorkspaceCollectionManager to support
    metadata-based multi-tenancy while maintaining backward compatibility.
    """

    def __init__(self, client: QdrantClient, config: ConfigManager):
        """Initialize the multi-tenant collection manager."""
        super().__init__(client, config)

        # Multi-tenant components
        self.registry = WorkspaceCollectionRegistry()
        self.isolation_manager = ProjectIsolationManager()
        self._multitenant_collections = {}

    async def create_workspace_collection(
        self,
        project_name: str,
        collection_type: str,
        enable_metadata_indexing: bool = True
    ) -> dict:
        """
        Create a workspace collection with multi-tenant metadata support.

        Args:
            project_name: Project name for tenant isolation
            collection_type: Workspace collection type (notes, docs, etc.)
            enable_metadata_indexing: Whether to create metadata indexes

        Returns:
            Dict: Creation result with success status and details
        """
        if not self.registry.is_multi_tenant_type(collection_type):
            return {
                "success": False,
                "error": f"Unsupported workspace collection type: {collection_type}"
            }

        try:
            # Generate collection name using existing naming strategy
            collection_name = f"{project_name}-{collection_type}"

            # Check if collection already exists
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Workspace collection already exists: {collection_name}")
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "message": "Collection already exists",
                    "existing": True
                }
            except Exception:
                # Collection doesn't exist, continue with creation
                pass

            # Create collection configuration
            config = CollectionConfig(
                name=collection_name,
                description=f"{collection_type.title()} for {project_name}",
                collection_type=collection_type,
                project_name=project_name,
                vector_size=self.config.get("embedding.vector_size", 384),
                distance_metric="Cosine",
                enable_sparse_vectors=True
            )

            # Create the collection
            success = await self._create_single_collection(config)

            if not success:
                return {
                    "success": False,
                    "error": f"Failed to create collection: {collection_name}"
                }

            # Create metadata indexes if requested
            if enable_metadata_indexing:
                await self._create_metadata_indexes(collection_name)

            # Store multi-tenant configuration
            mt_config = MultiTenantCollectionConfig(
                name=collection_name,
                display_name=f"{project_name} {collection_type.title()}",
                collection_type=collection_type,
                default_project_metadata=self.isolation_manager.get_tenant_metadata(
                    project_name, collection_type
                )
            )
            self._multitenant_collections[collection_name] = mt_config

            logger.info(f"Created workspace collection: {collection_name}")
            return {
                "success": True,
                "collection_name": collection_name,
                "collection_type": collection_type,
                "project_name": project_name,
                "metadata_indexed": enable_metadata_indexing
            }

        except Exception as e:
            logger.error(f"Failed to create workspace collection: {e}")
            return {
                "success": False,
                "error": f"Collection creation failed: {e}"
            }

    async def _create_metadata_indexes(self, collection_name: str) -> None:
        """Create metadata indexes for efficient tenant filtering."""
        try:
            # Index fields for efficient filtering
            index_fields = [
                "project_name", "tenant_namespace", "collection_type",
                "workspace_scope", "created_by", "access_level"
            ]

            for field in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=models.KeywordIndexParams()
                    )
                    logger.debug(f"Created index on {field} for collection {collection_name}")
                except Exception as e:
                    # Index might already exist, continue
                    logger.debug(f"Index creation for {field} skipped: {e}")

        except Exception as e:
            logger.warning(f"Failed to create metadata indexes for {collection_name}: {e}")

    async def initialize_workspace_collections(
        self,
        project_name: str,
        subprojects: list[str] | None = None,
        workspace_types: list[str] | None = None
    ) -> dict:
        """
        Initialize workspace collections for a project with multi-tenant support.

        Args:
            project_name: Main project name
            subprojects: Optional list of subproject names
            workspace_types: Optional list of workspace types to create

        Returns:
            Dict: Initialization results
        """
        if workspace_types is None:
            # Default workspace types for project initialization
            workspace_types = ["notes", "docs", "scratchbook"]

        results = {
            "success": True,
            "project_name": project_name,
            "collections_created": [],
            "collections_existing": [],
            "errors": []
        }

        # Store project info for filtering
        self._project_info = {
            "main_project": project_name,
            "subprojects": subprojects or []
        }

        # Create workspace collections for main project
        for workspace_type in workspace_types:
            try:
                result = await self.create_workspace_collection(
                    project_name=project_name,
                    collection_type=workspace_type
                )

                if result["success"]:
                    if result.get("existing", False):
                        results["collections_existing"].append(result["collection_name"])
                    else:
                        results["collections_created"].append(result["collection_name"])
                else:
                    results["errors"].append(f"{workspace_type}: {result['error']}")

            except Exception as e:
                results["errors"].append(f"{workspace_type}: {str(e)}")

        # Create workspace collections for subprojects
        if subprojects:
            for subproject in subprojects:
                for workspace_type in workspace_types:
                    try:
                        result = await self.create_workspace_collection(
                            project_name=subproject,
                            collection_type=workspace_type
                        )

                        if result["success"]:
                            if result.get("existing", False):
                                results["collections_existing"].append(result["collection_name"])
                            else:
                                results["collections_created"].append(result["collection_name"])
                        else:
                            results["errors"].append(f"{subproject}-{workspace_type}: {result['error']}")

                    except Exception as e:
                        results["errors"].append(f"{subproject}-{workspace_type}: {str(e)}")

        # Set success based on whether any errors occurred
        results["success"] = len(results["errors"]) == 0

        logger.info(f"Workspace collections initialized for {project_name}:")
        logger.info(f"  Created: {len(results['collections_created'])}")
        logger.info(f"  Existing: {len(results['collections_existing'])}")
        logger.info(f"  Errors: {len(results['errors'])}")

        return results

    def get_workspace_collection_types(self) -> set[str]:
        """Get all supported workspace collection types."""
        return self.registry.get_workspace_types()

    def is_workspace_collection_type(self, collection_type: str) -> bool:
        """Check if a collection type is a supported workspace type."""
        return self.registry.is_multi_tenant_type(collection_type)

    def get_project_isolation_filter(self, project_name: str, collection_type: str | None = None) -> models.Filter:
        """Get Qdrant filter for project isolation."""
        return self.isolation_manager.create_workspace_filter(
            project_name=project_name,
            collection_type=collection_type,
            include_shared=True
        )

    def enrich_document_metadata(
        self,
        base_metadata: dict,
        project_name: str,
        collection_type: str
    ) -> dict:
        """Enrich document metadata with project isolation fields."""
        return self.isolation_manager.enrich_document_metadata(
            base_metadata, project_name, collection_type
        )
