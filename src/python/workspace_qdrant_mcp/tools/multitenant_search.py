"""
Multi-tenant search tools with project metadata filtering.

This module extends the existing search capabilities to support multi-tenant
workspace collections with automatic project context filtering. It maintains
backward compatibility while adding project isolation features.

Key Features:
    - Automatic project context detection and filtering
    - Workspace collection type-aware search
    - Metadata-based tenant isolation
    - Cross-project search capabilities when authorized
    - Enhanced search result enrichment with project context
"""

from typing import Dict, List, Optional, Union
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from python.common.core.client import QdrantWorkspaceClient
from python.common.core.multitenant_collections import (
    MultiTenantWorkspaceCollectionManager,
    ProjectIsolationManager,
    WorkspaceCollectionRegistry
)
from python.common.core.hybrid_search import HybridSearchEngine
from .search import search_workspace as base_search_workspace


class MultiTenantSearchEngine:
    """Enhanced search engine with multi-tenant project isolation."""

    def __init__(self, client: QdrantWorkspaceClient):
        self.client = client
        self.isolation_manager = ProjectIsolationManager()
        self.registry = WorkspaceCollectionRegistry()

    async def search_workspace_with_project_context(
        self,
        query: str,
        project_name: Optional[str] = None,
        workspace_types: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
        include_shared: bool = True,
        cross_project_search: bool = False
    ) -> Dict:
        """
        Search workspace collections with automatic project context filtering.

        Args:
            query: Natural language search query
            project_name: Project context for filtering (auto-detected if None)
            workspace_types: Specific workspace types to search (notes, docs, etc.)
            collections: Specific collections to search (overrides project filtering)
            mode: Search mode (hybrid, dense, sparse)
            limit: Maximum number of results
            score_threshold: Minimum relevance score
            include_shared: Include shared workspace collections
            cross_project_search: Allow searching across multiple projects

        Returns:
            Dict: Enhanced search results with project context
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(self.client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            if not project_name and not cross_project_search:
                return {"error": "No project context available and cross-project search disabled"}

            # Determine target collections
            target_collections = []

            if collections:
                # Use explicitly specified collections
                target_collections = collections
            elif cross_project_search:
                # Search all workspace collections
                target_collections = self._get_all_workspace_collections()
            else:
                # Filter by project context
                target_collections = self._get_project_collections(
                    project_name, workspace_types, include_shared
                )

            if not target_collections:
                return {
                    "query": query,
                    "mode": mode,
                    "project_name": project_name,
                    "workspace_types": workspace_types,
                    "collections_searched": [],
                    "total_results": 0,
                    "results": [],
                    "message": "No accessible collections found for search"
                }

            # Perform the search using enhanced search functionality with multi-tenant aggregation
            search_result = await base_search_workspace(
                client=self.client,
                query=query,
                collections=target_collections,
                mode=mode,
                limit=limit,
                score_threshold=score_threshold,
                enable_multi_tenant_aggregation=True,
                enable_deduplication=True,
                score_aggregation_method="max_score"
            )

            # Enhance results with project context
            if "results" in search_result:
                search_result["results"] = self._enrich_search_results(
                    search_result["results"],
                    project_name,
                    workspace_types
                )

            # Add project context information
            search_result.update({
                "project_name": project_name,
                "workspace_types": workspace_types,
                "include_shared": include_shared,
                "cross_project_search": cross_project_search
            })

            return search_result

        except Exception as e:
            logger.error(f"Multi-tenant search failed: {e}")
            return {"error": f"Multi-tenant search failed: {e}"}

    async def search_workspace_by_metadata(
        self,
        metadata_filter: Dict,
        project_name: Optional[str] = None,
        workspace_types: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        limit: int = 10,
        include_shared: bool = True
    ) -> Dict:
        """
        Search workspace collections by metadata with project filtering.

        Args:
            metadata_filter: Base metadata filter conditions
            project_name: Project context for filtering
            workspace_types: Specific workspace types to search
            collections: Specific collections to search
            limit: Maximum number of results
            include_shared: Include shared workspace collections

        Returns:
            Dict: Search results with project metadata filtering
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(self.client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            # Build enhanced metadata filter with project context
            enhanced_filter = self._build_project_metadata_filter(
                base_filter=metadata_filter,
                project_name=project_name,
                workspace_types=workspace_types,
                include_shared=include_shared
            )

            # Determine target collections
            if collections:
                target_collections = collections
            else:
                target_collections = self._get_project_collections(
                    project_name, workspace_types, include_shared
                )

            if not target_collections:
                return {
                    "filter": metadata_filter,
                    "project_name": project_name,
                    "collections_searched": [],
                    "total_results": 0,
                    "results": [],
                    "message": "No accessible collections found"
                }

            # Execute search across collections
            all_results = []
            collections_searched = []

            for collection_name in target_collections:
                try:
                    # Validate collection exists
                    available_collections = self.client.list_collections()
                    if collection_name not in available_collections:
                        logger.warning(f"Collection {collection_name} not found, skipping")
                        continue

                    # Search collection with enhanced filter
                    results = self.client.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=enhanced_filter,
                        limit=limit,
                        with_payload=True,
                    )

                    # Process results
                    for result in results[0]:  # results is (points, next_page_offset)
                        all_results.append({
                            "id": result.id,
                            "payload": result.payload,
                            "collection": collection_name,
                            "project_name": result.payload.get("project_name"),
                            "collection_type": result.payload.get("collection_type"),
                            "workspace_scope": result.payload.get("workspace_scope")
                        })

                    collections_searched.append(collection_name)

                except Exception as e:
                    logger.error(f"Search failed for collection {collection_name}: {e}")
                    continue

            # Sort by relevance if multiple collections
            if len(collections_searched) > 1:
                all_results.sort(key=lambda x: x.get("payload", {}).get("priority", 3), reverse=True)

            # Apply global limit
            all_results = all_results[:limit]

            return {
                "filter": metadata_filter,
                "enhanced_filter": self._filter_to_dict(enhanced_filter),
                "project_name": project_name,
                "workspace_types": workspace_types,
                "collections_searched": collections_searched,
                "total_results": len(all_results),
                "results": all_results
            }

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return {"error": f"Metadata search failed: {e}"}

    def _get_all_workspace_collections(self) -> List[str]:
        """Get all workspace collections across all projects."""
        try:
            all_collections = self.client.list_collections()
            workspace_collections = []

            for collection_name in all_collections:
                # Check if collection follows workspace naming pattern
                for workspace_type in self.registry.get_workspace_types():
                    if collection_name.endswith(f"-{workspace_type}"):
                        workspace_collections.append(collection_name)
                        break

            return workspace_collections
        except Exception as e:
            logger.error(f"Failed to get all workspace collections: {e}")
            return []

    def _get_project_collections(
        self,
        project_name: Optional[str],
        workspace_types: Optional[List[str]],
        include_shared: bool = True
    ) -> List[str]:
        """Get collections for a specific project context using enhanced selector."""
        if not project_name:
            return []

        try:
            # Use enhanced collection selector for better multi-tenant support
            from python.common.core.collections import CollectionSelector

            project_detector = getattr(self.client, 'project_detector', None)
            collection_selector = CollectionSelector(
                self.client.client, self.client.config, project_detector
            )

            # Get code collections for the project
            selection_result = collection_selector.select_collections_by_type(
                'code_collection',
                project_name=project_name,
                workspace_types=workspace_types,
                include_shared=include_shared
            )

            # Combine code collections and shared collections
            project_collections = []
            project_collections.extend(selection_result.get('code_collections', []))
            project_collections.extend(selection_result.get('shared_collections', []))

            # Apply fallback if needed
            if not project_collections:
                project_collections = selection_result.get('fallback_collections', [])

            logger.debug(
                f"Enhanced project collection selection",
                project_name=project_name,
                workspace_types=workspace_types,
                selected_count=len(project_collections),
                collections=project_collections[:3]  # Log first 3 for debugging
            )

            return project_collections

        except Exception as e:
            logger.error(f"Enhanced project collection selection failed: {e}")
            # Fallback to original logic
            return self._get_project_collections_fallback(
                project_name, workspace_types, include_shared
            )

    def _get_project_collections_fallback(
        self,
        project_name: Optional[str],
        workspace_types: Optional[List[str]],
        include_shared: bool = True
    ) -> List[str]:
        """Original project collection selection logic as fallback."""
        try:
            all_collections = self.client.list_collections()
            project_collections = []

            # Target workspace types
            target_types = workspace_types or list(self.registry.get_workspace_types())

            for workspace_type in target_types:
                # Direct project collection
                project_collection = f"{project_name}-{workspace_type}"
                if project_collection in all_collections:
                    project_collections.append(project_collection)

                # Include shared collections if requested
                if include_shared and workspace_type == "scratchbook":
                    # Scratchbook is typically shared across projects
                    if "scratchbook" in all_collections:
                        project_collections.append("scratchbook")

            return project_collections

        except Exception as e:
            logger.error(f"Failed to get project collections for {project_name}: {e}")
            return []

    def _build_project_metadata_filter(
        self,
        base_filter: Dict,
        project_name: Optional[str],
        workspace_types: Optional[List[str]],
        include_shared: bool = True
    ) -> models.Filter:
        """Build Qdrant filter with project metadata constraints."""
        conditions = []

        # Add base filter conditions
        for key, value in base_filter.items():
            if isinstance(value, str):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            elif isinstance(value, (int, float)):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            elif isinstance(value, list):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=value))
                )

        # Add project context constraints
        if project_name:
            project_conditions = [
                models.FieldCondition(
                    key="project_name",
                    match=models.MatchValue(value=project_name)
                )
            ]

            # Include shared workspace scope if requested
            if include_shared:
                scope_condition = models.Filter(
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
                project_conditions.append(scope_condition)

            conditions.extend(project_conditions)

        # Add workspace type constraints
        if workspace_types:
            if len(workspace_types) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="collection_type",
                        match=models.MatchValue(value=workspace_types[0])
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="collection_type",
                        match=models.MatchAny(any=workspace_types)
                    )
                )

        return models.Filter(must=conditions)

    def _enrich_search_results(
        self,
        results: List[Dict],
        project_name: Optional[str],
        workspace_types: Optional[List[str]]
    ) -> List[Dict]:
        """Enrich search results with project context information."""
        enriched_results = []

        for result in results:
            enriched_result = result.copy()

            # Extract project context from payload
            payload = result.get("payload", {})

            enriched_result.update({
                "project_context": {
                    "project_name": payload.get("project_name"),
                    "collection_type": payload.get("collection_type"),
                    "workspace_scope": payload.get("workspace_scope"),
                    "tenant_namespace": payload.get("tenant_namespace")
                },
                "access_info": {
                    "created_by": payload.get("created_by"),
                    "access_level": payload.get("access_level"),
                    "team_access": payload.get("team_access", [])
                },
                "metadata_enriched": True
            })

            enriched_results.append(enriched_result)

        return enriched_results

    def _filter_to_dict(self, qdrant_filter: models.Filter) -> Dict:
        """Convert Qdrant filter to dictionary for logging/debugging."""
        try:
            return qdrant_filter.dict() if qdrant_filter else {}
        except Exception:
            return {"error": "Could not serialize filter"}


# Convenience functions for backward compatibility and easy access

async def search_workspace_with_project_context(
    client: QdrantWorkspaceClient,
    query: str,
    project_name: Optional[str] = None,
    workspace_types: Optional[List[str]] = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7,
    include_shared: bool = True
) -> Dict:
    """
    Search workspace collections with automatic project context filtering.

    Convenience function that creates a MultiTenantSearchEngine and performs
    a project-aware search.
    """
    search_engine = MultiTenantSearchEngine(client)
    return await search_engine.search_workspace_with_project_context(
        query=query,
        project_name=project_name,
        workspace_types=workspace_types,
        mode=mode,
        limit=limit,
        score_threshold=score_threshold,
        include_shared=include_shared
    )


async def search_workspace_by_metadata_with_project_context(
    client: QdrantWorkspaceClient,
    metadata_filter: Dict,
    project_name: Optional[str] = None,
    workspace_types: Optional[List[str]] = None,
    limit: int = 10,
    include_shared: bool = True
) -> Dict:
    """
    Search workspace collections by metadata with project filtering.

    Convenience function for metadata-based search with project context.
    """
    search_engine = MultiTenantSearchEngine(client)
    return await search_engine.search_workspace_by_metadata(
        metadata_filter=metadata_filter,
        project_name=project_name,
        workspace_types=workspace_types,
        limit=limit,
        include_shared=include_shared
    )