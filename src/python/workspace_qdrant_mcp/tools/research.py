"""
Four-mode research interface implementation.

This module implements the core research functionality for the workspace-qdrant-mcp
system, supporting four distinct research modes as specified in PRD v2.0.

Research Modes:
- 'project': Search within current project and its subprojects (default)
- 'collection': Search within a specific collection
- 'global': Search across global knowledge collections
- 'all': Search across all available collections

Features:
- Version-aware filtering with latest/all preferences
- Relationship information inclusion for version tracking
- Archived collection support
- Configurable result limits and score thresholds

This implementation is designed to be called by MCP tools while remaining
testable as a standalone function.
"""

from typing import Any

from loguru import logger

from common.core.client import QdrantWorkspaceClient
from common.core.collection_naming import build_project_collection_name
from .search import search_workspace

# logger imported from loguru


async def research_workspace(
    client: QdrantWorkspaceClient,
    query: str,
    mode: str = "project",
    target_collection: str = None,
    include_relationships: bool = False,
    version_preference: str = "latest",
    include_archived: bool = False,
    limit: int = 10,
    score_threshold: float = 0.7,
) -> dict:
    """
    Four-mode research interface for workspace collections.

    Provides comprehensive search capabilities across different contexts
    within the workspace, with version-aware filtering and relationship
    tracking as specified in PRD v2.0.

    Args:
        client: Initialized workspace client
        query: Natural language search query
        mode: Research mode - 'project', 'collection', 'global', or 'all'
        target_collection: Required for 'collection' mode - specific collection name
        include_relationships: Include version relationship information in results
        version_preference: 'latest' to filter to latest versions only, 'all' for everything
        include_archived: Include archived collections in 'all' mode
        limit: Maximum number of results to return
        score_threshold: Minimum relevance score threshold

    Returns:
        Dict containing:
        - results: List of search results with relevance scores
        - total_results: Number of results returned
        - research_context: Context information about the search
        - error: Error message if search failed

    Raises:
        None - All errors are captured and returned in the response dict
    """

    # Validate client initialization
    if not client or not client.initialized:
        return {"error": "Workspace client not initialized"}

    # Validate query
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}

    # Validate mode
    valid_modes = ["project", "collection", "global", "all"]
    if mode not in valid_modes:
        return {"error": f"Invalid mode '{mode}'. Must be one of: {valid_modes}"}

    # Validate collection mode requirements
    if mode == "collection" and not target_collection:
        return {"error": "target_collection required for collection mode"}

    try:
        # Determine collections to search based on mode
        collections = []

        if mode == "project":
            # Search current project and subprojects
            project_info = getattr(client, "project_info", {})
            main_project = project_info.get("main_project", "default")
            subprojects = project_info.get("subprojects", [])

            # Add main project collections
            collections.extend([
                build_project_collection_name(main_project, "scratchbook"),
                build_project_collection_name(main_project, "docs")
            ])

            # Add subproject collections
            for subproject in subprojects:
                collections.extend([
                    build_project_collection_name(subproject, "scratchbook"),
                    build_project_collection_name(subproject, "docs")
                ])

        elif mode == "collection":
            # Search specific collection only
            collections = [target_collection]

        elif mode == "global":
            # Search global knowledge collections
            workspace_config = getattr(client.config, "workspace", None)
            if workspace_config:
                global_collections = getattr(
                    workspace_config, "global_collections", None
                )
                if global_collections:
                    collections = global_collections
                else:
                    # Default global collections if not configured - use display names
                    collections = ["memory", "technical-books", "standards"]
            else:
                collections = ["memory", "technical-books", "standards"]

        elif mode == "all":
            # Search all available collections
            if hasattr(client.collection_manager, 'list_searchable_collections') and not include_archived:
                # Use searchable collections by default (excludes system collections)
                collections = client.collection_manager.list_searchable_collections()
            else:
                # Fallback to full collection list
                collections = client.list_collections()

                # Optionally filter out archived collections
                if not include_archived:
                    collections = [c for c in collections if not c.endswith("_archive")]

        # Perform the search
        search_results = await search_workspace(
            client=client,
            query=query,
            collections=collections,
            mode="hybrid",  # Always use hybrid search for research
            limit=limit,
            score_threshold=score_threshold,
        )

        # Check for search errors
        if "error" in search_results:
            return {"error": f"Search failed: {search_results['error']}"}

        results = search_results.get("results", [])

        # Apply version filtering if requested
        if version_preference == "latest":
            filtered_results = []
            for result in results:
                payload = result.get("payload", {})
                # Default to latest if no version info
                is_latest = payload.get("is_latest", True)
                if is_latest:
                    filtered_results.append(result)
            results = filtered_results

        # Add relationship information if requested
        if include_relationships:
            for result in results:
                payload = result.get("payload", {})
                version_info = {}

                # Extract version-related metadata
                if "version" in payload:
                    version_info["version"] = payload["version"]
                if "is_latest" in payload:
                    version_info["is_latest"] = payload["is_latest"]
                if "supersedes" in payload:
                    version_info["supersedes"] = payload["supersedes"]
                if "document_type" in payload:
                    version_info["document_type"] = payload["document_type"]

                if version_info:
                    result["version_info"] = version_info

        # Build research context
        research_context = {
            "mode": mode,
            "collections_searched": collections,
            "version_preference": version_preference,
            "include_archived": include_archived,
        }

        if mode == "collection":
            research_context["target_collection"] = target_collection

        return {
            "results": results,
            "total_results": len(results),
            "research_context": research_context,
            "query": query,
            "search_params": {
                "limit": limit,
                "score_threshold": score_threshold,
                "include_relationships": include_relationships,
            },
        }

    except Exception as e:
        logger.error("Research failed: %s", e)
        return {"error": f"Research failed: {str(e)}"}
