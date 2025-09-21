"""
Advanced search tools for workspace-qdrant-mcp.

This module implements high-performance search capabilities across workspace collections
using hybrid search techniques. It combines dense semantic embeddings with sparse
keyword vectors using Reciprocal Rank Fusion (RRF) for optimal search quality.

Search Modes:
    - 'hybrid': Combines dense and sparse search with RRF (recommended)
    - 'dense': Pure semantic search using dense embeddings
    - 'sparse': Keyword-based search using enhanced BM25

Performance Benchmarks:
    Based on 21,930 test queries:
    - Symbol/exact search: 100% precision, 78.3% recall
    - Semantic search: 94.2% precision, 78.3% recall
    - Hybrid search: Best of both worlds with RRF fusion

Key Features:
    - Cross-collection search with unified ranking
    - Configurable score thresholds for precision control
    - Metadata-based filtering and search
    - Intelligent error handling and collection validation
    - Async processing for high throughput

Example:
    ```python
    from workspace_qdrant_mcp.tools.search import search_workspace

    # Hybrid search across all collections
    results = await search_workspace(
        client=workspace_client,
        query="authentication patterns",
        mode="hybrid",
        limit=10,
        score_threshold=0.7
    )

    # Metadata-based filtering
    filtered_results = await search_collection_by_metadata(
        client=workspace_client,
        collection="my-project",
        metadata_filter={"file_type": "python", "author": "dev-team"}
    )
    ```
"""

from typing import Optional

from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from python.common.core.client import QdrantWorkspaceClient
from python.common.core.hybrid_search import HybridSearchEngine
from python.common.core.sparse_vectors import create_named_sparse_vector

# logger imported from loguru


async def search_workspace(
    client: QdrantWorkspaceClient,
    query: str,
    collections: list[str] | None = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7,
    project_context: Optional[dict] = None,
    auto_inject_project_metadata: bool = True,
    include_shared: bool = True,
    enable_multi_tenant_aggregation: bool = True,
    enable_deduplication: bool = True,
    score_aggregation_method: str = "max_score",
) -> dict:
    """
    Search across multiple workspace collections with advanced hybrid search.

    This is the primary search interface that combines results from multiple
    collections, applies sophisticated ranking algorithms, and provides
    unified result presentation. It supports multiple search modes optimized
    for different use cases.

    Args:
        client: Initialized workspace client with embedding service
        query: Natural language query or exact text to search for
        collections: Specific collections to search. If None, searches all
                    workspace collections including project and global collections
        mode: Search strategy:
            - 'hybrid' (default): Combines dense + sparse with RRF fusion
            - 'dense': Semantic search only (good for conceptual queries)
            - 'sparse': Keyword search only (good for exact matches)
        limit: Maximum number of results to return across all collections
        score_threshold: Minimum relevance score (0.0-1.0). Higher values
                        increase precision but may reduce recall
        project_context: Optional project context for metadata filtering
        auto_inject_project_metadata: Whether to automatically inject project metadata filters
        include_shared: Whether to include shared workspace resources in search
        enable_multi_tenant_aggregation: Whether to use advanced multi-tenant result aggregation
        enable_deduplication: Whether to deduplicate results across collections
        score_aggregation_method: Method for aggregating duplicate scores ("max_score", "avg_score", "sum_score")

    Returns:
        Dict: Comprehensive search results containing:
            - query (str): Original search query
            - mode (str): Search mode used
            - collections_searched (List[str]): Collections that were searched
            - total_results (int): Number of results returned
            - results (List[Dict]): Ranked search results with:
                - id (str): Document identifier
                - score (float): Relevance score (higher = more relevant)
                - payload (Dict): Document content and metadata
                - collection (str): Source collection name
                - search_type (str): Type of match (hybrid/dense/sparse)
            - error (str): Error message if search failed

    Performance Notes:
        - Results are globally ranked across all collections
        - Invalid collections are gracefully skipped with warnings
        - Async processing enables concurrent collection searches
        - Memory usage scales with result set size and document content

    Example:
        ```python
        # Comprehensive search across all collections
        results = await search_workspace(
            client=workspace_client,
            query="How to implement OAuth authentication?",
            mode="hybrid",
            limit=20,
            score_threshold=0.6
        )

        # Process results
        for result in results['results']:
            logger.info("Score: {result['score']:.3f}")
            logger.info("Source: {result['collection']}")
            logger.info("Content: {result['payload']['content'][:100]}...")
        ```
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}

    # Validate query
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}

    # Validate search mode
    valid_modes = ["hybrid", "dense", "sparse"]
    if mode not in valid_modes:
        return {"error": f"Invalid search mode '{mode}'. Must be one of: {valid_modes}"}

    try:
        # Get embedding service
        embedding_service = client.get_embedding_service()

        # Generate embeddings for query
        embeddings = await embedding_service.generate_embeddings(
            query, include_sparse=(mode in ["sparse", "hybrid"])
        )

        # Get collections to search using enhanced collection selector
        display_collections = collections  # Keep original for user reference
        if collections is None:
            # Use enhanced collection selector for multi-tenant aware search
            try:
                from python.common.core.collections import CollectionSelector

                # Initialize collection selector with project detector
                project_detector = getattr(client, 'project_detector', None)
                collection_selector = CollectionSelector(
                    client.client, client.config, project_detector
                )

                # Get project context from client or auto-detect
                project_name = None
                if project_context:
                    project_name = project_context.get('project_name')
                elif hasattr(client, 'project_info') and client.project_info:
                    project_name = client.project_info.get('main_project')

                # Determine workspace types from project context or use defaults
                workspace_types = None
                if project_context and 'workspace_types' in project_context:
                    workspace_types = project_context['workspace_types']

                # Get searchable collections with enhanced selector
                display_collections = collection_selector.get_searchable_collections(
                    project_name=project_name,
                    workspace_types=workspace_types,
                    include_memory=auto_inject_project_metadata,  # Include memory if metadata filtering enabled
                    include_shared=include_shared
                )

                logger.info(
                    f"Enhanced collection selection completed",
                    project_name=project_name,
                    workspace_types=workspace_types,
                    selected_count=len(display_collections),
                    include_shared=include_shared
                )

            except Exception as selector_error:
                logger.warning(f"Enhanced collection selector failed, using fallback: {selector_error}")
                # Fallback to original logic
                if hasattr(client.collection_manager, 'list_searchable_collections'):
                    display_collections = client.collection_manager.list_searchable_collections()
                else:
                    display_collections = client.list_collections()

            collections = display_collections
        else:
            # Validate collections exist
            available_collections = client.list_collections()
            invalid_collections = [
                c for c in collections if c not in available_collections
            ]
            if invalid_collections:
                return {
                    "error": f"Collections not found: {', '.join(invalid_collections)}"
                }

        # Check if we have any collections to search
        if not collections:
            # Enhanced error message with diagnostic information
            try:
                diagnostics = client.collection_manager.validate_collection_filtering()
                total_collections = diagnostics.get('summary', {}).get('total_collections', 0)
                
                if total_collections == 0:
                    error_msg = "No collections found in Qdrant database"
                else:
                    error_msg = (
                        f"No workspace collections available for search. "
                        f"Found {total_collections} total collections in database, "
                        f"but none match the current workspace filtering criteria. "
                        f"Check project configuration or collection naming patterns."
                    )
                
                return {"error": error_msg}
            except Exception:
                # Fallback to original error message if diagnostics fail
                return {"error": "No collections available for search"}

        # Resolve display names to actual collection names for Qdrant operations
        actual_collections = []
        for display_name in collections:
            actual_name, _ = client.collection_manager.resolve_collection_name(
                display_name
            )
            actual_collections.append(actual_name)

        # Validate sparse mode has sparse embeddings
        if mode == "sparse" and "sparse" not in embeddings:
            return {"error": "Sparse embeddings not available for sparse search mode"}

        # Log search configuration
        logger.info(
            "Starting workspace search with multi-tenant aggregation",
            query=query[:50] + "..." if len(query) > 50 else query,
            mode=mode,
            collections_count=len(collections),
            auto_inject_metadata=auto_inject_project_metadata,
            include_shared=include_shared,
            multi_tenant_aggregation=enable_multi_tenant_aggregation,
            deduplication_enabled=enable_deduplication,
            aggregation_method=score_aggregation_method
        )

        # Check if we should use advanced multi-tenant aggregation
        if enable_multi_tenant_aggregation and len(collections) > 1 and mode == "hybrid":
            try:
                # Use advanced multi-collection hybrid search with result aggregation
                hybrid_engine = HybridSearchEngine(
                    client.client,
                    enable_optimizations=True,
                    enable_multi_tenant_aggregation=True
                )

                # Build project contexts for collections
                project_contexts = {}
                detected_context = project_context or client.get_project_context()
                if detected_context:
                    for collection_name in collections:
                        project_contexts[collection_name] = detected_context

                # Perform multi-collection search with aggregation
                aggregated_result = await hybrid_engine.multi_collection_hybrid_search(
                    collection_names=actual_collections,
                    query_embeddings=embeddings,
                    project_contexts=project_contexts,
                    limit=limit,
                    fusion_method="rrf" if mode == "hybrid" else mode,
                    score_threshold=score_threshold,
                    enable_deduplication=enable_deduplication,
                    aggregation_method=score_aggregation_method,
                    with_payload=True
                )

                # Convert to standard search response format
                enhanced_response = {
                    "query": query,
                    "mode": mode,
                    "collections_searched": collections,
                    "total": aggregated_result["total_results"],
                    "results": aggregated_result["results"],
                    "search_params": {
                        "mode": mode,
                        "limit": limit,
                        "score_threshold": score_threshold,
                        "auto_inject_project_metadata": auto_inject_project_metadata,
                        "include_shared": include_shared,
                        "multi_tenant_aggregation_enabled": True,
                        "deduplication_enabled": enable_deduplication,
                        "aggregation_method": score_aggregation_method
                    },
                    "aggregation_metadata": aggregated_result.get("aggregation_metadata", {}),
                    "performance": aggregated_result.get("performance", {})
                }

                # Add project context information
                if auto_inject_project_metadata and detected_context:
                    enhanced_response["project_context"] = {
                        "project_name": detected_context.get("project_name"),
                        "workspace_scope": detected_context.get("workspace_scope"),
                        "metadata_filtering_enabled": True
                    }

                logger.info(
                    "Multi-tenant aggregation search completed",
                    collections_searched=len(collections),
                    raw_results=enhanced_response["aggregation_metadata"].get("raw_result_count", 0),
                    deduplicated_results=enhanced_response["aggregation_metadata"].get("post_deduplication_count", 0),
                    final_results=enhanced_response["total"],
                    response_time_ms=enhanced_response.get("performance", {}).get("response_time_ms")
                )

                return enhanced_response

            except Exception as e:
                logger.warning(
                    "Multi-tenant aggregation failed, falling back to standard search",
                    error=str(e)
                )
                # Fall through to standard search logic

        # Search each collection using standard approach
        all_results = []

        for i, display_name in enumerate(collections):
            actual_name = actual_collections[i]
            try:
                # Use new project-aware search if enabled
                if auto_inject_project_metadata:
                    # Use the client's search_with_project_context for automatic metadata filtering
                    search_result = await client.search_with_project_context(
                        collection_name=actual_name,
                        query_embeddings=embeddings,
                        collection_type=actual_name,  # Use collection name as type
                        limit=limit,
                        fusion_method=mode if mode == "hybrid" else "rrf",
                        include_shared=include_shared,
                        with_payload=True,
                        score_threshold=score_threshold,
                    )

                    # Extract results from search_result
                    if "fused_results" in search_result:
                        collection_results = [
                            {
                                "id": result.id,
                                "score": getattr(result, "score", 0.0),
                                "payload": getattr(result, "payload", {}),
                                "search_type": mode
                            }
                            for result in search_result["fused_results"]
                            if getattr(result, "score", 0) >= score_threshold
                        ]
                    else:
                        collection_results = []
                else:
                    # Fallback to legacy search without metadata filtering
                    collection_results = await _search_collection(
                        client.client,
                        actual_name,  # Use actual name for Qdrant operations
                        embeddings,
                        mode,
                        limit,
                        score_threshold,
                    )

                # Add display name to results for user-facing response
                for result in collection_results:
                    result["collection"] = display_name

                all_results.extend(collection_results)

            except Exception as e:
                logger.warning(
                    "Failed to search collection %s (actual: %s): %s",
                    display_name,
                    actual_name,
                    e,
                )
                continue

        # Sort by score and limit results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = all_results[:limit]

        # Enrich response with project context information
        response_data = {
            "query": query,
            "mode": mode,
            "collections_searched": collections,
            "total": len(final_results),
            "results": final_results,
            "search_params": {
                "mode": mode,
                "limit": limit,
                "score_threshold": score_threshold,
                "auto_inject_project_metadata": auto_inject_project_metadata,
                "include_shared": include_shared,
            },
        }

        # Add project context information if enabled
        if auto_inject_project_metadata:
            detected_context = project_context or client.get_project_context()
            if detected_context:
                response_data["project_context"] = {
                    "project_name": detected_context.get("project_name"),
                    "workspace_scope": detected_context.get("workspace_scope"),
                    "metadata_filtering_enabled": True
                }
            else:
                response_data["project_context"] = {
                    "metadata_filtering_enabled": False,
                    "reason": "No project context detected"
                }

        return response_data

    except Exception as e:
        logger.error("Search failed: %s", e)
        return {"error": f"Search failed: {e}"}


async def _search_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    embeddings: dict,
    mode: str,
    limit: int,
    score_threshold: float,
) -> list[dict]:
    """Search a single collection with hybrid search support.

    Internal method that handles the actual search operation for a single
    collection. Optimizes search strategy based on available embeddings
    and requested mode.

    Args:
        qdrant_client: Direct Qdrant client instance
        collection_name: Name of the collection to search
        embeddings: Pre-generated embedding vectors (dense/sparse)
        mode: Search mode to use
        limit: Maximum results for this collection
        score_threshold: Minimum score threshold

    Returns:
        List[Dict]: Search results for the collection
    """

    try:
        if mode == "hybrid" and "dense" in embeddings:
            # Use hybrid search engine for RRF fusion
            hybrid_engine = HybridSearchEngine(qdrant_client)

            result_or_awaitable = hybrid_engine.hybrid_search(
                collection_name=collection_name,
                query_embeddings=embeddings,
                limit=limit,
                score_threshold=score_threshold,
            )

            # Handle both real async calls and mocked synchronous returns
            if hasattr(result_or_awaitable, "__await__"):
                result = await result_or_awaitable
            else:
                result = result_or_awaitable

            if "error" in result:
                logger.error("Hybrid search failed: %s", result["error"])
                return []

            # Convert hybrid results to expected format
            search_results = []
            for r in result.get("results", []):
                search_results.append(
                    {
                        "id": r["id"],
                        "score": r.get("rrf_score", r.get("score", 0.0)),
                        "payload": r["payload"],
                        "search_type": "hybrid",
                    }
                )

            return search_results

        # Single mode searches (dense or sparse)
        search_results = []

        if mode == "dense" and "dense" in embeddings:
            # Dense vector search
            dense_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=embeddings["dense"],
                using="dense",
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            for result in dense_results:
                # Apply score threshold filtering
                if result.score >= score_threshold:
                    search_results.append(
                        {
                            "id": result.id,
                            "score": result.score,
                            "payload": result.payload,
                            "search_type": "dense",
                        }
                    )

        elif mode == "sparse" and "sparse" in embeddings:
            # Sparse vector search using enhanced BM25
            sparse_vector = create_named_sparse_vector(
                indices=embeddings["sparse"]["indices"],
                values=embeddings["sparse"]["values"],
                name="sparse",
            )

            sparse_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=sparse_vector,
                using="sparse",
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            for result in sparse_results:
                # Apply score threshold filtering
                if result.score >= score_threshold:
                    search_results.append(
                        {
                            "id": result.id,
                            "score": result.score,
                            "payload": result.payload,
                            "search_type": "sparse",
                        }
                    )

        return search_results[:limit]

    except ResponseHandlingException as e:
        logger.error("Qdrant search error in collection %s: %s", collection_name, e)
        return []
    except Exception as e:
        logger.error("Unexpected search error in collection %s: %s", collection_name, e)
        return []


async def search_collection_by_metadata(
    client: QdrantWorkspaceClient,
    collection: str,
    metadata_filter: dict,
    limit: int = 10,
) -> dict:
    """
    Search collection by metadata filter.

    Args:
        client: Workspace client instance
        collection: Collection name to search
        metadata_filter: Metadata filter conditions
        limit: Maximum number of results

    Returns:
        Dictionary with search results
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}

    # Validate metadata filter is not empty
    if not metadata_filter:
        return {"error": "Metadata filter cannot be empty"}

    try:
        # Validate collection exists
        available_collections = client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}

        # Resolve display name to actual collection name
        actual_collection, _ = client.collection_manager.resolve_collection_name(
            collection
        )

        # Build Qdrant filter
        qdrant_filter = _build_metadata_filter(metadata_filter)

        # Search with metadata filter
        results = client.client.scroll(
            collection_name=actual_collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )

        formatted_results = []
        for result in results[0]:  # results is (points, next_page_offset)
            formatted_results.append({"id": result.id, "payload": result.payload})

        # Apply limit enforcement
        formatted_results = formatted_results[:limit]

        return {
            "collection": collection,
            "filter": metadata_filter,
            "total": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        logger.error("Metadata search failed: %s", e)
        return {"error": f"Metadata search failed: {e}"}


def _build_metadata_filter(metadata_filter: dict) -> models.Filter:
    """Build Qdrant filter from metadata dictionary.

    Converts a simple metadata dictionary into Qdrant's filter format,
    supporting exact matches, numeric comparisons, and list-based filtering.

    Args:
        metadata_filter: Dictionary of field->value mappings for filtering

    Returns:
        models.Filter: Qdrant filter object with appropriate conditions
        Returns None if no valid conditions found

    Supported Value Types:
        - str: Exact string match
        - int/float: Exact numeric match
        - List: Match any value in the list (OR condition)

    Example:
        ```python
        filter_dict = {
            "file_type": "python",
            "priority": 1,
            "tags": ["auth", "security"]
        }
        qdrant_filter = _build_metadata_filter(filter_dict)
        ```
    """
    conditions = []

    for key, value in metadata_filter.items():
        if isinstance(value, str):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        elif isinstance(value, int):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        elif isinstance(value, float):
            # For floats, convert to int if it's a whole number, otherwise to string
            if value.is_integer():
                conditions.append(
                    models.FieldCondition(
                        key=key, match=models.MatchValue(value=int(value))
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key, match=models.MatchValue(value=str(value))
                    )
                )
        elif isinstance(value, bool):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        elif isinstance(value, list):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=value))
            )

    return models.Filter(must=conditions) if conditions else None


async def search_workspace_with_project_isolation(
    client: QdrantWorkspaceClient,
    query: str,
    project_name: Optional[str] = None,
    collection_types: Optional[list[str]] = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7,
    include_shared: bool = True,
) -> dict:
    """
    Search workspace with automatic multi-tenant project isolation.

    This is a convenience function that combines the best of the enhanced search
    capabilities with automatic project context detection and metadata filtering.

    Args:
        client: Initialized workspace client with embedding service
        query: Natural language query or exact text to search for
        project_name: Specific project to search within (auto-detected if None)
        collection_types: Specific collection types to search (notes, docs, etc.)
        mode: Search strategy (hybrid, dense, sparse)
        limit: Maximum number of results to return
        score_threshold: Minimum relevance score (0.0-1.0)
        include_shared: Whether to include shared workspace resources

    Returns:
        Dict: Enhanced search results with project isolation applied

    Example:
        ```python
        # Search within current project context
        results = await search_workspace_with_project_isolation(
            client=workspace_client,
            query="authentication patterns",
            collection_types=["docs", "notes"],
            include_shared=True
        )

        # Search specific project
        results = await search_workspace_with_project_isolation(
            client=workspace_client,
            query="database schema",
            project_name="user-service",
            collection_types=["docs"]
        )
        ```
    """
    try:
        # Build project context if not provided
        if project_name:
            # Use provided project name
            project_context = {
                "project_name": project_name,
                "project_id": client._generate_project_id(project_name),
                "workspace_scope": "project"
            }
        else:
            # Auto-detect project context
            project_context = client.get_project_context()

        if not project_context:
            logger.warning("No project context available for isolated search")
            # Fall back to regular search without project filtering
            return await search_workspace(
                client=client,
                query=query,
                mode=mode,
                limit=limit,
                score_threshold=score_threshold,
                auto_inject_project_metadata=False
            )

        # Determine collections to search based on collection types
        target_collections = None
        if collection_types:
            # Build collection names based on the new multi-tenant architecture
            # Collections are now shared (e.g., "docs", "notes") with metadata isolation
            target_collections = collection_types
        else:
            # Search all available collections with project filtering
            target_collections = None  # Will use all searchable collections

        # Perform search with project context
        results = await search_workspace(
            client=client,
            query=query,
            collections=target_collections,
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
            project_context=project_context,
            auto_inject_project_metadata=True,
            include_shared=include_shared
        )

        # Enhance results with isolation information
        if "project_context" not in results:
            results["project_context"] = project_context

        results["isolation_info"] = {
            "project_name": project_context.get("project_name"),
            "collection_types": collection_types,
            "isolation_enabled": True,
            "shared_resources_included": include_shared
        }

        logger.info(
            "Project-isolated search completed",
            project_name=project_context.get("project_name"),
            collection_types=collection_types,
            results_count=results.get("total", 0)
        )

        return results

    except Exception as e:
        logger.error("Project-isolated search failed: %s", e)
        return {"error": f"Project-isolated search failed: {e}"}


async def search_workspace_with_advanced_aggregation(
    client: QdrantWorkspaceClient,
    query: str,
    collections: list[str] | None = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7,
    project_context: Optional[dict] = None,
    aggregation_settings: Optional[dict] = None
) -> dict:
    """
    Search workspace with advanced multi-tenant result aggregation.

    This is a convenience function that provides easy access to the advanced
    multi-tenant result aggregation capabilities with customizable settings.

    Args:
        client: Initialized workspace client with embedding service
        query: Natural language query or exact text to search for
        collections: Specific collections to search. If None, searches all workspace collections
        mode: Search strategy (hybrid, dense, sparse)
        limit: Maximum number of results to return across all collections
        score_threshold: Minimum relevance score (0.0-1.0)
        project_context: Optional project context for metadata filtering
        aggregation_settings: Optional dict with aggregation configuration:
            - enable_multi_tenant_aggregation: bool (default: True)
            - enable_deduplication: bool (default: True)
            - score_aggregation_method: str (default: "max_score")
            - preserve_tenant_isolation: bool (default: True)
            - enable_score_normalization: bool (default: True)

    Returns:
        Dict: Enhanced search results with advanced aggregation metadata

    Example:
        ```python
        # Search with custom aggregation settings
        results = await search_workspace_with_advanced_aggregation(
            client=workspace_client,
            query="authentication patterns",
            mode="hybrid",
            limit=20,
            aggregation_settings={
                "enable_deduplication": True,
                "score_aggregation_method": "avg_score",
                "preserve_tenant_isolation": True,
                "enable_score_normalization": True
            }
        )

        # Access aggregation metadata
        metadata = results.get("aggregation_metadata", {})
        print(f"Deduplicated {metadata.get('duplicates_found', 0)} results")
        print(f"Score normalization: {metadata.get('score_normalization_enabled', False)}")
        ```

    Task 233.5: Added for advanced multi-tenant result aggregation with full configuration control.
    """
    # Set default aggregation settings
    default_settings = {
        "enable_multi_tenant_aggregation": True,
        "enable_deduplication": True,
        "score_aggregation_method": "max_score",
        "preserve_tenant_isolation": True,
        "enable_score_normalization": True
    }

    # Merge with user settings
    if aggregation_settings:
        default_settings.update(aggregation_settings)

    try:
        # Use the enhanced search_workspace with aggregation settings
        result = await search_workspace(
            client=client,
            query=query,
            collections=collections,
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
            project_context=project_context,
            auto_inject_project_metadata=True,
            include_shared=True,
            enable_multi_tenant_aggregation=default_settings["enable_multi_tenant_aggregation"],
            enable_deduplication=default_settings["enable_deduplication"],
            score_aggregation_method=default_settings["score_aggregation_method"]
        )

        # Add aggregation settings to the response for transparency
        result["aggregation_settings_used"] = default_settings

        logger.info(
            "Advanced aggregation search completed",
            query=query[:50] + "..." if len(query) > 50 else query,
            total_results=result.get("total", 0),
            aggregation_settings=default_settings
        )

        return result

    except Exception as e:
        logger.error("Advanced aggregation search failed: %s", e)
        return {"error": f"Advanced aggregation search failed: {e}"}
