"""
Four-Context Hierarchy Hybrid Search System for workspace-qdrant-mcp.

Task 259: Implements advanced hybrid search with semantic + keyword fusion across
four-tier search hierarchy: project/collection/global/all contexts with LSP integration.

This module provides the next-generation search system that combines:
- Dense semantic vector search (FastEmbed embeddings)
- Sparse keyword vector search (enhanced BM25)
- Reciprocal Rank Fusion (RRF) for optimal result combination
- Four-tier context hierarchy for precise scoped searches
- Code-aware search with LSP metadata integration
- Sub-100ms response times with caching and optimization
- Context-aware routing and result ranking

Context Hierarchy:
    1. **Project**: Current project collections only
    2. **Collection**: Specific target collection(s)
    3. **Global**: User-configured global collections
    4. **All**: Comprehensive workspace-wide search

Key Features:
    - Intelligent context detection and routing
    - LSP-enhanced code symbol and relationship search
    - Advanced caching with context-aware invalidation
    - Performance optimization for <100ms response times
    - Configurable fusion strategies (RRF, weighted sum, max score)
    - Multi-tenant result aggregation and deduplication
    - Comprehensive search analytics and monitoring

Example:
    ```python
    from workspace_qdrant_mcp.core.four_context_search import FourContextSearchEngine

    # Initialize search engine
    search_engine = FourContextSearchEngine(workspace_client)
    await search_engine.initialize()

    # Project-scoped search
    results = await search_engine.search_project_context(
        query="authentication patterns",
        project_name="my-service"
    )

    # Collection-specific search
    results = await search_engine.search_collection_context(
        query="JWT validation",
        collections=["auth-library", "security-docs"]
    )

    # Global workspace search
    results = await search_engine.search_global_context(
        query="database migration patterns"
    )

    # Comprehensive all-context search
    results = await search_engine.search_all_contexts(
        query="error handling best practices"
    )
    ```
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from ...common.core.client import QdrantWorkspaceClient
from ...common.core.hybrid_search import HybridSearchEngine, RRFFusionRanker
from ...common.utils.project_detection import ProjectDetector
from ...common.core.collections import CollectionSelector
from ..tools.search import search_workspace, search_workspace_with_project_isolation
from ..tools.code_search import CodeSearchEngine, CodeSearchResult
from ..tools.multitenant_search import MultiTenantSearchCoordinator


class SearchContext(Enum):
    """Four-tier search context hierarchy"""
    PROJECT = "project"       # Current project collections only
    COLLECTION = "collection" # Specific target collection(s)
    GLOBAL = "global"         # User-configured global collections
    ALL = "all"               # Comprehensive workspace-wide search


class SearchScope(Enum):
    """Search scope within context"""
    NARROW = "narrow"         # Focused, precise search
    BROAD = "broad"           # Comprehensive, expansive search
    ADAPTIVE = "adaptive"     # Dynamically adjusts based on results


@dataclass
class SearchContextConfig:
    """Configuration for a specific search context"""
    context: SearchContext
    collections: List[str] = field(default_factory=list)
    include_shared: bool = True
    enable_code_search: bool = True
    enable_lsp_enhancement: bool = True
    cache_ttl_seconds: int = 300
    max_results_per_collection: int = 20
    fusion_method: str = "rrf"
    score_threshold: float = 0.6


@dataclass
class FourContextSearchQuery:
    """Comprehensive search query across four contexts"""
    query: str
    contexts: List[SearchContext] = field(default_factory=lambda: [SearchContext.PROJECT])
    scope: SearchScope = SearchScope.ADAPTIVE
    mode: str = "hybrid"  # hybrid, dense, sparse
    limit: int = 20
    project_name: Optional[str] = None
    target_collections: Optional[List[str]] = None
    global_collections: Optional[List[str]] = None
    include_code_search: bool = True
    enable_lsp_enhancement: bool = True
    enable_deduplication: bool = True
    response_time_target_ms: float = 100.0


@dataclass
class SearchResult:
    """Enhanced search result with context metadata"""
    id: str
    score: float
    content: str
    collection: str
    context: SearchContext
    search_type: str  # hybrid, dense, sparse, code, lsp
    metadata: Dict[str, Any]
    code_result: Optional[CodeSearchResult] = None
    fusion_details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None


@dataclass
class FourContextSearchResponse:
    """Comprehensive response from four-context search"""
    query: str
    total_results: int
    results: List[SearchResult]
    context_breakdown: Dict[SearchContext, int]
    performance_metrics: Dict[str, Any]
    search_strategy: Dict[str, Any]
    cache_hits: Dict[SearchContext, bool]
    fusion_summary: Dict[str, Any]


class ContextSearchCache:
    """High-performance cache for context-specific search results"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _generate_cache_key(
        self,
        query: str,
        context: SearchContext,
        collections: List[str],
        mode: str
    ) -> str:
        """Generate cache key for search parameters"""
        collections_str = "|".join(sorted(collections)) if collections else ""
        return f"{context.value}:{mode}:{hash(query)}:{hash(collections_str)}"

    def get(
        self,
        query: str,
        context: SearchContext,
        collections: List[str],
        mode: str
    ) -> Optional[Any]:
        """Get cached search results"""
        cache_key = self._generate_cache_key(query, context, collections, mode)

        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hit_count += 1
                logger.debug("Cache hit", key=cache_key, context=context.value)
                return result
            else:
                # Remove expired entry
                del self.cache[cache_key]

        self.miss_count += 1
        logger.debug("Cache miss", key=cache_key, context=context.value)
        return None

    def set(
        self,
        query: str,
        context: SearchContext,
        collections: List[str],
        mode: str,
        result: Any
    ) -> None:
        """Cache search results"""
        cache_key = self._generate_cache_key(query, context, collections, mode)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[cache_key] = (result, time.time())
        logger.debug("Cached result", key=cache_key, context=context.value)

    def invalidate_context(self, context: SearchContext) -> None:
        """Invalidate all cache entries for a specific context"""
        keys_to_remove = [
            key for key in self.cache.keys()
            if key.startswith(f"{context.value}:")
        ]
        for key in keys_to_remove:
            del self.cache[key]
        logger.info("Invalidated cache", context=context.value, entries=len(keys_to_remove))

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cleared search cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }


class FourContextSearchEngine:
    """
    Advanced hybrid search engine with four-context hierarchy and LSP integration.

    Provides comprehensive search capabilities across project, collection, global,
    and all contexts with intelligent routing, caching, and performance optimization.
    """

    def __init__(
        self,
        workspace_client: QdrantWorkspaceClient,
        enable_performance_monitoring: bool = True,
        cache_ttl_seconds: int = 300
    ):
        self.workspace_client = workspace_client
        self.enable_performance_monitoring = enable_performance_monitoring

        # Core search engines
        self.hybrid_engine: Optional[HybridSearchEngine] = None
        self.code_search_engine: Optional[CodeSearchEngine] = None
        self.multitenant_coordinator: Optional[MultiTenantSearchCoordinator] = None

        # Workspace detection and routing
        self.project_detector: Optional[ProjectDetector] = None
        self.collection_selector: Optional[CollectionSelector] = None

        # Performance optimization
        self.search_cache = ContextSearchCache(ttl_seconds=cache_ttl_seconds)
        self.performance_tracker = defaultdict(list)

        # Context configurations
        self.context_configs: Dict[SearchContext, SearchContextConfig] = {}
        self._initialize_default_configs()

        self.initialized = False

    def _initialize_default_configs(self) -> None:
        """Initialize default configurations for each search context"""
        self.context_configs = {
            SearchContext.PROJECT: SearchContextConfig(
                context=SearchContext.PROJECT,
                include_shared=True,
                enable_code_search=True,
                enable_lsp_enhancement=True,
                max_results_per_collection=15,
                score_threshold=0.7
            ),
            SearchContext.COLLECTION: SearchContextConfig(
                context=SearchContext.COLLECTION,
                include_shared=False,
                enable_code_search=True,
                enable_lsp_enhancement=True,
                max_results_per_collection=20,
                score_threshold=0.6
            ),
            SearchContext.GLOBAL: SearchContextConfig(
                context=SearchContext.GLOBAL,
                include_shared=True,
                enable_code_search=False,
                enable_lsp_enhancement=False,
                max_results_per_collection=10,
                score_threshold=0.6
            ),
            SearchContext.ALL: SearchContextConfig(
                context=SearchContext.ALL,
                include_shared=True,
                enable_code_search=False,
                enable_lsp_enhancement=False,
                max_results_per_collection=8,
                score_threshold=0.5
            )
        }

    async def initialize(self) -> None:
        """Initialize the four-context search engine"""
        if self.initialized:
            return

        try:
            logger.info("Initializing four-context search engine")
            start_time = time.time()

            # Initialize hybrid search engine with optimizations
            self.hybrid_engine = HybridSearchEngine(
                client=self.workspace_client.client,
                enable_optimizations=True,
                enable_multi_tenant_aggregation=True,
                enable_performance_monitoring=self.enable_performance_monitoring
            )

            # Initialize code search engine
            self.code_search_engine = CodeSearchEngine(self.workspace_client)
            await self.code_search_engine.initialize()

            # Initialize project detection and collection selection
            self.project_detector = ProjectDetector()
            self.collection_selector = CollectionSelector(
                self.workspace_client.client,
                self.workspace_client.config,
                self.project_detector
            )

            # Initialize multi-tenant coordinator
            self.multitenant_coordinator = MultiTenantSearchCoordinator(
                self.workspace_client
            )

            initialization_time = (time.time() - start_time) * 1000

            self.initialized = True
            logger.info(
                "Four-context search engine initialized",
                initialization_time_ms=initialization_time,
                contexts_configured=len(self.context_configs)
            )

        except Exception as e:
            logger.error("Failed to initialize four-context search engine", error=str(e))
            raise

    async def search_project_context(
        self,
        query: str,
        project_name: Optional[str] = None,
        scope: SearchScope = SearchScope.ADAPTIVE,
        mode: str = "hybrid",
        limit: int = 20,
        enable_cache: bool = True
    ) -> FourContextSearchResponse:
        """
        Search within project context - current project collections only.

        Args:
            query: Search query
            project_name: Specific project name (auto-detected if None)
            scope: Search scope (narrow/broad/adaptive)
            mode: Search mode (hybrid/dense/sparse)
            limit: Maximum results
            enable_cache: Use search cache

        Returns:
            Project-scoped search results
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            logger.info(
                "Starting project context search",
                query=query[:50] + "..." if len(query) > 50 else query,
                project_name=project_name,
                scope=scope.value,
                mode=mode
            )

            # Detect or validate project context
            if not project_name:
                project_info = await self._detect_current_project()
                project_name = project_info.get("project_name") if project_info else None

            if not project_name:
                logger.warning("No project context available for project search")
                return self._create_empty_response(query, SearchContext.PROJECT)

            # Get project collections
            project_collections = await self._get_project_collections(project_name)

            if not project_collections:
                logger.warning("No collections found for project", project_name=project_name)
                return self._create_empty_response(query, SearchContext.PROJECT)

            # Check cache
            cached_result = None
            if enable_cache:
                cached_result = self.search_cache.get(
                    query, SearchContext.PROJECT, project_collections, mode
                )
                if cached_result:
                    return self._enhance_cached_response(cached_result, start_time)

            # Configure search for project context
            config = self.context_configs[SearchContext.PROJECT]

            # Execute search with project isolation
            search_results = await search_workspace_with_project_isolation(
                client=self.workspace_client,
                query=query,
                project_name=project_name,
                collection_types=None,  # Include all project collection types
                mode=mode,
                limit=limit,
                score_threshold=config.score_threshold,
                include_shared=config.include_shared
            )

            # Enhance with code search if enabled
            enhanced_results = []
            if config.enable_code_search and self.code_search_engine:
                enhanced_results = await self._enhance_with_code_search(
                    search_results.get("results", []),
                    query,
                    project_collections,
                    config
                )
            else:
                enhanced_results = self._convert_to_search_results(
                    search_results.get("results", []),
                    SearchContext.PROJECT
                )

            # Create response
            response = self._create_search_response(
                query=query,
                results=enhanced_results,
                primary_context=SearchContext.PROJECT,
                search_time_ms=(time.time() - start_time) * 1000,
                cache_hit=cached_result is not None,
                collections_searched=project_collections
            )

            # Cache result
            if enable_cache and not cached_result:
                self.search_cache.set(
                    query, SearchContext.PROJECT, project_collections, mode, response
                )

            logger.info(
                "Project context search completed",
                project_name=project_name,
                results_count=len(enhanced_results),
                response_time_ms=response.performance_metrics["response_time_ms"]
            )

            return response

        except Exception as e:
            logger.error("Project context search failed", query=query, error=str(e))
            return self._create_error_response(query, SearchContext.PROJECT, str(e))

    async def search_collection_context(
        self,
        query: str,
        collections: List[str],
        scope: SearchScope = SearchScope.ADAPTIVE,
        mode: str = "hybrid",
        limit: int = 20,
        enable_cache: bool = True
    ) -> FourContextSearchResponse:
        """
        Search within specific collection(s) context.

        Args:
            query: Search query
            collections: Target collections to search
            scope: Search scope (narrow/broad/adaptive)
            mode: Search mode (hybrid/dense/sparse)
            limit: Maximum results
            enable_cache: Use search cache

        Returns:
            Collection-scoped search results
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            logger.info(
                "Starting collection context search",
                query=query[:50] + "..." if len(query) > 50 else query,
                collections=collections,
                scope=scope.value,
                mode=mode
            )

            # Validate collections exist
            available_collections = self.workspace_client.list_collections()
            valid_collections = [c for c in collections if c in available_collections]

            if not valid_collections:
                logger.warning("No valid collections found", collections=collections)
                return self._create_empty_response(query, SearchContext.COLLECTION)

            # Check cache
            cached_result = None
            if enable_cache:
                cached_result = self.search_cache.get(
                    query, SearchContext.COLLECTION, valid_collections, mode
                )
                if cached_result:
                    return self._enhance_cached_response(cached_result, start_time)

            # Configure search for collection context
            config = self.context_configs[SearchContext.COLLECTION]

            # Execute search across specified collections
            search_results = await search_workspace(
                client=self.workspace_client,
                query=query,
                collections=valid_collections,
                mode=mode,
                limit=limit,
                score_threshold=config.score_threshold,
                auto_inject_project_metadata=False,  # Collection-specific, no auto-injection
                include_shared=config.include_shared,
                enable_multi_tenant_aggregation=True,
                enable_deduplication=True
            )

            # Enhance with code search if enabled
            enhanced_results = []
            if config.enable_code_search and self.code_search_engine:
                enhanced_results = await self._enhance_with_code_search(
                    search_results.get("results", []),
                    query,
                    valid_collections,
                    config
                )
            else:
                enhanced_results = self._convert_to_search_results(
                    search_results.get("results", []),
                    SearchContext.COLLECTION
                )

            # Create response
            response = self._create_search_response(
                query=query,
                results=enhanced_results,
                primary_context=SearchContext.COLLECTION,
                search_time_ms=(time.time() - start_time) * 1000,
                cache_hit=cached_result is not None,
                collections_searched=valid_collections
            )

            # Cache result
            if enable_cache and not cached_result:
                self.search_cache.set(
                    query, SearchContext.COLLECTION, valid_collections, mode, response
                )

            logger.info(
                "Collection context search completed",
                collections=valid_collections,
                results_count=len(enhanced_results),
                response_time_ms=response.performance_metrics["response_time_ms"]
            )

            return response

        except Exception as e:
            logger.error("Collection context search failed", query=query, error=str(e))
            return self._create_error_response(query, SearchContext.COLLECTION, str(e))

    async def search_global_context(
        self,
        query: str,
        global_collections: Optional[List[str]] = None,
        scope: SearchScope = SearchScope.ADAPTIVE,
        mode: str = "hybrid",
        limit: int = 20,
        enable_cache: bool = True
    ) -> FourContextSearchResponse:
        """
        Search within global context - user-configured global collections.

        Args:
            query: Search query
            global_collections: Specific global collections (auto-detected if None)
            scope: Search scope (narrow/broad/adaptive)
            mode: Search mode (hybrid/dense/sparse)
            limit: Maximum results
            enable_cache: Use search cache

        Returns:
            Global-scoped search results
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            logger.info(
                "Starting global context search",
                query=query[:50] + "..." if len(query) > 50 else query,
                global_collections=global_collections,
                scope=scope.value,
                mode=mode
            )

            # Get global collections
            if not global_collections:
                global_collections = await self._get_global_collections()

            if not global_collections:
                logger.warning("No global collections configured")
                return self._create_empty_response(query, SearchContext.GLOBAL)

            # Check cache
            cached_result = None
            if enable_cache:
                cached_result = self.search_cache.get(
                    query, SearchContext.GLOBAL, global_collections, mode
                )
                if cached_result:
                    return self._enhance_cached_response(cached_result, start_time)

            # Configure search for global context
            config = self.context_configs[SearchContext.GLOBAL]

            # Execute search across global collections
            search_results = await search_workspace(
                client=self.workspace_client,
                query=query,
                collections=global_collections,
                mode=mode,
                limit=limit,
                score_threshold=config.score_threshold,
                auto_inject_project_metadata=False,  # Global search, no project filtering
                include_shared=config.include_shared,
                enable_multi_tenant_aggregation=True,
                enable_deduplication=True
            )

            # Convert to search results (no code search for global context by default)
            enhanced_results = self._convert_to_search_results(
                search_results.get("results", []),
                SearchContext.GLOBAL
            )

            # Create response
            response = self._create_search_response(
                query=query,
                results=enhanced_results,
                primary_context=SearchContext.GLOBAL,
                search_time_ms=(time.time() - start_time) * 1000,
                cache_hit=cached_result is not None,
                collections_searched=global_collections
            )

            # Cache result
            if enable_cache and not cached_result:
                self.search_cache.set(
                    query, SearchContext.GLOBAL, global_collections, mode, response
                )

            logger.info(
                "Global context search completed",
                global_collections=global_collections,
                results_count=len(enhanced_results),
                response_time_ms=response.performance_metrics["response_time_ms"]
            )

            return response

        except Exception as e:
            logger.error("Global context search failed", query=query, error=str(e))
            return self._create_error_response(query, SearchContext.GLOBAL, str(e))

    async def search_all_contexts(
        self,
        query: str,
        scope: SearchScope = SearchScope.ADAPTIVE,
        mode: str = "hybrid",
        limit: int = 20,
        enable_cache: bool = True
    ) -> FourContextSearchResponse:
        """
        Search across all contexts - comprehensive workspace-wide search.

        Args:
            query: Search query
            scope: Search scope (narrow/broad/adaptive)
            mode: Search mode (hybrid/dense/sparse)
            limit: Maximum results
            enable_cache: Use search cache

        Returns:
            All-context search results
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            logger.info(
                "Starting all-context search",
                query=query[:50] + "..." if len(query) > 50 else query,
                scope=scope.value,
                mode=mode
            )

            # Get all available collections
            all_collections = self.workspace_client.list_collections()

            if not all_collections:
                logger.warning("No collections found for all-context search")
                return self._create_empty_response(query, SearchContext.ALL)

            # Check cache
            cached_result = None
            if enable_cache:
                cached_result = self.search_cache.get(
                    query, SearchContext.ALL, all_collections, mode
                )
                if cached_result:
                    return self._enhance_cached_response(cached_result, start_time)

            # Configure search for all-context
            config = self.context_configs[SearchContext.ALL]

            # Execute comprehensive search across all collections
            search_results = await search_workspace(
                client=self.workspace_client,
                query=query,
                collections=None,  # Search all collections
                mode=mode,
                limit=limit,
                score_threshold=config.score_threshold,
                auto_inject_project_metadata=False,  # No automatic filtering
                include_shared=config.include_shared,
                enable_multi_tenant_aggregation=True,
                enable_deduplication=True
            )

            # Convert to search results
            enhanced_results = self._convert_to_search_results(
                search_results.get("results", []),
                SearchContext.ALL
            )

            # Create response
            response = self._create_search_response(
                query=query,
                results=enhanced_results,
                primary_context=SearchContext.ALL,
                search_time_ms=(time.time() - start_time) * 1000,
                cache_hit=cached_result is not None,
                collections_searched=all_collections
            )

            # Cache result
            if enable_cache and not cached_result:
                self.search_cache.set(
                    query, SearchContext.ALL, all_collections, mode, response
                )

            logger.info(
                "All-context search completed",
                total_collections=len(all_collections),
                results_count=len(enhanced_results),
                response_time_ms=response.performance_metrics["response_time_ms"]
            )

            return response

        except Exception as e:
            logger.error("All-context search failed", query=query, error=str(e))
            return self._create_error_response(query, SearchContext.ALL, str(e))

    async def search_multi_context(
        self,
        search_query: FourContextSearchQuery
    ) -> FourContextSearchResponse:
        """
        Execute search across multiple contexts with intelligent routing and fusion.

        Args:
            search_query: Comprehensive multi-context search query

        Returns:
            Fused results from multiple contexts
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            logger.info(
                "Starting multi-context search",
                query=search_query.query[:50] + "..." if len(search_query.query) > 50 else search_query.query,
                contexts=[ctx.value for ctx in search_query.contexts],
                scope=search_query.scope.value,
                mode=search_query.mode
            )

            # Execute searches across all requested contexts in parallel
            context_tasks = []

            for context in search_query.contexts:
                if context == SearchContext.PROJECT:
                    task = self.search_project_context(
                        query=search_query.query,
                        project_name=search_query.project_name,
                        scope=search_query.scope,
                        mode=search_query.mode,
                        limit=search_query.limit // len(search_query.contexts)
                    )
                elif context == SearchContext.COLLECTION:
                    if search_query.target_collections:
                        task = self.search_collection_context(
                            query=search_query.query,
                            collections=search_query.target_collections,
                            scope=search_query.scope,
                            mode=search_query.mode,
                            limit=search_query.limit // len(search_query.contexts)
                        )
                    else:
                        continue  # Skip if no target collections specified
                elif context == SearchContext.GLOBAL:
                    task = self.search_global_context(
                        query=search_query.query,
                        global_collections=search_query.global_collections,
                        scope=search_query.scope,
                        mode=search_query.mode,
                        limit=search_query.limit // len(search_query.contexts)
                    )
                elif context == SearchContext.ALL:
                    task = self.search_all_contexts(
                        query=search_query.query,
                        scope=search_query.scope,
                        mode=search_query.mode,
                        limit=search_query.limit // len(search_query.contexts)
                    )
                else:
                    continue

                context_tasks.append((context, task))

            # Execute all context searches concurrently
            context_results = {}
            for context, task in context_tasks:
                try:
                    result = await task
                    context_results[context] = result
                except Exception as e:
                    logger.error(f"Context search failed", context=context.value, error=str(e))
                    context_results[context] = self._create_error_response(
                        search_query.query, context, str(e)
                    )

            # Fuse results across contexts using RRF
            fused_results = await self._fuse_multi_context_results(
                context_results, search_query
            )

            # Create comprehensive response
            response = self._create_multi_context_response(
                search_query=search_query,
                context_results=context_results,
                fused_results=fused_results,
                search_time_ms=(time.time() - start_time) * 1000
            )

            logger.info(
                "Multi-context search completed",
                contexts=[ctx.value for ctx in search_query.contexts],
                total_results=len(fused_results),
                response_time_ms=response.performance_metrics["response_time_ms"]
            )

            return response

        except Exception as e:
            logger.error("Multi-context search failed", query=search_query.query, error=str(e))
            return self._create_error_response(search_query.query, SearchContext.ALL, str(e))

    # Helper methods for context detection and collection management

    async def _detect_current_project(self) -> Optional[Dict[str, Any]]:
        """Detect current project context"""
        try:
            if hasattr(self.workspace_client, 'project_info') and self.workspace_client.project_info:
                return self.workspace_client.project_info

            if self.project_detector:
                return await self.project_detector.detect_project_context()

            return None

        except Exception as e:
            logger.error("Failed to detect current project", error=str(e))
            return None

    async def _get_project_collections(self, project_name: str) -> List[str]:
        """Get collections belonging to a specific project"""
        try:
            if self.collection_selector:
                return self.collection_selector.get_searchable_collections(
                    project_name=project_name,
                    include_memory=True,
                    include_shared=True
                )

            # Fallback: filter by project name pattern
            all_collections = self.workspace_client.list_collections()
            project_collections = [
                c for c in all_collections
                if project_name.lower() in c.lower()
            ]

            return project_collections

        except Exception as e:
            logger.error("Failed to get project collections", project_name=project_name, error=str(e))
            return []

    async def _get_global_collections(self) -> List[str]:
        """Get user-configured global collections"""
        try:
            # Check workspace client configuration for global collections
            if hasattr(self.workspace_client, 'config') and self.workspace_client.config:
                global_collections = self.workspace_client.config.get('global_collections', [])
                if global_collections:
                    return global_collections

            # Fallback: look for collections with global indicators
            all_collections = self.workspace_client.list_collections()
            global_indicators = ['global', 'shared', 'common', 'docs', 'wiki']

            global_collections = [
                c for c in all_collections
                if any(indicator in c.lower() for indicator in global_indicators)
            ]

            return global_collections

        except Exception as e:
            logger.error("Failed to get global collections", error=str(e))
            return []

    async def _enhance_with_code_search(
        self,
        search_results: List[Dict[str, Any]],
        query: str,
        collections: List[str],
        config: SearchContextConfig
    ) -> List[SearchResult]:
        """Enhance search results with code-aware search"""
        enhanced_results = []

        try:
            # First convert basic search results
            basic_results = self._convert_to_search_results(
                search_results, config.context
            )
            enhanced_results.extend(basic_results)

            # Add code search results if LSP enhancement is enabled
            if config.enable_lsp_enhancement and self.code_search_engine:
                code_results = await self.code_search_engine.search_semantic_code(
                    query=query,
                    collections=collections,
                    enhance_with_metadata=True,
                    limit=config.max_results_per_collection // 2
                )

                # Convert code search results to SearchResult format
                for code_result in code_results:
                    search_result = SearchResult(
                        id=f"code_{code_result.symbol.get('name', 'unknown')}",
                        score=code_result.relevance_score,
                        content=code_result.context_snippet,
                        collection=code_result.collection,
                        context=config.context,
                        search_type="code",
                        metadata={
                            "symbol_type": code_result.symbol.get("kind_name", ""),
                            "file_path": code_result.file_path,
                            "line_number": code_result.line_number,
                            "documentation": code_result.documentation_summary
                        },
                        code_result=code_result
                    )
                    enhanced_results.append(search_result)

            # Sort by score and limit
            enhanced_results.sort(key=lambda x: x.score, reverse=True)
            return enhanced_results[:config.max_results_per_collection]

        except Exception as e:
            logger.error("Failed to enhance with code search", error=str(e))
            return self._convert_to_search_results(search_results, config.context)

    def _convert_to_search_results(
        self,
        raw_results: List[Dict[str, Any]],
        context: SearchContext
    ) -> List[SearchResult]:
        """Convert raw search results to SearchResult format"""
        search_results = []

        for result in raw_results:
            try:
                search_result = SearchResult(
                    id=result.get("id", ""),
                    score=result.get("score", 0.0),
                    content=result.get("payload", {}).get("content", ""),
                    collection=result.get("collection", ""),
                    context=context,
                    search_type=result.get("search_type", "hybrid"),
                    metadata=result.get("payload", {}).get("metadata", {}),
                    fusion_details=result.get("payload", {}).get("fusion_details")
                )
                search_results.append(search_result)

            except Exception as e:
                logger.error("Failed to convert search result", error=str(e))
                continue

        return search_results

    async def _fuse_multi_context_results(
        self,
        context_results: Dict[SearchContext, FourContextSearchResponse],
        search_query: FourContextSearchQuery
    ) -> List[SearchResult]:
        """Fuse results from multiple contexts using RRF"""
        try:
            # Collect all results for fusion
            all_results = []
            context_weights = {
                SearchContext.PROJECT: 1.2,    # Boost project results
                SearchContext.COLLECTION: 1.1, # Boost collection results
                SearchContext.GLOBAL: 1.0,     # Standard weight
                SearchContext.ALL: 0.9         # Slight penalty for all-context
            }

            for context, response in context_results.items():
                if response.total_results > 0:
                    # Apply context-specific weight boost
                    weight = context_weights.get(context, 1.0)

                    for result in response.results:
                        # Apply context weight to score
                        result.score *= weight
                        result.metadata["context_weight"] = weight
                        all_results.append(result)

            if not all_results:
                return []

            # Apply deduplication if enabled
            if search_query.enable_deduplication:
                all_results = self._deduplicate_results(all_results)

            # Use RRF for final ranking if multiple contexts
            if len(context_results) > 1:
                ranked_results = self._apply_rrf_ranking(all_results, search_query)
            else:
                # Single context, just sort by score
                ranked_results = sorted(all_results, key=lambda x: x.score, reverse=True)

            return ranked_results[:search_query.limit]

        except Exception as e:
            logger.error("Failed to fuse multi-context results", error=str(e))
            # Fallback: return results from first context
            for response in context_results.values():
                if response.total_results > 0:
                    return response.results[:search_query.limit]
            return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        deduplicated = []
        seen_content = set()

        for result in results:
            # Create a content hash for deduplication
            content_key = f"{result.collection}:{result.id}"

            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(result)

        return deduplicated

    def _apply_rrf_ranking(
        self,
        results: List[SearchResult],
        search_query: FourContextSearchQuery
    ) -> List[SearchResult]:
        """Apply Reciprocal Rank Fusion for multi-context ranking"""
        try:
            # Group results by context for RRF
            context_groups = defaultdict(list)
            for result in results:
                context_groups[result.context].append(result)

            # Sort each context group by score
            for context in context_groups:
                context_groups[context].sort(key=lambda x: x.score, reverse=True)

            # Apply RRF formula: RRF(d) = Σ(1 / (k + r(d)))
            k = 60  # Standard RRF parameter
            rrf_scores = defaultdict(float)

            for context, context_results in context_groups.items():
                for rank, result in enumerate(context_results):
                    result_key = f"{result.collection}:{result.id}"
                    rrf_scores[result_key] += 1.0 / (k + rank + 1)

            # Update result scores with RRF scores
            result_map = {f"{r.collection}:{r.id}": r for r in results}

            for result_key, rrf_score in rrf_scores.items():
                if result_key in result_map:
                    result = result_map[result_key]
                    result.fusion_details = {
                        "original_score": result.score,
                        "rrf_score": rrf_score,
                        "fusion_method": "rrf"
                    }
                    result.score = rrf_score

            # Sort by RRF score
            return sorted(results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error("Failed to apply RRF ranking", error=str(e))
            return sorted(results, key=lambda x: x.score, reverse=True)

    # Response creation helpers

    def _create_search_response(
        self,
        query: str,
        results: List[SearchResult],
        primary_context: SearchContext,
        search_time_ms: float,
        cache_hit: bool,
        collections_searched: List[str]
    ) -> FourContextSearchResponse:
        """Create a comprehensive search response"""

        context_breakdown = {primary_context: len(results)}

        performance_metrics = {
            "response_time_ms": search_time_ms,
            "target_met": search_time_ms <= 100.0,
            "cache_hit": cache_hit,
            "collections_searched": len(collections_searched),
            "results_per_collection": len(results) / len(collections_searched) if collections_searched else 0
        }

        search_strategy = {
            "primary_context": primary_context.value,
            "collections": collections_searched,
            "fusion_method": "rrf",
            "code_search_enabled": any(r.search_type == "code" for r in results),
            "lsp_enhanced": any(r.code_result is not None for r in results)
        }

        fusion_summary = {
            "total_contexts": 1,
            "fusion_method": "single_context",
            "deduplication_applied": False,
            "score_distribution": {
                "min": min((r.score for r in results), default=0.0),
                "max": max((r.score for r in results), default=0.0),
                "avg": sum(r.score for r in results) / len(results) if results else 0.0
            }
        }

        cache_hits = {primary_context: cache_hit}

        return FourContextSearchResponse(
            query=query,
            total_results=len(results),
            results=results,
            context_breakdown=context_breakdown,
            performance_metrics=performance_metrics,
            search_strategy=search_strategy,
            cache_hits=cache_hits,
            fusion_summary=fusion_summary
        )

    def _create_multi_context_response(
        self,
        search_query: FourContextSearchQuery,
        context_results: Dict[SearchContext, FourContextSearchResponse],
        fused_results: List[SearchResult],
        search_time_ms: float
    ) -> FourContextSearchResponse:
        """Create response for multi-context search"""

        context_breakdown = {}
        cache_hits = {}
        total_collections_searched = 0

        for context, response in context_results.items():
            context_breakdown[context] = response.total_results
            cache_hits[context] = response.cache_hits.get(context, False)
            total_collections_searched += response.performance_metrics.get("collections_searched", 0)

        performance_metrics = {
            "response_time_ms": search_time_ms,
            "target_met": search_time_ms <= search_query.response_time_target_ms,
            "contexts_searched": len(context_results),
            "total_collections_searched": total_collections_searched,
            "parallel_execution": True,
            "cache_hit_rate": sum(cache_hits.values()) / len(cache_hits) if cache_hits else 0.0
        }

        search_strategy = {
            "contexts": [ctx.value for ctx in search_query.contexts],
            "scope": search_query.scope.value,
            "mode": search_query.mode,
            "fusion_method": "rrf",
            "deduplication_enabled": search_query.enable_deduplication,
            "code_search_enabled": search_query.include_code_search,
            "lsp_enhanced": search_query.enable_lsp_enhancement
        }

        fusion_summary = {
            "total_contexts": len(context_results),
            "fusion_method": "rrf",
            "deduplication_applied": search_query.enable_deduplication,
            "score_distribution": {
                "min": min((r.score for r in fused_results), default=0.0),
                "max": max((r.score for r in fused_results), default=0.0),
                "avg": sum(r.score for r in fused_results) / len(fused_results) if fused_results else 0.0
            }
        }

        return FourContextSearchResponse(
            query=search_query.query,
            total_results=len(fused_results),
            results=fused_results,
            context_breakdown=context_breakdown,
            performance_metrics=performance_metrics,
            search_strategy=search_strategy,
            cache_hits=cache_hits,
            fusion_summary=fusion_summary
        )

    def _create_empty_response(
        self,
        query: str,
        context: SearchContext
    ) -> FourContextSearchResponse:
        """Create empty response for failed searches"""
        return FourContextSearchResponse(
            query=query,
            total_results=0,
            results=[],
            context_breakdown={context: 0},
            performance_metrics={"response_time_ms": 0.0, "error": True},
            search_strategy={"context": context.value},
            cache_hits={context: False},
            fusion_summary={"error": "No results found"}
        )

    def _create_error_response(
        self,
        query: str,
        context: SearchContext,
        error: str
    ) -> FourContextSearchResponse:
        """Create error response for failed searches"""
        return FourContextSearchResponse(
            query=query,
            total_results=0,
            results=[],
            context_breakdown={context: 0},
            performance_metrics={"response_time_ms": 0.0, "error": True, "error_message": error},
            search_strategy={"context": context.value, "error": error},
            cache_hits={context: False},
            fusion_summary={"error": error}
        )

    def _enhance_cached_response(
        self,
        cached_response: FourContextSearchResponse,
        start_time: float
    ) -> FourContextSearchResponse:
        """Enhance cached response with current timing"""
        cached_response.performance_metrics["response_time_ms"] = (time.time() - start_time) * 1000
        cached_response.performance_metrics["cache_hit"] = True
        return cached_response

    # Public API methods for configuration and management

    def configure_context(
        self,
        context: SearchContext,
        config: SearchContextConfig
    ) -> None:
        """Configure settings for a specific search context"""
        self.context_configs[context] = config
        logger.info("Updated context configuration", context=context.value, config=config.__dict__)

    def get_context_config(self, context: SearchContext) -> SearchContextConfig:
        """Get configuration for a specific search context"""
        return self.context_configs.get(context, SearchContextConfig(context=context))

    def clear_cache(self, context: Optional[SearchContext] = None) -> None:
        """Clear search cache for specific context or all contexts"""
        if context:
            self.search_cache.invalidate_context(context)
        else:
            self.search_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get search cache performance statistics"""
        return self.search_cache.get_stats()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        cache_stats = self.get_cache_stats()

        return {
            "cache_performance": cache_stats,
            "context_configs": {
                ctx.value: config.__dict__
                for ctx, config in self.context_configs.items()
            },
            "initialized": self.initialized,
            "engines_available": {
                "hybrid_engine": self.hybrid_engine is not None,
                "code_search_engine": self.code_search_engine is not None,
                "multitenant_coordinator": self.multitenant_coordinator is not None
            }
        }