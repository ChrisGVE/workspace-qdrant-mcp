"""
Advanced Code-Aware Search Interface for Workspace Qdrant MCP

This module implements sophisticated search capabilities that leverage LSP-extracted
code metadata to provide developer-friendly search experiences. It builds on the
foundation of LSP metadata extraction (Task #121) and smart ingestion router (Task #122)
to deliver code-intelligence enhanced search.

Key Features:
    - Symbol definition lookup (functions, classes, methods by name/signature)
    - Function signature search (by parameter types, return types)
    - Type-based searches (find functions returning specific types)
    - Dependency queries (imports, exports, relationships)
    - Natural language code queries with semantic + metadata fusion
    - Multi-collection search with relevance ranking
    - Result enrichment with context snippets and related symbols
    - Fuzzy matching for incomplete queries
    - Documentation synchronization checks

Search Modes:
    - 'symbol': Find specific code symbols by name/type
    - 'signature': Search by function signatures and types
    - 'dependency': Find imports, exports, and relationships
    - 'semantic': Natural language enhanced with code context
    - 'fuzzy': Approximate matching for incomplete queries
    - 'documentation': Search code documentation and comments

Example:
    ```python
    from workspace_qdrant_mcp.tools.code_search import CodeSearchEngine
    
    # Initialize code search engine
    search_engine = CodeSearchEngine(workspace_client)
    await search_engine.initialize()
    
    # Find function definitions
    results = await search_engine.search_symbols(
        query="authenticate user",
        symbol_types=["function", "method"],
        collections=["my-project"]
    )
    
    # Search by function signature
    results = await search_engine.search_by_signature(
        parameter_types=["str", "dict"],
        return_type="bool",
        collections=["my-project"]
    )
    
    # Natural language code query
    results = await search_engine.search_semantic_code(
        query="find authentication patterns with JWT tokens",
        collections=["my-project", "auth-library"]
    )
    ```
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from fuzzywuzzy import fuzz, process

from common.core.client import QdrantWorkspaceClient
from common.core.error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from common.core.hybrid_search import HybridSearchEngine
from common.core.lsp_metadata_extractor import SymbolKind, RelationshipType, CodeSymbol
from common.core.sparse_vectors import create_named_sparse_vector
from loguru import logger
from .search import search_workspace, search_collection_by_metadata

# logger imported from loguru


class CodeSearchMode(Enum):
    """Code-aware search modes"""
    SYMBOL = "symbol"                    # Find specific code symbols by name/type
    SIGNATURE = "signature"              # Search by function signatures and types
    DEPENDENCY = "dependency"            # Find imports, exports, relationships
    SEMANTIC = "semantic"                # Natural language enhanced with code context
    FUZZY = "fuzzy"                     # Approximate matching for incomplete queries
    DOCUMENTATION = "documentation"      # Search code documentation and comments
    USAGE = "usage"                     # Find usage examples and call sites


class SymbolSearchType(Enum):
    """Types of symbol searches"""
    EXACT_NAME = "exact_name"           # Exact symbol name match
    PARTIAL_NAME = "partial_name"       # Partial symbol name match
    BY_KIND = "by_kind"                 # Search by symbol kind (function, class, etc.)
    BY_SIGNATURE = "by_signature"       # Search by type signature
    BY_PARENT = "by_parent"            # Search by parent class/namespace


@dataclass
class CodeSearchQuery:
    """Represents a code-aware search query with metadata filters"""
    query: str
    mode: CodeSearchMode
    symbol_types: Optional[List[str]] = None  # Filter by symbol kinds
    languages: Optional[List[str]] = None     # Filter by programming languages
    file_patterns: Optional[List[str]] = None # Filter by file patterns
    collections: Optional[List[str]] = None   # Target collections
    include_deprecated: bool = False          # Include deprecated symbols
    include_experimental: bool = False        # Include experimental symbols
    fuzzy_threshold: int = 80                # Fuzzy matching threshold (0-100)
    max_results: int = 20                    # Maximum results to return
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode.value,
            "symbol_types": self.symbol_types,
            "languages": self.languages,
            "file_patterns": self.file_patterns,
            "collections": self.collections,
            "include_deprecated": self.include_deprecated,
            "include_experimental": self.include_experimental,
            "fuzzy_threshold": self.fuzzy_threshold,
            "max_results": self.max_results
        }


@dataclass
class CodeSearchResult:
    """Enhanced search result with code intelligence"""
    symbol: Dict[str, Any]                   # CodeSymbol data
    relevance_score: float                   # Combined relevance score
    context_snippet: str                     # Code context around the symbol
    related_symbols: List[Dict[str, Any]]    # Related symbols (imports, calls, etc.)
    usage_examples: List[str]                # Usage example snippets
    documentation_summary: Optional[str]     # Extracted documentation summary
    file_path: str                          # Source file path
    line_number: int                        # Line number in source
    collection: str                         # Source collection
    search_type: str                        # Type of match found
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "relevance_score": self.relevance_score,
            "context_snippet": self.context_snippet,
            "related_symbols": self.related_symbols,
            "usage_examples": self.usage_examples,
            "documentation_summary": self.documentation_summary,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "collection": self.collection,
            "search_type": self.search_type
        }


@dataclass
class SignatureSearchQuery:
    """Query for function signature-based searches"""
    parameter_types: Optional[List[str]] = None    # Required parameter types
    return_type: Optional[str] = None              # Required return type
    parameter_names: Optional[List[str]] = None    # Parameter name patterns
    function_name_pattern: Optional[str] = None    # Function name pattern
    visibility: Optional[str] = None               # public, private, protected
    modifiers: Optional[List[str]] = None          # static, async, etc.
    exact_match: bool = False                      # Require exact type matching
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_types": self.parameter_types,
            "return_type": self.return_type,
            "parameter_names": self.parameter_names,
            "function_name_pattern": self.function_name_pattern,
            "visibility": self.visibility,
            "modifiers": self.modifiers,
            "exact_match": self.exact_match
        }


class CodeSearchEngine:
    """
    Advanced code-aware search engine that leverages LSP metadata for 
    intelligent code searches across workspace collections.
    """
    
    def __init__(self, workspace_client: QdrantWorkspaceClient):
        self.workspace_client = workspace_client
        self.hybrid_engine: Optional[HybridSearchEngine] = None
        self.symbol_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.relationship_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the code search engine with workspace collections"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing code search engine")
            
            # Initialize hybrid search engine
            self.hybrid_engine = HybridSearchEngine(
                qdrant_client=self.workspace_client.qdrant_client,
                embedding_service=self.workspace_client.embedding_service
            )
            await self.hybrid_engine.initialize()
            
            # Build symbol and relationship caches
            await self._build_code_intelligence_caches()
            
            self.initialized = True
            logger.info("Code search engine initialized successfully",
                       symbol_cache_size=len(self.symbol_cache),
                       relationship_cache_size=len(self.relationship_cache))
                       
        except Exception as e:
            logger.error("Failed to initialize code search engine", error=str(e))
            raise WorkspaceError(
                f"Code search initialization failed: {e}",
                category=ErrorCategory.INITIALIZATION,
                severity=ErrorSeverity.HIGH
            )
    
    async def _build_code_intelligence_caches(self) -> None:
        """Build caches of code symbols and relationships from collections"""
        try:
            # Get all workspace collections
            collections = await self.workspace_client.list_collections()
            code_collections = [c for c in collections if self._is_code_collection(c)]
            
            logger.info("Building code intelligence caches",
                       total_collections=len(collections),
                       code_collections=len(code_collections))
            
            for collection in code_collections:
                await self._cache_collection_symbols(collection)
                await self._cache_collection_relationships(collection)
                
        except Exception as e:
            logger.error("Failed to build code intelligence caches", error=str(e))
            # Non-fatal error - search can still work without caches
    
    def _is_code_collection(self, collection_name: str) -> bool:
        """Check if a collection likely contains code metadata"""
        # Heuristic: collections with code-related names or metadata
        code_indicators = ['code', 'source', 'src', 'lib', 'project', 'repo']
        return any(indicator in collection_name.lower() for indicator in code_indicators)
    
    async def _cache_collection_symbols(self, collection: str) -> None:
        """Cache symbol metadata from a collection"""
        try:
            # Search for documents with symbol metadata
            symbol_filter = {"content_type": "code_symbol"}
            results = await search_collection_by_metadata(
                self.workspace_client,
                collection,
                symbol_filter,
                limit=1000
            )
            
            symbols = []
            for result in results.get("results", []):
                if "payload" in result and "symbol" in result["payload"]:
                    symbols.append(result["payload"]["symbol"])
                    
            self.symbol_cache[collection] = symbols
            logger.debug("Cached symbols for collection",
                        collection=collection,
                        symbol_count=len(symbols))
                        
        except Exception as e:
            logger.warning("Failed to cache symbols for collection",
                          collection=collection,
                          error=str(e))
    
    async def _cache_collection_relationships(self, collection: str) -> None:
        """Cache relationship metadata from a collection"""
        try:
            # Search for documents with relationship metadata
            relationship_filter = {"content_type": "code_relationship"}
            results = await search_collection_by_metadata(
                self.workspace_client,
                collection,
                relationship_filter,
                limit=1000
            )
            
            relationships = []
            for result in results.get("results", []):
                if "payload" in result and "relationship" in result["payload"]:
                    relationships.append(result["payload"]["relationship"])
                    
            self.relationship_cache[collection] = relationships
            logger.debug("Cached relationships for collection",
                        collection=collection,
                        relationship_count=len(relationships))
                        
        except Exception as e:
            logger.warning("Failed to cache relationships for collection",
                          collection=collection,
                          error=str(e))
    
    async def search_symbols(
        self,
        query: str,
        symbol_types: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        search_type: SymbolSearchType = SymbolSearchType.PARTIAL_NAME,
        limit: int = 20
    ) -> List[CodeSearchResult]:
        """
        Search for code symbols by name, type, or signature.
        
        Args:
            query: Symbol name or pattern to search for
            symbol_types: Filter by symbol kinds (function, class, method, etc.)
            collections: Target collections to search
            search_type: Type of symbol search to perform
            limit: Maximum number of results to return
            
        Returns:
            List of enriched code search results
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching for code symbols",
                       query=query,
                       symbol_types=symbol_types,
                       search_type=search_type.value)
            
            # Build search filters
            filters = await self._build_symbol_filters(
                query, symbol_types, search_type
            )
            
            # Get target collections
            target_collections = collections or await self._get_code_collections()
            
            results = []
            for collection in target_collections:
                collection_results = await self._search_symbols_in_collection(
                    collection, query, filters, search_type, limit
                )
                results.extend(collection_results)
            
            # Rank and merge results
            ranked_results = await self._rank_symbol_results(results, query, limit)
            
            # Enrich results with context and relationships
            enriched_results = []
            for result in ranked_results:
                enriched = await self._enrich_symbol_result(result)
                enriched_results.append(enriched)
            
            logger.info("Symbol search completed",
                       query=query,
                       results_found=len(enriched_results))
            
            return enriched_results
            
        except Exception as e:
            logger.error("Symbol search failed", query=query, error=str(e))
            raise WorkspaceError(
                f"Symbol search failed: {e}",
                category=ErrorCategory.SEARCH,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_by_signature(
        self,
        signature_query: SignatureSearchQuery,
        collections: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[CodeSearchResult]:
        """
        Search for functions/methods by their signatures.
        
        Args:
            signature_query: Signature search parameters
            collections: Target collections to search
            limit: Maximum number of results to return
            
        Returns:
            List of enriched code search results matching the signature
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching by function signature",
                       signature_query=signature_query.to_dict())
            
            # Get target collections
            target_collections = collections or await self._get_code_collections()
            
            results = []
            for collection in target_collections:
                collection_results = await self._search_signatures_in_collection(
                    collection, signature_query, limit
                )
                results.extend(collection_results)
            
            # Rank results by signature match quality
            ranked_results = await self._rank_signature_results(
                results, signature_query, limit
            )
            
            # Enrich results with context and usage examples
            enriched_results = []
            for result in ranked_results:
                enriched = await self._enrich_signature_result(result, signature_query)
                enriched_results.append(enriched)
            
            logger.info("Signature search completed",
                       results_found=len(enriched_results))
            
            return enriched_results
            
        except Exception as e:
            logger.error("Signature search failed", error=str(e))
            raise WorkspaceError(
                f"Signature search failed: {e}",
                category=ErrorCategory.SEARCH,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_semantic_code(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        enhance_with_metadata: bool = True,
        limit: int = 20
    ) -> List[CodeSearchResult]:
        """
        Perform semantic search enhanced with code metadata.
        
        Combines natural language understanding with code structure awareness
        to find relevant code snippets, functions, and patterns.
        
        Args:
            query: Natural language query about code functionality
            collections: Target collections to search
            enhance_with_metadata: Use code metadata to enhance search
            limit: Maximum number of results to return
            
        Returns:
            List of semantically relevant code search results
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Performing semantic code search",
                       query=query,
                       enhance_with_metadata=enhance_with_metadata)
            
            # Get target collections
            target_collections = collections or await self._get_code_collections()
            
            # Perform hybrid semantic search
            semantic_results = await search_workspace(
                client=self.workspace_client,
                query=query,
                collections=target_collections,
                mode="hybrid",
                limit=limit * 2,  # Get more results for filtering
                score_threshold=0.6
            )
            
            results = []
            for result in semantic_results.get("results", []):
                # Filter for code-related content
                if self._is_code_result(result):
                    code_result = await self._convert_to_code_result(
                        result, query, "semantic"
                    )
                    if code_result:
                        results.append(code_result)
            
            # Enhance with metadata if requested
            if enhance_with_metadata:
                results = await self._enhance_semantic_results_with_metadata(
                    results, query
                )
            
            # Rank by relevance and code-specific factors
            ranked_results = await self._rank_semantic_code_results(
                results, query, limit
            )
            
            logger.info("Semantic code search completed",
                       query=query,
                       results_found=len(ranked_results))
            
            return ranked_results[:limit]
            
        except Exception as e:
            logger.error("Semantic code search failed", query=query, error=str(e))
            raise WorkspaceError(
                f"Semantic code search failed: {e}",
                category=ErrorCategory.SEARCH,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_dependencies(
        self,
        query: str,
        dependency_types: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[CodeSearchResult]:
        """
        Search for code dependencies, imports, exports, and relationships.
        
        Args:
            query: Module/symbol name to find dependencies for
            dependency_types: Filter by relationship types (imports, exports, etc.)
            collections: Target collections to search
            limit: Maximum number of results to return
            
        Returns:
            List of dependency-related code search results
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching for code dependencies",
                       query=query,
                       dependency_types=dependency_types)
            
            # Get target collections
            target_collections = collections or await self._get_code_collections()
            
            results = []
            for collection in target_collections:
                # Search in relationship cache
                if collection in self.relationship_cache:
                    cache_results = await self._search_cached_relationships(
                        collection, query, dependency_types
                    )
                    results.extend(cache_results)
                
                # Search in collection metadata
                collection_results = await self._search_dependency_metadata(
                    collection, query, dependency_types
                )
                results.extend(collection_results)
            
            # Rank by dependency relevance
            ranked_results = await self._rank_dependency_results(
                results, query, limit
            )
            
            # Enrich with relationship context
            enriched_results = []
            for result in ranked_results:
                enriched = await self._enrich_dependency_result(result, query)
                enriched_results.append(enriched)
            
            logger.info("Dependency search completed",
                       query=query,
                       results_found=len(enriched_results))
            
            return enriched_results
            
        except Exception as e:
            logger.error("Dependency search failed", query=query, error=str(e))
            raise WorkspaceError(
                f"Dependency search failed: {e}",
                category=ErrorCategory.SEARCH,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_fuzzy(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        threshold: int = 80,
        limit: int = 20
    ) -> List[CodeSearchResult]:
        """
        Perform fuzzy matching search for incomplete or approximate queries.
        
        Args:
            query: Partial or approximate query string
            collections: Target collections to search
            threshold: Fuzzy matching threshold (0-100)
            limit: Maximum number of results to return
            
        Returns:
            List of fuzzy-matched code search results
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Performing fuzzy code search",
                       query=query,
                       threshold=threshold)
            
            # Get target collections
            target_collections = collections or await self._get_code_collections()
            
            results = []
            for collection in target_collections:
                # Search symbol cache with fuzzy matching
                if collection in self.symbol_cache:
                    cache_results = await self._fuzzy_search_symbols(
                        collection, query, threshold
                    )
                    results.extend(cache_results)
                
                # Fuzzy search in collection content
                collection_results = await self._fuzzy_search_collection(
                    collection, query, threshold
                )
                results.extend(collection_results)
            
            # Rank by fuzzy match score
            ranked_results = await self._rank_fuzzy_results(
                results, query, threshold, limit
            )
            
            # Enrich results
            enriched_results = []
            for result in ranked_results:
                enriched = await self._enrich_fuzzy_result(result, query)
                enriched_results.append(enriched)
            
            logger.info("Fuzzy search completed",
                       query=query,
                       results_found=len(enriched_results))
            
            return enriched_results
            
        except Exception as e:
            logger.error("Fuzzy search failed", query=query, error=str(e))
            raise WorkspaceError(
                f"Fuzzy search failed: {e}",
                category=ErrorCategory.SEARCH,
                severity=ErrorSeverity.MEDIUM
            )
    
    # Helper methods for search implementation
    
    async def _get_code_collections(self) -> List[str]:
        """Get list of collections that contain code metadata"""
        all_collections = await self.workspace_client.list_collections()
        code_collections = [c for c in all_collections if self._is_code_collection(c)]
        return code_collections
    
    async def _build_symbol_filters(
        self,
        query: str,
        symbol_types: Optional[List[str]],
        search_type: SymbolSearchType
    ) -> Dict[str, Any]:
        """Build metadata filters for symbol search"""
        filters = {"content_type": "code_symbol"}
        
        if symbol_types:
            # Map symbol type names to SymbolKind values
            kind_values = []
            for symbol_type in symbol_types:
                try:
                    kind = SymbolKind[symbol_type.upper()]
                    kind_values.append(kind.value)
                except KeyError:
                    logger.warning(f"Unknown symbol type: {symbol_type}")
            
            if kind_values:
                filters["symbol.kind"] = {"$in": kind_values}
        
        return filters
    
    async def _search_symbols_in_collection(
        self,
        collection: str,
        query: str,
        filters: Dict[str, Any],
        search_type: SymbolSearchType,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for symbols within a specific collection"""
        try:
            # First try cache-based search
            cache_results = []
            if collection in self.symbol_cache:
                cache_results = await self._search_cached_symbols(
                    collection, query, search_type
                )
            
            # Then search in collection with metadata filters
            metadata_results = await search_collection_by_metadata(
                self.workspace_client,
                collection,
                filters,
                limit=limit
            )
            
            # Combine and deduplicate results
            combined_results = cache_results + metadata_results.get("results", [])
            seen_ids = set()
            unique_results = []
            
            for result in combined_results:
                result_id = result.get("id")
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error("Failed to search symbols in collection",
                        collection=collection,
                        error=str(e))
            return []
    
    async def _search_cached_symbols(
        self,
        collection: str,
        query: str,
        search_type: SymbolSearchType
    ) -> List[Dict[str, Any]]:
        """Search symbols in the cache"""
        if collection not in self.symbol_cache:
            return []
        
        symbols = self.symbol_cache[collection]
        matching_symbols = []
        
        for symbol in symbols:
            match_score = self._calculate_symbol_match_score(
                symbol, query, search_type
            )
            if match_score > 0.3:  # Minimum match threshold
                result = {
                    "id": f"{collection}_{symbol.get('name', 'unknown')}",
                    "score": match_score,
                    "payload": {"symbol": symbol, "content_type": "code_symbol"},
                    "collection": collection
                }
                matching_symbols.append(result)
        
        # Sort by match score
        matching_symbols.sort(key=lambda x: x["score"], reverse=True)
        return matching_symbols
    
    def _calculate_symbol_match_score(
        self,
        symbol: Dict[str, Any],
        query: str,
        search_type: SymbolSearchType
    ) -> float:
        """Calculate match score for a symbol against query"""
        query_lower = query.lower()
        symbol_name = symbol.get("name", "").lower()
        
        if search_type == SymbolSearchType.EXACT_NAME:
            return 1.0 if symbol_name == query_lower else 0.0
        elif search_type == SymbolSearchType.PARTIAL_NAME:
            if query_lower in symbol_name:
                # Prefer exact matches, then prefix matches, then substring matches
                if symbol_name == query_lower:
                    return 1.0
                elif symbol_name.startswith(query_lower):
                    return 0.8
                else:
                    return 0.5
        elif search_type == SymbolSearchType.BY_KIND:
            kind_name = symbol.get("kind_name", "").lower()
            if query_lower in kind_name:
                return 0.7
        elif search_type == SymbolSearchType.BY_SIGNATURE:
            # Check type signature if available
            type_info = symbol.get("type_info", {})
            signature = type_info.get("type_signature", "").lower()
            if query_lower in signature:
                return 0.6
        
        return 0.0
    
    async def _search_signatures_in_collection(
        self,
        collection: str,
        signature_query: SignatureSearchQuery,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for function signatures within a collection"""
        try:
            # Build filters for signature search
            filters = {"content_type": "code_symbol"}
            
            # Filter for function-like symbols
            function_kinds = [SymbolKind.FUNCTION.value, SymbolKind.METHOD.value, 
                            SymbolKind.CONSTRUCTOR.value]
            filters["symbol.kind"] = {"$in": function_kinds}
            
            # Search with metadata filters
            results = await search_collection_by_metadata(
                self.workspace_client,
                collection,
                filters,
                limit=limit * 2  # Get more results for signature matching
            )
            
            # Filter by signature matching
            matching_results = []
            for result in results.get("results", []):
                symbol = result.get("payload", {}).get("symbol", {})
                if self._matches_signature_query(symbol, signature_query):
                    matching_results.append(result)
            
            return matching_results[:limit]
            
        except Exception as e:
            logger.error("Failed to search signatures in collection",
                        collection=collection,
                        error=str(e))
            return []
    
    def _matches_signature_query(
        self,
        symbol: Dict[str, Any],
        signature_query: SignatureSearchQuery
    ) -> bool:
        """Check if a symbol matches the signature query"""
        type_info = symbol.get("type_info", {})
        
        # Check return type
        if signature_query.return_type:
            return_type = type_info.get("return_type", "")
            if signature_query.exact_match:
                if return_type != signature_query.return_type:
                    return False
            else:
                if signature_query.return_type.lower() not in return_type.lower():
                    return False
        
        # Check parameter types
        if signature_query.parameter_types:
            param_types = [p.get("type", "") for p in type_info.get("parameter_types", [])]
            
            if signature_query.exact_match:
                if param_types != signature_query.parameter_types:
                    return False
            else:
                # Check if all required types are present (order doesn't matter)
                param_types_lower = [t.lower() for t in param_types]
                required_types_lower = [t.lower() for t in signature_query.parameter_types]
                
                for required_type in required_types_lower:
                    if not any(required_type in param_type for param_type in param_types_lower):
                        return False
        
        # Check function name pattern
        if signature_query.function_name_pattern:
            symbol_name = symbol.get("name", "").lower()
            pattern = signature_query.function_name_pattern.lower()
            if pattern not in symbol_name:
                return False
        
        # Check visibility
        if signature_query.visibility:
            symbol_visibility = symbol.get("visibility", "public").lower()
            if symbol_visibility != signature_query.visibility.lower():
                return False
        
        # Check modifiers
        if signature_query.modifiers:
            symbol_modifiers = [m.lower() for m in symbol.get("modifiers", [])]
            required_modifiers = [m.lower() for m in signature_query.modifiers]
            
            for required_modifier in required_modifiers:
                if required_modifier not in symbol_modifiers:
                    return False
        
        return True
    
    def _is_code_result(self, result: Dict[str, Any]) -> bool:
        """Check if a search result contains code-related content"""
        payload = result.get("payload", {})
        content_type = payload.get("content_type", "")
        
        # Check for explicit code content types
        if content_type in ["code_symbol", "code_relationship", "source_code"]:
            return True
        
        # Check for code-related metadata
        metadata = payload.get("metadata", {})
        file_type = metadata.get("file_type", "")
        file_path = metadata.get("file_path", "")
        
        # Check file extensions for code files
        code_extensions = {
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
            ".rs", ".go", ".rb", ".php", ".cs", ".swift", ".kt", ".scala"
        }
        
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in code_extensions:
                return True
        
        if file_type in ["python", "javascript", "typescript", "java", "cpp", "c",
                        "rust", "go", "ruby", "php", "csharp", "swift", "kotlin", "scala"]:
            return True
        
        return False
    
    async def _convert_to_code_result(
        self,
        result: Dict[str, Any],
        query: str,
        search_type: str
    ) -> Optional[CodeSearchResult]:
        """Convert a search result to a CodeSearchResult"""
        try:
            payload = result.get("payload", {})
            symbol_data = payload.get("symbol", {})
            
            # Extract basic information
            file_path = payload.get("metadata", {}).get("file_path", "")
            line_number = symbol_data.get("range", {}).get("start", {}).get("line", 0)
            collection = result.get("collection", "unknown")
            
            # Build context snippet
            context_snippet = await self._build_context_snippet(symbol_data)
            
            # Create CodeSearchResult
            code_result = CodeSearchResult(
                symbol=symbol_data,
                relevance_score=result.get("score", 0.0),
                context_snippet=context_snippet,
                related_symbols=[],  # Will be populated in enrichment
                usage_examples=[],   # Will be populated in enrichment
                documentation_summary=self._extract_documentation_summary(symbol_data),
                file_path=file_path,
                line_number=line_number,
                collection=collection,
                search_type=search_type
            )
            
            return code_result
            
        except Exception as e:
            logger.error("Failed to convert search result to CodeSearchResult",
                        error=str(e))
            return None
    
    async def _build_context_snippet(self, symbol_data: Dict[str, Any]) -> str:
        """Build a context snippet around the symbol"""
        context_before = symbol_data.get("context_before", [])
        context_after = symbol_data.get("context_after", [])
        symbol_name = symbol_data.get("name", "")
        
        context_lines = []
        
        # Add context before
        context_lines.extend(context_before)
        
        # Add symbol line (or approximation)
        if not context_before and not context_after:
            # Fallback: create a minimal context from symbol info
            kind_name = symbol_data.get("kind_name", "").lower()
            if kind_name in ["function", "method"]:
                type_info = symbol_data.get("type_info", {})
                signature = type_info.get("type_signature", "")
                context_lines.append(signature or f"def {symbol_name}():")
            elif kind_name == "class":
                context_lines.append(f"class {symbol_name}:")
            else:
                context_lines.append(f"{symbol_name}")
        
        # Add context after
        context_lines.extend(context_after)
        
        return "\n".join(context_lines).strip()
    
    def _extract_documentation_summary(self, symbol_data: Dict[str, Any]) -> Optional[str]:
        """Extract a summary from symbol documentation"""
        documentation = symbol_data.get("documentation", {})
        if not documentation:
            return None
        
        docstring = documentation.get("docstring", "")
        if docstring:
            # Extract first sentence or first line as summary
            first_sentence = docstring.split('.')[0].strip()
            if len(first_sentence) > 10:  # Reasonable length for summary
                return first_sentence[:200] + "..." if len(first_sentence) > 200 else first_sentence
        
        # Try inline comments
        inline_comments = documentation.get("inline_comments", [])
        if inline_comments:
            return inline_comments[0][:100] + "..." if len(inline_comments[0]) > 100 else inline_comments[0]
        
        return None
    
    # Ranking methods for different search types
    
    async def _rank_symbol_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Rank symbol search results by relevance"""
        if not results:
            return []
        
        # Calculate composite scores
        for result in results:
            symbol = result.get("payload", {}).get("symbol", {})
            base_score = result.get("score", 0.0)
            
            # Boost exact name matches
            symbol_name = symbol.get("name", "").lower()
            if symbol_name == query.lower():
                base_score *= 1.5
            elif symbol_name.startswith(query.lower()):
                base_score *= 1.2
            
            # Boost function/class symbols over variables
            kind_name = symbol.get("kind_name", "").lower()
            if kind_name in ["function", "method", "class"]:
                base_score *= 1.1
            
            # Boost public symbols
            visibility = symbol.get("visibility", "public").lower()
            if visibility == "public":
                base_score *= 1.05
            
            result["composite_score"] = base_score
        
        # Sort by composite score
        ranked_results = sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True)
        return ranked_results[:limit]
    
    async def _rank_signature_results(
        self,
        results: List[Dict[str, Any]],
        signature_query: SignatureSearchQuery,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Rank signature search results by match quality"""
        if not results:
            return []
        
        for result in results:
            symbol = result.get("payload", {}).get("symbol", {})
            base_score = result.get("score", 0.0)
            
            # Calculate signature match quality
            match_quality = self._calculate_signature_match_quality(symbol, signature_query)
            composite_score = base_score * (1.0 + match_quality)
            
            result["composite_score"] = composite_score
        
        ranked_results = sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True)
        return ranked_results[:limit]
    
    def _calculate_signature_match_quality(
        self,
        symbol: Dict[str, Any],
        signature_query: SignatureSearchQuery
    ) -> float:
        """Calculate how well a symbol matches the signature query"""
        quality_score = 0.0
        type_info = symbol.get("type_info", {})
        
        # Exact parameter type matches boost score significantly
        if signature_query.parameter_types:
            param_types = [p.get("type", "") for p in type_info.get("parameter_types", [])]
            if signature_query.exact_match and param_types == signature_query.parameter_types:
                quality_score += 0.8
            elif not signature_query.exact_match:
                matches = sum(1 for req_type in signature_query.parameter_types
                            if any(req_type.lower() in p_type.lower() for p_type in param_types))
                if signature_query.parameter_types:
                    quality_score += 0.6 * (matches / len(signature_query.parameter_types))
        
        # Exact return type match
        if signature_query.return_type:
            return_type = type_info.get("return_type", "")
            if signature_query.exact_match and return_type == signature_query.return_type:
                quality_score += 0.3
            elif not signature_query.exact_match and signature_query.return_type.lower() in return_type.lower():
                quality_score += 0.2
        
        # Function name pattern match
        if signature_query.function_name_pattern:
            symbol_name = symbol.get("name", "")
            if signature_query.function_name_pattern.lower() in symbol_name.lower():
                quality_score += 0.1
        
        return quality_score
    
    async def _rank_semantic_code_results(
        self,
        results: List[CodeSearchResult],
        query: str,
        limit: int
    ) -> List[CodeSearchResult]:
        """Rank semantic search results with code-specific factors"""
        if not results:
            return []
        
        for result in results:
            base_score = result.relevance_score
            
            # Boost based on symbol importance
            symbol = result.symbol
            kind_name = symbol.get("kind_name", "").lower()
            
            if kind_name in ["function", "method"]:
                base_score *= 1.2
            elif kind_name == "class":
                base_score *= 1.1
            
            # Boost if documentation matches query
            if result.documentation_summary:
                query_words = set(query.lower().split())
                doc_words = set(result.documentation_summary.lower().split())
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    base_score *= (1.0 + 0.1 * overlap)
            
            # Boost public symbols
            visibility = symbol.get("visibility", "public").lower()
            if visibility == "public":
                base_score *= 1.05
            
            result.relevance_score = base_score
        
        # Sort by enhanced relevance score
        ranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return ranked_results[:limit]
    
    async def _rank_dependency_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Rank dependency search results by relevance"""
        if not results:
            return []
        
        for result in results:
            base_score = result.get("score", 0.0)
            
            # Boost direct imports/exports
            relationship_type = result.get("relationship_type", "")
            if relationship_type in ["imports", "exports"]:
                base_score *= 1.3
            elif relationship_type == "calls":
                base_score *= 1.1
            
            result["composite_score"] = base_score
        
        ranked_results = sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True)
        return ranked_results[:limit]
    
    async def _rank_fuzzy_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        threshold: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Rank fuzzy search results by match quality"""
        if not results:
            return []
        
        # Calculate fuzzy match scores
        for result in results:
            symbol = result.get("payload", {}).get("symbol", {})
            symbol_name = symbol.get("name", "")
            
            # Use fuzzywuzzy for better fuzzy matching
            fuzzy_score = fuzz.ratio(query.lower(), symbol_name.lower())
            composite_score = result.get("score", 0.0) * (fuzzy_score / 100.0)
            
            result["fuzzy_score"] = fuzzy_score
            result["composite_score"] = composite_score
        
        # Filter by threshold and sort
        filtered_results = [r for r in results if r.get("fuzzy_score", 0) >= threshold]
        ranked_results = sorted(filtered_results, key=lambda x: x.get("composite_score", 0), reverse=True)
        
        return ranked_results[:limit]
    
    # Result enrichment methods
    
    async def _enrich_symbol_result(self, result: Dict[str, Any]) -> CodeSearchResult:
        """Enrich a symbol result with additional context and relationships"""
        code_result = await self._convert_to_code_result(result, "", "symbol")
        if not code_result:
            return None
        
        # Find related symbols
        code_result.related_symbols = await self._find_related_symbols(code_result.symbol)
        
        # Find usage examples
        code_result.usage_examples = await self._find_usage_examples(code_result.symbol)
        
        return code_result
    
    async def _enrich_signature_result(
        self,
        result: Dict[str, Any],
        signature_query: SignatureSearchQuery
    ) -> CodeSearchResult:
        """Enrich a signature result with additional context"""
        code_result = await self._convert_to_code_result(result, "", "signature")
        if not code_result:
            return None
        
        # Find related symbols with similar signatures
        code_result.related_symbols = await self._find_similar_signatures(
            code_result.symbol, signature_query
        )
        
        # Find usage examples
        code_result.usage_examples = await self._find_usage_examples(code_result.symbol)
        
        return code_result
    
    async def _enrich_dependency_result(
        self,
        result: Dict[str, Any],
        query: str
    ) -> CodeSearchResult:
        """Enrich a dependency result with relationship context"""
        code_result = await self._convert_to_code_result(result, query, "dependency")
        if not code_result:
            return None
        
        # Find dependency chain
        code_result.related_symbols = await self._find_dependency_chain(code_result.symbol, query)
        
        return code_result
    
    async def _enrich_fuzzy_result(
        self,
        result: Dict[str, Any],
        query: str
    ) -> CodeSearchResult:
        """Enrich a fuzzy search result"""
        code_result = await self._convert_to_code_result(result, query, "fuzzy")
        if not code_result:
            return None
        
        # Add fuzzy match information to context
        fuzzy_score = result.get("fuzzy_score", 0)
        if fuzzy_score < 100:
            original_snippet = code_result.context_snippet
            code_result.context_snippet = f"# Fuzzy match ({fuzzy_score}% similarity)\n{original_snippet}"
        
        return code_result
    
    async def _find_related_symbols(self, symbol: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find symbols related to the given symbol"""
        related = []
        
        try:
            symbol_name = symbol.get("name", "")
            parent_symbol = symbol.get("parent_symbol")
            
            # Find symbols in the same class/namespace
            if parent_symbol:
                for collection, symbols in self.symbol_cache.items():
                    for cached_symbol in symbols:
                        if (cached_symbol.get("parent_symbol") == parent_symbol and 
                            cached_symbol.get("name") != symbol_name):
                            related.append(cached_symbol)
                            if len(related) >= 5:  # Limit related symbols
                                break
                    if len(related) >= 5:
                        break
            
            # Find child symbols
            children = symbol.get("children", [])
            for child_name in children:
                for collection, symbols in self.symbol_cache.items():
                    for cached_symbol in symbols:
                        if cached_symbol.get("name") == child_name:
                            related.append(cached_symbol)
                            break
        
        except Exception as e:
            logger.error("Failed to find related symbols", error=str(e))
        
        return related
    
    async def _find_similar_signatures(
        self,
        symbol: Dict[str, Any],
        signature_query: SignatureSearchQuery
    ) -> List[Dict[str, Any]]:
        """Find symbols with similar signatures"""
        similar = []
        
        try:
            symbol_type_info = symbol.get("type_info", {})
            
            for collection, symbols in self.symbol_cache.items():
                for cached_symbol in symbols:
                    if cached_symbol.get("name") == symbol.get("name"):
                        continue  # Skip the same symbol
                    
                    if self._has_similar_signature(cached_symbol, symbol_type_info):
                        similar.append(cached_symbol)
                        if len(similar) >= 3:  # Limit similar signatures
                            break
                if len(similar) >= 3:
                    break
        
        except Exception as e:
            logger.error("Failed to find similar signatures", error=str(e))
        
        return similar
    
    def _has_similar_signature(
        self,
        symbol: Dict[str, Any],
        target_type_info: Dict[str, Any]
    ) -> bool:
        """Check if two symbols have similar signatures"""
        symbol_type_info = symbol.get("type_info", {})
        
        # Compare return types
        target_return = target_type_info.get("return_type", "").lower()
        symbol_return = symbol_type_info.get("return_type", "").lower()
        
        if target_return and symbol_return and target_return == symbol_return:
            return True
        
        # Compare parameter count
        target_params = target_type_info.get("parameter_types", [])
        symbol_params = symbol_type_info.get("parameter_types", [])
        
        if len(target_params) == len(symbol_params) and len(target_params) > 0:
            return True
        
        return False
    
    async def _find_usage_examples(self, symbol: Dict[str, Any]) -> List[str]:
        """Find usage examples for a symbol"""
        examples = []
        
        try:
            symbol_name = symbol.get("name", "")
            kind_name = symbol.get("kind_name", "").lower()
            
            # Generate basic usage examples based on symbol type
            if kind_name == "function":
                type_info = symbol.get("type_info", {})
                params = type_info.get("parameter_types", [])
                param_names = [p.get("name", f"arg{i}") for i, p in enumerate(params)]
                
                if param_names:
                    example = f"{symbol_name}({', '.join(param_names)})"
                else:
                    example = f"{symbol_name}()"
                examples.append(example)
                
            elif kind_name == "class":
                examples.append(f"{symbol_name}()")
                examples.append(f"instance = {symbol_name}()")
                
            elif kind_name == "method":
                parent = symbol.get("parent_symbol", "obj")
                examples.append(f"{parent}.{symbol_name}()")
                
            elif kind_name in ["variable", "constant"]:
                examples.append(f"value = {symbol_name}")
        
        except Exception as e:
            logger.error("Failed to generate usage examples", error=str(e))
        
        return examples
    
    async def _find_dependency_chain(
        self,
        symbol: Dict[str, Any],
        query: str
    ) -> List[Dict[str, Any]]:
        """Find the dependency chain for a symbol"""
        chain = []
        
        try:
            # Search through relationship cache
            for collection, relationships in self.relationship_cache.items():
                for relationship in relationships:
                    source = relationship.get("source", "")
                    target = relationship.get("target", "")
                    rel_type = relationship.get("type", "")
                    
                    if query.lower() in source.lower() or query.lower() in target.lower():
                        chain.append({
                            "relationship_type": rel_type,
                            "source": source,
                            "target": target,
                            "collection": collection
                        })
                        
                        if len(chain) >= 5:  # Limit chain length
                            break
                if len(chain) >= 5:
                    break
        
        except Exception as e:
            logger.error("Failed to find dependency chain", error=str(e))
        
        return chain
    
    # Fuzzy search implementations
    
    async def _fuzzy_search_symbols(
        self,
        collection: str,
        query: str,
        threshold: int
    ) -> List[Dict[str, Any]]:
        """Perform fuzzy search in symbol cache"""
        if collection not in self.symbol_cache:
            return []
        
        symbols = self.symbol_cache[collection]
        matches = []
        
        # Extract symbol names for fuzzy matching
        symbol_names = [(symbol, symbol.get("name", "")) for symbol in symbols]
        
        # Use process.extract for efficient fuzzy matching
        try:
            fuzzy_matches = process.extract(
                query,
                [name for _, name in symbol_names],
                limit=20,
                score_cutoff=threshold
            )
            
            # Convert back to result format
            for match_name, score in fuzzy_matches:
                # Find the original symbol
                for symbol, name in symbol_names:
                    if name == match_name:
                        result = {
                            "id": f"{collection}_fuzzy_{symbol.get('name', 'unknown')}",
                            "score": score / 100.0,  # Normalize to 0-1
                            "fuzzy_score": score,
                            "payload": {"symbol": symbol, "content_type": "code_symbol"},
                            "collection": collection
                        }
                        matches.append(result)
                        break
        
        except Exception as e:
            logger.error("Fuzzy search failed", collection=collection, error=str(e))
        
        return matches
    
    async def _fuzzy_search_collection(
        self,
        collection: str,
        query: str,
        threshold: int
    ) -> List[Dict[str, Any]]:
        """Perform fuzzy search in collection content"""
        # This would typically involve more sophisticated fuzzy text search
        # For now, we'll do a basic implementation
        try:
            # Use a lower score threshold for semantic search as fallback
            semantic_results = await search_workspace(
                client=self.workspace_client,
                query=query,
                collections=[collection],
                mode="hybrid",
                limit=10,
                score_threshold=0.4
            )
            
            # Convert to fuzzy search format
            fuzzy_results = []
            for result in semantic_results.get("results", []):
                # Calculate fuzzy score based on content similarity
                content = result.get("payload", {}).get("content", "")
                fuzzy_score = fuzz.partial_ratio(query.lower(), content.lower())
                
                if fuzzy_score >= threshold:
                    result["fuzzy_score"] = fuzzy_score
                    fuzzy_results.append(result)
            
            return fuzzy_results
        
        except Exception as e:
            logger.error("Collection fuzzy search failed", collection=collection, error=str(e))
            return []
    
    # Dependency and relationship search methods
    
    async def _search_cached_relationships(
        self,
        collection: str,
        query: str,
        dependency_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Search cached relationships for dependencies"""
        if collection not in self.relationship_cache:
            return []
        
        relationships = self.relationship_cache[collection]
        matches = []
        
        for relationship in relationships:
            if self._matches_dependency_query(relationship, query, dependency_types):
                result = {
                    "id": f"{collection}_rel_{len(matches)}",
                    "score": 0.8,  # Fixed score for cached relationships
                    "payload": {"relationship": relationship, "content_type": "code_relationship"},
                    "collection": collection,
                    "relationship_type": relationship.get("type", "unknown")
                }
                matches.append(result)
        
        return matches
    
    async def _search_dependency_metadata(
        self,
        collection: str,
        query: str,
        dependency_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Search collection metadata for dependencies"""
        try:
            # Build filters for dependency search
            filters = {"content_type": "code_relationship"}
            
            if dependency_types:
                filters["relationship.type"] = {"$in": dependency_types}
            
            results = await search_collection_by_metadata(
                self.workspace_client,
                collection,
                filters,
                limit=50
            )
            
            # Filter by query match
            filtered_results = []
            for result in results.get("results", []):
                relationship = result.get("payload", {}).get("relationship", {})
                if self._matches_dependency_query(relationship, query, dependency_types):
                    result["relationship_type"] = relationship.get("type", "unknown")
                    filtered_results.append(result)
            
            return filtered_results
        
        except Exception as e:
            logger.error("Dependency metadata search failed",
                        collection=collection,
                        error=str(e))
            return []
    
    def _matches_dependency_query(
        self,
        relationship: Dict[str, Any],
        query: str,
        dependency_types: Optional[List[str]]
    ) -> bool:
        """Check if a relationship matches the dependency query"""
        # Check dependency type filter
        if dependency_types:
            rel_type = relationship.get("type", "")
            if rel_type not in dependency_types:
                return False
        
        # Check query match in source or target
        query_lower = query.lower()
        source = relationship.get("source", "").lower()
        target = relationship.get("target", "").lower()
        
        return query_lower in source or query_lower in target
    
    async def _enhance_semantic_results_with_metadata(
        self,
        results: List[CodeSearchResult],
        query: str
    ) -> List[CodeSearchResult]:
        """Enhance semantic search results with code metadata"""
        enhanced_results = []
        
        for result in results:
            try:
                # Enhance with type information
                if result.symbol:
                    type_info = result.symbol.get("type_info", {})
                    if type_info:
                        # Add type information to context
                        type_summary = self._create_type_summary(type_info)
                        if type_summary:
                            result.context_snippet = f"{type_summary}\n\n{result.context_snippet}"
                
                # Enhance with related symbols
                if not result.related_symbols:
                    result.related_symbols = await self._find_related_symbols(result.symbol)
                
                enhanced_results.append(result)
                
            except Exception as e:
                logger.error("Failed to enhance result with metadata", error=str(e))
                # Add the original result even if enhancement fails
                enhanced_results.append(result)
        
        return enhanced_results
    
    def _create_type_summary(self, type_info: Dict[str, Any]) -> Optional[str]:
        """Create a summary of type information"""
        summary_parts = []
        
        # Add type signature if available
        type_signature = type_info.get("type_signature")
        if type_signature:
            summary_parts.append(f"Signature: {type_signature}")
        
        # Add return type
        return_type = type_info.get("return_type")
        if return_type:
            summary_parts.append(f"Returns: {return_type}")
        
        # Add parameter info
        param_types = type_info.get("parameter_types", [])
        if param_types:
            param_summary = ", ".join([f"{p.get('name', 'param')}: {p.get('type', 'unknown')}" 
                                     for p in param_types])
            summary_parts.append(f"Parameters: {param_summary}")
        
        return " | ".join(summary_parts) if summary_parts else None


# High-level convenience functions for common search patterns

async def search_code_symbols(
    workspace_client: QdrantWorkspaceClient,
    symbol_name: str,
    symbol_types: Optional[List[str]] = None,
    collections: Optional[List[str]] = None,
    exact_match: bool = False
) -> List[CodeSearchResult]:
    """
    Convenience function for searching code symbols.
    
    Args:
        workspace_client: Initialized workspace client
        symbol_name: Name or pattern of symbol to search for
        symbol_types: Filter by symbol kinds (function, class, method, etc.)
        collections: Target collections to search
        exact_match: Whether to require exact name matching
        
    Returns:
        List of code search results
    """
    search_engine = CodeSearchEngine(workspace_client)
    await search_engine.initialize()
    
    search_type = SymbolSearchType.EXACT_NAME if exact_match else SymbolSearchType.PARTIAL_NAME
    
    return await search_engine.search_symbols(
        query=symbol_name,
        symbol_types=symbol_types,
        collections=collections,
        search_type=search_type
    )


async def search_function_signatures(
    workspace_client: QdrantWorkspaceClient,
    parameter_types: Optional[List[str]] = None,
    return_type: Optional[str] = None,
    function_name_pattern: Optional[str] = None,
    collections: Optional[List[str]] = None,
    exact_match: bool = False
) -> List[CodeSearchResult]:
    """
    Convenience function for searching by function signatures.
    
    Args:
        workspace_client: Initialized workspace client
        parameter_types: Required parameter types
        return_type: Required return type
        function_name_pattern: Function name pattern
        collections: Target collections to search
        exact_match: Whether to require exact type matching
        
    Returns:
        List of code search results matching the signature
    """
    search_engine = CodeSearchEngine(workspace_client)
    await search_engine.initialize()
    
    signature_query = SignatureSearchQuery(
        parameter_types=parameter_types,
        return_type=return_type,
        function_name_pattern=function_name_pattern,
        exact_match=exact_match
    )
    
    return await search_engine.search_by_signature(
        signature_query=signature_query,
        collections=collections
    )


async def search_code_semantically(
    workspace_client: QdrantWorkspaceClient,
    query: str,
    collections: Optional[List[str]] = None,
    enhance_with_metadata: bool = True
) -> List[CodeSearchResult]:
    """
    Convenience function for semantic code search.
    
    Args:
        workspace_client: Initialized workspace client
        query: Natural language query about code functionality
        collections: Target collections to search
        enhance_with_metadata: Use code metadata to enhance search
        
    Returns:
        List of semantically relevant code search results
    """
    search_engine = CodeSearchEngine(workspace_client)
    await search_engine.initialize()
    
    return await search_engine.search_semantic_code(
        query=query,
        collections=collections,
        enhance_with_metadata=enhance_with_metadata
    )