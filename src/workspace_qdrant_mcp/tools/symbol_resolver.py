"""
Symbol Definition Lookup System for Workspace Qdrant MCP

This module implements a high-performance symbol resolution system that provides
instant O(1) access to symbol definitions, locations, and context across
workspace collections. It builds upon the LSP metadata extraction (Task #121)
and code search infrastructure (Task #125) to deliver fast, accurate symbol lookup.

Key Features:
    - O(1) symbol definition lookup using hash-based indexing
    - Symbol disambiguation for overloaded functions with parameter matching  
    - Cross-reference tracking for usage analysis and impact assessment
    - Symbol hierarchy navigation for inheritance and module structure
    - Workspace-wide search with project-scoped filtering
    - Context-aware symbol resolution with surrounding code
    - Performance optimization with intelligent caching

Architecture:
    - SymbolIndex: Hash-based O(1) lookup data structure
    - SymbolEntry: Optimized symbol representation for fast access
    - SymbolResolver: Main resolution engine with disambiguation
    - CrossReferenceTracker: Usage analysis and dependency tracking
    - DisambiguationEngine: Parameter-based overload resolution

Example:
    ```python
    from workspace_qdrant_mcp.tools.symbol_resolver import SymbolResolver
    
    # Initialize symbol resolver
    resolver = SymbolResolver(workspace_client)
    await resolver.initialize()
    
    # Instant symbol lookup
    definitions = await resolver.find_symbol_definitions("authenticate")
    
    # Disambiguate overloaded functions
    specific_def = await resolver.resolve_symbol_with_params(
        "authenticate", ["str", "dict"]
    )
    
    # Find all references
    references = await resolver.find_all_references("User.login")
    
    # Navigate hierarchy
    inheritance = await resolver.get_symbol_hierarchy("BaseClass")
    ```
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from ..core.client import QdrantWorkspaceClient
from ..core.error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from ..core.lsp_metadata_extractor import (
    CodeSymbol, SymbolKind, SymbolRelationship, RelationshipType, 
    Position, Range, TypeInformation, Documentation
)
from .code_search import CodeSearchEngine, CodeSearchResult, SymbolSearchType
from .search import search_workspace, search_collection_by_metadata

logger = structlog.get_logger(__name__)


class SymbolScope(Enum):
    """Symbol visibility and scope levels"""
    GLOBAL = "global"           # Module/package level symbols
    CLASS = "class"             # Class-level symbols (methods, properties)
    FUNCTION = "function"       # Function-level symbols (parameters, locals)
    BLOCK = "block"            # Block-level symbols (loop variables, etc.)


class SymbolAccessPattern(Enum):
    """Common symbol access patterns for optimization"""
    DEFINITION = "definition"   # Looking up symbol definition
    REFERENCE = "reference"     # Finding symbol references
    HIERARCHY = "hierarchy"     # Navigating class/module hierarchy
    SIGNATURE = "signature"     # Type signature lookup
    CONTEXT = "context"        # Context-aware resolution


@dataclass
class SymbolLocation:
    """Precise symbol location information"""
    file_path: str              # Absolute file path
    file_uri: str              # File URI for LSP integration
    line: int                  # 0-based line number
    column: int                # 0-based column number
    end_line: int              # End position line
    end_column: int            # End position column
    collection: str            # Source collection name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_uri": self.file_uri,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "collection": self.collection
        }
    
    def get_range_key(self) -> str:
        """Generate unique key for location range"""
        return f"{self.file_path}:{self.line}:{self.column}-{self.end_line}:{self.end_column}"


@dataclass  
class SymbolEntry:
    """
    Optimized symbol entry for fast O(1) lookups.
    
    This is the core data structure stored in the hash-based symbol index.
    It contains all essential information for instant symbol resolution while
    maintaining references to full symbol metadata when needed.
    """
    
    # Core identity
    name: str                   # Symbol name
    qualified_name: str         # Fully qualified name (namespace.class.symbol)
    symbol_id: str             # Unique identifier hash
    kind: SymbolKind           # Symbol type (function, class, variable, etc.)
    
    # Location and source
    location: SymbolLocation    # Precise location information
    language: str              # Programming language
    
    # Fast access metadata
    signature: Optional[str] = None        # Function/method signature
    return_type: Optional[str] = None      # Return type for functions
    parameter_types: List[str] = field(default_factory=list)  # Parameter types
    visibility: str = "public"             # public, private, protected
    scope: SymbolScope = SymbolScope.GLOBAL # Symbol scope level
    
    # Context and documentation
    context_before: List[str] = field(default_factory=list)  # Lines before symbol
    context_after: List[str] = field(default_factory=list)   # Lines after symbol
    documentation_summary: Optional[str] = None              # Brief documentation
    
    # Hierarchy and relationships
    parent_symbol: Optional[str] = None     # Parent symbol qualified name
    child_symbols: List[str] = field(default_factory=list)   # Child symbols
    related_symbols: List[str] = field(default_factory=list) # Related symbols
    
    # Metadata flags
    is_deprecated: bool = False
    is_experimental: bool = False
    is_exported: bool = True
    confidence_score: float = 1.0          # Resolution confidence (0.0-1.0)
    
    # Reference to full symbol data (lazy loaded)
    full_symbol: Optional[CodeSymbol] = None
    
    def get_symbol_hash(self) -> str:
        """Generate unique hash for this symbol"""
        if not hasattr(self, '_cached_hash'):
            content = f"{self.qualified_name}:{self.location.file_path}:{self.signature or ''}"
            self._cached_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._cached_hash
    
    def matches_signature(self, parameter_types: List[str], return_type: Optional[str] = None) -> float:
        """
        Calculate signature match score for disambiguation.
        
        Returns:
            Float between 0.0 and 1.0 indicating match quality
        """
        if not self.parameter_types and not parameter_types:
            # Both have no parameters
            base_score = 1.0
        elif not self.parameter_types or not parameter_types:
            # One has parameters, other doesn't
            base_score = 0.2
        else:
            # Compare parameter lists
            matches = 0
            total = max(len(self.parameter_types), len(parameter_types))
            
            for i in range(min(len(self.parameter_types), len(parameter_types))):
                self_param = self.parameter_types[i].lower()
                target_param = parameter_types[i].lower()
                
                if self_param == target_param:
                    matches += 2
                elif target_param in self_param or self_param in target_param:
                    matches += 1
            
            base_score = matches / (total * 2) if total > 0 else 0.0
        
        # Boost score if return type matches
        if return_type and self.return_type:
            if self.return_type.lower() == return_type.lower():
                base_score = min(1.0, base_score * 1.3)
            elif return_type.lower() in self.return_type.lower():
                base_score = min(1.0, base_score * 1.1)
        
        return base_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "symbol_id": self.symbol_id,
            "kind": self.kind.name,
            "location": self.location.to_dict(),
            "language": self.language,
            "signature": self.signature,
            "return_type": self.return_type,
            "parameter_types": self.parameter_types,
            "visibility": self.visibility,
            "scope": self.scope.value,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "documentation_summary": self.documentation_summary,
            "parent_symbol": self.parent_symbol,
            "child_symbols": self.child_symbols,
            "related_symbols": self.related_symbols,
            "is_deprecated": self.is_deprecated,
            "is_experimental": self.is_experimental,
            "is_exported": self.is_exported,
            "confidence_score": self.confidence_score
        }


@dataclass
class SymbolResolutionResult:
    """Result of symbol resolution with confidence and context"""
    symbol: SymbolEntry
    match_confidence: float     # How confident we are in this match (0.0-1.0)
    resolution_method: str      # How this symbol was resolved
    disambiguation_info: Optional[Dict[str, Any]] = None  # Details about disambiguation
    related_symbols: List[SymbolEntry] = field(default_factory=list)
    resolution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol.to_dict(),
            "match_confidence": self.match_confidence,
            "resolution_method": self.resolution_method,
            "disambiguation_info": self.disambiguation_info,
            "related_symbols": [s.to_dict() for s in self.related_symbols],
            "resolution_time_ms": self.resolution_time_ms
        }


class SymbolIndex:
    """
    High-performance hash-based symbol index for O(1) lookups.
    
    This class implements multiple indexing strategies to support different
    access patterns while maintaining O(1) average case performance.
    """
    
    def __init__(self, initial_capacity: int = 10000):
        """
        Initialize symbol index with specified capacity.
        
        Args:
            initial_capacity: Initial capacity for hash tables
        """
        self.initial_capacity = initial_capacity
        
        # Primary indexes for O(1) lookup
        self.by_name: Dict[str, List[SymbolEntry]] = defaultdict(list)
        self.by_qualified_name: Dict[str, SymbolEntry] = {}
        self.by_symbol_id: Dict[str, SymbolEntry] = {}
        
        # Secondary indexes for filtered lookups  
        self.by_kind: Dict[SymbolKind, List[SymbolEntry]] = defaultdict(list)
        self.by_file: Dict[str, List[SymbolEntry]] = defaultdict(list)
        self.by_collection: Dict[str, List[SymbolEntry]] = defaultdict(list)
        self.by_language: Dict[str, List[SymbolEntry]] = defaultdict(list)
        self.by_parent: Dict[str, List[SymbolEntry]] = defaultdict(list)
        
        # Signature-based indexes for disambiguation
        self.by_signature_hash: Dict[str, List[SymbolEntry]] = defaultdict(list)
        self.function_overloads: Dict[str, List[SymbolEntry]] = defaultdict(list)
        
        # Performance tracking
        self.index_size = 0
        self.lookup_count = 0
        self.hit_count = 0
        self.build_time_ms = 0.0
        
        logger.info("Symbol index initialized", initial_capacity=initial_capacity)
    
    def add_symbol(self, symbol: SymbolEntry) -> None:
        """Add symbol to all relevant indexes"""
        start_time = time.perf_counter()
        
        # Primary indexes
        self.by_name[symbol.name].append(symbol)
        self.by_qualified_name[symbol.qualified_name] = symbol
        self.by_symbol_id[symbol.symbol_id] = symbol
        
        # Secondary indexes
        self.by_kind[symbol.kind].append(symbol)
        self.by_file[symbol.location.file_path].append(symbol)
        self.by_collection[symbol.location.collection].append(symbol)
        self.by_language[symbol.language].append(symbol)
        
        if symbol.parent_symbol:
            self.by_parent[symbol.parent_symbol].append(symbol)
        
        # Signature-based indexes
        if symbol.signature:
            sig_hash = self._get_signature_hash(symbol.signature)
            self.by_signature_hash[sig_hash].append(symbol)
        
        # Function overload tracking
        if symbol.kind in [SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CONSTRUCTOR]:
            self.function_overloads[symbol.name].append(symbol)
        
        self.index_size += 1
        self.build_time_ms += (time.perf_counter() - start_time) * 1000
        
        logger.debug("Symbol added to index", 
                    name=symbol.name, 
                    qualified_name=symbol.qualified_name,
                    kind=symbol.kind.name)
    
    def find_by_name(self, name: str) -> List[SymbolEntry]:
        """O(1) lookup by symbol name"""
        self.lookup_count += 1
        results = self.by_name.get(name, [])
        if results:
            self.hit_count += 1
        return results
    
    def find_by_qualified_name(self, qualified_name: str) -> Optional[SymbolEntry]:
        """O(1) lookup by fully qualified name"""
        self.lookup_count += 1
        result = self.by_qualified_name.get(qualified_name)
        if result:
            self.hit_count += 1
        return result
    
    def find_by_kind(self, kind: SymbolKind, limit: int = 100) -> List[SymbolEntry]:
        """Find symbols by kind with optional limit"""
        self.lookup_count += 1
        results = self.by_kind.get(kind, [])[:limit]
        if results:
            self.hit_count += 1
        return results
    
    def find_by_file(self, file_path: str) -> List[SymbolEntry]:
        """Find all symbols in a file"""
        self.lookup_count += 1
        results = self.by_file.get(file_path, [])
        if results:
            self.hit_count += 1
        return results
    
    def find_by_collection(self, collection: str) -> List[SymbolEntry]:
        """Find all symbols in a collection"""
        self.lookup_count += 1
        results = self.by_collection.get(collection, [])
        if results:
            self.hit_count += 1
        return results
    
    def find_children(self, parent_qualified_name: str) -> List[SymbolEntry]:
        """Find child symbols of a parent symbol"""
        self.lookup_count += 1
        results = self.by_parent.get(parent_qualified_name, [])
        if results:
            self.hit_count += 1
        return results
    
    def find_overloads(self, function_name: str) -> List[SymbolEntry]:
        """Find all overloads of a function"""
        self.lookup_count += 1
        results = self.function_overloads.get(function_name, [])
        if results:
            self.hit_count += 1
        return results
    
    def _get_signature_hash(self, signature: str) -> str:
        """Generate hash for function signature"""
        # Normalize signature for hashing
        normalized = signature.lower().replace(' ', '')
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index performance statistics"""
        hit_rate = (self.hit_count / max(1, self.lookup_count)) * 100
        
        return {
            "index_size": self.index_size,
            "lookup_count": self.lookup_count,
            "hit_count": self.hit_count,
            "hit_rate_percent": hit_rate,
            "build_time_ms": self.build_time_ms,
            "by_name_entries": len(self.by_name),
            "by_qualified_name_entries": len(self.by_qualified_name),
            "function_overloads": len(self.function_overloads),
            "collections": len(self.by_collection),
            "languages": len(self.by_language)
        }
    
    def clear(self) -> None:
        """Clear all indexes"""
        self.by_name.clear()
        self.by_qualified_name.clear()
        self.by_symbol_id.clear()
        self.by_kind.clear()
        self.by_file.clear()
        self.by_collection.clear()
        self.by_language.clear()
        self.by_parent.clear()
        self.by_signature_hash.clear()
        self.function_overloads.clear()
        
        self.index_size = 0
        logger.info("Symbol index cleared")


class DisambiguationEngine:
    """
    Symbol disambiguation engine for resolving overloaded functions and methods.
    
    Uses parameter type matching, context analysis, and usage patterns to
    determine the most appropriate symbol when multiple candidates exist.
    """
    
    def __init__(self):
        self.disambiguation_cache: Dict[str, List[SymbolEntry]] = {}
        self.disambiguation_count = 0
        self.cache_hits = 0
        
    def disambiguate_symbols(
        self,
        candidates: List[SymbolEntry],
        parameter_types: Optional[List[str]] = None,
        return_type: Optional[str] = None,
        context_file: Optional[str] = None,
        context_line: Optional[int] = None
    ) -> List[SymbolEntry]:
        """
        Disambiguate symbol candidates based on various criteria.
        
        Args:
            candidates: List of symbol candidates to disambiguate
            parameter_types: Expected parameter types for function resolution
            return_type: Expected return type
            context_file: File context for disambiguation
            context_line: Line context for disambiguation
            
        Returns:
            List of symbols ranked by disambiguation confidence
        """
        if len(candidates) <= 1:
            return candidates
        
        self.disambiguation_count += 1
        
        # Generate cache key
        cache_key = self._generate_disambiguation_key(
            candidates, parameter_types, return_type, context_file
        )
        
        # Check cache
        if cache_key in self.disambiguation_cache:
            self.cache_hits += 1
            return self.disambiguation_cache[cache_key]
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_disambiguation_score(
                candidate, parameter_types, return_type, context_file, context_line
            )
            scored_candidates.append((candidate, score))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        result = [candidate for candidate, _ in scored_candidates]
        
        # Cache result
        self.disambiguation_cache[cache_key] = result
        
        logger.debug("Symbol disambiguation completed",
                    candidates_count=len(candidates),
                    top_score=scored_candidates[0][1] if scored_candidates else 0.0)
        
        return result
    
    def _calculate_disambiguation_score(
        self,
        candidate: SymbolEntry,
        parameter_types: Optional[List[str]],
        return_type: Optional[str],
        context_file: Optional[str],
        context_line: Optional[int]
    ) -> float:
        """Calculate disambiguation score for a symbol candidate"""
        score = 0.0
        
        # Base confidence score
        score += candidate.confidence_score * 0.3
        
        # Signature matching (most important for functions)
        if parameter_types and candidate.kind in [SymbolKind.FUNCTION, SymbolKind.METHOD]:
            signature_score = candidate.matches_signature(parameter_types, return_type)
            score += signature_score * 0.5
        
        # Context proximity (prefer symbols from same file)
        if context_file:
            if candidate.location.file_path == context_file:
                score += 0.3
            elif Path(candidate.location.file_path).parent == Path(context_file).parent:
                score += 0.1
        
        # Visibility preference (public > protected > private)
        if candidate.visibility == "public":
            score += 0.2
        elif candidate.visibility == "protected":
            score += 0.1
        
        # Deprecation penalty
        if candidate.is_deprecated:
            score -= 0.2
        
        # Export preference
        if candidate.is_exported:
            score += 0.1
        
        # Scope preference (global/class > function > block)
        scope_scores = {
            SymbolScope.GLOBAL: 0.15,
            SymbolScope.CLASS: 0.12,
            SymbolScope.FUNCTION: 0.08,
            SymbolScope.BLOCK: 0.05
        }
        score += scope_scores.get(candidate.scope, 0.0)
        
        return min(1.0, max(0.0, score))  # Clamp to [0.0, 1.0]
    
    def _generate_disambiguation_key(
        self,
        candidates: List[SymbolEntry],
        parameter_types: Optional[List[str]],
        return_type: Optional[str],
        context_file: Optional[str]
    ) -> str:
        """Generate cache key for disambiguation request"""
        candidate_ids = sorted([c.symbol_id for c in candidates])
        param_str = ','.join(parameter_types or [])
        key_content = f"{':'.join(candidate_ids)}:{param_str}:{return_type or ''}:{context_file or ''}"
        return hashlib.md5(key_content.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get disambiguation engine statistics"""
        cache_hit_rate = (self.cache_hits / max(1, self.disambiguation_count)) * 100
        
        return {
            "disambiguation_count": self.disambiguation_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "cache_size": len(self.disambiguation_cache)
        }


# Main SymbolResolver class implementation continues in next part...