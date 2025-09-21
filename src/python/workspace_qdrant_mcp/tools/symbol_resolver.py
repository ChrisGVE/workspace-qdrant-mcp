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


from python.common.core.client import QdrantWorkspaceClient
from python.common.core.error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from python.common.core.lsp_metadata_extractor import (
    CodeSymbol, SymbolKind, SymbolRelationship, RelationshipType,
    Position, Range, TypeInformation, Documentation
)
from loguru import logger
# Import code search components with fallback for missing dependencies
try:
    from .code_search import CodeSearchEngine, CodeSearchResult, SymbolSearchType
except ImportError:
    # Fallback definitions when code_search dependencies aren't available
    CodeSearchEngine = None
    CodeSearchResult = None
    SymbolSearchType = None
from .search import search_workspace, search_collection_by_metadata

# logger imported from loguru


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


class CrossReferenceTracker:
    """
    Cross-reference tracking system for symbol usage analysis and impact assessment.
    
    Tracks where symbols are used, referenced, imported, and modified to support
    impact analysis, refactoring assistance, and usage pattern analysis.
    """
    
    def __init__(self):
        # Reference tracking indexes
        self.symbol_references: Dict[str, List[SymbolLocation]] = defaultdict(list)
        self.file_imports: Dict[str, List[str]] = defaultdict(list)  # file -> imported symbols
        self.symbol_dependencies: Dict[str, Set[str]] = defaultdict(set)  # symbol -> depends on
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)  # symbol -> depended by
        
        # Usage pattern tracking
        self.usage_frequency: Dict[str, int] = defaultdict(int)
        self.last_accessed: Dict[str, float] = {}
        self.reference_contexts: Dict[str, List[str]] = defaultdict(list)  # usage contexts
        
        # Performance metrics
        self.tracking_enabled = True
        self.reference_count = 0
        self.impact_analysis_count = 0
        
    def add_symbol_reference(
        self, 
        symbol_qualified_name: str, 
        reference_location: SymbolLocation,
        context: Optional[str] = None
    ) -> None:
        """Add a reference to a symbol at a specific location"""
        if not self.tracking_enabled:
            return
            
        self.symbol_references[symbol_qualified_name].append(reference_location)
        self.usage_frequency[symbol_qualified_name] += 1
        self.last_accessed[symbol_qualified_name] = time.time()
        
        if context:
            self.reference_contexts[symbol_qualified_name].append(context)
        
        self.reference_count += 1
        
        logger.debug("Symbol reference added",
                    symbol=symbol_qualified_name,
                    location=reference_location.get_range_key())
    
    def add_symbol_dependency(self, from_symbol: str, to_symbol: str) -> None:
        """Add dependency relationship between symbols"""
        self.symbol_dependencies[from_symbol].add(to_symbol)
        self.reverse_dependencies[to_symbol].add(from_symbol)
        
        logger.debug("Symbol dependency added", from_symbol=from_symbol, to_symbol=to_symbol)
    
    def get_symbol_references(self, qualified_name: str) -> List[SymbolLocation]:
        """Get all references to a symbol"""
        return self.symbol_references.get(qualified_name, [])
    
    def get_symbol_dependencies(self, qualified_name: str, recursive: bool = False) -> Set[str]:
        """Get symbols that this symbol depends on"""
        if not recursive:
            return self.symbol_dependencies.get(qualified_name, set())
        
        # Recursive dependency resolution
        visited = set()
        dependencies = set()
        
        def collect_dependencies(symbol: str):
            if symbol in visited:
                return
            visited.add(symbol)
            
            direct_deps = self.symbol_dependencies.get(symbol, set())
            dependencies.update(direct_deps)
            
            for dep in direct_deps:
                collect_dependencies(dep)
        
        collect_dependencies(qualified_name)
        return dependencies
    
    def get_symbol_dependents(self, qualified_name: str, recursive: bool = False) -> Set[str]:
        """Get symbols that depend on this symbol"""
        if not recursive:
            return self.reverse_dependencies.get(qualified_name, set())
        
        # Recursive dependent resolution
        visited = set()
        dependents = set()
        
        def collect_dependents(symbol: str):
            if symbol in visited:
                return
            visited.add(symbol)
            
            direct_deps = self.reverse_dependencies.get(symbol, set())
            dependents.update(direct_deps)
            
            for dep in direct_deps:
                collect_dependents(dep)
        
        collect_dependents(qualified_name)
        return dependents
    
    def analyze_impact(self, qualified_name: str) -> Dict[str, Any]:
        """
        Analyze the impact of modifying or removing a symbol.
        
        Returns analysis including direct and transitive dependencies,
        usage frequency, and affected files.
        """
        self.impact_analysis_count += 1
        
        references = self.get_symbol_references(qualified_name)
        direct_dependents = self.get_symbol_dependents(qualified_name, recursive=False)
        transitive_dependents = self.get_symbol_dependents(qualified_name, recursive=True)
        
        affected_files = set()
        for ref in references:
            affected_files.add(ref.file_path)
        
        usage_freq = self.usage_frequency.get(qualified_name, 0)
        last_used = self.last_accessed.get(qualified_name, 0)
        
        impact_analysis = {
            "symbol": qualified_name,
            "reference_count": len(references),
            "direct_dependents": len(direct_dependents),
            "transitive_dependents": len(transitive_dependents),
            "affected_files": len(affected_files),
            "usage_frequency": usage_freq,
            "last_accessed": last_used,
            "impact_score": self._calculate_impact_score(
                len(references), len(transitive_dependents), usage_freq
            ),
            "risk_level": self._assess_risk_level(qualified_name),
            "affected_file_list": list(affected_files),
            "dependent_symbols": list(direct_dependents)[:10]  # Top 10 dependents
        }
        
        logger.info("Impact analysis completed",
                   symbol=qualified_name,
                   impact_score=impact_analysis["impact_score"],
                   risk_level=impact_analysis["risk_level"])
        
        return impact_analysis
    
    def _calculate_impact_score(
        self, 
        reference_count: int, 
        transitive_dependents: int, 
        usage_frequency: int
    ) -> float:
        """Calculate overall impact score (0.0-1.0)"""
        # Weighted formula considering multiple factors
        ref_score = min(1.0, reference_count / 100.0)  # Normalize around 100 references
        dep_score = min(1.0, transitive_dependents / 50.0)  # Normalize around 50 dependents
        freq_score = min(1.0, usage_frequency / 200.0)  # Normalize around 200 uses
        
        # Weighted average with emphasis on dependencies
        impact_score = (ref_score * 0.3 + dep_score * 0.5 + freq_score * 0.2)
        return impact_score
    
    def _assess_risk_level(self, qualified_name: str) -> str:
        """Assess risk level of modifying this symbol"""
        references = len(self.get_symbol_references(qualified_name))
        dependents = len(self.get_symbol_dependents(qualified_name, recursive=True))
        usage_freq = self.usage_frequency.get(qualified_name, 0)
        
        if dependents > 20 or references > 50 or usage_freq > 100:
            return "HIGH"
        elif dependents > 5 or references > 15 or usage_freq > 30:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_popular_symbols(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get most frequently used symbols"""
        return sorted(
            self.usage_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cross-reference tracking statistics"""
        return {
            "total_references": self.reference_count,
            "tracked_symbols": len(self.symbol_references),
            "dependencies": len(self.symbol_dependencies),
            "impact_analyses": self.impact_analysis_count,
            "average_references_per_symbol": (
                self.reference_count / max(1, len(self.symbol_references))
            ),
            "tracking_enabled": self.tracking_enabled
        }


class SymbolResolver:
    """
    High-performance symbol definition lookup system with O(1) resolution.
    
    This is the main interface for symbol resolution, providing instant access
    to symbol definitions, locations, and context across workspace collections.
    Integrates with existing Code Search and LSP Metadata systems.
    """
    
    def __init__(
        self, 
        workspace_client: QdrantWorkspaceClient,
        enable_cross_references: bool = True,
        index_capacity: int = 10000,
        cache_size: int = 1000
    ):
        """
        Initialize symbol resolver.
        
        Args:
            workspace_client: Initialized workspace client
            enable_cross_references: Enable cross-reference tracking
            index_capacity: Initial symbol index capacity  
            cache_size: Resolution result cache size
        """
        self.workspace_client = workspace_client
        self.enable_cross_references = enable_cross_references
        
        # Core components
        self.symbol_index = SymbolIndex(index_capacity)
        self.disambiguation_engine = DisambiguationEngine()
        self.cross_ref_tracker = CrossReferenceTracker() if enable_cross_references else None
        
        # Integration with existing systems
        self.code_search_engine: Optional[CodeSearchEngine] = None
        
        # Caching and performance
        self.resolution_cache: Dict[str, List[SymbolResolutionResult]] = {}
        self.cache_size = cache_size
        self.cache_ttl = 300.0  # 5 minutes TTL
        
        # Workspace state
        self.indexed_collections: Set[str] = set()
        self.last_index_update: float = 0.0
        self.index_update_threshold = 3600.0  # 1 hour
        
        # Performance tracking
        self.resolution_count = 0
        self.cache_hits = 0
        self.total_resolution_time_ms = 0.0
        
        # Initialization state
        self.initialized = False
        
        logger.info("Symbol resolver initialized",
                   enable_cross_references=enable_cross_references,
                   index_capacity=index_capacity,
                   cache_size=cache_size)
    
    async def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the symbol resolver by building indexes from workspace collections.
        
        Args:
            force_rebuild: Force complete index rebuild
        """
        if self.initialized and not force_rebuild:
            # Check if index needs updating
            current_time = time.time()
            if current_time - self.last_index_update < self.index_update_threshold:
                return
        
        logger.info("Initializing symbol resolver", force_rebuild=force_rebuild)
        start_time = time.perf_counter()
        
        try:
            # Initialize code search engine
            self.code_search_engine = CodeSearchEngine(self.workspace_client)
            await self.code_search_engine.initialize()
            
            # Get all workspace collections
            collections = await self.workspace_client.list_collections()
            logger.info("Found collections for indexing", count=len(collections))
            
            # Build symbol index from collections
            if force_rebuild:
                self.symbol_index.clear()
                self.indexed_collections.clear()
            
            for collection in collections:
                if collection not in self.indexed_collections or force_rebuild:
                    await self._index_collection_symbols(collection)
                    self.indexed_collections.add(collection)
            
            # Build cross-references if enabled
            if self.cross_ref_tracker:
                await self._build_cross_references()
            
            self.last_index_update = time.time()
            self.initialized = True
            
            init_time = (time.perf_counter() - start_time) * 1000
            index_stats = self.symbol_index.get_statistics()
            
            logger.info("Symbol resolver initialization completed",
                       init_time_ms=init_time,
                       symbols_indexed=index_stats["index_size"],
                       collections_indexed=len(self.indexed_collections))
                       
        except Exception as e:
            logger.error("Symbol resolver initialization failed", error=str(e))
            raise WorkspaceError(
                f"Symbol resolver initialization failed: {e}",
                category=ErrorCategory.INITIALIZATION,
                severity=ErrorSeverity.HIGH
            )
    
    async def find_symbol_definitions(
        self,
        symbol_name: str,
        collections: Optional[List[str]] = None,
        symbol_kinds: Optional[List[SymbolKind]] = None,
        context_file: Optional[str] = None,
        context_line: Optional[int] = None
    ) -> List[SymbolResolutionResult]:
        """
        Find all definitions of a symbol with instant O(1) lookup.
        
        Args:
            symbol_name: Name of symbol to find
            collections: Filter by specific collections
            symbol_kinds: Filter by symbol kinds
            context_file: Context file for disambiguation
            context_line: Context line for disambiguation
            
        Returns:
            List of symbol resolution results ranked by confidence
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        self.resolution_count += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(
            symbol_name, collections, symbol_kinds, context_file
        )
        
        if cache_key in self.resolution_cache:
            cached_result = self.resolution_cache[cache_key]
            # Check cache TTL
            if cached_result and time.time() - cached_result[0].resolution_time_ms < self.cache_ttl:
                self.cache_hits += 1
                logger.debug("Cache hit for symbol resolution", symbol=symbol_name)
                return cached_result
        
        try:
            # O(1) lookup by name
            candidates = self.symbol_index.find_by_name(symbol_name)
            
            if not candidates:
                logger.debug("No candidates found for symbol", symbol=symbol_name)
                return []
            
            # Apply filters
            if collections:
                candidates = [c for c in candidates if c.location.collection in collections]
            
            if symbol_kinds:
                candidates = [c for c in candidates if c.kind in symbol_kinds]
            
            # Disambiguate candidates
            disambiguated = self.disambiguation_engine.disambiguate_symbols(
                candidates, None, None, context_file, context_line
            )
            
            # Create resolution results
            results = []
            for symbol in disambiguated:
                resolution_time = (time.perf_counter() - start_time) * 1000
                result = SymbolResolutionResult(
                    symbol=symbol,
                    match_confidence=symbol.confidence_score,
                    resolution_method="index_lookup",
                    resolution_time_ms=resolution_time
                )
                
                # Find related symbols
                result.related_symbols = await self._find_related_symbols(symbol)
                
                results.append(result)
            
            # Cache result
            self._cache_resolution_result(cache_key, results)
            
            # Track cross-references
            if self.cross_ref_tracker and context_file and context_line is not None:
                for result in results:
                    ref_location = SymbolLocation(
                        file_path=context_file,
                        file_uri=f"file://{context_file}",
                        line=context_line,
                        column=0,
                        end_line=context_line,
                        end_column=0,
                        collection="unknown"  # Could be resolved from file
                    )
                    self.cross_ref_tracker.add_symbol_reference(
                        result.symbol.qualified_name, ref_location
                    )
            
            resolution_time = (time.perf_counter() - start_time) * 1000
            self.total_resolution_time_ms += resolution_time
            
            logger.debug("Symbol definitions found",
                        symbol=symbol_name,
                        candidates=len(candidates),
                        results=len(results),
                        resolution_time_ms=resolution_time)
            
            return results
            
        except Exception as e:
            logger.error("Symbol definition lookup failed", 
                        symbol=symbol_name, 
                        error=str(e))
            raise WorkspaceError(
                f"Symbol definition lookup failed: {e}",
                category=ErrorCategory.SEARCH,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def resolve_symbol_with_params(
        self,
        symbol_name: str,
        parameter_types: List[str],
        return_type: Optional[str] = None,
        collections: Optional[List[str]] = None
    ) -> List[SymbolResolutionResult]:
        """
        Resolve symbol with specific parameter signature for disambiguation.
        
        Args:
            symbol_name: Name of symbol to resolve
            parameter_types: Expected parameter types
            return_type: Expected return type
            collections: Filter by collections
            
        Returns:
            List of resolution results ranked by signature match quality
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        # Find function overloads
        candidates = self.symbol_index.find_overloads(symbol_name)
        
        if collections:
            candidates = [c for c in candidates if c.location.collection in collections]
        
        # Score candidates by signature match
        scored_candidates = []
        for candidate in candidates:
            match_score = candidate.matches_signature(parameter_types, return_type)
            if match_score > 0.1:  # Minimum threshold
                scored_candidates.append((candidate, match_score))
        
        # Sort by match score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        results = []
        for candidate, match_score in scored_candidates:
            resolution_time = (time.perf_counter() - start_time) * 1000
            result = SymbolResolutionResult(
                symbol=candidate,
                match_confidence=match_score,
                resolution_method="signature_match",
                disambiguation_info={
                    "parameter_types": parameter_types,
                    "return_type": return_type,
                    "signature_match_score": match_score
                },
                resolution_time_ms=resolution_time
            )
            results.append(result)
        
        logger.debug("Symbol resolved with parameters",
                    symbol=symbol_name,
                    parameter_types=parameter_types,
                    results=len(results))
        
        return results
    
    async def find_all_references(
        self, 
        qualified_name: str
    ) -> List[SymbolLocation]:
        """
        Find all references to a symbol across the workspace.
        
        Args:
            qualified_name: Fully qualified symbol name
            
        Returns:
            List of all reference locations
        """
        if not self.cross_ref_tracker:
            logger.warning("Cross-reference tracking is disabled")
            return []
        
        return self.cross_ref_tracker.get_symbol_references(qualified_name)
    
    async def get_symbol_hierarchy(
        self, 
        qualified_name: str, 
        include_children: bool = True,
        include_parents: bool = True
    ) -> Dict[str, Any]:
        """
        Get symbol hierarchy including parents and children.
        
        Args:
            qualified_name: Symbol to get hierarchy for
            include_children: Include child symbols
            include_parents: Include parent symbols
            
        Returns:
            Dictionary with hierarchy information
        """
        symbol = self.symbol_index.find_by_qualified_name(qualified_name)
        if not symbol:
            return {}
        
        hierarchy = {
            "symbol": symbol.to_dict(),
            "parents": [],
            "children": []
        }
        
        # Find parents
        if include_parents and symbol.parent_symbol:
            parent_chain = []
            current_parent = symbol.parent_symbol
            
            while current_parent:
                parent_symbol = self.symbol_index.find_by_qualified_name(current_parent)
                if parent_symbol:
                    parent_chain.append(parent_symbol.to_dict())
                    current_parent = parent_symbol.parent_symbol
                else:
                    break
            
            hierarchy["parents"] = parent_chain
        
        # Find children
        if include_children:
            children = self.symbol_index.find_children(qualified_name)
            hierarchy["children"] = [child.to_dict() for child in children]
        
        logger.debug("Symbol hierarchy retrieved",
                    symbol=qualified_name,
                    parents=len(hierarchy["parents"]),
                    children=len(hierarchy["children"]))
        
        return hierarchy
    
    async def analyze_symbol_impact(self, qualified_name: str) -> Dict[str, Any]:
        """
        Analyze impact of modifying or removing a symbol.
        
        Args:
            qualified_name: Symbol to analyze
            
        Returns:
            Impact analysis including dependencies and usage
        """
        if not self.cross_ref_tracker:
            logger.warning("Cross-reference tracking is disabled")
            return {}
        
        return self.cross_ref_tracker.analyze_impact(qualified_name)
    
    async def search_symbols_by_kind(
        self,
        kind: SymbolKind,
        collections: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[SymbolEntry]:
        """
        Search symbols by kind with optional collection filtering.
        
        Args:
            kind: Symbol kind to search for
            collections: Filter by collections
            limit: Maximum results to return
            
        Returns:
            List of matching symbols
        """
        candidates = self.symbol_index.find_by_kind(kind, limit)
        
        if collections:
            candidates = [c for c in candidates if c.location.collection in collections]
        
        return candidates[:limit]
    
    async def get_collection_symbols(self, collection: str) -> List[SymbolEntry]:
        """Get all symbols from a specific collection"""
        return self.symbol_index.find_by_collection(collection)
    
    async def get_file_symbols(self, file_path: str) -> List[SymbolEntry]:
        """Get all symbols from a specific file"""
        return self.symbol_index.find_by_file(file_path)
    
    # Helper methods
    
    async def _index_collection_symbols(self, collection: str) -> None:
        """Build symbol index from collection metadata"""
        try:
            logger.info("Indexing symbols from collection", collection=collection)
            
            # Search for code symbols in the collection
            symbol_filter = {"content_type": "code_symbol"}
            results = await search_collection_by_metadata(
                self.workspace_client,
                collection, 
                symbol_filter,
                limit=10000  # Large limit for comprehensive indexing
            )
            
            symbols_added = 0
            for result in results.get("results", []):
                symbol_data = result.get("payload", {}).get("symbol", {})
                if symbol_data:
                    symbol_entry = await self._create_symbol_entry(symbol_data, collection)
                    if symbol_entry:
                        self.symbol_index.add_symbol(symbol_entry)
                        symbols_added += 1
            
            logger.info("Collection indexing completed",
                       collection=collection,
                       symbols_added=symbols_added)
                       
        except Exception as e:
            logger.error("Failed to index collection",
                        collection=collection,
                        error=str(e))
    
    async def _create_symbol_entry(
        self, 
        symbol_data: Dict[str, Any], 
        collection: str
    ) -> Optional[SymbolEntry]:
        """Create SymbolEntry from symbol metadata"""
        try:
            name = symbol_data.get("name", "")
            if not name:
                return None
            
            # Extract location information
            range_data = symbol_data.get("range", {})
            start_pos = range_data.get("start", {})
            end_pos = range_data.get("end", {})
            
            file_uri = symbol_data.get("file_uri", "")
            file_path = file_uri.replace("file://", "") if file_uri.startswith("file://") else file_uri
            
            location = SymbolLocation(
                file_path=file_path,
                file_uri=file_uri,
                line=start_pos.get("line", 0),
                column=start_pos.get("character", 0),
                end_line=end_pos.get("line", 0),
                end_column=end_pos.get("character", 0),
                collection=collection
            )
            
            # Create qualified name
            parent_symbol = symbol_data.get("parent_symbol", "")
            qualified_name = f"{parent_symbol}.{name}" if parent_symbol else name
            
            # Extract type information
            type_info = symbol_data.get("type_info", {})
            signature = type_info.get("type_signature", "")
            return_type = type_info.get("return_type", "")
            parameter_types = [
                param.get("type", "") for param in type_info.get("parameter_types", [])
            ]
            
            # Create symbol entry
            kind_value = symbol_data.get("kind", 1)
            try:
                kind = SymbolKind(kind_value)
            except ValueError:
                kind = SymbolKind.VARIABLE
            
            symbol_entry = SymbolEntry(
                name=name,
                qualified_name=qualified_name,
                symbol_id=hashlib.sha256(f"{qualified_name}:{file_path}".encode()).hexdigest()[:16],
                kind=kind,
                location=location,
                language=symbol_data.get("language", "unknown"),
                signature=signature,
                return_type=return_type if return_type else None,
                parameter_types=parameter_types,
                visibility=symbol_data.get("visibility", "public"),
                scope=SymbolScope.GLOBAL,  # Could be refined based on context
                context_before=symbol_data.get("context_before", []),
                context_after=symbol_data.get("context_after", []),
                documentation_summary=self._extract_doc_summary(symbol_data),
                parent_symbol=parent_symbol if parent_symbol else None,
                child_symbols=symbol_data.get("children", []),
                is_deprecated=symbol_data.get("deprecated", False),
                is_experimental=symbol_data.get("experimental", False)
            )
            
            return symbol_entry
            
        except Exception as e:
            logger.debug("Failed to create symbol entry", 
                        symbol_name=symbol_data.get("name", "unknown"),
                        error=str(e))
            return None
    
    def _extract_doc_summary(self, symbol_data: Dict[str, Any]) -> Optional[str]:
        """Extract brief documentation summary"""
        doc = symbol_data.get("documentation", {})
        if not doc:
            return None
        
        docstring = doc.get("docstring", "")
        if docstring:
            # Extract first sentence
            first_sentence = docstring.split('.')[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:150] + "..." if len(first_sentence) > 150 else first_sentence
        
        # Try inline comments
        inline_comments = doc.get("inline_comments", [])
        if inline_comments:
            return inline_comments[0][:100] + "..." if len(inline_comments[0]) > 100 else inline_comments[0]
        
        return None
    
    async def _build_cross_references(self) -> None:
        """Build cross-reference relationships from indexed symbols"""
        if not self.cross_ref_tracker:
            return
        
        logger.info("Building cross-reference relationships")
        
        # This would typically involve analyzing relationships from LSP metadata
        # For now, we'll implement a basic version that could be enhanced
        
        # Process all symbols to build dependency relationships
        for collection in self.indexed_collections:
            try:
                # Search for relationship data in the collection
                relationship_filter = {"content_type": "code_relationship"}
                results = await search_collection_by_metadata(
                    self.workspace_client,
                    collection,
                    relationship_filter,
                    limit=5000
                )
                
                for result in results.get("results", []):
                    relationship_data = result.get("payload", {}).get("relationship", {})
                    if relationship_data:
                        from_symbol = relationship_data.get("from_symbol", "")
                        to_symbol = relationship_data.get("to_symbol", "")
                        
                        if from_symbol and to_symbol:
                            self.cross_ref_tracker.add_symbol_dependency(from_symbol, to_symbol)
                            
            except Exception as e:
                logger.debug("Failed to build cross-references for collection",
                           collection=collection, 
                           error=str(e))
    
    async def _find_related_symbols(self, symbol: SymbolEntry) -> List[SymbolEntry]:
        """Find symbols related to the given symbol"""
        related = []
        
        # Find siblings (same parent)
        if symbol.parent_symbol:
            siblings = self.symbol_index.find_children(symbol.parent_symbol)
            for sibling in siblings:
                if sibling.qualified_name != symbol.qualified_name:
                    related.append(sibling)
                    if len(related) >= 5:  # Limit related symbols
                        break
        
        # Find symbols with similar names
        if len(related) < 5:
            similar_name_pattern = symbol.name.lower()[:4]  # First 4 chars
            name_candidates = []
            for name, symbols in self.symbol_index.by_name.items():
                if name.lower().startswith(similar_name_pattern) and name != symbol.name:
                    name_candidates.extend(symbols)
            
            # Add up to remaining slots
            remaining = 5 - len(related)
            related.extend(name_candidates[:remaining])
        
        return related
    
    def _generate_cache_key(
        self,
        symbol_name: str,
        collections: Optional[List[str]],
        symbol_kinds: Optional[List[SymbolKind]], 
        context_file: Optional[str]
    ) -> str:
        """Generate cache key for resolution request"""
        collections_str = ','.join(sorted(collections)) if collections else ''
        kinds_str = ','.join(sorted([k.name for k in symbol_kinds])) if symbol_kinds else ''
        context_str = context_file or ''
        
        key_content = f"{symbol_name}:{collections_str}:{kinds_str}:{context_str}"
        return hashlib.md5(key_content.encode()).hexdigest()[:16]
    
    def _cache_resolution_result(
        self, 
        cache_key: str, 
        results: List[SymbolResolutionResult]
    ) -> None:
        """Cache resolution results with TTL"""
        # Limit cache size
        if len(self.resolution_cache) >= self.cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.resolution_cache.keys())[:100]
            for key in oldest_keys:
                del self.resolution_cache[key]
        
        self.resolution_cache[cache_key] = results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resolver statistics"""
        cache_hit_rate = (self.cache_hits / max(1, self.resolution_count)) * 100
        avg_resolution_time = (self.total_resolution_time_ms / max(1, self.resolution_count))
        
        stats = {
            "resolution_count": self.resolution_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "average_resolution_time_ms": avg_resolution_time,
            "indexed_collections": len(self.indexed_collections),
            "last_index_update": self.last_index_update,
            "initialized": self.initialized,
            "symbol_index": self.symbol_index.get_statistics(),
            "disambiguation_engine": self.disambiguation_engine.get_statistics()
        }
        
        if self.cross_ref_tracker:
            stats["cross_reference_tracker"] = self.cross_ref_tracker.get_statistics()
        
        return stats
    
    async def shutdown(self) -> None:
        """Clean up resolver resources"""
        logger.info("Shutting down symbol resolver")
        
        self.symbol_index.clear()
        self.resolution_cache.clear()
        self.indexed_collections.clear()
        
        if self.code_search_engine:
            # Code search engine doesn't have explicit shutdown
            self.code_search_engine = None
        
        self.initialized = False
        logger.info("Symbol resolver shutdown completed")


# Convenience functions for common operations

async def find_symbol_definition(
    workspace_client: QdrantWorkspaceClient,
    symbol_name: str,
    collections: Optional[List[str]] = None,
    context_file: Optional[str] = None
) -> List[SymbolResolutionResult]:
    """
    Convenience function to find symbol definition.
    
    Args:
        workspace_client: Initialized workspace client
        symbol_name: Name of symbol to find
        collections: Filter by specific collections
        context_file: Context file for disambiguation
        
    Returns:
        List of symbol resolution results
    """
    resolver = SymbolResolver(workspace_client)
    await resolver.initialize()
    
    try:
        return await resolver.find_symbol_definitions(
            symbol_name=symbol_name,
            collections=collections,
            context_file=context_file
        )
    finally:
        await resolver.shutdown()


async def resolve_function_overload(
    workspace_client: QdrantWorkspaceClient,
    function_name: str,
    parameter_types: List[str],
    return_type: Optional[str] = None,
    collections: Optional[List[str]] = None
) -> List[SymbolResolutionResult]:
    """
    Convenience function to resolve function overloads.
    
    Args:
        workspace_client: Initialized workspace client
        function_name: Name of function to resolve
        parameter_types: Expected parameter types
        return_type: Expected return type
        collections: Filter by specific collections
        
    Returns:
        List of resolution results ranked by signature match
    """
    resolver = SymbolResolver(workspace_client)
    await resolver.initialize()
    
    try:
        return await resolver.resolve_symbol_with_params(
            symbol_name=function_name,
            parameter_types=parameter_types,
            return_type=return_type,
            collections=collections
        )
    finally:
        await resolver.shutdown()


async def analyze_symbol_usage(
    workspace_client: QdrantWorkspaceClient,
    qualified_name: str
) -> Dict[str, Any]:
    """
    Convenience function to analyze symbol usage and impact.
    
    Args:
        workspace_client: Initialized workspace client  
        qualified_name: Fully qualified symbol name
        
    Returns:
        Usage analysis including references and dependencies
    """
    resolver = SymbolResolver(workspace_client, enable_cross_references=True)
    await resolver.initialize()
    
    try:
        return await resolver.analyze_symbol_impact(qualified_name)
    finally:
        await resolver.shutdown()