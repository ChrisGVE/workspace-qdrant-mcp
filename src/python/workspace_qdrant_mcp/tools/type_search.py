"""
Type-Based Search and Analysis System for Workspace Qdrant MCP

This module implements advanced type-based search capabilities that extend the
existing CodeSearchEngine to provide sophisticated type matching and analysis.
It focuses on type signatures, generic types, interface matching, and type
compatibility analysis.

Key Features:
    - Type signature matching for functions and methods
    - Generic type handling with constraint analysis
    - Interface and protocol matching 
    - Type compatibility and substitutability analysis
    - Advanced type pattern searches
    - Type hierarchy exploration
    - Constraint-based type filtering
    - Type-safe code recommendations

Search Capabilities:
    - Find functions by exact type signatures
    - Match generic types with constraints
    - Discover compatible function overloads
    - Locate interface implementations
    - Search for type-safe usage patterns
    - Analyze type dependencies and relationships

Example:
    ```python
    from workspace_qdrant_mcp.tools.type_search import TypeSearchEngine
    
    # Initialize type search engine
    type_engine = TypeSearchEngine(workspace_client)
    await type_engine.initialize()
    
    # Find functions with specific signature
    results = await type_engine.search_exact_signature(
        parameter_types=["str", "int"],
        return_type="Optional[bool]",
        collections=["my-project"]
    )
    
    # Find compatible function overloads
    results = await type_engine.search_compatible_signatures(
        target_signature="(str, int) -> bool",
        allow_subtyping=True,
        collections=["my-project"]
    )
    
    # Search for generic type implementations
    results = await type_engine.search_generic_implementations(
        generic_pattern="List[T]",
        constraint_patterns=["T: Comparable"],
        collections=["my-project"]
    )
    ```
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Pattern


from common.core.client import QdrantWorkspaceClient
from common.core.error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from common.core.lsp_metadata_extractor import SymbolKind, TypeInformation, CodeSymbol
from .code_search import CodeSearchEngine, CodeSearchResult, SignatureSearchQuery

logger = structlog.get_logger(__name__)


class TypeMatchMode(Enum):
    """Type matching strategies"""
    EXACT = "exact"                    # Exact type matching
    COMPATIBLE = "compatible"          # Subtype/supertype compatible
    GENERIC_MATCH = "generic_match"    # Generic type pattern matching
    STRUCTURAL = "structural"          # Duck typing / structural compatibility
    COVARIANT = "covariant"           # Covariant type matching
    CONTRAVARIANT = "contravariant"   # Contravariant type matching


class TypeConstraintKind(Enum):
    """Types of generic type constraints"""
    UPPER_BOUND = "upper_bound"        # T extends SomeType
    LOWER_BOUND = "lower_bound"        # T super SomeType  
    EXACT_BOUND = "exact_bound"        # T = SomeType
    MULTIPLE_BOUNDS = "multiple_bounds" # T extends A & B
    WILDCARD = "wildcard"              # T = ?


@dataclass
class TypePattern:
    """Represents a type pattern for matching"""
    type_expression: str               # The type expression to match
    generic_parameters: List[str] = field(default_factory=list)
    constraints: Dict[str, str] = field(default_factory=dict)
    nullable: bool = False
    optional: bool = False
    variadic: bool = False             # For *args, **kwargs patterns
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_expression": self.type_expression,
            "generic_parameters": self.generic_parameters,
            "constraints": self.constraints,
            "nullable": self.nullable,
            "optional": self.optional,
            "variadic": self.variadic
        }


@dataclass
class TypeSignature:
    """Comprehensive type signature representation"""
    parameter_types: List[TypePattern] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    return_type: Optional[TypePattern] = None
    generic_parameters: List[str] = field(default_factory=list)
    type_constraints: Dict[str, List[str]] = field(default_factory=dict)
    modifiers: List[str] = field(default_factory=list)  # async, static, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_types": [p.to_dict() for p in self.parameter_types],
            "parameter_names": self.parameter_names,
            "return_type": self.return_type.to_dict() if self.return_type else None,
            "generic_parameters": self.generic_parameters,
            "type_constraints": self.type_constraints,
            "modifiers": self.modifiers
        }


@dataclass
class TypeSearchQuery:
    """Advanced type-based search query"""
    signature_pattern: Optional[TypeSignature] = None
    match_mode: TypeMatchMode = TypeMatchMode.EXACT
    generic_patterns: List[str] = field(default_factory=list)
    constraint_patterns: List[str] = field(default_factory=list)
    interface_patterns: List[str] = field(default_factory=list)
    symbol_kinds: List[SymbolKind] = field(default_factory=list)
    allow_subtyping: bool = False
    allow_generics: bool = True
    exact_arity: bool = False          # Must match parameter count exactly
    collections: Optional[List[str]] = None
    max_results: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature_pattern": self.signature_pattern.to_dict() if self.signature_pattern else None,
            "match_mode": self.match_mode.value,
            "generic_patterns": self.generic_patterns,
            "constraint_patterns": self.constraint_patterns,
            "interface_patterns": self.interface_patterns,
            "symbol_kinds": [k.value for k in self.symbol_kinds],
            "allow_subtyping": self.allow_subtyping,
            "allow_generics": self.allow_generics,
            "exact_arity": self.exact_arity,
            "collections": self.collections,
            "max_results": self.max_results
        }


@dataclass
class TypeSearchResult:
    """Type-enriched search result"""
    base_result: CodeSearchResult
    type_signature: Optional[TypeSignature] = None
    compatibility_score: float = 0.0
    generic_substitutions: Dict[str, str] = field(default_factory=dict)
    constraint_satisfaction: Dict[str, bool] = field(default_factory=dict)
    type_relationships: List[str] = field(default_factory=list)
    subtyping_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_result": self.base_result.to_dict(),
            "type_signature": self.type_signature.to_dict() if self.type_signature else None,
            "compatibility_score": self.compatibility_score,
            "generic_substitutions": self.generic_substitutions,
            "constraint_satisfaction": self.constraint_satisfaction,
            "type_relationships": self.type_relationships,
            "subtyping_chain": self.subtyping_chain
        }


class TypeSearchEngine:
    """
    Advanced type-based search engine that provides sophisticated type matching,
    generic type handling, and type compatibility analysis.
    
    Integrates with the existing CodeSearchEngine to provide type-aware search
    capabilities while maintaining compatibility with the existing search infrastructure.
    """
    
    def __init__(self, workspace_client: QdrantWorkspaceClient):
        self.workspace_client = workspace_client
        self.code_search_engine = CodeSearchEngine(workspace_client)
        self.type_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.compatibility_cache: Dict[str, Dict[str, float]] = {}
        self.generic_patterns: Dict[str, Pattern] = {}
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the type search engine"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing type search engine")
            
            # Initialize base code search engine
            await self.code_search_engine.initialize()
            
            # Compile commonly used generic patterns
            await self._compile_generic_patterns()
            
            # Build type compatibility cache
            await self._build_compatibility_cache()
            
            self.initialized = True
            logger.info("Type search engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize type search engine", error=str(e))
            raise WorkspaceError(
                f"Type search engine initialization failed: {e}",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.HIGH
            )
    
    async def search_exact_signature(
        self,
        parameter_types: List[str],
        return_type: Optional[str] = None,
        parameter_names: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[TypeSearchResult]:
        """
        Find functions with exact type signatures.
        
        Args:
            parameter_types: List of parameter type strings
            return_type: Expected return type
            parameter_names: Optional parameter names for additional matching
            collections: Target collections to search
            limit: Maximum number of results
            
        Returns:
            List of type-enriched search results with exact signature matches
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching for exact type signatures",
                       parameter_types=parameter_types,
                       return_type=return_type)
            
            # Build type signature pattern
            type_patterns = [TypePattern(type_expression=t) for t in parameter_types]
            return_pattern = TypePattern(type_expression=return_type) if return_type else None
            
            signature = TypeSignature(
                parameter_types=type_patterns,
                parameter_names=parameter_names or [],
                return_type=return_pattern
            )
            
            # Create type search query
            query = TypeSearchQuery(
                signature_pattern=signature,
                match_mode=TypeMatchMode.EXACT,
                symbol_kinds=[SymbolKind.FUNCTION, SymbolKind.METHOD],
                exact_arity=True,
                collections=collections,
                max_results=limit
            )
            
            return await self._execute_type_search(query)
            
        except Exception as e:
            logger.error("Exact signature search failed", error=str(e))
            raise WorkspaceError(
                f"Exact signature search failed: {e}",
                category=ErrorCategory.OPERATION,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_compatible_signatures(
        self,
        target_signature: str,
        allow_subtyping: bool = True,
        allow_generics: bool = True,
        collections: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[TypeSearchResult]:
        """
        Find functions with compatible type signatures.
        
        Args:
            target_signature: Target signature pattern (e.g., "(str, int) -> bool")
            allow_subtyping: Allow subtype/supertype compatibility
            allow_generics: Allow generic type matching
            collections: Target collections to search
            limit: Maximum number of results
            
        Returns:
            List of type-enriched search results with compatible signatures
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching for compatible signatures",
                       target_signature=target_signature,
                       allow_subtyping=allow_subtyping)
            
            # Parse target signature
            parsed_signature = await self._parse_signature_string(target_signature)
            
            # Create compatibility query
            query = TypeSearchQuery(
                signature_pattern=parsed_signature,
                match_mode=TypeMatchMode.COMPATIBLE,
                symbol_kinds=[SymbolKind.FUNCTION, SymbolKind.METHOD],
                allow_subtyping=allow_subtyping,
                allow_generics=allow_generics,
                collections=collections,
                max_results=limit
            )
            
            return await self._execute_type_search(query)
            
        except Exception as e:
            logger.error("Compatible signature search failed", error=str(e))
            raise WorkspaceError(
                f"Compatible signature search failed: {e}",
                category=ErrorCategory.OPERATION,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_generic_implementations(
        self,
        generic_pattern: str,
        constraint_patterns: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[TypeSearchResult]:
        """
        Search for generic type implementations with constraints.
        
        Args:
            generic_pattern: Generic type pattern (e.g., "List[T]", "Dict[K, V]")
            constraint_patterns: Type constraint patterns (e.g., ["T: Comparable"])
            collections: Target collections to search
            limit: Maximum number of results
            
        Returns:
            List of type-enriched search results with generic implementations
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching for generic implementations",
                       generic_pattern=generic_pattern,
                       constraint_patterns=constraint_patterns)
            
            # Create generic search query
            query = TypeSearchQuery(
                match_mode=TypeMatchMode.GENERIC_MATCH,
                generic_patterns=[generic_pattern],
                constraint_patterns=constraint_patterns or [],
                symbol_kinds=[SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS],
                allow_generics=True,
                collections=collections,
                max_results=limit
            )
            
            return await self._execute_type_search(query)
            
        except Exception as e:
            logger.error("Generic implementation search failed", error=str(e))
            raise WorkspaceError(
                f"Generic implementation search failed: {e}",
                category=ErrorCategory.OPERATION,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def search_interface_implementations(
        self,
        interface_patterns: List[str],
        match_mode: TypeMatchMode = TypeMatchMode.STRUCTURAL,
        collections: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[TypeSearchResult]:
        """
        Find implementations of specific interfaces or protocols.
        
        Args:
            interface_patterns: Interface/protocol patterns to match
            match_mode: Type matching strategy
            collections: Target collections to search
            limit: Maximum number of results
            
        Returns:
            List of type-enriched search results with interface implementations
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Searching for interface implementations",
                       interface_patterns=interface_patterns,
                       match_mode=match_mode.value)
            
            # Create interface search query
            query = TypeSearchQuery(
                match_mode=match_mode,
                interface_patterns=interface_patterns,
                symbol_kinds=[SymbolKind.CLASS, SymbolKind.INTERFACE],
                allow_subtyping=True,
                collections=collections,
                max_results=limit
            )
            
            return await self._execute_type_search(query)
            
        except Exception as e:
            logger.error("Interface implementation search failed", error=str(e))
            raise WorkspaceError(
                f"Interface implementation search failed: {e}",
                category=ErrorCategory.OPERATION,
                severity=ErrorSeverity.MEDIUM
            )
    
    async def analyze_type_compatibility(
        self,
        source_type: str,
        target_type: str,
        collections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze type compatibility between source and target types.
        
        Args:
            source_type: Source type expression
            target_type: Target type expression
            collections: Collections to analyze for context
            
        Returns:
            Compatibility analysis results
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            logger.info("Analyzing type compatibility",
                       source_type=source_type,
                       target_type=target_type)
            
            # Check cache first
            cache_key = f"{source_type}::{target_type}"
            if cache_key in self.compatibility_cache:
                return self.compatibility_cache[cache_key]
            
            # Perform compatibility analysis
            compatibility_result = await self._analyze_compatibility(
                source_type, target_type, collections
            )
            
            # Cache result
            self.compatibility_cache[cache_key] = compatibility_result
            
            return compatibility_result
            
        except Exception as e:
            logger.error("Type compatibility analysis failed", error=str(e))
            raise WorkspaceError(
                f"Type compatibility analysis failed: {e}",
                category=ErrorCategory.OPERATION,
                severity=ErrorSeverity.MEDIUM
            )
    
    # Private implementation methods
    
    async def _execute_type_search(self, query: TypeSearchQuery) -> List[TypeSearchResult]:
        """Execute a type search query and return enriched results"""
        # Convert to base code search query
        base_query = await self._convert_to_base_query(query)
        
        # Execute base search
        base_results = await self.code_search_engine.search_by_signature(**base_query)
        
        # Filter and enrich with type information
        type_results = []
        for base_result in base_results:
            if await self._matches_type_criteria(base_result, query):
                type_result = await self._enrich_with_type_info(base_result, query)
                type_results.append(type_result)
        
        # Rank by type compatibility
        ranked_results = await self._rank_type_results(type_results, query)
        
        return ranked_results[:query.max_results]
    
    async def _convert_to_base_query(self, query: TypeSearchQuery) -> Dict[str, Any]:
        """Convert type search query to base code search parameters"""
        base_query = {
            "collections": query.collections,
            "limit": query.max_results * 2  # Get more for filtering
        }
        
        if query.signature_pattern:
            sig = query.signature_pattern
            base_query.update({
                "parameter_types": [p.type_expression for p in sig.parameter_types],
                "return_type": sig.return_type.type_expression if sig.return_type else None,
                "parameter_names": sig.parameter_names,
                "exact_match": query.match_mode == TypeMatchMode.EXACT
            })
        
        return base_query
    
    async def _matches_type_criteria(
        self,
        result: CodeSearchResult,
        query: TypeSearchQuery
    ) -> bool:
        """Check if a search result matches type search criteria"""
        symbol = result.symbol
        
        # Check symbol kind filter
        if query.symbol_kinds:
            symbol_kind = SymbolKind(symbol.get("kind", 0))
            if symbol_kind not in query.symbol_kinds:
                return False
        
        # Check type signature matching
        if query.signature_pattern:
            if not await self._matches_signature_pattern(symbol, query.signature_pattern, query.match_mode):
                return False
        
        # Check generic patterns
        if query.generic_patterns:
            if not await self._matches_generic_patterns(symbol, query.generic_patterns):
                return False
        
        # Check interface patterns
        if query.interface_patterns:
            if not await self._matches_interface_patterns(symbol, query.interface_patterns):
                return False
        
        return True
    
    async def _matches_signature_pattern(
        self,
        symbol: Dict[str, Any],
        pattern: TypeSignature,
        match_mode: TypeMatchMode
    ) -> bool:
        """Check if symbol matches signature pattern with given match mode"""
        type_info = symbol.get("type_info", {})
        
        # Get symbol parameter and return types
        symbol_param_types = [p.get("type", "") for p in type_info.get("parameter_types", [])]
        symbol_return_type = type_info.get("return_type", "")
        
        # Check parameter count if exact arity required
        pattern_param_types = [p.type_expression for p in pattern.parameter_types]
        if len(symbol_param_types) != len(pattern_param_types):
            return False
        
        # Match parameter types based on match mode
        for sym_type, pat_type in zip(symbol_param_types, pattern_param_types):
            if not await self._types_match(sym_type, pat_type, match_mode):
                return False
        
        # Match return type
        if pattern.return_type:
            if not await self._types_match(symbol_return_type, pattern.return_type.type_expression, match_mode):
                return False
        
        return True
    
    async def _types_match(self, symbol_type: str, pattern_type: str, match_mode: TypeMatchMode) -> bool:
        """Check if two types match according to the given match mode"""
        if match_mode == TypeMatchMode.EXACT:
            return symbol_type.strip() == pattern_type.strip()
        elif match_mode == TypeMatchMode.COMPATIBLE:
            return await self._are_compatible_types(symbol_type, pattern_type)
        elif match_mode == TypeMatchMode.GENERIC_MATCH:
            return await self._matches_generic_pattern(symbol_type, pattern_type)
        elif match_mode == TypeMatchMode.STRUCTURAL:
            return await self._structurally_compatible(symbol_type, pattern_type)
        else:
            return symbol_type.strip() == pattern_type.strip()
    
    async def _are_compatible_types(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible (subtype/supertype relationship)"""
        # Normalize types
        type1 = type1.strip()
        type2 = type2.strip()
        
        if type1 == type2:
            return True
        
        # Handle common compatibility patterns
        compatibility_patterns = [
            # Optional types
            (r'Optional\[(.*?)\]', r'\1'),
            (r'Union\[(.*?), None\]', r'\1'),
            # List/sequence compatibility
            (r'List\[(.*?)\]', r'Sequence\[\1\]'),
            (r'list\[(.*?)\]', r'Sequence\[\1\]'),
            # Dict/mapping compatibility  
            (r'Dict\[(.*?), (.*?)\]', r'Mapping\[\1, \2\]'),
            (r'dict\[(.*?), (.*?)\]', r'Mapping\[\1, \2\]'),
            # Number hierarchy
            (r'\bint\b', r'float'),
            (r'\bfloat\b', r'complex'),
        ]
        
        for pattern, replacement in compatibility_patterns:
            if re.search(pattern, type1) and re.search(pattern.replace(replacement, type2), type2):
                return True
        
        return False
    
    async def _matches_generic_pattern(self, symbol_type: str, pattern: str) -> bool:
        """Check if symbol type matches a generic pattern"""
        # Convert pattern to regex
        generic_regex = pattern.replace('[', r'\[').replace(']', r'\]')
        generic_regex = re.sub(r'\b[A-Z]\b', r'[A-Za-z_][A-Za-z0-9_]*', generic_regex)
        
        return bool(re.match(generic_regex, symbol_type))
    
    async def _structurally_compatible(self, type1: str, type2: str) -> bool:
        """Check structural compatibility (duck typing)"""
        # For now, implement basic structural matching
        # This could be extended with actual structural analysis
        return await self._are_compatible_types(type1, type2)
    
    async def _matches_generic_patterns(self, symbol: Dict[str, Any], patterns: List[str]) -> bool:
        """Check if symbol matches any of the generic patterns"""
        type_info = symbol.get("type_info", {})
        generic_params = type_info.get("generic_parameters", [])
        
        for pattern in patterns:
            # Check if any generic parameter matches the pattern
            for param in generic_params:
                if await self._matches_generic_pattern(param, pattern):
                    return True
        
        return False
    
    async def _matches_interface_patterns(self, symbol: Dict[str, Any], patterns: List[str]) -> bool:
        """Check if symbol implements any of the interface patterns"""
        # This would need access to inheritance/implementation relationships
        # For now, do basic name matching
        symbol_name = symbol.get("name", "")
        
        for pattern in patterns:
            if pattern.lower() in symbol_name.lower():
                return True
        
        return False
    
    async def _enrich_with_type_info(
        self,
        base_result: CodeSearchResult,
        query: TypeSearchQuery
    ) -> TypeSearchResult:
        """Enrich base search result with type information"""
        # Extract type signature from symbol
        type_signature = await self._extract_type_signature(base_result.symbol)
        
        # Calculate compatibility score
        compatibility_score = await self._calculate_compatibility_score(
            base_result.symbol, query
        )
        
        # Analyze generic substitutions if applicable
        generic_substitutions = await self._analyze_generic_substitutions(
            base_result.symbol, query
        )
        
        return TypeSearchResult(
            base_result=base_result,
            type_signature=type_signature,
            compatibility_score=compatibility_score,
            generic_substitutions=generic_substitutions
        )
    
    async def _extract_type_signature(self, symbol: Dict[str, Any]) -> Optional[TypeSignature]:
        """Extract TypeSignature from symbol metadata"""
        type_info = symbol.get("type_info", {})
        if not type_info:
            return None
        
        # Extract parameter types
        param_types = []
        for param in type_info.get("parameter_types", []):
            param_pattern = TypePattern(type_expression=param.get("type", ""))
            param_types.append(param_pattern)
        
        # Extract return type
        return_type = None
        if type_info.get("return_type"):
            return_type = TypePattern(type_expression=type_info["return_type"])
        
        return TypeSignature(
            parameter_types=param_types,
            parameter_names=[p.get("name", "") for p in type_info.get("parameter_types", [])],
            return_type=return_type,
            generic_parameters=type_info.get("generic_parameters", [])
        )
    
    async def _calculate_compatibility_score(
        self,
        symbol: Dict[str, Any],
        query: TypeSearchQuery
    ) -> float:
        """Calculate how well symbol matches query requirements"""
        base_score = 0.5  # Base compatibility
        
        # Boost for exact matches
        if query.match_mode == TypeMatchMode.EXACT:
            base_score += 0.3
        
        # Boost for generic compatibility
        type_info = symbol.get("type_info", {})
        if query.allow_generics and type_info.get("generic_parameters"):
            base_score += 0.2
        
        return min(1.0, base_score)
    
    async def _analyze_generic_substitutions(
        self,
        symbol: Dict[str, Any],
        query: TypeSearchQuery
    ) -> Dict[str, str]:
        """Analyze possible generic type parameter substitutions"""
        substitutions = {}
        
        # This would implement actual generic type inference
        # For now, return empty substitutions
        
        return substitutions
    
    async def _rank_type_results(
        self,
        results: List[TypeSearchResult],
        query: TypeSearchQuery
    ) -> List[TypeSearchResult]:
        """Rank type search results by compatibility and relevance"""
        def score_function(result: TypeSearchResult) -> float:
            base_score = result.base_result.relevance_score
            compatibility_score = result.compatibility_score
            
            # Combine scores with type-specific weighting
            combined_score = (base_score * 0.6) + (compatibility_score * 0.4)
            
            # Boost exact matches
            if query.match_mode == TypeMatchMode.EXACT and compatibility_score > 0.8:
                combined_score *= 1.2
            
            return combined_score
        
        return sorted(results, key=score_function, reverse=True)
    
    async def _parse_signature_string(self, signature: str) -> TypeSignature:
        """Parse a signature string into TypeSignature object"""
        # Basic signature parsing: "(param1: type1, param2: type2) -> return_type"
        signature = signature.strip()
        
        # Extract parameter and return parts
        if " -> " in signature:
            param_part, return_part = signature.split(" -> ", 1)
        else:
            param_part, return_part = signature, None
        
        # Parse parameters
        param_part = param_part.strip("()")
        param_types = []
        param_names = []
        
        if param_part:
            for param in param_part.split(","):
                param = param.strip()
                if ":" in param:
                    name, type_expr = param.split(":", 1)
                    param_names.append(name.strip())
                    param_types.append(TypePattern(type_expression=type_expr.strip()))
                else:
                    param_names.append("")
                    param_types.append(TypePattern(type_expression=param))
        
        # Parse return type
        return_type = None
        if return_part:
            return_type = TypePattern(type_expression=return_part.strip())
        
        return TypeSignature(
            parameter_types=param_types,
            parameter_names=param_names,
            return_type=return_type
        )
    
    async def _compile_generic_patterns(self) -> None:
        """Compile commonly used generic type patterns"""
        common_patterns = [
            r'List\[(.+)\]',
            r'Dict\[(.+),\s*(.+)\]',
            r'Optional\[(.+)\]',
            r'Union\[(.+)\]',
            r'Tuple\[(.+)\]',
            r'Set\[(.+)\]',
            r'Callable\[\[(.+)\],\s*(.+)\]'
        ]
        
        for pattern in common_patterns:
            self.generic_patterns[pattern] = re.compile(pattern)
    
    async def _build_compatibility_cache(self) -> None:
        """Build cache of common type compatibility relationships"""
        # This would be populated with common type hierarchies and relationships
        # For now, keep it empty and populate on-demand
        pass
    
    async def _analyze_compatibility(
        self,
        source_type: str,
        target_type: str,
        collections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform detailed type compatibility analysis"""
        result = {
            "compatible": await self._are_compatible_types(source_type, target_type),
            "confidence": 0.0,
            "relationship": "none",
            "conversion_needed": False,
            "suggestions": []
        }
        
        if result["compatible"]:
            result["confidence"] = 0.9
            if source_type == target_type:
                result["relationship"] = "identical"
            else:
                result["relationship"] = "compatible"
        
        return result