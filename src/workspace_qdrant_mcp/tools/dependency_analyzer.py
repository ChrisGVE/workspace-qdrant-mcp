"""
Relationship and Dependency Query Engine for Workspace Qdrant MCP

This module implements a comprehensive dependency analysis system that provides
deep insights into code relationships, call graphs, inheritance hierarchies,
and refactoring impact analysis. It builds upon the LSP metadata extraction
(Task #121), code search interface (Task #125), and symbol resolver (Task #126).

Key Features:
    - Call graph analysis for function usage tracking and impact assessment
    - Import dependency tracking with circular dependency detection
    - Inheritance hierarchy analysis with method override detection
    - Module/package dependency visualization with external library tracking
    - Refactoring impact analysis showing potential breakage points
    - Relationship queries like 'what calls this function' and 'what depends on this class'
    - Graph traversal algorithms for dependency analysis
    - Visualization helpers for dependency graphs

Architecture:
    - DependencyNode: Represents code entities in dependency graphs
    - DependencyGraph: Base graph structure for all relationship types
    - CallGraph: Function and method call relationship tracking
    - ImportGraph: Module and package import dependency tracking
    - InheritanceGraph: Class inheritance and interface implementation
    - ImpactAnalyzer: Refactoring impact assessment engine
    - DependencyQueryEngine: High-level query interface
    - DependencyAnalyzer: Main orchestrating analysis engine

Example:
    ```python
    from workspace_qdrant_mcp.tools.dependency_analyzer import DependencyAnalyzer
    
    # Initialize dependency analyzer
    analyzer = DependencyAnalyzer(workspace_client)
    await analyzer.initialize()
    
    # Find what calls a function
    callers = await analyzer.query_engine.find_callers("authenticate")
    
    # Find what a function calls
    callees = await analyzer.query_engine.find_callees("process_login")
    
    # Check for circular dependencies
    cycles = await analyzer.import_graph.find_circular_dependencies()
    
    # Analyze refactoring impact
    impact = await analyzer.impact_analyzer.analyze_function_change(
        "User.authenticate", collections=["my-project"]
    )
    
    # Get inheritance hierarchy
    hierarchy = await analyzer.inheritance_graph.get_class_hierarchy("BaseAuth")
    ```
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
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
from .symbol_resolver import SymbolResolver, SymbolLocation

logger = structlog.get_logger(__name__)


class DependencyType(Enum):
    """Types of dependencies for analysis"""
    CALL = "call"                    # Function/method call dependencies
    IMPORT = "import"                # Module import dependencies  
    INHERITANCE = "inheritance"      # Class inheritance dependencies
    COMPOSITION = "composition"      # Object composition dependencies
    REFERENCE = "reference"          # Variable/symbol reference dependencies
    DEFINITION = "definition"        # Symbol definition dependencies


class ImpactLevel(Enum):
    """Impact severity levels for refactoring analysis"""
    LOW = "low"                     # Minor changes, backward compatible
    MEDIUM = "medium"               # Moderate changes, may require updates
    HIGH = "high"                   # Major changes, likely breaking
    CRITICAL = "critical"           # Critical changes, definitely breaking


class CircularDependencyType(Enum):
    """Types of circular dependencies"""
    IMPORT = "import"               # Circular import dependencies
    INHERITANCE = "inheritance"     # Circular inheritance (invalid)
    CALL = "call"                  # Circular call dependencies (recursion)
    COMPOSITION = "composition"     # Circular composition dependencies


@dataclass
class DependencyNode:
    """Represents a code entity in dependency graphs"""
    identifier: str                 # Unique identifier (fully qualified name)
    symbol_kind: SymbolKind         # Type of symbol (function, class, etc.)
    location: SymbolLocation        # Source location information
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __hash__(self) -> int:
        return hash(self.identifier)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DependencyNode):
            return False
        return self.identifier == other.identifier
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "symbol_kind": self.symbol_kind.name if isinstance(self.symbol_kind, SymbolKind) else str(self.symbol_kind),
            "location": self.location.to_dict(),
            "metadata": self.metadata
        }


@dataclass
class DependencyEdge:
    """Represents a relationship between dependency nodes"""
    source: DependencyNode          # Source node
    target: DependencyNode          # Target node
    dependency_type: DependencyType # Type of dependency
    relationship_data: Dict[str, Any] = field(default_factory=dict)  # Additional relationship info
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.identifier,
            "target": self.target.identifier,
            "dependency_type": self.dependency_type.value,
            "relationship_data": self.relationship_data
        }


@dataclass
class CircularDependency:
    """Represents a circular dependency cycle"""
    cycle_type: CircularDependencyType
    nodes: List[DependencyNode]      # Nodes involved in the cycle
    severity: ImpactLevel            # Severity of the circular dependency
    resolution_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_type": self.cycle_type.value,
            "nodes": [node.identifier for node in self.nodes],
            "severity": self.severity.value,
            "resolution_suggestions": self.resolution_suggestions
        }


@dataclass
class RefactoringImpact:
    """Analysis of potential impact from code changes"""
    target_symbol: str              # Symbol being changed
    impact_level: ImpactLevel       # Overall impact severity
    affected_symbols: List[DependencyNode]  # Symbols that would be affected
    breaking_changes: List[str]     # Specific breaking changes identified
    suggested_migrations: List[str] = field(default_factory=list)  # Migration suggestions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_symbol": self.target_symbol,
            "impact_level": self.impact_level.value,
            "affected_symbols": [node.to_dict() for node in self.affected_symbols],
            "breaking_changes": self.breaking_changes,
            "suggested_migrations": self.suggested_migrations
        }


class DependencyGraph(ABC):
    """Base class for all dependency graph types"""
    
    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: List[DependencyEdge] = []
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self._last_updated = None
    
    def add_node(self, node: DependencyNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.identifier] = node
        if node.identifier not in self.adjacency_list:
            self.adjacency_list[node.identifier] = set()
        if node.identifier not in self.reverse_adjacency_list:
            self.reverse_adjacency_list[node.identifier] = set()
    
    def add_edge(self, edge: DependencyEdge) -> None:
        """Add an edge to the graph"""
        self.edges.append(edge)
        self.adjacency_list[edge.source.identifier].add(edge.target.identifier)
        self.reverse_adjacency_list[edge.target.identifier].add(edge.source.identifier)
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get direct neighbors (targets) of a node"""
        return self.adjacency_list.get(node_id, set())
    
    def get_reverse_neighbors(self, node_id: str) -> Set[str]:
        """Get reverse neighbors (sources) of a node"""
        return self.reverse_adjacency_list.get(node_id, set())
    
    def find_paths(self, source_id: str, target_id: str, max_depth: int = 10) -> List[List[str]]:
        """Find all paths between two nodes with maximum depth"""
        paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target_id:
                paths.append(path + [current])
                return
            if current in path:  # Avoid cycles in path finding
                return
                
            for neighbor in self.get_neighbors(current):
                dfs(neighbor, path + [current], depth + 1)
        
        dfs(source_id, [], 0)
        return paths
    
    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs_cycles(node: str, path: List[str]):
            if node in rec_stack:
                # Found a cycle - extract the cycle part
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) > 2:  # Ignore self-loops for now
                    cycles.append(cycle)
                return
            
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.get_neighbors(node):
                dfs_cycles(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs_cycles(node_id, [])
        
        return cycles
    
    @abstractmethod
    async def build_graph(self, collections: Optional[List[str]] = None) -> None:
        """Build the dependency graph from workspace data"""
        pass
    
    def clear(self) -> None:
        """Clear all graph data"""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency_list.clear()
        self.reverse_adjacency_list.clear()
        self._last_updated = None


class CallGraph(DependencyGraph):
    """Call graph for function and method call dependencies"""
    
    def __init__(self, workspace_client: QdrantWorkspaceClient, symbol_resolver: SymbolResolver):
        super().__init__()
        self.workspace_client = workspace_client
        self.symbol_resolver = symbol_resolver
    
    async def build_graph(self, collections: Optional[List[str]] = None) -> None:
        """Build call graph from LSP call relationship data"""
        try:
            logger.info("Building call graph", collections=collections)
            
            # Search for all symbols with call relationships
            search_result = await search_workspace(
                self.workspace_client,
                query="*",  # Get all documents
                collections=collections,
                mode="sparse",
                limit=10000
            )
            
            processed_symbols = set()
            
            for result in search_result.get("results", []):
                if "lsp_metadata" not in result.get("payload", {}):
                    continue
                
                lsp_data = result["payload"]["lsp_metadata"]
                if not isinstance(lsp_data, dict):
                    continue
                
                # Extract symbols and their relationships
                for symbol_data in lsp_data.get("symbols", []):
                    if not isinstance(symbol_data, dict):
                        continue
                    
                    symbol_id = symbol_data.get("identifier", "")
                    if not symbol_id or symbol_id in processed_symbols:
                        continue
                    
                    processed_symbols.add(symbol_id)
                    
                    # Create node for this symbol
                    if symbol_data.get("kind") in ["function", "method"]:
                        location = SymbolLocation(
                            file_path=result["payload"].get("file_path", ""),
                            file_uri=result["payload"].get("file_uri", ""),
                            line=symbol_data.get("range", {}).get("start", {}).get("line", 0),
                            column=symbol_data.get("range", {}).get("start", {}).get("character", 0),
                            end_line=symbol_data.get("range", {}).get("end", {}).get("line", 0),
                            end_column=symbol_data.get("range", {}).get("end", {}).get("character", 0),
                            collection=result["collection"]
                        )
                        
                        node = DependencyNode(
                            identifier=symbol_id,
                            symbol_kind=SymbolKind.FUNCTION if symbol_data.get("kind") == "function" else SymbolKind.METHOD,
                            location=location,
                            metadata={
                                "signature": symbol_data.get("signature", ""),
                                "return_type": symbol_data.get("return_type", ""),
                                "parameters": symbol_data.get("parameters", [])
                            }
                        )
                        
                        self.add_node(node)
                        
                        # Process call relationships
                        for relationship in symbol_data.get("relationships", []):
                            if relationship.get("type") == "calls":
                                target_id = relationship.get("target", "")
                                if target_id:
                                    # We may not have the target node yet, create a placeholder
                                    if target_id not in self.nodes:
                                        target_location = SymbolLocation(
                                            file_path="",
                                            file_uri="",
                                            line=0,
                                            column=0,
                                            end_line=0,
                                            end_column=0,
                                            collection=""
                                        )
                                        target_node = DependencyNode(
                                            identifier=target_id,
                                            symbol_kind=SymbolKind.FUNCTION,  # Default assumption
                                            location=target_location
                                        )
                                        self.add_node(target_node)
                                    
                                    # Create call edge
                                    edge = DependencyEdge(
                                        source=node,
                                        target=self.nodes[target_id],
                                        dependency_type=DependencyType.CALL,
                                        relationship_data={
                                            "call_site": relationship.get("location", {}),
                                            "arguments": relationship.get("arguments", [])
                                        }
                                    )
                                    self.add_edge(edge)
            
            self._last_updated = time.time()
            logger.info("Call graph built successfully", 
                       nodes=len(self.nodes), 
                       edges=len(self.edges))
            
        except Exception as e:
            logger.error("Failed to build call graph", error=str(e))
            raise WorkspaceError(
                f"Failed to build call graph: {str(e)}",
                ErrorCategory.ANALYSIS_ERROR,
                ErrorSeverity.HIGH
            )
    
    async def find_callers(self, function_name: str, collections: Optional[List[str]] = None) -> List[DependencyNode]:
        """Find all functions that call the specified function"""
        if not self.nodes:
            await self.build_graph(collections)
        
        callers = []
        for caller_id in self.get_reverse_neighbors(function_name):
            if caller_id in self.nodes:
                callers.append(self.nodes[caller_id])
        
        return callers
    
    async def find_callees(self, function_name: str, collections: Optional[List[str]] = None) -> List[DependencyNode]:
        """Find all functions called by the specified function"""
        if not self.nodes:
            await self.build_graph(collections)
        
        callees = []
        for callee_id in self.get_neighbors(function_name):
            if callee_id in self.nodes:
                callees.append(self.nodes[callee_id])
        
        return callees
    
    async def find_call_chains(self, source_function: str, target_function: str, 
                              max_depth: int = 5, collections: Optional[List[str]] = None) -> List[List[str]]:
        """Find all call chains from source to target function"""
        if not self.nodes:
            await self.build_graph(collections)
        
        return self.find_paths(source_function, target_function, max_depth)
    
    async def find_recursive_calls(self, collections: Optional[List[str]] = None) -> List[List[str]]:
        """Find all recursive call cycles"""
        if not self.nodes:
            await self.build_graph(collections)
        
        cycles = self.find_cycles()
        return [cycle for cycle in cycles if len(cycle) > 1]  # Filter out direct self-calls


class ImportGraph(DependencyGraph):
    """Import dependency graph for module and package relationships"""
    
    def __init__(self, workspace_client: QdrantWorkspaceClient, symbol_resolver: SymbolResolver):
        super().__init__()
        self.workspace_client = workspace_client
        self.symbol_resolver = symbol_resolver
        self.external_dependencies: Set[str] = set()
    
    async def build_graph(self, collections: Optional[List[str]] = None) -> None:
        """Build import dependency graph from LSP import data"""
        try:
            logger.info("Building import graph", collections=collections)
            
            # Search for all documents with import metadata
            search_result = await search_workspace(
                self.workspace_client,
                query="*",
                collections=collections,
                mode="sparse",
                limit=10000
            )
            
            processed_modules = set()
            
            for result in search_result.get("results", []):
                if "lsp_metadata" not in result.get("payload", {}):
                    continue
                
                lsp_data = result["payload"]["lsp_metadata"]
                if not isinstance(lsp_data, dict):
                    continue
                
                file_path = result["payload"].get("file_path", "")
                collection = result["collection"]
                
                # Create module node
                module_id = self._get_module_identifier(file_path)
                if module_id not in processed_modules:
                    processed_modules.add(module_id)
                    
                    location = SymbolLocation(
                        file_path=file_path,
                        file_uri=result["payload"].get("file_uri", ""),
                        line=0,
                        column=0,
                        end_line=0,
                        end_column=0,
                        collection=collection
                    )
                    
                    module_node = DependencyNode(
                        identifier=module_id,
                        symbol_kind=SymbolKind.MODULE,
                        location=location,
                        metadata={
                            "file_path": file_path,
                            "language": lsp_data.get("language", "unknown")
                        }
                    )
                    
                    self.add_node(module_node)
                
                # Process import relationships
                for symbol_data in lsp_data.get("symbols", []):
                    if not isinstance(symbol_data, dict):
                        continue
                    
                    for relationship in symbol_data.get("relationships", []):
                        if relationship.get("type") == "imports":
                            imported_module = relationship.get("target", "")
                            if imported_module:
                                # Check if it's an external dependency
                                is_external = self._is_external_dependency(imported_module, collections)
                                if is_external:
                                    self.external_dependencies.add(imported_module)
                                
                                # Create or get target module node
                                if imported_module not in self.nodes:
                                    target_location = SymbolLocation(
                                        file_path="",
                                        file_uri="",
                                        line=0,
                                        column=0,
                                        end_line=0,
                                        end_column=0,
                                        collection="external" if is_external else collection
                                    )
                                    target_node = DependencyNode(
                                        identifier=imported_module,
                                        symbol_kind=SymbolKind.MODULE,
                                        location=target_location,
                                        metadata={"external": is_external}
                                    )
                                    self.add_node(target_node)
                                
                                # Create import edge
                                edge = DependencyEdge(
                                    source=module_node,
                                    target=self.nodes[imported_module],
                                    dependency_type=DependencyType.IMPORT,
                                    relationship_data={
                                        "import_type": relationship.get("import_type", "standard"),
                                        "symbols": relationship.get("symbols", []),
                                        "alias": relationship.get("alias", "")
                                    }
                                )
                                self.add_edge(edge)
            
            self._last_updated = time.time()
            logger.info("Import graph built successfully",
                       nodes=len(self.nodes),
                       edges=len(self.edges),
                       external_deps=len(self.external_dependencies))
            
        except Exception as e:
            logger.error("Failed to build import graph", error=str(e))
            raise WorkspaceError(
                f"Failed to build import graph: {str(e)}",
                ErrorCategory.ANALYSIS_ERROR,
                ErrorSeverity.HIGH
            )
    
    def _get_module_identifier(self, file_path: str) -> str:
        """Convert file path to module identifier"""
        path = Path(file_path)
        if path.suffix in ['.py']:
            # Python module path
            parts = path.parts
            # Remove common prefixes and file extension
            if 'src' in parts:
                src_index = parts.index('src')
                parts = parts[src_index + 1:]
            return '.'.join(parts).replace('.py', '')
        elif path.suffix in ['.js', '.ts']:
            # JavaScript/TypeScript module path
            return str(path).replace('\\', '/')
        else:
            return str(path)
    
    def _is_external_dependency(self, module_name: str, collections: Optional[List[str]]) -> bool:
        """Determine if a module is an external dependency"""
        # Simple heuristics for external dependencies
        external_patterns = [
            # Python standard library and third-party
            'os', 'sys', 'json', 'time', 'asyncio', 'typing', 'dataclasses',
            'numpy', 'pandas', 'requests', 'flask', 'django', 'fastapi',
            # JavaScript/Node.js
            'react', 'vue', 'angular', 'express', 'lodash', 'axios',
            # Common prefixes
            '@', 'node_modules'
        ]
        
        return any(module_name.startswith(pattern) for pattern in external_patterns)
    
    async def find_circular_dependencies(self, collections: Optional[List[str]] = None) -> List[CircularDependency]:
        """Find circular import dependencies"""
        if not self.nodes:
            await self.build_graph(collections)
        
        cycles = self.find_cycles()
        circular_deps = []
        
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            
            # Convert cycle node IDs to nodes
            cycle_nodes = []
            for node_id in cycle[:-1]:  # Remove duplicate at end
                if node_id in self.nodes:
                    cycle_nodes.append(self.nodes[node_id])
            
            if not cycle_nodes:
                continue
            
            # Determine severity based on cycle length and type
            severity = ImpactLevel.LOW
            if len(cycle_nodes) == 2:
                severity = ImpactLevel.HIGH  # Direct circular imports are serious
            elif len(cycle_nodes) > 5:
                severity = ImpactLevel.MEDIUM  # Long cycles are complex but manageable
            
            # Generate resolution suggestions
            suggestions = []
            if len(cycle_nodes) == 2:
                suggestions.append("Extract common functionality to a separate module")
                suggestions.append("Use late imports (import inside functions)")
                suggestions.append("Redesign module interfaces to remove circular dependency")
            else:
                suggestions.append("Analyze dependency chain and identify unnecessary imports")
                suggestions.append("Consider dependency inversion principle")
                suggestions.append("Extract shared abstractions to break cycles")
            
            circular_dep = CircularDependency(
                cycle_type=CircularDependencyType.IMPORT,
                nodes=cycle_nodes,
                severity=severity,
                resolution_suggestions=suggestions
            )
            circular_deps.append(circular_dep)
        
        return circular_deps
    
    async def get_module_dependencies(self, module_name: str, 
                                    include_external: bool = True,
                                    collections: Optional[List[str]] = None) -> Dict[str, List[DependencyNode]]:
        """Get all dependencies for a specific module"""
        if not self.nodes:
            await self.build_graph(collections)
        
        if module_name not in self.nodes:
            return {"imports": [], "imported_by": []}
        
        imports = []
        for import_id in self.get_neighbors(module_name):
            if import_id in self.nodes:
                node = self.nodes[import_id]
                if include_external or not node.metadata.get("external", False):
                    imports.append(node)
        
        imported_by = []
        for importer_id in self.get_reverse_neighbors(module_name):
            if importer_id in self.nodes:
                node = self.nodes[importer_id]
                if include_external or not node.metadata.get("external", False):
                    imported_by.append(node)
        
        return {
            "imports": imports,
            "imported_by": imported_by
        }


class InheritanceGraph(DependencyGraph):
    """Inheritance hierarchy graph for class relationships"""
    
    def __init__(self, workspace_client: QdrantWorkspaceClient, symbol_resolver: SymbolResolver):
        super().__init__()
        self.workspace_client = workspace_client
        self.symbol_resolver = symbol_resolver
        self.method_overrides: Dict[str, List[str]] = defaultdict(list)
    
    async def build_graph(self, collections: Optional[List[str]] = None) -> None:
        """Build inheritance graph from LSP class hierarchy data"""
        try:
            logger.info("Building inheritance graph", collections=collections)
            
            # Search for class symbols with inheritance relationships
            search_result = await search_workspace(
                self.workspace_client,
                query="*",
                collections=collections,
                mode="sparse",
                limit=10000
            )
            
            processed_classes = set()
            
            for result in search_result.get("results", []):
                if "lsp_metadata" not in result.get("payload", {}):
                    continue
                
                lsp_data = result["payload"]["lsp_metadata"]
                if not isinstance(lsp_data, dict):
                    continue
                
                # Process class symbols and their inheritance
                for symbol_data in lsp_data.get("symbols", []):
                    if not isinstance(symbol_data, dict):
                        continue
                    
                    symbol_id = symbol_data.get("identifier", "")
                    symbol_kind = symbol_data.get("kind", "")
                    
                    if symbol_kind not in ["class", "interface"] or not symbol_id:
                        continue
                    
                    if symbol_id in processed_classes:
                        continue
                    
                    processed_classes.add(symbol_id)
                    
                    # Create class node
                    location = SymbolLocation(
                        file_path=result["payload"].get("file_path", ""),
                        file_uri=result["payload"].get("file_uri", ""),
                        line=symbol_data.get("range", {}).get("start", {}).get("line", 0),
                        column=symbol_data.get("range", {}).get("start", {}).get("character", 0),
                        end_line=symbol_data.get("range", {}).get("end", {}).get("line", 0),
                        end_column=symbol_data.get("range", {}).get("end", {}).get("character", 0),
                        collection=result["collection"]
                    )
                    
                    class_node = DependencyNode(
                        identifier=symbol_id,
                        symbol_kind=SymbolKind.CLASS if symbol_kind == "class" else SymbolKind.INTERFACE,
                        location=location,
                        metadata={
                            "methods": symbol_data.get("methods", []),
                            "properties": symbol_data.get("properties", []),
                            "abstract": symbol_data.get("abstract", False),
                            "accessibility": symbol_data.get("accessibility", "public")
                        }
                    )
                    
                    self.add_node(class_node)
                    
                    # Process inheritance relationships
                    for relationship in symbol_data.get("relationships", []):
                        rel_type = relationship.get("type", "")
                        if rel_type in ["extends", "implements"]:
                            parent_id = relationship.get("target", "")
                            if parent_id:
                                # Create or get parent node
                                if parent_id not in self.nodes:
                                    parent_location = SymbolLocation(
                                        file_path="",
                                        file_uri="",
                                        line=0,
                                        column=0,
                                        end_line=0,
                                        end_column=0,
                                        collection=""
                                    )
                                    parent_node = DependencyNode(
                                        identifier=parent_id,
                                        symbol_kind=SymbolKind.CLASS,
                                        location=parent_location
                                    )
                                    self.add_node(parent_node)
                                
                                # Create inheritance edge
                                edge = DependencyEdge(
                                    source=class_node,
                                    target=self.nodes[parent_id],
                                    dependency_type=DependencyType.INHERITANCE,
                                    relationship_data={
                                        "inheritance_type": rel_type,
                                        "access_modifier": relationship.get("access_modifier", "public")
                                    }
                                )
                                self.add_edge(edge)
                    
                    # Process method overrides
                    for method in symbol_data.get("methods", []):
                        if method.get("overrides"):
                            parent_method = method.get("overrides")
                            self.method_overrides[parent_method].append(f"{symbol_id}.{method.get('name', '')}")
            
            self._last_updated = time.time()
            logger.info("Inheritance graph built successfully",
                       classes=len(self.nodes),
                       inheritance_edges=len(self.edges),
                       method_overrides=len(self.method_overrides))
            
        except Exception as e:
            logger.error("Failed to build inheritance graph", error=str(e))
            raise WorkspaceError(
                f"Failed to build inheritance graph: {str(e)}",
                ErrorCategory.ANALYSIS_ERROR,
                ErrorSeverity.HIGH
            )
    
    async def get_class_hierarchy(self, class_name: str, 
                                collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get complete inheritance hierarchy for a class"""
        if not self.nodes:
            await self.build_graph(collections)
        
        if class_name not in self.nodes:
            return {}
        
        # Find all ancestors (parents)
        ancestors = []
        visited_ancestors = set()
        
        def find_ancestors(node_id: str, depth: int = 0):
            if node_id in visited_ancestors or depth > 10:  # Prevent infinite recursion
                return
            visited_ancestors.add(node_id)
            
            for parent_id in self.get_neighbors(node_id):
                if parent_id in self.nodes:
                    ancestors.append({
                        "class": self.nodes[parent_id],
                        "depth": depth + 1
                    })
                    find_ancestors(parent_id, depth + 1)
        
        find_ancestors(class_name)
        
        # Find all descendants (children)
        descendants = []
        visited_descendants = set()
        
        def find_descendants(node_id: str, depth: int = 0):
            if node_id in visited_descendants or depth > 10:
                return
            visited_descendants.add(node_id)
            
            for child_id in self.get_reverse_neighbors(node_id):
                if child_id in self.nodes:
                    descendants.append({
                        "class": self.nodes[child_id],
                        "depth": depth + 1
                    })
                    find_descendants(child_id, depth + 1)
        
        find_descendants(class_name)
        
        return {
            "target_class": self.nodes[class_name].to_dict(),
            "ancestors": [item["class"].to_dict() for item in sorted(ancestors, key=lambda x: x["depth"])],
            "descendants": [item["class"].to_dict() for item in sorted(descendants, key=lambda x: x["depth"])],
            "method_overrides": self.method_overrides.get(class_name, [])
        }
    
    async def find_method_overrides(self, method_name: str, 
                                  collections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find all overrides of a specific method"""
        if not self.nodes:
            await self.build_graph(collections)
        
        overrides = []
        for base_method, override_list in self.method_overrides.items():
            if method_name in base_method:
                for override_method in override_list:
                    class_name = override_method.split('.')[0]
                    if class_name in self.nodes:
                        overrides.append({
                            "base_method": base_method,
                            "override_method": override_method,
                            "class": self.nodes[class_name].to_dict()
                        })
        
        return overrides
    
    async def find_interface_implementations(self, interface_name: str,
                                          collections: Optional[List[str]] = None) -> List[DependencyNode]:
        """Find all classes that implement a specific interface"""
        if not self.nodes:
            await self.build_graph(collections)
        
        implementations = []
        for child_id in self.get_reverse_neighbors(interface_name):
            if child_id in self.nodes:
                # Check if the relationship is 'implements'
                for edge in self.edges:
                    if (edge.target.identifier == interface_name and 
                        edge.source.identifier == child_id and 
                        edge.relationship_data.get("inheritance_type") == "implements"):
                        implementations.append(edge.source)
                        break
        
        return implementations


class ImpactAnalyzer:
    """Analyzes refactoring impact using dependency graphs"""
    
    def __init__(self, call_graph: CallGraph, import_graph: ImportGraph, 
                 inheritance_graph: InheritanceGraph):
        self.call_graph = call_graph
        self.import_graph = import_graph
        self.inheritance_graph = inheritance_graph
    
    async def analyze_function_change(self, function_name: str, 
                                    change_type: str = "signature",
                                    collections: Optional[List[str]] = None) -> RefactoringImpact:
        """Analyze impact of changing a function"""
        logger.info("Analyzing function change impact", 
                   function=function_name, 
                   change_type=change_type)
        
        # Find all callers of this function
        callers = await self.call_graph.find_callers(function_name, collections)
        
        # Determine impact level based on number of callers and change type
        impact_level = ImpactLevel.LOW
        breaking_changes = []
        
        if change_type == "signature":
            if len(callers) > 10:
                impact_level = ImpactLevel.HIGH
                breaking_changes.append(f"Signature change affects {len(callers)} calling functions")
            elif len(callers) > 3:
                impact_level = ImpactLevel.MEDIUM
                breaking_changes.append(f"Signature change affects {len(callers)} calling functions")
            else:
                breaking_changes.append("Function signature change requires caller updates")
                
        elif change_type == "removal":
            if callers:
                impact_level = ImpactLevel.CRITICAL
                breaking_changes.append(f"Function removal breaks {len(callers)} calling functions")
                
        elif change_type == "rename":
            if callers:
                impact_level = ImpactLevel.HIGH
                breaking_changes.append(f"Function rename requires updating {len(callers)} callers")
        
        # Generate migration suggestions
        suggestions = []
        if change_type == "signature":
            suggestions.append("Add function overload for backward compatibility")
            suggestions.append("Use deprecation warnings before removing old signature")
            suggestions.append("Update all callers to use new signature")
        elif change_type == "removal":
            suggestions.append("Mark function as deprecated first")
            suggestions.append("Provide alternative function recommendations")
            suggestions.append("Migrate all callers to alternatives")
        elif change_type == "rename":
            suggestions.append("Keep old function as deprecated wrapper")
            suggestions.append("Use IDE refactoring tools for safe renaming")
            suggestions.append("Update documentation and examples")
        
        return RefactoringImpact(
            target_symbol=function_name,
            impact_level=impact_level,
            affected_symbols=callers,
            breaking_changes=breaking_changes,
            suggested_migrations=suggestions
        )
    
    async def analyze_class_change(self, class_name: str, 
                                 change_type: str = "method_change",
                                 collections: Optional[List[str]] = None) -> RefactoringImpact:
        """Analyze impact of changing a class"""
        logger.info("Analyzing class change impact",
                   class_name=class_name,
                   change_type=change_type)
        
        # Get class hierarchy
        hierarchy = await self.inheritance_graph.get_class_hierarchy(class_name, collections)
        
        affected_symbols = []
        breaking_changes = []
        impact_level = ImpactLevel.LOW
        
        # Analyze descendants (subclasses)
        descendants = hierarchy.get("descendants", [])
        if descendants:
            affected_symbols.extend([DependencyNode(
                identifier=desc["identifier"],
                symbol_kind=SymbolKind.CLASS,
                location=SymbolLocation(
                    file_path=desc["location"]["file_path"],
                    file_uri=desc["location"]["file_uri"],
                    line=desc["location"]["line"],
                    column=desc["location"]["column"],
                    end_line=desc["location"]["end_line"],
                    end_column=desc["location"]["end_column"],
                    collection=desc["location"]["collection"]
                )
            ) for desc in descendants])
            
            if change_type == "method_change":
                impact_level = ImpactLevel.MEDIUM
                breaking_changes.append(f"Method changes may affect {len(descendants)} subclasses")
            elif change_type == "interface_change":
                impact_level = ImpactLevel.HIGH
                breaking_changes.append(f"Interface changes will affect {len(descendants)} subclasses")
        
        # Generate suggestions based on change type
        suggestions = []
        if change_type == "method_change":
            suggestions.append("Review method overrides in subclasses")
            suggestions.append("Test all subclasses after changes")
            suggestions.append("Consider adding new methods instead of changing existing ones")
        elif change_type == "interface_change":
            suggestions.append("Use adapter pattern for interface changes")
            suggestions.append("Provide migration path for existing implementations")
            suggestions.append("Version the interface if possible")
        
        return RefactoringImpact(
            target_symbol=class_name,
            impact_level=impact_level,
            affected_symbols=affected_symbols,
            breaking_changes=breaking_changes,
            suggested_migrations=suggestions
        )
    
    async def analyze_module_change(self, module_name: str,
                                  change_type: str = "api_change", 
                                  collections: Optional[List[str]] = None) -> RefactoringImpact:
        """Analyze impact of changing a module"""
        logger.info("Analyzing module change impact",
                   module_name=module_name,
                   change_type=change_type)
        
        # Get module dependencies
        dependencies = await self.import_graph.get_module_dependencies(module_name, True, collections)
        
        imported_by = dependencies.get("imported_by", [])
        affected_symbols = imported_by
        
        breaking_changes = []
        impact_level = ImpactLevel.LOW
        
        if change_type == "api_change":
            if len(imported_by) > 20:
                impact_level = ImpactLevel.CRITICAL
                breaking_changes.append(f"API changes affect {len(imported_by)} importing modules")
            elif len(imported_by) > 5:
                impact_level = ImpactLevel.HIGH
                breaking_changes.append(f"API changes affect {len(imported_by)} importing modules")
            else:
                impact_level = ImpactLevel.MEDIUM
                breaking_changes.append("API changes require import updates")
        
        elif change_type == "removal":
            if imported_by:
                impact_level = ImpactLevel.CRITICAL
                breaking_changes.append(f"Module removal breaks {len(imported_by)} importing modules")
        
        # Generate migration suggestions
        suggestions = []
        if change_type == "api_change":
            suggestions.append("Use semantic versioning for API changes")
            suggestions.append("Provide deprecated compatibility layer")
            suggestions.append("Update all importing modules")
            suggestions.append("Publish migration guide")
        elif change_type == "removal":
            suggestions.append("Mark module as deprecated first")
            suggestions.append("Provide alternative modules")
            suggestions.append("Migrate all imports to alternatives")
        
        return RefactoringImpact(
            target_symbol=module_name,
            impact_level=impact_level,
            affected_symbols=affected_symbols,
            breaking_changes=breaking_changes,
            suggested_migrations=suggestions
        )


class DependencyQueryEngine:
    """High-level query interface for dependency analysis"""
    
    def __init__(self, dependency_analyzer: 'DependencyAnalyzer'):
        self.analyzer = dependency_analyzer
    
    async def find_callers(self, function_name: str, collections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find all functions that call the specified function"""
        callers = await self.analyzer.call_graph.find_callers(function_name, collections)
        return [caller.to_dict() for caller in callers]
    
    async def find_callees(self, function_name: str, collections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find all functions called by the specified function"""
        callees = await self.analyzer.call_graph.find_callees(function_name, collections)
        return [callee.to_dict() for callee in callees]
    
    async def find_dependencies(self, symbol_name: str, dependency_type: str = "all",
                              collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Find all dependencies of a specific symbol"""
        result = {"symbol": symbol_name, "dependencies": {}}
        
        if dependency_type in ["all", "calls"]:
            callees = await self.analyzer.call_graph.find_callees(symbol_name, collections)
            result["dependencies"]["calls"] = [callee.to_dict() for callee in callees]
        
        if dependency_type in ["all", "imports"]:
            if symbol_name in self.analyzer.import_graph.nodes:
                deps = await self.analyzer.import_graph.get_module_dependencies(symbol_name, True, collections)
                result["dependencies"]["imports"] = [dep.to_dict() for dep in deps.get("imports", [])]
        
        if dependency_type in ["all", "inheritance"]:
            hierarchy = await self.analyzer.inheritance_graph.get_class_hierarchy(symbol_name, collections)
            result["dependencies"]["inheritance"] = {
                "ancestors": hierarchy.get("ancestors", []),
                "descendants": hierarchy.get("descendants", [])
            }
        
        return result
    
    async def find_dependents(self, symbol_name: str, dependency_type: str = "all",
                            collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Find all symbols that depend on the specified symbol"""
        result = {"symbol": symbol_name, "dependents": {}}
        
        if dependency_type in ["all", "callers"]:
            callers = await self.analyzer.call_graph.find_callers(symbol_name, collections)
            result["dependents"]["callers"] = [caller.to_dict() for caller in callers]
        
        if dependency_type in ["all", "importers"]:
            if symbol_name in self.analyzer.import_graph.nodes:
                deps = await self.analyzer.import_graph.get_module_dependencies(symbol_name, True, collections)
                result["dependents"]["importers"] = [dep.to_dict() for dep in deps.get("imported_by", [])]
        
        if dependency_type in ["all", "subclasses"]:
            hierarchy = await self.analyzer.inheritance_graph.get_class_hierarchy(symbol_name, collections)
            result["dependents"]["subclasses"] = hierarchy.get("descendants", [])
        
        return result
    
    async def analyze_circular_dependencies(self, collections: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Find all circular dependencies in the codebase"""
        result = {}
        
        # Import circular dependencies
        import_cycles = await self.analyzer.import_graph.find_circular_dependencies(collections)
        result["import_cycles"] = [cycle.to_dict() for cycle in import_cycles]
        
        # Call circular dependencies (recursion)
        call_cycles = await self.analyzer.call_graph.find_recursive_calls(collections)
        result["recursive_calls"] = call_cycles
        
        return result
    
    async def get_dependency_statistics(self, collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get overall dependency statistics for the codebase"""
        # Ensure graphs are built
        await self.analyzer.ensure_graphs_built(collections)
        
        return {
            "call_graph": {
                "functions": len(self.analyzer.call_graph.nodes),
                "calls": len(self.analyzer.call_graph.edges),
                "average_calls_per_function": (
                    len(self.analyzer.call_graph.edges) / max(len(self.analyzer.call_graph.nodes), 1)
                )
            },
            "import_graph": {
                "modules": len(self.analyzer.import_graph.nodes),
                "imports": len(self.analyzer.import_graph.edges),
                "external_dependencies": len(self.analyzer.import_graph.external_dependencies)
            },
            "inheritance_graph": {
                "classes": len(self.analyzer.inheritance_graph.nodes),
                "inheritance_relations": len(self.analyzer.inheritance_graph.edges),
                "method_overrides": len(self.analyzer.inheritance_graph.method_overrides)
            }
        }


class DependencyAnalyzer:
    """Main dependency analysis engine that orchestrates all dependency graphs"""
    
    def __init__(self, workspace_client: QdrantWorkspaceClient):
        self.workspace_client = workspace_client
        self.symbol_resolver: Optional[SymbolResolver] = None
        self.call_graph: Optional[CallGraph] = None
        self.import_graph: Optional[ImportGraph] = None
        self.inheritance_graph: Optional[InheritanceGraph] = None
        self.impact_analyzer: Optional[ImpactAnalyzer] = None
        self.query_engine: Optional[DependencyQueryEngine] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all dependency analysis components"""
        try:
            logger.info("Initializing dependency analyzer")
            
            # Initialize symbol resolver
            self.symbol_resolver = SymbolResolver(self.workspace_client)
            await self.symbol_resolver.initialize()
            
            # Initialize dependency graphs
            self.call_graph = CallGraph(self.workspace_client, self.symbol_resolver)
            self.import_graph = ImportGraph(self.workspace_client, self.symbol_resolver)
            self.inheritance_graph = InheritanceGraph(self.workspace_client, self.symbol_resolver)
            
            # Initialize impact analyzer
            self.impact_analyzer = ImpactAnalyzer(
                self.call_graph, self.import_graph, self.inheritance_graph
            )
            
            # Initialize query engine
            self.query_engine = DependencyQueryEngine(self)
            
            self._initialized = True
            logger.info("Dependency analyzer initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize dependency analyzer", error=str(e))
            raise WorkspaceError(
                f"Failed to initialize dependency analyzer: {str(e)}",
                ErrorCategory.INITIALIZATION_ERROR,
                ErrorSeverity.CRITICAL
            )
    
    async def ensure_graphs_built(self, collections: Optional[List[str]] = None) -> None:
        """Ensure all dependency graphs are built"""
        if not self._initialized:
            await self.initialize()
        
        tasks = []
        
        if not self.call_graph.nodes:
            tasks.append(self.call_graph.build_graph(collections))
        
        if not self.import_graph.nodes:
            tasks.append(self.import_graph.build_graph(collections))
        
        if not self.inheritance_graph.nodes:
            tasks.append(self.inheritance_graph.build_graph(collections))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def rebuild_graphs(self, collections: Optional[List[str]] = None) -> None:
        """Rebuild all dependency graphs from scratch"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Rebuilding dependency graphs", collections=collections)
        
        # Clear existing graphs
        self.call_graph.clear()
        self.import_graph.clear()
        self.inheritance_graph.clear()
        
        # Rebuild all graphs
        await self.ensure_graphs_built(collections)
        
        logger.info("Dependency graphs rebuilt successfully")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the current dependency analysis state"""
        return {
            "initialized": self._initialized,
            "graphs_built": {
                "call_graph": len(self.call_graph.nodes) > 0 if self.call_graph else False,
                "import_graph": len(self.import_graph.nodes) > 0 if self.import_graph else False,
                "inheritance_graph": len(self.inheritance_graph.nodes) > 0 if self.inheritance_graph else False
            },
            "last_updated": {
                "call_graph": getattr(self.call_graph, '_last_updated', None) if self.call_graph else None,
                "import_graph": getattr(self.import_graph, '_last_updated', None) if self.import_graph else None,
                "inheritance_graph": getattr(self.inheritance_graph, '_last_updated', None) if self.inheritance_graph else None
            }
        }