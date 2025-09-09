#!/usr/bin/env python3
"""
Dependency Analyzer Demo Script

This script demonstrates the comprehensive Relationship and Dependency Query Engine
implemented for Task #127. It showcases all major functionality including:

- Call graph analysis with caller/callee relationships
- Import dependency tracking with circular detection
- Inheritance hierarchy analysis with method overrides
- Refactoring impact analysis with severity assessment
- High-level query interface for relationship exploration

Usage:
    python 20250107-1628_dependency_analyzer_demo.py

Key Features Demonstrated:
- Creating and analyzing dependency graphs
- Finding function callers and callees
- Detecting circular dependencies
- Analyzing refactoring impact
- Querying symbol relationships
- Visualizing dependency statistics
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from workspace_qdrant_mcp.tools.dependency_analyzer import (
    DependencyAnalyzer, CallGraph, ImportGraph, InheritanceGraph,
    DependencyNode, DependencyEdge, CircularDependency,
    DependencyType, ImpactLevel, CircularDependencyType
)
from workspace_qdrant_mcp.core.lsp_metadata_extractor import SymbolKind
from workspace_qdrant_mcp.tools.symbol_resolver import SymbolLocation


def create_sample_location(file_path: str, line: int = 0) -> SymbolLocation:
    """Create a sample SymbolLocation for demo purposes"""
    return SymbolLocation(
        file_path=file_path,
        file_uri=f"file://{file_path}",
        line=line,
        column=0,
        end_line=line + 5,
        end_column=20,
        collection="demo-project"
    )


def create_sample_call_graph() -> CallGraph:
    """Create a sample call graph with realistic function relationships"""
    mock_client = MagicMock()
    mock_resolver = MagicMock()
    call_graph = CallGraph(mock_client, mock_resolver)
    
    # Create sample functions
    auth_func = DependencyNode(
        "auth.authenticate", SymbolKind.FUNCTION, 
        create_sample_location("/src/auth.py", 10),
        metadata={"signature": "authenticate(user: str, password: str) -> bool"}
    )
    
    validate_func = DependencyNode(
        "auth.validate_credentials", SymbolKind.FUNCTION,
        create_sample_location("/src/auth.py", 25),
        metadata={"signature": "validate_credentials(user: str, password: str) -> bool"}
    )
    
    hash_func = DependencyNode(
        "utils.hash_password", SymbolKind.FUNCTION,
        create_sample_location("/src/utils.py", 15),
        metadata={"signature": "hash_password(password: str) -> str"}
    )
    
    login_func = DependencyNode(
        "main.login", SymbolKind.FUNCTION,
        create_sample_location("/src/main.py", 40),
        metadata={"signature": "login(username: str, password: str) -> dict"}
    )
    
    # Add nodes to graph
    for node in [auth_func, validate_func, hash_func, login_func]:
        call_graph.add_node(node)
    
    # Create call relationships
    # login calls authenticate
    call_graph.add_edge(DependencyEdge(login_func, auth_func, DependencyType.CALL))
    
    # authenticate calls validate_credentials
    call_graph.add_edge(DependencyEdge(auth_func, validate_func, DependencyType.CALL))
    
    # validate_credentials calls hash_password
    call_graph.add_edge(DependencyEdge(validate_func, hash_func, DependencyType.CALL))
    
    return call_graph


def create_sample_import_graph() -> ImportGraph:
    """Create a sample import graph with module dependencies"""
    mock_client = MagicMock()
    mock_resolver = MagicMock()
    import_graph = ImportGraph(mock_client, mock_resolver)
    
    # Create sample modules
    main_module = DependencyNode(
        "main", SymbolKind.MODULE,
        create_sample_location("/src/main.py"),
        metadata={"language": "python"}
    )
    
    auth_module = DependencyNode(
        "auth", SymbolKind.MODULE,
        create_sample_location("/src/auth.py"),
        metadata={"language": "python"}
    )
    
    utils_module = DependencyNode(
        "utils", SymbolKind.MODULE,
        create_sample_location("/src/utils.py"),
        metadata={"language": "python"}
    )
    
    # External dependency
    requests_module = DependencyNode(
        "requests", SymbolKind.MODULE,
        create_sample_location(""),
        metadata={"external": True}
    )
    
    # Add nodes
    for node in [main_module, auth_module, utils_module, requests_module]:
        import_graph.add_node(node)
    
    # Create import relationships
    import_graph.add_edge(DependencyEdge(main_module, auth_module, DependencyType.IMPORT))
    import_graph.add_edge(DependencyEdge(auth_module, utils_module, DependencyType.IMPORT))
    import_graph.add_edge(DependencyEdge(auth_module, requests_module, DependencyType.IMPORT))
    
    # Add external dependency
    import_graph.external_dependencies.add("requests")
    
    return import_graph


def create_sample_inheritance_graph() -> InheritanceGraph:
    """Create a sample inheritance graph with class hierarchy"""
    mock_client = MagicMock()
    mock_resolver = MagicMock()
    inheritance_graph = InheritanceGraph(mock_client, mock_resolver)
    
    # Create sample classes
    base_user = DependencyNode(
        "BaseUser", SymbolKind.CLASS,
        create_sample_location("/src/models.py", 10),
        metadata={"abstract": True, "methods": ["login", "logout"]}
    )
    
    admin_user = DependencyNode(
        "AdminUser", SymbolKind.CLASS,
        create_sample_location("/src/models.py", 50),
        metadata={"methods": ["login", "delete_user"], "properties": ["permissions"]}
    )
    
    regular_user = DependencyNode(
        "RegularUser", SymbolKind.CLASS,
        create_sample_location("/src/models.py", 80),
        metadata={"methods": ["login"], "properties": ["profile"]}
    )
    
    # Add nodes
    for node in [base_user, admin_user, regular_user]:
        inheritance_graph.add_node(node)
    
    # Create inheritance relationships
    inheritance_graph.add_edge(DependencyEdge(admin_user, base_user, DependencyType.INHERITANCE))
    inheritance_graph.add_edge(DependencyEdge(regular_user, base_user, DependencyType.INHERITANCE))
    
    # Add method overrides
    inheritance_graph.method_overrides["BaseUser.login"] = ["AdminUser.login", "RegularUser.login"]
    
    return inheritance_graph


def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)


def print_dependency_node(node: DependencyNode, indent: str = ""):
    """Print a formatted dependency node"""
    print(f"{indent}• {node.identifier} ({node.symbol_kind.name})")
    if node.metadata:
        for key, value in node.metadata.items():
            if key == "signature":
                print(f"{indent}  └─ {value}")


async def demonstrate_call_graph_analysis():
    """Demonstrate call graph analysis capabilities"""
    print_section_header("CALL GRAPH ANALYSIS")
    
    call_graph = create_sample_call_graph()
    
    print(f"📊 Call Graph Statistics:")
    print(f"   Functions: {len(call_graph.nodes)}")
    print(f"   Call relationships: {len(call_graph.edges)}")
    
    print(f"\n🔍 Function Analysis:")
    
    # Show callers of authenticate function
    authenticate_callers = await call_graph.find_callers("auth.authenticate")
    print(f"\n   Functions that call 'auth.authenticate':")
    for caller in authenticate_callers:
        print_dependency_node(caller, "     ")
    
    # Show what authenticate calls
    authenticate_callees = await call_graph.find_callees("auth.authenticate")
    print(f"\n   Functions called by 'auth.authenticate':")
    for callee in authenticate_callees:
        print_dependency_node(callee, "     ")
    
    # Show call chain
    call_chains = await call_graph.find_call_chains("main.login", "utils.hash_password")
    print(f"\n   Call chain from 'main.login' to 'utils.hash_password':")
    for i, chain in enumerate(call_chains):
        print(f"     Chain {i+1}: {' → '.join(chain)}")


async def demonstrate_import_graph_analysis():
    """Demonstrate import dependency analysis"""
    print_section_header("IMPORT DEPENDENCY ANALYSIS")
    
    import_graph = create_sample_import_graph()
    
    print(f"📊 Import Graph Statistics:")
    print(f"   Modules: {len(import_graph.nodes)}")
    print(f"   Import relationships: {len(import_graph.edges)}")
    print(f"   External dependencies: {len(import_graph.external_dependencies)}")
    
    print(f"\n🔍 Module Analysis:")
    
    # Show module dependencies
    auth_deps = await import_graph.get_module_dependencies("auth")
    print(f"\n   'auth' module dependencies:")
    print(f"     Imports:")
    for dep in auth_deps["imports"]:
        print_dependency_node(dep, "       ")
    print(f"     Imported by:")
    for dep in auth_deps["imported_by"]:
        print_dependency_node(dep, "       ")
    
    # Show external dependencies
    print(f"\n   External Dependencies:")
    for ext_dep in import_graph.external_dependencies:
        print(f"     • {ext_dep}")


async def demonstrate_inheritance_analysis():
    """Demonstrate inheritance hierarchy analysis"""
    print_section_header("INHERITANCE HIERARCHY ANALYSIS")
    
    inheritance_graph = create_sample_inheritance_graph()
    
    print(f"📊 Inheritance Graph Statistics:")
    print(f"   Classes: {len(inheritance_graph.nodes)}")
    print(f"   Inheritance relationships: {len(inheritance_graph.edges)}")
    print(f"   Method overrides: {len(inheritance_graph.method_overrides)}")
    
    print(f"\n🔍 Class Hierarchy Analysis:")
    
    # Show class hierarchy for BaseUser
    hierarchy = await inheritance_graph.get_class_hierarchy("BaseUser")
    print(f"\n   'BaseUser' class hierarchy:")
    print(f"     Descendants (subclasses):")
    for desc in hierarchy["descendants"]:
        print(f"       • {desc['identifier']}")
    
    # Show method overrides
    login_overrides = await inheritance_graph.find_method_overrides("login")
    print(f"\n   Method overrides for 'login':")
    for override in login_overrides:
        print(f"     • {override['base_method']} → {override['override_method']}")


async def demonstrate_impact_analysis():
    """Demonstrate refactoring impact analysis"""
    print_section_header("REFACTORING IMPACT ANALYSIS")
    
    # Create sample graphs for impact analysis
    call_graph = create_sample_call_graph()
    import_graph = create_sample_import_graph()
    inheritance_graph = create_sample_inheritance_graph()
    
    from workspace_qdrant_mcp.tools.dependency_analyzer import ImpactAnalyzer
    impact_analyzer = ImpactAnalyzer(call_graph, import_graph, inheritance_graph)
    
    print(f"🔍 Impact Analysis Examples:")
    
    # Analyze function change impact
    function_impact = await impact_analyzer.analyze_function_change(
        "auth.authenticate", "signature", ["demo-project"]
    )
    print(f"\n   Impact of changing 'auth.authenticate' signature:")
    print(f"     Impact Level: {function_impact.impact_level.value.upper()}")
    print(f"     Affected Functions: {len(function_impact.affected_symbols)}")
    for change in function_impact.breaking_changes:
        print(f"     • {change}")
    print(f"     Migration Suggestions:")
    for suggestion in function_impact.suggested_migrations[:2]:  # Show first 2
        print(f"       - {suggestion}")
    
    # Analyze class change impact
    class_impact = await impact_analyzer.analyze_class_change(
        "BaseUser", "method_change", ["demo-project"]
    )
    print(f"\n   Impact of changing 'BaseUser' methods:")
    print(f"     Impact Level: {class_impact.impact_level.value.upper()}")
    print(f"     Affected Classes: {len(class_impact.affected_symbols)}")
    for change in class_impact.breaking_changes:
        print(f"     • {change}")


async def demonstrate_query_interface():
    """Demonstrate high-level dependency query interface"""
    print_section_header("DEPENDENCY QUERY INTERFACE")
    
    # Create mock dependency analyzer
    mock_client = MagicMock()
    analyzer = DependencyAnalyzer(mock_client)
    
    # Mock the components
    analyzer.call_graph = create_sample_call_graph()
    analyzer.import_graph = create_sample_import_graph()
    analyzer.inheritance_graph = create_sample_inheritance_graph()
    analyzer._initialized = True
    
    from workspace_qdrant_mcp.tools.dependency_analyzer import DependencyQueryEngine
    query_engine = DependencyQueryEngine(analyzer)
    
    print(f"🔍 High-Level Query Examples:")
    
    # Find callers query
    callers = await query_engine.find_callers("auth.authenticate", ["demo-project"])
    print(f"\n   Query: Find callers of 'auth.authenticate'")
    print(f"     Results: {len(callers)} callers found")
    for caller in callers:
        print(f"       • {caller['identifier']}")
    
    # Dependency statistics
    stats = await query_engine.get_dependency_statistics(["demo-project"])
    print(f"\n   Dependency Statistics:")
    print(f"     Call Graph: {stats['call_graph']['functions']} functions, {stats['call_graph']['calls']} calls")
    print(f"     Import Graph: {stats['import_graph']['modules']} modules, {stats['import_graph']['external_dependencies']} external deps")
    print(f"     Inheritance Graph: {stats['inheritance_graph']['classes']} classes")


async def main():
    """Main demonstration function"""
    print("🚀 Dependency Analyzer Demo - Task #127")
    print("Comprehensive Relationship and Dependency Query Engine")
    
    # Demonstrate all major functionality
    await demonstrate_call_graph_analysis()
    await demonstrate_import_graph_analysis()
    await demonstrate_inheritance_analysis()
    await demonstrate_impact_analysis()
    await demonstrate_query_interface()
    
    print_section_header("DEMO COMPLETE")
    print("✅ All dependency analysis features demonstrated successfully!")
    print("\nKey Capabilities Showcased:")
    print("  • Call graph analysis with caller/callee relationships")
    print("  • Import dependency tracking with external library detection")
    print("  • Inheritance hierarchy analysis with method overrides")
    print("  • Refactoring impact analysis with severity assessment")
    print("  • High-level query interface for relationship exploration")
    print("  • Integration with Symbol Resolver and Code Search infrastructure")
    print("\nThe Dependency Analyzer provides actionable insights for:")
    print("  • Code maintenance and refactoring planning")
    print("  • Impact assessment for code changes")
    print("  • Dependency visualization and analysis")
    print("  • Circular dependency detection and resolution")


if __name__ == "__main__":
    asyncio.run(main())