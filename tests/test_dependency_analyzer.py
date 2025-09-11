"""
Comprehensive tests for the Relationship and Dependency Query Engine

This test suite validates all functionality of the dependency analyzer including:
- Call graph analysis and traversal
- Import dependency tracking with circular detection
- Inheritance hierarchy analysis with method overrides
- Refactoring impact analysis
- Relationship query interface
- Integration with existing symbol resolver and search infrastructure
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from common.tools.dependency_analyzer import (
    DependencyAnalyzer, CallGraph, ImportGraph, InheritanceGraph,
    ImpactAnalyzer, DependencyQueryEngine,
    DependencyNode, DependencyEdge, CircularDependency, RefactoringImpact,
    DependencyType, ImpactLevel, CircularDependencyType
)
from common.core.lsp_metadata_extractor import SymbolKind
from common.tools.symbol_resolver import SymbolLocation
from common.core.error_handling import WorkspaceError


class TestDependencyNode:
    """Test DependencyNode data structure"""
    
    def test_dependency_node_creation(self):
        """Test creating dependency nodes"""
        location = SymbolLocation(
            file_path="/test/file.py",
            file_uri="file:///test/file.py",
            line=10,
            column=5,
            end_line=15,
            end_column=10,
            collection="test-collection"
        )
        
        node = DependencyNode(
            identifier="test.function",
            symbol_kind=SymbolKind.FUNCTION,
            location=location,
            metadata={"signature": "function(arg: str) -> bool"}
        )
        
        assert node.identifier == "test.function"
        assert node.symbol_kind == SymbolKind.FUNCTION
        assert node.location.file_path == "/test/file.py"
        assert node.metadata["signature"] == "function(arg: str) -> bool"
    
    def test_dependency_node_equality(self):
        """Test dependency node equality comparison"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "")
        
        node1 = DependencyNode("test.func", SymbolKind.FUNCTION, location)
        node2 = DependencyNode("test.func", SymbolKind.FUNCTION, location)
        node3 = DependencyNode("other.func", SymbolKind.FUNCTION, location)
        
        assert node1 == node2
        assert node1 != node3
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)
    
    def test_dependency_node_to_dict(self):
        """Test dependency node serialization"""
        location = SymbolLocation(
            file_path="/test/file.py",
            file_uri="file:///test/file.py",
            line=10,
            column=5,
            end_line=15,
            end_column=10,
            collection="test-collection"
        )
        
        node = DependencyNode(
            identifier="test.function",
            symbol_kind=SymbolKind.FUNCTION,
            location=location,
            metadata={"signature": "function()"}
        )
        
        result = node.to_dict()
        
        assert result["identifier"] == "test.function"
        assert result["symbol_kind"] == "FUNCTION"
        assert result["location"]["file_path"] == "/test/file.py"
        assert result["metadata"]["signature"] == "function()"


class TestDependencyGraph:
    """Test base dependency graph functionality"""
    
    @pytest.fixture
    def mock_workspace_client(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_symbol_resolver(self):
        resolver = MagicMock()
        resolver.initialize = AsyncMock()
        return resolver
    
    @pytest.fixture
    def call_graph(self, mock_workspace_client, mock_symbol_resolver):
        return CallGraph(mock_workspace_client, mock_symbol_resolver)
    
    def test_graph_add_node(self, call_graph):
        """Test adding nodes to graph"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "")
        node = DependencyNode("test.func", SymbolKind.FUNCTION, location)
        
        call_graph.add_node(node)
        
        assert "test.func" in call_graph.nodes
        assert call_graph.nodes["test.func"] == node
        assert "test.func" in call_graph.adjacency_list
        assert "test.func" in call_graph.reverse_adjacency_list
    
    def test_graph_add_edge(self, call_graph):
        """Test adding edges to graph"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "")
        node1 = DependencyNode("func1", SymbolKind.FUNCTION, location)
        node2 = DependencyNode("func2", SymbolKind.FUNCTION, location)
        
        call_graph.add_node(node1)
        call_graph.add_node(node2)
        
        edge = DependencyEdge(node1, node2, DependencyType.CALL)
        call_graph.add_edge(edge)
        
        assert edge in call_graph.edges
        assert "func2" in call_graph.adjacency_list["func1"]
        assert "func1" in call_graph.reverse_adjacency_list["func2"]
    
    def test_graph_find_paths(self, call_graph):
        """Test finding paths between nodes"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "")
        
        # Create a chain: A -> B -> C
        node_a = DependencyNode("A", SymbolKind.FUNCTION, location)
        node_b = DependencyNode("B", SymbolKind.FUNCTION, location)
        node_c = DependencyNode("C", SymbolKind.FUNCTION, location)
        
        call_graph.add_node(node_a)
        call_graph.add_node(node_b)
        call_graph.add_node(node_c)
        
        call_graph.add_edge(DependencyEdge(node_a, node_b, DependencyType.CALL))
        call_graph.add_edge(DependencyEdge(node_b, node_c, DependencyType.CALL))
        
        paths = call_graph.find_paths("A", "C")
        
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C"]
    
    def test_graph_find_cycles(self, call_graph):
        """Test finding cycles in graph"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "")
        
        # Create a cycle: A -> B -> A
        node_a = DependencyNode("A", SymbolKind.FUNCTION, location)
        node_b = DependencyNode("B", SymbolKind.FUNCTION, location)
        
        call_graph.add_node(node_a)
        call_graph.add_node(node_b)
        
        call_graph.add_edge(DependencyEdge(node_a, node_b, DependencyType.CALL))
        call_graph.add_edge(DependencyEdge(node_b, node_a, DependencyType.CALL))
        
        cycles = call_graph.find_cycles()
        
        assert len(cycles) >= 1
        # Should find the cycle A -> B -> A
        found_cycle = False
        for cycle in cycles:
            if len(cycle) == 3 and "A" in cycle and "B" in cycle:
                found_cycle = True
        assert found_cycle


@pytest.mark.asyncio
class TestCallGraph:
    """Test call graph analysis functionality"""
    
    @pytest.fixture
    def mock_workspace_client(self):
        client = MagicMock()
        return client
    
    @pytest.fixture
    def mock_symbol_resolver(self):
        resolver = MagicMock()
        resolver.initialize = AsyncMock()
        return resolver
    
    @pytest.fixture
    def call_graph(self, mock_workspace_client, mock_symbol_resolver):
        return CallGraph(mock_workspace_client, mock_symbol_resolver)
    
    @pytest.fixture
    def sample_lsp_data(self):
        """Sample LSP data for testing"""
        return {
            "results": [
                {
                    "collection": "test-project",
                    "payload": {
                        "file_path": "/test/module.py",
                        "file_uri": "file:///test/module.py",
                        "lsp_metadata": {
                            "symbols": [
                                {
                                    "identifier": "module.authenticate",
                                    "kind": "function",
                                    "signature": "authenticate(user: str, password: str) -> bool",
                                    "return_type": "bool",
                                    "parameters": ["user: str", "password: str"],
                                    "range": {
                                        "start": {"line": 10, "character": 0},
                                        "end": {"line": 15, "character": 20}
                                    },
                                    "relationships": [
                                        {
                                            "type": "calls",
                                            "target": "module.validate_credentials",
                                            "location": {"line": 12, "character": 4}
                                        }
                                    ]
                                },
                                {
                                    "identifier": "module.validate_credentials", 
                                    "kind": "function",
                                    "signature": "validate_credentials(user: str, password: str) -> bool",
                                    "relationships": []
                                }
                            ]
                        }
                    }
                }
            ]
        }
    
    async def test_build_call_graph(self, call_graph, sample_lsp_data):
        """Test building call graph from LSP data"""
        with patch('src.workspace_qdrant_mcp.tools.dependency_analyzer.search_workspace') as mock_search:
            mock_search.return_value = sample_lsp_data
            
            await call_graph.build_graph(["test-project"])
            
            # Should have created nodes for both functions
            assert "module.authenticate" in call_graph.nodes
            assert "module.validate_credentials" in call_graph.nodes
            
            # Should have created call relationship
            assert len(call_graph.edges) == 1
            edge = call_graph.edges[0]
            assert edge.source.identifier == "module.authenticate"
            assert edge.target.identifier == "module.validate_credentials"
            assert edge.dependency_type == DependencyType.CALL
    
    async def test_find_callers(self, call_graph, sample_lsp_data):
        """Test finding function callers"""
        with patch('src.workspace_qdrant_mcp.tools.dependency_analyzer.search_workspace') as mock_search:
            mock_search.return_value = sample_lsp_data
            
            await call_graph.build_graph(["test-project"])
            
            # Find callers of validate_credentials
            callers = await call_graph.find_callers("module.validate_credentials")
            
            assert len(callers) == 1
            assert callers[0].identifier == "module.authenticate"
    
    async def test_find_callees(self, call_graph, sample_lsp_data):
        """Test finding function callees"""
        with patch('src.workspace_qdrant_mcp.tools.dependency_analyzer.search_workspace') as mock_search:
            mock_search.return_value = sample_lsp_data
            
            await call_graph.build_graph(["test-project"])
            
            # Find callees of authenticate
            callees = await call_graph.find_callees("module.authenticate")
            
            assert len(callees) == 1
            assert callees[0].identifier == "module.validate_credentials"
    
    async def test_find_call_chains(self, call_graph):
        """Test finding call chains between functions"""
        # Manually build a call chain for testing
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        node_a = DependencyNode("A", SymbolKind.FUNCTION, location)
        node_b = DependencyNode("B", SymbolKind.FUNCTION, location)
        node_c = DependencyNode("C", SymbolKind.FUNCTION, location)
        
        call_graph.add_node(node_a)
        call_graph.add_node(node_b)
        call_graph.add_node(node_c)
        
        call_graph.add_edge(DependencyEdge(node_a, node_b, DependencyType.CALL))
        call_graph.add_edge(DependencyEdge(node_b, node_c, DependencyType.CALL))
        
        chains = await call_graph.find_call_chains("A", "C")
        
        assert len(chains) == 1
        assert chains[0] == ["A", "B", "C"]
    
    async def test_find_recursive_calls(self, call_graph):
        """Test finding recursive call patterns"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        # Create recursive pattern: A -> B -> A
        node_a = DependencyNode("recursive_func", SymbolKind.FUNCTION, location)
        node_b = DependencyNode("helper_func", SymbolKind.FUNCTION, location)
        
        call_graph.add_node(node_a)
        call_graph.add_node(node_b)
        
        call_graph.add_edge(DependencyEdge(node_a, node_b, DependencyType.CALL))
        call_graph.add_edge(DependencyEdge(node_b, node_a, DependencyType.CALL))
        
        recursive_calls = await call_graph.find_recursive_calls()
        
        assert len(recursive_calls) >= 1
        # Should find the recursive cycle
        found_recursion = any("recursive_func" in cycle and "helper_func" in cycle for cycle in recursive_calls)
        assert found_recursion


@pytest.mark.asyncio  
class TestImportGraph:
    """Test import dependency graph functionality"""
    
    @pytest.fixture
    def mock_workspace_client(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_symbol_resolver(self):
        resolver = MagicMock()
        resolver.initialize = AsyncMock()
        return resolver
    
    @pytest.fixture
    def import_graph(self, mock_workspace_client, mock_symbol_resolver):
        return ImportGraph(mock_workspace_client, mock_symbol_resolver)
    
    @pytest.fixture
    def sample_import_data(self):
        """Sample data with import relationships"""
        return {
            "results": [
                {
                    "collection": "test-project",
                    "payload": {
                        "file_path": "/test/main.py",
                        "file_uri": "file:///test/main.py",
                        "lsp_metadata": {
                            "language": "python",
                            "symbols": [
                                {
                                    "identifier": "main.main_function",
                                    "kind": "function",
                                    "relationships": [
                                        {
                                            "type": "imports",
                                            "target": "auth.module",
                                            "import_type": "standard",
                                            "symbols": ["authenticate"]
                                        },
                                        {
                                            "type": "imports", 
                                            "target": "os",
                                            "import_type": "standard"
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            ]
        }
    
    async def test_build_import_graph(self, import_graph, sample_import_data):
        """Test building import graph from LSP data"""
        with patch('src.workspace_qdrant_mcp.tools.dependency_analyzer.search_workspace') as mock_search:
            mock_search.return_value = sample_import_data
            
            await import_graph.build_graph(["test-project"])
            
            # Should have created module nodes
            assert "test.main" in import_graph.nodes  # Module from file path
            assert "auth.module" in import_graph.nodes
            assert "os" in import_graph.nodes
            
            # Should have identified external dependencies
            assert "os" in import_graph.external_dependencies
            
            # Should have created import edges
            assert len(import_graph.edges) >= 2
    
    def test_module_identifier_conversion(self, import_graph):
        """Test conversion of file paths to module identifiers"""
        # Test Python module conversion
        py_id = import_graph._get_module_identifier("/src/package/module.py")
        assert py_id == "package.module"
        
        # Test JavaScript module conversion  
        js_id = import_graph._get_module_identifier("/src/components/Button.js")
        assert js_id == "src/components/Button.js"
    
    def test_external_dependency_detection(self, import_graph):
        """Test detection of external dependencies"""
        assert import_graph._is_external_dependency("os", None)
        assert import_graph._is_external_dependency("numpy", None)
        assert import_graph._is_external_dependency("react", None)
        assert not import_graph._is_external_dependency("my.custom.module", None)
    
    async def test_find_circular_dependencies(self, import_graph):
        """Test finding circular import dependencies"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        # Create circular import: A -> B -> A
        node_a = DependencyNode("module.a", SymbolKind.MODULE, location)
        node_b = DependencyNode("module.b", SymbolKind.MODULE, location)
        
        import_graph.add_node(node_a)
        import_graph.add_node(node_b)
        
        import_graph.add_edge(DependencyEdge(node_a, node_b, DependencyType.IMPORT))
        import_graph.add_edge(DependencyEdge(node_b, node_a, DependencyType.IMPORT))
        
        circular_deps = await import_graph.find_circular_dependencies()
        
        assert len(circular_deps) >= 1
        found_circular = any(
            dep.cycle_type == CircularDependencyType.IMPORT 
            and len(dep.nodes) == 2 
            for dep in circular_deps
        )
        assert found_circular
    
    async def test_get_module_dependencies(self, import_graph):
        """Test getting module dependencies"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        # Create dependency chain
        main_module = DependencyNode("main", SymbolKind.MODULE, location)
        auth_module = DependencyNode("auth", SymbolKind.MODULE, location)
        utils_module = DependencyNode("utils", SymbolKind.MODULE, location)
        
        import_graph.add_node(main_module)
        import_graph.add_node(auth_module)
        import_graph.add_node(utils_module)
        
        # main imports auth, auth imports utils  
        import_graph.add_edge(DependencyEdge(main_module, auth_module, DependencyType.IMPORT))
        import_graph.add_edge(DependencyEdge(auth_module, utils_module, DependencyType.IMPORT))
        
        # utils is imported by auth
        import_graph.add_edge(DependencyEdge(utils_module, main_module, DependencyType.IMPORT))
        
        deps = await import_graph.get_module_dependencies("main")
        
        assert "imports" in deps
        assert "imported_by" in deps
        assert len(deps["imports"]) >= 1
        assert any(node.identifier == "auth" for node in deps["imports"])


@pytest.mark.asyncio
class TestInheritanceGraph:
    """Test inheritance hierarchy analysis"""
    
    @pytest.fixture
    def mock_workspace_client(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_symbol_resolver(self):
        resolver = MagicMock()
        resolver.initialize = AsyncMock()
        return resolver
    
    @pytest.fixture
    def inheritance_graph(self, mock_workspace_client, mock_symbol_resolver):
        return InheritanceGraph(mock_workspace_client, mock_symbol_resolver)
    
    @pytest.fixture
    def sample_inheritance_data(self):
        """Sample data with inheritance relationships"""
        return {
            "results": [
                {
                    "collection": "test-project",
                    "payload": {
                        "file_path": "/test/classes.py",
                        "file_uri": "file:///test/classes.py", 
                        "lsp_metadata": {
                            "symbols": [
                                {
                                    "identifier": "User",
                                    "kind": "class",
                                    "methods": [
                                        {"name": "login", "overrides": "BaseUser.login"},
                                        {"name": "logout"}
                                    ],
                                    "properties": ["username", "email"],
                                    "abstract": False,
                                    "range": {
                                        "start": {"line": 5, "character": 0},
                                        "end": {"line": 20, "character": 0}
                                    },
                                    "relationships": [
                                        {
                                            "type": "extends",
                                            "target": "BaseUser",
                                            "access_modifier": "public"
                                        }
                                    ]
                                },
                                {
                                    "identifier": "BaseUser",
                                    "kind": "class", 
                                    "methods": [{"name": "login"}, {"name": "validate"}],
                                    "abstract": True,
                                    "relationships": []
                                }
                            ]
                        }
                    }
                }
            ]
        }
    
    async def test_build_inheritance_graph(self, inheritance_graph, sample_inheritance_data):
        """Test building inheritance graph from LSP data"""
        with patch('src.workspace_qdrant_mcp.tools.dependency_analyzer.search_workspace') as mock_search:
            mock_search.return_value = sample_inheritance_data
            
            await inheritance_graph.build_graph(["test-project"])
            
            # Should have created class nodes
            assert "User" in inheritance_graph.nodes
            assert "BaseUser" in inheritance_graph.nodes
            
            # Should have created inheritance edge
            assert len(inheritance_graph.edges) == 1
            edge = inheritance_graph.edges[0]
            assert edge.source.identifier == "User"
            assert edge.target.identifier == "BaseUser"
            assert edge.dependency_type == DependencyType.INHERITANCE
            
            # Should have tracked method overrides
            assert "BaseUser.login" in inheritance_graph.method_overrides
            assert "User.login" in inheritance_graph.method_overrides["BaseUser.login"]
    
    async def test_get_class_hierarchy(self, inheritance_graph):
        """Test getting complete class hierarchy"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        # Create inheritance chain: BaseClass -> MiddleClass -> DerivedClass
        base_class = DependencyNode("BaseClass", SymbolKind.CLASS, location)
        middle_class = DependencyNode("MiddleClass", SymbolKind.CLASS, location)
        derived_class = DependencyNode("DerivedClass", SymbolKind.CLASS, location)
        
        inheritance_graph.add_node(base_class)
        inheritance_graph.add_node(middle_class)
        inheritance_graph.add_node(derived_class)
        
        inheritance_graph.add_edge(DependencyEdge(middle_class, base_class, DependencyType.INHERITANCE))
        inheritance_graph.add_edge(DependencyEdge(derived_class, middle_class, DependencyType.INHERITANCE))
        
        hierarchy = await inheritance_graph.get_class_hierarchy("MiddleClass")
        
        assert "target_class" in hierarchy
        assert "ancestors" in hierarchy
        assert "descendants" in hierarchy
        
        # Should find BaseClass as ancestor
        ancestors = hierarchy["ancestors"]
        assert len(ancestors) >= 1
        assert any(ancestor["identifier"] == "BaseClass" for ancestor in ancestors)
        
        # Should find DerivedClass as descendant
        descendants = hierarchy["descendants"]
        assert len(descendants) >= 1
        assert any(descendant["identifier"] == "DerivedClass" for descendant in descendants)
    
    async def test_find_method_overrides(self, inheritance_graph):
        """Test finding method overrides"""
        # Setup method override data
        inheritance_graph.method_overrides["BaseClass.render"] = [
            "DerivedClass.render",
            "AnotherClass.render"
        ]
        
        overrides = await inheritance_graph.find_method_overrides("render")
        
        assert len(overrides) >= 1
        found_override = any(
            "BaseClass.render" in override["base_method"] and
            "DerivedClass.render" in override["override_method"]
            for override in overrides
        )
        assert found_override
    
    async def test_find_interface_implementations(self, inheritance_graph):
        """Test finding interface implementations"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        # Create interface implementation
        interface_node = DependencyNode("Drawable", SymbolKind.INTERFACE, location)
        impl_node = DependencyNode("Circle", SymbolKind.CLASS, location)
        
        inheritance_graph.add_node(interface_node)
        inheritance_graph.add_node(impl_node)
        
        edge = DependencyEdge(
            impl_node, interface_node, DependencyType.INHERITANCE,
            relationship_data={"inheritance_type": "implements"}
        )
        inheritance_graph.add_edge(edge)
        
        implementations = await inheritance_graph.find_interface_implementations("Drawable")
        
        assert len(implementations) == 1
        assert implementations[0].identifier == "Circle"


@pytest.mark.asyncio
class TestImpactAnalyzer:
    """Test refactoring impact analysis"""
    
    @pytest.fixture
    def mock_graphs(self):
        call_graph = MagicMock()
        import_graph = MagicMock()
        inheritance_graph = MagicMock()
        return call_graph, import_graph, inheritance_graph
    
    @pytest.fixture
    def impact_analyzer(self, mock_graphs):
        call_graph, import_graph, inheritance_graph = mock_graphs
        return ImpactAnalyzer(call_graph, import_graph, inheritance_graph)
    
    async def test_analyze_function_change(self, impact_analyzer, mock_graphs):
        """Test analyzing impact of function changes"""
        call_graph, _, _ = mock_graphs
        
        # Mock function has many callers
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        caller_nodes = [
            DependencyNode(f"caller_{i}", SymbolKind.FUNCTION, location)
            for i in range(15)  # 15 callers = HIGH impact
        ]
        
        call_graph.find_callers = AsyncMock(return_value=caller_nodes)
        
        impact = await impact_analyzer.analyze_function_change(
            "popular_function", "signature", ["test-project"]
        )
        
        assert impact.target_symbol == "popular_function"
        assert impact.impact_level == ImpactLevel.HIGH
        assert len(impact.affected_symbols) == 15
        assert len(impact.breaking_changes) >= 1
        assert "15 calling functions" in impact.breaking_changes[0]
    
    async def test_analyze_function_removal(self, impact_analyzer, mock_graphs):
        """Test analyzing impact of function removal"""
        call_graph, _, _ = mock_graphs
        
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        caller_nodes = [DependencyNode("caller", SymbolKind.FUNCTION, location)]
        
        call_graph.find_callers = AsyncMock(return_value=caller_nodes)
        
        impact = await impact_analyzer.analyze_function_change(
            "removed_function", "removal", ["test-project"]
        )
        
        assert impact.impact_level == ImpactLevel.CRITICAL
        assert "Function removal breaks" in impact.breaking_changes[0]
        assert any("deprecated" in suggestion for suggestion in impact.suggested_migrations)
    
    async def test_analyze_class_change(self, impact_analyzer, mock_graphs):
        """Test analyzing impact of class changes"""
        _, _, inheritance_graph = mock_graphs
        
        # Mock class hierarchy with descendants  
        hierarchy_data = {
            "descendants": [
                {
                    "identifier": "DerivedClass1",
                    "location": {
                        "file_path": "/test/file.py",
                        "file_uri": "file:///test/file.py",
                        "line": 10,
                        "column": 0,
                        "end_line": 20,
                        "end_column": 0,
                        "collection": "test"
                    }
                },
                {
                    "identifier": "DerivedClass2", 
                    "location": {
                        "file_path": "/test/file2.py",
                        "file_uri": "file:///test/file2.py",
                        "line": 5,
                        "column": 0,
                        "end_line": 15,
                        "end_column": 0,
                        "collection": "test"
                    }
                }
            ]
        }
        
        inheritance_graph.get_class_hierarchy = AsyncMock(return_value=hierarchy_data)
        
        impact = await impact_analyzer.analyze_class_change(
            "BaseClass", "interface_change", ["test-project"]
        )
        
        assert impact.target_symbol == "BaseClass"
        assert impact.impact_level == ImpactLevel.HIGH
        assert len(impact.affected_symbols) == 2
        assert "2 subclasses" in impact.breaking_changes[0]
    
    async def test_analyze_module_change(self, impact_analyzer, mock_graphs):
        """Test analyzing impact of module changes"""
        _, import_graph, _ = mock_graphs
        
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        importer_nodes = [
            DependencyNode(f"importer_{i}", SymbolKind.MODULE, location)
            for i in range(25)  # 25 importers = CRITICAL impact
        ]
        
        import_graph.get_module_dependencies = AsyncMock(return_value={
            "imported_by": importer_nodes,
            "imports": []
        })
        
        impact = await impact_analyzer.analyze_module_change(
            "popular_module", "api_change", ["test-project"]
        )
        
        assert impact.impact_level == ImpactLevel.CRITICAL
        assert len(impact.affected_symbols) == 25
        assert "25 importing modules" in impact.breaking_changes[0]


@pytest.mark.asyncio
class TestDependencyQueryEngine:
    """Test high-level dependency query interface"""
    
    @pytest.fixture
    def mock_dependency_analyzer(self):
        analyzer = MagicMock()
        analyzer.call_graph = MagicMock()
        analyzer.import_graph = MagicMock()
        analyzer.inheritance_graph = MagicMock()
        return analyzer
    
    @pytest.fixture
    def query_engine(self, mock_dependency_analyzer):
        return DependencyQueryEngine(mock_dependency_analyzer)
    
    async def test_find_callers_query(self, query_engine, mock_dependency_analyzer):
        """Test find callers query interface"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        caller_nodes = [DependencyNode("caller1", SymbolKind.FUNCTION, location)]
        
        mock_dependency_analyzer.call_graph.find_callers = AsyncMock(return_value=caller_nodes)
        
        result = await query_engine.find_callers("test_function", ["test-project"])
        
        assert len(result) == 1
        assert result[0]["identifier"] == "caller1"
    
    async def test_find_dependencies_all(self, query_engine, mock_dependency_analyzer):
        """Test finding all dependencies of a symbol"""
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        
        # Mock call dependencies
        callees = [DependencyNode("callee1", SymbolKind.FUNCTION, location)]
        mock_dependency_analyzer.call_graph.find_callees = AsyncMock(return_value=callees)
        
        # Mock import dependencies
        mock_dependency_analyzer.import_graph.nodes = {"test_symbol": MagicMock()}
        mock_dependency_analyzer.import_graph.get_module_dependencies = AsyncMock(return_value={
            "imports": [DependencyNode("imported_module", SymbolKind.MODULE, location)],
            "imported_by": []
        })
        
        # Mock inheritance dependencies
        mock_dependency_analyzer.inheritance_graph.get_class_hierarchy = AsyncMock(return_value={
            "ancestors": [{"identifier": "BaseClass"}],
            "descendants": [{"identifier": "DerivedClass"}]
        })
        
        result = await query_engine.find_dependencies("test_symbol", "all", ["test-project"])
        
        assert "dependencies" in result
        assert "calls" in result["dependencies"]
        assert "imports" in result["dependencies"]
        assert "inheritance" in result["dependencies"]
        assert len(result["dependencies"]["calls"]) == 1
    
    async def test_analyze_circular_dependencies(self, query_engine, mock_dependency_analyzer):
        """Test analyzing circular dependencies"""
        from common.tools.dependency_analyzer import CircularDependency, CircularDependencyType
        
        # Mock circular import dependencies
        location = SymbolLocation("", "", 0, 0, 0, 0, "test")
        circular_dep = CircularDependency(
            CircularDependencyType.IMPORT,
            [DependencyNode("module.a", SymbolKind.MODULE, location)],
            ImpactLevel.HIGH
        )
        
        mock_dependency_analyzer.import_graph.find_circular_dependencies = AsyncMock(
            return_value=[circular_dep]
        )
        mock_dependency_analyzer.call_graph.find_recursive_calls = AsyncMock(
            return_value=[["func1", "func2", "func1"]]
        )
        
        result = await query_engine.analyze_circular_dependencies(["test-project"])
        
        assert "import_cycles" in result
        assert "recursive_calls" in result
        assert len(result["import_cycles"]) == 1
        assert len(result["recursive_calls"]) == 1
    
    async def test_get_dependency_statistics(self, query_engine, mock_dependency_analyzer):
        """Test getting dependency statistics"""
        # Mock graph statistics
        mock_dependency_analyzer.call_graph.nodes = {"func1": MagicMock(), "func2": MagicMock()}
        mock_dependency_analyzer.call_graph.edges = [MagicMock(), MagicMock()]
        
        mock_dependency_analyzer.import_graph.nodes = {"mod1": MagicMock()}
        mock_dependency_analyzer.import_graph.edges = [MagicMock()]
        mock_dependency_analyzer.import_graph.external_dependencies = {"numpy", "requests"}
        
        mock_dependency_analyzer.inheritance_graph.nodes = {"Class1": MagicMock()}
        mock_dependency_analyzer.inheritance_graph.edges = [MagicMock()]
        mock_dependency_analyzer.inheritance_graph.method_overrides = {"method1": ["override1"]}
        
        mock_dependency_analyzer.ensure_graphs_built = AsyncMock()
        
        stats = await query_engine.get_dependency_statistics(["test-project"])
        
        assert "call_graph" in stats
        assert "import_graph" in stats
        assert "inheritance_graph" in stats
        
        assert stats["call_graph"]["functions"] == 2
        assert stats["call_graph"]["calls"] == 2
        assert stats["import_graph"]["external_dependencies"] == 2
        assert stats["inheritance_graph"]["method_overrides"] == 1


@pytest.mark.asyncio
class TestDependencyAnalyzer:
    """Test main dependency analyzer orchestration"""
    
    @pytest.fixture
    def mock_workspace_client(self):
        return MagicMock()
    
    @pytest.fixture
    def dependency_analyzer(self, mock_workspace_client):
        return DependencyAnalyzer(mock_workspace_client)
    
    async def test_initialize(self, dependency_analyzer):
        """Test dependency analyzer initialization"""
        with patch('src.workspace_qdrant_mcp.tools.dependency_analyzer.SymbolResolver') as mock_resolver:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_resolver.return_value = mock_instance
            
            await dependency_analyzer.initialize()
            
            assert dependency_analyzer._initialized is True
            assert dependency_analyzer.symbol_resolver is not None
            assert dependency_analyzer.call_graph is not None
            assert dependency_analyzer.import_graph is not None
            assert dependency_analyzer.inheritance_graph is not None
            assert dependency_analyzer.impact_analyzer is not None
            assert dependency_analyzer.query_engine is not None
    
    async def test_ensure_graphs_built(self, dependency_analyzer):
        """Test ensuring dependency graphs are built"""
        # Mock initialization
        dependency_analyzer._initialized = True
        dependency_analyzer.call_graph = MagicMock()
        dependency_analyzer.import_graph = MagicMock()
        dependency_analyzer.inheritance_graph = MagicMock()
        
        # Mock empty graphs (need building)
        dependency_analyzer.call_graph.nodes = {}
        dependency_analyzer.import_graph.nodes = {}
        dependency_analyzer.inheritance_graph.nodes = {}
        
        # Mock build methods
        dependency_analyzer.call_graph.build_graph = AsyncMock()
        dependency_analyzer.import_graph.build_graph = AsyncMock()
        dependency_analyzer.inheritance_graph.build_graph = AsyncMock()
        
        await dependency_analyzer.ensure_graphs_built(["test-project"])
        
        # All build methods should have been called
        dependency_analyzer.call_graph.build_graph.assert_called_once_with(["test-project"])
        dependency_analyzer.import_graph.build_graph.assert_called_once_with(["test-project"])
        dependency_analyzer.inheritance_graph.build_graph.assert_called_once_with(["test-project"])
    
    async def test_rebuild_graphs(self, dependency_analyzer):
        """Test rebuilding dependency graphs"""
        # Mock initialization
        dependency_analyzer._initialized = True
        dependency_analyzer.call_graph = MagicMock()
        dependency_analyzer.import_graph = MagicMock()
        dependency_analyzer.inheritance_graph = MagicMock()
        
        # Mock graphs with existing data
        dependency_analyzer.call_graph.nodes = {"func1": MagicMock()}
        dependency_analyzer.import_graph.nodes = {"mod1": MagicMock()}  
        dependency_analyzer.inheritance_graph.nodes = {"class1": MagicMock()}
        
        # Mock clear and build methods
        dependency_analyzer.call_graph.clear = MagicMock()
        dependency_analyzer.import_graph.clear = MagicMock()
        dependency_analyzer.inheritance_graph.clear = MagicMock()
        
        dependency_analyzer.call_graph.build_graph = AsyncMock()
        dependency_analyzer.import_graph.build_graph = AsyncMock()
        dependency_analyzer.inheritance_graph.build_graph = AsyncMock()
        
        await dependency_analyzer.rebuild_graphs(["test-project"])
        
        # Clear methods should have been called
        dependency_analyzer.call_graph.clear.assert_called_once()
        dependency_analyzer.import_graph.clear.assert_called_once()
        dependency_analyzer.inheritance_graph.clear.assert_called_once()
        
        # Build methods should have been called
        dependency_analyzer.call_graph.build_graph.assert_called_once_with(["test-project"])
        dependency_analyzer.import_graph.build_graph.assert_called_once_with(["test-project"])
        dependency_analyzer.inheritance_graph.build_graph.assert_called_once_with(["test-project"])
    
    def test_get_analysis_summary(self, dependency_analyzer):
        """Test getting analysis summary"""
        # Mock partial initialization
        dependency_analyzer._initialized = True
        dependency_analyzer.call_graph = MagicMock()
        dependency_analyzer.import_graph = MagicMock()
        dependency_analyzer.inheritance_graph = MagicMock()
        
        # Mock some graphs built, some not
        dependency_analyzer.call_graph.nodes = {"func1": MagicMock()}
        dependency_analyzer.import_graph.nodes = {}
        dependency_analyzer.inheritance_graph.nodes = {"class1": MagicMock()}
        
        dependency_analyzer.call_graph._last_updated = 1234567890
        dependency_analyzer.import_graph._last_updated = None
        dependency_analyzer.inheritance_graph._last_updated = 1234567891
        
        summary = dependency_analyzer.get_analysis_summary()
        
        assert summary["initialized"] is True
        assert summary["graphs_built"]["call_graph"] is True
        assert summary["graphs_built"]["import_graph"] is False
        assert summary["graphs_built"]["inheritance_graph"] is True
        
        assert summary["last_updated"]["call_graph"] == 1234567890
        assert summary["last_updated"]["import_graph"] is None
        assert summary["last_updated"]["inheritance_graph"] == 1234567891


if __name__ == "__main__":
    pytest.main([__file__, "-v"])