"""
Comprehensive test suite for Symbol Definition Lookup System (Task #126)

Tests cover:
- Symbol resolution accuracy
- Lookup performance benchmarking (O(1) validation)
- Disambiguation correctness with parameter matching
- Cross-reference accuracy and usage tracking
- Hierarchy navigation for inheritance and containment
- Workspace filtering validation

Test Categories:
1. Core functionality tests
2. Performance benchmarks
3. Disambiguation tests
4. Cross-reference tests
5. Hierarchy navigation tests
6. Integration tests
7. Edge case and error handling tests
"""

import asyncio
import pytest
import time
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

from src.workspace_qdrant_mcp.tools.symbol_resolver import (
    SymbolResolver, SymbolIndex, SymbolEntry, SymbolLocation,
    DisambiguationEngine, CrossReferenceTracker, SymbolResolutionResult,
    SymbolKind, SymbolScope, find_symbol_definition, resolve_function_overload,
    analyze_symbol_usage
)
from src.workspace_qdrant_mcp.core.lsp_metadata_extractor import (
    CodeSymbol, Range, Position, TypeInformation
)
from src.workspace_qdrant_mcp.core.client import QdrantWorkspaceClient


class TestSymbolIndex:
    """Test the core hash-based symbol index for O(1) lookups"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.index = SymbolIndex(initial_capacity=100)
        self.sample_symbols = self._create_sample_symbols()
        
        # Add sample symbols to index
        for symbol in self.sample_symbols:
            self.index.add_symbol(symbol)
    
    def _create_sample_symbols(self) -> List[SymbolEntry]:
        """Create sample symbols for testing"""
        return [
            SymbolEntry(
                name="authenticate",
                qualified_name="auth.authenticate",
                symbol_id="auth001",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation(
                    file_path="/src/auth.py",
                    file_uri="file:///src/auth.py",
                    line=10,
                    column=0,
                    end_line=10,
                    end_column=20,
                    collection="main"
                ),
                language="python",
                signature="authenticate(username: str, password: str) -> bool",
                return_type="bool",
                parameter_types=["str", "str"]
            ),
            SymbolEntry(
                name="authenticate",
                qualified_name="api.authenticate",
                symbol_id="api001",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation(
                    file_path="/src/api.py",
                    file_uri="file:///src/api.py",
                    line=25,
                    column=0,
                    end_line=25,
                    end_column=20,
                    collection="main"
                ),
                language="python",
                signature="authenticate(token: str) -> dict",
                return_type="dict",
                parameter_types=["str"]
            ),
            SymbolEntry(
                name="User",
                qualified_name="models.User",
                symbol_id="user001",
                kind=SymbolKind.CLASS,
                location=SymbolLocation(
                    file_path="/src/models.py",
                    file_uri="file:///src/models.py",
                    line=5,
                    column=0,
                    end_line=50,
                    end_column=0,
                    collection="main"
                ),
                language="python"
            ),
            SymbolEntry(
                name="login",
                qualified_name="models.User.login",
                symbol_id="user_login001",
                kind=SymbolKind.METHOD,
                location=SymbolLocation(
                    file_path="/src/models.py",
                    file_uri="file:///src/models.py",
                    line=15,
                    column=4,
                    end_line=20,
                    end_column=0,
                    collection="main"
                ),
                language="python",
                parent_symbol="models.User"
            )
        ]
    
    def test_o1_lookup_by_name(self):
        """Test O(1) lookup performance by name"""
        start_time = time.perf_counter()
        
        # Perform multiple lookups
        for _ in range(1000):
            results = self.index.find_by_name("authenticate")
            assert len(results) == 2  # Two overloads
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # Should be very fast (< 10ms for 1000 lookups)
        assert elapsed < 10.0, f"Lookup took {elapsed:.2f}ms, expected < 10ms"
    
    def test_o1_lookup_by_qualified_name(self):
        """Test O(1) lookup by qualified name"""
        result = self.index.find_by_qualified_name("auth.authenticate")
        
        assert result is not None
        assert result.name == "authenticate"
        assert result.qualified_name == "auth.authenticate"
        assert result.location.file_path == "/src/auth.py"
    
    def test_lookup_by_kind(self):
        """Test lookup by symbol kind"""
        functions = self.index.find_by_kind(SymbolKind.FUNCTION)
        classes = self.index.find_by_kind(SymbolKind.CLASS)
        methods = self.index.find_by_kind(SymbolKind.METHOD)
        
        assert len(functions) == 2
        assert len(classes) == 1
        assert len(methods) == 1
        
        # Verify correct kinds
        for func in functions:
            assert func.kind == SymbolKind.FUNCTION
    
    def test_lookup_by_collection(self):
        """Test lookup by collection"""
        main_symbols = self.index.find_by_collection("main")
        assert len(main_symbols) == 4
        
        # Test non-existent collection
        empty_symbols = self.index.find_by_collection("nonexistent")
        assert len(empty_symbols) == 0
    
    def test_find_children(self):
        """Test finding child symbols"""
        children = self.index.find_children("models.User")
        assert len(children) == 1
        assert children[0].name == "login"
        assert children[0].parent_symbol == "models.User"
    
    def test_find_overloads(self):
        """Test finding function overloads"""
        overloads = self.index.find_overloads("authenticate")
        assert len(overloads) == 2
        
        # Should include both function overloads
        qualified_names = {o.qualified_name for o in overloads}
        assert "auth.authenticate" in qualified_names
        assert "api.authenticate" in qualified_names
    
    def test_index_statistics(self):
        """Test index statistics tracking"""
        stats = self.index.get_statistics()
        
        assert stats["index_size"] == 4
        assert stats["by_name_entries"] >= 3  # authenticate, User, login
        assert stats["function_overloads"] >= 1
        assert stats["collections"] == 1
        assert stats["languages"] == 1


class TestDisambiguationEngine:
    """Test symbol disambiguation with parameter matching"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = DisambiguationEngine()
        self.candidates = self._create_disambiguation_candidates()
    
    def _create_disambiguation_candidates(self) -> List[SymbolEntry]:
        """Create candidates for disambiguation testing"""
        return [
            SymbolEntry(
                name="process",
                qualified_name="service.process",
                symbol_id="process001",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation("/src/service.py", "file:///src/service.py", 10, 0, 10, 20, "main"),
                language="python",
                signature="process(data: str) -> str",
                return_type="str",
                parameter_types=["str"],
                confidence_score=0.9
            ),
            SymbolEntry(
                name="process",
                qualified_name="service.process",
                symbol_id="process002",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation("/src/service.py", "file:///src/service.py", 20, 0, 20, 20, "main"),
                language="python",
                signature="process(data: dict, options: dict = None) -> dict",
                return_type="dict",
                parameter_types=["dict", "dict"],
                confidence_score=0.8
            ),
            SymbolEntry(
                name="process",
                qualified_name="utils.process",
                symbol_id="process003",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation("/src/utils.py", "file:///src/utils.py", 5, 0, 5, 20, "utils"),
                language="python",
                signature="process(items: list) -> list",
                return_type="list",
                parameter_types=["list"],
                confidence_score=0.7,
                is_deprecated=True
            )
        ]
    
    def test_signature_matching_exact(self):
        """Test exact parameter signature matching"""
        results = self.engine.disambiguate_symbols(
            self.candidates,
            parameter_types=["str"],
            return_type="str"
        )
        
        # First result should be exact match
        assert len(results) >= 1
        assert results[0].qualified_name == "service.process"
        assert results[0].parameter_types == ["str"]
        assert results[0].return_type == "str"
    
    def test_signature_matching_partial(self):
        """Test partial parameter signature matching"""
        results = self.engine.disambiguate_symbols(
            self.candidates,
            parameter_types=["dict"],
            return_type="dict"
        )
        
        # Should prefer the dict-based overload
        assert len(results) >= 1
        assert results[0].parameter_types == ["dict", "dict"]
        assert results[0].return_type == "dict"
    
    def test_context_preference(self):
        """Test context-based disambiguation"""
        results = self.engine.disambiguate_symbols(
            self.candidates,
            context_file="/src/service.py"
        )
        
        # Should prefer symbols from same file
        top_results = results[:2]
        for result in top_results:
            assert result.location.file_path == "/src/service.py"
    
    def test_deprecation_penalty(self):
        """Test deprecation penalty in disambiguation"""
        results = self.engine.disambiguate_symbols(self.candidates)
        
        # Deprecated symbol should be ranked lower
        deprecated_positions = [
            i for i, result in enumerate(results)
            if result.is_deprecated
        ]
        
        # Deprecated symbol should not be first
        assert len(deprecated_positions) > 0
        assert deprecated_positions[0] > 0  # Not in first position
    
    def test_disambiguation_caching(self):
        """Test disambiguation result caching"""
        # First call
        results1 = self.engine.disambiguate_symbols(
            self.candidates,
            parameter_types=["str"]
        )
        
        # Second call (should use cache)
        results2 = self.engine.disambiguate_symbols(
            self.candidates,
            parameter_types=["str"]
        )
        
        assert results1 == results2
        
        stats = self.engine.get_statistics()
        assert stats["cache_hits"] > 0


class TestCrossReferenceTracker:
    """Test cross-reference tracking and impact analysis"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = CrossReferenceTracker()
        self._populate_test_data()
    
    def _populate_test_data(self):
        """Populate tracker with test reference data"""
        # Add symbol references
        locations = [
            SymbolLocation("/src/main.py", "file:///src/main.py", 5, 10, 5, 20, "main"),
            SymbolLocation("/src/app.py", "file:///src/app.py", 15, 5, 15, 15, "main"),
            SymbolLocation("/src/tests.py", "file:///src/tests.py", 25, 0, 25, 10, "test")
        ]
        
        for location in locations:
            self.tracker.add_symbol_reference("auth.authenticate", location)
        
        # Add symbol dependencies
        self.tracker.add_symbol_dependency("api.login", "auth.authenticate")
        self.tracker.add_symbol_dependency("service.process", "auth.authenticate")
        self.tracker.add_symbol_dependency("auth.authenticate", "models.User")
    
    def test_reference_tracking(self):
        """Test basic reference tracking"""
        references = self.tracker.get_symbol_references("auth.authenticate")
        
        assert len(references) == 3
        
        # Check file paths
        file_paths = {ref.file_path for ref in references}
        expected_paths = {"/src/main.py", "/src/app.py", "/src/tests.py"}
        assert file_paths == expected_paths
    
    def test_dependency_tracking(self):
        """Test dependency relationship tracking"""
        # Test direct dependencies
        deps = self.tracker.get_symbol_dependencies("auth.authenticate", recursive=False)
        assert "models.User" in deps
        assert len(deps) == 1
        
        # Test reverse dependencies
        dependents = self.tracker.get_symbol_dependents("auth.authenticate", recursive=False)
        assert "api.login" in dependents
        assert "service.process" in dependents
        assert len(dependents) == 2
    
    def test_recursive_dependencies(self):
        """Test recursive dependency resolution"""
        # Add more complex dependency chain
        self.tracker.add_symbol_dependency("models.User", "database.Connection")
        self.tracker.add_symbol_dependency("database.Connection", "config.Settings")
        
        # Test recursive dependencies
        recursive_deps = self.tracker.get_symbol_dependencies("auth.authenticate", recursive=True)
        
        assert "models.User" in recursive_deps
        assert "database.Connection" in recursive_deps
        assert "config.Settings" in recursive_deps
        assert len(recursive_deps) >= 3
    
    def test_impact_analysis(self):
        """Test comprehensive impact analysis"""
        analysis = self.tracker.analyze_impact("auth.authenticate")
        
        # Verify analysis structure
        assert analysis["symbol"] == "auth.authenticate"
        assert analysis["reference_count"] == 3
        assert analysis["direct_dependents"] == 2
        assert analysis["affected_files"] == 3
        assert analysis["usage_frequency"] >= 3
        assert analysis["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert "impact_score" in analysis
        assert isinstance(analysis["affected_file_list"], list)
        assert isinstance(analysis["dependent_symbols"], list)
    
    def test_popular_symbols(self):
        """Test popular symbol tracking"""
        # Add more usage for different symbols
        location = SymbolLocation("/src/test2.py", "file:///src/test2.py", 10, 0, 10, 10, "test")
        
        for _ in range(5):
            self.tracker.add_symbol_reference("utils.helper", location)
        
        popular = self.tracker.get_popular_symbols(limit=5)
        
        assert len(popular) >= 2
        assert isinstance(popular[0], tuple)
        
        # Most popular should be utils.helper (5 uses)
        assert popular[0][0] == "utils.helper"
        assert popular[0][1] == 5
    
    def test_statistics(self):
        """Test tracker statistics"""
        stats = self.tracker.get_statistics()
        
        assert stats["total_references"] >= 3
        assert stats["tracked_symbols"] >= 1
        assert stats["dependencies"] >= 3
        assert stats["tracking_enabled"] is True
        assert stats["average_references_per_symbol"] > 0


@pytest.fixture
async def mock_workspace_client():
    """Create mock workspace client for testing"""
    client = Mock(spec=QdrantWorkspaceClient)
    client.list_collections = AsyncMock(return_value=["main", "test", "docs"])
    return client


@pytest.fixture
async def symbol_resolver(mock_workspace_client):
    """Create symbol resolver with mocked dependencies"""
    with patch('src.workspace_qdrant_mcp.tools.symbol_resolver.CodeSearchEngine'):
        with patch('src.workspace_qdrant_mcp.tools.symbol_resolver.search_collection_by_metadata') as mock_search:
            # Mock collection search results
            mock_search.return_value = {
                "results": [
                    {
                        "payload": {
                            "symbol": {
                                "name": "test_function",
                                "kind": 12,  # FUNCTION
                                "file_uri": "file:///test.py",
                                "range": {
                                    "start": {"line": 0, "character": 0},
                                    "end": {"line": 0, "character": 20}
                                },
                                "type_info": {
                                    "type_signature": "test_function() -> None",
                                    "parameter_types": [],
                                    "return_type": "None"
                                },
                                "language": "python",
                                "parent_symbol": "",
                                "visibility": "public"
                            }
                        }
                    }
                ]
            }
            
            resolver = SymbolResolver(mock_workspace_client)
            await resolver.initialize()
            return resolver


class TestSymbolResolver:
    """Test the main SymbolResolver class functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, symbol_resolver):
        """Test resolver initialization"""
        assert symbol_resolver.initialized is True
        assert len(symbol_resolver.indexed_collections) > 0
        
        stats = symbol_resolver.get_statistics()
        assert stats["initialized"] is True
        assert stats["indexed_collections"] > 0
    
    @pytest.mark.asyncio
    async def test_find_symbol_definitions(self, symbol_resolver):
        """Test finding symbol definitions"""
        results = await symbol_resolver.find_symbol_definitions("test_function")
        
        assert len(results) > 0
        assert all(isinstance(r, SymbolResolutionResult) for r in results)
        
        # Check first result
        result = results[0]
        assert result.symbol.name == "test_function"
        assert result.symbol.kind == SymbolKind.FUNCTION
        assert result.match_confidence > 0
        assert result.resolution_method == "index_lookup"
    
    @pytest.mark.asyncio
    async def test_symbol_resolution_with_collections_filter(self, symbol_resolver):
        """Test symbol resolution with collection filtering"""
        results = await symbol_resolver.find_symbol_definitions(
            "test_function", 
            collections=["main"]
        )
        
        # Should only return symbols from specified collections
        for result in results:
            assert result.symbol.location.collection in ["main"]
    
    @pytest.mark.asyncio
    async def test_symbol_resolution_caching(self, symbol_resolver):
        """Test symbol resolution result caching"""
        # First call
        start_time = time.perf_counter()
        results1 = await symbol_resolver.find_symbol_definitions("test_function")
        first_call_time = (time.perf_counter() - start_time) * 1000
        
        # Second call (should be cached)
        start_time = time.perf_counter()
        results2 = await symbol_resolver.find_symbol_definitions("test_function")
        second_call_time = (time.perf_counter() - start_time) * 1000
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        # Second call should be significantly faster (cached)
        assert second_call_time < first_call_time * 0.5
        
        # Verify cache hit in statistics
        stats = symbol_resolver.get_statistics()
        assert stats["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_resolve_symbol_with_params(self, symbol_resolver):
        """Test resolving symbols with parameter matching"""
        # Add a symbol with parameters to the index
        symbol_with_params = SymbolEntry(
            name="process_data",
            qualified_name="service.process_data",
            symbol_id="process_data001",
            kind=SymbolKind.FUNCTION,
            location=SymbolLocation("/service.py", "file:///service.py", 10, 0, 10, 30, "main"),
            language="python",
            signature="process_data(data: str, options: dict) -> dict",
            parameter_types=["str", "dict"],
            return_type="dict"
        )
        symbol_resolver.symbol_index.add_symbol(symbol_with_params)
        
        results = await symbol_resolver.resolve_symbol_with_params(
            "process_data",
            parameter_types=["str", "dict"],
            return_type="dict"
        )
        
        assert len(results) > 0
        result = results[0]
        assert result.resolution_method == "signature_match"
        assert result.disambiguation_info is not None
        assert result.match_confidence > 0.8  # High confidence for exact match
    
    @pytest.mark.asyncio
    async def test_search_symbols_by_kind(self, symbol_resolver):
        """Test searching symbols by kind"""
        functions = await symbol_resolver.search_symbols_by_kind(SymbolKind.FUNCTION)
        assert len(functions) > 0
        assert all(symbol.kind == SymbolKind.FUNCTION for symbol in functions)
        
        classes = await symbol_resolver.search_symbols_by_kind(SymbolKind.CLASS)
        # Should return empty list for classes if none exist
        assert isinstance(classes, list)
    
    @pytest.mark.asyncio
    async def test_get_symbol_hierarchy(self, symbol_resolver):
        """Test symbol hierarchy navigation"""
        # Add parent and child symbols
        parent_symbol = SymbolEntry(
            name="BaseClass",
            qualified_name="models.BaseClass",
            symbol_id="base001",
            kind=SymbolKind.CLASS,
            location=SymbolLocation("/models.py", "file:///models.py", 5, 0, 20, 0, "main"),
            language="python"
        )
        
        child_symbol = SymbolEntry(
            name="method",
            qualified_name="models.BaseClass.method",
            symbol_id="method001",
            kind=SymbolKind.METHOD,
            location=SymbolLocation("/models.py", "file:///models.py", 10, 4, 15, 0, "main"),
            language="python",
            parent_symbol="models.BaseClass"
        )
        
        symbol_resolver.symbol_index.add_symbol(parent_symbol)
        symbol_resolver.symbol_index.add_symbol(child_symbol)
        
        hierarchy = await symbol_resolver.get_symbol_hierarchy("models.BaseClass")
        
        assert "symbol" in hierarchy
        assert "children" in hierarchy
        assert "parents" in hierarchy
        
        # Should have one child
        assert len(hierarchy["children"]) == 1
        assert hierarchy["children"][0]["name"] == "method"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, symbol_resolver):
        """Test O(1) performance characteristics"""
        # Add multiple symbols for testing
        symbols_to_add = []
        for i in range(1000):
            symbol = SymbolEntry(
                name=f"func_{i}",
                qualified_name=f"module.func_{i}",
                symbol_id=f"func_{i:04d}",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation(f"/file_{i}.py", f"file:///file_{i}.py", 1, 0, 1, 20, "perf_test"),
                language="python"
            )
            symbols_to_add.append(symbol)
        
        # Add symbols to index
        start_time = time.perf_counter()
        for symbol in symbols_to_add:
            symbol_resolver.symbol_index.add_symbol(symbol)
        index_time = (time.perf_counter() - start_time) * 1000
        
        # Test lookup performance
        lookup_times = []
        for i in range(100):
            start_time = time.perf_counter()
            results = await symbol_resolver.find_symbol_definitions(f"func_{i}")
            lookup_time = (time.perf_counter() - start_time) * 1000
            lookup_times.append(lookup_time)
        
        avg_lookup_time = sum(lookup_times) / len(lookup_times)
        max_lookup_time = max(lookup_times)
        
        # Performance assertions
        assert avg_lookup_time < 5.0, f"Average lookup time {avg_lookup_time:.2f}ms too slow"
        assert max_lookup_time < 20.0, f"Max lookup time {max_lookup_time:.2f}ms too slow"
        
        # Index size shouldn't significantly affect lookup time (O(1) characteristic)
        stats = symbol_resolver.get_statistics()
        assert stats["average_resolution_time_ms"] < 10.0


class TestConvenienceFunctions:
    """Test convenience functions for common operations"""
    
    @pytest.mark.asyncio
    async def test_find_symbol_definition_convenience(self, mock_workspace_client):
        """Test convenience function for finding symbol definitions"""
        with patch('src.workspace_qdrant_mcp.tools.symbol_resolver.SymbolResolver') as MockResolver:
            mock_resolver_instance = Mock()
            mock_resolver_instance.initialize = AsyncMock()
            mock_resolver_instance.find_symbol_definitions = AsyncMock(return_value=[])
            mock_resolver_instance.shutdown = AsyncMock()
            MockResolver.return_value = mock_resolver_instance
            
            results = await find_symbol_definition(
                mock_workspace_client,
                "test_symbol"
            )
            
            # Verify resolver was used correctly
            MockResolver.assert_called_once_with(mock_workspace_client)
            mock_resolver_instance.initialize.assert_called_once()
            mock_resolver_instance.find_symbol_definitions.assert_called_once_with(
                symbol_name="test_symbol",
                collections=None,
                context_file=None
            )
            mock_resolver_instance.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resolve_function_overload_convenience(self, mock_workspace_client):
        """Test convenience function for resolving function overloads"""
        with patch('src.workspace_qdrant_mcp.tools.symbol_resolver.SymbolResolver') as MockResolver:
            mock_resolver_instance = Mock()
            mock_resolver_instance.initialize = AsyncMock()
            mock_resolver_instance.resolve_symbol_with_params = AsyncMock(return_value=[])
            mock_resolver_instance.shutdown = AsyncMock()
            MockResolver.return_value = mock_resolver_instance
            
            results = await resolve_function_overload(
                mock_workspace_client,
                "test_function",
                ["str", "int"]
            )
            
            # Verify resolver was used correctly
            mock_resolver_instance.resolve_symbol_with_params.assert_called_once_with(
                symbol_name="test_function",
                parameter_types=["str", "int"],
                return_type=None,
                collections=None
            )


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios"""
    
    def test_empty_symbol_index(self):
        """Test behavior with empty symbol index"""
        index = SymbolIndex()
        
        # All lookups should return empty results
        assert index.find_by_name("nonexistent") == []
        assert index.find_by_qualified_name("nonexistent") is None
        assert index.find_by_kind(SymbolKind.FUNCTION) == []
        assert index.find_by_collection("nonexistent") == []
    
    def test_invalid_symbol_data_handling(self):
        """Test handling of invalid symbol data"""
        index = SymbolIndex()
        
        # Try to add symbol with missing required fields
        incomplete_symbol = SymbolEntry(
            name="",  # Empty name
            qualified_name="",
            symbol_id="invalid001",
            kind=SymbolKind.FUNCTION,
            location=SymbolLocation("", "", 0, 0, 0, 0, ""),
            language=""
        )
        
        # Should handle gracefully without crashing
        index.add_symbol(incomplete_symbol)
        
        # Lookup should still work but return empty
        results = index.find_by_name("")
        assert len(results) == 1  # Symbol was added despite being incomplete
    
    @pytest.mark.asyncio
    async def test_disambiguation_with_no_candidates(self):
        """Test disambiguation with empty candidate list"""
        engine = DisambiguationEngine()
        results = engine.disambiguate_symbols([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_disambiguation_with_single_candidate(self):
        """Test disambiguation with single candidate"""
        engine = DisambiguationEngine()
        single_candidate = [SymbolEntry(
            name="single",
            qualified_name="test.single",
            symbol_id="single001",
            kind=SymbolKind.FUNCTION,
            location=SymbolLocation("/test.py", "file:///test.py", 1, 0, 1, 10, "test"),
            language="python"
        )]
        
        results = engine.disambiguate_symbols(single_candidate)
        assert len(results) == 1
        assert results[0] == single_candidate[0]
    
    def test_cross_reference_disabled_tracking(self):
        """Test cross-reference tracker when disabled"""
        tracker = CrossReferenceTracker()
        tracker.tracking_enabled = False
        
        location = SymbolLocation("/test.py", "file:///test.py", 1, 0, 1, 10, "test")
        
        # Should not track when disabled
        tracker.add_symbol_reference("test.symbol", location)
        
        references = tracker.get_symbol_references("test.symbol")
        assert len(references) == 0  # Nothing should be tracked
    
    @pytest.mark.asyncio
    async def test_resolver_statistics_accuracy(self, symbol_resolver):
        """Test accuracy of resolver statistics"""
        # Perform some operations
        await symbol_resolver.find_symbol_definitions("test_function")
        await symbol_resolver.find_symbol_definitions("test_function")  # Cache hit
        await symbol_resolver.find_symbol_definitions("nonexistent")
        
        stats = symbol_resolver.get_statistics()
        
        # Verify statistics accuracy
        assert stats["resolution_count"] >= 3
        assert stats["cache_hits"] >= 1
        assert stats["cache_hit_rate_percent"] > 0
        assert stats["indexed_collections"] > 0
        
        # Verify nested statistics exist
        assert "symbol_index" in stats
        assert "disambiguation_engine" in stats
        if symbol_resolver.cross_ref_tracker:
            assert "cross_reference_tracker" in stats


# Performance benchmark suite
class TestPerformanceBenchmarks:
    """Dedicated performance benchmarking tests"""
    
    @pytest.mark.asyncio
    async def test_large_scale_indexing_performance(self):
        """Test performance with large number of symbols"""
        index = SymbolIndex(initial_capacity=10000)
        
        # Create 10,000 symbols
        symbols = []
        for i in range(10000):
            symbol = SymbolEntry(
                name=f"symbol_{i}",
                qualified_name=f"module_{i // 100}.symbol_{i}",
                symbol_id=f"sym_{i:05d}",
                kind=SymbolKind.FUNCTION if i % 2 == 0 else SymbolKind.VARIABLE,
                location=SymbolLocation(
                    f"/file_{i // 100}.py", 
                    f"file:///file_{i // 100}.py", 
                    i % 1000, 0, i % 1000, 20, 
                    f"collection_{i // 1000}"
                ),
                language="python"
            )
            symbols.append(symbol)
        
        # Time indexing
        start_time = time.perf_counter()
        for symbol in symbols:
            index.add_symbol(symbol)
        indexing_time = (time.perf_counter() - start_time) * 1000
        
        # Time lookups
        lookup_times = []
        for i in range(0, 1000, 10):  # Test every 10th symbol
            start_time = time.perf_counter()
            result = index.find_by_name(f"symbol_{i}")
            lookup_time = (time.perf_counter() - start_time) * 1000
            lookup_times.append(lookup_time)
        
        avg_lookup_time = sum(lookup_times) / len(lookup_times)
        
        # Performance assertions
        assert indexing_time < 2000, f"Indexing 10K symbols took {indexing_time:.0f}ms (>2s)"
        assert avg_lookup_time < 1.0, f"Average lookup time {avg_lookup_time:.3f}ms too slow"
        
        stats = index.get_statistics()
        assert stats["index_size"] == 10000
        assert stats["hit_rate_percent"] > 90  # Most lookups should hit
    
    def test_memory_efficiency(self):
        """Test memory efficiency of symbol storage"""
        import sys
        
        index = SymbolIndex()
        
        # Measure baseline memory
        baseline_size = sys.getsizeof(index)
        
        # Add 1000 symbols and measure memory growth
        for i in range(1000):
            symbol = SymbolEntry(
                name=f"test_{i}",
                qualified_name=f"module.test_{i}",
                symbol_id=f"test_{i:04d}",
                kind=SymbolKind.FUNCTION,
                location=SymbolLocation(
                    "/test.py", "file:///test.py", i, 0, i, 20, "test"
                ),
                language="python"
            )
            index.add_symbol(symbol)
        
        final_size = sys.getsizeof(index)
        memory_per_symbol = (final_size - baseline_size) / 1000
        
        # Should be reasonable memory usage per symbol (< 1KB per symbol for index overhead)
        assert memory_per_symbol < 1024, f"Memory per symbol {memory_per_symbol:.0f} bytes too high"


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run only performance benchmarks
        pytest.main([__file__ + "::TestPerformanceBenchmarks", "-v"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])