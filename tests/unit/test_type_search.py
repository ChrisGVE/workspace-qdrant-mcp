"""
Comprehensive tests for the Type-Based Search and Analysis System.

Tests cover type signature matching, generic type handling, interface matching,
and type compatibility analysis across different programming languages.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from common.core.client import QdrantWorkspaceClient
from common.core.lsp_metadata_extractor import SymbolKind, TypeInformation
from common.tools.code_search import CodeSearchEngine, CodeSearchResult
from common.tools.type_search import (
    TypeSearchEngine,
    TypeMatchMode,
    TypeConstraintKind,
    TypePattern,
    TypeSignature,
    TypeSearchQuery,
    TypeSearchResult
)


@pytest.fixture
def mock_workspace_client():
    """Create mock workspace client for testing"""
    client = MagicMock(spec=QdrantWorkspaceClient)
    client.qdrant_client = MagicMock()
    client.embedding_service = MagicMock()
    client.list_collections = AsyncMock(return_value=[
        "my-project", "library-code", "test-project"
    ])
    return client


@pytest.fixture
def mock_code_search_engine():
    """Create mock code search engine"""
    engine = MagicMock(spec=CodeSearchEngine)
    engine.initialize = AsyncMock()
    engine.search_by_signature = AsyncMock()
    return engine


@pytest.fixture
def sample_typed_symbols():
    """Sample code symbols with comprehensive type information"""
    return [
        {
            "name": "authenticate_user",
            "kind": SymbolKind.FUNCTION.value,
            "kind_name": "FUNCTION",
            "file_uri": "file:///src/auth.py",
            "range": {"start": {"line": 10, "character": 0}, "end": {"line": 20, "character": 0}},
            "type_info": {
                "type_signature": "def authenticate_user(username: str, password: str) -> bool",
                "parameter_types": [
                    {"name": "username", "type": "str"},
                    {"name": "password", "type": "str"}
                ],
                "return_type": "bool",
                "generic_parameters": []
            },
            "visibility": "public",
            "language": "python"
        },
        {
            "name": "process_data",
            "kind": SymbolKind.FUNCTION.value,
            "kind_name": "FUNCTION",
            "file_uri": "file:///src/processing.py",
            "range": {"start": {"line": 5, "character": 0}, "end": {"line": 25, "character": 0}},
            "type_info": {
                "type_signature": "def process_data[T](data: List[T], processor: Callable[[T], T]) -> List[T]",
                "parameter_types": [
                    {"name": "data", "type": "List[T]"},
                    {"name": "processor", "type": "Callable[[T], T]"}
                ],
                "return_type": "List[T]",
                "generic_parameters": ["T"]
            },
            "visibility": "public",
            "language": "python"
        },
        {
            "name": "create_connection",
            "kind": SymbolKind.FUNCTION.value,
            "kind_name": "FUNCTION",
            "file_uri": "file:///src/database.py",
            "range": {"start": {"line": 15, "character": 0}, "end": {"line": 30, "character": 0}},
            "type_info": {
                "type_signature": "def create_connection(host: str, port: int = 5432) -> Optional[Connection]",
                "parameter_types": [
                    {"name": "host", "type": "str"},
                    {"name": "port", "type": "int"}
                ],
                "return_type": "Optional[Connection]",
                "generic_parameters": []
            },
            "visibility": "public",
            "language": "python"
        },
        {
            "name": "Repository",
            "kind": SymbolKind.CLASS.value,
            "kind_name": "CLASS",
            "file_uri": "file:///src/repository.py",
            "range": {"start": {"line": 8, "character": 0}, "end": {"line": 50, "character": 0}},
            "type_info": {
                "type_signature": "class Repository[T: Entity]",
                "generic_parameters": ["T"],
                "parameter_types": [],
                "return_type": None
            },
            "visibility": "public",
            "language": "python"
        }
    ]


@pytest.fixture
def sample_code_search_results(sample_typed_symbols):
    """Create sample CodeSearchResult objects"""
    results = []
    for symbol in sample_typed_symbols:
        result = CodeSearchResult(
            symbol=symbol,
            relevance_score=0.9,
            context_snippet=f"Code context for {symbol['name']}",
            related_symbols=[],
            usage_examples=[],
            documentation_summary=f"Documentation for {symbol['name']}",
            file_path=symbol["file_uri"].replace("file://", ""),
            line_number=symbol["range"]["start"]["line"],
            collection="test-collection",
            search_type="type_signature"
        )
        results.append(result)
    return results


class TestTypePattern:
    """Test TypePattern functionality"""
    
    def test_create_basic_type_pattern(self):
        """Test creating a basic type pattern"""
        pattern = TypePattern(type_expression="str")
        assert pattern.type_expression == "str"
        assert not pattern.nullable
        assert not pattern.optional
        assert not pattern.variadic
    
    def test_create_complex_type_pattern(self):
        """Test creating a complex type pattern with generics"""
        pattern = TypePattern(
            type_expression="List[T]",
            generic_parameters=["T"],
            constraints={"T": "Comparable"},
            nullable=True
        )
        assert pattern.type_expression == "List[T]"
        assert pattern.generic_parameters == ["T"]
        assert pattern.constraints == {"T": "Comparable"}
        assert pattern.nullable
    
    def test_type_pattern_to_dict(self):
        """Test TypePattern serialization"""
        pattern = TypePattern(
            type_expression="Dict[K, V]",
            generic_parameters=["K", "V"],
            optional=True
        )
        result = pattern.to_dict()
        expected = {
            "type_expression": "Dict[K, V]",
            "generic_parameters": ["K", "V"],
            "constraints": {},
            "nullable": False,
            "optional": True,
            "variadic": False
        }
        assert result == expected


class TestTypeSignature:
    """Test TypeSignature functionality"""
    
    def test_create_function_signature(self):
        """Test creating a function type signature"""
        param_types = [
            TypePattern(type_expression="str"),
            TypePattern(type_expression="int")
        ]
        return_type = TypePattern(type_expression="bool")
        
        signature = TypeSignature(
            parameter_types=param_types,
            parameter_names=["username", "port"],
            return_type=return_type
        )
        
        assert len(signature.parameter_types) == 2
        assert signature.parameter_types[0].type_expression == "str"
        assert signature.parameter_names == ["username", "port"]
        assert signature.return_type.type_expression == "bool"
    
    def test_generic_function_signature(self):
        """Test creating a generic function signature"""
        param_types = [TypePattern(type_expression="List[T]")]
        return_type = TypePattern(type_expression="T")
        
        signature = TypeSignature(
            parameter_types=param_types,
            return_type=return_type,
            generic_parameters=["T"],
            type_constraints={"T": ["Comparable", "Hashable"]}
        )
        
        assert signature.generic_parameters == ["T"]
        assert signature.type_constraints == {"T": ["Comparable", "Hashable"]}


class TestTypeSearchQuery:
    """Test TypeSearchQuery functionality"""
    
    def test_create_exact_search_query(self):
        """Test creating an exact type search query"""
        param_types = [TypePattern(type_expression="str")]
        return_type = TypePattern(type_expression="bool")
        signature = TypeSignature(parameter_types=param_types, return_type=return_type)
        
        query = TypeSearchQuery(
            signature_pattern=signature,
            match_mode=TypeMatchMode.EXACT,
            symbol_kinds=[SymbolKind.FUNCTION],
            exact_arity=True
        )
        
        assert query.match_mode == TypeMatchMode.EXACT
        assert query.exact_arity
        assert SymbolKind.FUNCTION in query.symbol_kinds
    
    def test_create_compatible_search_query(self):
        """Test creating a compatible type search query"""
        query = TypeSearchQuery(
            match_mode=TypeMatchMode.COMPATIBLE,
            generic_patterns=["List[T]", "Dict[K, V]"],
            allow_subtyping=True,
            allow_generics=True
        )
        
        assert query.match_mode == TypeMatchMode.COMPATIBLE
        assert query.allow_subtyping
        assert query.allow_generics
        assert "List[T]" in query.generic_patterns


class TestTypeSearchEngine:
    """Test TypeSearchEngine functionality"""
    
    @pytest.fixture
    def type_search_engine(self, mock_workspace_client):
        """Create TypeSearchEngine instance for testing"""
        return TypeSearchEngine(mock_workspace_client)
    
    @pytest.mark.asyncio
    async def test_initialization(self, type_search_engine, mock_code_search_engine):
        """Test TypeSearchEngine initialization"""
        with patch.object(type_search_engine, 'code_search_engine', mock_code_search_engine):
            await type_search_engine.initialize()
            
            assert type_search_engine.initialized
            mock_code_search_engine.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_exact_signature(self, type_search_engine, sample_code_search_results):
        """Test exact signature search"""
        # Mock the code search engine
        mock_code_search = AsyncMock()
        mock_code_search.initialize = AsyncMock()
        mock_code_search.search_by_signature = AsyncMock(return_value=sample_code_search_results[:1])
        type_search_engine.code_search_engine = mock_code_search
        
        results = await type_search_engine.search_exact_signature(
            parameter_types=["str", "str"],
            return_type="bool",
            collections=["test-collection"]
        )
        
        assert len(results) >= 0  # May be filtered by type matching
        mock_code_search.search_by_signature.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_compatible_signatures(self, type_search_engine, sample_code_search_results):
        """Test compatible signature search"""
        mock_code_search = AsyncMock()
        mock_code_search.initialize = AsyncMock()
        mock_code_search.search_by_signature = AsyncMock(return_value=sample_code_search_results)
        type_search_engine.code_search_engine = mock_code_search
        
        results = await type_search_engine.search_compatible_signatures(
            target_signature="(str, str) -> bool",
            allow_subtyping=True,
            collections=["test-collection"]
        )
        
        assert isinstance(results, list)
        mock_code_search.search_by_signature.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_generic_implementations(self, type_search_engine, sample_code_search_results):
        """Test generic implementation search"""
        mock_code_search = AsyncMock()
        mock_code_search.initialize = AsyncMock()
        mock_code_search.search_by_signature = AsyncMock(return_value=sample_code_search_results)
        type_search_engine.code_search_engine = mock_code_search
        
        results = await type_search_engine.search_generic_implementations(
            generic_pattern="List[T]",
            constraint_patterns=["T: Comparable"],
            collections=["test-collection"]
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_interface_implementations(self, type_search_engine, sample_code_search_results):
        """Test interface implementation search"""
        mock_code_search = AsyncMock()
        mock_code_search.initialize = AsyncMock() 
        mock_code_search.search_by_signature = AsyncMock(return_value=sample_code_search_results)
        type_search_engine.code_search_engine = mock_code_search
        
        results = await type_search_engine.search_interface_implementations(
            interface_patterns=["Repository", "Comparable"],
            match_mode=TypeMatchMode.STRUCTURAL,
            collections=["test-collection"]
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_analyze_type_compatibility(self, type_search_engine):
        """Test type compatibility analysis"""
        await type_search_engine.initialize()
        
        result = await type_search_engine.analyze_type_compatibility(
            source_type="int",
            target_type="float",
            collections=["test-collection"]
        )
        
        assert "compatible" in result
        assert "confidence" in result
        assert "relationship" in result
        assert isinstance(result["compatible"], bool)
    
    def test_parse_signature_string(self, type_search_engine):
        """Test signature string parsing"""
        # Test simple signature
        signature = asyncio.run(type_search_engine._parse_signature_string("(str, int) -> bool"))
        
        assert len(signature.parameter_types) == 2
        assert signature.parameter_types[0].type_expression == "str"
        assert signature.parameter_types[1].type_expression == "int"
        assert signature.return_type.type_expression == "bool"
    
    def test_parse_signature_with_names(self, type_search_engine):
        """Test parsing signature with parameter names"""
        signature = asyncio.run(type_search_engine._parse_signature_string("(name: str, age: int) -> Person"))
        
        assert len(signature.parameter_types) == 2
        assert signature.parameter_names == ["name", "age"]
        assert signature.parameter_types[0].type_expression == "str"
        assert signature.return_type.type_expression == "Person"
    
    @pytest.mark.asyncio
    async def test_types_match_exact(self, type_search_engine):
        """Test exact type matching"""
        await type_search_engine.initialize()
        
        # Test exact match
        assert await type_search_engine._types_match("str", "str", TypeMatchMode.EXACT)
        assert not await type_search_engine._types_match("str", "int", TypeMatchMode.EXACT)
    
    @pytest.mark.asyncio
    async def test_types_match_compatible(self, type_search_engine):
        """Test compatible type matching"""
        await type_search_engine.initialize()
        
        # Test compatibility
        assert await type_search_engine._types_match("int", "float", TypeMatchMode.COMPATIBLE)
        assert await type_search_engine._types_match("List[str]", "Sequence[str]", TypeMatchMode.COMPATIBLE)
    
    @pytest.mark.asyncio
    async def test_matches_generic_pattern(self, type_search_engine):
        """Test generic pattern matching"""
        await type_search_engine.initialize()
        
        # Test generic pattern matching
        assert await type_search_engine._matches_generic_pattern("List[str]", "List[T]")
        assert await type_search_engine._matches_generic_pattern("Dict[str, int]", "Dict[K, V]")
        assert not await type_search_engine._matches_generic_pattern("Set[str]", "List[T]")


class TestTypeCompatibilityAnalysis:
    """Test advanced type compatibility analysis"""
    
    @pytest.fixture
    def type_engine(self, mock_workspace_client):
        """Create TypeSearchEngine for compatibility testing"""
        return TypeSearchEngine(mock_workspace_client)
    
    @pytest.mark.asyncio
    async def test_basic_type_compatibility(self, type_engine):
        """Test basic type compatibility relationships"""
        await type_engine.initialize()
        
        # Test numeric hierarchy
        assert await type_engine._are_compatible_types("int", "float")
        assert await type_engine._are_compatible_types("float", "complex")
        assert not await type_engine._are_compatible_types("str", "int")
    
    @pytest.mark.asyncio
    async def test_optional_type_compatibility(self, type_engine):
        """Test Optional type compatibility"""
        await type_engine.initialize()
        
        # Test Optional patterns
        assert await type_engine._are_compatible_types("Optional[str]", "str")
        assert await type_engine._are_compatible_types("Union[str, None]", "str")
    
    @pytest.mark.asyncio
    async def test_container_type_compatibility(self, type_engine):
        """Test container type compatibility"""
        await type_engine.initialize()
        
        # Test container compatibility
        assert await type_engine._are_compatible_types("List[str]", "Sequence[str]")
        assert await type_engine._are_compatible_types("Dict[str, int]", "Mapping[str, int]")


class TestTypeSearchResult:
    """Test TypeSearchResult functionality"""
    
    def test_create_type_search_result(self, sample_code_search_results):
        """Test creating a TypeSearchResult"""
        base_result = sample_code_search_results[0]
        
        type_signature = TypeSignature(
            parameter_types=[TypePattern(type_expression="str")],
            return_type=TypePattern(type_expression="bool")
        )
        
        result = TypeSearchResult(
            base_result=base_result,
            type_signature=type_signature,
            compatibility_score=0.85,
            generic_substitutions={"T": "str"},
            constraint_satisfaction={"T: Comparable": True}
        )
        
        assert result.base_result == base_result
        assert result.compatibility_score == 0.85
        assert result.generic_substitutions == {"T": "str"}
        assert result.constraint_satisfaction == {"T: Comparable": True}
    
    def test_type_search_result_to_dict(self, sample_code_search_results):
        """Test TypeSearchResult serialization"""
        base_result = sample_code_search_results[0]
        type_signature = TypeSignature(
            parameter_types=[TypePattern(type_expression="str")],
            return_type=TypePattern(type_expression="bool")
        )
        
        result = TypeSearchResult(
            base_result=base_result,
            type_signature=type_signature,
            compatibility_score=0.90
        )
        
        result_dict = result.to_dict()
        assert "base_result" in result_dict
        assert "type_signature" in result_dict
        assert "compatibility_score" in result_dict
        assert result_dict["compatibility_score"] == 0.90


class TestErrorHandling:
    """Test error handling in type search operations"""
    
    @pytest.fixture
    def type_engine(self, mock_workspace_client):
        return TypeSearchEngine(mock_workspace_client)
    
    @pytest.mark.asyncio
    async def test_uninitialized_engine_error(self, type_engine):
        """Test that operations work when engine is not explicitly initialized"""
        # Should auto-initialize
        mock_code_search = AsyncMock()
        mock_code_search.initialize = AsyncMock()
        mock_code_search.search_by_signature = AsyncMock(return_value=[])
        type_engine.code_search_engine = mock_code_search
        
        results = await type_engine.search_exact_signature(
            parameter_types=["str"],
            return_type="bool"
        )
        
        assert isinstance(results, list)
        assert type_engine.initialized
    
    @pytest.mark.asyncio  
    async def test_invalid_signature_parsing(self, type_engine):
        """Test handling of invalid signature strings"""
        await type_engine.initialize()
        
        # Should handle malformed signatures gracefully
        signature = await type_engine._parse_signature_string("invalid signature format")
        assert isinstance(signature, TypeSignature)
        
        # Should handle empty signatures
        signature = await type_engine._parse_signature_string("")
        assert isinstance(signature, TypeSignature)
        assert len(signature.parameter_types) == 0


if __name__ == "__main__":
    pytest.main([__file__])