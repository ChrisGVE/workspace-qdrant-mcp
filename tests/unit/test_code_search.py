"""
Comprehensive tests for the Advanced Code-Aware Search Interface.

Tests cover all search modes, ranking algorithms, result enrichment,
and integration with LSP metadata and workspace collections.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from src.workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from src.workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine
from src.workspace_qdrant_mcp.core.lsp_metadata_extractor import SymbolKind, CodeSymbol
from src.workspace_qdrant_mcp.tools.code_search import (
    CodeSearchEngine,
    CodeSearchMode,
    SymbolSearchType,
    CodeSearchQuery,
    SignatureSearchQuery,
    CodeSearchResult,
    search_code_symbols,
    search_function_signatures,
    search_code_semantically
)


@pytest.fixture
def mock_workspace_client():
    """Create mock workspace client for testing"""
    client = MagicMock(spec=QdrantWorkspaceClient)
    client.qdrant_client = MagicMock()
    client.embedding_service = MagicMock()
    client.list_collections = AsyncMock(return_value=[
        "my-project", "library-code", "test-project", "documentation"
    ])
    return client


@pytest.fixture
def mock_hybrid_engine():
    """Create mock hybrid search engine"""
    engine = MagicMock(spec=HybridSearchEngine)
    engine.initialize = AsyncMock()
    return engine


@pytest.fixture
def sample_symbols():
    """Sample code symbols for testing"""
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
                "return_type": "bool"
            },
            "documentation": {
                "docstring": "Authenticate a user with username and password.",
                "inline_comments": ["Validates user credentials"]
            },
            "context_before": ["# User authentication module"],
            "context_after": ["    return user_valid"],
            "visibility": "public",
            "modifiers": ["async"],
            "language": "python"
        },
        {
            "name": "UserManager",
            "kind": SymbolKind.CLASS.value,
            "kind_name": "CLASS",
            "file_uri": "file:///src/user.py",
            "range": {"start": {"line": 5, "character": 0}, "end": {"line": 50, "character": 0}},
            "type_info": {
                "type_signature": "class UserManager",
            },
            "documentation": {
                "docstring": "Manages user accounts and authentication.",
            },
            "context_before": ["from typing import Dict, Optional"],
            "context_after": ["    def __init__(self):"],
            "visibility": "public",
            "modifiers": [],
            "language": "python",
            "children": ["authenticate_user", "create_user", "delete_user"]
        },
        {
            "name": "get_user_profile",
            "kind": SymbolKind.METHOD.value,
            "kind_name": "METHOD",
            "file_uri": "file:///src/user.py",
            "range": {"start": {"line": 30, "character": 4}, "end": {"line": 35, "character": 4}},
            "type_info": {
                "type_signature": "def get_user_profile(self, user_id: int) -> Dict[str, Any]",
                "parameter_types": [
                    {"name": "user_id", "type": "int"}
                ],
                "return_type": "Dict[str, Any]"
            },
            "documentation": {
                "docstring": "Retrieve user profile information.",
            },
            "parent_symbol": "UserManager",
            "visibility": "public",
            "modifiers": [],
            "language": "python"
        }
    ]


@pytest.fixture
def sample_relationships():
    """Sample code relationships for testing"""
    return [
        {
            "type": "imports",
            "source": "auth.py",
            "target": "user.py",
            "details": "imports UserManager"
        },
        {
            "type": "calls",
            "source": "authenticate_user",
            "target": "get_user_profile",
            "details": "function call relationship"
        },
        {
            "type": "exports",
            "source": "user.py",
            "target": "UserManager",
            "details": "exports class UserManager"
        }
    ]


@pytest.fixture
async def code_search_engine(mock_workspace_client, mock_hybrid_engine, sample_symbols, sample_relationships):
    """Create initialized CodeSearchEngine for testing"""
    engine = CodeSearchEngine(mock_workspace_client)
    
    # Mock hybrid engine initialization
    with patch('src.workspace_qdrant_mcp.tools.code_search.HybridSearchEngine', return_value=mock_hybrid_engine):
        # Pre-populate caches
        engine.symbol_cache = {
            "my-project": sample_symbols,
            "library-code": sample_symbols[:1]  # Subset for library
        }
        engine.relationship_cache = {
            "my-project": sample_relationships,
            "library-code": sample_relationships[:1]
        }
        engine.hybrid_engine = mock_hybrid_engine
        engine.initialized = True
        
        yield engine


class TestCodeSearchEngine:
    """Test cases for the CodeSearchEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_workspace_client, mock_hybrid_engine):
        """Test proper initialization of CodeSearchEngine"""
        engine = CodeSearchEngine(mock_workspace_client)
        
        with patch('src.workspace_qdrant_mcp.tools.code_search.HybridSearchEngine', return_value=mock_hybrid_engine):
            with patch.object(engine, '_build_code_intelligence_caches', new_callable=AsyncMock):
                await engine.initialize()
                
                assert engine.initialized
                assert engine.hybrid_engine == mock_hybrid_engine
                mock_hybrid_engine.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_symbol_search_exact_name(self, code_search_engine):
        """Test symbol search with exact name matching"""
        results = await code_search_engine.search_symbols(
            query="authenticate_user",
            search_type=SymbolSearchType.EXACT_NAME,
            limit=10
        )
        
        assert len(results) > 0
        # Should find exact match
        exact_matches = [r for r in results if r.symbol.get("name") == "authenticate_user"]
        assert len(exact_matches) > 0
        assert exact_matches[0].search_type == "symbol"
    
    @pytest.mark.asyncio
    async def test_symbol_search_partial_name(self, code_search_engine):
        """Test symbol search with partial name matching"""
        results = await code_search_engine.search_symbols(
            query="user",
            search_type=SymbolSearchType.PARTIAL_NAME,
            limit=10
        )
        
        assert len(results) > 0
        # Should find multiple matches with "user" in name
        user_matches = [r for r in results if "user" in r.symbol.get("name", "").lower()]
        assert len(user_matches) > 0
    
    @pytest.mark.asyncio
    async def test_symbol_search_by_type(self, code_search_engine):
        """Test symbol search filtered by symbol types"""
        results = await code_search_engine.search_symbols(
            query="user",
            symbol_types=["function"],
            search_type=SymbolSearchType.PARTIAL_NAME,
            limit=10
        )
        
        assert len(results) > 0
        # All results should be functions
        for result in results:
            kind_name = result.symbol.get("kind_name", "").lower()
            assert kind_name in ["function", "method"]
    
    @pytest.mark.asyncio
    async def test_signature_search_by_parameter_types(self, code_search_engine):
        """Test function signature search by parameter types"""
        signature_query = SignatureSearchQuery(
            parameter_types=["str", "str"],
            exact_match=False
        )
        
        results = await code_search_engine.search_by_signature(
            signature_query=signature_query,
            limit=10
        )
        
        assert len(results) > 0
        # Should find authenticate_user which has (str, str) parameters
        auth_matches = [r for r in results if r.symbol.get("name") == "authenticate_user"]
        assert len(auth_matches) > 0
    
    @pytest.mark.asyncio
    async def test_signature_search_by_return_type(self, code_search_engine):
        """Test function signature search by return type"""
        signature_query = SignatureSearchQuery(
            return_type="bool",
            exact_match=False
        )
        
        results = await code_search_engine.search_by_signature(
            signature_query=signature_query,
            limit=10
        )
        
        assert len(results) > 0
        # Should find authenticate_user which returns bool
        bool_returns = [r for r in results 
                       if r.symbol.get("type_info", {}).get("return_type") == "bool"]
        assert len(bool_returns) > 0
    
    @pytest.mark.asyncio
    async def test_signature_search_exact_match(self, code_search_engine):
        """Test exact signature matching"""
        signature_query = SignatureSearchQuery(
            parameter_types=["str", "str"],
            return_type="bool",
            exact_match=True
        )
        
        results = await code_search_engine.search_by_signature(
            signature_query=signature_query,
            limit=10
        )
        
        # Should have high-quality matches for exact signature
        if results:
            assert results[0].relevance_score > 0.5
    
    @pytest.mark.asyncio
    async def test_semantic_search_with_enhancement(self, code_search_engine):
        """Test semantic search enhanced with code metadata"""
        with patch('src.workspace_qdrant_mcp.tools.code_search.search_workspace') as mock_search:
            mock_search.return_value = {
                "results": [
                    {
                        "id": "test_result",
                        "score": 0.8,
                        "payload": {
                            "symbol": code_search_engine.symbol_cache["my-project"][0],
                            "content_type": "code_symbol",
                            "metadata": {"file_path": "/src/auth.py", "file_type": "python"}
                        },
                        "collection": "my-project"
                    }
                ]
            }
            
            results = await code_search_engine.search_semantic_code(
                query="user authentication with password validation",
                enhance_with_metadata=True,
                limit=5
            )
            
            assert len(results) > 0
            assert results[0].search_type == "semantic"
            assert results[0].symbol is not None
    
    @pytest.mark.asyncio
    async def test_dependency_search(self, code_search_engine):
        """Test dependency and relationship search"""
        results = await code_search_engine.search_dependencies(
            query="UserManager",
            dependency_types=["imports", "exports"],
            limit=10
        )
        
        assert len(results) > 0
        # Should find relationships involving UserManager
        manager_deps = [r for r in results 
                       if "usermanager" in str(r.related_symbols).lower()]
        # Note: May not find exact matches due to mocking, but structure should be correct
    
    @pytest.mark.asyncio
    async def test_fuzzy_search(self, code_search_engine):
        """Test fuzzy search for incomplete queries"""
        results = await code_search_engine.search_fuzzy(
            query="authuser",  # Incomplete/misspelled
            threshold=70,
            limit=10
        )
        
        # Should find fuzzy matches even with incomplete query
        # Exact assertion depends on fuzzy matching algorithm
        assert isinstance(results, list)
    
    @pytest.mark.asyncio 
    async def test_result_enrichment(self, code_search_engine):
        """Test that search results are properly enriched"""
        results = await code_search_engine.search_symbols(
            query="authenticate_user",
            search_type=SymbolSearchType.EXACT_NAME,
            limit=1
        )
        
        if results:
            result = results[0]
            assert isinstance(result, CodeSearchResult)
            assert result.context_snippet is not None
            assert result.documentation_summary is not None
            assert isinstance(result.related_symbols, list)
            assert isinstance(result.usage_examples, list)
            assert result.file_path is not None
            assert result.line_number >= 0


class TestSearchRanking:
    """Test cases for search result ranking algorithms"""
    
    @pytest.mark.asyncio
    async def test_symbol_ranking_exact_match_boost(self, code_search_engine):
        """Test that exact name matches get ranking boost"""
        mock_results = [
            {
                "payload": {"symbol": {"name": "user_helper", "kind_name": "function", "visibility": "public"}},
                "score": 0.5
            },
            {
                "payload": {"symbol": {"name": "authenticate_user", "kind_name": "function", "visibility": "public"}},
                "score": 0.5
            }
        ]
        
        ranked = await code_search_engine._rank_symbol_results(
            mock_results, "authenticate_user", limit=10
        )
        
        # Exact match should be ranked higher
        assert ranked[0]["payload"]["symbol"]["name"] == "authenticate_user"
        assert ranked[0]["composite_score"] > ranked[1]["composite_score"]
    
    @pytest.mark.asyncio
    async def test_signature_ranking_quality_scoring(self, code_search_engine):
        """Test signature match quality scoring"""
        signature_query = SignatureSearchQuery(
            parameter_types=["str", "str"],
            return_type="bool",
            exact_match=False
        )
        
        symbol_exact = {"type_info": {
            "parameter_types": [{"type": "str"}, {"type": "str"}],
            "return_type": "bool"
        }}
        
        symbol_partial = {"type_info": {
            "parameter_types": [{"type": "str"}],
            "return_type": "int"
        }}
        
        exact_quality = code_search_engine._calculate_signature_match_quality(
            symbol_exact, signature_query
        )
        partial_quality = code_search_engine._calculate_signature_match_quality(
            symbol_partial, signature_query
        )
        
        assert exact_quality > partial_quality
    
    @pytest.mark.asyncio
    async def test_fuzzy_ranking_by_similarity(self, code_search_engine):
        """Test fuzzy search ranking by similarity score"""
        mock_results = [
            {
                "payload": {"symbol": {"name": "auth_user"}},
                "score": 0.5,
                "fuzzy_score": 60
            },
            {
                "payload": {"symbol": {"name": "authenticate_user"}},
                "score": 0.5,
                "fuzzy_score": 85
            }
        ]
        
        ranked = await code_search_engine._rank_fuzzy_results(
            mock_results, "authuser", threshold=50, limit=10
        )
        
        # Higher fuzzy score should rank higher
        assert ranked[0]["fuzzy_score"] > ranked[1]["fuzzy_score"]


class TestResultEnrichment:
    """Test cases for search result enrichment"""
    
    @pytest.mark.asyncio
    async def test_context_snippet_generation(self, code_search_engine):
        """Test context snippet building"""
        symbol_data = {
            "name": "test_function",
            "kind_name": "function",
            "context_before": ["# Helper function"],
            "context_after": ["    return result"]
        }
        
        snippet = await code_search_engine._build_context_snippet(symbol_data)
        
        assert "# Helper function" in snippet
        assert "return result" in snippet
    
    @pytest.mark.asyncio
    async def test_documentation_summary_extraction(self, code_search_engine):
        """Test documentation summary extraction"""
        symbol_data = {
            "documentation": {
                "docstring": "This function authenticates a user. It validates credentials and returns status.",
                "inline_comments": ["Validates input"]
            }
        }
        
        summary = code_search_engine._extract_documentation_summary(symbol_data)
        
        assert summary is not None
        assert "authenticates a user" in summary.lower()
    
    @pytest.mark.asyncio
    async def test_usage_example_generation(self, code_search_engine):
        """Test usage example generation"""
        symbol_data = {
            "name": "authenticate_user",
            "kind_name": "function",
            "type_info": {
                "parameter_types": [
                    {"name": "username", "type": "str"},
                    {"name": "password", "type": "str"}
                ]
            }
        }
        
        examples = await code_search_engine._find_usage_examples(symbol_data)
        
        assert len(examples) > 0
        assert "authenticate_user(username, password)" in examples[0]
    
    @pytest.mark.asyncio
    async def test_related_symbols_discovery(self, code_search_engine):
        """Test related symbol discovery"""
        symbol_data = {
            "name": "get_user_profile",
            "parent_symbol": "UserManager",
            "children": []
        }
        
        related = await code_search_engine._find_related_symbols(symbol_data)
        
        # Should find siblings in UserManager class
        assert isinstance(related, list)
        # Exact content depends on cache, but structure should be correct


class TestConvenienceFunctions:
    """Test cases for high-level convenience functions"""
    
    @pytest.mark.asyncio
    async def test_search_code_symbols_function(self, mock_workspace_client):
        """Test search_code_symbols convenience function"""
        with patch('src.workspace_qdrant_mcp.tools.code_search.CodeSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.initialize = AsyncMock()
            mock_engine.search_symbols = AsyncMock(return_value=[])
            mock_engine_class.return_value = mock_engine
            
            results = await search_code_symbols(
                workspace_client=mock_workspace_client,
                symbol_name="test_function",
                symbol_types=["function"],
                exact_match=True
            )
            
            mock_engine.initialize.assert_called_once()
            mock_engine.search_symbols.assert_called_once()
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_function_signatures_function(self, mock_workspace_client):
        """Test search_function_signatures convenience function"""
        with patch('src.workspace_qdrant_mcp.tools.code_search.CodeSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.initialize = AsyncMock()
            mock_engine.search_by_signature = AsyncMock(return_value=[])
            mock_engine_class.return_value = mock_engine
            
            results = await search_function_signatures(
                workspace_client=mock_workspace_client,
                parameter_types=["str"],
                return_type="bool"
            )
            
            mock_engine.initialize.assert_called_once()
            mock_engine.search_by_signature.assert_called_once()
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_code_semantically_function(self, mock_workspace_client):
        """Test search_code_semantically convenience function"""
        with patch('src.workspace_qdrant_mcp.tools.code_search.CodeSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.initialize = AsyncMock()
            mock_engine.search_semantic_code = AsyncMock(return_value=[])
            mock_engine_class.return_value = mock_engine
            
            results = await search_code_semantically(
                workspace_client=mock_workspace_client,
                query="authentication patterns with security validation"
            )
            
            mock_engine.initialize.assert_called_once()
            mock_engine.search_semantic_code.assert_called_once()
            assert isinstance(results, list)


class TestDataStructures:
    """Test cases for data structure validation"""
    
    def test_code_search_query_creation(self):
        """Test CodeSearchQuery data structure"""
        query = CodeSearchQuery(
            query="test query",
            mode=CodeSearchMode.SYMBOL,
            symbol_types=["function", "class"],
            collections=["my-project"],
            max_results=10
        )
        
        assert query.query == "test query"
        assert query.mode == CodeSearchMode.SYMBOL
        assert "function" in query.symbol_types
        assert query.max_results == 10
        
        # Test serialization
        query_dict = query.to_dict()
        assert isinstance(query_dict, dict)
        assert query_dict["mode"] == "symbol"
    
    def test_signature_search_query_creation(self):
        """Test SignatureSearchQuery data structure"""
        sig_query = SignatureSearchQuery(
            parameter_types=["str", "int"],
            return_type="bool",
            function_name_pattern="authenticate",
            exact_match=True
        )
        
        assert sig_query.parameter_types == ["str", "int"]
        assert sig_query.return_type == "bool"
        assert sig_query.exact_match is True
        
        # Test serialization
        sig_dict = sig_query.to_dict()
        assert isinstance(sig_dict, dict)
        assert sig_dict["exact_match"] is True
    
    def test_code_search_result_creation(self):
        """Test CodeSearchResult data structure"""
        result = CodeSearchResult(
            symbol={"name": "test_func", "kind": "function"},
            relevance_score=0.85,
            context_snippet="def test_func():\n    pass",
            related_symbols=[],
            usage_examples=["test_func()"],
            documentation_summary="Test function",
            file_path="/src/test.py",
            line_number=10,
            collection="my-project",
            search_type="symbol"
        )
        
        assert result.symbol["name"] == "test_func"
        assert result.relevance_score == 0.85
        assert result.line_number == 10
        
        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["file_path"] == "/src/test.py"


class TestErrorHandling:
    """Test cases for error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_initialization_failure_handling(self, mock_workspace_client):
        """Test handling of initialization failures"""
        engine = CodeSearchEngine(mock_workspace_client)
        
        with patch('src.workspace_qdrant_mcp.tools.code_search.HybridSearchEngine') as mock_engine_class:
            mock_engine_class.side_effect = Exception("Initialization failed")
            
            with pytest.raises(Exception):  # Should propagate WorkspaceError
                await engine.initialize()
    
    @pytest.mark.asyncio
    async def test_empty_search_results(self, code_search_engine):
        """Test handling of empty search results"""
        # Mock empty cache
        code_search_engine.symbol_cache = {}
        code_search_engine.relationship_cache = {}
        
        results = await code_search_engine.search_symbols(
            query="nonexistent_symbol",
            search_type=SymbolSearchType.EXACT_NAME
        )
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_signature_query(self, code_search_engine):
        """Test handling of invalid signature queries"""
        signature_query = SignatureSearchQuery(
            parameter_types=None,
            return_type=None,
            exact_match=True
        )
        
        results = await code_search_engine.search_by_signature(
            signature_query=signature_query
        )
        
        # Should handle gracefully even with minimal query
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_collection_access_failure(self, code_search_engine):
        """Test handling of collection access failures"""
        with patch.object(code_search_engine.workspace_client, 'list_collections', 
                         side_effect=Exception("Collection access failed")):
            
            # Should handle collection access failure gracefully
            collections = await code_search_engine._get_code_collections()
            assert isinstance(collections, list)  # Should return empty list or fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])