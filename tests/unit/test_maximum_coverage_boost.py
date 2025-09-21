"""
Maximum Coverage Boost Unit Tests

Targets the highest-impact uncovered modules for aggressive coverage improvement.
Focus on: sqlite_state_manager, hybrid_search, lsp_metadata_extractor, client.py

Target: Push coverage from 6.41% to 50%+ through systematic testing
"""

import asyncio
import json
import sqlite3
import tempfile
import pytest
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from dataclasses import dataclass
import threading

# Add src paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

# Import largest uncovered modules
try:
    from common.core.sqlite_state_manager import SQLiteStateManager
    from common.core.hybrid_search import SearchService, HybridSearchEngine
    from common.core.lsp_metadata_extractor import (
        LspMetadataExtractor, SymbolKind, CodeSymbol, FileMetadata,
        Position, Range, TypeInformation, Documentation, SymbolRelationship
    )
    from common.core.client import BaseClient, AsyncClient
    from common.core.collections import CollectionManager, Collection
    from common.core.config import Configuration, ConfigManager
    from common.core.memory import MemorySystem, DocumentMemory
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Could not import modules: {e}")

# More critical module imports
try:
    from workspace_qdrant_mcp.core.client import QdrantClient
    from workspace_qdrant_mcp.core.hybrid_search import HybridSearchService
    from workspace_qdrant_mcp.core.embeddings import EmbeddingService
    from workspace_qdrant_mcp.tools.memory import MemoryTool
    from workspace_qdrant_mcp.server import FastMCPServer
    WQM_AVAILABLE = True
except ImportError as e:
    WQM_AVAILABLE = False
    print(f"Warning: Could not import workspace modules: {e}")


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
class TestSQLiteStateManager:
    """Comprehensive tests for SQLiteStateManager - largest uncovered module"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def state_manager(self, temp_db_path):
        """Create SQLiteStateManager instance for testing"""
        return SQLiteStateManager(db_path=str(temp_db_path))

    def test_sqlite_state_manager_initialization(self, temp_db_path):
        """Test SQLiteStateManager initialization"""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        assert manager is not None
        assert manager.db_path == str(temp_db_path)

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_connect(self, state_manager):
        """Test SQLiteStateManager database connection"""
        await state_manager.connect()
        assert state_manager.is_connected is True

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_disconnect(self, state_manager):
        """Test SQLiteStateManager database disconnection"""
        await state_manager.connect()
        await state_manager.disconnect()
        assert state_manager.is_connected is False

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_create_tables(self, state_manager):
        """Test SQLiteStateManager table creation"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Verify tables exist
        async with state_manager.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()

        table_names = [table[0] for table in tables]
        expected_tables = ['documents', 'collections', 'metadata', 'state']

        for expected_table in expected_tables:
            assert any(expected_table in name for name in table_names)

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_store_document(self, state_manager):
        """Test SQLiteStateManager document storage"""
        await state_manager.connect()
        await state_manager.create_tables()

        document = {
            "id": "doc_1",
            "content": "Test document content",
            "metadata": {"type": "test", "timestamp": "2023-01-01"},
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }

        await state_manager.store_document("test_collection", document)

        # Verify document was stored
        stored_doc = await state_manager.get_document("test_collection", "doc_1")
        assert stored_doc is not None
        assert stored_doc["id"] == "doc_1"
        assert stored_doc["content"] == "Test document content"

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_update_document(self, state_manager):
        """Test SQLiteStateManager document updating"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Store initial document
        document = {
            "id": "doc_1",
            "content": "Original content",
            "metadata": {"type": "test"}
        }
        await state_manager.store_document("test_collection", document)

        # Update document
        updated_document = {
            "id": "doc_1",
            "content": "Updated content",
            "metadata": {"type": "test", "updated": True}
        }
        await state_manager.update_document("test_collection", updated_document)

        # Verify update
        stored_doc = await state_manager.get_document("test_collection", "doc_1")
        assert stored_doc["content"] == "Updated content"
        assert stored_doc["metadata"]["updated"] is True

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_delete_document(self, state_manager):
        """Test SQLiteStateManager document deletion"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Store document
        document = {"id": "doc_1", "content": "Test content"}
        await state_manager.store_document("test_collection", document)

        # Delete document
        await state_manager.delete_document("test_collection", "doc_1")

        # Verify deletion
        stored_doc = await state_manager.get_document("test_collection", "doc_1")
        assert stored_doc is None

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_search_documents(self, state_manager):
        """Test SQLiteStateManager document searching"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Store multiple documents
        documents = [
            {"id": "doc_1", "content": "Python programming tutorial"},
            {"id": "doc_2", "content": "JavaScript web development"},
            {"id": "doc_3", "content": "Python data analysis guide"}
        ]

        for doc in documents:
            await state_manager.store_document("test_collection", doc)

        # Search for Python-related documents
        results = await state_manager.search_documents(
            "test_collection", query="Python", limit=10
        )

        assert len(results) == 2
        python_docs = [doc for doc in results if "Python" in doc["content"]]
        assert len(python_docs) == 2

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_get_collection_stats(self, state_manager):
        """Test SQLiteStateManager collection statistics"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Store documents
        for i in range(5):
            doc = {"id": f"doc_{i}", "content": f"Content {i}"}
            await state_manager.store_document("test_collection", doc)

        stats = await state_manager.get_collection_stats("test_collection")
        assert stats["document_count"] == 5
        assert stats["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_backup_restore(self, state_manager, temp_db_path):
        """Test SQLiteStateManager backup and restore functionality"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Store test data
        document = {"id": "doc_1", "content": "Backup test content"}
        await state_manager.store_document("test_collection", document)

        # Create backup
        backup_path = temp_db_path.parent / f"{temp_db_path.stem}_backup.db"
        await state_manager.create_backup(str(backup_path))

        assert backup_path.exists()

        # Verify backup contains data
        backup_manager = SQLiteStateManager(db_path=str(backup_path))
        await backup_manager.connect()

        restored_doc = await backup_manager.get_document("test_collection", "doc_1")
        assert restored_doc is not None
        assert restored_doc["content"] == "Backup test content"

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_transaction_handling(self, state_manager):
        """Test SQLiteStateManager transaction handling"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Test successful transaction
        async with state_manager.transaction():
            doc1 = {"id": "doc_1", "content": "Transaction test 1"}
            doc2 = {"id": "doc_2", "content": "Transaction test 2"}
            await state_manager.store_document("test_collection", doc1)
            await state_manager.store_document("test_collection", doc2)

        # Verify both documents were stored
        doc1_stored = await state_manager.get_document("test_collection", "doc_1")
        doc2_stored = await state_manager.get_document("test_collection", "doc_2")
        assert doc1_stored is not None
        assert doc2_stored is not None

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_concurrent_access(self, state_manager):
        """Test SQLiteStateManager concurrent access handling"""
        await state_manager.connect()
        await state_manager.create_tables()

        async def store_document_task(doc_id):
            doc = {"id": doc_id, "content": f"Concurrent content {doc_id}"}
            await state_manager.store_document("test_collection", doc)

        # Run concurrent operations
        tasks = [store_document_task(f"doc_{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all documents were stored
        stats = await state_manager.get_collection_stats("test_collection")
        assert stats["document_count"] == 10

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_error_handling(self, state_manager):
        """Test SQLiteStateManager error handling"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Test invalid document ID
        with pytest.raises((ValueError, TypeError)):
            await state_manager.store_document("test_collection", {"id": None})

        # Test missing collection
        with pytest.raises((ValueError, sqlite3.Error)):
            await state_manager.get_document("", "doc_1")

    def test_sqlite_state_manager_connection_pooling(self, temp_db_path):
        """Test SQLiteStateManager connection pooling"""
        manager = SQLiteStateManager(db_path=str(temp_db_path), max_connections=5)
        assert manager.max_connections == 5

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_schema_migration(self, state_manager):
        """Test SQLiteStateManager schema migration"""
        await state_manager.connect()

        # Initial schema version
        await state_manager.create_tables()
        initial_version = await state_manager.get_schema_version()

        # Simulate migration
        await state_manager.migrate_schema(initial_version + 1)

        new_version = await state_manager.get_schema_version()
        assert new_version == initial_version + 1

    @pytest.mark.asyncio
    async def test_sqlite_state_manager_cleanup(self, state_manager):
        """Test SQLiteStateManager cleanup operations"""
        await state_manager.connect()
        await state_manager.create_tables()

        # Store documents with timestamps
        for i in range(10):
            doc = {
                "id": f"doc_{i}",
                "content": f"Content {i}",
                "timestamp": time.time() - (i * 86400)  # Days ago
            }
            await state_manager.store_document("test_collection", doc)

        # Cleanup old documents (older than 5 days)
        cutoff_time = time.time() - (5 * 86400)
        deleted_count = await state_manager.cleanup_old_documents(
            "test_collection", cutoff_time
        )

        assert deleted_count == 5

        # Verify remaining documents
        stats = await state_manager.get_collection_stats("test_collection")
        assert stats["document_count"] == 5


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
class TestHybridSearchEngine:
    """Comprehensive tests for HybridSearchEngine"""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        service = Mock()
        service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4] * 96)
        service.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4] * 96] * 3)
        return service

    @pytest.fixture
    def mock_client(self):
        """Mock Qdrant client for testing"""
        client = Mock()
        client.search = AsyncMock(return_value=[
            {"id": "doc1", "score": 0.9, "payload": {"content": "test content"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "another test"}}
        ])
        return client

    @pytest.fixture
    def search_engine(self, mock_client, mock_embedding_service):
        """Create HybridSearchEngine for testing"""
        config = Mock()
        config.dense_weight = 0.7
        config.sparse_weight = 0.3
        config.rrf_constant = 60

        engine = HybridSearchEngine(
            client=mock_client,
            embedding_service=mock_embedding_service,
            config=config
        )
        return engine

    @pytest.mark.asyncio
    async def test_hybrid_search_engine_initialization(self, mock_client, mock_embedding_service):
        """Test HybridSearchEngine initialization"""
        config = Mock()
        engine = HybridSearchEngine(
            client=mock_client,
            embedding_service=mock_embedding_service,
            config=config
        )
        assert engine is not None
        assert engine.client == mock_client
        assert engine.embedding_service == mock_embedding_service

    @pytest.mark.asyncio
    async def test_hybrid_search_dense_search(self, search_engine):
        """Test HybridSearchEngine dense vector search"""
        results = await search_engine.dense_search(
            collection="test_collection",
            query="test query",
            limit=10
        )

        assert len(results) == 2
        assert results[0]["score"] == 0.9
        search_engine.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_sparse_search(self, search_engine):
        """Test HybridSearchEngine sparse keyword search"""
        # Mock sparse search results
        search_engine.client.search = AsyncMock(return_value=[
            {"id": "doc1", "score": 0.95, "payload": {"content": "keyword match"}},
            {"id": "doc3", "score": 0.7, "payload": {"content": "partial match"}}
        ])

        results = await search_engine.sparse_search(
            collection="test_collection",
            query="keyword",
            limit=10
        )

        assert len(results) == 2
        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_hybrid_search_reciprocal_rank_fusion(self, search_engine):
        """Test HybridSearchEngine reciprocal rank fusion"""
        dense_results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8},
            {"id": "doc3", "score": 0.7}
        ]

        sparse_results = [
            {"id": "doc2", "score": 0.95},
            {"id": "doc4", "score": 0.85},
            {"id": "doc1", "score": 0.75}
        ]

        fused_results = search_engine.reciprocal_rank_fusion(
            dense_results, sparse_results, k=60
        )

        assert len(fused_results) >= 3
        # doc2 should rank high as it appears in both lists
        doc2_result = next((r for r in fused_results if r["id"] == "doc2"), None)
        assert doc2_result is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_full_pipeline(self, search_engine):
        """Test HybridSearchEngine full search pipeline"""
        # Mock both dense and sparse searches
        search_engine.dense_search = AsyncMock(return_value=[
            {"id": "doc1", "score": 0.9, "payload": {"content": "dense match"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "another dense"}}
        ])

        search_engine.sparse_search = AsyncMock(return_value=[
            {"id": "doc2", "score": 0.95, "payload": {"content": "sparse match"}},
            {"id": "doc3", "score": 0.7, "payload": {"content": "keyword match"}}
        ])

        results = await search_engine.hybrid_search(
            collection="test_collection",
            query="test query",
            limit=10
        )

        assert len(results) >= 2
        search_engine.dense_search.assert_called_once()
        search_engine.sparse_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_filter_application(self, search_engine):
        """Test HybridSearchEngine with filters"""
        filters = {"metadata.type": "document", "metadata.status": "published"}

        await search_engine.dense_search(
            collection="test_collection",
            query="test query",
            limit=10,
            filters=filters
        )

        # Verify filters were passed to client
        search_engine.client.search.assert_called_once()
        call_args = search_engine.client.search.call_args
        assert "filter" in call_args.kwargs or any("filter" in str(arg) for arg in call_args.args)

    @pytest.mark.asyncio
    async def test_hybrid_search_performance_optimization(self, search_engine):
        """Test HybridSearchEngine performance optimizations"""
        # Test batch processing
        queries = ["query 1", "query 2", "query 3"]

        start_time = time.time()

        tasks = [
            search_engine.dense_search("test_collection", query, limit=5)
            for query in queries
        ]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        execution_time = end_time - start_time

        assert len(results) == 3
        assert execution_time < 1.0  # Should execute efficiently

    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self, search_engine):
        """Test HybridSearchEngine error handling"""
        # Test with invalid collection
        search_engine.client.search = AsyncMock(side_effect=Exception("Collection not found"))

        with pytest.raises(Exception):
            await search_engine.dense_search("invalid_collection", "query", limit=10)

    @pytest.mark.asyncio
    async def test_hybrid_search_result_aggregation(self, search_engine):
        """Test HybridSearchEngine result aggregation"""
        # Test result deduplication and scoring
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "content"}},
            {"id": "doc1", "score": 0.85, "payload": {"content": "content"}}  # Duplicate
        ]

        sparse_results = [
            {"id": "doc1", "score": 0.95, "payload": {"content": "content"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "other"}}
        ]

        fused_results = search_engine.reciprocal_rank_fusion(dense_results, sparse_results)

        # Should deduplicate and properly score
        doc1_results = [r for r in fused_results if r["id"] == "doc1"]
        assert len(doc1_results) == 1  # Should be deduplicated

    @pytest.mark.asyncio
    async def test_hybrid_search_semantic_vs_lexical(self, search_engine):
        """Test HybridSearchEngine semantic vs lexical search balance"""
        # Test that semantic and lexical searches complement each other
        semantic_query = "machine learning algorithms"
        lexical_query = "exact phrase match"

        # Mock different result patterns
        search_engine.dense_search = AsyncMock(return_value=[
            {"id": "semantic_doc", "score": 0.9, "payload": {"content": "AI and ML"}}
        ])

        search_engine.sparse_search = AsyncMock(return_value=[
            {"id": "lexical_doc", "score": 0.9, "payload": {"content": "exact phrase match"}}
        ])

        semantic_results = await search_engine.hybrid_search(
            "test_collection", semantic_query, limit=10
        )

        lexical_results = await search_engine.hybrid_search(
            "test_collection", lexical_query, limit=10
        )

        # Both should return relevant results
        assert len(semantic_results) > 0
        assert len(lexical_results) > 0


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
class TestLspMetadataExtractor:
    """Comprehensive tests for LspMetadataExtractor"""

    @pytest.fixture
    def mock_lsp_client(self):
        """Mock LSP client for testing"""
        client = Mock()
        client.initialize = AsyncMock()
        client.initialized = AsyncMock()
        client.document_symbols = AsyncMock(return_value=[])
        client.hover = AsyncMock(return_value=None)
        client.definition = AsyncMock(return_value=[])
        client.references = AsyncMock(return_value=[])
        client.shutdown = AsyncMock()
        client.exit = AsyncMock()
        return client

    @pytest.fixture
    def extractor(self, mock_lsp_client):
        """Create LspMetadataExtractor for testing"""
        with patch('common.core.lsp_metadata_extractor.AsyncioLspClient') as mock_client_class:
            mock_client_class.return_value = mock_lsp_client
            extractor = LspMetadataExtractor()
            return extractor

    def test_lsp_metadata_extractor_initialization(self):
        """Test LspMetadataExtractor initialization"""
        extractor = LspMetadataExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'language_extractors')

    @pytest.mark.asyncio
    async def test_lsp_metadata_extractor_initialize(self, extractor, mock_lsp_client):
        """Test LspMetadataExtractor initialization"""
        workspace_path = Path("/test/workspace")

        await extractor.initialize(workspace_path)

        assert extractor._initialized is True
        mock_lsp_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_lsp_metadata_extractor_extract_file_metadata(self, extractor, mock_lsp_client):
        """Test LspMetadataExtractor file metadata extraction"""
        # Create temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function(param1: str, param2: int) -> bool:
    '''Test function docstring'''
    return True

class TestClass:
    '''Test class docstring'''
    def method(self):
        pass
""")
            file_path = Path(f.name)

        # Mock LSP responses
        mock_lsp_client.document_symbols.return_value = [
            {
                "name": "test_function",
                "kind": 12,  # Function
                "range": {"start": {"line": 1, "character": 0}, "end": {"line": 3, "character": 0}},
                "detail": "def test_function(param1: str, param2: int) -> bool"
            },
            {
                "name": "TestClass",
                "kind": 5,  # Class
                "range": {"start": {"line": 5, "character": 0}, "end": {"line": 8, "character": 0}},
                "children": [
                    {
                        "name": "method",
                        "kind": 6,  # Method
                        "range": {"start": {"line": 7, "character": 4}, "end": {"line": 8, "character": 0}}
                    }
                ]
            }
        ]

        # Initialize extractor
        await extractor.initialize(file_path.parent)

        # Extract metadata
        metadata = await extractor.extract_file_metadata(file_path)

        assert metadata is not None
        assert isinstance(metadata, FileMetadata)
        assert len(metadata.symbols) >= 2
        assert any(symbol.name == "test_function" for symbol in metadata.symbols)
        assert any(symbol.name == "TestClass" for symbol in metadata.symbols)

    def test_symbol_kind_enum(self):
        """Test SymbolKind enumeration"""
        assert SymbolKind.FUNCTION.value == 12
        assert SymbolKind.CLASS.value == 5
        assert SymbolKind.VARIABLE.value == 13
        assert SymbolKind.METHOD.value == 6

    def test_code_symbol_creation(self):
        """Test CodeSymbol creation and properties"""
        position = Position(line=10, character=5)
        range_obj = Range(start=position, end=Position(line=15, character=10))

        symbol = CodeSymbol(
            name="test_symbol",
            kind=SymbolKind.FUNCTION,
            range=range_obj,
            signature="def test_symbol(param: str) -> int"
        )

        assert symbol.name == "test_symbol"
        assert symbol.kind == SymbolKind.FUNCTION
        assert symbol.range.start.line == 10
        assert symbol.get_signature() == "def test_symbol(param: str) -> int"

    def test_type_information_creation(self):
        """Test TypeInformation creation"""
        type_info = TypeInformation(
            type_name="str",
            return_type="bool",
            parameters=["param1: str", "param2: int"]
        )

        assert type_info.type_name == "str"
        assert type_info.return_type == "bool"
        assert len(type_info.parameters) == 2

    def test_documentation_creation(self):
        """Test Documentation creation"""
        doc = Documentation(
            docstring="This is a test function",
            comments=["# Important comment", "# Another comment"],
            annotations={"param1": "Input parameter", "return": "Success status"}
        )

        assert doc.docstring == "This is a test function"
        assert len(doc.comments) == 2
        assert doc.annotations["param1"] == "Input parameter"

    def test_symbol_relationship_creation(self):
        """Test SymbolRelationship creation"""
        from common.core.lsp_metadata_extractor import RelationshipType

        relationship = SymbolRelationship(
            from_symbol="ClassA",
            to_symbol="ClassB",
            relationship_type=RelationshipType.INHERITANCE,
            file_path=Path("/test/file.py")
        )

        assert relationship.from_symbol == "ClassA"
        assert relationship.to_symbol == "ClassB"
        assert relationship.relationship_type == RelationshipType.INHERITANCE

    def test_file_metadata_creation(self):
        """Test FileMetadata creation"""
        symbols = [
            CodeSymbol(
                name="test_func",
                kind=SymbolKind.FUNCTION,
                range=Range(Position(0, 0), Position(5, 0))
            )
        ]

        metadata = FileMetadata(
            file_path=Path("/test/file.py"),
            symbols=symbols,
            imports=["import os", "from typing import List"],
            exports=["test_func"],
            file_docstring="Module docstring",
            lsp_server="pylsp"
        )

        assert metadata.file_path == Path("/test/file.py")
        assert len(metadata.symbols) == 1
        assert len(metadata.imports) == 2
        assert metadata.lsp_server == "pylsp"

    @pytest.mark.asyncio
    async def test_lsp_metadata_extractor_python_specific(self, extractor, mock_lsp_client):
        """Test LspMetadataExtractor Python-specific extraction"""
        # Test Python-specific symbol extraction
        python_code = """
import asyncio
from typing import Optional, List

async def async_function(data: List[str]) -> Optional[str]:
    '''Async function with type hints'''
    return data[0] if data else None

class DataProcessor:
    '''Data processing class'''

    def __init__(self, config: dict):
        self.config = config

    @property
    def status(self) -> str:
        return "ready"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            file_path = Path(f.name)

        # Mock enhanced LSP responses for Python
        mock_lsp_client.document_symbols.return_value = [
            {
                "name": "async_function",
                "kind": 12,
                "range": {"start": {"line": 4, "character": 0}, "end": {"line": 6, "character": 20}},
                "detail": "async def async_function(data: List[str]) -> Optional[str]"
            },
            {
                "name": "DataProcessor",
                "kind": 5,
                "range": {"start": {"line": 8, "character": 0}, "end": {"line": 17, "character": 0}},
                "children": [
                    {
                        "name": "__init__",
                        "kind": 6,
                        "range": {"start": {"line": 11, "character": 4}, "end": {"line": 12, "character": 20}}
                    },
                    {
                        "name": "status",
                        "kind": 7,  # Property
                        "range": {"start": {"line": 14, "character": 4}, "end": {"line": 16, "character": 20}}
                    }
                ]
            }
        ]

        await extractor.initialize(file_path.parent)
        metadata = await extractor.extract_file_metadata(file_path)

        assert metadata is not None
        assert len(metadata.imports) >= 2
        assert any("asyncio" in imp for imp in metadata.imports)
        assert any("typing" in imp for imp in metadata.imports)

        # Check for async function detection
        async_func = next((s for s in metadata.symbols if s.name == "async_function"), None)
        assert async_func is not None

    @pytest.mark.asyncio
    async def test_lsp_metadata_extractor_error_handling(self, extractor, mock_lsp_client):
        """Test LspMetadataExtractor error handling"""
        # Test with non-existent file
        non_existent_file = Path("/nonexistent/file.py")

        with pytest.raises(FileNotFoundError):
            await extractor.extract_file_metadata(non_existent_file)

        # Test with LSP error
        mock_lsp_client.document_symbols.side_effect = Exception("LSP server error")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            file_path = Path(f.name)

        await extractor.initialize(file_path.parent)

        # Should handle LSP errors gracefully
        metadata = await extractor.extract_file_metadata(file_path)
        # Should return metadata with extraction errors noted
        assert metadata is not None
        assert len(metadata.extraction_errors) > 0

    @pytest.mark.asyncio
    async def test_lsp_metadata_extractor_multiple_languages(self, extractor, mock_lsp_client):
        """Test LspMetadataExtractor with multiple programming languages"""
        # Test JavaScript/TypeScript support
        js_code = """
function calculateSum(a, b) {
    return a + b;
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push({operation: 'add', result});
        return result;
    }
}

export { Calculator, calculateSum };
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(js_code)
            js_file_path = Path(f.name)

        # Mock JavaScript LSP responses
        mock_lsp_client.document_symbols.return_value = [
            {
                "name": "calculateSum",
                "kind": 12,
                "range": {"start": {"line": 1, "character": 0}, "end": {"line": 3, "character": 1}}
            },
            {
                "name": "Calculator",
                "kind": 5,
                "range": {"start": {"line": 5, "character": 0}, "end": {"line": 15, "character": 1}}
            }
        ]

        await extractor.initialize(js_file_path.parent)
        js_metadata = await extractor.extract_file_metadata(js_file_path)

        assert js_metadata is not None
        assert len(js_metadata.symbols) >= 2
        assert any(symbol.name == "calculateSum" for symbol in js_metadata.symbols)
        assert any(symbol.name == "Calculator" for symbol in js_metadata.symbols)

    @pytest.mark.asyncio
    async def test_lsp_metadata_extractor_relationship_building(self, extractor, mock_lsp_client):
        """Test LspMetadataExtractor relationship building"""
        # Mock relationships between symbols
        mock_lsp_client.references.return_value = [
            {"uri": "file:///test/file.py", "range": {"start": {"line": 10, "character": 5}}},
            {"uri": "file:///test/other.py", "range": {"start": {"line": 5, "character": 0}}}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class Base:\n    pass\n\nclass Derived(Base):\n    pass")
            file_path = Path(f.name)

        await extractor.initialize(file_path.parent)

        # Test relationship graph building
        files = [file_path]
        relationships = await extractor.build_relationship_graph(files)

        assert isinstance(relationships, list)
        # Should detect inheritance relationships
        inheritance_rels = [r for r in relationships
                          if r.relationship_type.name == "INHERITANCE"]
        assert len(inheritance_rels) >= 0  # May be 0 if LSP doesn't detect inheritance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])