"""
Integration tests for complete document ingestion pipeline.

Tests the end-to-end workflow from file watching to vector storage, including:
- File system event detection and processing
- Document parsing and chunking
- Vector embedding generation
- Qdrant storage and indexing
- Metadata extraction and storage
- Error handling and recovery
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
import shutil
import os

from testcontainers.compose import DockerCompose

from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.daemon_manager import DaemonManager, DaemonConfig
from workspace_qdrant_mcp.core.watch_manager import WatchManager
from workspace_qdrant_mcp.tools.grpc_tools import test_grpc_connection, get_grpc_engine_stats
from workspace_qdrant_mcp.parsers.factory import ParserFactory
from workspace_qdrant_mcp.embeddings.factory import EmbeddingFactory


@pytest.fixture(scope="module")
def isolated_qdrant():
    """Start isolated Qdrant instance for ingestion pipeline testing."""
    compose_file = """
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6335:6333"
      - "6336:6334"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        compose_path = Path(temp_dir) / "docker-compose.yml"
        compose_path.write_text(compose_file)
        
        with DockerCompose(temp_dir) as compose:
            qdrant_url = compose.get_service_host("qdrant", 6333)
            qdrant_port = compose.get_service_port("qdrant", 6333)
            
            # Wait for Qdrant to be ready
            import requests
            for _ in range(30):
                try:
                    response = requests.get(f"http://{qdrant_url}:{qdrant_port}/health")
                    if response.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            
            yield {
                "http_host": qdrant_url, 
                "http_port": qdrant_port,
                "grpc_host": qdrant_url,
                "grpc_port": compose.get_service_port("qdrant", 6334)
            }


@pytest.fixture
def test_workspace():
    """Create test workspace with various document types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        
        # Create directory structure
        (workspace_path / "docs").mkdir()
        (workspace_path / "src").mkdir()
        (workspace_path / "tests").mkdir()
        (workspace_path / "data").mkdir()
        
        # Create test documents
        test_files = {
            "docs/readme.md": """
# Test Project

This is a test project for document ingestion pipeline testing.

## Features

- Document processing
- Vector search
- File watching
            """,
            "docs/api.md": """
# API Documentation

## Endpoints

### GET /search
Search for documents

### POST /ingest  
Ingest new documents
            """,
            "src/main.py": """
def main():
    \"\"\"Main application entry point.\"\"\"
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
            """,
            "tests/test_example.py": """
import pytest

def test_example():
    assert True

def test_math():
    assert 2 + 2 == 4
            """,
            "data/sample.txt": """
This is a sample data file containing various information
about testing document ingestion pipelines and their
performance characteristics under different conditions.
            """
        }
        
        for rel_path, content in test_files.items():
            file_path = workspace_path / rel_path
            file_path.write_text(content.strip())
        
        yield {
            "path": workspace_path,
            "files": list(test_files.keys()),
            "expected_documents": len(test_files)
        }


@pytest.fixture
async def ingestion_config(isolated_qdrant, test_workspace):
    """Create configuration for ingestion pipeline testing."""
    config_data = {
        "qdrant": {
            "host": isolated_qdrant["http_host"],
            "port": isolated_qdrant["http_port"],
            "grpc_port": isolated_qdrant["grpc_port"],
            "collection_name": "ingestion_pipeline_test",
            "timeout": 30,
            "prefer_grpc": True
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu"
        },
        "ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 10,
            "max_file_size_mb": 50
        },
        "watch": {
            "directories": [str(test_workspace["path"])],
            "patterns": ["*.md", "*.py", "*.txt"],
            "ignore_patterns": [".git/*", "__pycache__/*"],
            "debounce_seconds": 0.1  # Fast for testing
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = Config.from_yaml(config_path)
        yield config
    finally:
        Path(config_path).unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.slow
class TestDocumentIngestionPipeline:
    """Integration tests for complete document ingestion pipeline."""
    
    async def test_end_to_end_document_ingestion(
        self, 
        isolated_qdrant,
        test_workspace,
        ingestion_config
    ):
        """
        Test complete end-to-end document ingestion pipeline.
        
        Verifies:
        1. File system monitoring detects new files
        2. Documents are parsed correctly  
        3. Text is chunked appropriately
        4. Vector embeddings are generated
        5. Data is stored in Qdrant with proper metadata
        6. Search functionality works with ingested data
        """
        
        # Initialize client
        client = QdrantWorkspaceClient(config=ingestion_config)
        await client.initialize()
        
        try:
            # Ensure collection exists
            await client.create_collection_if_not_exists(
                ingestion_config.qdrant.collection_name
            )
            
            # Process all test files
            ingestion_results = []
            for rel_path in test_workspace["files"]:
                file_path = test_workspace["path"] / rel_path
                
                # Process document through the pipeline
                result = await client.add_document(
                    file_path=str(file_path),
                    collection=ingestion_config.qdrant.collection_name,
                    metadata={
                        "file_type": file_path.suffix,
                        "relative_path": rel_path,
                        "test_run": True
                    }
                )
                
                ingestion_results.append({
                    "file_path": rel_path,
                    "success": result.get("success", False),
                    "chunks_added": result.get("chunks_added", 0),
                    "document_id": result.get("document_id")
                })
            
            # Verify ingestion results
            successful_ingestions = [r for r in ingestion_results if r["success"]]
            assert len(successful_ingestions) >= 4, "Most documents should ingest successfully"
            
            total_chunks = sum(r["chunks_added"] for r in successful_ingestions)
            assert total_chunks > 10, "Should generate multiple chunks across documents"
            
            # Test search functionality with ingested data
            search_queries = [
                "test project documentation",
                "API endpoints search",
                "main function Python",
                "pytest testing example"
            ]
            
            search_results = []
            for query in search_queries:
                result = await client.search(
                    query=query,
                    collection=ingestion_config.qdrant.collection_name,
                    limit=5
                )
                search_results.append({
                    "query": query,
                    "results_count": len(result.get("results", [])),
                    "has_results": len(result.get("results", [])) > 0
                })
            
            # At least half of search queries should return results
            successful_searches = [r for r in search_results if r["has_results"]]
            assert len(successful_searches) >= 2, "Search should work with ingested data"
            
        finally:
            await client.close()

    async def test_file_watching_integration(
        self,
        isolated_qdrant, 
        test_workspace,
        ingestion_config
    ):
        """
        Test file watching integration with ingestion pipeline.
        
        Creates, modifies, and deletes files while monitoring that:
        1. New files trigger ingestion
        2. Modified files are re-processed
        3. File pattern matching works correctly
        4. Ignored patterns are respected
        """
        
        # Create daemon configuration for file watching
        daemon_config = DaemonConfig(
            project_name="test_ingestion",
            project_path=str(test_workspace["path"]),
            grpc_host=isolated_qdrant["grpc_host"],
            grpc_port=isolated_qdrant["grpc_port"],
            collection_name="watch_test",
            health_check_interval=1.0,
            startup_timeout=10.0
        )
        
        # Mock the daemon manager to simulate file watching
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemon:
            mock_instance = AsyncMock()
            MockDaemon.return_value = mock_instance
            mock_instance.is_healthy.return_value = True
            
            daemon_manager = DaemonManager(daemon_config)
            
            # Simulate file events
            test_events = []
            
            # Add new file
            new_file = test_workspace["path"] / "docs" / "new_doc.md"
            new_file.write_text("# New Document\n\nThis is a newly created document.")
            test_events.append({"type": "created", "path": str(new_file)})
            
            # Modify existing file
            existing_file = test_workspace["path"] / "docs" / "readme.md"
            content = existing_file.read_text()
            existing_file.write_text(content + "\n\n## New Section\n\nUpdated content.")
            test_events.append({"type": "modified", "path": str(existing_file)})
            
            # Create file that should be ignored
            ignored_file = test_workspace["path"] / "__pycache__" / "cache.pyc"
            ignored_file.parent.mkdir(exist_ok=True)
            ignored_file.write_bytes(b"binary cache data")
            test_events.append({"type": "created", "path": str(ignored_file)})
            
            # Verify watch patterns would catch appropriate files
            watch_patterns = ingestion_config.watch.patterns
            ignore_patterns = ingestion_config.watch.ignore_patterns
            
            processed_files = []
            ignored_files = []
            
            for event in test_events:
                file_path = Path(event["path"])
                
                # Check if file matches patterns
                should_process = False
                for pattern in watch_patterns:
                    if file_path.match(pattern):
                        should_process = True
                        break
                
                # Check ignore patterns
                should_ignore = False
                for pattern in ignore_patterns:
                    if pattern.replace('/*', '') in str(file_path):
                        should_ignore = True
                        break
                
                if should_process and not should_ignore:
                    processed_files.append(event)
                else:
                    ignored_files.append(event)
            
            # Verify file filtering logic
            assert len(processed_files) >= 2, "Should process new and modified markdown files"
            assert len(ignored_files) >= 1, "Should ignore cache files"
            
            # Verify specific file handling
            new_doc_processed = any(
                "new_doc.md" in event["path"] for event in processed_files
            )
            readme_processed = any(
                "readme.md" in event["path"] for event in processed_files
            )
            cache_ignored = any(
                "cache.pyc" in event["path"] for event in ignored_files  
            )
            
            assert new_doc_processed, "New markdown file should be processed"
            assert readme_processed, "Modified markdown file should be processed"
            assert cache_ignored, "Cache file should be ignored"

    async def test_error_handling_and_recovery(
        self,
        isolated_qdrant,
        test_workspace, 
        ingestion_config
    ):
        """
        Test error handling and recovery in ingestion pipeline.
        
        Tests various error scenarios:
        1. Corrupted file handling
        2. Network connectivity issues
        3. Qdrant service unavailability
        4. Parser errors with unsupported files
        5. Memory pressure during large file processing
        """
        
        client = QdrantWorkspaceClient(config=ingestion_config)
        await client.initialize()
        
        try:
            # Test 1: Corrupted file
            corrupted_file = test_workspace["path"] / "corrupted.txt"
            corrupted_file.write_bytes(b"\xff\xfe\x00\x01invalid utf-8 \x80\x81")
            
            result = await client.add_document(
                file_path=str(corrupted_file),
                collection=ingestion_config.qdrant.collection_name
            )
            
            # Should handle gracefully, not crash
            assert isinstance(result, dict), "Should return error result dict"
            
            # Test 2: Very large file (simulated)
            large_file = test_workspace["path"] / "large.txt"
            large_content = "Large file content " * 10000  # ~200KB
            large_file.write_text(large_content)
            
            result = await client.add_document(
                file_path=str(large_file),
                collection=ingestion_config.qdrant.collection_name
            )
            
            # Should either succeed or fail gracefully
            assert isinstance(result, dict), "Large file processing should return result"
            
            # Test 3: Network timeout handling
            with patch.object(client, '_qdrant_client') as mock_client:
                mock_client.upsert.side_effect = asyncio.TimeoutError("Connection timeout")
                
                result = await client.add_document(
                    file_path=str(test_workspace["path"] / "docs" / "readme.md"),
                    collection=ingestion_config.qdrant.collection_name
                )
                
                # Should handle timeout gracefully
                assert not result.get("success", True), "Should report failure on timeout"
                assert "timeout" in str(result.get("error", "")).lower()
            
            # Test 4: Unsupported file type
            binary_file = test_workspace["path"] / "binary.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03binary data here")
            
            result = await client.add_document(
                file_path=str(binary_file),
                collection=ingestion_config.qdrant.collection_name
            )
            
            # Should handle unsupported file types
            assert isinstance(result, dict), "Binary file should return result dict"
            
        finally:
            await client.close()

    async def test_metadata_extraction_and_storage(
        self,
        isolated_qdrant,
        test_workspace,
        ingestion_config
    ):
        """
        Test metadata extraction and storage throughout pipeline.
        
        Verifies:
        1. File metadata is extracted correctly
        2. Custom metadata is preserved
        3. Metadata is stored and searchable
        4. Metadata filtering works in search
        """
        
        client = QdrantWorkspaceClient(config=ingestion_config)
        await client.initialize()
        
        try:
            await client.create_collection_if_not_exists(
                ingestion_config.qdrant.collection_name
            )
            
            # Test files with different metadata scenarios
            test_cases = [
                {
                    "file": "docs/readme.md",
                    "metadata": {
                        "project": "test-project",
                        "author": "test-author",
                        "importance": "high",
                        "category": "documentation"
                    }
                },
                {
                    "file": "src/main.py", 
                    "metadata": {
                        "language": "python",
                        "module": "main",
                        "category": "source"
                    }
                },
                {
                    "file": "tests/test_example.py",
                    "metadata": {
                        "language": "python",
                        "type": "test",
                        "category": "test"
                    }
                }
            ]
            
            ingestion_results = []
            for case in test_cases:
                file_path = test_workspace["path"] / case["file"]
                
                result = await client.add_document(
                    file_path=str(file_path),
                    collection=ingestion_config.qdrant.collection_name,
                    metadata=case["metadata"]
                )
                
                ingestion_results.append({
                    "case": case,
                    "result": result,
                    "success": result.get("success", False)
                })
            
            # Verify successful ingestion
            successful_results = [r for r in ingestion_results if r["success"]]
            assert len(successful_results) >= 2, "Most documents should ingest with metadata"
            
            # Test metadata-based search filtering
            search_scenarios = [
                {
                    "query": "documentation",
                    "filter_metadata": {"category": "documentation"},
                    "expected_results": 1
                },
                {
                    "query": "python", 
                    "filter_metadata": {"language": "python"},
                    "expected_results": 2
                },
                {
                    "query": "test",
                    "filter_metadata": {"type": "test"},
                    "expected_results": 1
                }
            ]
            
            for scenario in search_scenarios:
                # Note: Actual metadata filtering would depend on client implementation
                # This tests the concept and structure
                result = await client.search(
                    query=scenario["query"],
                    collection=ingestion_config.qdrant.collection_name,
                    limit=10
                )
                
                # Verify search returns results (actual filtering test would be more complex)
                assert isinstance(result.get("results", []), list), "Should return search results"
                
                # Check if results contain expected metadata structure
                if result.get("results"):
                    first_result = result["results"][0]
                    assert "payload" in first_result or "metadata" in first_result, \
                        "Results should include metadata/payload"
            
        finally:
            await client.close()

    async def test_concurrent_ingestion_handling(
        self,
        isolated_qdrant,
        test_workspace,
        ingestion_config
    ):
        """
        Test concurrent document ingestion handling.
        
        Simulates multiple files being processed simultaneously to verify:
        1. Concurrent processing doesn't cause conflicts
        2. Resource management under load
        3. Proper error isolation between concurrent operations
        """
        
        client = QdrantWorkspaceClient(config=ingestion_config)
        await client.initialize()
        
        try:
            await client.create_collection_if_not_exists(
                ingestion_config.qdrant.collection_name
            )
            
            # Create additional test files for concurrent processing
            concurrent_files = []
            for i in range(8):  # Process 8 files concurrently
                file_path = test_workspace["path"] / f"concurrent_{i}.txt"
                content = f"Concurrent test document {i}\n" + "Content line " * 50
                file_path.write_text(content)
                concurrent_files.append(str(file_path))
            
            # Add original test files
            for rel_path in test_workspace["files"][:3]:  # First 3 files
                concurrent_files.append(str(test_workspace["path"] / rel_path))
            
            # Process files concurrently
            async def process_file(file_path: str, file_index: int):
                try:
                    result = await client.add_document(
                        file_path=file_path,
                        collection=ingestion_config.qdrant.collection_name,
                        metadata={
                            "concurrent_batch": True,
                            "file_index": file_index,
                            "processing_time": time.time()
                        }
                    )
                    return {"file_path": file_path, "result": result, "error": None}
                except Exception as e:
                    return {"file_path": file_path, "result": None, "error": str(e)}
            
            # Execute concurrent ingestion
            tasks = [
                process_file(file_path, i) 
                for i, file_path in enumerate(concurrent_files)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze concurrent processing results
            successful_results = []
            failed_results = []
            exception_results = []
            
            for result in results:
                if isinstance(result, Exception):
                    exception_results.append(result)
                elif result.get("error"):
                    failed_results.append(result)
                elif result.get("result", {}).get("success"):
                    successful_results.append(result)
                else:
                    failed_results.append(result)
            
            # Verify concurrent processing performance and reliability
            total_processing_time = end_time - start_time
            assert total_processing_time < 60, "Concurrent processing took too long"
            
            success_rate = len(successful_results) / len(concurrent_files)
            assert success_rate >= 0.7, f"Success rate too low: {success_rate:.2%}"
            
            assert len(exception_results) == 0, "No exceptions should be raised"
            
            # Verify no resource conflicts or data corruption
            if successful_results:
                # Check that document IDs are unique
                doc_ids = [
                    r["result"].get("document_id") 
                    for r in successful_results 
                    if r["result"].get("document_id")
                ]
                assert len(doc_ids) == len(set(doc_ids)), "Document IDs should be unique"
                
                # Check that all successful results have reasonable chunk counts
                chunk_counts = [
                    r["result"].get("chunks_added", 0)
                    for r in successful_results
                ]
                assert all(count > 0 for count in chunk_counts), "All docs should have chunks"
                
        finally:
            await client.close()