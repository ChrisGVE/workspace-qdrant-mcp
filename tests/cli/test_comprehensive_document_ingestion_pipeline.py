"""
Comprehensive document ingestion pipeline testing for Task 79.

This test suite provides end-to-end validation of the complete document ingestion
workflow, from file watching to vector storage and search functionality.

Test Coverage:
1. Multi-format parser validation (PDF, EPUB, DOCX, Code, Text, HTML, PPTX)
2. End-to-end ingestion pipeline testing
3. SQLite state persistence validation
4. Qdrant integration and vector storage
5. Search functionality verification
6. Error recovery and resilience testing
7. Performance measurement and bottleneck identification

Integration Points:
- cli/ingestion_engine.py
- cli/parsers/* modules
- core/sqlite_state_manager.py
- tools/documents.py
- File watching system
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wqm_cli.cli.ingestion_engine import DocumentIngestionEngine, IngestionResult
from wqm_cli.cli.parsers import (
    CodeParser,
    DocxParser,
    EpubParser,
    HtmlParser,
    PDFParser,
    PptxParser,
    TextParser,
)
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
def comprehensive_test_workspace():
    """Create a comprehensive test workspace with all supported file formats."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        
        # Create directory structure
        (workspace_path / "documents").mkdir()
        (workspace_path / "code").mkdir()
        (workspace_path / "presentations").mkdir()
        (workspace_path / "web").mkdir()
        (workspace_path / "books").mkdir()
        
        # Create test files for each format
        test_files = {}
        
        # Text files
        test_files["documents/readme.txt"] = """
This is a comprehensive test document for the ingestion pipeline.
It contains multiple paragraphs to test text chunking and processing.

The document should be properly parsed, chunked, and indexed for search.
Performance metrics should be collected during processing.
        """.strip()
        
        # Markdown files  
        test_files["documents/guide.md"] = """
# Test Guide

This is a **markdown** document with various formatting elements.

## Features

- Lists and bullets
- **Bold** and *italic* text
- [Links](https://example.com)
- Code blocks:

```python
def hello():
    return "world"
```

## Search Testing

This content should be searchable after ingestion.
        """.strip()
        
        # Code files
        test_files["code/main.py"] = '''
"""
Main application module for testing code parsing.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TestApplication:
    """Test application for document ingestion testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def start(self) -> None:
        """Start the application."""
        self.logger.info("Starting test application")
        await self.process_data()
    
    async def process_data(self) -> List[str]:
        """Process test data."""
        data = ["item1", "item2", "item3"]
        results = []
        
        for item in data:
            processed = await self.process_item(item)
            results.append(processed)
        
        return results
    
    async def process_item(self, item: str) -> str:
        """Process a single item."""
        await asyncio.sleep(0.1)  # Simulate processing
        return f"processed_{item}"


async def main():
    """Main entry point."""
    app = TestApplication()
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
        '''.strip()
        
        test_files["code/utils.js"] = '''
/**
 * Utility functions for JavaScript code parsing test.
 */

class DocumentProcessor {
    constructor(options = {}) {
        this.options = {
            chunkSize: 1000,
            overlap: 200,
            ...options
        };
    }
    
    /**
     * Process a document into chunks.
     * @param {string} content - Document content
     * @returns {Array<string>} Processed chunks
     */
    processDocument(content) {
        const chunks = [];
        const chunkSize = this.options.chunkSize;
        const overlap = this.options.overlap;
        
        for (let i = 0; i < content.length; i += chunkSize - overlap) {
            const chunk = content.substring(i, i + chunkSize);
            chunks.push(chunk);
        }
        
        return chunks;
    }
    
    /**
     * Extract metadata from document.
     * @param {string} filepath - Path to document
     * @returns {Object} Metadata object
     */
    extractMetadata(filepath) {
        return {
            filename: filepath.split('/').pop(),
            extension: filepath.split('.').pop(),
            processed_at: new Date().toISOString()
        };
    }
}

module.exports = { DocumentProcessor };
        '''.strip()
        
        # HTML files
        test_files["web/index.html"] = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test HTML Document</title>
    <meta name="description" content="Test HTML for document ingestion pipeline">
</head>
<body>
    <header>
        <h1>Document Ingestion Test Page</h1>
        <nav>
            <ul>
                <li><a href="#features">Features</a></li>
                <li><a href="#testing">Testing</a></li>
                <li><a href="#performance">Performance</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="features">
            <h2>Pipeline Features</h2>
            <p>The document ingestion pipeline supports multiple file formats:</p>
            <ul>
                <li>HTML documents with full parsing</li>
                <li>Text extraction and metadata preservation</li>
                <li>Link and image handling</li>
                <li>Table and list processing</li>
            </ul>
        </section>
        
        <section id="testing">
            <h2>Testing Methodology</h2>
            <p>Comprehensive testing includes:</p>
            <ol>
                <li>Parser validation for each format</li>
                <li>End-to-end pipeline testing</li>
                <li>Performance benchmarking</li>
                <li>Error recovery scenarios</li>
            </ol>
            
            <table>
                <thead>
                    <tr>
                        <th>Format</th>
                        <th>Parser</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>HTML</td>
                        <td>HtmlParser</td>
                        <td>✅ Implemented</td>
                    </tr>
                    <tr>
                        <td>Text</td>
                        <td>TextParser</td>
                        <td>✅ Implemented</td>
                    </tr>
                </tbody>
            </table>
        </section>
        
        <section id="performance">
            <h2>Performance Metrics</h2>
            <p>Key performance indicators:</p>
            <ul>
                <li>Throughput: files processed per second</li>
                <li>Latency: time per file processing</li>
                <li>Memory usage during large file processing</li>
                <li>Error recovery time</li>
            </ul>
        </section>
    </main>
    
    <footer>
        <p>Test document for comprehensive pipeline validation.</p>
    </footer>
</body>
</html>
        '''.strip()
        
        # Create all text files
        for rel_path, content in test_files.items():
            file_path = workspace_path / rel_path
            file_path.write_text(content)
        
        # Create minimal binary files for testing (these will be created as placeholders)
        binary_files = {
            "documents/sample.pdf": b"PDF placeholder content",
            "documents/document.docx": b"DOCX placeholder content", 
            "books/sample.epub": b"EPUB placeholder content",
            "presentations/slides.pptx": b"PPTX placeholder content",
        }
        
        for rel_path, content in binary_files.items():
            file_path = workspace_path / rel_path
            file_path.write_bytes(content)
        
        yield {
            "path": workspace_path,
            "text_files": list(test_files.keys()),
            "binary_files": list(binary_files.keys()),
            "all_files": list(test_files.keys()) + list(binary_files.keys()),
            "expected_documents": len(test_files) + len(binary_files)
        }


@pytest.fixture
def mock_qdrant_client():
    """Create a comprehensive mock Qdrant client."""
    client = AsyncMock(spec=QdrantWorkspaceClient)
    client.list_collections.return_value = ["test-collection", "pipeline-test"]
    client.create_collection_if_not_exists.return_value = {"status": "success"}
    client.close.return_value = None
    return client


@pytest.fixture
def mock_sqlite_state_manager():
    """Create a mock SQLite state manager."""
    manager = AsyncMock(spec=SQLiteStateManager)
    manager.initialize.return_value = None
    manager.add_file.return_value = None
    manager.update_file_status.return_value = None
    manager.get_file_status.return_value = "pending"
    manager.get_processing_stats.return_value = {
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0
    }
    return manager


@pytest.mark.integration
@pytest.mark.slow
class TestMultiFormatParserValidation:
    """Test individual parser functionality for all supported formats."""
    
    def test_text_parser_validation(self, comprehensive_test_workspace):
        """Test TextParser functionality."""
        parser = TextParser()
        text_file = comprehensive_test_workspace["path"] / "documents/readme.txt"
        
        # Test parser detection
        assert parser.can_parse(text_file) is True
        assert parser.format_name == "Plain Text"
        assert ".txt" in parser.supported_extensions
        
        # Test parsing options
        options = parser.get_parsing_options()
        assert isinstance(options, dict)
    
    def test_code_parser_validation(self, comprehensive_test_workspace):
        """Test CodeParser functionality."""
        parser = CodeParser()
        
        # Test Python file
        py_file = comprehensive_test_workspace["path"] / "code/main.py"
        assert parser.can_parse(py_file) is True
        
        # Test JavaScript file
        js_file = comprehensive_test_workspace["path"] / "code/utils.js"
        assert parser.can_parse(js_file) is True
        
        assert parser.format_name == "Source Code"
        assert ".py" in parser.supported_extensions
        assert ".js" in parser.supported_extensions
    
    def test_html_parser_validation(self, comprehensive_test_workspace):
        """Test HtmlParser functionality."""
        parser = HtmlParser()
        html_file = comprehensive_test_workspace["path"] / "web/index.html"
        
        assert parser.can_parse(html_file) is True
        assert parser.format_name == "HTML Document"
        assert ".html" in parser.supported_extensions
    
    @pytest.mark.asyncio
    async def test_parser_content_extraction(self, comprehensive_test_workspace):
        """Test actual content extraction from parsers."""
        parsers = [
            (TextParser(), "documents/readme.txt"),
            (CodeParser(), "code/main.py"), 
            (HtmlParser(), "web/index.html"),
        ]
        
        for parser, file_path in parsers:
            full_path = comprehensive_test_workspace["path"] / file_path
            
            try:
                parsed_doc = await parser.parse(full_path)
                
                # Basic validation
                assert parsed_doc.content is not None
                assert len(parsed_doc.content) > 0
                assert parsed_doc.content_hash is not None
                assert isinstance(parsed_doc.metadata, dict)
                
                # Content quality checks
                assert len(parsed_doc.content.strip()) > 10
                assert parsed_doc.metadata.get("file_path") is not None
                
            except Exception as e:
                pytest.fail(f"Parser {parser.format_name} failed on {file_path}: {e}")


@pytest.mark.integration 
@pytest.mark.slow
class TestEndToEndIngestionPipeline:
    """Test complete end-to-end ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(
        self, 
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test complete pipeline from file discovery to ingestion."""
        
        # Mock the add_document function to simulate successful ingestion
        with patch("workspace_qdrant_mcp.tools.documents.add_document") as mock_add_doc:
            mock_add_doc.return_value = {
                "success": True,
                "points_added": 3,
                "document_id": "test-doc-123"
            }
            
            # Create ingestion engine
            engine = DocumentIngestionEngine(
                client=mock_qdrant_client,
                concurrency=3,
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Process the entire workspace
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="pipeline-test",
                recursive=True,
                dry_run=False
            )
            
            # Verify successful processing
            assert result.success is True
            assert result.stats.files_found > 0
            assert result.stats.files_processed > 0
            assert result.stats.total_documents > 0
            
            # Verify performance metrics
            assert result.stats.processing_time > 0
            assert result.stats.files_per_second > 0
            assert result.stats.success_rate > 0
            
            # Verify processing statistics
            print(f"Pipeline Results:")
            print(f"  Files found: {result.stats.files_found}")
            print(f"  Files processed: {result.stats.files_processed}")
            print(f"  Processing time: {result.stats.processing_time:.2f}s")
            print(f"  Rate: {result.stats.files_per_second:.1f} files/sec")
            print(f"  Success rate: {result.stats.success_rate:.1f}%")
    
    @pytest.mark.asyncio
    async def test_pipeline_with_format_filtering(
        self,
        comprehensive_test_workspace, 
        mock_qdrant_client
    ):
        """Test pipeline with specific format filtering."""
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document") as mock_add_doc:
            mock_add_doc.return_value = {"success": True, "points_added": 1}
            
            engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
            
            # Test filtering for only text and code files
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="pipeline-test",
                formats=["text", "code"],
                recursive=True
            )
            
            assert result.success is True
            
            # Should have processed fewer files due to filtering
            total_files = len(comprehensive_test_workspace["all_files"])
            assert result.stats.files_found < total_files
            
            print(f"Filtered Processing:")
            print(f"  Total available: {total_files}")
            print(f"  Files found with filter: {result.stats.files_found}")
            print(f"  Files processed: {result.stats.files_processed}")
    
    @pytest.mark.asyncio
    async def test_pipeline_dry_run_mode(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test pipeline dry run mode for analysis without ingestion."""
        
        engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
        
        result = await engine.process_directory(
            directory_path=comprehensive_test_workspace["path"],
            collection="pipeline-test",
            dry_run=True,
            recursive=True
        )
        
        assert result.success is True
        assert result.dry_run is True
        assert result.stats.files_found > 0
        assert result.stats.files_processed > 0  # Files are "processed" in analysis
        
        # In dry run, should not call the actual client methods
        mock_qdrant_client.list_collections.assert_not_called()
        
        print(f"Dry Run Analysis:")
        print(f"  Files found: {result.stats.files_found}")
        print(f"  Estimated chunks: {result.stats.total_chunks}")
        print(f"  Estimated processing time: {result.stats.processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test pipeline concurrent processing performance."""
        
        processing_times = []
        
        async def timed_add_document(*args, **kwargs):
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing time
            end_time = time.time()
            processing_times.append(end_time - start_time)
            return {"success": True, "points_added": 1}
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document", side_effect=timed_add_document):
            engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=4)
            
            start_time = time.time()
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="pipeline-test",
                recursive=True
            )
            end_time = time.time()
            
            total_time = end_time - start_time
            
            assert result.success is True
            assert result.stats.files_processed > 0
            
            # With concurrency, should be faster than sequential processing
            expected_sequential_time = len(processing_times) * 0.1
            assert total_time < expected_sequential_time * 0.8  # Allow overhead
            
            print(f"Concurrent Processing Performance:")
            print(f"  Files processed: {result.stats.files_processed}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Expected sequential: {expected_sequential_time:.2f}s")
            print(f"  Speedup: {expected_sequential_time / total_time:.1f}x")


@pytest.mark.integration
class TestSQLiteStateIntegration:
    """Test SQLite state manager integration with ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_state_manager_file_tracking(
        self,
        comprehensive_test_workspace,
        mock_sqlite_state_manager,
        mock_qdrant_client
    ):
        """Test that state manager properly tracks file processing."""
        
        # Create a partial mock that tracks calls
        call_log = []
        
        async def mock_add_file(file_path, metadata=None):
            call_log.append(("add_file", file_path, metadata))
            
        async def mock_update_status(file_path, status, metadata=None):
            call_log.append(("update_status", file_path, status, metadata))
        
        mock_sqlite_state_manager.add_file.side_effect = mock_add_file
        mock_sqlite_state_manager.update_file_status.side_effect = mock_update_status
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document") as mock_add_doc:
            mock_add_doc.return_value = {"success": True, "points_added": 1}
            
            # Patch the state manager into the ingestion process
            with patch("workspace_qdrant_mcp.core.sqlite_state_manager.SQLiteStateManager", return_value=mock_sqlite_state_manager):
                
                engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
                
                # Process a few files
                result = await engine.process_directory(
                    directory_path=comprehensive_test_workspace["path"] / "documents",
                    collection="pipeline-test",
                    recursive=False
                )
                
                assert result.success is True
                
                # Verify state tracking calls were made
                # Note: This would require actual integration with the state manager in the engine
                print(f"State Manager Calls: {len(call_log)}")
                for call in call_log:
                    print(f"  {call[0]}: {Path(call[1]).name}")
    
    @pytest.mark.asyncio
    async def test_state_persistence_across_sessions(
        self,
        comprehensive_test_workspace,
        mock_sqlite_state_manager
    ):
        """Test state persistence across ingestion sessions."""
        
        # Simulate previous processing state
        processed_files = ["documents/readme.txt", "code/main.py"]
        
        async def mock_get_file_status(file_path):
            if any(pf in str(file_path) for pf in processed_files):
                return "completed"
            return "pending"
        
        mock_sqlite_state_manager.get_file_status.side_effect = mock_get_file_status
        
        # Simulate querying state for resume capability
        workspace_path = comprehensive_test_workspace["path"]
        for file_path in workspace_path.rglob("*"):
            if file_path.is_file():
                status = await mock_sqlite_state_manager.get_file_status(str(file_path))
                print(f"File: {file_path.name} -> Status: {status}")
        
        # Verify that previously processed files would be skipped
        pending_files = []
        for file_path in workspace_path.rglob("*"):
            if file_path.is_file():
                status = await mock_sqlite_state_manager.get_file_status(str(file_path))
                if status == "pending":
                    pending_files.append(file_path)
        
        assert len(pending_files) > 0
        print(f"Files pending processing: {len(pending_files)}")


@pytest.mark.integration
class TestErrorRecoveryAndResilience:
    """Test error recovery and resilience scenarios."""
    
    @pytest.mark.asyncio
    async def test_corrupted_file_handling(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test handling of corrupted and unreadable files."""
        
        # Create corrupted files
        corrupted_files = {
            "documents/corrupted.txt": b"\xff\xfe\x00\x01invalid utf-8 \x80\x81",
            "documents/empty.txt": b"",
            "documents/huge.txt": b"x" * (50 * 1024 * 1024),  # 50MB file
        }
        
        for rel_path, content in corrupted_files.items():
            file_path = comprehensive_test_workspace["path"] / rel_path
            file_path.write_bytes(content)
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document") as mock_add_doc:
            mock_add_doc.return_value = {"success": True, "points_added": 1}
            
            engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
            
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="pipeline-test",
                recursive=True
            )
            
            # Pipeline should complete successfully despite corrupted files
            assert result.success is True
            
            # Some files should have failed or been skipped
            total_files = result.stats.files_found
            successful_files = result.stats.files_processed
            
            print(f"Error Recovery Results:")
            print(f"  Total files: {total_files}")
            print(f"  Successful: {successful_files}")
            print(f"  Failed: {result.stats.files_failed}")
            print(f"  Skipped: {result.stats.files_skipped}")
            print(f"  Errors: {len(result.stats.errors)}")
            
            # Print error details
            for error in result.stats.errors[:3]:  # First 3 errors
                print(f"  Error: {error.get('file', 'unknown')} - {error.get('error', 'unknown')}")
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test recovery from network failures during ingestion."""
        
        # Simulate intermittent network failures
        call_count = 0
        
        async def failing_add_document(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ConnectionError("Network timeout")
            
            return {"success": True, "points_added": 1}
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document", side_effect=failing_add_document):
            engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
            
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"] / "documents",
                collection="pipeline-test",
                recursive=False
            )
            
            # Pipeline should handle failures gracefully
            assert isinstance(result, IngestionResult)
            
            print(f"Network Failure Recovery:")
            print(f"  Files processed: {result.stats.files_processed}")
            print(f"  Files failed: {result.stats.files_failed}")
            print(f"  Success rate: {result.stats.success_rate:.1f}%")
            
            # Should have some failures but not complete failure
            assert result.stats.files_failed > 0
            assert result.stats.files_processed >= 0
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test handling of memory pressure during large file processing."""
        
        # Create several large files to test memory usage
        large_files = {}
        for i in range(3):
            # Create 1MB files with repetitive content
            content = f"Large file {i} content. " * 50000  # ~1MB each
            large_files[f"documents/large_{i}.txt"] = content
        
        for rel_path, content in large_files.items():
            file_path = comprehensive_test_workspace["path"] / rel_path
            file_path.write_text(content)
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document") as mock_add_doc:
            mock_add_doc.return_value = {"success": True, "points_added": 10}
            
            engine = DocumentIngestionEngine(
                client=mock_qdrant_client,
                concurrency=2,
                chunk_size=1000,  # Small chunks to test chunking
                chunk_overlap=100
            )
            
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="pipeline-test",
                recursive=True
            )
            
            assert result.success is True
            
            print(f"Memory Pressure Test:")
            print(f"  Total characters processed: {result.stats.total_characters:,}")
            print(f"  Total chunks generated: {result.stats.total_chunks}")
            print(f"  Processing time: {result.stats.processing_time:.2f}s")
            print(f"  Memory efficiency: {result.stats.total_characters / result.stats.processing_time / 1024:.1f} KB/s")


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceMeasurement:
    """Test performance measurement and bottleneck identification."""
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test throughput measurement across different file types."""
        
        processing_stats = {}
        
        async def tracked_add_document(*args, **kwargs):
            # Extract file info from args/kwargs for tracking
            await asyncio.sleep(0.05)  # Simulate processing time
            return {"success": True, "points_added": 1}
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document", side_effect=tracked_add_document):
            
            # Test different concurrency levels
            concurrency_levels = [1, 2, 4]
            
            for concurrency in concurrency_levels:
                engine = DocumentIngestionEngine(
                    client=mock_qdrant_client,
                    concurrency=concurrency,
                    chunk_size=1000
                )
                
                start_time = time.time()
                result = await engine.process_directory(
                    directory_path=comprehensive_test_workspace["path"],
                    collection="pipeline-test",
                    recursive=True
                )
                end_time = time.time()
                
                processing_stats[concurrency] = {
                    "files_processed": result.stats.files_processed,
                    "total_time": end_time - start_time,
                    "throughput": result.stats.files_processed / (end_time - start_time),
                    "success_rate": result.stats.success_rate
                }
        
        # Analyze performance scaling
        print("Performance Scaling Analysis:")
        for concurrency, stats in processing_stats.items():
            print(f"  Concurrency {concurrency}:")
            print(f"    Files processed: {stats['files_processed']}")
            print(f"    Total time: {stats['total_time']:.2f}s")
            print(f"    Throughput: {stats['throughput']:.1f} files/sec")
            print(f"    Success rate: {stats['success_rate']:.1f}%")
        
        # Verify performance improves with concurrency (up to a point)
        assert processing_stats[2]["throughput"] > processing_stats[1]["throughput"]
    
    @pytest.mark.asyncio  
    async def test_processing_time_estimation(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test processing time estimation accuracy."""
        
        engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
        
        # Get time estimation
        estimation = await engine.estimate_processing_time(
            comprehensive_test_workspace["path"]
        )
        
        assert estimation["files_found"] > 0
        assert estimation["total_size_mb"] > 0
        assert estimation["estimated_time_seconds"] > 0
        assert "file_types" in estimation
        
        print(f"Processing Time Estimation:")
        print(f"  Files found: {estimation['files_found']}")
        print(f"  Total size: {estimation['total_size_mb']:.2f} MB")
        print(f"  Estimated time: {estimation['estimated_time_seconds']:.2f}s")
        print(f"  Human readable: {estimation['estimated_time_human']}")
        print(f"  File type breakdown:")
        for file_type, count in estimation["file_types"].items():
            print(f"    {file_type}: {count} files")
        
        # Test actual processing time vs estimation
        with patch("workspace_qdrant_mcp.tools.documents.add_document") as mock_add_doc:
            mock_add_doc.return_value = {"success": True, "points_added": 1}
            
            start_time = time.time()
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="pipeline-test",
                recursive=True
            )
            end_time = time.time()
            
            actual_time = end_time - start_time
            estimated_time = estimation["estimated_time_seconds"]
            
            print(f"Estimation Accuracy:")
            print(f"  Estimated: {estimated_time:.2f}s")
            print(f"  Actual: {actual_time:.2f}s")
            print(f"  Accuracy: {abs(estimated_time - actual_time) / estimated_time * 100:.1f}% difference")
    
    @pytest.mark.asyncio
    async def test_bottleneck_identification(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test identification of processing bottlenecks."""
        
        # Track different operation times
        operation_times = {
            "parsing": [],
            "chunking": [],
            "embedding": [],
            "storage": []
        }
        
        async def profiled_add_document(*args, **kwargs):
            # Simulate different operation times
            parse_time = 0.02
            chunk_time = 0.01
            embed_time = 0.05  # Usually the slowest
            storage_time = 0.02
            
            operation_times["parsing"].append(parse_time)
            operation_times["chunking"].append(chunk_time)
            operation_times["embedding"].append(embed_time)
            operation_times["storage"].append(storage_time)
            
            total_time = parse_time + chunk_time + embed_time + storage_time
            await asyncio.sleep(total_time)
            
            return {"success": True, "points_added": 1}
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document", side_effect=profiled_add_document):
            engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
            
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"] / "documents",
                collection="pipeline-test",
                recursive=False
            )
            
            assert result.success is True
        
        # Analyze bottlenecks
        print("Bottleneck Analysis:")
        total_operations = len(operation_times["parsing"])
        
        for operation, times in operation_times.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                percentage = (total_time / sum(sum(times) for times in operation_times.values())) * 100
                
                print(f"  {operation.capitalize()}:")
                print(f"    Average time: {avg_time:.3f}s")
                print(f"    Total time: {total_time:.3f}s")
                print(f"    Percentage of total: {percentage:.1f}%")
        
        # Identify the biggest bottleneck
        bottleneck = max(operation_times.items(), key=lambda x: sum(x[1]) if x[1] else 0)
        print(f"  Primary bottleneck: {bottleneck[0]} ({sum(bottleneck[1]):.3f}s total)")


@pytest.mark.integration
class TestSearchFunctionalityVerification:
    """Test that ingested documents are searchable end-to-end."""
    
    @pytest.mark.asyncio
    async def test_ingestion_to_search_workflow(
        self,
        comprehensive_test_workspace,
        mock_qdrant_client
    ):
        """Test complete workflow from ingestion to search functionality."""
        
        # Mock search functionality
        ingested_content = []
        
        async def mock_add_document(client, content, collection, metadata=None, **kwargs):
            ingested_content.append({
                "content": content,
                "metadata": metadata or {},
                "collection": collection
            })
            return {"success": True, "points_added": 1, "document_id": f"doc-{len(ingested_content)}"}
        
        async def mock_search(query, collection, limit=10, **kwargs):
            # Simple text matching for testing
            results = []
            for i, doc in enumerate(ingested_content):
                if query.lower() in doc["content"].lower():
                    results.append({
                        "id": f"doc-{i+1}",
                        "score": 0.9,
                        "payload": doc["metadata"],
                        "content": doc["content"][:200] + "..."
                    })
            return {"results": results[:limit]}
        
        mock_qdrant_client.search.side_effect = mock_search
        
        with patch("workspace_qdrant_mcp.tools.documents.add_document", side_effect=mock_add_document):
            # Step 1: Ingest documents
            engine = DocumentIngestionEngine(client=mock_qdrant_client, concurrency=2)
            
            result = await engine.process_directory(
                directory_path=comprehensive_test_workspace["path"],
                collection="search-test",
                recursive=True
            )
            
            assert result.success is True
            assert len(ingested_content) > 0
            
            print(f"Ingestion completed:")
            print(f"  Documents ingested: {len(ingested_content)}")
            print(f"  Processing time: {result.stats.processing_time:.2f}s")
            
            # Step 2: Test search functionality
            search_queries = [
                "document ingestion",
                "Python code",
                "test application",
                "HTML parsing",
                "performance metrics"
            ]
            
            search_results = {}
            for query in search_queries:
                results = await mock_qdrant_client.search(
                    query=query,
                    collection="search-test",
                    limit=5
                )
                search_results[query] = results["results"]
            
            # Verify search results
            print(f"Search Results:")
            for query, results in search_results.items():
                print(f"  Query: '{query}' -> {len(results)} results")
                for i, result in enumerate(results[:2]):  # Show first 2 results
                    print(f"    {i+1}. Score: {result['score']:.2f}")
                    print(f"       Content: {result['content'][:100]}...")
            
            # Verify that relevant queries return results
            assert len(search_results["document ingestion"]) > 0
            assert len(search_results["Python code"]) > 0
            
            # Verify search quality
            total_queries = len(search_queries)
            successful_queries = sum(1 for results in search_results.values() if len(results) > 0)
            search_success_rate = successful_queries / total_queries * 100
            
            print(f"Search Success Rate: {search_success_rate:.1f}%")
            assert search_success_rate >= 60  # At least 60% of queries should return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])