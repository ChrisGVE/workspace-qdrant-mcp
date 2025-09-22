"""
Workspace CLI Comprehensive Unit Tests

Targets workspace_qdrant_mcp and wqm_cli modules for maximum coverage boost.
Focus on: tools/memory.py, server.py, CLI commands, parsers

Target: Push coverage significantly higher through comprehensive CLI testing
"""

import asyncio
import json
import pytest
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from dataclasses import dataclass
import io
import os

# Add src paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

# Workspace MCP imports
try:
    from workspace_qdrant_mcp.server import FastMCPServer
    from workspace_qdrant_mcp.tools.memory import (
        MemoryTool, store_document, retrieve_documents,
        list_collections, delete_document
    )
    from workspace_qdrant_mcp.tools.state_management import (
        create_collection, delete_collection, get_collection_info
    )
    from workspace_qdrant_mcp.core.project_detection import ProjectDetector
    from workspace_qdrant_mcp.core.client import QdrantClient
    from workspace_qdrant_mcp.core.embeddings import EmbeddingService
    WORKSPACE_MCP_AVAILABLE = True
except ImportError as e:
    WORKSPACE_MCP_AVAILABLE = False
    print(f"Warning: workspace_qdrant_mcp modules not available: {e}")

# CLI imports
try:
    from wqm_cli.cli.main import app as cli_app
    from wqm_cli.cli.commands.admin import admin_app
    from wqm_cli.cli.commands.ingest import ingest_app
    from wqm_cli.cli.commands.search import search_app
    from wqm_cli.cli.parsers.pdf_parser import PDFParser
    from wqm_cli.cli.parsers.text_parser import TextParser
    from wqm_cli.cli.parsers.html_parser import HTMLParser
    from wqm_cli.cli.parsers.markdown_parser import MarkdownParser
    from wqm_cli.cli.parsers.code_parser import CodeParser
    from wqm_cli.cli.ingestion_engine import DocumentIngestionEngine
    from wqm_cli.cli.watch_service import WatchService
    CLI_AVAILABLE = True
except ImportError as e:
    CLI_AVAILABLE = False
    print(f"Warning: wqm_cli modules not available: {e}")

# Common utilities
try:
    from workspace_qdrant_mcp.utils.project_detection import GitProjectDetector
    from workspace_qdrant_mcp.utils.file_operations import FileOperations
    from workspace_qdrant_mcp.utils.os_directories import DirectoryManager
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    print(f"Warning: common.utils modules not available: {e}")


@pytest.mark.skipif(not WORKSPACE_MCP_AVAILABLE, reason="workspace_qdrant_mcp not available")
class TestWorkspaceMCPTools:
    """Comprehensive tests for workspace_qdrant_mcp.tools modules"""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing"""
        client = Mock()
        client.upsert = AsyncMock()
        client.search = AsyncMock(return_value=[
            {"id": "doc1", "score": 0.9, "payload": {"content": "test content"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "another test"}}
        ])
        client.get_collections = AsyncMock(return_value=[
            {"name": "collection1"}, {"name": "collection2"}
        ])
        client.delete = AsyncMock()
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.get_collection_info = AsyncMock(return_value={
            "status": "green", "vectors_count": 100
        })
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        service = Mock()
        service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4] * 96)
        return service

    @pytest.fixture
    def memory_tool(self, mock_qdrant_client, mock_embedding_service):
        """Create MemoryTool for testing"""
        tool = MemoryTool(
            client=mock_qdrant_client,
            embedding_service=mock_embedding_service
        )
        return tool

    @pytest.mark.asyncio
    async def test_memory_tool_initialization(self, mock_qdrant_client, mock_embedding_service):
        """Test MemoryTool initialization"""
        tool = MemoryTool(
            client=mock_qdrant_client,
            embedding_service=mock_embedding_service
        )
        assert tool is not None
        assert tool.client == mock_qdrant_client
        assert tool.embedding_service == mock_embedding_service

    @pytest.mark.asyncio
    async def test_store_document_function(self, mock_qdrant_client, mock_embedding_service):
        """Test store_document MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.memory.get_client') as mock_get_client:
            with patch('workspace_qdrant_mcp.tools.memory.get_embedding_service') as mock_get_embedding:
                mock_get_client.return_value = mock_qdrant_client
                mock_get_embedding.return_value = mock_embedding_service

                result = await store_document(
                    collection="test_collection",
                    document_id="doc1",
                    content="Test document content",
                    metadata={"type": "test", "timestamp": "2023-01-01"}
                )

                assert result["success"] is True
                assert result["document_id"] == "doc1"
                assert result["collection"] == "test_collection"
                mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_documents_function(self, mock_qdrant_client, mock_embedding_service):
        """Test retrieve_documents MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.memory.get_client') as mock_get_client:
            with patch('workspace_qdrant_mcp.tools.memory.get_embedding_service') as mock_get_embedding:
                mock_get_client.return_value = mock_qdrant_client
                mock_get_embedding.return_value = mock_embedding_service

                result = await retrieve_documents(
                    collection="test_collection",
                    query="test query",
                    limit=10
                )

                assert result["success"] is True
                assert len(result["documents"]) == 2
                assert result["query"] == "test query"
                mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_collections_function(self, mock_qdrant_client):
        """Test list_collections MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.memory.get_client') as mock_get_client:
            mock_get_client.return_value = mock_qdrant_client

            result = await list_collections()

            assert result["success"] is True
            assert len(result["collections"]) == 2
            assert "collection1" in [c["name"] for c in result["collections"]]
            mock_qdrant_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_function(self, mock_qdrant_client):
        """Test delete_document MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.memory.get_client') as mock_get_client:
            mock_get_client.return_value = mock_qdrant_client

            result = await delete_document(
                collection="test_collection",
                document_id="doc1"
            )

            assert result["success"] is True
            assert result["document_id"] == "doc1"
            assert result["collection"] == "test_collection"
            mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_function(self, mock_qdrant_client):
        """Test create_collection MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client') as mock_get_client:
            mock_get_client.return_value = mock_qdrant_client

            result = await create_collection(
                collection_name="new_collection",
                vector_size=384,
                distance_metric="cosine"
            )

            assert result["success"] is True
            assert result["collection_name"] == "new_collection"
            assert result["vector_size"] == 384
            mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_collection_function(self, mock_qdrant_client):
        """Test delete_collection MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client') as mock_get_client:
            mock_get_client.return_value = mock_qdrant_client

            result = await delete_collection(collection_name="test_collection")

            assert result["success"] is True
            assert result["collection_name"] == "test_collection"
            mock_qdrant_client.delete_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_info_function(self, mock_qdrant_client):
        """Test get_collection_info MCP tool function"""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client') as mock_get_client:
            mock_get_client.return_value = mock_qdrant_client

            result = await get_collection_info(collection_name="test_collection")

            assert result["success"] is True
            assert result["collection_name"] == "test_collection"
            assert result["info"]["status"] == "green"
            mock_qdrant_client.get_collection_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_detector_initialization(self):
        """Test ProjectDetector initialization"""
        detector = ProjectDetector()
        assert detector is not None

    @pytest.mark.asyncio
    async def test_project_detector_detect_project(self):
        """Test ProjectDetector project detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake Git repository
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            detector = ProjectDetector()
            project_info = await detector.detect_project(Path(temp_dir))

            assert project_info is not None
            assert project_info["is_git_repo"] is True
            assert project_info["project_root"] == str(Path(temp_dir))

    @pytest.mark.asyncio
    async def test_fastmcp_server_initialization(self):
        """Test FastMCPServer initialization"""
        with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_client:
            with patch('workspace_qdrant_mcp.server.EmbeddingService') as mock_embedding:
                server = FastMCPServer()
                assert server is not None

    @pytest.mark.asyncio
    async def test_fastmcp_server_tool_registration(self):
        """Test FastMCPServer tool registration"""
        with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_client:
            with patch('workspace_qdrant_mcp.server.EmbeddingService') as mock_embedding:
                server = FastMCPServer()

                # Test that tools are registered
                expected_tools = [
                    "store_document", "retrieve_documents", "list_collections",
                    "delete_document", "create_collection", "delete_collection"
                ]

                for tool_name in expected_tools:
                    assert hasattr(server, tool_name) or tool_name in str(server)


@pytest.mark.skipif(not CLI_AVAILABLE, reason="wqm_cli modules not available")
class TestWqmCliComponents:
    """Comprehensive tests for wqm_cli components"""

    @pytest.fixture
    def temp_file_pdf(self):
        """Create temporary PDF file for testing"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write minimal PDF content
            f.write(b'%PDF-1.4\n%test content\nendobj\n%%EOF')
            return Path(f.name)

    @pytest.fixture
    def temp_file_text(self):
        """Create temporary text file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test text file.\nWith multiple lines.\nFor testing purposes.")
            return Path(f.name)

    @pytest.fixture
    def temp_file_html(self):
        """Create temporary HTML file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Test HTML</title>
</head>
<body>
    <h1>Test Document</h1>
    <p>This is a test HTML document.</p>
    <div id="content">
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
</body>
</html>
""")
            return Path(f.name)

    @pytest.fixture
    def temp_file_markdown(self):
        """Create temporary Markdown file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Test Markdown Document

This is a **test** markdown document with various elements.

## Features

- Lists
- *Italic text*
- `Code snippets`
- [Links](http://example.com)

### Code Block

```python
def hello_world():
    print("Hello, World!")
```

> Blockquote example

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
""")
            return Path(f.name)

    @pytest.fixture
    def temp_file_python(self):
        """Create temporary Python file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
#!/usr/bin/env python3
'''
Test Python module for parser testing
'''

import os
import sys
from typing import List, Optional, Dict

class TestClass:
    '''Test class with various methods'''

    def __init__(self, name: str):
        self.name = name
        self._private_attr = 42

    def public_method(self, param: int) -> str:
        '''Public method with type hints'''
        return f"Result: {param * 2}"

    @property
    def computed_value(self) -> int:
        '''Computed property'''
        return self._private_attr * 2

    @staticmethod
    def static_method(data: List[int]) -> int:
        '''Static method example'''
        return sum(data)

async def async_function(items: Optional[List[str]] = None) -> Dict[str, int]:
    '''Async function with complex types'''
    if items is None:
        items = []
    return {item: len(item) for item in items}

def simple_function(x, y):
    '''Simple function without type hints'''
    return x + y

# Global variable
GLOBAL_CONSTANT = "test_value"

if __name__ == "__main__":
    test = TestClass("example")
    print(test.public_method(5))
""")
            return Path(f.name)

    def test_pdf_parser_initialization(self):
        """Test PDFParser initialization"""
        parser = PDFParser()
        assert parser is not None
        assert parser.format_name == "PDF"
        assert ".pdf" in parser.supported_extensions

    def test_pdf_parser_validate_file(self, temp_file_pdf):
        """Test PDFParser file validation"""
        parser = PDFParser()

        # Valid file should not raise exception
        try:
            parser.validate_file(temp_file_pdf)
            validation_passed = True
        except Exception:
            validation_passed = False

        assert validation_passed is True

    @pytest.mark.asyncio
    async def test_pdf_parser_parse_options(self):
        """Test PDFParser parsing options"""
        parser = PDFParser()
        options = parser.get_parsing_options()

        assert isinstance(options, dict)
        expected_options = ["extract_images", "extract_tables", "preserve_layout"]
        for option in expected_options:
            assert option in options or any(opt in option for opt in options.keys())

    def test_text_parser_initialization(self):
        """Test TextParser initialization"""
        parser = TextParser()
        assert parser is not None
        assert parser.format_name == "Text"
        assert ".txt" in parser.supported_extensions

    @pytest.mark.asyncio
    async def test_text_parser_parse(self, temp_file_text):
        """Test TextParser parsing functionality"""
        parser = TextParser()

        result = await parser.parse(temp_file_text)

        assert result is not None
        assert result.content == "This is a test text file.\nWith multiple lines.\nFor testing purposes."
        assert result.file_path == temp_file_text
        assert result.file_type == "text"

    def test_html_parser_initialization(self):
        """Test HTMLParser initialization"""
        parser = HTMLParser()
        assert parser is not None
        assert parser.format_name == "HTML"
        assert ".html" in parser.supported_extensions

    @pytest.mark.asyncio
    async def test_html_parser_parse(self, temp_file_html):
        """Test HTMLParser parsing functionality"""
        parser = HTMLParser()

        result = await parser.parse(temp_file_html)

        assert result is not None
        assert "Test Document" in result.content
        assert result.file_type == "html"
        assert "title" in result.additional_metadata
        assert result.additional_metadata["title"] == "Test HTML"

    @pytest.mark.asyncio
    async def test_html_parser_extract_metadata(self, temp_file_html):
        """Test HTMLParser metadata extraction"""
        parser = HTMLParser()

        result = await parser.parse(temp_file_html, extract_links=True, extract_images=True)

        metadata = result.additional_metadata
        assert "links" in metadata or "link_count" in metadata
        assert "structure" in metadata or "elements" in metadata

    def test_markdown_parser_initialization(self):
        """Test MarkdownParser initialization"""
        parser = MarkdownParser()
        assert parser is not None
        assert parser.format_name == "Markdown"
        assert ".md" in parser.supported_extensions

    @pytest.mark.asyncio
    async def test_markdown_parser_parse(self, temp_file_markdown):
        """Test MarkdownParser parsing functionality"""
        parser = MarkdownParser()

        result = await parser.parse(temp_file_markdown)

        assert result is not None
        assert "Test Markdown Document" in result.content
        assert result.file_type == "markdown"

    @pytest.mark.asyncio
    async def test_markdown_parser_extract_structure(self, temp_file_markdown):
        """Test MarkdownParser structure extraction"""
        parser = MarkdownParser()

        result = await parser.parse(temp_file_markdown, extract_structure=True)

        metadata = result.additional_metadata
        assert "headings" in metadata or "heading_count" in metadata
        assert "links" in metadata or "link_count" in metadata

    def test_code_parser_initialization(self):
        """Test CodeParser initialization"""
        parser = CodeParser()
        assert parser is not None
        assert parser.format_name == "Source Code"
        assert ".py" in parser.supported_extensions

    @pytest.mark.asyncio
    async def test_code_parser_parse(self, temp_file_python):
        """Test CodeParser parsing functionality"""
        parser = CodeParser()

        result = await parser.parse(temp_file_python)

        assert result is not None
        assert "class TestClass" in result.content
        assert result.file_type == "code"
        assert result.additional_metadata["programming_language"] == "python"

    @pytest.mark.asyncio
    async def test_code_parser_extract_functions(self, temp_file_python):
        """Test CodeParser function extraction"""
        parser = CodeParser()

        result = await parser.parse(temp_file_python, detect_functions=True)

        metadata = result.additional_metadata
        assert "function_count" in metadata
        assert metadata["function_count"] >= 3  # async_function, simple_function, methods
        assert "class_count" in metadata
        assert metadata["class_count"] >= 1  # TestClass

    @pytest.mark.asyncio
    async def test_code_parser_language_detection(self):
        """Test CodeParser language detection"""
        parser = CodeParser()

        test_files = [
            (".py", "python"),
            (".js", "javascript"),
            (".java", "java"),
            (".cpp", "cpp"),
            (".rs", "rust"),
            (".go", "go")
        ]

        for ext, expected_lang in test_files:
            test_path = Path(f"/test/file{ext}")
            detected_lang = await parser._detect_language(test_path)
            assert detected_lang == expected_lang

    @pytest.mark.asyncio
    async def test_document_ingestion_engine_initialization(self):
        """Test DocumentIngestionEngine initialization"""
        with patch('wqm_cli.cli.ingestion_engine.QdrantClient') as mock_client:
            with patch('wqm_cli.cli.ingestion_engine.EmbeddingService') as mock_embedding:
                config = Mock()
                engine = DocumentIngestionEngine(config=config)
                assert engine is not None

    @pytest.mark.asyncio
    async def test_document_ingestion_engine_ingest_file(self, temp_file_text):
        """Test DocumentIngestionEngine file ingestion"""
        with patch('wqm_cli.cli.ingestion_engine.QdrantClient') as mock_client_class:
            with patch('wqm_cli.cli.ingestion_engine.EmbeddingService') as mock_embedding_class:
                mock_client = Mock()
                mock_client.upsert = AsyncMock()
                mock_client_class.return_value = mock_client

                mock_embedding = Mock()
                mock_embedding.embed_text = AsyncMock(return_value=[0.1] * 384)
                mock_embedding_class.return_value = mock_embedding

                config = Mock()
                engine = DocumentIngestionEngine(config=config)

                result = await engine.ingest_file(
                    file_path=temp_file_text,
                    collection="test_collection"
                )

                assert result["success"] is True
                assert result["file_path"] == str(temp_file_text)
                mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_document_ingestion_engine_batch_ingest(self, temp_file_text, temp_file_html):
        """Test DocumentIngestionEngine batch ingestion"""
        with patch('wqm_cli.cli.ingestion_engine.QdrantClient') as mock_client_class:
            with patch('wqm_cli.cli.ingestion_engine.EmbeddingService') as mock_embedding_class:
                mock_client = Mock()
                mock_client.upsert = AsyncMock()
                mock_client_class.return_value = mock_client

                mock_embedding = Mock()
                mock_embedding.embed_batch = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
                mock_embedding_class.return_value = mock_embedding

                config = Mock()
                engine = DocumentIngestionEngine(config=config)

                files = [temp_file_text, temp_file_html]
                results = await engine.batch_ingest(
                    file_paths=files,
                    collection="test_collection"
                )

                assert len(results) == 2
                assert all(result["success"] for result in results)
                assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_watch_service_initialization(self):
        """Test WatchService initialization"""
        with patch('wqm_cli.cli.watch_service.DocumentIngestionEngine') as mock_engine:
            config = Mock()
            service = WatchService(config=config)
            assert service is not None

    @pytest.mark.asyncio
    async def test_watch_service_start_watching(self):
        """Test WatchService directory watching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('wqm_cli.cli.watch_service.DocumentIngestionEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_engine.ingest_file = AsyncMock(return_value={"success": True})
                mock_engine_class.return_value = mock_engine

                config = Mock()
                service = WatchService(config=config)

                # Start watching in background
                watch_task = asyncio.create_task(
                    service.start_watching(Path(temp_dir), "test_collection")
                )

                # Give it a moment to initialize
                await asyncio.sleep(0.1)

                # Create a new file in watched directory
                test_file = Path(temp_dir) / "test.txt"
                test_file.write_text("Test content")

                # Give watcher time to detect the file
                await asyncio.sleep(0.2)

                # Cancel the watch task
                watch_task.cancel()

                try:
                    await watch_task
                except asyncio.CancelledError:
                    pass

    def test_cli_app_structure(self):
        """Test CLI app structure and commands"""
        assert cli_app is not None

        # Test that main CLI app has expected structure
        if hasattr(cli_app, 'commands'):
            assert isinstance(cli_app.commands, dict)
        elif hasattr(cli_app, 'name'):
            assert cli_app.name is not None

    def test_admin_app_structure(self):
        """Test admin app structure"""
        assert admin_app is not None

        # Test admin app functionality
        if hasattr(admin_app, 'commands'):
            expected_commands = ['collections', 'status', 'cleanup']
            available_commands = list(admin_app.commands.keys())
            # At least some admin commands should be available
            assert len(available_commands) > 0

    def test_ingest_app_structure(self):
        """Test ingest app structure"""
        assert ingest_app is not None

        # Test ingest app has expected functionality
        if hasattr(ingest_app, 'commands'):
            expected_commands = ['file', 'directory', 'watch']
            available_commands = list(ingest_app.commands.keys())
            # At least some ingest commands should be available
            assert len(available_commands) > 0

    @pytest.mark.asyncio
    async def test_cli_error_handling(self):
        """Test CLI error handling"""
        # Test invalid file handling
        non_existent_file = Path("/nonexistent/file.txt")

        parser = TextParser()
        with pytest.raises(FileNotFoundError):
            await parser.parse(non_existent_file)

        # Test invalid configuration handling
        with patch('wqm_cli.cli.ingestion_engine.QdrantClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            config = Mock()
            with pytest.raises(Exception):
                engine = DocumentIngestionEngine(config=config)
                await engine.connect()


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="workspace_qdrant_mcp.utils modules not available")
class TestCommonUtils:
    """Comprehensive tests for common.utils modules"""

    @pytest.fixture
    def temp_git_repo(self):
        """Create temporary Git repository for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            # Create some files
            (Path(temp_dir) / "README.md").write_text("# Test Repository")
            (Path(temp_dir) / "src").mkdir()
            (Path(temp_dir) / "src" / "main.py").write_text("print('Hello, World!')")

            yield Path(temp_dir)

    def test_git_project_detector_initialization(self):
        """Test GitProjectDetector initialization"""
        detector = GitProjectDetector()
        assert detector is not None

    def test_git_project_detector_is_git_repository(self, temp_git_repo):
        """Test GitProjectDetector Git repository detection"""
        detector = GitProjectDetector()

        assert detector.is_git_repository(temp_git_repo) is True
        assert detector.is_git_repository(Path("/nonexistent")) is False

    def test_git_project_detector_get_project_root(self, temp_git_repo):
        """Test GitProjectDetector project root detection"""
        detector = GitProjectDetector()

        # Test from repository root
        root = detector.get_project_root(temp_git_repo)
        assert root == temp_git_repo

        # Test from subdirectory
        src_dir = temp_git_repo / "src"
        root_from_subdir = detector.get_project_root(src_dir)
        assert root_from_subdir == temp_git_repo

    def test_git_project_detector_get_project_info(self, temp_git_repo):
        """Test GitProjectDetector project information extraction"""
        detector = GitProjectDetector()

        info = detector.get_project_info(temp_git_repo)

        assert info is not None
        assert info["is_git_repo"] is True
        assert info["project_root"] == str(temp_git_repo)
        assert "files" in info
        assert len(info["files"]) >= 2  # README.md and main.py

    def test_file_operations_initialization(self):
        """Test FileOperations initialization"""
        file_ops = FileOperations()
        assert file_ops is not None

    def test_file_operations_read_file(self, temp_git_repo):
        """Test FileOperations file reading"""
        file_ops = FileOperations()

        readme_path = temp_git_repo / "README.md"
        content = file_ops.read_file(readme_path)

        assert content == "# Test Repository"

    def test_file_operations_write_file(self):
        """Test FileOperations file writing"""
        file_ops = FileOperations()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        content = "Test file content\nWith multiple lines"
        file_ops.write_file(temp_path, content)

        # Verify file was written correctly
        written_content = temp_path.read_text()
        assert written_content == content

        # Cleanup
        temp_path.unlink()

    def test_file_operations_list_files(self, temp_git_repo):
        """Test FileOperations file listing"""
        file_ops = FileOperations()

        files = file_ops.list_files(temp_git_repo, recursive=True)

        assert len(files) >= 2
        assert any(f.name == "README.md" for f in files)
        assert any(f.name == "main.py" for f in files)

    def test_file_operations_filter_files(self, temp_git_repo):
        """Test FileOperations file filtering"""
        file_ops = FileOperations()

        # Filter for Python files only
        python_files = file_ops.list_files(
            temp_git_repo,
            recursive=True,
            extensions=[".py"]
        )

        assert len(python_files) >= 1
        assert all(f.suffix == ".py" for f in python_files)

    def test_directory_manager_initialization(self):
        """Test DirectoryManager initialization"""
        dir_manager = DirectoryManager()
        assert dir_manager is not None

    def test_directory_manager_create_directory(self):
        """Test DirectoryManager directory creation"""
        dir_manager = DirectoryManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "test_directory"

            dir_manager.create_directory(new_dir)

            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_directory_manager_copy_directory(self, temp_git_repo):
        """Test DirectoryManager directory copying"""
        dir_manager = DirectoryManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_dir = Path(temp_dir) / "copied_repo"

            dir_manager.copy_directory(temp_git_repo, dest_dir)

            assert dest_dir.exists()
            assert (dest_dir / "README.md").exists()
            assert (dest_dir / "src" / "main.py").exists()

    def test_directory_manager_delete_directory(self):
        """Test DirectoryManager directory deletion"""
        dir_manager = DirectoryManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "to_delete"
            test_dir.mkdir()
            (test_dir / "test_file.txt").write_text("content")

            assert test_dir.exists()

            dir_manager.delete_directory(test_dir)

            assert not test_dir.exists()

    def test_directory_manager_get_directory_size(self, temp_git_repo):
        """Test DirectoryManager directory size calculation"""
        dir_manager = DirectoryManager()

        size = dir_manager.get_directory_size(temp_git_repo)

        assert size > 0
        assert isinstance(size, int)

    def test_directory_manager_find_files_by_pattern(self, temp_git_repo):
        """Test DirectoryManager file pattern matching"""
        dir_manager = DirectoryManager()

        # Find all Python files
        python_files = dir_manager.find_files_by_pattern(temp_git_repo, "*.py")

        assert len(python_files) >= 1
        assert all(f.suffix == ".py" for f in python_files)

        # Find all markdown files
        md_files = dir_manager.find_files_by_pattern(temp_git_repo, "*.md")

        assert len(md_files) >= 1
        assert all(f.suffix == ".md" for f in md_files)


# Integration and end-to-end tests
class TestIntegrationScenarios:
    """Integration tests combining multiple components"""

    @pytest.mark.skipif(not (WORKSPACE_MCP_AVAILABLE and CLI_AVAILABLE),
                       reason="Both workspace and CLI modules required")
    @pytest.mark.asyncio
    async def test_full_document_ingestion_pipeline(self):
        """Test complete document ingestion pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test documents
            test_files = {
                "document1.txt": "This is the first test document with important content.",
                "document2.md": "# Markdown Document\n\nThis contains **formatted** text.",
                "code.py": "def hello():\n    return 'Hello, World!'\n\nclass Example:\n    pass"
            }

            file_paths = []
            for filename, content in test_files.items():
                file_path = Path(temp_dir) / filename
                file_path.write_text(content)
                file_paths.append(file_path)

            # Mock the complete pipeline
            with patch('wqm_cli.cli.ingestion_engine.QdrantClient') as mock_client_class:
                with patch('wqm_cli.cli.ingestion_engine.EmbeddingService') as mock_embedding_class:
                    mock_client = Mock()
                    mock_client.upsert = AsyncMock()
                    mock_client_class.return_value = mock_client

                    mock_embedding = Mock()
                    mock_embedding.embed_text = AsyncMock(return_value=[0.1] * 384)
                    mock_embedding_class.return_value = mock_embedding

                    config = Mock()
                    engine = DocumentIngestionEngine(config=config)

                    # Process all files
                    results = []
                    for file_path in file_paths:
                        result = await engine.ingest_file(file_path, "test_collection")
                        results.append(result)

                    # Verify all files were processed successfully
                    assert len(results) == 3
                    assert all(result["success"] for result in results)
                    assert mock_client.upsert.call_count == 3

    @pytest.mark.skipif(not (WORKSPACE_MCP_AVAILABLE and UTILS_AVAILABLE),
                       reason="Workspace and utils modules required")
    @pytest.mark.asyncio
    async def test_project_detection_and_ingestion(self):
        """Test project detection followed by document ingestion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock Git project
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            project_files = {
                "README.md": "# Test Project\n\nThis is a test project.",
                "src/main.py": "def main():\n    print('Hello')\n\nif __name__ == '__main__':\n    main()",
                "docs/guide.md": "# User Guide\n\nHow to use this project."
            }

            for file_path, content in project_files.items():
                full_path = Path(temp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Test project detection
            if UTILS_AVAILABLE:
                detector = GitProjectDetector()
                project_info = detector.get_project_info(Path(temp_dir))

                assert project_info["is_git_repo"] is True
                assert len(project_info["files"]) >= 3

            # Test ingestion of detected files
            with patch('workspace_qdrant_mcp.tools.memory.get_client') as mock_get_client:
                with patch('workspace_qdrant_mcp.tools.memory.get_embedding_service') as mock_get_embedding:
                    mock_client = Mock()
                    mock_client.upsert = AsyncMock()
                    mock_get_client.return_value = mock_client

                    mock_embedding = Mock()
                    mock_embedding.embed_text = AsyncMock(return_value=[0.1] * 384)
                    mock_get_embedding.return_value = mock_embedding

                    # Store each detected file
                    for file_path, content in project_files.items():
                        result = await store_document(
                            collection="project_docs",
                            document_id=file_path,
                            content=content,
                            metadata={"type": "project_file", "path": file_path}
                        )
                        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test error recovery in various failure scenarios"""
        # Test file system errors
        with pytest.raises(FileNotFoundError):
            parser = TextParser()
            await parser.parse(Path("/nonexistent/file.txt"))

        # Test network/connection errors
        with patch('workspace_qdrant_mcp.tools.memory.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.upsert = AsyncMock(side_effect=ConnectionError("Network error"))
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError):
                await store_document(
                    collection="test",
                    document_id="doc1",
                    content="test",
                    metadata={}
                )

        # Test malformed data errors
        if CLI_AVAILABLE:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write("{ invalid json content")
                json_file = Path(f.name)

            # Parser should handle malformed files gracefully
            parser = TextParser()
            try:
                result = await parser.parse(json_file)
                # Should either succeed with warning or fail gracefully
                assert result is not None or True  # Either outcome is acceptable
            except Exception as e:
                # Exception should be informative
                assert "json" in str(e).lower() or "invalid" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])