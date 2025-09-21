"""
Unit tests for context injection workflows and document processing.

Tests the comprehensive document processing pipeline including file parsing,
content extraction, embedding generation, and context injection for LLM workflows.
Validates integration with project detection and workspace management components.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.core.config import Config, EmbeddingConfig
from common.core.embeddings import EmbeddingService
from wqm_cli.cli.parsers.base import DocumentParser, ParsedDocument
from wqm_cli.cli.parsers.text_parser import TextParser
from wqm_cli.cli.parsers.markdown_parser import MarkdownParser
from wqm_cli.cli.parsers.pdf_parser import PDFParser
from wqm_cli.cli.parsers.html_parser import HtmlParser
from wqm_cli.cli.parsers.file_detector import detect_file_type, FileDetector
from wqm_cli.cli.parsers.exceptions import ParsingError


class MockContextInjector:
    """Mock context injector for testing document processing workflows."""

    def __init__(self, embedding_service: EmbeddingService, project_detector=None):
        self.embedding_service = embedding_service
        self.project_detector = project_detector
        self.processed_documents = []
        self.injection_rules = {}
        self.workspace_context = {}

    async def process_document(
        self,
        file_path: str,
        content: str = None,
        metadata: Dict[str, Any] = None
    ) -> ParsedDocument:
        """Process a document through the full context injection pipeline."""
        if content is None:
            # Auto-detect and parse file
            parser = await self._detect_parser(file_path)
            document = await parser.parse(file_path)
        else:
            # Create document from provided content
            document = ParsedDocument.create(
                content=content,
                file_path=file_path,
                file_type="text",
                additional_metadata=metadata or {}
            )

        # Generate embeddings
        embeddings = await self.embedding_service.generate_embeddings(
            document.content,
            include_sparse=True
        )

        # Add embeddings to document
        document.dense_vector = embeddings.get("dense", [])
        document.sparse_vector = embeddings.get("sparse", {})

        # Apply injection rules
        document = await self._apply_injection_rules(document)

        # Store processed document
        self.processed_documents.append(document)

        return document

    async def process_documents_batch(
        self,
        file_paths: List[str],
        batch_size: int = 10
    ) -> List[ParsedDocument]:
        """Process multiple documents in batches."""
        results = []

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.process_document(path) for path in batch]
            )
            results.extend(batch_results)

        return results

    async def inject_llm_rules(self, document: ParsedDocument, rules: Dict[str, Any]):
        """Inject LLM-specific rules and context into document."""
        self.injection_rules[document.file_path] = rules

        # Add rule metadata to document
        if hasattr(document, 'llm_context'):
            document.llm_context.update(rules)
        else:
            document.llm_context = rules.copy()

        return document

    async def extract_workspace_context(self, project_root: str) -> Dict[str, Any]:
        """Extract workspace context for LLM injection."""
        if self.project_detector:
            project_info = await self.project_detector.detect_project(project_root)
        else:
            project_info = {"type": "unknown", "root": project_root}

        context = {
            "project": project_info,
            "workspace_root": project_root,
            "processed_files": len(self.processed_documents),
            "injection_rules": len(self.injection_rules)
        }

        self.workspace_context = context
        return context

    async def _detect_parser(self, file_path: str) -> DocumentParser:
        """Detect appropriate parser for file."""
        file_type, parser_type, confidence = detect_file_type(file_path)

        parser_map = {
            "text": TextParser(),
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "html": HtmlParser()
        }

        return parser_map.get(parser_type, TextParser())

    async def _apply_injection_rules(self, document: ParsedDocument) -> ParsedDocument:
        """Apply context injection rules to document."""
        rules = self.injection_rules.get(document.file_path, {})

        if "content_filter" in rules:
            # Apply content filtering
            filter_rules = rules["content_filter"]
            if filter_rules.get("remove_comments", False):
                # Remove comment lines (simplified)
                lines = document.content.split('\n')
                lines = [line for line in lines if not line.strip().startswith('#')]
                document.content = '\n'.join(lines)

        if "metadata_enrichment" in rules:
            # Enrich metadata based on rules
            enrichment = rules["metadata_enrichment"]
            document.metadata.update(enrichment)

        return document


class TestContextInjectionWorkflows:
    """Test context injection workflows and document processing pipelines."""

    @pytest.fixture
    def embedding_config(self):
        """Create embedding configuration for testing."""
        return EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            enable_sparse_vectors=True,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=50,
        )

    @pytest.fixture
    def config(self, embedding_config):
        """Create full config with embedding settings."""
        config = Config()
        config.embedding = embedding_config
        return config

    @pytest.fixture
    async def embedding_service(self, config):
        """Create and initialize embedding service."""
        service = EmbeddingService(config)

        # Mock the embedding models
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()
        service.initialized = True

        # Mock embedding generation
        service.dense_model.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        service.bm25_encoder.encode.return_value = {
            "indices": [1, 3, 5],
            "values": [0.8, 0.6, 0.9]
        }

        return service

    @pytest.fixture
    def mock_project_detector(self):
        """Create mock project detector."""
        detector = MagicMock()
        detector.detect_project = AsyncMock(return_value={
            "type": "python",
            "root": "/test/project",
            "name": "test-project",
            "submodules": []
        })
        return detector

    @pytest.fixture
    async def context_injector(self, embedding_service, mock_project_detector):
        """Create context injector for testing."""
        return MockContextInjector(embedding_service, mock_project_detector)

    @pytest.fixture
    def temp_files(self):
        """Create temporary test files with various formats."""
        temp_dir = tempfile.mkdtemp()
        files = {}

        # Text file
        text_file = Path(temp_dir) / "test.txt"
        text_file.write_text("This is a test text file.\nIt has multiple lines.\nFor testing purposes.")
        files["text"] = str(text_file)

        # Markdown file
        md_file = Path(temp_dir) / "test.md"
        md_file.write_text("# Test Document\n\nThis is **bold** text.\n\n## Section\n\n- Item 1\n- Item 2")
        files["markdown"] = str(md_file)

        # Python file
        py_file = Path(temp_dir) / "test.py"
        py_file.write_text("# Python comment\ndef hello():\n    print('Hello world')\n    return 42")
        files["python"] = str(py_file)

        # HTML file
        html_file = Path(temp_dir) / "test.html"
        html_file.write_text("<html><head><title>Test</title></head><body><h1>Test Page</h1><p>Content</p></body></html>")
        files["html"] = str(html_file)

        yield files

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_single_document_processing(self, context_injector, temp_files):
        """Test processing of a single document through the pipeline."""
        text_file = temp_files["text"]

        document = await context_injector.process_document(text_file)

        assert isinstance(document, ParsedDocument)
        assert document.file_path == text_file
        assert document.file_type == "text"
        assert "test text file" in document.content
        assert hasattr(document, 'dense_vector')
        assert hasattr(document, 'sparse_vector')
        assert len(document.dense_vector) == 4  # Mock embedding size
        assert "indices" in document.sparse_vector
        assert len(context_injector.processed_documents) == 1

    @pytest.mark.asyncio
    async def test_batch_document_processing(self, context_injector, temp_files):
        """Test batch processing of multiple documents."""
        file_paths = list(temp_files.values())

        documents = await context_injector.process_documents_batch(file_paths, batch_size=2)

        assert len(documents) == 4
        assert all(isinstance(doc, ParsedDocument) for doc in documents)
        assert all(hasattr(doc, 'dense_vector') for doc in documents)
        assert all(hasattr(doc, 'sparse_vector') for doc in documents)
        assert len(context_injector.processed_documents) == 4

        # Verify different file types were processed
        file_types = {doc.file_type for doc in documents}
        assert "text" in file_types
        assert "markdown" in file_types

    @pytest.mark.asyncio
    async def test_content_provided_processing(self, context_injector):
        """Test processing when content is provided directly."""
        content = "Direct content for processing without file reading."
        metadata = {"source": "manual", "priority": "high"}

        document = await context_injector.process_document(
            "/virtual/path.txt",
            content=content,
            metadata=metadata
        )

        assert document.content == content
        assert document.metadata["source"] == "manual"
        assert document.metadata["priority"] == "high"
        assert hasattr(document, 'dense_vector')

    @pytest.mark.asyncio
    async def test_embedding_generation_integration(self, context_injector, temp_files):
        """Test integration with embedding generation."""
        text_file = temp_files["text"]

        with patch.object(context_injector.embedding_service, 'generate_embeddings') as mock_generate:
            mock_generate.return_value = {
                "dense": [0.5, 0.6, 0.7, 0.8],
                "sparse": {"indices": [2, 4, 6], "values": [0.9, 0.8, 0.7]}
            }

            document = await context_injector.process_document(text_file)

            mock_generate.assert_called_once()
            assert document.dense_vector == [0.5, 0.6, 0.7, 0.8]
            assert document.sparse_vector == {"indices": [2, 4, 6], "values": [0.9, 0.8, 0.7]}

    @pytest.mark.asyncio
    async def test_llm_rule_injection(self, context_injector, temp_files):
        """Test LLM rule injection into documents."""
        text_file = temp_files["text"]

        # Process document first
        document = await context_injector.process_document(text_file)

        # Inject LLM rules
        rules = {
            "role": "assistant",
            "context_type": "documentation",
            "priority": "high",
            "metadata_enrichment": {"category": "test", "importance": 5}
        }

        enhanced_document = await context_injector.inject_llm_rules(document, rules)

        assert hasattr(enhanced_document, 'llm_context')
        assert enhanced_document.llm_context["role"] == "assistant"
        assert enhanced_document.llm_context["context_type"] == "documentation"
        assert document.file_path in context_injector.injection_rules

    @pytest.mark.asyncio
    async def test_workspace_context_extraction(self, context_injector, mock_project_detector):
        """Test extraction of workspace context for LLM injection."""
        project_root = "/test/project"

        # Process some documents first
        await context_injector.process_document("/test/file1.txt", content="Content 1")
        await context_injector.process_document("/test/file2.txt", content="Content 2")

        context = await context_injector.extract_workspace_context(project_root)

        assert context["workspace_root"] == project_root
        assert context["processed_files"] == 2
        assert context["project"]["type"] == "python"
        assert context["project"]["name"] == "test-project"
        mock_project_detector.detect_project.assert_called_once_with(project_root)

    @pytest.mark.asyncio
    async def test_content_filtering_rules(self, context_injector, temp_files):
        """Test content filtering through injection rules."""
        py_file = temp_files["python"]

        # Set up content filtering rules
        rules = {
            "content_filter": {
                "remove_comments": True
            }
        }

        document = await context_injector.process_document(py_file)
        await context_injector.inject_llm_rules(document, rules)

        # Apply rules by reprocessing
        document = await context_injector._apply_injection_rules(document)

        # Comments should be removed
        assert "# Python comment" not in document.content
        assert "def hello():" in document.content
        assert "print('Hello world')" in document.content

    @pytest.mark.asyncio
    async def test_metadata_enrichment_rules(self, context_injector, temp_files):
        """Test metadata enrichment through injection rules."""
        text_file = temp_files["text"]

        rules = {
            "metadata_enrichment": {
                "category": "documentation",
                "language": "english",
                "priority": 10,
                "tags": ["test", "example"]
            }
        }

        document = await context_injector.process_document(text_file)
        await context_injector.inject_llm_rules(document, rules)
        document = await context_injector._apply_injection_rules(document)

        assert document.metadata["category"] == "documentation"
        assert document.metadata["language"] == "english"
        assert document.metadata["priority"] == 10
        assert document.metadata["tags"] == ["test", "example"]

    @pytest.mark.asyncio
    async def test_parser_detection_and_integration(self, context_injector, temp_files):
        """Test automatic parser detection and integration."""
        # Test different file types get correct parsers
        md_file = temp_files["markdown"]
        html_file = temp_files["html"]

        md_doc = await context_injector.process_document(md_file)
        html_doc = await context_injector.process_document(html_file)

        # Markdown should be processed as markdown
        assert md_doc.file_type == "markdown"
        assert "Test Document" in md_doc.content

        # HTML should be processed as HTML
        assert html_doc.file_type == "html"
        assert "Test Page" in html_doc.content
        assert "<html>" not in html_doc.content  # HTML tags should be stripped

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_files(self, context_injector):
        """Test error handling for unsupported file types."""
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            f.write(b"Unknown binary content")
            f.flush()

            # Should fallback to text parser but may have issues
            try:
                document = await context_injector.process_document(f.name)
                # If it succeeds, it should use TextParser as fallback
                assert document.file_type == "text"
            except Exception as e:
                # Or it might fail, which is also acceptable for unknown formats
                assert isinstance(e, (UnicodeDecodeError, ParsingError))
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_missing_files(self, context_injector):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            await context_injector.process_document("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_memory_management_large_batch(self, context_injector):
        """Test memory management during large batch processing."""
        # Create many temporary files
        temp_dir = tempfile.mkdtemp()
        try:
            file_paths = []
            for i in range(50):  # Create 50 small files
                file_path = Path(temp_dir) / f"file_{i}.txt"
                file_path.write_text(f"Content for file {i}\nLine 2\nLine 3")
                file_paths.append(str(file_path))

            # Process in small batches
            documents = await context_injector.process_documents_batch(
                file_paths,
                batch_size=5
            )

            assert len(documents) == 50
            assert len(context_injector.processed_documents) == 50

            # Verify all documents have embeddings
            for doc in documents:
                assert hasattr(doc, 'dense_vector')
                assert hasattr(doc, 'sparse_vector')
                assert len(doc.dense_vector) > 0

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, context_injector, temp_files):
        """Test concurrent document processing."""
        file_paths = list(temp_files.values())

        # Process files concurrently
        tasks = [
            context_injector.process_document(path)
            for path in file_paths
        ]

        documents = await asyncio.gather(*tasks)

        assert len(documents) == 4
        assert all(isinstance(doc, ParsedDocument) for doc in documents)

        # Check that all files were processed
        processed_paths = {doc.file_path for doc in documents}
        expected_paths = set(file_paths)
        assert processed_paths == expected_paths

    @pytest.mark.asyncio
    async def test_embedding_service_failure_handling(self, context_injector, temp_files):
        """Test handling of embedding service failures."""
        text_file = temp_files["text"]

        with patch.object(context_injector.embedding_service, 'generate_embeddings') as mock_generate:
            mock_generate.side_effect = RuntimeError("Embedding generation failed")

            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                await context_injector.process_document(text_file)

    @pytest.mark.asyncio
    async def test_workspace_context_without_project_detector(self, embedding_service):
        """Test workspace context extraction without project detector."""
        injector = MockContextInjector(embedding_service, project_detector=None)

        context = await injector.extract_workspace_context("/test/project")

        assert context["project"]["type"] == "unknown"
        assert context["project"]["root"] == "/test/project"
        assert context["workspace_root"] == "/test/project"

    def test_context_injector_initialization(self, embedding_service, mock_project_detector):
        """Test proper initialization of context injector."""
        injector = MockContextInjector(embedding_service, mock_project_detector)

        assert injector.embedding_service == embedding_service
        assert injector.project_detector == mock_project_detector
        assert injector.processed_documents == []
        assert injector.injection_rules == {}
        assert injector.workspace_context == {}

    @pytest.mark.asyncio
    async def test_text_chunking_integration(self, context_injector):
        """Test integration with text chunking for large documents."""
        # Create a large document that requires chunking
        large_content = "This is a test sentence. " * 200  # ~5000 characters

        document = await context_injector.process_document(
            "/large/document.txt",
            content=large_content
        )

        # Should still process successfully
        assert document.content == large_content
        assert hasattr(document, 'dense_vector')
        assert len(document.dense_vector) > 0

    @pytest.mark.asyncio
    async def test_sparse_vector_integration(self, context_injector, temp_files):
        """Test integration with sparse vector generation."""
        text_file = temp_files["text"]

        # Verify sparse vector generation is enabled
        assert context_injector.embedding_service.config.embedding.enable_sparse_vectors

        document = await context_injector.process_document(text_file)

        assert hasattr(document, 'sparse_vector')
        assert isinstance(document.sparse_vector, dict)
        assert "indices" in document.sparse_vector
        assert "values" in document.sparse_vector
        assert len(document.sparse_vector["indices"]) == len(document.sparse_vector["values"])


class TestContextInjectionErrorHandling:
    """Test error handling in context injection workflows."""

    @pytest.fixture
    async def embedding_service_mock(self):
        """Create mock embedding service for error testing."""
        service = MagicMock()
        service.config = MagicMock()
        service.config.embedding.enable_sparse_vectors = True
        return service

    @pytest.mark.asyncio
    async def test_parser_initialization_failure(self, embedding_service_mock):
        """Test handling of parser initialization failures."""
        injector = MockContextInjector(embedding_service_mock)

        with patch('workspace_qdrant_mcp.cli.parsers.file_detector.detect_file_type') as mock_detect:
            mock_detect.side_effect = Exception("Parser detection failed")

            with pytest.raises(Exception, match="Parser detection failed"):
                await injector.process_document("/test/file.txt")

    @pytest.mark.asyncio
    async def test_document_creation_failure(self, embedding_service_mock):
        """Test handling of document creation failures."""
        injector = MockContextInjector(embedding_service_mock)

        with patch('workspace_qdrant_mcp.cli.parsers.base.ParsedDocument.create') as mock_create:
            mock_create.side_effect = ValueError("Document creation failed")

            with pytest.raises(ValueError, match="Document creation failed"):
                await injector.process_document("/test/file.txt", content="test")

    @pytest.mark.asyncio
    async def test_injection_rule_application_failure(self, embedding_service_mock):
        """Test handling of injection rule application failures."""
        injector = MockContextInjector(embedding_service_mock)
        injector.embedding_service.generate_embeddings = AsyncMock(return_value={
            "dense": [0.1, 0.2], "sparse": {}
        })

        # Mock rule application to fail
        with patch.object(injector, '_apply_injection_rules') as mock_apply:
            mock_apply.side_effect = RuntimeError("Rule application failed")

            with pytest.raises(RuntimeError, match="Rule application failed"):
                await injector.process_document("/test/file.txt", content="test")


class TestContextInjectionPerformance:
    """Test performance characteristics of context injection workflows."""

    @pytest.fixture
    async def performance_embedding_service(self, config):
        """Create embedding service with performance monitoring."""
        service = EmbeddingService(config)

        # Mock with timing simulation
        async def mock_generate_embeddings(text, include_sparse=True):
            await asyncio.sleep(0.01)  # Simulate processing time
            return {
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 2, 3], "values": [0.8, 0.7, 0.6]}
            }

        service.generate_embeddings = mock_generate_embeddings
        service.initialized = True

        return service

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, performance_embedding_service):
        """Test performance of batch processing."""
        injector = MockContextInjector(performance_embedding_service)

        # Create multiple test files
        temp_dir = tempfile.mkdtemp()
        try:
            file_paths = []
            for i in range(10):
                file_path = Path(temp_dir) / f"file_{i}.txt"
                file_path.write_text(f"Test content {i}")
                file_paths.append(str(file_path))

            start_time = asyncio.get_event_loop().time()

            documents = await injector.process_documents_batch(file_paths, batch_size=3)

            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            assert len(documents) == 10
            assert processing_time < 2.0  # Should complete within 2 seconds

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, performance_embedding_service):
        """Test memory usage during document processing."""
        import psutil
        import os

        injector = MockContextInjector(performance_embedding_service)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process multiple documents
        for i in range(20):
            await injector.process_document(
                f"/test/file_{i}.txt",
                content=f"Test content {i} " * 100  # ~1000 chars each
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 20 small docs)
        assert memory_increase < 50 * 1024 * 1024  # 50MB limit

    @pytest.mark.asyncio
    async def test_concurrent_processing_scalability(self, performance_embedding_service):
        """Test scalability of concurrent processing."""
        injector = MockContextInjector(performance_embedding_service)

        # Test different levels of concurrency
        concurrency_levels = [1, 5, 10]

        for concurrency in concurrency_levels:
            start_time = asyncio.get_event_loop().time()

            # Create concurrent tasks
            tasks = [
                injector.process_document(
                    f"/test/file_{i}.txt",
                    content=f"Content {i}"
                ) for i in range(concurrency)
            ]

            documents = await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            assert len(documents) == concurrency
            # Higher concurrency shouldn't take proportionally longer due to async processing
            if concurrency > 1:
                assert processing_time < concurrency * 0.02  # Less than 20ms per doc