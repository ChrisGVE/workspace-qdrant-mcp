"""
Unit tests for document processing workflow integration.

Tests the end-to-end document processing pipeline including file detection,
parser selection, content extraction, embedding generation, and integration
with workspace management and project detection systems.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

from common.core.config import Config, EmbeddingConfig
from common.core.embeddings import EmbeddingService
from wqm_cli.cli.parsers.base import DocumentParser, ParsedDocument
from wqm_cli.cli.parsers.text_parser import TextParser
from wqm_cli.cli.parsers.markdown_parser import MarkdownParser
from wqm_cli.cli.parsers.pdf_parser import PDFParser
from wqm_cli.cli.parsers.html_parser import HtmlParser
from wqm_cli.cli.parsers.docx_parser import DocxParser
from wqm_cli.cli.parsers.file_detector import detect_file_type, FileDetector
from wqm_cli.cli.parsers.exceptions import ParsingError, handle_parsing_error
from common.utils.project_detection import ProjectDetector


class DocumentProcessingWorkflow:
    """Complete document processing workflow for testing."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        project_detector: Optional[ProjectDetector] = None
    ):
        self.embedding_service = embedding_service
        self.project_detector = project_detector
        self.file_detector = FileDetector()
        self.parser_registry = self._initialize_parsers()
        self.processing_stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "total_time": 0.0
        }

    def _initialize_parsers(self) -> Dict[str, DocumentParser]:
        """Initialize all available document parsers."""
        return {
            "text": TextParser(),
            "markdown": MarkdownParser(),
            "pdf": PDFParser(),
            "html": HtmlParser(),
            "docx": DocxParser()
        }

    async def process_document(
        self,
        file_path: str,
        **processing_options
    ) -> ParsedDocument:
        """Process a single document through the complete workflow."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: File type detection
            file_type, parser_type, confidence = detect_file_type(file_path)

            # Step 2: Parser selection
            parser = self._select_parser(parser_type, file_path)

            # Step 3: Document parsing
            document = await parser.parse(file_path, **processing_options)

            # Step 4: Content preprocessing
            document = await self._preprocess_content(document)

            # Step 5: Embedding generation
            document = await self._generate_embeddings(document)

            # Step 6: Metadata enrichment
            document = await self._enrich_metadata(document, file_type, confidence)

            # Step 7: Workspace context integration
            if self.project_detector:
                document = await self._integrate_workspace_context(document, file_path)

            self.processing_stats["processed"] += 1

        except Exception as e:
            self.processing_stats["failed"] += 1
            raise

        finally:
            end_time = asyncio.get_event_loop().time()
            self.processing_stats["total_time"] += end_time - start_time

        return document

    async def process_directory(
        self,
        directory_path: str,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        batch_size: int = 10
    ) -> List[ParsedDocument]:
        """Process all documents in a directory."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        # Collect files based on patterns
        files = self._collect_files(directory, file_patterns, exclude_patterns)

        # Process in batches
        results = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)

        return results

    def _select_parser(self, parser_type: str, file_path: str) -> DocumentParser:
        """Select appropriate parser based on file type."""
        parser = self.parser_registry.get(parser_type)

        if not parser:
            # Fallback to text parser for unknown types
            parser = self.parser_registry["text"]

        # Verify parser can handle the file
        if not parser.can_parse(file_path):
            raise ValueError(f"No suitable parser found for {file_path}")

        return parser

    async def _preprocess_content(self, document: ParsedDocument) -> ParsedDocument:
        """Preprocess document content for better embedding quality."""
        # Text cleaning and normalization
        content = document.content

        # Remove excessive whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # Update content
        document.content = content.strip()

        # Update metadata with preprocessing info
        document.metadata["preprocessed"] = True
        document.metadata["original_length"] = len(document.content)

        return document

    async def _generate_embeddings(self, document: ParsedDocument) -> ParsedDocument:
        """Generate embeddings for document content."""
        if not self.embedding_service.initialized:
            raise RuntimeError("Embedding service not initialized")

        # Check if content needs chunking
        chunk_size = self.embedding_service.config.embedding.chunk_size
        if len(document.content) > chunk_size:
            # Process document in chunks
            chunks = self.embedding_service.chunk_text(document.content)

            # Generate embeddings for all chunks
            chunk_embeddings = await self.embedding_service.generate_embeddings_batch(
                chunks,
                include_sparse=self.embedding_service.config.embedding.enable_sparse_vectors
            )

            # Combine chunk embeddings (simple averaging for dense, union for sparse)
            document.dense_vector = self._combine_dense_embeddings(
                [emb["dense"] for emb in chunk_embeddings]
            )

            if "sparse" in chunk_embeddings[0]:
                document.sparse_vector = self._combine_sparse_embeddings(
                    [emb["sparse"] for emb in chunk_embeddings]
                )

            document.metadata["chunked"] = True
            document.metadata["chunk_count"] = len(chunks)

        else:
            # Single embedding for small documents
            embeddings = await self.embedding_service.generate_embeddings(
                document.content,
                include_sparse=self.embedding_service.config.embedding.enable_sparse_vectors
            )

            document.dense_vector = embeddings["dense"]
            if "sparse" in embeddings:
                document.sparse_vector = embeddings["sparse"]

            document.metadata["chunked"] = False

        return document

    def _combine_dense_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Combine multiple dense embeddings by averaging."""
        if not embeddings:
            return []

        # Average all embeddings
        vector_size = len(embeddings[0])
        combined = [0.0] * vector_size

        for embedding in embeddings:
            for i, value in enumerate(embedding):
                combined[i] += value

        # Average
        for i in range(vector_size):
            combined[i] /= len(embeddings)

        return combined

    def _combine_sparse_embeddings(self, embeddings: List[Dict]) -> Dict:
        """Combine multiple sparse embeddings by union."""
        if not embeddings:
            return {"indices": [], "values": []}

        # Collect all indices and values
        index_value_map = {}

        for embedding in embeddings:
            indices = embedding.get("indices", [])
            values = embedding.get("values", [])

            for idx, val in zip(indices, values):
                if idx in index_value_map:
                    # Sum values for same index
                    index_value_map[idx] += val
                else:
                    index_value_map[idx] = val

        # Convert back to sparse format
        sorted_items = sorted(index_value_map.items())
        return {
            "indices": [idx for idx, _ in sorted_items],
            "values": [val for _, val in sorted_items]
        }

    async def _enrich_metadata(
        self,
        document: ParsedDocument,
        file_type: str,
        detection_confidence: float
    ) -> ParsedDocument:
        """Enrich document metadata with additional information."""
        # Add detection information
        document.metadata["detected_file_type"] = file_type
        document.metadata["detection_confidence"] = detection_confidence

        # Add embedding metadata
        document.metadata["embedding_model"] = self.embedding_service.config.embedding.model
        document.metadata["has_dense_vector"] = hasattr(document, "dense_vector")
        document.metadata["has_sparse_vector"] = hasattr(document, "sparse_vector")

        # Content analysis
        content = document.content
        document.metadata["word_count"] = len(content.split())
        document.metadata["character_count"] = len(content)
        document.metadata["line_count"] = content.count('\n') + 1

        # Language detection (simplified)
        document.metadata["estimated_language"] = self._detect_language(content)

        return document

    def _detect_language(self, content: str) -> str:
        """Simple language detection based on common words."""
        content_lower = content.lower()

        # English indicators
        english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for"]
        english_score = sum(1 for word in english_words if word in content_lower)

        # Python code indicators
        python_words = ["def", "class", "import", "from", "if", "else", "for", "while"]
        python_score = sum(1 for word in python_words if word in content_lower)

        if python_score > english_score:
            return "python"
        elif english_score > 0:
            return "english"
        else:
            return "unknown"

    async def _integrate_workspace_context(
        self,
        document: ParsedDocument,
        file_path: str
    ) -> ParsedDocument:
        """Integrate workspace and project context."""
        try:
            # Get project information
            project_info = await self.project_detector.detect_project(file_path)

            document.metadata["project_root"] = project_info.get("root")
            document.metadata["project_name"] = project_info.get("name")
            document.metadata["project_type"] = project_info.get("type")

            # Calculate relative path from project root
            if project_info.get("root"):
                try:
                    rel_path = Path(file_path).relative_to(Path(project_info["root"]))
                    document.metadata["relative_path"] = str(rel_path)
                except ValueError:
                    # File is outside project root
                    document.metadata["relative_path"] = file_path

        except Exception as e:
            # Don't fail document processing due to project detection issues
            document.metadata["project_detection_error"] = str(e)

        return document

    def _collect_files(
        self,
        directory: Path,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> List[str]:
        """Collect files from directory based on patterns."""
        import fnmatch

        files = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            # Check include patterns
            if file_patterns:
                if not any(fnmatch.fnmatch(file_path.name, pattern) for pattern in file_patterns):
                    continue

            # Check exclude patterns
            if exclude_patterns:
                if any(fnmatch.fnmatch(file_path.name, pattern) for pattern in exclude_patterns):
                    continue

            files.append(str(file_path))

        return files

    async def _process_batch(self, file_paths: List[str]) -> List[ParsedDocument]:
        """Process a batch of files concurrently."""
        tasks = []
        for file_path in file_paths:
            try:
                task = self.process_document(file_path)
                tasks.append(task)
            except Exception as e:
                # Skip files that can't be processed
                self.processing_stats["skipped"] += 1
                continue

        if not tasks:
            return []

        # Process concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        documents = []
        for result in results:
            if isinstance(result, Exception):
                self.processing_stats["failed"] += 1
            else:
                documents.append(result)

        return documents

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = sum([
            self.processing_stats["processed"],
            self.processing_stats["failed"],
            self.processing_stats["skipped"]
        ])

        return {
            **self.processing_stats,
            "total_documents": total,
            "success_rate": self.processing_stats["processed"] / max(1, total),
            "avg_processing_time": self.processing_stats["total_time"] / max(1, self.processing_stats["processed"])
        }


class TestDocumentProcessingWorkflow:
    """Test the complete document processing workflow."""

    @pytest.fixture
    def embedding_config(self):
        """Create embedding configuration for testing."""
        return EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            enable_sparse_vectors=True,
            chunk_size=500,  # Smaller for testing
            chunk_overlap=100,
            batch_size=10,
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

        # Mock embedding generation with different sizes
        def mock_embed(texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        def mock_sparse_encode(text):
            return {"indices": [1, 3, 5], "values": [0.8, 0.6, 0.9]}

        service.dense_model.embed.side_effect = mock_embed
        service.bm25_encoder.encode.side_effect = mock_sparse_encode

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
    async def workflow(self, embedding_service, mock_project_detector):
        """Create document processing workflow."""
        return DocumentProcessingWorkflow(embedding_service, mock_project_detector)

    @pytest.fixture
    def temp_test_directory(self):
        """Create temporary directory with various file types."""
        temp_dir = tempfile.mkdtemp()
        test_dir = Path(temp_dir)

        # Create various file types
        files = {}

        # Text files
        (test_dir / "simple.txt").write_text("This is a simple text file.")
        (test_dir / "python.py").write_text("def hello():\n    print('Hello world!')")
        files["text_files"] = 2

        # Markdown files
        (test_dir / "readme.md").write_text("# Project\n\nThis is a **test** project.")
        files["markdown_files"] = 1

        # HTML files
        (test_dir / "page.html").write_text(
            "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
        )
        files["html_files"] = 1

        # Large text file for chunking test
        large_content = "This is a sentence for testing chunking. " * 50  # ~2000 chars
        (test_dir / "large.txt").write_text(large_content)
        files["large_files"] = 1

        # Subdirectory
        sub_dir = test_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested.txt").write_text("Nested file content.")
        files["nested_files"] = 1

        files["total_files"] = sum(files.values())
        files["temp_dir"] = str(test_dir)

        yield files

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_single_document_processing(self, workflow, temp_test_directory):
        """Test processing of a single document through complete workflow."""
        text_file = Path(temp_test_directory["temp_dir"]) / "simple.txt"

        document = await workflow.process_document(str(text_file))

        # Verify document structure
        assert isinstance(document, ParsedDocument)
        assert document.file_path == str(text_file)
        assert document.file_type == "text"
        assert "simple text file" in document.content

        # Verify embeddings
        assert hasattr(document, "dense_vector")
        assert hasattr(document, "sparse_vector")
        assert len(document.dense_vector) == 4

        # Verify metadata enrichment
        assert document.metadata["detected_file_type"] == "text"
        assert "detection_confidence" in document.metadata
        assert document.metadata["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert document.metadata["word_count"] > 0
        assert document.metadata["estimated_language"] == "english"

        # Verify project integration
        assert document.metadata["project_type"] == "python"
        assert document.metadata["project_name"] == "test-project"

    @pytest.mark.asyncio
    async def test_large_document_chunking(self, workflow, temp_test_directory):
        """Test processing of large documents with chunking."""
        large_file = Path(temp_test_directory["temp_dir"]) / "large.txt"

        document = await workflow.process_document(str(large_file))

        # Verify chunking occurred
        assert document.metadata["chunked"] is True
        assert document.metadata["chunk_count"] > 1

        # Verify embeddings were combined
        assert hasattr(document, "dense_vector")
        assert len(document.dense_vector) == 4  # Combined embedding

    @pytest.mark.asyncio
    async def test_directory_processing(self, workflow, temp_test_directory):
        """Test processing of entire directory."""
        temp_dir = temp_test_directory["temp_dir"]
        expected_files = temp_test_directory["total_files"]

        documents = await workflow.process_directory(temp_dir, batch_size=3)

        assert len(documents) == expected_files

        # Verify different file types were processed
        file_types = {doc.file_type for doc in documents}
        assert "text" in file_types
        assert "markdown" in file_types
        assert "html" in file_types

        # Verify all have embeddings
        for doc in documents:
            assert hasattr(doc, "dense_vector")
            assert len(doc.dense_vector) > 0

    @pytest.mark.asyncio
    async def test_file_pattern_filtering(self, workflow, temp_test_directory):
        """Test directory processing with file pattern filtering."""
        temp_dir = temp_test_directory["temp_dir"]

        # Process only Python files
        documents = await workflow.process_directory(
            temp_dir,
            file_patterns=["*.py"],
            batch_size=2
        )

        assert len(documents) == 1
        assert documents[0].file_path.endswith(".py")
        assert "def hello" in documents[0].content

    @pytest.mark.asyncio
    async def test_exclude_pattern_filtering(self, workflow, temp_test_directory):
        """Test directory processing with exclude patterns."""
        temp_dir = temp_test_directory["temp_dir"]

        # Exclude large files
        documents = await workflow.process_directory(
            temp_dir,
            exclude_patterns=["large.*"],
            batch_size=2
        )

        # Should not include the large.txt file
        file_names = [Path(doc.file_path).name for doc in documents]
        assert "large.txt" not in file_names
        assert len(documents) == temp_test_directory["total_files"] - 1

    @pytest.mark.asyncio
    async def test_parser_selection_logic(self, workflow, temp_test_directory):
        """Test parser selection for different file types."""
        temp_dir = Path(temp_test_directory["temp_dir"])

        # Test markdown file gets markdown parser
        md_file = temp_dir / "readme.md"
        with patch.object(workflow, '_select_parser', wraps=workflow._select_parser) as mock_select:
            document = await workflow.process_document(str(md_file))

            mock_select.assert_called_once()
            args = mock_select.call_args[0]
            assert args[0] == "markdown"  # parser_type

        assert document.file_type == "markdown"
        assert "Project" in document.content

    @pytest.mark.asyncio
    async def test_content_preprocessing(self, workflow):
        """Test content preprocessing functionality."""
        # Create content with excessive whitespace
        content_with_whitespace = "Line 1    \n\n\n\n\nLine 2   \n\n\n   Line 3"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content_with_whitespace)
            f.flush()

            try:
                document = await workflow.process_document(f.name)

                # Verify whitespace was normalized
                assert "\n\n\n\n" not in document.content
                assert document.metadata["preprocessed"] is True
                assert document.metadata["original_length"] > 0

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_embedding_generation_integration(self, workflow):
        """Test integration with embedding generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for embedding generation.")
            f.flush()

            try:
                # Verify embedding service calls
                with patch.object(workflow.embedding_service, 'generate_embeddings') as mock_generate:
                    mock_generate.return_value = {
                        "dense": [0.5, 0.6, 0.7, 0.8],
                        "sparse": {"indices": [1, 2], "values": [0.9, 0.8]}
                    }

                    document = await workflow.process_document(f.name)

                    mock_generate.assert_called_once()
                    assert document.dense_vector == [0.5, 0.6, 0.7, 0.8]
                    assert document.sparse_vector == {"indices": [1, 2], "values": [0.9, 0.8]}

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_metadata_enrichment(self, workflow, temp_test_directory):
        """Test metadata enrichment functionality."""
        text_file = Path(temp_test_directory["temp_dir"]) / "simple.txt"

        document = await workflow.process_document(str(text_file))

        # Check all expected metadata fields
        expected_fields = [
            "detected_file_type", "detection_confidence", "embedding_model",
            "has_dense_vector", "has_sparse_vector", "word_count",
            "character_count", "line_count", "estimated_language",
            "preprocessed", "original_length"
        ]

        for field in expected_fields:
            assert field in document.metadata, f"Missing metadata field: {field}"

    @pytest.mark.asyncio
    async def test_workspace_context_integration(self, workflow, mock_project_detector):
        """Test workspace context integration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            f.flush()

            try:
                document = await workflow.process_document(f.name)

                # Verify project detector was called
                mock_project_detector.detect_project.assert_called_once_with(f.name)

                # Verify project metadata was added
                assert document.metadata["project_root"] == "/test/project"
                assert document.metadata["project_name"] == "test-project"
                assert document.metadata["project_type"] == "python"

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_missing_file(self, workflow):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            await workflow.process_document("/nonexistent/file.txt")

        # Verify stats were updated
        stats = workflow.get_processing_stats()
        assert stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_format(self, workflow):
        """Test error handling for unsupported file formats."""
        # Create a file with unknown extension
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b"Unknown content")
            f.flush()

            try:
                # Should either succeed with text parser or fail gracefully
                try:
                    document = await workflow.process_document(f.name)
                    # If it succeeds, should use text parser
                    assert document.file_type == "text"
                except (UnicodeDecodeError, ValueError):
                    # Or fail gracefully for binary content
                    pass

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_corrupted_file(self, workflow):
        """Test error handling for corrupted files."""
        # Create a corrupted text file (binary content with .txt extension)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'\x00\x01\x02\x03invalid text content\xff\xfe')
            f.flush()

            try:
                # Should either handle gracefully or raise appropriate error
                try:
                    document = await workflow.process_document(f.name)
                    # If it processes, content should be handled somehow
                    assert isinstance(document, ParsedDocument)
                except (UnicodeDecodeError, ParsingError):
                    # Or raise appropriate parsing error
                    pass

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, workflow, temp_test_directory):
        """Test batch processing handles individual file errors gracefully."""
        temp_dir = Path(temp_test_directory["temp_dir"])

        # Add a corrupted file
        corrupted_file = temp_dir / "corrupted.txt"
        corrupted_file.write_bytes(b'\x00\x01\x02invalid\xff\xfe')

        # Process directory - should handle the corrupted file gracefully
        documents = await workflow.process_directory(str(temp_dir), batch_size=2)

        # Should process most files successfully
        stats = workflow.get_processing_stats()
        assert stats["processed"] >= temp_test_directory["total_files"]
        assert stats["success_rate"] > 0.8  # At least 80% success rate

    @pytest.mark.asyncio
    async def test_processing_statistics(self, workflow, temp_test_directory):
        """Test processing statistics collection."""
        temp_dir = temp_test_directory["temp_dir"]

        # Process directory
        documents = await workflow.process_directory(temp_dir)

        stats = workflow.get_processing_stats()

        assert stats["processed"] == len(documents)
        assert stats["total_documents"] == stats["processed"] + stats["failed"] + stats["skipped"]
        assert 0 <= stats["success_rate"] <= 1
        assert stats["avg_processing_time"] > 0
        assert stats["total_time"] > 0

    @pytest.mark.asyncio
    async def test_embedding_chunking_logic(self, workflow):
        """Test chunking logic for large documents."""
        # Create content larger than chunk size (500 chars)
        large_content = "This is a test sentence for chunking logic. " * 20  # ~900 chars

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            f.flush()

            try:
                with patch.object(workflow.embedding_service, 'chunk_text') as mock_chunk:
                    mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]

                    with patch.object(workflow.embedding_service, 'generate_embeddings_batch') as mock_batch:
                        mock_batch.return_value = [
                            {"dense": [0.1, 0.2], "sparse": {"indices": [1], "values": [0.8]}},
                            {"dense": [0.3, 0.4], "sparse": {"indices": [2], "values": [0.7]}},
                            {"dense": [0.5, 0.6], "sparse": {"indices": [3], "values": [0.6]}}
                        ]

                        document = await workflow.process_document(f.name)

                        # Verify chunking was used
                        mock_chunk.assert_called_once()
                        mock_batch.assert_called_once()

                        # Verify embeddings were combined
                        assert document.metadata["chunked"] is True
                        assert document.metadata["chunk_count"] == 3

                        # Verify dense embeddings were averaged
                        expected_dense = [0.3, 0.4]  # Average of chunks
                        assert document.dense_vector == expected_dense

                        # Verify sparse embeddings were combined
                        expected_sparse = {"indices": [1, 2, 3], "values": [0.8, 0.7, 0.6]}
                        assert document.sparse_vector == expected_sparse

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_language_detection(self, workflow):
        """Test language detection functionality."""
        # Test English content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("The quick brown fox jumps over the lazy dog and runs.")
            f.flush()

            try:
                document = await workflow.process_document(f.name)
                assert document.metadata["estimated_language"] == "english"
            finally:
                Path(f.name).unlink()

        # Test Python code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test():\n    if True:\n        for i in range(10):\n            print(i)")
            f.flush()

            try:
                document = await workflow.process_document(f.name)
                assert document.metadata["estimated_language"] == "python"
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_workflow_without_project_detector(self, embedding_service):
        """Test workflow functionality without project detector."""
        workflow = DocumentProcessingWorkflow(embedding_service, project_detector=None)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content without project detection.")
            f.flush()

            try:
                document = await workflow.process_document(f.name)

                # Should still process successfully
                assert isinstance(document, ParsedDocument)
                assert hasattr(document, "dense_vector")

                # Project metadata should not be present
                assert "project_root" not in document.metadata
                assert "project_name" not in document.metadata

            finally:
                Path(f.name).unlink()

    def test_parser_registry_initialization(self, workflow):
        """Test parser registry initialization."""
        parsers = workflow.parser_registry

        assert "text" in parsers
        assert "markdown" in parsers
        assert "pdf" in parsers
        assert "html" in parsers
        assert "docx" in parsers

        assert isinstance(parsers["text"], TextParser)
        assert isinstance(parsers["markdown"], MarkdownParser)

    def test_dense_embedding_combination(self, workflow):
        """Test dense embedding combination logic."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]

        combined = workflow._combine_dense_embeddings(embeddings)

        expected = [0.4, 0.5, 0.6]  # Average of all embeddings
        assert combined == expected

    def test_sparse_embedding_combination(self, workflow):
        """Test sparse embedding combination logic."""
        embeddings = [
            {"indices": [1, 2], "values": [0.8, 0.6]},
            {"indices": [2, 3], "values": [0.7, 0.9]},
            {"indices": [1, 4], "values": [0.5, 0.8]}
        ]

        combined = workflow._combine_sparse_embeddings(embeddings)

        # Index 1: 0.8 + 0.5 = 1.3
        # Index 2: 0.6 + 0.7 = 1.3
        # Index 3: 0.9
        # Index 4: 0.8
        expected_indices = [1, 2, 3, 4]
        expected_values = [1.3, 1.3, 0.9, 0.8]

        assert combined["indices"] == expected_indices
        assert combined["values"] == expected_values