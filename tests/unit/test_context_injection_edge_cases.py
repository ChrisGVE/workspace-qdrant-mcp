"""
Unit tests for context injection edge cases and error handling.

Tests error scenarios, edge cases, and boundary conditions in the context
injection workflow including unsupported file types, memory constraints,
processing failures, and recovery mechanisms.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.core.config import Config, EmbeddingConfig
from common.core.embeddings import EmbeddingService
from wqm_cli.cli.parsers.base import DocumentParser, ParsedDocument
from wqm_cli.cli.parsers.exceptions import ParsingError, ParsingTimeout, FileFormatError
from wqm_cli.cli.parsers.file_detector import detect_file_type


class EdgeCaseContextInjector:
    """Context injector with enhanced error handling for testing edge cases."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.processing_limits = {
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "max_content_length": 1024 * 1024,  # 1MB
            "max_batch_size": 100,
            "timeout_seconds": 30
        }
        self.error_stats = {
            "parsing_errors": 0,
            "embedding_errors": 0,
            "timeout_errors": 0,
            "memory_errors": 0,
            "unsupported_format_errors": 0
        }
        self.processed_documents = []
        self.failed_documents = []
        self.retry_attempts = {}

    async def process_document_with_retry(
        self,
        file_path: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[ParsedDocument]:
        """Process document with retry logic for transient failures."""
        for attempt in range(max_retries + 1):
            try:
                return await self.process_document_with_limits(file_path)
            except (asyncio.TimeoutError, MemoryError, OSError) as e:
                if attempt == max_retries:
                    self.failed_documents.append({
                        "file_path": file_path,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempts": attempt + 1
                    })
                    raise

                # Wait before retry
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                self.retry_attempts[file_path] = attempt + 1

        return None

    async def process_document_with_limits(self, file_path: str) -> ParsedDocument:
        """Process document with resource limits and safety checks."""
        # Check file size limit
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.processing_limits["max_file_size"]:
                raise ValueError(f"File too large: {file_size} bytes")
        except OSError as e:
            self.error_stats["parsing_errors"] += 1
            raise ParsingError(f"Cannot access file: {e}")

        # Process with timeout
        try:
            return await asyncio.wait_for(
                self._process_document_internal(file_path),
                timeout=self.processing_limits["timeout_seconds"]
            )
        except asyncio.TimeoutError:
            self.error_stats["timeout_errors"] += 1
            raise ParsingTimeout(f"Processing timeout for {file_path}")

    async def _process_document_internal(self, file_path: str) -> ParsedDocument:
        """Internal document processing with error handling."""
        try:
            # File type detection with fallback
            try:
                file_type, parser_type, confidence = detect_file_type(file_path)
                if confidence < 0.5:
                    # Low confidence, try content-based detection
                    file_type, parser_type = await self._fallback_detection(file_path)
            except Exception:
                # Detection failed, use extension-based fallback
                file_type, parser_type = self._extension_fallback(file_path)

            # Content extraction with limits
            try:
                content = await self._extract_content_safely(file_path, parser_type)
            except UnicodeDecodeError:
                # Try alternative encodings
                content = await self._extract_content_with_encoding_detection(file_path)
            except Exception as e:
                self.error_stats["parsing_errors"] += 1
                raise ParsingError(f"Content extraction failed: {e}")

            # Content length check
            if len(content) > self.processing_limits["max_content_length"]:
                content = await self._handle_large_content(content, file_path)

            # Create document
            document = ParsedDocument.create(
                content=content,
                file_path=file_path,
                file_type=file_type,
                additional_metadata={
                    "parser_type": parser_type,
                    "confidence": confidence if 'confidence' in locals() else 0.0,
                    "processing_method": "safe_extraction"
                }
            )

            # Generate embeddings with error handling
            try:
                document = await self._generate_embeddings_safely(document)
            except Exception as e:
                self.error_stats["embedding_errors"] += 1
                # Continue without embeddings if generation fails
                document.metadata["embedding_error"] = str(e)

            self.processed_documents.append(document)
            return document

        except MemoryError:
            self.error_stats["memory_errors"] += 1
            raise
        except Exception as e:
            self.error_stats["parsing_errors"] += 1
            raise ParsingError(f"Document processing failed: {e}")

    async def _extract_content_safely(self, file_path: str, parser_type: str) -> str:
        """Extract content with safety checks and limits."""
        # Simplified content extraction based on parser type
        path = Path(file_path)

        if parser_type == "binary" or not path.suffix:
            raise FileFormatError(f"Unsupported format: {path.suffix}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read in chunks to avoid memory issues
                content = ""
                chunk_size = 8192
                while len(content) < self.processing_limits["max_content_length"]:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content += chunk

                return content
        except UnicodeDecodeError:
            raise  # Let caller handle encoding detection

    async def _extract_content_with_encoding_detection(self, file_path: str) -> str:
        """Extract content with automatic encoding detection."""
        import chardet

        # Read raw bytes for encoding detection
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)  # Sample for encoding detection

        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read(self.processing_limits["max_content_length"])
        except UnicodeDecodeError:
            # Final fallback to latin-1 (never fails)
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read(self.processing_limits["max_content_length"])

    async def _handle_large_content(self, content: str, file_path: str) -> str:
        """Handle content that exceeds size limits."""
        max_length = self.processing_limits["max_content_length"]

        # Intelligent truncation - try to break at sentence boundaries
        if len(content) > max_length:
            truncated = content[:max_length]

            # Find last sentence boundary
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')

            if last_period > max_length * 0.8:  # Within last 20%
                truncated = truncated[:last_period + 1]
            elif last_newline > max_length * 0.8:
                truncated = truncated[:last_newline]

            # Add truncation marker
            truncated += "\n\n[CONTENT TRUNCATED]"

            return truncated

        return content

    async def _generate_embeddings_safely(self, document: ParsedDocument) -> ParsedDocument:
        """Generate embeddings with error handling and fallbacks."""
        try:
            # Check if embedding service is available
            if not self.embedding_service.initialized:
                raise RuntimeError("Embedding service not initialized")

            # Generate embeddings with timeout
            embeddings = await asyncio.wait_for(
                self.embedding_service.generate_embeddings(
                    document.content,
                    include_sparse=self.embedding_service.config.embedding.enable_sparse_vectors
                ),
                timeout=10.0  # 10-second timeout for embedding generation
            )

            document.dense_vector = embeddings.get("dense", [])
            document.sparse_vector = embeddings.get("sparse", {})
            document.metadata["has_embeddings"] = True

        except asyncio.TimeoutError:
            document.metadata["embedding_timeout"] = True
            document.metadata["has_embeddings"] = False
        except Exception as e:
            document.metadata["embedding_error"] = str(e)
            document.metadata["has_embeddings"] = False

        return document

    async def _fallback_detection(self, file_path: str) -> tuple[str, str]:
        """Fallback file type detection based on content analysis."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline() for _ in range(5)]
                content_sample = ''.join(first_lines)

            # Simple heuristics
            if any(keyword in content_sample.lower() for keyword in ['def ', 'import ', 'class ']):
                return "python", "text"
            elif content_sample.strip().startswith('<!DOCTYPE html') or '<html' in content_sample:
                return "html", "html"
            elif content_sample.strip().startswith('#') and any(marker in content_sample for marker in ['**', '*', '##']):
                return "markdown", "markdown"
            else:
                return "text", "text"

        except Exception:
            return "unknown", "text"

    def _extension_fallback(self, file_path: str) -> tuple[str, str]:
        """Extension-based fallback detection."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        extension_map = {
            '.py': ('python', 'text'),
            '.js': ('javascript', 'text'),
            '.ts': ('typescript', 'text'),
            '.md': ('markdown', 'markdown'),
            '.html': ('html', 'html'),
            '.htm': ('html', 'html'),
            '.txt': ('text', 'text'),
            '.log': ('text', 'text'),
            '.json': ('json', 'text'),
            '.xml': ('xml', 'text'),
            '.csv': ('csv', 'text'),
        }

        return extension_map.get(suffix, ('unknown', 'text'))

    async def process_batch_with_error_isolation(
        self,
        file_paths: List[str],
        batch_size: int = 10,
        fail_fast: bool = False
    ) -> tuple[List[ParsedDocument], List[Dict[str, Any]]]:
        """Process batch with error isolation - failures don't stop processing."""
        successful_documents = []
        failed_documents = []

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]

            # Process batch with error isolation
            batch_results = await self._process_batch_isolated(batch, fail_fast)

            for result in batch_results:
                if isinstance(result, ParsedDocument):
                    successful_documents.append(result)
                else:
                    failed_documents.append(result)

        return successful_documents, failed_documents

    async def _process_batch_isolated(
        self,
        file_paths: List[str],
        fail_fast: bool
    ) -> List[Any]:
        """Process batch with individual error isolation."""
        tasks = []

        for file_path in file_paths:
            task = self._process_single_with_error_capture(file_path)
            tasks.append(task)

        if fail_fast:
            return await asyncio.gather(*tasks)
        else:
            # Use return_exceptions to capture both results and exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error dictionaries
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_dict = {
                        "file_path": file_paths[i],
                        "error": str(result),
                        "error_type": type(result).__name__
                    }
                    processed_results.append(error_dict)
                else:
                    processed_results.append(result)

            return processed_results

    async def _process_single_with_error_capture(self, file_path: str) -> Any:
        """Process single file with comprehensive error capture."""
        try:
            return await self.process_document_with_retry(file_path, max_retries=2)
        except Exception as e:
            # Convert exception to error dictionary
            return {
                "file_path": file_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "recoverable": isinstance(e, (asyncio.TimeoutError, OSError))
            }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_processed = len(self.processed_documents)
        total_failed = len(self.failed_documents)
        total_attempts = total_processed + total_failed

        return {
            **self.error_stats,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_attempts": total_attempts,
            "success_rate": total_processed / max(1, total_attempts),
            "retry_count": len(self.retry_attempts),
            "avg_retries": sum(self.retry_attempts.values()) / max(1, len(self.retry_attempts))
        }

    async def simulate_memory_pressure(self, target_usage_mb: int = 100):
        """Simulate memory pressure for testing memory handling."""
        # Allocate memory to simulate pressure
        memory_hog = bytearray(target_usage_mb * 1024 * 1024)

        try:
            await asyncio.sleep(0.1)  # Let memory allocation take effect
            yield memory_hog
        finally:
            del memory_hog
            await asyncio.sleep(0.1)  # Let GC clean up

    async def test_concurrent_processing_limits(
        self,
        file_paths: List[str],
        max_concurrent: int = 5
    ) -> List[ParsedDocument]:
        """Test processing with concurrency limits."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(file_path: str):
            async with semaphore:
                return await self.process_document_with_retry(file_path)

        tasks = [process_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        documents = [r for r in results if isinstance(r, ParsedDocument)]
        return documents


class TestContextInjectionEdgeCases:
    """Test edge cases and error scenarios in context injection."""

    @pytest.fixture
    def embedding_config(self):
        """Create embedding configuration for testing."""
        return EmbeddingConfig(
            model="test-model",
            enable_sparse_vectors=True,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=10,
        )

    @pytest.fixture
    def config(self, embedding_config):
        """Create configuration for testing."""
        config = Config()
        config.embedding = embedding_config
        return config

    @pytest.fixture
    async def embedding_service(self, config):
        """Create mock embedding service for edge case testing."""
        service = EmbeddingService(config)

        # Mock with error simulation capabilities
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()
        service.initialized = True

        return service

    @pytest.fixture
    async def edge_case_injector(self, embedding_service):
        """Create edge case context injector."""
        return EdgeCaseContextInjector(embedding_service)

    @pytest.fixture
    def problematic_files(self):
        """Create problematic test files for edge case testing."""
        temp_dir = tempfile.mkdtemp()
        files = {}

        # Very large file (simulated)
        large_file = Path(temp_dir) / "large_file.txt"
        large_content = "This is a large file content. " * 10000  # ~300KB
        large_file.write_text(large_content)
        files["large"] = str(large_file)

        # Binary file with text extension
        binary_file = Path(temp_dir) / "fake_text.txt"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xFF\xFE\xFD\xFC')
        files["binary"] = str(binary_file)

        # Empty file
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.write_text("")
        files["empty"] = str(empty_file)

        # File with problematic encoding
        encoding_file = Path(temp_dir) / "encoding_issue.txt"
        try:
            encoding_file.write_bytes("Café naïve résumé".encode('iso-8859-1'))
        except:
            encoding_file.write_text("Cafe naive resume")  # Fallback
        files["encoding"] = str(encoding_file)

        # File with excessive whitespace
        whitespace_file = Path(temp_dir) / "whitespace.txt"
        whitespace_content = "\n\n\n   \t\t\t   \n\n\n" * 100
        whitespace_file.write_text(whitespace_content)
        files["whitespace"] = str(whitespace_file)

        # Unknown extension
        unknown_file = Path(temp_dir) / "unknown.xyz"
        unknown_file.write_text("Content with unknown extension")
        files["unknown"] = str(unknown_file)

        files["temp_dir"] = temp_dir

        yield files

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_large_file_handling(self, edge_case_injector, problematic_files):
        """Test handling of large files with size limits."""
        large_file = problematic_files["large"]

        # Should process successfully with truncation
        document = await edge_case_injector.process_document_with_limits(large_file)

        assert isinstance(document, ParsedDocument)
        assert len(document.content) <= edge_case_injector.processing_limits["max_content_length"]
        assert "[CONTENT TRUNCATED]" in document.content

    @pytest.mark.asyncio
    async def test_binary_file_handling(self, edge_case_injector, problematic_files):
        """Test handling of binary files with text extensions."""
        binary_file = problematic_files["binary"]

        # Should handle gracefully with encoding detection
        try:
            document = await edge_case_injector.process_document_with_limits(binary_file)
            # If successful, should have processed with fallback encoding
            assert isinstance(document, ParsedDocument)
        except (FileFormatError, ParsingError):
            # Or should fail with appropriate error
            pass

    @pytest.mark.asyncio
    async def test_empty_file_handling(self, edge_case_injector, problematic_files):
        """Test handling of empty files."""
        empty_file = problematic_files["empty"]

        document = await edge_case_injector.process_document_with_limits(empty_file)

        assert isinstance(document, ParsedDocument)
        assert document.content == ""
        assert document.metadata["processing_method"] == "safe_extraction"

    @pytest.mark.asyncio
    async def test_encoding_issue_handling(self, edge_case_injector, problematic_files):
        """Test handling of files with encoding issues."""
        encoding_file = problematic_files["encoding"]

        document = await edge_case_injector.process_document_with_limits(encoding_file)

        assert isinstance(document, ParsedDocument)
        # Should have content (possibly with encoding conversion)
        assert len(document.content) > 0

    @pytest.mark.asyncio
    async def test_processing_timeout_handling(self, edge_case_injector, problematic_files):
        """Test timeout handling during processing."""
        # Mock slow processing
        original_extract = edge_case_injector._extract_content_safely

        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(35)  # Longer than timeout
            return await original_extract(*args, **kwargs)

        edge_case_injector._extract_content_safely = slow_extract

        with pytest.raises(ParsingTimeout):
            await edge_case_injector.process_document_with_limits(problematic_files["large"])

        assert edge_case_injector.error_stats["timeout_errors"] == 1

    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, edge_case_injector, problematic_files):
        """Test handling of embedding generation failures."""
        # Mock embedding service failure
        edge_case_injector.embedding_service.generate_embeddings = AsyncMock(
            side_effect=RuntimeError("Embedding service unavailable")
        )

        document = await edge_case_injector.process_document_with_limits(
            problematic_files["empty"]
        )

        # Should complete processing without embeddings
        assert isinstance(document, ParsedDocument)
        assert "embedding_error" in document.metadata
        assert document.metadata["has_embeddings"] is False

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, edge_case_injector, problematic_files):
        """Test retry mechanism for transient failures."""
        call_count = 0

        async def failing_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise OSError("Transient failure")
            return await edge_case_injector._process_document_internal(*args, **kwargs)

        edge_case_injector._process_document_internal = failing_process

        # Should succeed after retries
        document = await edge_case_injector.process_document_with_retry(
            problematic_files["empty"],
            max_retries=3
        )

        assert isinstance(document, ParsedDocument)
        assert call_count == 3
        assert problematic_files["empty"] in edge_case_injector.retry_attempts

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, edge_case_injector, problematic_files):
        """Test behavior when retries are exhausted."""
        # Mock persistent failure
        edge_case_injector._process_document_internal = AsyncMock(
            side_effect=OSError("Persistent failure")
        )

        with pytest.raises(OSError, match="Persistent failure"):
            await edge_case_injector.process_document_with_retry(
                problematic_files["empty"],
                max_retries=2
            )

        # Should track failed document
        assert len(edge_case_injector.failed_documents) == 1
        assert edge_case_injector.failed_documents[0]["error_type"] == "OSError"

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, edge_case_injector, problematic_files):
        """Test batch processing with mixed success/failure scenarios."""
        file_paths = [
            problematic_files["empty"],     # Should succeed
            problematic_files["encoding"],  # Should succeed
            "/nonexistent/file.txt",       # Should fail
            problematic_files["whitespace"] # Should succeed
        ]

        successful, failed = await edge_case_injector.process_batch_with_error_isolation(
            file_paths,
            batch_size=2,
            fail_fast=False
        )

        # Should have some successes and some failures
        assert len(successful) >= 2
        assert len(failed) >= 1

        # Check error tracking
        nonexistent_error = next(
            (err for err in failed if "/nonexistent/file.txt" in err["file_path"]),
            None
        )
        assert nonexistent_error is not None

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, edge_case_injector, problematic_files):
        """Test handling under memory pressure conditions."""
        # Simulate memory pressure
        async with edge_case_injector.simulate_memory_pressure(50):  # 50MB
            try:
                document = await edge_case_injector.process_document_with_limits(
                    problematic_files["large"]
                )
                assert isinstance(document, ParsedDocument)
            except MemoryError:
                # Memory error should be properly categorized
                assert edge_case_injector.error_stats["memory_errors"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_processing_limits(self, edge_case_injector, problematic_files):
        """Test concurrent processing with resource limits."""
        file_paths = list(problematic_files.values())[:-1]  # Exclude temp_dir

        # Process with concurrency limit
        documents = await edge_case_injector.test_concurrent_processing_limits(
            file_paths,
            max_concurrent=2
        )

        # Should process some files successfully
        assert len(documents) >= 2
        assert all(isinstance(doc, ParsedDocument) for doc in documents)

    @pytest.mark.asyncio
    async def test_file_access_permission_errors(self, edge_case_injector):
        """Test handling of file permission errors."""
        # Create file and remove read permissions (Unix-like systems)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            f.flush()

            try:
                # Remove read permissions
                os.chmod(f.name, 0o000)

                with pytest.raises(ParsingError):
                    await edge_case_injector.process_document_with_limits(f.name)

                assert edge_case_injector.error_stats["parsing_errors"] > 0

            finally:
                # Restore permissions for cleanup
                try:
                    os.chmod(f.name, 0o644)
                    os.unlink(f.name)
                except:
                    pass

    @pytest.mark.asyncio
    async def test_malformed_file_detection(self, edge_case_injector):
        """Test detection and handling of malformed files."""
        # Create file that looks like one type but is another
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Write PDF header but with .txt extension
            f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\n')
            f.flush()

            try:
                # Should detect mismatch and handle appropriately
                document = await edge_case_injector.process_document_with_limits(f.name)

                # Should either succeed with fallback or fail gracefully
                if isinstance(document, ParsedDocument):
                    assert document.metadata["parser_type"] is not None
                else:
                    pytest.fail("Should have produced a document or raised appropriate error")

            except (FileFormatError, ParsingError):
                # Acceptable outcome for malformed files
                pass
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_embedding_timeout_handling(self, edge_case_injector, problematic_files):
        """Test timeout handling during embedding generation."""
        # Mock slow embedding generation
        async def slow_embeddings(*args, **kwargs):
            await asyncio.sleep(15)  # Longer than embedding timeout
            return {"dense": [0.1, 0.2], "sparse": {}}

        edge_case_injector.embedding_service.generate_embeddings = slow_embeddings

        document = await edge_case_injector.process_document_with_limits(
            problematic_files["empty"]
        )

        # Should complete without embeddings
        assert isinstance(document, ParsedDocument)
        assert document.metadata.get("embedding_timeout") is True
        assert document.metadata.get("has_embeddings") is False

    @pytest.mark.asyncio
    async def test_unknown_file_extension_handling(self, edge_case_injector, problematic_files):
        """Test handling of files with unknown extensions."""
        unknown_file = problematic_files["unknown"]

        document = await edge_case_injector.process_document_with_limits(unknown_file)

        assert isinstance(document, ParsedDocument)
        assert document.file_type in ["unknown", "text"]  # Should fallback to text
        assert document.content == "Content with unknown extension"

    def test_error_statistics_tracking(self, edge_case_injector):
        """Test comprehensive error statistics tracking."""
        # Simulate various errors
        edge_case_injector.error_stats["parsing_errors"] = 5
        edge_case_injector.error_stats["embedding_errors"] = 2
        edge_case_injector.error_stats["timeout_errors"] = 1
        edge_case_injector.processed_documents = [Mock()] * 10
        edge_case_injector.failed_documents = [Mock()] * 3
        edge_case_injector.retry_attempts = {"file1": 2, "file2": 1}

        stats = edge_case_injector.get_error_statistics()

        assert stats["parsing_errors"] == 5
        assert stats["embedding_errors"] == 2
        assert stats["timeout_errors"] == 1
        assert stats["total_processed"] == 10
        assert stats["total_failed"] == 3
        assert stats["total_attempts"] == 13
        assert stats["success_rate"] == 10/13
        assert stats["retry_count"] == 2
        assert stats["avg_retries"] == 1.5

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_embeddings(self, problematic_files):
        """Test graceful degradation when embedding service is unavailable."""
        # Create injector without embedding service
        config = Config()
        config.embedding = EmbeddingConfig(
            model="test-model",
            enable_sparse_vectors=False,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=10,
        )

        # Mock uninitialized embedding service
        embedding_service = MagicMock()
        embedding_service.initialized = False

        injector = EdgeCaseContextInjector(embedding_service)

        document = await injector.process_document_with_limits(problematic_files["empty"])

        # Should process document without embeddings
        assert isinstance(document, ParsedDocument)
        assert document.metadata.get("has_embeddings") is False
        assert "embedding_error" in document.metadata

    @pytest.mark.asyncio
    async def test_content_truncation_intelligence(self, edge_case_injector):
        """Test intelligent content truncation at sentence boundaries."""
        # Create content with clear sentence boundaries
        content = "First sentence. Second sentence. " * 1000  # Will exceed limits

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()

            try:
                document = await edge_case_injector.process_document_with_limits(f.name)

                assert isinstance(document, ParsedDocument)
                assert len(document.content) <= edge_case_injector.processing_limits["max_content_length"]
                assert "[CONTENT TRUNCATED]" in document.content

                # Should end with a complete sentence
                content_before_marker = document.content.split("[CONTENT TRUNCATED]")[0]
                assert content_before_marker.rstrip().endswith('.')

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_fallback_detection_mechanisms(self, edge_case_injector):
        """Test multiple fallback mechanisms for file type detection."""
        # Create file with misleading extension but clear content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("def hello():\n    print('Python code')\n    return 42")
            f.flush()

            try:
                # Detection should use content analysis fallback
                file_type, parser_type = await edge_case_injector._fallback_detection(f.name)

                assert file_type == "python"
                assert parser_type == "text"

            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_errors(self, edge_case_injector, problematic_files):
        """Test that resources are properly cleaned up even when errors occur."""
        import gc
        import weakref

        # Track object creation
        initial_objects = len(gc.get_objects())

        # Process files with expected errors
        try:
            await edge_case_injector.process_document_with_limits("/nonexistent/file.txt")
        except:
            pass

        try:
            await edge_case_injector.process_document_with_limits(problematic_files["binary"])
        except:
            pass

        # Force garbage collection
        gc.collect()

        # Check that we haven't leaked too many objects
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Allow some increase but not excessive
        assert object_increase < 1000, f"Potential memory leak: {object_increase} new objects"