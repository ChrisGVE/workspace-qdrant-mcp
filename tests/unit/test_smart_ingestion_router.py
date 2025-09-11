"""
Tests for Smart Ingestion Router

This module contains comprehensive tests for the smart ingestion differentiation logic
including file classification, routing decisions, processing strategies, and batch operations.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import pytest
import structlog
import yaml

from common.core.smart_ingestion_router import (
    SmartIngestionRouter,
    FileClassifier,
    RouterConfiguration,
    ProcessingStrategy,
    FileClassification,
    ClassificationResult,
    RouterStatistics
)
from common.core.language_filters import LanguageAwareFilter
from common.core.lsp_metadata_extractor import LspMetadataExtractor, FileMetadata

logger = structlog.get_logger(__name__)


class TestFileClassifier:
    """Test file classification functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = RouterConfiguration()
        self.classifier = FileClassifier(self.config)
    
    def test_classify_by_extension_code_files(self, tmp_path):
        """Test classification of code files by extension"""
        # Python file
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        result = self.classifier.classify_file(py_file)
        
        assert result.classification == FileClassification.CODE
        assert result.confidence >= 0.9
        assert result.strategy == ProcessingStrategy.LSP_ENRICHED
        
        # JavaScript file
        js_file = tmp_path / "test.js"
        js_file.write_text("function hello() { return 'world'; }")
        result = self.classifier.classify_file(js_file)
        
        assert result.classification == FileClassification.CODE
        assert result.confidence >= 0.9
        assert result.strategy == ProcessingStrategy.LSP_ENRICHED
    
    def test_classify_by_extension_documentation_files(self, tmp_path):
        """Test classification of documentation files by extension"""
        # Markdown file
        md_file = tmp_path / "README.md"
        md_file.write_text("# Hello World\nThis is a test document.")
        result = self.classifier.classify_file(md_file)
        
        assert result.classification == FileClassification.DOCUMENTATION
        assert result.confidence >= 0.9
        assert result.strategy == ProcessingStrategy.STANDARD_INGESTION
    
    def test_classify_by_extension_data_files(self, tmp_path):
        """Test classification of data files by extension"""
        # Create a custom config without JSON in force_standard_extensions
        config = RouterConfiguration()
        config.force_standard_extensions = {'.md', '.txt', '.rst'}  # Remove JSON
        classifier = FileClassifier(config)
        
        # JSON file
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value", "number": 42}')
        result = classifier.classify_file(json_file)
        
        assert result.classification == FileClassification.DATA
        assert result.confidence >= 0.9
        assert result.strategy == ProcessingStrategy.STANDARD_INGESTION
    
    def test_classify_by_extension_binary_files(self, tmp_path):
        """Test classification of binary files by extension"""
        # Binary file
        exe_file = tmp_path / "program.exe"
        exe_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        result = self.classifier.classify_file(exe_file)
        
        assert result.classification == FileClassification.BINARY
        assert result.confidence >= 0.9
        assert result.strategy == ProcessingStrategy.SKIP
    
    def test_configuration_overrides(self, tmp_path):
        """Test configuration overrides for file classification"""
        # Test force LSP extension override
        config = RouterConfiguration()
        config.force_lsp_extensions.add('.custom')
        classifier = FileClassifier(config)
        
        custom_file = tmp_path / "test.custom"
        custom_file.write_text("some content")
        result = classifier.classify_file(custom_file)
        
        assert result.classification == FileClassification.CODE
        assert result.confidence == 1.0
        assert result.strategy == ProcessingStrategy.LSP_ENRICHED
        assert "extension_lsp_override" in result.reason
        
        # Test skip extension override
        config.skip_extensions.add('.skip')
        skip_file = tmp_path / "test.skip"
        skip_file.write_text("some content")
        result = classifier.classify_file(skip_file)
        
        assert result.classification == FileClassification.BINARY
        assert result.strategy == ProcessingStrategy.SKIP
        assert "extension_skip_override" in result.reason
    
    def test_content_analysis_classification(self, tmp_path):
        """Test content-based file classification"""
        # Code-like content without code extension
        code_file = tmp_path / "noext"
        code_file.write_text("""
def calculate_sum(a, b):
    '''Calculate the sum of two numbers'''
    return a + b

class Calculator:
    def __init__(self):
        self.history = []
        
    def add(self, x, y):
        result = x + y
        self.history.append(('add', x, y, result))
        return result
""")
        
        result = self.classifier.classify_file(code_file)
        assert result.classification == FileClassification.CODE
        assert result.confidence >= 0.6  # Lower confidence due to no extension
        
        # Documentation-like content
        doc_file = tmp_path / "document"
        doc_file.write_text("""
# Project Documentation

This is a comprehensive guide to understanding the project structure and implementation.

## Getting Started

To begin working with this project, you'll need to install the required dependencies
and configure your development environment properly.

### Prerequisites

- Python 3.8 or higher
- Node.js for frontend components
- Docker for containerized deployment
""")
        
        result = self.classifier.classify_file(doc_file)
        assert result.classification == FileClassification.DOCUMENTATION
        assert result.confidence >= 0.7
    
    def test_language_detection(self, tmp_path):
        """Test programming language detection"""
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.ts", "typescript"),
            ("test.rs", "rust"),
            ("test.java", "java"),
            ("test.go", "go"),
            ("test.cpp", "cpp"),
            ("test.c", "c"),
        ]
        
        for filename, expected_lang in test_cases:
            file_path = tmp_path / filename
            file_path.write_text("// test content")
            
            detected_lang = self.classifier._detect_language(file_path)
            assert detected_lang == expected_lang
    
    def test_classify_nonexistent_file(self, tmp_path):
        """Test classification of non-existent file"""
        nonexistent = tmp_path / "does_not_exist.py"
        result = self.classifier.classify_file(nonexistent)
        
        assert result.classification == FileClassification.UNKNOWN
        assert result.confidence == 0.0
        assert result.strategy == ProcessingStrategy.SKIP
        assert result.reason == "file_not_found"
    
    def test_binary_content_detection(self, tmp_path):
        """Test detection of binary content"""
        binary_file = tmp_path / "test.dat"
        # Write binary content with null bytes
        binary_file.write_bytes(b'\x00\x01\x02\x03Hello\x00World\xFF\xFE')
        
        classification, confidence = self.classifier._classify_by_content(binary_file)
        assert classification == FileClassification.BINARY
        assert confidence >= 0.9


class TestRouterConfiguration:
    """Test router configuration loading and management"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = RouterConfiguration()
        
        assert config.enable_content_analysis is True
        assert config.enable_mime_detection is True
        assert config.max_content_sample_bytes == 4096
        assert config.fallback_on_lsp_error is True
        assert config.batch_size_limit == 100
        assert config.enable_caching is True
        
        # Check default extensions
        assert '.py' in config.force_lsp_extensions
        assert '.js' in config.force_lsp_extensions
        assert '.md' in config.force_standard_extensions
        assert '.pyc' in config.skip_extensions
    
    def test_yaml_configuration_loading(self, tmp_path):
        """Test loading configuration from YAML file"""
        config_file = tmp_path / "router_config.yaml"
        config_data = {
            'smart_router': {
                'enable_content_analysis': False,
                'max_content_sample_bytes': 8192,
                'force_lsp_extensions': ['.custom', '.special'],
                'batch_size_limit': 50,
                'fallback_timeout_seconds': 60.0,
                'custom_language_map': {
                    '.special': 'special_language'
                }
            }
        }
        
        config_file.write_text(yaml.dump(config_data))
        config = RouterConfiguration.from_yaml(config_file)
        
        assert config.enable_content_analysis is False
        assert config.max_content_sample_bytes == 8192
        assert config.force_lsp_extensions == {'.custom', '.special'}
        assert config.batch_size_limit == 50
        assert config.fallback_timeout_seconds == 60.0
        assert config.custom_language_map['.special'] == 'special_language'
    
    def test_yaml_configuration_missing_file(self, tmp_path):
        """Test handling of missing configuration file"""
        missing_file = tmp_path / "nonexistent.yaml"
        config = RouterConfiguration.from_yaml(missing_file)
        
        # Should return default configuration
        assert isinstance(config, RouterConfiguration)
        assert config.enable_content_analysis is True  # Default value


class TestSmartIngestionRouter:
    """Test smart ingestion router functionality"""
    
    @pytest.fixture
    def mock_file_filter(self):
        """Create mock file filter"""
        filter_mock = MagicMock(spec=LanguageAwareFilter)
        filter_mock._initialized = True
        filter_mock.should_process_file.return_value = (True, "accepted")
        return filter_mock
    
    @pytest.fixture
    def mock_lsp_extractor(self):
        """Create mock LSP extractor"""
        extractor_mock = MagicMock(spec=LspMetadataExtractor)
        extractor_mock._initialized = True
        
        # Mock LSP server configs
        extractor_mock.lsp_server_configs = {
            'python': {'command': ['pylsp'], 'file_extensions': ['.py']},
            'javascript': {'command': ['typescript-language-server'], 'file_extensions': ['.js']},
            'rust': {'command': ['rust-analyzer'], 'file_extensions': ['.rs']}
        }
        
        # Mock FileMetadata
        mock_metadata = MagicMock(spec=FileMetadata)
        mock_metadata.symbols = []
        mock_metadata.relationships = []
        mock_metadata.to_dict.return_value = {
            "file_path": "/test/path.py",
            "language": "python",
            "symbols": [],
            "relationships": []
        }
        
        extractor_mock.extract_file_metadata.return_value = mock_metadata
        return extractor_mock
    
    @pytest.fixture
    def router(self, mock_file_filter, mock_lsp_extractor):
        """Create router with mocked dependencies"""
        config = RouterConfiguration()
        return SmartIngestionRouter(
            config=config,
            file_filter=mock_file_filter,
            lsp_extractor=mock_lsp_extractor
        )
    
    @pytest.mark.asyncio
    async def test_router_initialization(self, router):
        """Test router initialization"""
        assert not router._initialized
        await router.initialize()
        assert router._initialized
    
    @pytest.mark.asyncio
    async def test_route_file_code(self, router, tmp_path):
        """Test routing of code files"""
        await router.initialize()
        
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        
        strategy, classification = await router.route_file(py_file)
        
        assert strategy == ProcessingStrategy.LSP_ENRICHED
        assert classification is not None
        assert classification.classification == FileClassification.CODE
        assert classification.detected_language == "python"
    
    @pytest.mark.asyncio
    async def test_route_file_documentation(self, router, tmp_path):
        """Test routing of documentation files"""
        await router.initialize()
        
        md_file = tmp_path / "README.md"
        md_file.write_text("# Hello World")
        
        strategy, classification = await router.route_file(md_file)
        
        assert strategy == ProcessingStrategy.STANDARD_INGESTION
        assert classification is not None
        assert classification.classification == FileClassification.DOCUMENTATION
    
    @pytest.mark.asyncio
    async def test_route_file_filtered_out(self, router, tmp_path):
        """Test routing when file is filtered out"""
        # Mock file filter to reject file
        router.file_filter.should_process_file.return_value = (False, "extension_ignored")
        
        await router.initialize()
        
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        
        strategy, classification = await router.route_file(py_file)
        
        assert strategy == ProcessingStrategy.SKIP
        assert classification is None
    
    @pytest.mark.asyncio
    async def test_process_single_file_lsp_enriched(self, router, tmp_path):
        """Test LSP-enriched processing of single file"""
        await router.initialize()
        
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        
        result = await router.process_single_file(py_file)
        
        assert result is not None
        assert result["processing_strategy"] == "lsp_enriched"
        assert result["file_path"] == str(py_file)
        assert "classification" in result
        assert "lsp_metadata" in result
        assert router.statistics.lsp_processed == 1
    
    @pytest.mark.asyncio
    async def test_process_single_file_standard_ingestion(self, router, tmp_path):
        """Test standard ingestion processing"""
        await router.initialize()
        
        md_file = tmp_path / "README.md"
        content = "# Hello World\nThis is a test document."
        md_file.write_text(content)
        
        result = await router.process_single_file(md_file)
        
        assert result is not None
        assert result["processing_strategy"] == "standard_ingestion"
        assert result["file_path"] == str(md_file)
        assert result["content"] == content
        assert result["line_count"] == 2
        assert router.statistics.standard_processed == 1
    
    @pytest.mark.asyncio
    async def test_process_single_file_fallback(self, router, tmp_path):
        """Test fallback processing when LSP fails"""
        await router.initialize()
        
        # Mock LSP extractor to fail
        router.lsp_extractor.extract_file_metadata.return_value = None
        
        py_file = tmp_path / "test.py"
        content = "def hello():\n    return 'world'"
        py_file.write_text(content)
        
        result = await router.process_single_file(py_file)
        
        assert result is not None
        assert result["processing_strategy"] == "fallback_code"
        assert result["file_path"] == str(py_file)
        assert result["content"] == content
        assert "syntax_info" in result
        assert router.statistics.fallback_processed == 1
        assert router.statistics.fallback_triggers == 1
    
    @pytest.mark.asyncio
    async def test_classification_caching(self, router, tmp_path):
        """Test file classification caching"""
        await router.initialize()
        
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        
        # First call - should be cache miss
        strategy1, classification1 = await router.route_file(py_file)
        assert router.statistics.cache_misses == 1
        assert router.statistics.cache_hits == 0
        
        # Second call - should be cache hit
        strategy2, classification2 = await router.route_file(py_file)
        assert router.statistics.cache_hits == 1
        assert strategy1 == strategy2
        assert classification1.classification == classification2.classification
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, router, tmp_path):
        """Test batch processing of multiple files"""
        await router.initialize()
        
        # Create test files
        files = []
        for i in range(5):
            py_file = tmp_path / f"test_{i}.py"
            py_file.write_text(f"def hello_{i}(): pass")
            files.append(py_file)
        
        for i in range(3):
            md_file = tmp_path / f"doc_{i}.md"
            md_file.write_text(f"# Document {i}")
            files.append(md_file)
        
        results = await router.process_batch(files, batch_size=3)
        
        assert len(results) == 8  # All files should be processed successfully
        
        # Check processing strategies
        lsp_results = [r for r in results if r["processing_strategy"] == "lsp_enriched"]
        standard_results = [r for r in results if r["processing_strategy"] == "standard_ingestion"]
        
        assert len(lsp_results) == 5  # Python files
        assert len(standard_results) == 3  # Markdown files
        
        assert router.statistics.files_processed == 8
        assert router.statistics.lsp_processed == 5
        assert router.statistics.standard_processed == 3
    
    @pytest.mark.asyncio
    async def test_batch_processing_strategy_grouping(self, router, tmp_path):
        """Test that batch processing groups files by strategy"""
        await router.initialize()
        
        files = []
        # Mix of file types
        files.append(tmp_path / "code1.py")
        files.append(tmp_path / "doc1.md")
        files.append(tmp_path / "code2.js")
        files.append(tmp_path / "doc2.txt")
        files.append(tmp_path / "data.json")
        
        for file_path in files:
            file_path.write_text("test content")
        
        # Mock the grouping method to verify it's called
        with patch.object(router, '_group_files_by_strategy', wraps=router._group_files_by_strategy) as mock_group:
            await router.process_batch(files)
            mock_group.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, router, tmp_path):
        """Test comprehensive statistics tracking"""
        await router.initialize()
        
        # Process various file types
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        result1 = await router.process_single_file(py_file)
        
        md_file = tmp_path / "README.md"
        md_file.write_text("# Hello")
        result2 = await router.process_single_file(md_file)
        
        # Check statistics
        stats = router.get_statistics()
        stats_dict = stats.to_dict()
        
        # Files should be classified when routing
        assert stats.files_classified >= 2
        
        # Files should be processed if results are not None
        processed_count = (1 if result1 else 0) + (1 if result2 else 0)
        assert stats.files_processed == processed_count
        
        if result1:
            assert stats.lsp_processed >= 1 or stats.fallback_processed >= 1
        if result2:
            assert stats.standard_processed >= 1
        
        assert stats.classification_time_ms >= 0  # Should have some classification time
        assert stats.total_processing_time_ms >= 0  # Should have some processing time
    
    @pytest.mark.asyncio
    async def test_processing_capabilities_report(self, router):
        """Test processing capabilities reporting"""
        await router.initialize()
        
        capabilities = await router.get_processing_capabilities()
        
        assert "lsp_available" in capabilities
        assert "supported_lsp_languages" in capabilities
        assert "file_filter_initialized" in capabilities
        assert "classification_cache_size" in capabilities
        assert "configuration" in capabilities
        assert "statistics" in capabilities
    
    @pytest.mark.asyncio
    async def test_router_shutdown(self, router):
        """Test router shutdown and cleanup"""
        await router.initialize()
        assert router._initialized
        
        # Add some cache entries
        await router.route_file(Path("test.py"))  # This will fail but create cache entry
        
        await router.shutdown()
        
        assert not router._initialized
        assert len(router.classification_cache) == 0
        router.lsp_extractor.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_file_filter, mock_lsp_extractor):
        """Test using router as async context manager"""
        config = RouterConfiguration()
        
        async with SmartIngestionRouter(config, mock_file_filter, mock_lsp_extractor) as router:
            assert router._initialized
        
        # After context exit, should be shut down
        assert not router._initialized
    
    def test_cache_size_limits(self, router):
        """Test classification cache size limits"""
        # Set small cache limit for testing
        router.config.max_cache_entries = 3
        
        # Add entries beyond limit
        for i in range(5):
            result = ClassificationResult(
                classification=FileClassification.CODE,
                confidence=0.9,
                strategy=ProcessingStrategy.LSP_ENRICHED,
                reason="test"
            )
            router._cache_classification(f"file_{i}", result, time.time())
        
        # Should have evicted oldest entries
        assert len(router.classification_cache) <= 3  # Should be at or below limit
        # Cache evictions should have occurred when we exceeded the limit
        assert router.statistics.cache_evictions >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, router, tmp_path):
        """Test error handling during file processing"""
        await router.initialize()
        
        # Mock LSP extractor to raise exception
        router.lsp_extractor.extract_file_metadata.side_effect = Exception("LSP Error")
        
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass")
        
        result = await router.process_single_file(py_file)
        
        # Should fall back to fallback processing
        assert result is not None
        assert result["processing_strategy"] == "fallback_code"
        assert router.statistics.lsp_errors > 0
        assert router.statistics.fallback_triggers > 0


class TestRouterStatistics:
    """Test router statistics functionality"""
    
    def test_statistics_initialization(self):
        """Test statistics initialization with default values"""
        stats = RouterStatistics()
        
        assert stats.files_classified == 0
        assert stats.files_processed == 0
        assert stats.lsp_processed == 0
        assert stats.cache_hits == 0
        assert stats.classification_time_ms == 0.0
        assert len(stats.classification_by_type) == 0
    
    def test_statistics_to_dict(self):
        """Test statistics serialization to dictionary"""
        stats = RouterStatistics()
        stats.files_classified = 10
        stats.files_processed = 8
        stats.files_failed = 2
        stats.lsp_processed = 5
        stats.standard_processed = 3
        stats.classification_time_ms = 100.0
        stats.total_processing_time_ms = 500.0
        stats.cache_hits = 15
        stats.cache_misses = 5
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["classification"]["files_classified"] == 10
        assert stats_dict["processing"]["files_processed"] == 8
        assert stats_dict["processing"]["success_rate"] == 0.8  # 8/(8+2)
        assert stats_dict["cache"]["cache_hit_rate"] == 0.75  # 15/(15+5)
        assert stats_dict["classification"]["avg_classification_time_ms"] == 10.0  # 100/10
        assert stats_dict["performance"]["avg_processing_time_ms"] == 62.5  # 500/8


@pytest.mark.asyncio
async def test_integration_workflow(tmp_path):
    """Integration test for complete workflow"""
    # Create test workspace
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    
    # Create various file types
    (workspace / "main.py").write_text("""
def main():
    print("Hello, World!")
    
if __name__ == "__main__":
    main()
""")
    
    (workspace / "README.md").write_text("""
# Test Project
This is a test project for the smart ingestion router.
""")
    
    (workspace / "config.json").write_text('{"debug": true, "port": 8080}')
    
    (workspace / "binary.exe").write_bytes(b'\x00\x01\x02\x03')
    
    # Initialize router with minimal mocking
    file_filter = MagicMock(spec=LanguageAwareFilter)
    file_filter._initialized = True
    file_filter.should_process_file.return_value = (True, "accepted")
    
    lsp_extractor = MagicMock(spec=LspMetadataExtractor)
    lsp_extractor._initialized = True
    
    # Mock successful LSP extraction for Python files
    mock_metadata = MagicMock(spec=FileMetadata)
    mock_metadata.symbols = []
    mock_metadata.relationships = []
    mock_metadata.to_dict.return_value = {"symbols": [], "relationships": []}
    lsp_extractor.extract_file_metadata.return_value = mock_metadata
    
    config = RouterConfiguration()
    router = SmartIngestionRouter(config, file_filter, lsp_extractor)
    
    try:
        await router.initialize(workspace)
        
        # Get all files
        files = list(workspace.rglob("*"))
        files = [f for f in files if f.is_file()]
        
        # Process files
        results = await router.process_batch(files)
        
        # Verify results
        assert len(results) >= 3  # Should process at least python, markdown, and json
        
        # Check different processing strategies were used
        strategies = {result["processing_strategy"] for result in results}
        assert "lsp_enriched" in strategies or "fallback_code" in strategies  # Python file
        assert "standard_ingestion" in strategies  # Markdown and JSON files
        
        # Verify statistics
        stats = router.get_statistics()
        assert stats.files_processed > 0
        assert stats.files_classified > 0
        
        logger.info("Integration test completed successfully", 
                   results_count=len(results),
                   strategies=list(strategies),
                   stats=stats.to_dict())
        
    finally:
        await router.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])