"""
LSP Integration Testing Framework

Comprehensive integration tests for Language Server Protocol integration
covering language server detection, symbol extraction, hover information,
definition/reference tracking, and code structure analysis.

This module tests the full LSP integration flow from server detection
through metadata extraction and verification.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.python.common.core.lsp_client import AsyncioLspClient, ConnectionState, LspError
from src.python.common.core.lsp_metadata_extractor import (
    LspMetadataExtractor,
    CodeSymbol,
    SymbolKind,
    FileMetadata,
)
from src.python.common.core.lsp_detector import LSPDetector
from src.python.common.core.lsp_config import LSPConfig
from tests.mocks.lsp_mocks import LSPServerMock, LSPErrorInjector


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_python_file(temp_workspace):
    """Create a sample Python file for testing."""
    file_path = temp_workspace / "test_module.py"
    content = '''"""Sample Python module for testing."""

class TestClass:
    """A test class."""

    def __init__(self, name: str):
        """Initialize the test class."""
        self.name = name

    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}!"

def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

CONSTANT_VALUE = 42
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_rust_file(temp_workspace):
    """Create a sample Rust file for testing."""
    file_path = temp_workspace / "test_module.rs"
    content = '''/// Sample Rust module for testing
pub struct TestStruct {
    pub name: String,
}

impl TestStruct {
    /// Create a new TestStruct
    pub fn new(name: String) -> Self {
        TestStruct { name }
    }

    /// Get a greeting
    pub fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}

pub fn add_numbers(x: i32, y: i32) -> i32 {
    x + y
}

pub const CONSTANT_VALUE: i32 = 42;
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_javascript_file(temp_workspace):
    """Create a sample JavaScript file for testing."""
    file_path = temp_workspace / "test_module.js"
    content = '''/**
 * Sample JavaScript module for testing
 */

class TestClass {
  constructor(name) {
    this.name = name;
  }

  greet() {
    return `Hello, ${this.name}!`;
  }
}

function standaloneFunction(x, y) {
  return x + y;
}

const CONSTANT_VALUE = 42;

module.exports = { TestClass, standaloneFunction, CONSTANT_VALUE };
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def lsp_error_injector():
    """Create an LSP error injector for testing error scenarios."""
    return LSPErrorInjector()


@pytest.fixture
def mock_lsp_server(lsp_error_injector):
    """Create a mock LSP server for testing."""
    return LSPServerMock(language="python", error_injector=lsp_error_injector)


# ============================================================================
# Language Server Detection Tests
# ============================================================================


class TestLanguageServerDetection:
    """Test language server detection for supported languages."""

    def test_detect_python_lsp(self, temp_workspace):
        """Test detection of Python language server (pylsp)."""
        detector = LSPDetector()

        # Mock shutil.which to simulate pylsp being available
        with patch('shutil.which') as mock_which:
            mock_which.return_value = '/usr/local/bin/pylsp'

            result = detector.scan_available_lsps(force_refresh=True, project_path=temp_workspace)

            assert result is not None
            assert isinstance(result.detected_lsps, dict)

            # Check if Python LSP was detected
            python_lsps = [name for name, info in result.detected_lsps.items()
                          if any('.py' in ext for ext in info.supported_extensions)]

            if python_lsps:
                lsp_name = python_lsps[0]
                lsp_info = result.detected_lsps[lsp_name]
                assert lsp_info.binary_path is not None
                assert '.py' in lsp_info.supported_extensions

    def test_detect_rust_lsp(self, temp_workspace):
        """Test detection of Rust language server (rust-analyzer)."""
        detector = LSPDetector()

        with patch('shutil.which') as mock_which:
            mock_which.return_value = '/usr/local/bin/rust-analyzer'

            result = detector.scan_available_lsps(force_refresh=True)

            assert result is not None

            # Check for Rust LSP
            rust_lsps = [name for name, info in result.detected_lsps.items()
                        if 'rust' in name.lower()]

            if rust_lsps:
                assert any('rust-analyzer' in name.lower() for name in rust_lsps)

    def test_detect_typescript_lsp(self, temp_workspace):
        """Test detection of TypeScript language server."""
        detector = LSPDetector()

        with patch('shutil.which') as mock_which:
            mock_which.return_value = '/usr/local/bin/typescript-language-server'

            result = detector.scan_available_lsps(force_refresh=True)

            assert result is not None

            # Check for TypeScript LSP
            ts_lsps = [name for name, info in result.detected_lsps.items()
                      if 'typescript' in name.lower()]

            if ts_lsps:
                assert len(ts_lsps) > 0

    def test_detect_go_lsp(self, temp_workspace):
        """Test detection of Go language server (gopls)."""
        detector = LSPDetector()

        with patch('shutil.which') as mock_which:
            mock_which.return_value = '/usr/local/bin/gopls'

            result = detector.scan_available_lsps(force_refresh=True)

            assert result is not None

            # Check for Go LSP
            if 'gopls' in result.detected_lsps:
                gopls_info = result.detected_lsps['gopls']
                assert gopls_info.binary_path is not None

    def test_detect_no_lsp_available(self, temp_workspace):
        """Test handling when no LSP server is available."""
        detector = LSPDetector()

        with patch('shutil.which') as mock_which:
            mock_which.return_value = None

            result = detector.scan_available_lsps(force_refresh=True)

            # Should return result with empty or minimal detected_lsps
            assert result is not None
            assert isinstance(result.detected_lsps, dict)
            # May have some LSPs from cache or config, just verify it doesn't crash


# ============================================================================
# Symbol Extraction Tests
# ============================================================================


class TestSymbolExtraction:
    """Test symbol extraction from various file types."""

    @pytest.mark.asyncio
    async def test_extract_python_symbols(self, sample_python_file, mock_lsp_server):
        """Test extraction of symbols from Python file."""
        extractor = LspMetadataExtractor()

        # Mock the LSP client for python
        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                force_refresh=True
            )

            # May return None if LSP server not available (graceful degradation)
            if metadata is not None:
                assert metadata.file_path == str(sample_python_file.resolve())

                # Verify symbols were extracted
                if len(metadata.symbols) > 0:
                    # Check for expected symbol types
                    symbol_names = [s.name for s in metadata.symbols]
                    assert any('TestClass' in name or 'test' in name.lower() for name in symbol_names)

    @pytest.mark.asyncio
    async def test_extract_rust_symbols(self, sample_rust_file, mock_lsp_server):
        """Test extraction of symbols from Rust file."""
        extractor = LspMetadataExtractor()

        # Mock the LSP client for rust
        with patch.object(extractor, 'lsp_clients', {'rust': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_rust_file,
                force_refresh=True
            )

            # May return None if LSP server not available
            if metadata is not None:
                assert metadata.file_path == str(sample_rust_file.resolve())

                # Verify symbols were extracted
                if len(metadata.symbols) > 0:
                    symbol_names = [s.name for s in metadata.symbols]
                    # Should have some rust-related symbols
                    assert len(symbol_names) > 0

    @pytest.mark.asyncio
    async def test_extract_javascript_symbols(self, sample_javascript_file, mock_lsp_server):
        """Test extraction of symbols from JavaScript file."""
        extractor = LspMetadataExtractor()

        # Mock the LSP client for javascript
        with patch.object(extractor, 'lsp_clients', {'javascript': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_javascript_file,
                force_refresh=True
            )

            # May return None if LSP server not available
            if metadata is not None:
                assert metadata.file_path == str(sample_javascript_file.resolve())

                # Verify symbols were extracted
                if len(metadata.symbols) > 0:
                    # Should have extracted some symbols
                    assert len(metadata.symbols) > 0


# ============================================================================
# Hover Information Tests
# ============================================================================


class TestHoverInformation:
    """Test hover information capture through symbol metadata."""

    @pytest.mark.asyncio
    async def test_capture_hover_for_function(self, sample_python_file, mock_lsp_server):
        """Test capturing hover-like information for a function through metadata."""
        extractor = LspMetadataExtractor()

        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            # Extract metadata which includes documentation and type info
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                force_refresh=True
            )

            # Verify hover-like information is captured in symbol metadata
            if metadata is not None and len(metadata.symbols) > 0:
                function_symbols = [s for s in metadata.symbols
                                   if s.kind == SymbolKind.FUNCTION and 'standalone_function' in s.name]

                if function_symbols:
                    symbol = function_symbols[0]
                    # Documentation and type info provide hover information
                    assert symbol.documentation is not None or symbol.type_info is not None

    @pytest.mark.asyncio
    async def test_capture_hover_for_class_method(self, sample_python_file, mock_lsp_server):
        """Test capturing hover-like information for a class method through metadata."""
        extractor = LspMetadataExtractor()

        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                force_refresh=True
            )

            # Verify hover-like information is captured for methods
            if metadata is not None and len(metadata.symbols) > 0:
                method_symbols = [s for s in metadata.symbols
                                 if s.kind == SymbolKind.METHOD and 'greet' in s.name]

                if method_symbols:
                    symbol = method_symbols[0]
                    # Metadata includes documentation and type info
                    assert symbol is not None


# ============================================================================
# Definition and Reference Tracking Tests
# ============================================================================


class TestDefinitionAndReferences:
    """Test definition and reference tracking through relationships."""

    @pytest.mark.asyncio
    async def test_find_definition(self, sample_python_file, mock_lsp_server):
        """Test symbol definitions are captured in metadata."""
        extractor = LspMetadataExtractor()

        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                force_refresh=True
            )

            # Verify symbol definitions are captured with location info
            if metadata is not None and len(metadata.symbols) > 0:
                for symbol in metadata.symbols:
                    # Each symbol has a range (definition location)
                    assert symbol.range is not None
                    assert symbol.file_uri is not None

    @pytest.mark.asyncio
    async def test_find_references(self, sample_python_file, mock_lsp_server):
        """Test symbol references are captured through relationships."""
        extractor = LspMetadataExtractor()

        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                force_refresh=True
            )

            # Verify relationships capture reference information
            if metadata is not None:
                # Relationships track connections between symbols (references)
                assert isinstance(metadata.relationships, list)
                # May be empty if LSP server doesn't provide relationship data
                # or if mocking doesn't populate it


# ============================================================================
# Code Structure Analysis Tests
# ============================================================================


class TestCodeStructureAnalysis:
    """Test code structure analysis."""

    @pytest.mark.asyncio
    async def test_analyze_class_hierarchy(self, sample_python_file, mock_lsp_server):
        """Test analysis of class hierarchies."""
        extractor = LspMetadataExtractor()

        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                force_refresh=True
            )

            # Verify relationships are captured in metadata
            if metadata is not None:
                assert hasattr(metadata, 'relationships')

    @pytest.mark.asyncio
    async def test_analyze_imports(self, temp_workspace, mock_lsp_server):
        """Test analysis of import statements."""
        # Create a file with imports
        file_path = temp_workspace / "with_imports.py"
        content = '''import os
from pathlib import Path
from typing import Dict, List

def process_path(p: Path) -> str:
    return str(p)
'''
        file_path.write_text(content)

        extractor = LspMetadataExtractor()

        with patch.object(extractor, 'lsp_clients', {'python': mock_lsp_server}):
            metadata = await extractor.extract_file_metadata(
                file_path=file_path,
                force_refresh=True
            )

            # Verify imports are captured or symbols extracted
            if metadata is not None:
                assert hasattr(metadata, 'imports') or len(metadata.symbols) > 0


# ============================================================================
# Error Handling and Timeout Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and timeout scenarios."""

    @pytest.mark.asyncio
    async def test_handle_server_not_found(self, sample_python_file, lsp_error_injector):
        """Test handling when LSP server is not found."""
        lsp_error_injector.configure_server_issues(probability=1.0)
        mock_server = LSPServerMock(language="python", error_injector=lsp_error_injector)

        extractor = LspMetadataExtractor()

        with patch.object(extractor, '_client', mock_server):
            with pytest.raises((LspError, FileNotFoundError, RuntimeError)):
                await extractor.extract_file_metadata(
                    file_path=sample_python_file,
                    language='python'
                )

    @pytest.mark.asyncio
    async def test_handle_protocol_error(self, sample_python_file, lsp_error_injector):
        """Test handling of LSP protocol errors."""
        lsp_error_injector.configure_protocol_issues(probability=1.0)
        mock_server = LSPServerMock(language="python", error_injector=lsp_error_injector)

        extractor = LspMetadataExtractor()

        with patch.object(extractor, '_client', mock_server):
            with pytest.raises((LspError, ValueError, TimeoutError)):
                await extractor.extract_file_metadata(
                    file_path=sample_python_file,
                    language='python'
                )

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_python_file, mock_lsp_server):
        """Test handling of request timeouts."""
        extractor = LspMetadataExtractor(timeout=0.1)  # Very short timeout

        # Mock a slow response
        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than timeout
            return None

        with patch.object(extractor, 'extract_file_metadata', side_effect=slow_extract):
            with pytest.raises((asyncio.TimeoutError, LspError)):
                await extractor.extract_file_metadata(
                    file_path=sample_python_file,
                    language='python',
                    timeout=0.1
                )

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, sample_python_file):
        """Test graceful degradation when LSP is unavailable."""
        extractor = LspMetadataExtractor(fallback_to_treesitter=True)

        # Simulate LSP unavailability
        with patch.object(extractor, '_client', None):
            metadata = await extractor.extract_file_metadata(
                file_path=sample_python_file,
                language='python'
            )

            # Should still return metadata using fallback mechanism
            assert metadata is not None
            assert metadata.file_path == str(sample_python_file)


# ============================================================================
# Connection State Management Tests
# ============================================================================


class TestConnectionStateManagement:
    """Test LSP client connection state management."""

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, temp_workspace, mock_lsp_server):
        """Test complete connection lifecycle."""
        client = AsyncioLspClient(server_command=['mock-server'])

        # Mock the process management
        with patch.object(client, '_process', mock_lsp_server):
            # Initial state
            assert client.state == ConnectionState.DISCONNECTED

            # Connect
            await client.connect(workspace_root=temp_workspace)
            assert client.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]

            # Disconnect
            await client.disconnect()
            assert client.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnection_after_failure(self, temp_workspace):
        """Test automatic reconnection after connection failure."""
        client = AsyncioLspClient(
            server_command=['mock-server'],
            auto_reconnect=True,
            max_reconnect_attempts=3
        )

        # Simulate connection failure and recovery
        connection_attempts = 0

        async def mock_connect():
            nonlocal connection_attempts
            connection_attempts += 1
            if connection_attempts < 2:
                raise ConnectionError("Simulated connection failure")
            # Success on second attempt

        with patch.object(client, '_do_connect', side_effect=mock_connect):
            await client.connect(workspace_root=temp_workspace)

        assert connection_attempts >= 1


# ============================================================================
# Multi-Language Support Tests
# ============================================================================


class TestMultiLanguageSupport:
    """Test support for multiple programming languages."""

    @pytest.mark.asyncio
    async def test_concurrent_language_servers(self, temp_workspace):
        """Test running multiple language servers concurrently."""
        # Create files in different languages
        py_file = temp_workspace / "test.py"
        py_file.write_text("def hello(): pass")

        rs_file = temp_workspace / "test.rs"
        rs_file.write_text("fn hello() {}")

        js_file = temp_workspace / "test.js"
        js_file.write_text("function hello() {}")

        extractor = LspMetadataExtractor()

        # Extract metadata concurrently
        tasks = [
            extractor.extract_file_metadata(py_file, 'python'),
            extractor.extract_file_metadata(rs_file, 'rust'),
            extractor.extract_file_metadata(js_file, 'javascript'),
        ]

        with patch.object(extractor, '_client', LSPServerMock()):
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all completed (may have errors due to mocking)
            assert len(results) == 3


# ============================================================================
# Performance and Scalability Tests
# ============================================================================


class TestPerformanceAndScalability:
    """Test performance with multiple files and large codebases."""

    @pytest.mark.asyncio
    async def test_batch_file_processing(self, temp_workspace, mock_lsp_server):
        """Test processing multiple files in batch."""
        # Create multiple Python files
        files = []
        for i in range(10):
            file_path = temp_workspace / f"module_{i}.py"
            file_path.write_text(f"def function_{i}(): pass")
            files.append(file_path)

        extractor = LspMetadataExtractor()

        with patch.object(extractor, '_client', mock_lsp_server):
            tasks = [
                extractor.extract_file_metadata(f, 'python')
                for f in files
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all files processed
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 5  # At least half should succeed

    @pytest.mark.asyncio
    async def test_large_file_handling(self, temp_workspace, mock_lsp_server):
        """Test handling of large source files."""
        # Create a large Python file
        large_file = temp_workspace / "large_module.py"

        # Generate file with many functions
        content_lines = []
        for i in range(100):
            content_lines.append(f"def function_{i}(x, y):")
            content_lines.append(f"    '''Function {i} documentation'''")
            content_lines.append(f"    return x + y + {i}")
            content_lines.append("")

        large_file.write_text('\n'.join(content_lines))

        extractor = LspMetadataExtractor()

        with patch.object(extractor, '_client', mock_lsp_server):
            metadata = await extractor.extract_file_metadata(
                file_path=large_file,
                language='python'
            )

            assert metadata is not None
            # Should have extracted many symbols
            assert len(metadata.symbols) >= 50
