"""
Tests for file type detection system.
"""

import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from workspace_qdrant_mcp.cli.parsers.file_detector import (
    FileDetector,
    FileTypeDetectionError,
    UnsupportedFileTypeError,
    detect_file_type,
    get_supported_extensions,
    get_supported_mime_types,
    is_supported_file,
)


class TestFileDetector:
    """Test file type detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FileDetector()

    def test_detect_text_file(self, tmp_path):
        """Test detection of plain text file."""
        # Create a test text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, world!\nThis is a test file.")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(text_file)
        
        assert mime_type == "text/plain"
        assert parser_type == "text"
        assert confidence > 0.5

    def test_detect_markdown_file(self, tmp_path):
        """Test detection of Markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello World\n\nThis is a **markdown** file.")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(md_file)
        
        assert mime_type == "text/markdown"
        assert parser_type == "markdown"
        assert confidence > 0.5

    def test_detect_pdf_magic_number(self, tmp_path):
        """Test PDF detection using magic numbers."""
        # Create a minimal PDF file with PDF magic number
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%This is a test PDF")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(pdf_file)
        
        assert mime_type == "application/pdf"
        assert parser_type == "pdf"
        assert confidence >= 0.8

    def test_detect_html_file(self, tmp_path):
        """Test detection of HTML file."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<!DOCTYPE html>\n<html><body>Test</body></html>")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(html_file)
        
        assert "html" in mime_type.lower()
        assert parser_type == "html"
        assert confidence > 0.5

    def test_detect_docx_file(self, tmp_path):
        """Test detection of DOCX file."""
        # Create a minimal DOCX file (ZIP with specific structure)
        docx_file = tmp_path / "test.docx"
        
        with zipfile.ZipFile(docx_file, 'w') as zip_file:
            zip_file.writestr("word/document.xml", "<?xml version='1.0'?><document/>")
            zip_file.writestr("[Content_Types].xml", "<?xml version='1.0'?><Types/>")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(docx_file)
        
        assert mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert parser_type == "docx"
        assert confidence >= 0.8

    def test_detect_epub_file(self, tmp_path):
        """Test detection of EPUB file."""
        # Create a minimal EPUB file
        epub_file = tmp_path / "test.epub"
        
        with zipfile.ZipFile(epub_file, 'w') as zip_file:
            zip_file.writestr("META-INF/container.xml", "<?xml version='1.0'?><container/>")
            zip_file.writestr("mimetype", "application/epub+zip")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(epub_file)
        
        assert mime_type == "application/epub+zip"
        assert parser_type == "epub"
        assert confidence >= 0.8

    def test_detect_code_file(self, tmp_path):
        """Test detection of code files."""
        # Create a Python file
        py_file = tmp_path / "test.py"
        py_file.write_text("print('Hello, world!')\n")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(py_file)
        
        assert mime_type == "text/x-python"
        assert parser_type == "code"
        assert confidence > 0.5

    def test_extension_fallback(self, tmp_path):
        """Test extension-based fallback detection."""
        # Create a file with known extension but no magic number
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": "data"}')
        
        # Disable magic detection to force extension fallback
        detector = FileDetector(enable_magic=False)
        mime_type, parser_type, confidence = detector.detect_file_type(test_file)
        
        assert mime_type == "application/json"
        assert parser_type == "text"
        assert confidence <= 0.7  # Lower confidence for extension-only

    def test_text_content_analysis(self, tmp_path):
        """Test text file detection through content analysis."""
        # Create a file without extension
        text_file = tmp_path / "no_extension"
        text_file.write_text("This is plain text content without extension.")
        
        detector = FileDetector(enable_magic=False)
        mime_type, parser_type, confidence = detector.detect_file_type(text_file)
        
        assert mime_type == "text/plain"
        assert parser_type == "text"
        assert confidence <= 0.6

    def test_binary_file_detection(self, tmp_path):
        """Test detection of binary files."""
        # Create a binary file with null bytes
        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
        
        detector = FileDetector(enable_magic=False)
        mime_type, parser_type, confidence = detector.detect_file_type(binary_file)
        
        assert mime_type == "application/octet-stream"
        assert parser_type is None  # Should not have a parser for binary files
        assert confidence <= 0.2

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.detector.detect_file_type("nonexistent_file.txt")

    def test_directory_path(self, tmp_path):
        """Test handling of directory paths."""
        with pytest.raises(FileTypeDetectionError):
            self.detector.detect_file_type(tmp_path)

    def test_unsupported_file_type(self, tmp_path):
        """Test handling of unsupported file types."""
        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_bytes(b"\x89\x50\x4E\x47")  # PNG magic number
        
        detector = FileDetector(enable_magic=False)
        
        with pytest.raises(UnsupportedFileTypeError):
            detector.detect_file_type(unsupported_file)

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        extensions = get_supported_extensions()
        
        assert isinstance(extensions, list)
        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".md" in extensions
        assert len(extensions) > 10

    def test_get_supported_mime_types(self):
        """Test getting list of supported MIME types."""
        mime_types = get_supported_mime_types()
        
        assert isinstance(mime_types, list)
        assert "text/plain" in mime_types
        assert "application/pdf" in mime_types
        assert "text/markdown" in mime_types
        assert len(mime_types) > 10

    def test_is_supported_file(self, tmp_path):
        """Test checking if file is supported."""
        # Supported file
        text_file = tmp_path / "test.txt"
        text_file.write_text("Test content")
        
        assert is_supported_file(text_file) is True
        
        # Unsupported file
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"\x89\x50\x4E\x47")
        
        assert is_supported_file(image_file) is False

    @patch('workspace_qdrant_mcp.cli.parsers.file_detector.HAS_MAGIC', False)
    def test_without_magic_library(self, tmp_path):
        """Test detection when python-magic is not available."""
        detector = FileDetector()
        
        text_file = tmp_path / "test.txt"
        text_file.write_text("Test content")
        
        mime_type, parser_type, confidence = detector.detect_file_type(text_file)
        
        # Should still work with extension-based detection
        assert mime_type == "text/plain"
        assert parser_type == "text"
        assert confidence <= 0.7  # Lower confidence without magic

    def test_zip_based_format_detection_error(self, tmp_path):
        """Test handling of corrupted ZIP-based files."""
        # Create a corrupted DOCX file
        docx_file = tmp_path / "corrupted.docx"
        docx_file.write_bytes(b"PK\x03\x04corrupted_zip_data")
        
        # Should fall back to extension-based detection
        mime_type, parser_type, confidence = self.detector.detect_file_type(docx_file)
        
        # Should detect as DOCX based on extension fallback
        assert "wordprocessingml" in mime_type or parser_type == "docx"

    def test_utf8_bom_detection(self, tmp_path):
        """Test detection of UTF-8 BOM in text files."""
        # Create a file with UTF-8 BOM
        bom_file = tmp_path / "bom.txt"
        bom_file.write_bytes(b'\xef\xbb\xbfHello World')
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(bom_file)
        
        assert mime_type == "text/plain"
        assert parser_type == "text"
        assert confidence >= 0.5

    def test_confidence_scoring(self, tmp_path):
        """Test confidence scoring for different detection methods."""
        text_file = tmp_path / "test.txt" 
        text_file.write_text("Test content")
        
        # Magic detection should have higher confidence
        if self.detector.enable_magic:
            _, _, magic_confidence = self.detector.detect_file_type(text_file)
            
            # Extension-only detection
            detector_no_magic = FileDetector(enable_magic=False)
            _, _, ext_confidence = detector_no_magic.detect_file_type(text_file)
            
            assert magic_confidence >= ext_confidence

    def test_multiple_extension_formats(self, tmp_path):
        """Test files with multiple potential formats."""
        # Create a .yml file (could be YAML or text)
        yaml_file = tmp_path / "config.yml" 
        yaml_file.write_text("key: value\nother_key: other_value")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(yaml_file)
        
        assert mime_type == "text/yaml"
        assert parser_type == "text"  # YAML handled as text
        assert confidence > 0.5

    def test_empty_file(self, tmp_path):
        """Test detection of empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        mime_type, parser_type, confidence = self.detector.detect_file_type(empty_file)
        
        assert mime_type == "text/plain"
        assert parser_type == "text"
        assert confidence > 0.0  # Should still detect based on extension


class TestFileDetectorIntegration:
    """Integration tests for file detector."""

    def test_real_file_detection(self):
        """Test detection on actual files if available."""
        # This test would require actual sample files
        # For now, just test that the module functions are callable
        extensions = get_supported_extensions()
        mime_types = get_supported_mime_types()
        
        assert len(extensions) > 0
        assert len(mime_types) > 0
        
        # Test non-existent file
        assert is_supported_file("definitely_nonexistent_file.xyz") is False

    def test_detector_consistency(self, tmp_path):
        """Test that multiple detectors give consistent results."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Consistent content")
        
        detector1 = FileDetector()
        detector2 = FileDetector()
        
        result1 = detector1.detect_file_type(text_file)
        result2 = detector2.detect_file_type(text_file)
        
        assert result1[0] == result2[0]  # Same MIME type
        assert result1[1] == result2[1]  # Same parser type