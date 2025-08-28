"""
Unit tests for CLI document parsers.

Tests the document parser implementations for text, markdown, and PDF formats
including format detection, content extraction, metadata generation, and error handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from workspace_qdrant_mcp.cli.parsers.base import DocumentParser, ParsedDocument
from workspace_qdrant_mcp.cli.parsers.text_parser import TextParser
from workspace_qdrant_mcp.cli.parsers.markdown_parser import MarkdownParser
from workspace_qdrant_mcp.cli.parsers.pdf_parser import PDFParser


class TestDocumentParserBase:
    """Test the base DocumentParser interface."""
    
    def test_parsed_document_creation(self):
        """Test ParsedDocument creation with auto-generated metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            f.flush()
            
            doc = ParsedDocument.create(
                content="test content",
                file_path=f.name,
                file_type="text",
                additional_metadata={"author": "test"}
            )
            
            assert doc.content == "test content"
            assert doc.file_type == "text"
            assert doc.metadata["author"] == "test"
            assert doc.metadata["content_length"] == 12
            assert "filename" in doc.metadata
            assert len(doc.content_hash) == 64  # SHA256 hash length
            assert doc.parsed_at is not None
    
    def test_parsed_document_hash_consistency(self):
        """Test that identical content produces identical hashes."""
        doc1 = ParsedDocument.create("test content", "/tmp/file1.txt", "text")
        doc2 = ParsedDocument.create("test content", "/tmp/file2.txt", "text")
        
        assert doc1.content_hash == doc2.content_hash
    
    def test_parsed_document_hash_uniqueness(self):
        """Test that different content produces different hashes."""
        doc1 = ParsedDocument.create("content A", "/tmp/file1.txt", "text")
        doc2 = ParsedDocument.create("content B", "/tmp/file2.txt", "text")
        
        assert doc1.content_hash != doc2.content_hash


class TestTextParser:
    """Test the plain text parser."""
    
    @pytest.fixture
    def parser(self):
        return TextParser()
    
    def test_supported_extensions(self, parser):
        """Test that parser reports correct supported extensions."""
        extensions = parser.supported_extensions
        assert '.txt' in extensions
        assert '.py' in extensions
        assert '.md' not in extensions  # Should be handled by MarkdownParser
    
    def test_format_name(self, parser):
        """Test format name reporting."""
        assert parser.format_name == "Plain Text"
    
    def test_can_parse(self, parser):
        """Test file format detection."""
        assert parser.can_parse("test.txt")
        assert parser.can_parse("script.py")
        assert not parser.can_parse("document.pdf")
    
    @pytest.mark.asyncio
    async def test_parse_simple_text(self, parser):
        """Test parsing a simple text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            content = "Hello, world!\nThis is a test file."
            f.write(content)
            f.flush()
            
            result = await parser.parse(f.name)
            
            assert isinstance(result, ParsedDocument)
            assert result.content.strip() == content
            assert result.file_type == "text"
            assert result.metadata["word_count"] == 7
            assert result.metadata["character_count"] == len(content)
            
            Path(f.name).unlink()  # Cleanup
    
    @pytest.mark.asyncio
    async def test_parse_with_encoding_detection(self, parser):
        """Test automatic encoding detection."""
        with patch('workspace_qdrant_mcp.cli.parsers.text_parser.chardet.detect') as mock_detect:
            mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                content = "Test with special chars: café"
                f.write(content)
                f.flush()
                
                result = await parser.parse(f.name, detect_encoding=True)
                
                assert result.content.strip() == content
                assert result.parsing_info["encoding"] == "utf-8"
                assert result.parsing_info["encoding_confidence"] == 0.9
                
                Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_parse_with_content_cleaning(self, parser):
        """Test content cleaning functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Content with excessive whitespace
            content = "Line 1\n\n\n\nLine 2   \n   \n\nLine 3\n\n\n"
            f.write(content)
            f.flush()
            
            result = await parser.parse(f.name, clean_content=True)
            
            # Should have reduced excessive whitespace
            assert result.content.count('\n\n\n') == 0
            assert result.parsing_info["content_cleaned"] is True
            assert result.parsing_info["size_reduction"] > 0
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_parse_programming_file(self, parser):
        """Test parsing of programming files with language detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            content = 'def hello():\n    print("Hello, world!")'
            f.write(content)
            f.flush()
            
            result = await parser.parse(f.name)
            
            assert result.metadata["content_type"] == "code"
            assert result.metadata["language"] == "python"
            assert result.metadata["word_count"] > 0
            
            Path(f.name).unlink()
    
    def test_content_analysis(self, parser):
        """Test text analysis functionality."""
        content = "Hello world.\n\nThis is a test with multiple words.\nAnother line."
        stats = parser._analyze_text(content)
        
        assert stats["word_count"] == 11
        assert stats["line_count"] == 4
        assert stats["paragraph_count"] == 2
        assert stats["character_count"] == len(content)
    
    def test_language_detection(self, parser):
        """Test programming language detection."""
        assert parser._detect_language('.py') == 'python'
        assert parser._detect_language('.js') == 'javascript'
        assert parser._detect_language('.txt') == 'text'
    
    def test_get_parsing_options(self, parser):
        """Test parsing options reporting."""
        options = parser.get_parsing_options()
        assert "encoding" in options
        assert "clean_content" in options
        assert options["clean_content"]["default"] is True


class TestMarkdownParser:
    """Test the Markdown parser."""
    
    @pytest.fixture
    def parser(self):
        return MarkdownParser()
    
    def test_supported_extensions(self, parser):
        """Test supported extensions."""
        extensions = parser.supported_extensions
        assert '.md' in extensions
        assert '.markdown' in extensions
    
    def test_format_name(self, parser):
        """Test format name."""
        assert parser.format_name == "Markdown"
    
    @pytest.mark.asyncio
    async def test_parse_simple_markdown(self, parser):
        """Test parsing basic Markdown content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            content = """# Title
            
This is **bold** text and *italic* text.

## Subsection

- Item 1
- Item 2

[Link](https://example.com)
"""
            f.write(content)
            f.flush()
            
            result = await parser.parse(f.name)
            
            assert isinstance(result, ParsedDocument)
            assert result.file_type == "markdown"
            assert "Title" in result.content
            assert result.metadata["heading_count"] == 2
            assert result.metadata["link_count"] == 1
            assert result.metadata["word_count"] > 0
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_parse_with_frontmatter(self, parser):
        """Test parsing Markdown with YAML frontmatter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            content = """---
title: Test Document
author: Test Author
tags: [test, markdown]
---

# Main Content

This is the document content.
"""
            f.write(content)
            f.flush()
            
            result = await parser.parse(f.name, extract_frontmatter=True)
            
            assert result.metadata["fm_title"] == "Test Document"
            assert result.metadata["fm_author"] == "Test Author"
            assert "Main Content" in result.content
            assert result.parsing_info["has_frontmatter"] is True
            
            Path(f.name).unlink()
    
    def test_frontmatter_extraction(self, parser):
        """Test YAML frontmatter extraction."""
        content = """---
title: Test
author: Author
---

Content here"""
        
        frontmatter, remaining = parser._extract_frontmatter(content)
        
        # Should work even without yaml library
        assert "title" in frontmatter or frontmatter == {}
        assert "Content here" in remaining
    
    def test_markdown_structure_analysis(self, parser):
        """Test analysis of Markdown structure."""
        content = """# Main Title

## Section 1

This is a paragraph with a [link](http://example.com).

### Subsection

```python
code block
```

- List item 1
- List item 2

![Image](image.png)
"""
        
        structure = parser._analyze_markdown_structure(content)
        
        assert structure["heading_count"] == 3
        assert structure["code_block_count"] == 1
        assert structure["link_count"] == 1
        assert structure["image_count"] == 1
        assert structure["max_heading_level"] == 3
        assert structure["min_heading_level"] == 1
    
    def test_structured_text_conversion(self, parser):
        """Test conversion to structured plain text."""
        content = """# Main Title

## Section

This is **bold** text.

- Item 1
- Item 2

```code
some code
```
"""
        
        result = parser._convert_to_structured_text(content, True, False)
        
        assert "Main Title" in result
        assert "Section" in result
        assert "bold" in result  # Bold formatting removed
        assert "• Item 1" in result  # List converted
        assert "[CODE BLOCK" in result  # Code block marker
    
    def test_get_parsing_options(self, parser):
        """Test parsing options."""
        options = parser.get_parsing_options()
        assert "extract_frontmatter" in options
        assert "preserve_structure" in options
        assert "include_code_blocks" in options


class TestPDFParser:
    """Test the PDF parser."""
    
    @pytest.fixture
    def parser(self):
        return PDFParser()
    
    def test_supported_extensions(self, parser):
        """Test supported extensions."""
        assert parser.supported_extensions == ['.pdf']
    
    def test_format_name(self, parser):
        """Test format name."""
        assert parser.format_name == "PDF Document"
    
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.HAS_PYPDF2', False)
    def test_can_parse_without_pypdf2(self, parser):
        """Test parser availability check without PyPDF2."""
        assert not parser.can_parse("test.pdf")
    
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.HAS_PYPDF2', True)
    def test_can_parse_with_pypdf2(self, parser):
        """Test parser availability with PyPDF2."""
        assert parser.can_parse("test.pdf")
        assert not parser.can_parse("test.txt")
    
    @pytest.mark.asyncio
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.HAS_PYPDF2', False)
    async def test_parse_without_pypdf2_raises_error(self, parser):
        """Test that parsing without PyPDF2 raises ImportError."""
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            with pytest.raises(ImportError, match="PDF parsing requires"):
                await parser.parse(f.name)
    
    @pytest.mark.asyncio
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.HAS_PYPDF2', True)
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.PyPDF2')
    async def test_parse_simple_pdf(self, mock_pypdf2, parser):
        """Test parsing a simple PDF."""
        # Mock PyPDF2 components
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is page content."
        
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {
            '/Title': 'Test Document',
            '/Author': 'Test Author'
        }
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            # Write some dummy PDF data
            f.write(b'%PDF-1.4 dummy content')
            f.flush()
            
            result = await parser.parse(f.name)
            
            assert isinstance(result, ParsedDocument)
            assert result.file_type == "pdf"
            assert "This is page content." in result.content
            assert result.metadata["title"] == "Test Document"
            assert result.metadata["author"] == "Test Author"
            assert result.metadata["page_count"] == 1
            assert result.parsing_info["total_pages"] == 1
    
    @pytest.mark.asyncio
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.HAS_PYPDF2', True)
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.PyPDF2')
    async def test_parse_encrypted_pdf_with_password(self, mock_pypdf2, parser):
        """Test parsing an encrypted PDF with password."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = True
        mock_reader.pages = []
        mock_reader.metadata = {}
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            f.write(b'%PDF-1.4 encrypted')
            f.flush()
            
            result = await parser.parse(f.name, password="test123")
            
            assert result.parsing_info["encrypted"] is True
            assert result.parsing_info["decrypted"] is True
            mock_reader.decrypt.assert_called_once_with("test123")
    
    @pytest.mark.asyncio
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.HAS_PYPDF2', True)
    @patch('workspace_qdrant_mcp.cli.parsers.pdf_parser.PyPDF2')
    async def test_parse_encrypted_pdf_without_password_raises_error(self, mock_pypdf2, parser):
        """Test that encrypted PDF without password raises error."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            f.write(b'%PDF-1.4 encrypted')
            f.flush()
            
            with pytest.raises(RuntimeError, match="PDF is encrypted but no password provided"):
                await parser.parse(f.name)
    
    def test_pdf_date_parsing(self, parser):
        """Test PDF date format parsing."""
        pdf_date = "D:20240115123456"
        result = parser._parse_pdf_date(pdf_date)
        assert result == "2024-01-15T12:34:56"
        
        # Test with timezone info
        pdf_date_tz = "D:20240115123456+05'00'"
        result_tz = parser._parse_pdf_date(pdf_date_tz)
        assert result_tz == "2024-01-15T12:34:56"
        
        # Test invalid format returns original
        invalid_date = "invalid"
        assert parser._parse_pdf_date(invalid_date) == invalid_date
    
    def test_pdf_text_cleaning(self, parser):
        """Test PDF text cleaning."""
        raw_text = "Line 1\x0cPage break\xa0Non-breaking space\n  \n\nLine 2"
        cleaned = parser._clean_pdf_text(raw_text)
        
        assert "\x0c" not in cleaned  # Form feed removed
        assert "\xa0" not in cleaned  # Non-breaking space converted
        assert "Page break" in cleaned
        assert "Non-breaking space" in cleaned
    
    def test_pdf_content_analysis(self, parser):
        """Test PDF content analysis."""
        content = "This is a test document with multiple words."
        analysis = parser._analyze_pdf_content(content, 1)
        
        assert analysis["word_count"] == 9
        assert analysis["character_count"] == len(content)
        assert analysis["pages_with_content"] == 1
        assert analysis["avg_words_per_page"] == 9.0
    
    def test_get_parsing_options(self, parser):
        """Test parsing options."""
        options = parser.get_parsing_options()
        assert "extract_metadata" in options
        assert "include_page_numbers" in options
        assert "max_pages" in options
        assert "password" in options


@pytest.mark.unit
class TestParserErrorHandling:
    """Test error handling across all parsers."""
    
    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_error(self):
        """Test that non-existent files raise FileNotFoundError."""
        parser = TextParser()
        
        with pytest.raises(FileNotFoundError):
            await parser.parse("/nonexistent/file.txt")
    
    @pytest.mark.asyncio
    async def test_unsupported_format_validation(self):
        """Test validation of unsupported file formats."""
        parser = TextParser()
        
        with pytest.raises(ValueError, match="File format not supported"):
            await parser.parse("test.unsupported")
    
    @pytest.mark.asyncio
    async def test_directory_instead_of_file_raises_error(self):
        """Test that directories raise appropriate errors."""
        parser = TextParser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with .txt extension
            fake_file = Path(tmpdir) / "fake.txt"
            fake_file.mkdir()
            
            with pytest.raises(ValueError, match="Path is not a file"):
                await parser.parse(str(fake_file))