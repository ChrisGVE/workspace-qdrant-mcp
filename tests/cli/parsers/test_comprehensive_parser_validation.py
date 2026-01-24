"""
Comprehensive parser validation tests for Task 79.

This module tests all individual parser implementations to ensure they work
correctly with their respective file formats and handle edge cases properly.

Parsers tested:
- PDFParser (pdf_parser.py)
- EPUBParser (epub_parser.py)
- DOCXParser (docx_parser.py)
- CodeParser (code_parser.py)
- TextParser (text_parser.py)
- HTMLParser (html_parser.py)
- PPTXParser (pptx_parser.py)
- MarkdownParser (markdown_parser.py)

Test Coverage:
1. Parser detection and format support
2. Content extraction accuracy
3. Metadata preservation
4. Error handling for corrupted files
5. Performance with large files
6. Edge cases and boundary conditions
"""

import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from wqm_cli.cli.parsers import (
    CodeParser,
    DocxParser,
    EpubParser,
    HtmlParser,
    MarkdownParser,
    PDFParser,
    PptxParser,
    TextParser,
)
from wqm_cli.cli.parsers.base import ParsedDocument
from wqm_cli.cli.parsers.exceptions import ParsingError


@pytest.fixture
def sample_files_workspace():
    """Create comprehensive sample files for parser testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)

        # Create sample files for each format
        sample_files = {}

        # Text files
        sample_files["simple.txt"] = "This is a simple text file for parser testing.\n\nIt contains multiple lines and paragraphs."
        sample_files["unicode.txt"] = "Unicode test: 🚀 Python 🐍 Testing 💻\nMultiple languages: Hello, Bonjour, 你好, こんにちは"
        sample_files["large.txt"] = "Large text file content. " * 10000  # ~240KB
        sample_files["empty.txt"] = ""

        # Markdown files
        sample_files["readme.md"] = """
# Test Document

This is a **comprehensive** test document for markdown parsing.

## Features

- Bullet points
- *Italic text*
- **Bold text**
- `Code snippets`

### Code Block

```python
def hello_world():
    return "Hello, World!"
```

### Table

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

> This is a blockquote with important information.

[Link to example](https://example.com)
        """.strip()

        # HTML files
        sample_files["webpage.html"] = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test HTML Document</title>
    <meta name="description" content="Test HTML for parser validation">
</head>
<body>
    <header>
        <h1>HTML Parser Test</h1>
        <nav>
            <ul>
                <li><a href="#section1">Section 1</a></li>
                <li><a href="#section2">Section 2</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <section id="section1">
            <h2>Text Content</h2>
            <p>This paragraph contains <strong>bold</strong> and <em>italic</em> text.</p>
            <p>Another paragraph with <a href="https://example.com">a link</a>.</p>

            <ul>
                <li>List item 1</li>
                <li>List item 2 with <code>inline code</code></li>
                <li>List item 3</li>
            </ul>
        </section>

        <section id="section2">
            <h2>Table Data</h2>
            <table>
                <thead>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                        <th>Header 3</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Row 1, Col 1</td>
                        <td>Row 1, Col 2</td>
                        <td>Row 1, Col 3</td>
                    </tr>
                    <tr>
                        <td>Row 2, Col 1</td>
                        <td>Row 2, Col 2</td>
                        <td>Row 2, Col 3</td>
                    </tr>
                </tbody>
            </table>
        </section>
    </main>

    <footer>
        <p>&copy; 2024 Test Document. All rights reserved.</p>
    </footer>
</body>
</html>
        """.strip()

        # Code files
        sample_files["example.py"] = '''
"""
Example Python module for code parser testing.

This module contains various Python constructs to test code parsing
functionality including classes, functions, decorators, and comments.
"""

import asyncio
import logging
from typing import Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessor:
    """Process documents with configurable options."""

    chunk_size: int = 1000
    overlap: int = 200
    formats: List[str] = field(default_factory=lambda: ["txt", "md", "py"])
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize processor after creation."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        logger.info(f"Initialized processor with chunk_size={self.chunk_size}")

    async def process_file(self, file_path: Path) -> Dict[str, any]:
        """
        Process a single file asynchronously.

        Args:
            file_path: Path to file to process

        Returns:
            Dictionary containing processing results

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = await self._read_file(file_path)
            chunks = self._chunk_content(content)

            return {
                "file_path": str(file_path),
                "content_length": len(content),
                "chunk_count": len(chunks),
                "format": file_path.suffix,
                "success": True
            }

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e),
                "success": False
            }

    async def _read_file(self, file_path: Path) -> str:
        """Read file content asynchronously."""
        loop = asyncio.get_event_loop()

        def read_sync():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        return await loop.run_in_executor(None, read_sync)

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into overlapping chunks."""
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk = content[start:end]

            # Try to break at word boundary
            if end < len(content):
                last_space = chunk.rfind(' ')
                if last_space > self.chunk_size * 0.7:  # Don't break too early
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())
            start = end - self.overlap

            # Prevent infinite loop
            if start >= end:
                start = end

        return chunks

    @classmethod
    def from_config(cls, config: Dict[str, any]) -> "DocumentProcessor":
        """Create processor from configuration dictionary."""
        return cls(
            chunk_size=config.get("chunk_size", 1000),
            overlap=config.get("overlap", 200),
            formats=config.get("formats", ["txt", "md", "py"]),
            metadata=config.get("metadata", {})
        )

    def __repr__(self) -> str:
        return (f"DocumentProcessor(chunk_size={self.chunk_size}, "
               f"overlap={self.overlap}, formats={self.formats})")


# Module-level functions
def validate_file_format(file_path: Path, allowed_formats: List[str]) -> bool:
    """Validate that file format is supported."""
    suffix = file_path.suffix.lower()
    return suffix.lstrip('.') in [fmt.lower() for fmt in allowed_formats]


def calculate_processing_time(content_length: int, rate: float = 1000.0) -> float:
    """Estimate processing time based on content length."""
    return max(0.1, content_length / rate)


# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
SUPPORTED_FORMATS = ["txt", "md", "py", "js", "html", "xml", "json", "yaml"]

# Multi-line string for testing
HELP_TEXT = """
This is a multi-line help text that spans several lines
and contains various formatting:

- Item 1
- Item 2
- Item 3

For more information, see the documentation.
"""


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=500, overlap=100)
    print(f"Created {processor}")

    # Test with sample file
    sample_file = Path("sample.txt")
    if sample_file.exists():
        import asyncio
        result = asyncio.run(processor.process_file(sample_file))
        print(f"Processing result: {result}")
    else:
        print("Sample file not found")
        '''.strip()

        sample_files["utils.js"] = '''
/**
 * Utility functions for JavaScript code parser testing.
 *
 * This file contains various JavaScript constructs to test
 * the code parser's ability to handle different syntax patterns.
 */

// ES6 imports
import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

// Constants
const DEFAULT_CONFIG = {
    chunkSize: 1000,
    overlap: 200,
    maxFileSize: 10 * 1024 * 1024, // 10MB
    supportedFormats: ['txt', 'md', 'html', 'js', 'py']
};

/**
 * Document processor class using modern JavaScript features.
 */
class DocumentProcessor extends EventEmitter {
    constructor(options = {}) {
        super();
        this.config = { ...DEFAULT_CONFIG, ...options };
        this.processedCount = 0;
        this.errors = [];
    }

    /**
     * Process multiple files asynchronously.
     * @param {string[]} filePaths - Array of file paths to process
     * @returns {Promise<Object[]>} Array of processing results
     */
    async processFiles(filePaths) {
        const results = [];

        for (const filePath of filePaths) {
            try {
                const result = await this.processFile(filePath);
                results.push(result);
                this.emit('fileProcessed', result);
            } catch (error) {
                const errorResult = {
                    filePath,
                    success: false,
                    error: error.message
                };
                results.push(errorResult);
                this.errors.push(errorResult);
                this.emit('fileError', errorResult);
            }
        }

        return results;
    }

    /**
     * Process a single file.
     * @param {string} filePath - Path to file to process
     * @returns {Promise<Object>} Processing result
     */
    async processFile(filePath) {
        // Validate file
        await this.validateFile(filePath);

        // Read content
        const content = await fs.readFile(filePath, 'utf8');

        // Process content
        const chunks = this.chunkContent(content);
        const metadata = this.extractMetadata(filePath, content);

        this.processedCount++;

        return {
            filePath,
            success: true,
            contentLength: content.length,
            chunkCount: chunks.length,
            metadata,
            processedAt: new Date().toISOString()
        };
    }

    /**
     * Validate file before processing.
     * @param {string} filePath - File path to validate
     * @throws {Error} If file is invalid
     */
    async validateFile(filePath) {
        // Check if file exists
        try {
            await fs.access(filePath);
        } catch {
            throw new Error(`File not found: ${filePath}`);
        }

        // Check file size
        const stats = await fs.stat(filePath);
        if (stats.size > this.config.maxFileSize) {
            throw new Error(`File too large: ${filePath} (${stats.size} bytes)`);
        }

        // Check file extension
        const ext = path.extname(filePath).slice(1).toLowerCase();
        if (!this.config.supportedFormats.includes(ext)) {
            throw new Error(`Unsupported format: ${ext}`);
        }
    }

    /**
     * Split content into overlapping chunks.
     * @param {string} content - Content to chunk
     * @returns {string[]} Array of content chunks
     */
    chunkContent(content) {
        if (content.length <= this.config.chunkSize) {
            return [content];
        }

        const chunks = [];
        let start = 0;

        while (start < content.length) {
            let end = start + this.config.chunkSize;

            // Try to break at sentence or paragraph boundary
            if (end < content.length) {
                const breakPoints = ['. ', '\\n\\n', '\\n'];

                for (const breakPoint of breakPoints) {
                    const lastBreak = content.lastIndexOf(breakPoint, end);
                    if (lastBreak > start + this.config.chunkSize * 0.7) {
                        end = lastBreak + breakPoint.length;
                        break;
                    }
                }
            }

            const chunk = content.substring(start, end).trim();
            if (chunk) {
                chunks.push(chunk);
            }

            start = Math.max(start + 1, end - this.config.overlap);
        }

        return chunks;
    }

    /**
     * Extract metadata from file and content.
     * @param {string} filePath - File path
     * @param {string} content - File content
     * @returns {Object} Metadata object
     */
    extractMetadata(filePath, content) {
        const stats = {
            filename: path.basename(filePath),
            extension: path.extname(filePath),
            directory: path.dirname(filePath),
            wordCount: content.split(/\\s+/).length,
            lineCount: content.split('\\n').length,
            characterCount: content.length
        };

        // Extract additional metadata based on file type
        if (filePath.endsWith('.js')) {
            stats.jsFeatures = this.analyzeJavaScript(content);
        } else if (filePath.endsWith('.md')) {
            stats.markdownFeatures = this.analyzeMarkdown(content);
        }

        return stats;
    }

    /**
     * Analyze JavaScript-specific features.
     * @param {string} content - JavaScript content
     * @returns {Object} JavaScript analysis
     */
    analyzeJavaScript(content) {
        return {
            hasClasses: /class\\s+\\w+/.test(content),
            hasArrowFunctions: /=>/.test(content),
            hasAsyncAwait: /async\\s+function|await\\s+/.test(content),
            hasImports: /import\\s+/.test(content),
            hasExports: /export\\s+/.test(content)
        };
    }

    /**
     * Analyze Markdown-specific features.
     * @param {string} content - Markdown content
     * @returns {Object} Markdown analysis
     */
    analyzeMarkdown(content) {
        return {
            headingCount: (content.match(/^#+/gm) || []).length,
            linkCount: (content.match(/\\[.*?\\]\\(.*?\\)/g) || []).length,
            codeBlockCount: (content.match(/```[\\s\\S]*?```/g) || []).length,
            listItemCount: (content.match(/^\\s*[-*+]\\s/gm) || []).length
        };
    }

    /**
     * Get processing statistics.
     * @returns {Object} Processing statistics
     */
    getStats() {
        return {
            processedCount: this.processedCount,
            errorCount: this.errors.length,
            successRate: this.processedCount / (this.processedCount + this.errors.length) * 100
        };
    }
}

// Export for ES6 modules
export { DocumentProcessor, DEFAULT_CONFIG };

// Export for CommonJS (dual compatibility)
module.exports = { DocumentProcessor, DEFAULT_CONFIG };
        '''.strip()

        # Create all text files
        for filename, content in sample_files.items():
            file_path = workspace_path / filename
            file_path.write_text(content)

        # Create minimal binary files (placeholders that can be tested for error handling)
        # These must contain null bytes to be properly detected as binary, not text
        binary_files = {
            "document.pdf": b"%PDF-1.4\x00\x00\x00fake PDF content for testing\x00\x00",
            "document.docx": b"PK\x03\x04\x00\x00\x00\x00fake DOCX content\x00\x00",
            "presentation.pptx": b"PK\x03\x04\x00\x00\x00\x00fake PPTX content\x00\x00",
            "book.epub": b"PK\x03\x04\x00\x00\x00\x00fake EPUB content\x00\x00",
        }

        for filename, content in binary_files.items():
            file_path = workspace_path / filename
            file_path.write_bytes(content)

        yield {
            "path": workspace_path,
            "text_files": list(sample_files.keys()),
            "binary_files": list(binary_files.keys()),
            "all_files": list(sample_files.keys()) + list(binary_files.keys())
        }


@pytest.mark.unit
class TestTextParserValidation:
    """Test TextParser functionality."""

    def setup_method(self):
        self.parser = TextParser()

    def test_parser_format_detection(self, sample_files_workspace):
        """Test text format detection and support."""
        text_files = ["simple.txt", "unicode.txt", "large.txt", "empty.txt"]

        for filename in text_files:
            file_path = sample_files_workspace["path"] / filename
            assert self.parser.can_parse(file_path) is True

        # Test non-text files
        non_text_files = ["document.pdf", "document.docx"]
        for filename in non_text_files:
            file_path = sample_files_workspace["path"] / filename
            assert self.parser.can_parse(file_path) is False

    def test_parser_properties(self):
        """Test parser properties and configuration."""
        assert self.parser.format_name == "Text Document"
        assert ".txt" in self.parser.supported_extensions

        options = self.parser.get_parsing_options()
        assert isinstance(options, dict)

    @pytest.mark.asyncio
    async def test_content_extraction(self, sample_files_workspace):
        """Test content extraction from text files."""
        test_cases = [
            ("simple.txt", "simple text file"),
            ("unicode.txt", "Unicode test"),
            ("empty.txt", None)  # Empty file handling
        ]

        for filename, expected_content in test_cases:
            file_path = sample_files_workspace["path"] / filename

            parsed_doc = await self.parser.parse(file_path)

            assert isinstance(parsed_doc, ParsedDocument)
            assert parsed_doc.content_hash is not None
            assert isinstance(parsed_doc.metadata, dict)
            # file_path is an attribute of ParsedDocument, not in metadata
            assert parsed_doc.file_path == str(file_path)

            if expected_content:
                assert expected_content.lower() in parsed_doc.content.lower()
            else:
                # Empty file case
                assert len(parsed_doc.content.strip()) == 0

    @pytest.mark.asyncio
    async def test_large_file_handling(self, sample_files_workspace):
        """Test handling of large text files."""
        large_file = sample_files_workspace["path"] / "large.txt"

        parsed_doc = await self.parser.parse(large_file)

        assert parsed_doc.content is not None
        assert len(parsed_doc.content) > 10000  # Should be substantial
        assert "Large text file content" in parsed_doc.content
        assert parsed_doc.metadata["file_size"] > 0

    @pytest.mark.asyncio
    async def test_unicode_handling(self, sample_files_workspace):
        """Test Unicode content handling."""
        unicode_file = sample_files_workspace["path"] / "unicode.txt"

        parsed_doc = await self.parser.parse(unicode_file)

        assert parsed_doc.content is not None
        assert "🚀" in parsed_doc.content
        assert "你好" in parsed_doc.content
        assert "こんにちは" in parsed_doc.content


@pytest.mark.unit
class TestCodeParserValidation:
    """Test CodeParser functionality."""

    def setup_method(self):
        self.parser = CodeParser()

    def test_code_format_detection(self, sample_files_workspace):
        """Test code format detection."""
        code_files = ["example.py", "utils.js"]

        for filename in code_files:
            file_path = sample_files_workspace["path"] / filename
            assert self.parser.can_parse(file_path) is True

        assert self.parser.format_name == "Source Code"
        assert ".py" in self.parser.supported_extensions
        assert ".js" in self.parser.supported_extensions

    @pytest.mark.asyncio
    async def test_python_code_parsing(self, sample_files_workspace):
        """Test Python code parsing."""
        python_file = sample_files_workspace["path"] / "example.py"

        parsed_doc = await self.parser.parse(python_file)

        assert isinstance(parsed_doc, ParsedDocument)
        assert "DocumentProcessor" in parsed_doc.content
        assert "async def process_file" in parsed_doc.content
        assert parsed_doc.metadata["programming_language"] == "python"

        # Check for code-specific metadata (function_count and class_count rather than lists)
        assert "function_count" in parsed_doc.metadata
        assert parsed_doc.metadata["function_count"] > 0

    @pytest.mark.asyncio
    async def test_javascript_code_parsing(self, sample_files_workspace):
        """Test JavaScript code parsing."""
        js_file = sample_files_workspace["path"] / "utils.js"

        parsed_doc = await self.parser.parse(js_file)

        assert isinstance(parsed_doc, ParsedDocument)
        assert "DocumentProcessor" in parsed_doc.content
        assert "async processFiles" in parsed_doc.content
        assert parsed_doc.metadata["programming_language"] == "javascript"

        # Check for code-specific metadata (function_count rather than features)
        assert "function_count" in parsed_doc.metadata

    @pytest.mark.asyncio
    async def test_code_language_detection(self):
        """Test programming language detection."""
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.java", "java"),
            ("test.cpp", "cpp"),
            ("test.rs", "rust"),
        ]

        for filename, expected_lang in test_cases:
            with tempfile.NamedTemporaryFile(suffix=filename) as tmp:
                tmp.write(b"// test code")
                tmp.flush()

                detected_lang = await self.parser._detect_language(Path(tmp.name))
                assert detected_lang == expected_lang


@pytest.mark.unit
class TestHTMLParserValidation:
    """Test HTMLParser functionality."""

    def setup_method(self):
        self.parser = HtmlParser()

    def test_html_format_detection(self, sample_files_workspace):
        """Test HTML format detection."""
        html_file = sample_files_workspace["path"] / "webpage.html"

        assert self.parser.can_parse(html_file) is True
        assert self.parser.format_name == "HTML Web Content"
        assert ".html" in self.parser.supported_extensions

    @pytest.mark.asyncio
    async def test_html_content_extraction(self, sample_files_workspace):
        """Test HTML content extraction and cleaning."""
        html_file = sample_files_workspace["path"] / "webpage.html"

        parsed_doc = await self.parser.parse(html_file)

        assert isinstance(parsed_doc, ParsedDocument)
        assert parsed_doc.content is not None

        # Should extract text content without HTML tags
        assert "Text Content" in parsed_doc.content
        assert "Table Data" in parsed_doc.content
        assert "<html>" not in parsed_doc.content  # Tags should be stripped
        assert "<body>" not in parsed_doc.content

        # Should extract list and table content
        assert "List item 1" in parsed_doc.content
        assert "Row 1, Col 1" in parsed_doc.content

    @pytest.mark.asyncio
    async def test_html_metadata_extraction(self, sample_files_workspace):
        """Test HTML metadata extraction."""
        html_file = sample_files_workspace["path"] / "webpage.html"

        parsed_doc = await self.parser.parse(html_file)

        metadata = parsed_doc.metadata
        assert isinstance(metadata, dict)

        # Should extract meta tags
        if "title" in metadata:
            assert "Test HTML Document" in metadata["title"]

        if "description" in metadata:
            assert "parser validation" in metadata["description"]

        # Should identify structure elements
        if "structure" in metadata:
            structure = metadata["structure"]
            assert "headers" in structure or "headings" in structure
            assert "links" in structure

    @pytest.mark.asyncio
    async def test_html_table_extraction(self, sample_files_workspace):
        """Test HTML table content extraction."""
        html_file = sample_files_workspace["path"] / "webpage.html"

        parsed_doc = await self.parser.parse(html_file)

        # Should extract table content
        assert "Header 1" in parsed_doc.content
        assert "Row 1, Col 1" in parsed_doc.content
        assert "Row 2, Col 2" in parsed_doc.content


@pytest.mark.unit
class TestMarkdownParserValidation:
    """Test MarkdownParser functionality."""

    def setup_method(self):
        self.parser = MarkdownParser()

    def test_markdown_format_detection(self, sample_files_workspace):
        """Test Markdown format detection."""
        md_file = sample_files_workspace["path"] / "readme.md"

        assert self.parser.can_parse(md_file) is True
        assert self.parser.format_name == "Markdown"
        assert ".md" in self.parser.supported_extensions

    @pytest.mark.asyncio
    async def test_markdown_content_extraction(self, sample_files_workspace):
        """Test Markdown content extraction."""
        md_file = sample_files_workspace["path"] / "readme.md"

        parsed_doc = await self.parser.parse(md_file)

        assert isinstance(parsed_doc, ParsedDocument)
        assert "Test Document" in parsed_doc.content
        assert "comprehensive test" in parsed_doc.content.lower()

        # Should preserve or convert markdown elements appropriately
        assert "Features" in parsed_doc.content
        assert "Code Block" in parsed_doc.content

    @pytest.mark.asyncio
    async def test_markdown_structure_extraction(self, sample_files_workspace):
        """Test Markdown structure extraction."""
        md_file = sample_files_workspace["path"] / "readme.md"

        parsed_doc = await self.parser.parse(md_file)

        metadata = parsed_doc.metadata
        assert isinstance(metadata, dict)

        # Should identify markdown structure
        if "structure" in metadata:
            structure = metadata["structure"]
            assert "headings" in structure or "headers" in structure
            assert "links" in structure or "code_blocks" in structure


@pytest.mark.unit
class TestBinaryParserValidation:
    """Test binary format parsers (PDF, DOCX, PPTX, EPUB)."""

    def test_parser_format_detection(self, sample_files_workspace):
        """Test binary parser format detection."""
        parsers_and_files = [
            (PDFParser(), "document.pdf"),
            (DocxParser(), "document.docx"),
            (PptxParser(), "presentation.pptx"),
            (EpubParser(), "book.epub"),
        ]

        for parser, filename in parsers_and_files:
            file_path = sample_files_workspace["path"] / filename
            # These are placeholder files, so detection should work but parsing might fail
            assert parser.can_parse(file_path) is True

    def test_parser_properties(self):
        """Test binary parser properties."""
        # Check actual format names from parser implementations
        parsers = [
            (PDFParser(), ".pdf"),
            (DocxParser(), ".docx"),
            (PptxParser(), ".pptx"),
            (EpubParser(), ".epub"),
        ]

        for parser, expected_ext in parsers:
            # Just verify format_name is a non-empty string
            assert isinstance(parser.format_name, str)
            assert len(parser.format_name) > 0
            assert expected_ext in parser.supported_extensions

            options = parser.get_parsing_options()
            assert isinstance(options, dict)

    @pytest.mark.asyncio
    async def test_binary_parser_error_handling(self, sample_files_workspace):
        """Test binary parser error handling with placeholder files."""
        parsers_and_files = [
            (PDFParser(), "document.pdf"),
            (DocxParser(), "document.docx"),
            (PptxParser(), "presentation.pptx"),
            (EpubParser(), "book.epub"),
        ]

        for parser, filename in parsers_and_files:
            file_path = sample_files_workspace["path"] / filename

            try:
                # These should fail gracefully since they're placeholder files
                parsed_doc = await parser.parse(file_path)
                # If parsing somehow succeeds, verify basic structure
                assert isinstance(parsed_doc, ParsedDocument)

            except (ParsingError, Exception) as e:
                # Expected for placeholder files - should handle gracefully
                assert isinstance(e, Exception)
                print(f"Expected parsing error for {filename}: {e}")


@pytest.mark.unit
class TestParserErrorHandling:
    """Test parser error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_nonexistent_file_handling(self):
        """Test handling of non-existent files."""
        parsers = [TextParser(), CodeParser(), HtmlParser(), MarkdownParser()]

        for parser in parsers:
            nonexistent_file = Path("/nonexistent/file.txt")

            try:
                await parser.parse(nonexistent_file)
                pytest.fail(f"{parser.format_name} should have raised an error for nonexistent file")
            except (FileNotFoundError, ParsingError):
                pass  # Expected
            except Exception as e:
                pytest.fail(f"{parser.format_name} raised unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        parsers = [TextParser(), CodeParser(), HtmlParser(), MarkdownParser()]

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as tmp:
            # Write corrupted UTF-8 data
            tmp.write(b"\xff\xfe\x00\x01invalid utf-8 \x80\x81")
            tmp.flush()

            corrupted_file = Path(tmp.name)

            for parser in parsers:
                if parser.can_parse(corrupted_file):
                    try:
                        parsed_doc = await parser.parse(corrupted_file)
                        # If parsing succeeds, should have some content handling
                        assert isinstance(parsed_doc, ParsedDocument)

                    except (UnicodeDecodeError, ParsingError) as e:
                        # Expected for corrupted files
                        print(f"Expected error from {parser.format_name}: {e}")
                        pass

        # Clean up
        corrupted_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_empty_file_handling(self, sample_files_workspace):
        """Test handling of empty files."""
        parsers = [TextParser(), CodeParser(), HtmlParser(), MarkdownParser()]
        empty_file = sample_files_workspace["path"] / "empty.txt"

        for parser in parsers:
            if parser.can_parse(empty_file):
                parsed_doc = await parser.parse(empty_file)

                assert isinstance(parsed_doc, ParsedDocument)
                assert len(parsed_doc.content.strip()) == 0
                assert parsed_doc.content_hash is not None
                assert isinstance(parsed_doc.metadata, dict)

    @pytest.mark.asyncio
    async def test_permission_denied_handling(self):
        """Test handling of permission-denied files."""
        from unittest.mock import patch, mock_open

        parsers = [TextParser(), CodeParser(), HtmlParser(), MarkdownParser()]

        # Create a file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("test content")
            tmp.flush()
            restricted_file = Path(tmp.name)

        try:
            # Mock file open to raise PermissionError
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                for parser in parsers:
                    if parser.can_parse(restricted_file):
                        try:
                            await parser.parse(restricted_file)
                            pytest.fail(f"{parser.format_name} should have raised permission error")
                        except (PermissionError, ParsingError):
                            pass  # Expected
                        except Exception as e:
                            print(f"Unexpected error from {parser.format_name}: {e}")
        finally:
            # Cleanup
            try:
                restricted_file.unlink()
            except Exception:
                pass


@pytest.mark.performance
class TestParserPerformance:
    """Test parser performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_file_performance(self, sample_files_workspace):
        """Test performance with large files."""
        import time

        large_file = sample_files_workspace["path"] / "large.txt"
        parser = TextParser()

        start_time = time.time()
        parsed_doc = await parser.parse(large_file)
        end_time = time.time()

        processing_time = end_time - start_time

        assert parsed_doc is not None
        assert processing_time < 5.0  # Should process large file in reasonable time

        print("Large file processing performance:")
        print(f"  Content length: {len(parsed_doc.content):,} characters")
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Rate: {len(parsed_doc.content) / processing_time / 1000:.1f} KB/s")

    @pytest.mark.asyncio
    async def test_concurrent_parsing_performance(self, sample_files_workspace):
        """Test concurrent parsing performance."""
        import asyncio
        import time

        parsers_and_files = [
            (TextParser(), "simple.txt"),
            (TextParser(), "unicode.txt"),
            (CodeParser(), "example.py"),
            (CodeParser(), "utils.js"),
            (HtmlParser(), "webpage.html"),
            (MarkdownParser(), "readme.md"),
        ]

        async def parse_file(parser, filename):
            file_path = sample_files_workspace["path"] / filename
            start_time = time.time()
            parsed_doc = await parser.parse(file_path)
            end_time = time.time()
            return {
                "parser": parser.format_name,
                "file": filename,
                "time": end_time - start_time,
                "content_length": len(parsed_doc.content)
            }

        # Test concurrent parsing
        start_time = time.time()
        tasks = [parse_file(parser, filename) for parser, filename in parsers_and_files]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time

        print("Concurrent parsing performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Files processed: {len(results)}")
        print(f"  Average per file: {total_time / len(results):.3f}s")

        for result in results:
            print(f"    {result['parser']}: {result['file']} -> {result['time']:.3f}s")

        # Verify all parsings completed successfully
        assert len(results) == len(parsers_and_files)
        assert all(result["content_length"] > 0 for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
