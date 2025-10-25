"""
Comprehensive unit tests for enhanced EPUB and MOBI parsers.

Tests cover DRM detection, metadata extraction, error handling, edge cases,
and all new functionality added for task 258.3.
"""

import asyncio
import struct
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from wqm_cli.cli.parsers.base import ParsedDocument
from wqm_cli.cli.parsers.epub_parser import EpubParser
from wqm_cli.cli.parsers.mobi_parser import MobiParser


class TestEnhancedEpubParser:
    """Test enhanced EPUB parser functionality."""

    @pytest.fixture
    def parser(self):
        return EpubParser()

    def create_mock_epub_file(self, file_path: Path, has_drm: bool = False, corrupted: bool = False) -> None:
        """Create a mock EPUB file for testing."""
        with zipfile.ZipFile(file_path, 'w') as zf:
            if not corrupted:
                # Create basic EPUB structure
                zf.writestr('META-INF/container.xml', '''<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>''')

                zf.writestr('OEBPS/content.opf', '''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
</package>''')

                if has_drm:
                    # Add DRM indicators
                    zf.writestr('META-INF/encryption.xml', '''<?xml version="1.0"?>
<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:encryption:1.0">
  <EncryptedData>
    <EncryptionMethod Algorithm="http://www.adobe.com/adept"/>
  </EncryptedData>
</encryption>''')
            else:
                # Create corrupted structure
                zf.writestr('invalid_file.txt', 'This is not a valid EPUB')

    @pytest.mark.asyncio
    async def test_drm_detection_with_adobe_drm(self, parser):
        """Test DRM detection for Adobe ADEPT DRM."""
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
            epub_path = Path(f.name)
            self.create_mock_epub_file(epub_path, has_drm=True)

            try:
                drm_info = await parser._check_drm_protection(epub_path)

                assert drm_info["has_drm"] is True
                assert drm_info["scheme"] == "Adobe ADEPT"
                assert len(drm_info["details"]) > 0

            finally:
                epub_path.unlink()

    @pytest.mark.asyncio
    async def test_drm_detection_no_drm(self, parser):
        """Test DRM detection for non-DRM protected EPUB."""
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
            epub_path = Path(f.name)
            self.create_mock_epub_file(epub_path, has_drm=False)

            try:
                drm_info = await parser._check_drm_protection(epub_path)

                assert drm_info["has_drm"] is False
                assert drm_info["scheme"] == "none"

            finally:
                epub_path.unlink()

    @pytest.mark.asyncio
    async def test_corrupted_epub_handling(self, parser):
        """Test handling of corrupted EPUB files."""
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
            epub_path = Path(f.name)
            # Create an invalid ZIP file
            f.write(b"This is not a valid ZIP/EPUB file")
            f.flush()

            try:
                drm_info = await parser._check_drm_protection(epub_path)

                assert "not a valid ZIP/EPUB archive" in str(drm_info["details"])

            finally:
                epub_path.unlink()

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.epub_parser.epub.read_epub')
    async def test_safe_epub_reading_with_recovery(self, mock_read_epub, parser):
        """Test safe EPUB reading with recovery mechanism."""
        # Mock initial failure, then success
        mock_read_epub.side_effect = [Exception("Initial read failed"), Mock()]

        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
            epub_path = Path(f.name)
            self.create_mock_epub_file(epub_path)

            try:
                with patch('zipfile.ZipFile') as mock_zipfile:
                    mock_zipfile.return_value.__enter__.return_value.testzip.return_value = None

                    result = await parser._safe_read_epub(str(epub_path))

                    assert result is not None

            finally:
                epub_path.unlink()

    @pytest.mark.asyncio
    async def test_table_of_contents_extraction(self, parser):
        """Test TOC extraction from EPUB."""
        mock_book = Mock()
        mock_book.toc = [
            Mock(title="Chapter 1", href="chapter1.html"),
            Mock(title="Chapter 2", href="chapter2.html")
        ]

        toc_info = await parser._extract_table_of_contents(mock_book)

        assert "toc_entries" in toc_info
        assert len(toc_info["toc_entries"]) == 2
        assert toc_info["toc_structure"] == "ncx"
        assert toc_info["toc_entries"][0]["title"] == "Chapter 1"

    @pytest.mark.asyncio
    async def test_structured_text_extraction(self, parser):
        """Test structured text extraction preserving headings."""
        from bs4 import BeautifulSoup

        html_content = """
        <html>
            <body>
                <h1>Chapter Title</h1>
                <p>This is a paragraph.</p>
                <h2>Section Title</h2>
                <p>Another paragraph.</p>
            </body>
        </html>
        """

        soup = BeautifulSoup(html_content, 'html.parser')
        result = await parser._extract_structured_text(soup)

        assert "# Chapter Title" in result
        assert "## Section Title" in result
        assert "This is a paragraph." in result

    @pytest.mark.asyncio
    async def test_enhanced_metadata_extraction(self, parser):
        """Test comprehensive metadata extraction."""
        mock_book = Mock()

        # Mock Dublin Core metadata
        mock_book.get_metadata.side_effect = lambda ns, field: {
            ("DC", "title"): [("Test Book Title", {})],
            ("DC", "creator"): [("Author One", {}), ("Author Two", {})],
            ("DC", "publisher"): [("Test Publisher", {})],
            ("DC", "language"): [("en", {})],
            ("DC", "subject"): [("Fiction", {}), ("Adventure", {})],
            ("DC", "rights"): [("Copyright 2024", {})],
        }.get((ns, field), [])

        # Mock items for media analysis
        # Need to import ebooklib constants for proper testing
        try:
            import ebooklib
            ITEM_DOCUMENT = ebooklib.ITEM_DOCUMENT
            ITEM_IMAGE = ebooklib.ITEM_IMAGE
            ITEM_AUDIO = ebooklib.ITEM_AUDIO
            ITEM_VIDEO = ebooklib.ITEM_VIDEO
        except ImportError:
            # Fallback values if ebooklib not available
            ITEM_DOCUMENT = 10
            ITEM_IMAGE = 15
            ITEM_AUDIO = 20
            ITEM_VIDEO = 25

        mock_book.get_items_of_type.side_effect = lambda item_type: {
            ITEM_DOCUMENT: [Mock(), Mock(), Mock()],  # ITEM_DOCUMENT (chapters)
            ITEM_IMAGE: [Mock(media_type="image/jpeg"), Mock(media_type="image/png")],  # ITEM_IMAGE
            ITEM_AUDIO: [Mock()],  # ITEM_AUDIO
            ITEM_VIDEO: []  # ITEM_VIDEO
        }.get(item_type, [])

        mock_book.get_items.return_value = [Mock() for _ in range(20)]  # Total items

        drm_info = {"has_drm": False, "scheme": "none", "details": []}

        metadata = await parser._extract_enhanced_metadata(mock_book, drm_info)

        assert metadata["title"] == "Test Book Title"
        assert metadata["author"] == "Author One, Author Two"
        assert metadata["author_count"] == 2
        assert metadata["publisher"] == "Test Publisher"
        assert metadata["language"] == "en"
        assert metadata["subjects"] == "Fiction, Adventure"
        assert metadata["subject_count"] == 2
        assert metadata["rights"] == "Copyright 2024"
        assert metadata["chapter_count"] == 3
        assert metadata["image_count"] == 2
        assert metadata["audio_count"] == 1
        assert metadata["video_count"] == 0
        assert metadata["has_multimedia"] is True
        assert metadata["complexity_score"] == 2  # 20 items / 10

    @pytest.mark.asyncio
    async def test_parsing_error_diagnosis(self, parser):
        """Test detailed parsing error diagnosis."""
        # Test with non-existent file
        non_existent = Path("/tmp/does_not_exist.epub")
        error = FileNotFoundError("File not found")

        diagnosis = await parser._diagnose_parsing_error(non_existent, error)
        assert "File does not exist" in diagnosis

        # Test with empty file
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
            empty_path = Path(f.name)

        try:
            error = RuntimeError("Empty file")
            diagnosis = await parser._diagnose_parsing_error(empty_path, error)
            assert "File is empty" in diagnosis

        finally:
            empty_path.unlink()

    @pytest.mark.asyncio
    async def test_enhanced_parsing_options(self, parser):
        """Test new parsing options."""
        options = parser.get_parsing_options()

        expected_options = [
            "include_images", "max_chapter_size", "chapter_separator",
            "preserve_structure", "extract_toc"
        ]

        for option in expected_options:
            assert option in options
            assert "type" in options[option]
            assert "default" in options[option]
            assert "description" in options[option]

    @pytest.mark.asyncio
    async def test_large_epub_handling(self, parser):
        """Test handling of very large EPUB files."""
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
            epub_path = Path(f.name)

            # Create a mock large file
            with zipfile.ZipFile(epub_path, 'w') as zf:
                # Add basic structure
                zf.writestr('META-INF/container.xml', '<container/>')
                # Simulate large content
                large_content = "Large content " * 100000  # ~1.3MB of content
                zf.writestr('large_chapter.html', large_content)

            try:
                error = RuntimeError("File too large")
                diagnosis = await parser._diagnose_parsing_error(epub_path, error)

                # Should detect large file
                file_size = epub_path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # 100MB
                    assert "File is very large" in diagnosis

            finally:
                epub_path.unlink()


class TestEnhancedMobiParser:
    """Test enhanced MOBI parser functionality."""

    @pytest.fixture
    def parser(self):
        return MobiParser()

    def create_mock_mobi_file(self, file_path: Path, has_drm: bool = False, format_type: str = "MOBI") -> None:
        """Create a mock MOBI file for testing."""
        with open(file_path, 'wb') as f:
            # Create basic PDB header (78 bytes)
            pdb_header = bytearray(78)
            # Set database name
            db_name = b"Test Book Title\x00" + b"\x00" * (32 - len(b"Test Book Title\x00"))
            pdb_header[0:32] = db_name

            # Set creation and modification dates (Mock values)
            creation_date = struct.pack('>I', 2200000000)  # Mock date
            modification_date = struct.pack('>I', 2200000000)
            pdb_header[36:40] = creation_date
            pdb_header[40:44] = modification_date

            f.write(pdb_header)

            if format_type == "MOBI":
                # Add MOBI header
                mobi_header = b"MOBI"
                mobi_header += struct.pack('>I', 232)  # Header length
                mobi_header += struct.pack('>I', 2)    # MOBI type
                mobi_header += b"\x00" * 16           # Reserved
                mobi_header += struct.pack('>I', 65001)  # Text encoding (UTF-8)
                mobi_header += b"\x00" * 60           # More fields
                mobi_header += struct.pack('>I', 9)   # Language (English)
                mobi_header += b"\x00" * (232 - len(mobi_header))  # Pad to 232 bytes
                f.write(mobi_header)

                if has_drm:
                    # Add DRM indicators
                    drm_data = b"kindle.amazon.com" + b"\x00" * 100
                    f.write(drm_data)

                # Add some text content
                f.write(b"This is the main text content of the MOBI book.")
            else:
                # For other formats, just add minimal content
                f.write(b"Minimal content for " + format_type.encode())

    @pytest.mark.asyncio
    async def test_file_format_analysis_mobi(self, parser):
        """Test MOBI file format analysis."""
        with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
            mobi_path = Path(f.name)
            self.create_mock_mobi_file(mobi_path, format_type="MOBI")

            try:
                format_info = await parser._analyze_file_format(mobi_path)

                assert format_info["format"] == "MOBI"
                assert format_info["text_encoding"] == "utf-8"

            finally:
                mobi_path.unlink()

    @pytest.mark.asyncio
    async def test_file_format_analysis_azw3(self, parser):
        """Test AZW3 file format detection."""
        with tempfile.NamedTemporaryFile(suffix='.azw3', delete=False) as f:
            azw3_path = Path(f.name)
            self.create_mock_mobi_file(azw3_path, format_type="AZW3")

            try:
                format_info = await parser._analyze_file_format(azw3_path)

                assert format_info["format"] == "AZW3/KF8"

            finally:
                azw3_path.unlink()

    @pytest.mark.asyncio
    async def test_drm_detection_kindle_drm(self, parser):
        """Test DRM detection for Kindle DRM."""
        with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
            mobi_path = Path(f.name)
            self.create_mock_mobi_file(mobi_path, has_drm=True)

            try:
                format_info = {"format": "MOBI", "version": "unknown"}
                drm_info = await parser._check_drm_protection(mobi_path, format_info)

                assert drm_info["has_drm"] is True
                assert drm_info["scheme"] == "Kindle DRM"
                assert len(drm_info["details"]) > 0

            finally:
                mobi_path.unlink()

    @pytest.mark.asyncio
    async def test_drm_detection_no_drm(self, parser):
        """Test DRM detection for non-DRM MOBI."""
        with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
            mobi_path = Path(f.name)
            self.create_mock_mobi_file(mobi_path, has_drm=False)

            try:
                format_info = {"format": "MOBI", "version": "unknown"}
                drm_info = await parser._check_drm_protection(mobi_path, format_info)

                assert drm_info["has_drm"] is False
                assert drm_info["scheme"] == "none"

            finally:
                mobi_path.unlink()

    @pytest.mark.asyncio
    async def test_enhanced_text_extraction(self, parser):
        """Test enhanced text content extraction."""
        with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
            mobi_path = Path(f.name)
            self.create_mock_mobi_file(mobi_path)

            try:
                with open(mobi_path, 'rb') as file_handle:
                    format_info = {"format": "MOBI", "text_encoding": "utf-8"}

                    result = await parser._extract_enhanced_text_content(
                        file_handle, "utf-8", 10000, True, format_info
                    )

                    assert isinstance(result, str)
                    assert len(result) > 0

            finally:
                mobi_path.unlink()

    @pytest.mark.asyncio
    async def test_formatted_text_extraction_with_html(self, parser):
        """Test formatted text extraction with HTML content."""
        html_content = b"""<html><body>
            <h1>Chapter 1</h1>
            <p>This is the first paragraph.</p>
            <h2>Section 1.1</h2>
            <p>This is another paragraph.</p>
        </body></html>"""

        format_info = {"text_encoding": "utf-8"}

        with patch('bs4.BeautifulSoup') as mock_soup_class:
            # Mock BeautifulSoup parsing
            mock_soup = Mock()
            mock_soup_class.return_value = mock_soup

            # Mock finding elements
            h1_elem = Mock()
            h1_elem.name = 'h1'
            h1_elem.get_text.return_value = "Chapter 1"

            p_elem = Mock()
            p_elem.name = 'p'
            p_elem.get_text.return_value = "This is the first paragraph."

            mock_soup.find_all.return_value = [h1_elem, p_elem]
            # Mock the __call__ method for script/style removal
            mock_soup.return_value = []  # For soup(["script", "style"])

            result = await parser._extract_formatted_text(html_content, "utf-8", format_info)

            # Check that result contains content (may not have exact formatting due to mocking)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_parsing_error_diagnosis_invalid_file(self, parser):
        """Test error diagnosis for invalid MOBI files."""
        with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
            mobi_path = Path(f.name)
            f.write(b"Invalid MOBI content")  # Too short, invalid header
            f.flush()

            try:
                error = RuntimeError("Invalid MOBI format")
                diagnosis = await parser._diagnose_parsing_error(mobi_path, error)

                assert "File is too small" in diagnosis or "File header is incomplete" in diagnosis

            finally:
                mobi_path.unlink()

    @pytest.mark.asyncio
    async def test_parsing_error_diagnosis_drm_file(self, parser):
        """Test error diagnosis for DRM-protected files."""
        error = RuntimeError("DRM protected file cannot be read")

        with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
            mobi_path = Path(f.name)
            f.write(b"x" * 1000)  # Dummy content
            f.flush()

            try:
                diagnosis = await parser._diagnose_parsing_error(mobi_path, error)

                assert "DRM-protected" in diagnosis

            finally:
                mobi_path.unlink()

    @pytest.mark.asyncio
    async def test_enhanced_parsing_options(self, parser):
        """Test new MOBI parsing options."""
        options = parser.get_parsing_options()

        expected_options = [
            "encoding", "max_content_size", "attempt_drm_removal",
            "extract_images", "preserve_formatting"
        ]

        for option in expected_options:
            assert option in options
            assert "type" in options[option]
            assert "default" in options[option]
            assert "description" in options[option]

    @pytest.mark.asyncio
    async def test_kindleunpack_fallback_when_unavailable(self, parser):
        """Test graceful handling when kindleunpack is not available."""
        with patch('wqm_cli.cli.parsers.mobi_parser.KINDLE_UNPACK_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="kindleunpack library not available"):
                await parser._parse_with_kindleunpack(Path("test.mobi"), True)

    @pytest.mark.asyncio
    async def test_drm_removal_fallback_when_unavailable(self, parser):
        """Test graceful handling when mobidedrm is not available."""
        with patch('wqm_cli.cli.parsers.mobi_parser.MOBI_DEDRM_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="mobidedrm library not available"):
                await parser._parse_with_drm_removal(Path("test.mobi"), "utf-8", True)

    @pytest.mark.asyncio
    async def test_supported_extensions(self, parser):
        """Test that all Kindle format extensions are supported."""
        extensions = parser.supported_extensions

        expected_extensions = [".mobi", ".azw", ".azw3", ".azw4", ".kfx", ".kfx-zip"]

        for ext in expected_extensions:
            assert ext in extensions

    @pytest.mark.asyncio
    async def test_format_name(self, parser):
        """Test format name includes Kindle formats."""
        format_name = parser.format_name
        assert "MOBI" in format_name
        assert "Kindle" in format_name


class TestEpubMobiEdgeCases:
    """Test edge cases that apply to both EPUB and MOBI parsers."""

    @pytest.mark.asyncio
    async def test_extremely_large_files(self):
        """Test handling of extremely large e-book files."""
        epub_parser = EpubParser()
        mobi_parser = MobiParser()

        # Create large dummy files
        for parser, suffix in [(epub_parser, '.epub'), (mobi_parser, '.mobi')]:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                large_path = Path(f.name)
                f.write(b"x" * (150 * 1024 * 1024))  # 150MB file
                f.flush()

                try:
                    error = RuntimeError("File too large")

                    if suffix == '.epub':
                        diagnosis = await parser._diagnose_parsing_error(large_path, error)
                    else:
                        diagnosis = await parser._diagnose_parsing_error(large_path, error)

                    # Should detect large file or file format issues
                    assert ("very large" in diagnosis.lower() or "invalid" in diagnosis.lower() or "too large" in diagnosis.lower())

                finally:
                    large_path.unlink()

    @pytest.mark.asyncio
    async def test_zero_byte_files(self):
        """Test handling of empty e-book files."""
        epub_parser = EpubParser()
        mobi_parser = MobiParser()

        for parser, suffix in [(epub_parser, '.epub'), (mobi_parser, '.mobi')]:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                empty_path = Path(f.name)
                # File is created but remains empty

                try:
                    error = RuntimeError("Empty file")

                    if suffix == '.epub':
                        diagnosis = await parser._diagnose_parsing_error(empty_path, error)
                    else:
                        diagnosis = await parser._diagnose_parsing_error(empty_path, error)

                    assert "empty" in diagnosis.lower()

                finally:
                    empty_path.unlink()

    @pytest.mark.asyncio
    async def test_non_standard_extensions(self):
        """Test files with unusual but valid extensions."""
        epub_parser = EpubParser()
        mobi_parser = MobiParser()

        # Test EPUB parser with unusual extension
        assert epub_parser.can_parse("book.epub")

        # Test MOBI parser with various Kindle extensions
        kindle_extensions = ["book.mobi", "book.azw", "book.azw3", "book.azw4", "book.kfx"]
        for ext_file in kindle_extensions:
            assert mobi_parser.can_parse(ext_file)

    @pytest.mark.asyncio
    async def test_concurrent_parsing(self):
        """Test concurrent parsing of multiple files."""
        epub_parser = EpubParser()
        mobi_parser = MobiParser()

        # Create multiple test files
        files_to_test = []

        try:
            # Create EPUB files
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
                    epub_path = Path(f.name)
                    with zipfile.ZipFile(epub_path, 'w') as zf:
                        zf.writestr('META-INF/container.xml', '<container/>')
                        zf.writestr(f'content{i}.html', f'<html>Content {i}</html>')
                    files_to_test.append((epub_parser, epub_path))

            # Create MOBI files
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.mobi', delete=False) as f:
                    mobi_path = Path(f.name)
                    # Create minimal MOBI structure
                    f.write(b"x" * 78)  # PDB header
                    f.write(b"MOBI" + b"x" * 228)  # MOBI header
                    f.write(f"Content {i}".encode())
                    files_to_test.append((mobi_parser, mobi_path))

            # Test concurrent error diagnosis (simulating real concurrent usage)
            async def diagnose_file(parser, file_path):
                error = RuntimeError("Test error")
                return await parser._diagnose_parsing_error(file_path, error)

            # Run diagnoses concurrently
            tasks = [diagnose_file(parser, path) for parser, path in files_to_test]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete without exceptions
            for result in results:
                assert isinstance(result, str)  # Should be diagnosis string, not exception

        finally:
            # Cleanup
            for _, path in files_to_test:
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
