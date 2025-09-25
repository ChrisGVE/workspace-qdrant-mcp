"""
Comprehensive unit tests for ContentExtractor with edge cases and error conditions.

This test suite covers:
- Multiple parsing strategies (BeautifulSoup, readability, custom, hybrid)
- Content quality assessment and scoring
- Metadata extraction (title, description, keywords, author, dates)
- Text cleaning and normalization
- Language detection capabilities
- Link, image, and heading extraction
- Error handling and fallback mechanisms
- Content validation and filtering
- Performance with large documents
- Malformed HTML handling
- Edge cases with empty or minimal content
"""

import pytest
import time
from datetime import datetime
from unittest.mock import patch, MagicMock

from bs4 import BeautifulSoup

from common.core.content_extractor import (
    ContentExtractor,
    ExtractionConfig,
    ContentExtractionResult
)


class TestExtractionConfig:
    """Test ExtractionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExtractionConfig()

        assert config.preferred_strategy == "hybrid"
        assert "beautifulsoup" in config.fallback_strategies
        assert "custom" in config.fallback_strategies
        assert config.min_content_length == 100
        assert config.max_content_length == 1000000
        assert config.min_word_count == 10
        assert config.min_quality_score == 0.3
        assert config.extract_links is True
        assert config.extract_images is True
        assert config.extract_headings is True
        assert config.extract_metadata is True
        assert config.clean_whitespace is True
        assert config.remove_scripts is True
        assert config.remove_styles is True
        assert config.detect_language is True

    def test_custom_config(self):
        """Test custom configuration values."""
        quality_weights = {'content_length': 0.5, 'paragraph_density': 0.5}
        supported_languages = {'en', 'es', 'fr'}

        config = ExtractionConfig(
            preferred_strategy="beautifulsoup",
            fallback_strategies=["readability"],
            min_content_length=200,
            min_quality_score=0.5,
            extract_links=False,
            quality_weights=quality_weights,
            supported_languages=supported_languages
        )

        assert config.preferred_strategy == "beautifulsoup"
        assert config.fallback_strategies == ["readability"]
        assert config.min_content_length == 200
        assert config.min_quality_score == 0.5
        assert config.extract_links is False
        assert config.quality_weights == quality_weights
        assert config.supported_languages == supported_languages


class TestContentExtractionResult:
    """Test ContentExtractionResult dataclass."""

    def test_result_creation(self):
        """Test creating ContentExtractionResult instances."""
        result = ContentExtractionResult(url="http://example.com")
        assert result.url == "http://example.com"
        assert result.main_content is None
        assert result.quality_score == 0.0
        assert isinstance(result.links, list)
        assert isinstance(result.images, list)
        assert isinstance(result.headings, list)
        assert isinstance(result.keywords, list)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)

    def test_result_with_full_data(self):
        """Test ContentExtractionResult with all fields populated."""
        links = [{'url': 'http://example.com/link', 'text': 'Link'}]
        images = [{'url': 'http://example.com/image.jpg', 'alt': 'Image'}]
        headings = [{'level': 1, 'text': 'Title'}]
        keywords = ['test', 'content']
        errors = ['Minor error']
        warnings = ['Warning message']
        metadata = {'custom': 'data'}

        result = ContentExtractionResult(
            url="http://example.com",
            main_content="Test content",
            title="Test Title",
            description="Test description",
            keywords=keywords,
            author="Test Author",
            language="en",
            quality_score=0.8,
            content_length=100,
            word_count=20,
            paragraph_count=2,
            links=links,
            images=images,
            headings=headings,
            extraction_strategy="hybrid",
            processing_time=1.5,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

        assert result.main_content == "Test content"
        assert result.title == "Test Title"
        assert result.quality_score == 0.8
        assert result.links == links
        assert result.images == images
        assert result.headings == headings
        assert result.errors == errors
        assert result.warnings == warnings
        assert result.metadata == metadata


class TestContentExtractor:
    """Test ContentExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create ContentExtractor instance for testing."""
        return ContentExtractor()

    @pytest.fixture
    def custom_extractor(self):
        """Create ContentExtractor with custom configuration."""
        config = ExtractionConfig(
            preferred_strategy="beautifulsoup",
            min_content_length=50,
            min_quality_score=0.2
        )
        return ContentExtractor(config)

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.config is not None
        assert isinstance(extractor._content_selectors, list)
        assert isinstance(extractor._noise_selectors, list)
        assert isinstance(extractor._metadata_selectors, dict)
        assert 'article' in extractor._content_selectors
        assert 'script' in extractor._noise_selectors

    def test_initialization_with_custom_config(self, custom_extractor):
        """Test extractor initialization with custom config."""
        assert custom_extractor.config.preferred_strategy == "beautifulsoup"
        assert custom_extractor.config.min_content_length == 50
        assert custom_extractor.config.min_quality_score == 0.2

    def test_clean_html_removes_comments(self, extractor):
        """Test HTML cleaning removes comments."""
        html = """
        <html>
            <body>
                <!-- This is a comment -->
                <p>Content</p>
                <!-- Another comment -->
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        cleaned = extractor._clean_html(soup)

        # Comments should be removed
        assert '<!-- This is a comment -->' not in str(cleaned)
        assert '<!-- Another comment -->' not in str(cleaned)
        assert '<p>Content</p>' in str(cleaned)

    def test_clean_html_removes_scripts_and_styles(self, extractor):
        """Test HTML cleaning removes scripts and styles."""
        html = """
        <html>
            <head>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <p>Content</p>
                <script>console.log('inline');</script>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        cleaned = extractor._clean_html(soup)

        # Scripts and styles should be removed
        assert 'alert' not in str(cleaned)
        assert 'color: red' not in str(cleaned)
        assert 'console.log' not in str(cleaned)
        assert '<p>Content</p>' in str(cleaned)

    def test_clean_html_removes_noise_elements(self, extractor):
        """Test HTML cleaning removes noise elements."""
        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <header>Header</header>
                <main>
                    <article>Main content</article>
                </main>
                <aside class="sidebar">Sidebar</aside>
                <footer>Footer</footer>
                <div class="advertisement">Ad content</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        cleaned = extractor._clean_html(soup)

        # Noise elements should be removed
        assert 'Navigation' not in str(cleaned)
        assert 'Header' not in str(cleaned)
        assert 'Sidebar' not in str(cleaned)
        assert 'Footer' not in str(cleaned)
        assert 'Ad content' not in str(cleaned)
        assert 'Main content' in str(cleaned)

    def test_extract_text_beautifulsoup_with_article(self, extractor):
        """Test BeautifulSoup extraction with article tag."""
        html = """
        <html>
            <body>
                <div class="sidebar">Sidebar content</div>
                <article>
                    <h1>Article Title</h1>
                    <p>First paragraph with meaningful content.</p>
                    <p>Second paragraph with more content.</p>
                </article>
                <footer>Footer content</footer>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_beautifulsoup(soup)

        assert "Article Title" in content
        assert "First paragraph" in content
        assert "Second paragraph" in content
        assert "Sidebar content" not in content
        assert "Footer content" not in content

    def test_extract_text_beautifulsoup_with_content_class(self, extractor):
        """Test BeautifulSoup extraction with content class."""
        html = """
        <html>
            <body>
                <div class="header">Header</div>
                <div class="content">
                    <h2>Main Content</h2>
                    <p>This is the main content area.</p>
                </div>
                <div class="sidebar">Sidebar</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_beautifulsoup(soup)

        assert "Main Content" in content
        assert "This is the main content area." in content
        assert "Header" not in content
        assert "Sidebar" not in content

    def test_extract_text_beautifulsoup_fallback_to_body(self, extractor):
        """Test BeautifulSoup extraction falls back to body."""
        html = """
        <html>
            <body>
                <h1>Page Title</h1>
                <p>Some content without specific selectors.</p>
                <p>More content here.</p>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_beautifulsoup(soup)

        assert "Page Title" in content
        assert "Some content without specific selectors." in content
        assert "More content here." in content

    def test_extract_text_readability_scores_paragraphs(self, extractor):
        """Test readability extraction scores paragraphs correctly."""
        html = """
        <html>
            <body>
                <p>Short.</p>
                <p>This is a medium-length paragraph with good content that should score well in the readability algorithm.</p>
                <p><a href="#">Link heavy paragraph</a> with <a href="#">lots of links</a> and <a href="#">link text</a>.</p>
                <p>Another quality paragraph with proper sentence structure and reasonable length. This should also score well.</p>
                <div>This is not a paragraph tag but has good content that might be scored.</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_readability(soup)

        # Should include high-scoring paragraphs
        assert "medium-length paragraph" in content
        assert "Another quality paragraph" in content
        # Should exclude short paragraph
        assert "Short." not in content
        # May exclude link-heavy paragraph due to link density penalty

    def test_extract_text_custom_finds_title(self, extractor):
        """Test custom extraction finds and includes title."""
        html = """
        <html>
            <body>
                <h1>Main Article Title</h1>
                <div class="content">
                    <p>Article content goes here.</p>
                    <p>More article content.</p>
                </div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_custom(soup)

        assert "Main Article Title" in content
        assert "Article content goes here." in content

    def test_extract_text_custom_finds_content_by_class(self, extractor):
        """Test custom extraction finds content by class patterns."""
        html = """
        <html>
            <body>
                <div class="post-content">
                    <p>Post content in classed div.</p>
                </div>
                <div class="sidebar">Sidebar content</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_custom(soup)

        assert "Post content in classed div." in content
        assert "Sidebar content" not in content

    def test_extract_text_custom_finds_paragraph_heavy_div(self, extractor):
        """Test custom extraction finds div with most paragraphs."""
        html = """
        <html>
            <body>
                <div>
                    <p>Single paragraph div.</p>
                </div>
                <div>
                    <p>First paragraph in multi-paragraph div.</p>
                    <p>Second paragraph in multi-paragraph div.</p>
                    <p>Third paragraph in multi-paragraph div.</p>
                </div>
                <div>Short div with no paragraphs.</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_custom(soup)

        assert "First paragraph in multi-paragraph div." in content
        assert "Second paragraph in multi-paragraph div." in content
        assert "Third paragraph in multi-paragraph div." in content

    def test_extract_text_hybrid_chooses_best_strategy(self, extractor):
        """Test hybrid extraction chooses best strategy based on quality."""
        html = """
        <html>
            <body>
                <article>
                    <h1>High Quality Article</h1>
                    <p>This is a well-structured article with good content quality. It has proper paragraphs and meaningful text.</p>
                    <p>Multiple paragraphs indicate good structure. The content is substantial and informative.</p>
                    <p>Quality content continues with appropriate length and depth.</p>
                </article>
                <div class="noise">Noise content</div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = extractor._extract_text_hybrid(soup)

        assert "High Quality Article" in content
        assert "well-structured article" in content
        assert "Multiple paragraphs" in content
        assert "Noise content" not in content

    def test_clean_text_normalizes_whitespace(self, extractor):
        """Test text cleaning normalizes whitespace."""
        text = "  Multiple   spaces    between\n\n\nwords  and   \n\n\n\n  paragraphs  "
        cleaned = extractor._clean_text(text)

        assert "Multiple spaces between" in cleaned
        assert "words and" in cleaned
        assert "paragraphs" in cleaned
        # Should have normalized whitespace
        assert "   " not in cleaned
        assert cleaned.startswith("Multiple")
        assert cleaned.endswith("paragraphs")

    def test_clean_text_handles_empty_input(self, extractor):
        """Test text cleaning handles empty input."""
        assert extractor._clean_text("") == ""
        assert extractor._clean_text(None) == ""

    def test_extract_metadata_title_from_title_tag(self, extractor):
        """Test metadata extraction finds title in title tag."""
        html = """
        <html>
            <head><title>Page Title</title></head>
            <body><h1>Different Heading</h1></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_metadata(soup, "http://example.com")

        assert metadata['title'] == "Page Title"

    def test_extract_metadata_title_from_h1(self, extractor):
        """Test metadata extraction finds title in h1 when title tag missing."""
        html = """
        <html>
            <body><h1>Main Heading</h1><p>Content</p></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_metadata(soup, "http://example.com")

        assert metadata['title'] == "Main Heading"

    def test_extract_metadata_description_from_meta(self, extractor):
        """Test metadata extraction finds description in meta tag."""
        html = """
        <html>
            <head>
                <meta name="description" content="Page description text">
            </head>
            <body><p>Content</p></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_metadata(soup, "http://example.com")

        assert metadata['description'] == "Page description text"

    def test_extract_metadata_keywords_from_meta(self, extractor):
        """Test metadata extraction finds keywords in meta tag."""
        html = """
        <html>
            <head>
                <meta name="keywords" content="keyword1, keyword2, keyword3">
            </head>
            <body><p>Content</p></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_metadata(soup, "http://example.com")

        assert metadata['keywords'] == ["keyword1", "keyword2", "keyword3"]

    def test_extract_metadata_author_from_meta(self, extractor):
        """Test metadata extraction finds author in meta tag."""
        html = """
        <html>
            <head>
                <meta name="author" content="John Doe">
            </head>
            <body><p>Content</p></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_metadata(soup, "http://example.com")

        assert metadata['author'] == "John Doe"

    def test_extract_metadata_disabled(self):
        """Test metadata extraction can be disabled."""
        config = ExtractionConfig(extract_metadata=False)
        extractor = ContentExtractor(config)

        html = """
        <html>
            <head><title>Page Title</title></head>
            <body><p>Content</p></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_metadata(soup, "http://example.com")

        # Should return empty dict when disabled
        assert metadata == {}

    def test_extract_links_finds_absolute_and_relative(self, extractor):
        """Test link extraction finds and resolves URLs."""
        html = """
        <html>
            <body>
                <a href="http://example.com/absolute">Absolute Link</a>
                <a href="/relative">Relative Link</a>
                <a href="../parent">Parent Link</a>
                <a href="mailto:test@example.com">Email Link</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor._extract_links(soup, "http://example.com/page/")

        assert len(links) >= 3  # Should find at least 3 valid links

        link_urls = [link['url'] for link in links]
        assert "http://example.com/absolute" in link_urls
        assert "http://example.com/relative" in link_urls
        assert "http://example.com/" in link_urls  # Parent link resolved

    def test_extract_links_includes_text_and_title(self, extractor):
        """Test link extraction includes text and title attributes."""
        html = """
        <html>
            <body>
                <a href="/link" title="Link Title">Link Text</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor._extract_links(soup, "http://example.com")

        assert len(links) == 1
        link = links[0]
        assert link['text'] == "Link Text"
        assert link['title'] == "Link Title"
        assert link['url'] == "http://example.com/link"

    def test_extract_links_disabled(self):
        """Test link extraction can be disabled."""
        config = ExtractionConfig(extract_links=False)
        extractor = ContentExtractor(config)

        html = """
        <html>
            <body>
                <a href="/link">Link</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor._extract_links(soup, "http://example.com")

        assert links == []

    def test_extract_images_finds_and_resolves_urls(self, extractor):
        """Test image extraction finds and resolves URLs."""
        html = """
        <html>
            <body>
                <img src="http://example.com/absolute.jpg" alt="Absolute Image">
                <img src="/relative.png" alt="Relative Image">
                <img src="../parent.gif" alt="Parent Image">
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = extractor._extract_images(soup, "http://example.com/page/")

        assert len(images) == 3

        image_urls = [img['url'] for img in images]
        assert "http://example.com/absolute.jpg" in image_urls
        assert "http://example.com/relative.png" in image_urls
        assert "http://example.com/parent.gif" in image_urls

    def test_extract_images_includes_attributes(self, extractor):
        """Test image extraction includes alt, title, and dimension attributes."""
        html = """
        <html>
            <body>
                <img src="/image.jpg" alt="Image Alt" title="Image Title" width="100" height="200">
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = extractor._extract_images(soup, "http://example.com")

        assert len(images) == 1
        image = images[0]
        assert image['alt'] == "Image Alt"
        assert image['title'] == "Image Title"
        assert image['width'] == "100"
        assert image['height'] == "200"

    def test_extract_images_disabled(self):
        """Test image extraction can be disabled."""
        config = ExtractionConfig(extract_images=False)
        extractor = ContentExtractor(config)

        html = """
        <html>
            <body>
                <img src="/image.jpg" alt="Image">
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        images = extractor._extract_images(soup, "http://example.com")

        assert images == []

    def test_extract_headings_finds_all_levels(self, extractor):
        """Test heading extraction finds all heading levels."""
        html = """
        <html>
            <body>
                <h1 id="h1-id">Heading 1</h1>
                <h2>Heading 2</h2>
                <h3>Heading 3</h3>
                <h4>Heading 4</h4>
                <h5>Heading 5</h5>
                <h6>Heading 6</h6>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        headings = extractor._extract_headings(soup)

        assert len(headings) == 6

        levels = [h['level'] for h in headings]
        assert levels == [1, 2, 3, 4, 5, 6]

        texts = [h['text'] for h in headings]
        assert "Heading 1" in texts
        assert "Heading 6" in texts

        # Check ID extraction
        h1_heading = next(h for h in headings if h['level'] == 1)
        assert h1_heading['id'] == "h1-id"

    def test_extract_headings_disabled(self):
        """Test heading extraction can be disabled."""
        config = ExtractionConfig(extract_headings=False)
        extractor = ContentExtractor(config)

        html = """
        <html>
            <body>
                <h1>Heading</h1>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        headings = extractor._extract_headings(soup)

        assert headings == []

    def test_detect_language_russian(self, extractor):
        """Test language detection for Russian text."""
        text = "Это русский текст с кириллицей."
        language = extractor._detect_language(text)
        assert language == 'ru'

    def test_detect_language_chinese(self, extractor):
        """Test language detection for Chinese text."""
        text = "这是中文文本。"
        language = extractor._detect_language(text)
        assert language == 'zh'

    def test_detect_language_japanese(self, extractor):
        """Test language detection for Japanese text."""
        text = "これは日本語のテキストです。"
        language = extractor._detect_language(text)
        assert language == 'ja'

    def test_detect_language_korean(self, extractor):
        """Test language detection for Korean text."""
        text = "이것은 한국어 텍스트입니다."
        language = extractor._detect_language(text)
        assert language == 'ko'

    def test_detect_language_spanish(self, extractor):
        """Test language detection for Spanish text."""
        text = "Este es un texto en español con ñ."
        language = extractor._detect_language(text)
        assert language == 'es'

    def test_detect_language_french(self, extractor):
        """Test language detection for French text."""
        text = "Ceci est un texte en français avec des accents."
        language = extractor._detect_language(text)
        assert language == 'fr'

    def test_detect_language_english_default(self, extractor):
        """Test language detection defaults to English."""
        text = "This is English text without special characters."
        language = extractor._detect_language(text)
        assert language == 'en'

    def test_detect_language_disabled(self):
        """Test language detection can be disabled."""
        config = ExtractionConfig(detect_language=False)
        extractor = ContentExtractor(config)

        text = "Any text should return None."
        language = extractor._detect_language(text)
        assert language is None

    def test_detect_language_empty_text(self, extractor):
        """Test language detection with empty text."""
        assert extractor._detect_language("") is None
        assert extractor._detect_language(None) is None

    def test_calculate_quality_score_high_quality(self, extractor):
        """Test quality score calculation for high-quality content."""
        content = """
        This is a high-quality article with substantial content and good structure.
        It contains multiple well-formed paragraphs with meaningful information.

        Each paragraph is appropriately sized and contains valuable content
        that readers would find informative and engaging.

        The content maintains good balance between information density
        and readability, making it an excellent example of quality text.
        """

        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>First paragraph...</p>
                <h2>Subtitle</h2>
                <p>Second paragraph...</p>
                <p>Third paragraph...</p>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        score = extractor._calculate_content_quality_score(content, soup)

        assert score > 0.5  # Should be high quality

    def test_calculate_quality_score_low_quality(self, extractor):
        """Test quality score calculation for low-quality content."""
        content = "Short text."

        html = """
        <html>
            <body>
                <div>
                    <a href="#">Link</a>
                    <a href="#">Another link</a>
                    <a href="#">More links</a>
                    Short text.
                </div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        score = extractor._calculate_content_quality_score(content, soup)

        assert score < 0.5  # Should be low quality

    def test_calculate_quality_score_empty_content(self, extractor):
        """Test quality score calculation for empty content."""
        content = ""
        soup = BeautifulSoup("<html><body></body></html>", 'html.parser')
        score = extractor._calculate_content_quality_score(content, soup)

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_extract_content_successful(self, extractor):
        """Test successful content extraction."""
        html = """
        <html>
            <head>
                <title>Test Article</title>
                <meta name="description" content="Test description">
                <meta name="author" content="Test Author">
            </head>
            <body>
                <article>
                    <h1>Article Title</h1>
                    <p>This is the main article content with substantial text.</p>
                    <p>Second paragraph with more content for testing.</p>
                </article>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        assert result.url == "http://example.com"
        assert result.title == "Test Article"
        assert result.description == "Test description"
        assert result.author == "Test Author"
        assert "main article content" in result.main_content
        assert "Second paragraph" in result.main_content
        assert result.content_length > 0
        assert result.word_count > 0
        assert result.quality_score > 0
        assert result.processing_time is not None
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_extract_content_with_custom_strategy(self, extractor):
        """Test content extraction with custom strategy."""
        html = """
        <html>
            <body>
                <div class="post-content">
                    <p>Custom strategy content.</p>
                </div>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com", strategy="custom")

        assert result.extraction_strategy == "custom"
        assert "Custom strategy content" in result.main_content

    @pytest.mark.asyncio
    async def test_extract_content_unknown_strategy_fallback(self, extractor):
        """Test content extraction with unknown strategy falls back."""
        html = "<html><body><p>Test content.</p></body></html>"

        result = await extractor.extract_content(html, "http://example.com", strategy="unknown")

        assert result.extraction_strategy == "beautifulsoup"
        assert len(result.warnings) > 0
        assert "Unknown strategy" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_extract_content_fallback_strategies(self, custom_extractor):
        """Test content extraction uses fallback strategies when primary fails."""
        # Create HTML that might fail primary strategy but work with fallback
        html = """
        <html>
            <body>
                <div>
                    <p>Fallback strategy content that meets minimum requirements.</p>
                </div>
            </body>
        </html>
        """

        result = await custom_extractor.extract_content(html, "http://example.com")

        # Should extract content successfully using fallback
        assert "Fallback strategy content" in result.main_content
        assert result.content_length >= custom_extractor.config.min_content_length

    @pytest.mark.asyncio
    async def test_extract_content_malformed_html(self, extractor):
        """Test content extraction with malformed HTML."""
        html = """
        <html>
            <body>
                <p>Unclosed paragraph
                <div>Nested without closing
                    <span>Some content here
                </div>
                <p>Another paragraph</p>
            </body>
        """  # Note: intentionally malformed

        result = await extractor.extract_content(html, "http://example.com")

        # Should still extract some content despite malformation
        assert result.main_content is not None
        assert len(result.main_content) > 0

    @pytest.mark.asyncio
    async def test_extract_content_empty_html(self, extractor):
        """Test content extraction with empty HTML."""
        result = await extractor.extract_content("", "http://example.com")

        assert result.main_content == ""
        assert result.content_length == 0
        assert result.word_count == 0

    @pytest.mark.asyncio
    async def test_extract_content_exception_handling(self, extractor):
        """Test content extraction handles exceptions gracefully."""
        # Invalid HTML that might cause parsing errors
        invalid_html = "<<<invalid>>>html<<<content>>>"

        result = await extractor.extract_content(invalid_html, "http://example.com")

        # Should have fallback content
        assert result.main_content is not None
        assert result.extraction_strategy == "fallback"

    @pytest.mark.asyncio
    async def test_extract_content_with_links_and_images(self, extractor):
        """Test content extraction includes links and images."""
        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>Content with <a href="/link">a link</a>.</p>
                <img src="/image.jpg" alt="Test image">
                <h2>Subtitle</h2>
                <p>More content.</p>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        assert len(result.links) > 0
        assert len(result.images) > 0
        assert len(result.headings) > 0
        assert result.metadata['links_count'] == len(result.links)
        assert result.metadata['images_count'] == len(result.images)

    @pytest.mark.asyncio
    async def test_extract_content_performance_tracking(self, extractor):
        """Test content extraction tracks processing time."""
        html = "<html><body><p>Simple content</p></body></html>"

        start_time = time.time()
        result = await extractor.extract_content(html, "http://example.com")
        end_time = time.time()

        assert result.processing_time is not None
        assert result.processing_time > 0
        assert result.processing_time < (end_time - start_time + 0.1)  # Allow some tolerance

    def test_validate_content_successful(self, extractor):
        """Test content validation with valid content."""
        result = ContentExtractionResult(
            url="http://example.com",
            main_content="This is a substantial piece of content that meets all quality requirements and length thresholds.",
            content_length=150,
            word_count=20,
            quality_score=0.8
        )

        assert extractor.validate_content(result) is True

    def test_validate_content_too_short(self, extractor):
        """Test content validation fails for too-short content."""
        result = ContentExtractionResult(
            url="http://example.com",
            main_content="Short",
            content_length=5,
            word_count=1,
            quality_score=0.8
        )

        assert extractor.validate_content(result) is False

    def test_validate_content_too_few_words(self, extractor):
        """Test content validation fails for too few words."""
        result = ContentExtractionResult(
            url="http://example.com",
            main_content="A" * 200,  # Long but only one word
            content_length=200,
            word_count=1,
            quality_score=0.8
        )

        assert extractor.validate_content(result) is False

    def test_validate_content_low_quality(self, extractor):
        """Test content validation fails for low quality score."""
        result = ContentExtractionResult(
            url="http://example.com",
            main_content="This content is long enough and has enough words to meet length requirements.",
            content_length=150,
            word_count=20,
            quality_score=0.1  # Below threshold
        )

        assert extractor.validate_content(result) is False

    def test_validate_content_no_content(self, extractor):
        """Test content validation fails for empty content."""
        result = ContentExtractionResult(
            url="http://example.com",
            main_content=None
        )

        assert extractor.validate_content(result) is False

    def test_get_content_summary(self, extractor):
        """Test content summary generation."""
        result = ContentExtractionResult(
            url="http://example.com",
            main_content="Test content",
            title="Test Title",
            content_length=100,
            word_count=20,
            paragraph_count=2,
            quality_score=0.8,
            language="en",
            extraction_strategy="hybrid",
            processing_time=1.5,
            links=[{'url': 'http://example.com/link'}],
            images=[{'url': 'http://example.com/image.jpg'}],
            headings=[{'level': 1, 'text': 'Heading'}],
            errors=[],
            warnings=[]
        )

        summary = extractor.get_content_summary(result)

        assert summary['url'] == "http://example.com"
        assert summary['title'] == "Test Title"
        assert summary['content_length'] == 100
        assert summary['word_count'] == 20
        assert summary['quality_score'] == 0.8
        assert summary['language'] == "en"
        assert summary['extraction_strategy'] == "hybrid"
        assert summary['processing_time'] == 1.5
        assert summary['links_count'] == 1
        assert summary['images_count'] == 1
        assert summary['headings_count'] == 1
        assert summary['has_errors'] is False
        assert summary['has_warnings'] is False


class TestContentExtractorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self):
        """Test handling of Unicode content in various languages."""
        extractor = ContentExtractor()

        unicode_html = """
        <html>
            <head>
                <title>Unicode Test: émojis 🌍 and açcénts</title>
                <meta name="description" content="Testing ñ, ü, ç characters">
            </head>
            <body>
                <article>
                    <h1>多言語テスト Multilingual Test</h1>
                    <p>English with émojis 🚀 and special characters.</p>
                    <p>Español con acentos: niño, café, corazón.</p>
                    <p>Français avec accents: être, naïve, façon.</p>
                    <p>Русский текст с кириллицей: привет мир.</p>
                    <p>中文内容：你好世界。</p>
                    <p>日本語コンテンツ：こんにちは世界。</p>
                </article>
            </body>
        </html>
        """

        result = await extractor.extract_content(unicode_html, "http://example.com")

        assert result.title == "Unicode Test: émojis 🌍 and açcénts"
        assert "🚀" in result.main_content
        assert "niño" in result.main_content
        assert "être" in result.main_content
        assert "привет" in result.main_content
        assert "你好" in result.main_content
        assert "こんにちは" in result.main_content

    @pytest.mark.asyncio
    async def test_extremely_large_content(self):
        """Test handling of extremely large content."""
        config = ExtractionConfig(max_content_length=1000)
        extractor = ContentExtractor(config)

        # Create large HTML content
        large_content = "This is a repeated paragraph. " * 100
        html = f"""
        <html>
            <body>
                <article>
                    <p>{large_content}</p>
                </article>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        # Should still process but may be truncated or have warnings
        assert result.main_content is not None
        assert len(result.main_content) > 0

    @pytest.mark.asyncio
    async def test_deeply_nested_html(self):
        """Test handling of deeply nested HTML structures."""
        extractor = ContentExtractor()

        # Create deeply nested HTML
        nested_html = "<div>" * 20 + "<p>Deep content</p>" + "</div>" * 20
        html = f"""
        <html>
            <body>
                <article>
                    {nested_html}
                </article>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        assert "Deep content" in result.main_content

    @pytest.mark.asyncio
    async def test_content_with_only_noise_elements(self):
        """Test extraction from content with only noise elements."""
        extractor = ContentExtractor()

        html = """
        <html>
            <body>
                <nav>Navigation only</nav>
                <header>Header only</header>
                <aside>Sidebar only</aside>
                <footer>Footer only</footer>
                <div class="ads">Advertisement only</div>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        # Should have minimal or empty content after noise removal
        assert len(result.main_content.strip()) <= 50  # Very little content expected

    @pytest.mark.asyncio
    async def test_content_with_only_scripts_and_styles(self):
        """Test extraction from content with only scripts and styles."""
        extractor = ContentExtractor()

        html = """
        <html>
            <head>
                <script>console.log('test');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <script>alert('hello');</script>
                <noscript>JavaScript disabled</noscript>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        # Should extract very little or no meaningful content
        assert "console.log" not in result.main_content
        assert "color: red" not in result.main_content
        assert "alert" not in result.main_content

    @pytest.mark.asyncio
    async def test_extraction_with_all_features_disabled(self):
        """Test extraction with all optional features disabled."""
        config = ExtractionConfig(
            extract_links=False,
            extract_images=False,
            extract_headings=False,
            extract_metadata=False,
            detect_language=False
        )
        extractor = ContentExtractor(config)

        html = """
        <html>
            <head>
                <title>Test Title</title>
                <meta name="description" content="Test description">
            </head>
            <body>
                <h1>Main Heading</h1>
                <p>Content with <a href="/link">link</a> and <img src="/image.jpg" alt="image">.</p>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        assert len(result.links) == 0
        assert len(result.images) == 0
        assert len(result.headings) == 0
        assert result.title is None  # Metadata extraction disabled
        assert result.language is None  # Language detection disabled
        assert "Content with" in result.main_content  # Main content should still work

    def test_quality_score_with_custom_weights(self):
        """Test quality score calculation with custom weights."""
        quality_weights = {
            'content_length': 0.5,
            'paragraph_density': 0.5,
            'link_ratio': 0.0,
            'structural_quality': 0.0,
            'language_quality': 0.0
        }
        config = ExtractionConfig(quality_weights=quality_weights)
        extractor = ContentExtractor(config)

        content = "A" * 1000  # Long content
        soup = BeautifulSoup("<html><body><p>Test</p></body></html>", 'html.parser')

        score = extractor._calculate_content_quality_score(content, soup)

        # Score should be based primarily on content length and paragraph density
        assert score > 0

    @pytest.mark.asyncio
    async def test_malformed_metadata_handling(self):
        """Test handling of malformed metadata tags."""
        extractor = ContentExtractor()

        html = """
        <html>
            <head>
                <meta name="description">  <!-- Missing content attribute -->
                <meta content="Orphaned content">  <!-- Missing name attribute -->
                <meta name="keywords" content="">  <!-- Empty content -->
                <title></title>  <!-- Empty title -->
            </head>
            <body>
                <p>Content</p>
            </body>
        </html>
        """

        result = await extractor.extract_content(html, "http://example.com")

        # Should handle malformed metadata gracefully
        assert isinstance(result.keywords, list)  # Should not crash