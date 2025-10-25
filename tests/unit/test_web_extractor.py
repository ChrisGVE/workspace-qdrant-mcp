"""Comprehensive unit tests for the ContentExtractor component.

Tests cover all extraction strategies, edge cases, and quality assessment.
"""

from unittest.mock import patch

import pytest
from workspace_qdrant_mcp.web.extractor import (
    ContentExtractor,
    ContentQuality,
    ExtractedContent,
    ExtractionStrategy,
)


class TestContentExtractor:
    """Test ContentExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Content extractor instance for testing."""
        return ContentExtractor()

    @pytest.fixture
    def sample_html(self):
        """Sample HTML content for testing."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Test Article - Sample News</title>
            <meta name="description" content="This is a test article for content extraction testing.">
            <meta name="keywords" content="test, article, extraction, content">
            <meta name="author" content="Test Author">
            <meta property="og:title" content="Test Article OG Title">
            <meta property="og:description" content="Test article Open Graph description">
        </head>
        <body>
            <header>
                <nav class="navigation">
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                </nav>
            </header>

            <main>
                <article>
                    <h1>Main Article Title</h1>
                    <div class="byline">
                        By <span class="author">John Doe</span>
                        <time datetime="2023-01-15T10:00:00Z">January 15, 2023</time>
                    </div>

                    <h2>Introduction</h2>
                    <p>This is the first paragraph of the article. It contains important information about the topic.</p>
                    <p>This is the second paragraph with more detailed information. It helps explain the concept better.</p>

                    <h2>Main Content</h2>
                    <p>Here we have the main content of the article. This section contains the most important information.</p>

                    <h3>Subsection</h3>
                    <p>A subsection with additional details. This provides more context and examples.</p>

                    <p>Links to external resources: <a href="https://example.com">Example Link</a> and <a href="https://test.com">Test Link</a>.</p>

                    <img src="/test-image.jpg" alt="Test Image" title="Test Image Title">
                    <img src="https://example.com/remote-image.png" alt="Remote Image">
                </article>
            </main>

            <aside class="sidebar">
                <div class="ad">Advertisement content</div>
                <div class="social">
                    <a href="https://twitter.com/share">Share on Twitter</a>
                </div>
            </aside>

            <footer>
                <p>© 2023 Test Site</p>
            </footer>

            <script>
                console.log("Analytics tracking");
            </script>
        </body>
        </html>
        """

    @pytest.fixture
    def malformed_html(self):
        """Malformed HTML for testing error handling."""
        return """
        <html>
        <head>
            <title>Malformed HTML
        </head>
        <body>
            <h1>Unclosed heading
            <p>Paragraph with <span>unclosed span
            <div>Unclosed div
            Content without proper structure
        </body>
        """

    @pytest.fixture
    def minimal_html(self):
        """Minimal HTML for testing edge cases."""
        return "<html><head><title>Minimal</title></head><body><p>Short content.</p></body></html>"

    @pytest.fixture
    def empty_html(self):
        """Empty HTML for testing edge cases."""
        return ""

    def test_extract_basic_content(self, extractor, sample_html):
        """Test basic content extraction."""
        result = extractor.extract(sample_html, "https://example.com")

        assert result.title == "Test Article - Sample News"
        assert result.meta_description == "This is a test article for content extraction testing."
        assert "test" in result.meta_keywords
        assert "article" in result.meta_keywords
        assert result.author == "Test Author"
        assert result.language == "en"
        assert result.publish_date == "2023-01-15T10:00:00Z"

    def test_extract_structured_content(self, extractor, sample_html):
        """Test extraction of structured content."""
        result = extractor.extract(sample_html, "https://example.com")

        # Check headings
        assert len(result.headings) >= 3
        headings_dict = dict(result.headings)
        assert 1 in headings_dict
        assert "Main Article Title" in headings_dict[1]
        assert 2 in headings_dict
        assert 3 in headings_dict

        # Check paragraphs
        assert len(result.paragraphs) >= 4
        assert any("first paragraph" in p for p in result.paragraphs)

        # Check links
        assert len(result.links) >= 2
        link_urls = [url for url, text in result.links]
        assert "https://example.com" in link_urls
        assert "https://test.com" in link_urls

        # Check images
        assert len(result.images) >= 2
        image_srcs = [img.get('src', '') for img in result.images]
        assert any("test-image.jpg" in src for src in image_srcs)

    def test_content_cleaning(self, extractor, sample_html):
        """Test content cleaning and noise removal."""
        result = extractor.extract(sample_html, "https://example.com")

        # Should not contain navigation or ads
        assert "Home" not in result.content
        assert "About" not in result.content
        assert "Advertisement" not in result.content
        assert "Share on Twitter" not in result.content

        # Should not contain script content
        assert "Analytics tracking" not in result.content
        assert "console.log" not in result.content

        # Should contain main content
        assert "first paragraph" in result.content
        assert "main content" in result.content

    def test_quality_assessment(self, extractor, sample_html):
        """Test content quality assessment."""
        result = extractor.extract(sample_html, "https://example.com")
        quality = result.quality

        # Basic quality indicators
        assert quality.has_title is True
        assert quality.has_meta_description is True
        assert quality.has_structured_content is True
        assert quality.language_detected == "en"

        # Content metrics
        assert quality.word_count > 20
        assert quality.sentence_count > 5
        assert quality.paragraph_count >= 4
        assert quality.heading_count >= 3
        assert quality.link_count >= 2

        # Quality scores should be reasonable
        assert 0 <= quality.overall_score <= 100
        assert 0 <= quality.text_density <= 100
        assert 0 <= quality.structure_score <= 100
        assert 0 <= quality.readability_score <= 100

    def test_extract_empty_html(self, extractor, empty_html):
        """Test extraction from empty HTML."""
        result = extractor.extract(empty_html, "https://example.com")

        assert result.title == ""
        assert result.content == ""
        assert result.meta_description == ""
        assert result.quality.word_count == 0
        assert result.extraction_strategy == ExtractionStrategy.BEAUTIFULSOUP

    def test_extract_minimal_html(self, extractor, minimal_html):
        """Test extraction from minimal HTML."""
        result = extractor.extract(minimal_html, "https://example.com")

        assert result.title == "Minimal"
        assert "Short content" in result.content
        assert result.quality.word_count >= 2

    def test_extract_malformed_html(self, extractor, malformed_html):
        """Test extraction from malformed HTML."""
        result = extractor.extract(malformed_html, "https://example.com")

        # Should still extract some content despite malformed HTML
        assert result.title == "Malformed HTML"
        assert "Content without proper structure" in result.content
        assert result.extraction_strategy == ExtractionStrategy.BEAUTIFULSOUP

    def test_beautifulsoup_strategy(self, extractor, sample_html):
        """Test BeautifulSoup extraction strategy specifically."""
        # Force BeautifulSoup strategy
        extractor.strategies = [ExtractionStrategy.BEAUTIFULSOUP]
        result = extractor.extract(sample_html, "https://example.com")

        assert result.extraction_strategy == ExtractionStrategy.BEAUTIFULSOUP
        assert result.title == "Test Article - Sample News"
        assert len(result.headings) >= 3
        assert len(result.paragraphs) >= 4

    def test_lxml_strategy(self, extractor, sample_html):
        """Test lxml extraction strategy."""
        # Force lxml strategy
        extractor.strategies = [ExtractionStrategy.LXML]
        result = extractor.extract(sample_html, "https://example.com")

        assert result.extraction_strategy == ExtractionStrategy.LXML
        assert result.title == "Test Article - Sample News"
        assert result.content != ""

    def test_regex_strategy(self, extractor, sample_html):
        """Test regex extraction strategy."""
        # Force regex strategy
        extractor.strategies = [ExtractionStrategy.REGEX]
        result = extractor.extract(sample_html, "https://example.com")

        assert result.extraction_strategy == ExtractionStrategy.REGEX
        assert result.title == "Test Article - Sample News"
        assert result.meta_description == "This is a test article for content extraction testing."

    def test_simple_text_strategy(self, extractor, sample_html):
        """Test simple text extraction strategy."""
        # Force simple text strategy
        extractor.strategies = [ExtractionStrategy.SIMPLE_TEXT]
        result = extractor.extract(sample_html, "https://example.com")

        assert result.extraction_strategy == ExtractionStrategy.SIMPLE_TEXT
        assert result.content != ""
        assert "first paragraph" in result.content

    def test_strategy_fallback(self, extractor, sample_html):
        """Test fallback between extraction strategies."""
        with patch('bs4.BeautifulSoup') as mock_bs4:
            # Make BeautifulSoup fail
            mock_bs4.side_effect = Exception("BeautifulSoup failed")

            result = extractor.extract(sample_html, "https://example.com")

            # Should fall back to lxml or other strategy
            assert result.extraction_strategy != ExtractionStrategy.BEAUTIFULSOUP
            assert result.content != ""

    def test_extract_with_base_url(self, extractor, sample_html):
        """Test link and image URL resolution with base URL."""
        base_url = "https://test-site.com/articles/"
        result = extractor.extract(sample_html, base_url)

        # Check that relative URLs are resolved
        absolute_urls = [url for url, text in result.links]
        assert any(url.startswith("https://test-site.com") for url in absolute_urls)

        # Check image URLs
        for img in result.images:
            if 'src' in img and img['src'].startswith('/'):
                assert img['src'].startswith('https://test-site.com')

    def test_extract_without_base_url(self, extractor, sample_html):
        """Test extraction without base URL."""
        result = extractor.extract(sample_html, "")

        # Should still extract content
        assert result.title == "Test Article - Sample News"
        assert result.content != ""
        assert len(result.links) > 0

    def test_title_extraction_fallbacks(self, extractor):
        """Test title extraction with various fallback scenarios."""
        # Test with missing title tag
        html_no_title = """
        <html>
        <body>
            <h1>Main Heading</h1>
            <meta property="og:title" content="OG Title">
        </body>
        </html>
        """
        result = extractor.extract(html_no_title)
        assert result.title == "Main Heading"

        # Test with only OG title
        html_og_only = """
        <html>
        <head><meta property="og:title" content="Only OG Title"></head>
        <body><p>Content</p></body>
        </html>
        """
        result = extractor.extract(html_og_only)
        assert result.title == "Only OG Title"

    def test_meta_description_fallbacks(self, extractor):
        """Test meta description extraction with fallbacks."""
        html_og_desc = """
        <html>
        <head>
            <meta property="og:description" content="Open Graph description">
        </head>
        <body><p>Content</p></body>
        </html>
        """
        result = extractor.extract(html_og_desc)
        assert result.meta_description == "Open Graph description"

    def test_author_extraction_variants(self, extractor):
        """Test author extraction from various sources."""
        html_author = """
        <html>
        <head><meta name="author" content="Meta Author"></head>
        <body>
            <div class="author">Article Author</div>
            <span class="by-author">Byline Author</span>
        </body>
        </html>
        """
        result = extractor.extract(html_author)
        # Should prefer meta author
        assert result.author == "Meta Author"

        # Test with only class-based author
        html_class_author = """
        <html>
        <body><div class="author">Class Author</div></body>
        </html>
        """
        result = extractor.extract(html_class_author)
        assert result.author == "Class Author"

    def test_language_detection(self, extractor):
        """Test language detection from various sources."""
        html_lang = """
        <html lang="es">
        <head><meta http-equiv="content-language" content="fr"></head>
        <body><p>Content</p></body>
        </html>
        """
        result = extractor.extract(html_lang)
        # Should prefer html lang attribute
        assert result.language == "es"

    def test_content_quality_edge_cases(self, extractor):
        """Test content quality assessment edge cases."""
        # Very short content
        short_html = "<html><body><p>Short.</p></body></html>"
        result = extractor.extract(short_html)
        assert "Very short content" in result.quality.issues or "Short content" in result.quality.issues

        # No structure
        no_structure = "<html><body>Just plain text without structure.</body></html>"
        result = extractor.extract(no_structure)
        assert "No headings found" in result.quality.issues
        assert "Missing title" in result.quality.issues

    def test_text_cleaning(self, extractor):
        """Test text cleaning functionality."""
        html_messy = """
        <html>
        <body>
            <p>Text   with    multiple     spaces</p>
            <p>Text&nbsp;with&nbsp;entities&amp;more</p>
            <p>Text\n\nwith\nnewlines</p>
        </body>
        </html>
        """
        result = extractor.extract(html_messy)

        # Should normalize whitespace
        assert "multiple     spaces" not in result.content
        assert "Text with multiple spaces" in result.content

        # Should decode HTML entities
        assert "&nbsp;" not in result.content
        assert "&amp;" not in result.content

    def test_summary_generation(self, extractor, sample_html):
        """Test summary generation."""
        result = extractor.extract(sample_html)

        # Summary should be shorter than full content
        assert len(result.summary) < len(result.cleaned_content)
        assert result.summary.endswith('.')
        assert len(result.summary) > 0

        # Test with very short content
        short_html = "<html><body><p>Very short content here.</p></body></html>"
        result_short = extractor.extract(short_html)
        # Summary should be the full content for short text
        assert "Very short content here" in result_short.summary

    def test_lxml_extraction_edge_cases(self, extractor):
        """Test lxml extraction with edge cases."""
        # Test with invalid XML that might cause lxml to fail
        invalid_xml = "<html><body><p>Unclosed paragraph<div>Mixed content</body></html>"

        # Force lxml strategy
        extractor.strategies = [ExtractionStrategy.LXML]
        result = extractor.extract(invalid_xml)

        # Should handle gracefully
        assert result.content != ""

    def test_regex_extraction_edge_cases(self, extractor):
        """Test regex extraction with edge cases."""
        html_complex = """
        <html>
        <head>
            <title>Complex
            Title</title>
            <meta name="description" content="Description with 'quotes' and &entities;">
        </head>
        <body>
            <script>var title = "Fake Title";</script>
            <style>body { content: "Fake Content"; }</style>
            <p>Real content here.</p>
        </body>
        </html>
        """

        # Force regex strategy
        extractor.strategies = [ExtractionStrategy.REGEX]
        result = extractor.extract(html_complex)

        assert "Complex Title" in result.title
        assert "Description with" in result.meta_description
        assert "Fake Title" not in result.content  # Should remove script
        assert "Fake Content" not in result.content  # Should remove style
        assert "Real content" in result.content

    def test_image_extraction_comprehensive(self, extractor):
        """Test comprehensive image extraction."""
        html_images = """
        <html>
        <body>
            <img src="/local-image.jpg" alt="Local Image" title="Local Title">
            <img data-src="/lazy-image.png" alt="Lazy Loaded">
            <img src="https://example.com/remote.gif" alt="Remote Image">
            <img src="image-no-alt.jpg">
            <img alt="No source image">
        </body>
        </html>
        """

        result = extractor.extract(html_images, "https://test.com")

        # Should extract valid images
        assert len(result.images) >= 3

        # Check image attributes
        image_srcs = {img.get('src', '') for img in result.images}
        assert any('local-image.jpg' in src for src in image_srcs)
        assert any('lazy-image.png' in src for src in image_srcs)
        assert 'https://example.com/remote.gif' in image_srcs

        # Check alt text extraction
        image_alts = {img.get('alt', '') for img in result.images}
        assert 'Local Image' in image_alts
        assert 'Lazy Loaded' in image_alts

    def test_link_extraction_comprehensive(self, extractor):
        """Test comprehensive link extraction."""
        html_links = """
        <html>
        <body>
            <a href="/local-page">Local Link</a>
            <a href="https://external.com">External Link</a>
            <a href="mailto:test@example.com">Email Link</a>
            <a href="javascript:void(0)">JavaScript Link</a>
            <a>No href link</a>
            <a href="/empty-text"></a>
        </body>
        </html>
        """

        result = extractor.extract(html_links, "https://base.com")

        # Should extract links with text
        link_data = {(url, text) for url, text in result.links}

        # Check various link types
        assert any('base.com' in url and 'Local Link' in text for url, text in link_data)
        assert any('external.com' in url and 'External Link' in text for url, text in link_data)
        assert any('mailto:' in url and 'Email Link' in text for url, text in link_data)

    def test_processing_time_measurement(self, extractor, sample_html):
        """Test that processing time is measured."""
        result = extractor.extract(sample_html)
        assert result.processing_time >= 0
        assert isinstance(result.processing_time, float)

    def test_raw_html_storage(self, extractor, sample_html):
        """Test that raw HTML is stored (truncated)."""
        result = extractor.extract(sample_html)
        assert result.raw_html != ""
        assert len(result.raw_html) <= 10000  # Should be truncated

    def test_quality_score_calculation(self, extractor):
        """Test quality score calculation components."""
        # High quality content
        high_quality_html = """
        <html lang="en">
        <head>
            <title>Well Structured Article Title</title>
            <meta name="description" content="Comprehensive description of the article content.">
        </head>
        <body>
            <article>
                <h1>Main Title</h1>
                <h2>Section One</h2>
                <p>First paragraph with substantial content that explains the topic clearly.</p>
                <p>Second paragraph that builds on the first with additional information.</p>
                <h2>Section Two</h2>
                <p>Third paragraph with even more detailed information about the subject.</p>
                <h3>Subsection</h3>
                <p>Fourth paragraph in a subsection providing specific details and examples.</p>
            </article>
        </body>
        </html>
        """

        result = extractor.extract(high_quality_html)
        quality = result.quality

        # Should have high quality scores
        assert quality.overall_score > 50
        assert quality.structure_score > 60
        assert quality.has_title is True
        assert quality.has_meta_description is True
        assert quality.has_structured_content is True

    def test_content_extraction_robustness(self, extractor):
        """Test extraction robustness with various challenging inputs."""
        # Test with only whitespace
        whitespace_html = "<html><body>   \n\t   </body></html>"
        result = extractor.extract(whitespace_html)
        assert result.content == ""

        # Test with binary-like content
        binary_html = "<html><body>\x00\x01\x02Invalid content</body></html>"
        result = extractor.extract(binary_html)
        assert "Invalid content" in result.content
        assert "\x00" not in result.content  # Should be cleaned

        # Test with very long content
        long_content = "<html><body>" + "<p>" + "word " * 10000 + "</p>" + "</body></html>"
        result = extractor.extract(long_content)
        assert result.quality.word_count > 5000

    def test_extraction_strategy_selection(self, extractor):
        """Test extraction strategy selection and fallback logic."""
        html = "<html><body><p>Test content</p></body></html>"

        # Test that strategies are tried in order
        with patch.object(extractor, '_extract_with_beautifulsoup') as mock_bs4, \
             patch.object(extractor, '_extract_with_lxml') as mock_lxml:

            # Make first strategy fail
            mock_bs4.side_effect = Exception("BeautifulSoup failed")
            mock_lxml.return_value = ExtractedContent(
                content="Test content",
                extraction_strategy=ExtractionStrategy.LXML
            )

            result = extractor.extract(html)
            assert result.extraction_strategy == ExtractionStrategy.LXML
            assert mock_bs4.called
            assert mock_lxml.called


@pytest.mark.performance
def test_extraction_performance():
    """Test extraction performance with large content."""
    extractor = ContentExtractor()

    # Generate large HTML content
    large_html = """
    <html>
    <head><title>Large Document</title></head>
    <body>
    """ + "\n".join([f"<p>This is paragraph number {i} with some content.</p>" for i in range(1000)]) + """
    </body>
    </html>
    """

    import time
    start_time = time.time()
    result = extractor.extract(large_html)
    extraction_time = time.time() - start_time

    # Should complete within reasonable time (5 seconds)
    assert extraction_time < 5.0
    assert result.quality.word_count > 5000
    assert result.quality.paragraph_count == 1000


class TestContentQuality:
    """Test ContentQuality dataclass functionality."""

    def test_quality_initialization(self):
        """Test ContentQuality initialization."""
        quality = ContentQuality()

        assert quality.overall_score == 0.0
        assert quality.text_density == 0.0
        assert quality.structure_score == 0.0
        assert quality.readability_score == 0.0
        assert quality.word_count == 0
        assert quality.has_title is False
        assert quality.issues == []

    def test_quality_with_values(self):
        """Test ContentQuality with specific values."""
        issues = ["Missing title", "Short content"]
        quality = ContentQuality(
            overall_score=75.0,
            word_count=500,
            has_title=True,
            issues=issues
        )

        assert quality.overall_score == 75.0
        assert quality.word_count == 500
        assert quality.has_title is True
        assert quality.issues == issues


class TestExtractedContent:
    """Test ExtractedContent dataclass functionality."""

    def test_content_initialization(self):
        """Test ExtractedContent initialization."""
        content = ExtractedContent()

        assert content.title == ""
        assert content.content == ""
        assert content.meta_keywords == []
        assert content.headings == []
        assert content.paragraphs == []
        assert content.links == []
        assert content.images == []
        assert isinstance(content.quality, ContentQuality)
        assert content.extraction_strategy == ExtractionStrategy.BEAUTIFULSOUP

    def test_content_with_values(self):
        """Test ExtractedContent with specific values."""
        headings = [(1, "Title"), (2, "Subtitle")]
        links = [("https://example.com", "Example")]

        content = ExtractedContent(
            title="Test Title",
            content="Test content",
            headings=headings,
            links=links,
            extraction_strategy=ExtractionStrategy.LXML
        )

        assert content.title == "Test Title"
        assert content.content == "Test content"
        assert content.headings == headings
        assert content.links == links
        assert content.extraction_strategy == ExtractionStrategy.LXML
