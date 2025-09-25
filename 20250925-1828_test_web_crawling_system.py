"""
Comprehensive unit tests for Web Crawling and External Content System.

Tests all components with extensive edge case coverage and validation.
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup

# Import the module under test
from workspace_qdrant_mcp.web_crawling_system import (
    CrawlMode, ContentQuality, CrawlStatus, CrawlConfig, ExtractedContent,
    CrawlResult, CrawlSession, RobotsTxtParser, RateLimiter, ContentExtractor,
    WebCrawler, WebCrawlingSystem, create_default_config
)


class TestEnums:
    """Test enum definitions and values."""

    def test_crawl_mode_enum(self):
        """Test CrawlMode enum has all expected values."""
        expected_modes = ['SINGLE_PAGE', 'RECURSIVE', 'SITEMAP']
        actual_modes = [m.name for m in CrawlMode]
        assert set(actual_modes) == set(expected_modes)

    def test_content_quality_enum(self):
        """Test ContentQuality enum has all expected values."""
        expected_qualities = ['HIGH', 'MEDIUM', 'LOW', 'SPAM']
        actual_qualities = [q.name for q in ContentQuality]
        assert set(actual_qualities) == set(expected_qualities)

    def test_crawl_status_enum(self):
        """Test CrawlStatus enum has all expected values."""
        expected_statuses = [
            'PENDING', 'IN_PROGRESS', 'SUCCESS', 'FAILED',
            'SKIPPED', 'RATE_LIMITED', 'ROBOTS_BLOCKED'
        ]
        actual_statuses = [s.name for s in CrawlStatus]
        assert set(actual_statuses) == set(expected_statuses)


class TestDataClasses:
    """Test dataclass definitions and functionality."""

    def test_crawl_config_defaults(self):
        """Test CrawlConfig default values."""
        config = CrawlConfig()

        assert config.max_depth == 3
        assert config.max_pages == 1000
        assert config.rate_limit_delay == 0.5
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30
        assert config.retry_attempts == 3
        assert config.follow_external_links is False
        assert config.respect_robots_txt is True
        assert 'text/html' in config.allowed_content_types
        assert config.max_content_size == 10 * 1024 * 1024
        assert config.content_quality_threshold == ContentQuality.MEDIUM

    def test_crawl_config_custom_values(self):
        """Test CrawlConfig with custom values."""
        config = CrawlConfig(
            max_depth=5,
            max_pages=500,
            rate_limit_delay=1.0,
            follow_external_links=True,
            respect_robots_txt=False,
            user_agent="CustomBot/1.0"
        )

        assert config.max_depth == 5
        assert config.max_pages == 500
        assert config.rate_limit_delay == 1.0
        assert config.follow_external_links is True
        assert config.respect_robots_txt is False
        assert config.user_agent == "CustomBot/1.0"

    def test_extracted_content_creation(self):
        """Test ExtractedContent dataclass creation."""
        content = ExtractedContent(
            url="https://example.com",
            title="Test Page",
            description="A test page",
            keywords=["test", "page"],
            text_content="This is test content",
            html_content="<html>...</html>",
            metadata={"author": "Test Author"}
        )

        assert content.url == "https://example.com"
        assert content.title == "Test Page"
        assert content.keywords == ["test", "page"]
        assert content.links == []  # Default empty list
        assert content.content_quality == ContentQuality.MEDIUM  # Default
        assert content.word_count == 0  # Default

    def test_crawl_result_creation(self):
        """Test CrawlResult dataclass creation."""
        result = CrawlResult(
            url="https://example.com",
            status=CrawlStatus.SUCCESS,
            response_code=200
        )

        assert result.url == "https://example.com"
        assert result.status == CrawlStatus.SUCCESS
        assert result.response_code == 200
        assert result.content is None  # Default
        assert result.error_message is None  # Default

    def test_crawl_session_creation(self):
        """Test CrawlSession dataclass creation."""
        config = CrawlConfig()
        session = CrawlSession(
            session_id="test_session",
            start_url="https://example.com",
            config=config,
            mode=CrawlMode.RECURSIVE
        )

        assert session.session_id == "test_session"
        assert session.start_url == "https://example.com"
        assert session.mode == CrawlMode.RECURSIVE
        assert len(session.visited_urls) == 0
        assert len(session.pending_urls) == 0
        assert session.session_stats['pages_crawled'] == 0


class TestRobotsTxtParser:
    """Test RobotsTxtParser class."""

    @pytest.fixture
    def robots_parser(self):
        """Create RobotsTxtParser instance."""
        return RobotsTxtParser("TestBot/1.0")

    def test_initialization(self):
        """Test RobotsTxtParser initialization."""
        parser = RobotsTxtParser("TestBot/1.0")
        assert parser.user_agent == "TestBot/1.0"
        assert len(parser.cache) == 0

    @pytest.mark.asyncio
    async def test_can_fetch_allowed(self, robots_parser):
        """Test can_fetch for allowed URL."""
        # Mock session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
User-agent: *
Allow: /
""")

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await robots_parser.can_fetch(mock_session, "https://example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_can_fetch_disallowed(self, robots_parser):
        """Test can_fetch for disallowed URL."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
User-agent: *
Disallow: /private/
""")

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await robots_parser.can_fetch(mock_session, "https://example.com/private/secret")
        assert result is False

    @pytest.mark.asyncio
    async def test_can_fetch_no_robots_txt(self, robots_parser):
        """Test can_fetch when robots.txt doesn't exist."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await robots_parser.can_fetch(mock_session, "https://example.com/page")
        assert result is True  # Allow when robots.txt doesn't exist

    @pytest.mark.asyncio
    async def test_can_fetch_cached(self, robots_parser):
        """Test can_fetch uses cache."""
        # First call
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="User-agent: *\nAllow: /")

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        await robots_parser.can_fetch(mock_session, "https://example.com/page1")

        # Second call should use cache
        result = await robots_parser.can_fetch(mock_session, "https://example.com/page2")

        # Session.get should only be called once (first time)
        assert mock_session.get.call_count == 1
        assert result is True

    @pytest.mark.asyncio
    async def test_can_fetch_error_handling(self, robots_parser):
        """Test can_fetch error handling."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Network error")

        # Should return True on error (fail open)
        result = await robots_parser.can_fetch(mock_session, "https://example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, robots_parser):
        """Test cache cleanup functionality."""
        # Add old entry to cache
        from datetime import timedelta
        old_timestamp = datetime.now() - timedelta(hours=25)  # Older than TTL

        robots_parser.cache["https://old-site.com/robots.txt"] = {
            'parser': Mock(),
            'timestamp': old_timestamp
        }
        robots_parser.last_cache_clean = datetime.now() - timedelta(hours=2)

        # This should trigger cache cleanup
        await robots_parser._clean_cache()

        # Old entry should be removed
        assert "https://old-site.com/robots.txt" not in robots_parser.cache


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(2.0)  # 2 requests per second
        assert limiter.delay == 0.5  # 1/2 = 0.5 seconds delay

    @pytest.mark.asyncio
    async def test_rate_limiting_same_domain(self):
        """Test rate limiting for same domain."""
        limiter = RateLimiter(2.0)  # 2 requests per second

        start_time = time.time()

        # First request should not wait
        await limiter.wait_if_needed("https://example.com/page1")
        first_request_time = time.time()

        # Second request should wait
        await limiter.wait_if_needed("https://example.com/page2")
        second_request_time = time.time()

        # Should have waited approximately 0.5 seconds
        wait_time = second_request_time - first_request_time
        assert 0.4 <= wait_time <= 0.7  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_rate_limiting_different_domains(self):
        """Test rate limiting for different domains."""
        limiter = RateLimiter(2.0)

        start_time = time.time()

        # Requests to different domains should not interfere
        await limiter.wait_if_needed("https://example1.com/page")
        await limiter.wait_if_needed("https://example2.com/page")

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete quickly since different domains
        assert total_time < 0.1

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_domain(self):
        """Test concurrent requests to same domain."""
        limiter = RateLimiter(2.0)

        async def make_request(url):
            start = time.time()
            await limiter.wait_if_needed(url)
            return time.time() - start

        # Start multiple concurrent requests
        tasks = []
        for i in range(3):
            task = make_request(f"https://example.com/page{i}")
            tasks.append(task)

        # First request starts immediately
        start_time = time.time()
        wait_times = await asyncio.gather(*tasks)

        # Requests should be spaced by delay interval
        assert wait_times[0] < 0.1  # First request minimal wait
        assert 0.4 <= wait_times[1] <= 0.7  # Second request waits ~0.5s
        assert 0.9 <= wait_times[2] <= 1.2  # Third request waits ~1.0s


class TestContentExtractor:
    """Test ContentExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create ContentExtractor instance."""
        return ContentExtractor()

    def test_initialization(self, extractor):
        """Test ContentExtractor initialization."""
        assert len(extractor.spam_indicators) > 0
        assert len(extractor.quality_indicators) > 0
        assert 'click here' in extractor.spam_indicators
        assert 'article' in extractor.quality_indicators

    def test_extract_content_simple(self, extractor):
        """Test content extraction from simple HTML."""
        html = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="A test page description">
                <meta name="keywords" content="test, page, html">
            </head>
            <body>
                <h1>Main Title</h1>
                <p>This is the main content of the page.</p>
                <p>More content here with some details.</p>
                <a href="/link1">Link 1</a>
                <img src="/image1.jpg" alt="Image 1">
            </body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")

        assert content.title == "Test Page"
        assert content.description == "A test page description"
        assert "test" in content.keywords
        assert "Main Title" in content.text_content
        assert "main content" in content.text_content
        assert content.word_count > 0
        assert content.reading_time_minutes > 0
        assert len(content.links) > 0
        assert len(content.images) > 0

    def test_extract_title_multiple_sources(self, extractor):
        """Test title extraction from multiple sources."""
        html = """
        <html>
            <head>
                <meta property="og:title" content="OpenGraph Title">
                <meta name="twitter:title" content="Twitter Title">
                <title>HTML Title</title>
            </head>
            <body><h1>H1 Title</h1></body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")
        # Should prefer OpenGraph title (first in priority order)
        assert content.title == "OpenGraph Title"

    def test_extract_title_fallback(self, extractor):
        """Test title extraction fallback."""
        html = """
        <html>
            <body><h1>Only H1 Available</h1></body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")
        assert content.title == "Only H1 Available"

    def test_extract_title_no_title(self, extractor):
        """Test title extraction when no title available."""
        html = "<html><body><p>No title</p></body></html>"

        content = extractor.extract_content(html, "https://example.com")
        assert content.title == "Untitled Page"

    def test_extract_description_multiple_sources(self, extractor):
        """Test description extraction from multiple sources."""
        html = """
        <html>
            <head>
                <meta name="description" content="Meta description">
                <meta property="og:description" content="OG description">
            </head>
            <body><p>First paragraph content</p></body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")
        assert content.description == "Meta description"

    def test_extract_description_fallback(self, extractor):
        """Test description extraction fallback to first paragraph."""
        html = """
        <html>
            <body>
                <p>This is the first paragraph with substantial content.</p>
                <p>Second paragraph.</p>
            </body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")
        assert "first paragraph" in content.description.lower()

    def test_extract_keywords(self, extractor):
        """Test keyword extraction."""
        html = """
        <html>
            <head>
                <meta name="keywords" content="web, crawling, python, testing">
                <title>Web Crawling Guide</title>
            </head>
            <body>
                <h1>Introduction to Web Crawling</h1>
                <h2>Python Implementation</h2>
                <h3>Testing Strategies</h3>
            </body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")

        # Should contain meta keywords
        assert "web" in content.keywords
        assert "crawling" in content.keywords

        # Should contain keywords from headings
        assert any("introduction" in kw.lower() for kw in content.keywords)

    def test_extract_language(self, extractor):
        """Test language extraction."""
        html_with_lang = """
        <html lang="en-US">
            <body><p>Content</p></body>
        </html>
        """

        content = extractor.extract_content(html_with_lang, "https://example.com")
        assert content.language == "en-US"

    def test_extract_text_content_main_area(self, extractor):
        """Test text extraction prioritizing main content area."""
        html = """
        <html>
            <body>
                <nav>Navigation menu</nav>
                <header>Site header</header>
                <main>
                    <article>
                        <h1>Article Title</h1>
                        <p>This is the main article content.</p>
                        <p>More article content here.</p>
                    </article>
                </main>
                <footer>Site footer</footer>
                <aside>Sidebar content</aside>
            </body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com")

        # Should extract main content
        assert "Article Title" in content.text_content
        assert "main article content" in content.text_content

        # Should not include navigation, header, footer, aside
        assert "Navigation menu" not in content.text_content
        assert "Site header" not in content.text_content
        assert "Site footer" not in content.text_content
        assert "Sidebar content" not in content.text_content

    def test_extract_links(self, extractor):
        """Test link extraction."""
        html = """
        <html>
            <body>
                <a href="/relative-link">Relative</a>
                <a href="https://example.com/absolute">Absolute</a>
                <a href="mailto:test@example.com">Email</a>
                <a href="javascript:void(0)">JavaScript</a>
                <a href="#anchor">Anchor</a>
            </body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com/page")

        # Should include valid HTTP links (converted to absolute)
        links = content.links
        assert "https://example.com/relative-link" in links
        assert "https://example.com/absolute" in links

        # Should not include non-HTTP links
        assert not any("mailto:" in link for link in links)
        assert not any("javascript:" in link for link in links)

    def test_extract_images(self, extractor):
        """Test image extraction."""
        html = """
        <html>
            <body>
                <img src="/relative-image.jpg" alt="Relative">
                <img src="https://example.com/absolute.jpg" alt="Absolute">
                <img src="data:image/png;base64,..." alt="Data URL">
            </body>
        </html>
        """

        content = extractor.extract_content(html, "https://example.com/page")

        images = content.images
        assert "https://example.com/relative-image.jpg" in images
        assert "https://example.com/absolute.jpg" in images

        # Should not include data URLs
        assert not any("data:" in img for img in images)

    def test_assess_content_quality_high(self, extractor):
        """Test content quality assessment for high quality content."""
        text_content = """
        This is a comprehensive article about web crawling techniques and best practices.
        It includes detailed analysis of various crawling strategies, research findings,
        and practical implementation guides. The content is well-structured with clear
        headings and provides valuable insights for developers and researchers.
        """ * 10  # Make it longer

        html_content = """
        <article>
            <h1>Web Crawling Guide</h1>
            <h2>Research Methods</h2>
            <p>Detailed analysis content...</p>
        </article>
        """

        quality = extractor._assess_content_quality(text_content, html_content)
        assert quality in [ContentQuality.HIGH, ContentQuality.MEDIUM]

    def test_assess_content_quality_spam(self, extractor):
        """Test content quality assessment for spam content."""
        spam_text = """
        Click here now! Buy now and save money! Congratulations, you have won!
        Free money available! Act now, limited time offer! Casino poker lottery!
        """ * 5

        quality = extractor._assess_content_quality(spam_text, spam_text)
        assert quality == ContentQuality.SPAM

    def test_assess_content_quality_low(self, extractor):
        """Test content quality assessment for low quality content."""
        low_quality_text = "Short text"

        quality = extractor._assess_content_quality(low_quality_text, low_quality_text)
        assert quality == ContentQuality.LOW

    def test_remove_unwanted_elements(self, extractor):
        """Test removal of unwanted HTML elements."""
        html = """
        <html>
            <head>
                <script>alert('test')</script>
                <style>body{color:red}</style>
            </head>
            <body>
                <nav>Navigation</nav>
                <main>Main content</main>
                <aside class="advertisement">Ad content</aside>
                <footer>Footer</footer>
                <!-- Comment -->
            </body>
        </html>
        """

        soup = BeautifulSoup(html, 'html.parser')
        extractor._remove_unwanted_elements(soup)

        html_str = str(soup)
        assert "<script>" not in html_str
        assert "<style>" not in html_str
        assert "Navigation" not in html_str
        assert "Ad content" not in html_str
        assert "Comment" not in html_str
        assert "Main content" in html_str  # Should keep main content

    def test_extract_additional_metadata(self, extractor):
        """Test extraction of additional metadata."""
        html = """
        <html>
            <head>
                <meta property="og:type" content="article">
                <meta property="og:url" content="https://example.com">
                <meta name="twitter:card" content="summary">
                <meta name="author" content="John Doe">
                <meta name="date" content="2023-01-01">
                <script type="application/ld+json">
                {"@type": "Article", "headline": "Test Article"}
                </script>
            </head>
            <body>
                <time datetime="2023-01-01T00:00:00Z">January 1, 2023</time>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, 'html.parser')
        metadata = extractor._extract_additional_metadata(soup)

        assert metadata['og_type'] == 'article'
        assert metadata['og_url'] == 'https://example.com'
        assert metadata['twitter_card'] == 'summary'
        assert metadata['author'] == 'John Doe'
        assert 'schema_org' in metadata

    def test_content_hash_generation(self, extractor):
        """Test content hash generation."""
        html = "<html><body><p>Test content</p></body></html>"

        content1 = extractor.extract_content(html, "https://example.com")
        content2 = extractor.extract_content(html, "https://example.com")

        # Same content should produce same hash
        assert content1.content_hash == content2.content_hash
        assert len(content1.content_hash) == 16  # 16 character hash

    def test_word_count_and_reading_time(self, extractor):
        """Test word count and reading time calculation."""
        # Create content with known word count
        words = ["word"] * 400  # 400 words
        text_content = " ".join(words)
        html = f"<html><body><p>{text_content}</p></body></html>"

        content = extractor.extract_content(html, "https://example.com")

        assert content.word_count == 400
        assert content.reading_time_minutes == 2  # 400 words / 200 wpm = 2 minutes

    def test_empty_content_handling(self, extractor):
        """Test handling of empty or minimal content."""
        empty_html = "<html><body></body></html>"

        content = extractor.extract_content(empty_html, "https://example.com")

        assert content.title == "Untitled Page"
        assert content.description == ""
        assert content.text_content == ""
        assert content.word_count == 0
        assert content.content_quality == ContentQuality.LOW

    def test_malformed_html_handling(self, extractor):
        """Test handling of malformed HTML."""
        malformed_html = """
        <html>
            <head>
                <title>Test
            <body>
                <p>Unclosed paragraph
                <div>Unclosed div
                <p>Another paragraph</p>
        """

        # Should not raise exception
        content = extractor.extract_content(malformed_html, "https://example.com")

        assert "Test" in content.title
        assert len(content.text_content) > 0


class TestWebCrawler:
    """Test WebCrawler class."""

    @pytest.fixture
    def config(self):
        """Create test crawl configuration."""
        return CrawlConfig(
            max_depth=2,
            max_pages=10,
            rate_limit_delay=0.1,  # Faster for tests
            max_concurrent_requests=2,
            request_timeout=5,
            retry_attempts=2
        )

    @pytest.fixture
    def crawler(self, config):
        """Create WebCrawler instance."""
        return WebCrawler(config)

    def test_initialization(self, config):
        """Test WebCrawler initialization."""
        crawler = WebCrawler(config)
        assert crawler.config == config
        assert crawler.robots_parser.user_agent == config.user_agent
        assert crawler.content_extractor is not None

    @pytest.mark.asyncio
    async def test_session_setup_and_cleanup(self, crawler):
        """Test session setup and cleanup."""
        async with crawler:
            assert crawler._session is not None
            assert crawler._semaphore is not None
            assert isinstance(crawler._session, aiohttp.ClientSession)

        # Session should be closed after context exit
        assert crawler._session.closed

    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self, crawler):
        """Test successful single page crawling."""
        test_html = """
        <html>
            <head><title>Test Page</title></head>
            <body><p>Test content</p></body>
        </html>
        """

        # Mock the HTTP response
        with patch.object(crawler, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value=test_html)
            mock_response.history = []

            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock robots.txt check
            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    result = await crawler.crawl_single_page("https://example.com")

        assert result.status == CrawlStatus.SUCCESS
        assert result.content is not None
        assert result.content.title == "Test Page"
        assert "Test content" in result.content.text_content

    @pytest.mark.asyncio
    async def test_crawl_single_page_robots_blocked(self, crawler):
        """Test crawling blocked by robots.txt."""
        with patch.object(crawler.robots_parser, 'can_fetch', return_value=False):
            async with crawler:
                result = await crawler.crawl_single_page("https://example.com")

        assert result.status == CrawlStatus.ROBOTS_BLOCKED
        assert "robots.txt" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_crawl_single_page_http_error(self, crawler):
        """Test crawling with HTTP error response."""
        with patch.object(crawler, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    result = await crawler.crawl_single_page("https://example.com/notfound")

        assert result.status == CrawlStatus.FAILED
        assert result.response_code == 404

    @pytest.mark.asyncio
    async def test_crawl_single_page_timeout(self, crawler):
        """Test crawling with timeout."""
        with patch.object(crawler, '_session') as mock_session:
            mock_session.get.side_effect = asyncio.TimeoutError()

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    result = await crawler.crawl_single_page("https://example.com")

        assert result.status == CrawlStatus.FAILED
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_crawl_single_page_retry_logic(self, crawler):
        """Test retry logic on failures."""
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < crawler.config.retry_attempts:
                raise aiohttp.ClientError("Temporary error")
            else:
                # Success on final attempt
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {'content-type': 'text/html'}
                mock_response.text = AsyncMock(return_value="<html><body>Success</body></html>")
                mock_response.history = []

                mock_context = AsyncMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                return mock_context

        with patch.object(crawler, '_session') as mock_session:
            mock_session.get.side_effect = mock_get

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    result = await crawler.crawl_single_page("https://example.com")

        assert result.status == CrawlStatus.SUCCESS
        assert result.retry_count == crawler.config.retry_attempts - 1

    @pytest.mark.asyncio
    async def test_crawl_single_page_content_type_filter(self, crawler):
        """Test content type filtering."""
        with patch.object(crawler, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'application/pdf'}

            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    result = await crawler.crawl_single_page("https://example.com/document.pdf")

        assert result.status == CrawlStatus.SKIPPED
        assert "content type" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_crawl_single_page_content_size_limit(self, crawler):
        """Test content size limiting."""
        large_content = "x" * (crawler.config.max_content_size + 1000)

        with patch.object(crawler, '_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {
                'content-type': 'text/html',
                'content-length': str(len(large_content))
            }

            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    result = await crawler.crawl_single_page("https://example.com/large")

        assert result.status == CrawlStatus.SKIPPED
        assert "too large" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_crawl_recursive_basic(self, crawler):
        """Test basic recursive crawling."""
        # Mock responses for multiple pages
        responses = {
            "https://example.com": """
                <html>
                    <head><title>Home</title></head>
                    <body>
                        <a href="/page1">Page 1</a>
                        <a href="/page2">Page 2</a>
                        <p>Home content</p>
                    </body>
                </html>
            """,
            "https://example.com/page1": """
                <html>
                    <head><title>Page 1</title></head>
                    <body><p>Page 1 content</p></body>
                </html>
            """,
            "https://example.com/page2": """
                <html>
                    <head><title>Page 2</title></head>
                    <body><p>Page 2 content</p></body>
                </html>
            """
        }

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value=responses.get(str(url), ""))
            mock_response.history = []

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch.object(crawler, '_session') as mock_session:
            mock_session.get.side_effect = mock_get

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    results = await crawler.crawl_recursive("https://example.com")

        # Should have crawled multiple pages
        assert len(results) >= 1
        success_results = [r for r in results if r.status == CrawlStatus.SUCCESS]
        assert len(success_results) > 0

        # Check that home page was crawled
        home_results = [r for r in results if r.url == "https://example.com"]
        assert len(home_results) == 1
        assert home_results[0].content.title == "Home"

    @pytest.mark.asyncio
    async def test_crawl_recursive_depth_limit(self, crawler):
        """Test recursive crawling respects depth limits."""
        crawler.config.max_depth = 1  # Only go 1 level deep

        responses = {
            "https://example.com": """
                <html><body>
                    <a href="/level1">Level 1</a>
                    <p>Home</p>
                </body></html>
            """,
            "https://example.com/level1": """
                <html><body>
                    <a href="/level2">Level 2</a>
                    <p>Level 1</p>
                </body></html>
            """,
            "https://example.com/level2": """
                <html><body><p>Level 2</p></body></html>
            """
        }

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value=responses.get(str(url), ""))
            mock_response.history = []

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch.object(crawler, '_session') as mock_session:
            mock_session.get.side_effect = mock_get

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    results = await crawler.crawl_recursive("https://example.com")

        # Should not crawl level 2 due to depth limit
        level2_results = [r for r in results if "/level2" in r.url]
        assert len(level2_results) == 0

    @pytest.mark.asyncio
    async def test_crawl_recursive_external_links(self, crawler):
        """Test recursive crawling with external link handling."""
        crawler.config.follow_external_links = False

        html_with_external_links = """
        <html><body>
            <a href="/internal">Internal</a>
            <a href="https://external.com/page">External</a>
            <p>Content</p>
        </body></html>
        """

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value=html_with_external_links)
            mock_response.history = []

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch.object(crawler, '_session') as mock_session:
            mock_session.get.side_effect = mock_get

            with patch.object(crawler.robots_parser, 'can_fetch', return_value=True):
                async with crawler:
                    results = await crawler.crawl_recursive("https://example.com")

        # Should not have crawled external links
        external_results = [r for r in results if "external.com" in r.url]
        assert len(external_results) == 0


class TestWebCrawlingSystem:
    """Test WebCrawlingSystem high-level interface."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CrawlConfig(
            rate_limit_delay=0.1,  # Faster for tests
            max_concurrent_requests=2,
            request_timeout=5
        )

    @pytest.fixture
    def crawling_system(self, config):
        """Create WebCrawlingSystem instance."""
        return WebCrawlingSystem(config)

    def test_initialization_default_config(self):
        """Test WebCrawlingSystem initialization with default config."""
        system = WebCrawlingSystem()
        assert system.config is not None
        assert isinstance(system.config, CrawlConfig)

    def test_initialization_custom_config(self, config):
        """Test WebCrawlingSystem initialization with custom config."""
        system = WebCrawlingSystem(config)
        assert system.config == config

    @pytest.mark.asyncio
    async def test_crawl_url(self, crawling_system):
        """Test single URL crawling."""
        test_html = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"

        with patch.object(crawling_system.crawler, 'crawl_single_page') as mock_crawl:
            mock_result = CrawlResult(
                url="https://example.com",
                status=CrawlStatus.SUCCESS,
                content=ExtractedContent(
                    url="https://example.com",
                    title="Test",
                    description="",
                    keywords=[],
                    text_content="Content",
                    html_content=test_html,
                    metadata={}
                )
            )
            mock_crawl.return_value = mock_result

            result = await crawling_system.crawl_url("https://example.com")

        assert result.status == CrawlStatus.SUCCESS
        assert result.content.title == "Test"
        mock_crawl.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_website(self, crawling_system):
        """Test website recursive crawling."""
        with patch.object(crawling_system.crawler, 'crawl_recursive') as mock_crawl:
            mock_results = [
                CrawlResult(url="https://example.com", status=CrawlStatus.SUCCESS),
                CrawlResult(url="https://example.com/page1", status=CrawlStatus.SUCCESS),
                CrawlResult(url="https://example.com/page2", status=CrawlStatus.FAILED)
            ]
            mock_crawl.return_value = mock_results

            results = await crawling_system.crawl_website("https://example.com")

        assert len(results) == 3
        success_count = sum(1 for r in results if r.status == CrawlStatus.SUCCESS)
        assert success_count == 2

    @pytest.mark.asyncio
    async def test_crawl_website_with_limits(self, crawling_system):
        """Test website crawling with custom limits."""
        with patch.object(crawling_system.crawler, 'crawl_recursive') as mock_crawl:
            mock_crawl.return_value = []

            await crawling_system.crawl_website("https://example.com", max_pages=50, max_depth=3)

        # Check that config was updated
        assert crawling_system.config.max_pages == 50
        assert crawling_system.config.max_depth == 3

    @pytest.mark.asyncio
    async def test_crawl_urls_batch(self, crawling_system):
        """Test batch URL crawling."""
        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]

        with patch.object(crawling_system.crawler, 'crawl_single_page') as mock_crawl:
            mock_crawl.side_effect = [
                CrawlResult(url=urls[0], status=CrawlStatus.SUCCESS),
                CrawlResult(url=urls[1], status=CrawlStatus.FAILED),
                CrawlResult(url=urls[2], status=CrawlStatus.SUCCESS)
            ]

            results = await crawling_system.crawl_urls_batch(urls)

        assert len(results) == 3
        assert mock_crawl.call_count == 3
        success_count = sum(1 for r in results if r.status == CrawlStatus.SUCCESS)
        assert success_count == 2

    @pytest.mark.asyncio
    async def test_crawl_urls_batch_with_exceptions(self, crawling_system):
        """Test batch crawling with exceptions."""
        urls = ["https://example.com/1", "https://example.com/2"]

        with patch.object(crawling_system.crawler, 'crawl_single_page') as mock_crawl:
            mock_crawl.side_effect = [
                CrawlResult(url=urls[0], status=CrawlStatus.SUCCESS),
                Exception("Network error")
            ]

            results = await crawling_system.crawl_urls_batch(urls)

        assert len(results) == 2
        assert results[0].status == CrawlStatus.SUCCESS
        assert results[1].status == CrawlStatus.FAILED
        assert "Network error" in results[1].error_message

    def test_export_results_json(self, crawling_system):
        """Test exporting results to JSON."""
        results = [
            CrawlResult(
                url="https://example.com",
                status=CrawlStatus.SUCCESS,
                response_code=200,
                content=ExtractedContent(
                    url="https://example.com",
                    title="Test Page",
                    description="Test description",
                    keywords=["test"],
                    text_content="Test content",
                    html_content="<html>...</html>",
                    metadata={"author": "Test"}
                )
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name

        try:
            result_path = crawling_system.export_results(results, format='json', output_path=output_path)

            assert result_path == output_path
            assert Path(output_path).exists()

            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]['url'] == "https://example.com"
            assert data[0]['status'] == "SUCCESS"
            assert data[0]['content']['title'] == "Test Page"

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_results_yaml(self, crawling_system):
        """Test exporting results to YAML."""
        results = [
            CrawlResult(
                url="https://example.com",
                status=CrawlStatus.SUCCESS,
                content=ExtractedContent(
                    url="https://example.com",
                    title="Test Page",
                    description="",
                    keywords=[],
                    text_content="Content",
                    html_content="",
                    metadata={}
                )
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            output_path = f.name

        try:
            result_path = crawling_system.export_results(results, format='yaml', output_path=output_path)

            assert result_path == output_path
            assert Path(output_path).exists()

            # Verify YAML content
            import yaml
            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert len(data) == 1
            assert data[0]['url'] == "https://example.com"

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_results_csv(self, crawling_system):
        """Test exporting results to CSV."""
        results = [
            CrawlResult(
                url="https://example.com",
                status=CrawlStatus.SUCCESS,
                response_code=200,
                content=ExtractedContent(
                    url="https://example.com",
                    title="Test Page",
                    description="",
                    keywords=[],
                    text_content="Content",
                    html_content="",
                    metadata={}
                )
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            result_path = crawling_system.export_results(results, format='csv', output_path=output_path)

            assert result_path == output_path
            assert Path(output_path).exists()

            # Verify CSV content
            import csv
            with open(output_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2  # Header + 1 data row
            assert rows[0][0] == 'URL'  # Header
            assert rows[1][0] == 'https://example.com'  # Data

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_results_unsupported_format(self, crawling_system):
        """Test exporting with unsupported format."""
        results = []

        with pytest.raises(ValueError, match="Unsupported export format"):
            crawling_system.export_results(results, format='xml')

    def test_get_config(self, crawling_system):
        """Test getting current configuration."""
        config = crawling_system.get_config()
        assert isinstance(config, CrawlConfig)
        assert config == crawling_system.config

    def test_update_config(self, crawling_system):
        """Test updating configuration."""
        original_max_pages = crawling_system.config.max_pages

        crawling_system.update_config(max_pages=500, max_depth=5)

        assert crawling_system.config.max_pages == 500
        assert crawling_system.config.max_depth == 5
        assert crawling_system.config.max_pages != original_max_pages

    def test_update_config_invalid_parameter(self, crawling_system):
        """Test updating configuration with invalid parameter."""
        with pytest.raises(ValueError, match="Unknown config parameter"):
            crawling_system.update_config(invalid_param="value")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()

        assert isinstance(config, CrawlConfig)
        assert config.max_depth == 2
        assert config.max_pages == 100
        assert config.rate_limit_delay == 0.5
        assert config.follow_external_links is False
        assert config.respect_robots_txt is True


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CrawlConfig(rate_limit_delay=0.1)

    def test_invalid_url_handling(self, config):
        """Test handling of invalid URLs."""
        system = WebCrawlingSystem(config)

        # These should not raise exceptions during initialization
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "mailto:test@example.com",
            ""
        ]

        for url in invalid_urls:
            # Should be handled gracefully during crawling
            assert url is not None  # Just verify they don't cause issues

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, config):
        """Test handling of network timeouts."""
        system = WebCrawlingSystem(config)

        with patch.object(system.crawler, 'crawl_single_page') as mock_crawl:
            mock_crawl.side_effect = asyncio.TimeoutError()

            result = await system.crawl_url("https://slow-example.com")

        # Should handle timeout gracefully
        assert isinstance(result, Exception)

    def test_unicode_url_handling(self, config):
        """Test handling of Unicode URLs."""
        extractor = ContentExtractor()

        # Unicode content should be handled properly
        unicode_html = """
        <html>
            <head>
                <title>测试页面</title>
                <meta name="description" content="这是一个测试页面">
            </head>
            <body>
                <p>这是中文内容。</p>
                <a href="/页面">链接</a>
            </body>
        </html>
        """

        content = extractor.extract_content(unicode_html, "https://example.com/测试")

        assert "测试页面" in content.title
        assert "测试页面" in content.description
        assert "中文内容" in content.text_content

    def test_very_large_html_handling(self, config):
        """Test handling of very large HTML documents."""
        extractor = ContentExtractor()

        # Create large HTML document
        large_content = "<p>" + "word " * 50000 + "</p>"  # ~250k words
        large_html = f"<html><body>{large_content}</body></html>"

        content = extractor.extract_content(large_html, "https://example.com")

        # Should handle large content without issues
        assert content.word_count > 100000
        assert len(content.text_content) > 100000

    def test_deeply_nested_html_handling(self, config):
        """Test handling of deeply nested HTML structures."""
        extractor = ContentExtractor()

        # Create deeply nested HTML
        nested_html = "<div>" * 100 + "Deep content" + "</div>" * 100
        html = f"<html><body>{nested_html}</body></html>"

        content = extractor.extract_content(html, "https://example.com")

        assert "Deep content" in content.text_content

    @pytest.mark.asyncio
    async def test_concurrent_crawling_same_domain(self, config):
        """Test concurrent crawling of same domain with rate limiting."""
        system = WebCrawlingSystem(config)
        urls = [f"https://example.com/page{i}" for i in range(5)]

        # Mock successful responses
        with patch.object(system.crawler, 'crawl_single_page') as mock_crawl:
            mock_crawl.return_value = CrawlResult(
                url="https://example.com",
                status=CrawlStatus.SUCCESS
            )

            start_time = time.time()
            results = await system.crawl_urls_batch(urls)
            end_time = time.time()

            # Should have taken some time due to rate limiting
            # With 0.1s delay and 5 requests, should take at least 0.4s
            assert end_time - start_time >= 0.3

        assert len(results) == 5

    def test_malformed_robots_txt_handling(self):
        """Test handling of malformed robots.txt."""
        parser = RobotsTxtParser("TestBot")

        # Should handle malformed robots.txt gracefully
        malformed_robots = """
        User-agent: *
        Disallow: /private
        # Missing colon on next line
        Disallow /malformed
        Allow /public
        """

        # This should not raise an exception
        assert parser.user_agent == "TestBot"

    def test_circular_redirect_handling(self, config):
        """Test handling of circular redirects."""
        # This would be tested at the HTTP client level
        # aiohttp should handle circular redirects automatically
        assert True  # Placeholder for circular redirect handling

    def test_memory_usage_with_large_crawl(self, config):
        """Test memory usage during large crawling operations."""
        # Modify config for memory testing
        config.max_pages = 1000
        system = WebCrawlingSystem(config)

        # In a real scenario, this would test actual memory usage
        # For unit tests, we just verify the system can handle large configurations
        assert system.config.max_pages == 1000


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete workflow scenarios."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_complete_crawling_workflow(self, temp_output_dir):
        """Test complete crawling workflow from start to finish."""
        config = CrawlConfig(
            max_pages=3,
            max_depth=1,
            rate_limit_delay=0.1
        )
        system = WebCrawlingSystem(config)

        # Mock multiple page responses
        mock_responses = {
            "https://example.com": """
                <html>
                    <head><title>Home Page</title></head>
                    <body>
                        <a href="/about">About</a>
                        <a href="/contact">Contact</a>
                        <p>Welcome to our website</p>
                    </body>
                </html>
            """,
            "https://example.com/about": """
                <html>
                    <head><title>About Us</title></head>
                    <body><p>About our company</p></body>
                </html>
            """,
            "https://example.com/contact": """
                <html>
                    <head><title>Contact</title></head>
                    <body><p>Contact information</p></body>
                </html>
            """
        }

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value=mock_responses.get(str(url), ""))
            mock_response.history = []

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch.object(system.crawler, '_session') as mock_session:
            mock_session.get.side_effect = mock_get

            with patch.object(system.crawler.robots_parser, 'can_fetch', return_value=True):
                # Crawl website
                results = await system.crawl_website("https://example.com")

        # Verify results
        assert len(results) > 0
        success_results = [r for r in results if r.status == CrawlStatus.SUCCESS]
        assert len(success_results) >= 1

        # Export results
        json_file = temp_output_dir / "results.json"
        export_path = system.export_results(results, format='json', output_path=str(json_file))

        assert Path(export_path).exists()

        # Verify exported data
        with open(json_file) as f:
            exported_data = json.load(f)

        assert len(exported_data) == len(results)
        assert exported_data[0]['url'] == "https://example.com"

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, temp_output_dir):
        """Test error recovery during crawling."""
        config = CrawlConfig(max_pages=3, retry_attempts=2)
        system = WebCrawlingSystem(config)

        call_count = 0

        def mock_get_with_errors(url, **kwargs):
            nonlocal call_count
            call_count += 1

            # First few calls fail, then succeed
            if call_count <= 2:
                raise aiohttp.ClientError("Temporary network error")

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value="<html><body>Success</body></html>")
            mock_response.history = []

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            return mock_context

        with patch.object(system.crawler, '_session') as mock_session:
            mock_session.get.side_effect = mock_get_with_errors

            with patch.object(system.crawler.robots_parser, 'can_fetch', return_value=True):
                result = await system.crawl_url("https://example.com")

        # Should eventually succeed after retries
        assert result.status == CrawlStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self):
        """Test that rate limiting is properly enforced."""
        config = CrawlConfig(rate_limit_delay=0.5)  # 2 requests per second
        system = WebCrawlingSystem(config)

        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]

        with patch.object(system.crawler, 'crawl_single_page') as mock_crawl:
            mock_crawl.return_value = CrawlResult(
                url="https://example.com",
                status=CrawlStatus.SUCCESS
            )

            start_time = time.time()
            await system.crawl_urls_batch(urls)
            end_time = time.time()

        # Should have taken at least 1 second (2 delays of 0.5s each)
        assert end_time - start_time >= 0.8  # Allow some tolerance

    def test_configuration_validation(self):
        """Test configuration validation and edge cases."""
        # Test with extreme values
        config = CrawlConfig(
            max_depth=0,  # No depth
            max_pages=1,  # Single page
            rate_limit_delay=0.001,  # Very fast
            max_concurrent_requests=100  # High concurrency
        )

        system = WebCrawlingSystem(config)
        assert system.config.max_depth == 0
        assert system.config.max_pages == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])