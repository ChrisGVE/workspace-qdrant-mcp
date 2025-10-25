"""
Comprehensive unit tests for LinkDiscovery with edge cases and error conditions.

This test suite covers:
- URL normalization and canonicalization
- Link discovery from multiple sources (anchors, sitemaps, structured data)
- Recursive crawling with breadth-first and depth-first strategies
- Cycle detection and infinite loop prevention
- Domain filtering and URL pattern matching
- Link quality assessment and priority calculation
- Batch processing and queue management
- Error handling and edge cases
- Performance with large link sets
- Malformed HTML and broken link handling
"""

import asyncio
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup
from common.core.link_discovery import (
    DiscoveredLink,
    LinkDiscovery,
    LinkDiscoveryConfig,
    LinkExtractor,
    URLNormalizer,
)


class TestLinkDiscoveryConfig:
    """Test LinkDiscoveryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LinkDiscoveryConfig()

        assert config.max_depth == 3
        assert config.max_pages == 1000
        assert config.max_links_per_page == 100
        assert config.same_domain_only is True
        assert config.allowed_schemes == {'http', 'https'}
        assert config.discover_from_anchors is True
        assert config.discover_from_sitemaps is True
        assert config.discover_from_structured_data is True
        assert config.strategy == "breadth_first"
        assert config.batch_size == 10
        assert config.url_normalization is True

    def test_custom_config(self):
        """Test custom configuration values."""
        allowed_domains = {'example.com', 'test.com'}
        blocked_domains = {'spam.com'}
        blocked_patterns = [r'\.pdf$', r'/admin/']
        allowed_patterns = [r'/blog/']

        config = LinkDiscoveryConfig(
            max_depth=5,
            max_pages=500,
            same_domain_only=False,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            blocked_patterns=blocked_patterns,
            allowed_patterns=allowed_patterns,
            strategy="depth_first",
            batch_size=20
        )

        assert config.max_depth == 5
        assert config.max_pages == 500
        assert config.same_domain_only is False
        assert config.allowed_domains == allowed_domains
        assert config.blocked_domains == blocked_domains
        assert config.blocked_patterns == blocked_patterns
        assert config.allowed_patterns == allowed_patterns
        assert config.strategy == "depth_first"
        assert config.batch_size == 20


class TestDiscoveredLink:
    """Test DiscoveredLink dataclass."""

    def test_link_creation(self):
        """Test creating DiscoveredLink instances."""
        link = DiscoveredLink(
            url="http://example.com",
            source_url="http://source.com"
        )

        assert link.url == "http://example.com"
        assert link.source_url == "http://source.com"
        assert link.anchor_text is None
        assert link.depth == 0
        assert link.discovery_method == "html_anchor"
        assert link.is_internal is True
        assert link.crawl_priority == 0.5
        assert link.crawled is False
        assert isinstance(link.metadata, dict)

    def test_link_with_full_data(self):
        """Test DiscoveredLink with all fields populated."""
        discovery_time = datetime.now()
        metadata = {'custom': 'data'}

        link = DiscoveredLink(
            url="http://example.com/page",
            source_url="http://example.com",
            anchor_text="Test Link",
            title="Test Title",
            depth=2,
            discovery_method="json_ld",
            context="Context around link",
            position=5,
            is_internal=False,
            is_canonical=True,
            is_navigation=True,
            discovered_at=discovery_time,
            crawl_priority=0.8,
            crawled=True,
            crawl_attempted=True,
            metadata=metadata
        )

        assert link.anchor_text == "Test Link"
        assert link.title == "Test Title"
        assert link.depth == 2
        assert link.discovery_method == "json_ld"
        assert link.context == "Context around link"
        assert link.is_canonical is True
        assert link.crawl_priority == 0.8
        assert link.crawled is True
        assert link.metadata == metadata


class TestURLNormalizer:
    """Test URLNormalizer functionality."""

    @pytest.fixture
    def normalizer(self):
        """Create URLNormalizer instance for testing."""
        config = LinkDiscoveryConfig()
        return URLNormalizer(config)

    def test_normalize_absolute_url(self, normalizer):
        """Test normalization of absolute URLs."""
        url = "HTTP://Example.COM/Path/"
        normalized = normalizer.normalize_url(url)

        assert normalized == "http://example.com/Path"

    def test_normalize_relative_url(self, normalizer):
        """Test normalization of relative URLs."""
        url = "../relative/path"
        base_url = "http://example.com/base/page"
        normalized = normalizer.normalize_url(url, base_url)

        assert "example.com" in normalized
        assert "relative/path" in normalized

    def test_normalize_removes_default_ports(self, normalizer):
        """Test normalization removes default ports."""
        http_url = "http://example.com:80/path"
        https_url = "https://example.com:443/path"

        assert normalizer.normalize_url(http_url) == "http://example.com/path"
        assert normalizer.normalize_url(https_url) == "https://example.com/path"

    def test_normalize_removes_trailing_slash(self, normalizer):
        """Test normalization removes trailing slash from non-root paths."""
        url = "http://example.com/path/"
        root_url = "http://example.com/"

        assert normalizer.normalize_url(url) == "http://example.com/path"
        assert normalizer.normalize_url(root_url) == "http://example.com/"

    def test_normalize_query_params(self, normalizer):
        """Test normalization of query parameters."""
        url = "http://example.com/path?utm_source=test&param=value&utm_campaign=camp"
        normalized = normalizer.normalize_url(url)

        # Should remove UTM parameters but keep regular parameters
        assert "utm_source" not in normalized
        assert "utm_campaign" not in normalized
        assert "param=value" in normalized

    def test_normalize_empty_query_params(self, normalizer):
        """Test normalization with only tracking parameters."""
        url = "http://example.com/path?utm_source=test&fbclid=abc123"
        normalized = normalizer.normalize_url(url)

        # Should remove query string entirely when only tracking params
        assert "?" not in normalized
        assert normalized == "http://example.com/path"

    def test_normalize_handles_malformed_urls(self, normalizer):
        """Test normalization handles malformed URLs gracefully."""
        malformed_urls = [
            "not-a-url",
            "http://",
            "://example.com",
            "",
            None
        ]

        for url in malformed_urls:
            # Should either normalize or return original without crashing
            result = normalizer.normalize_url(url)
            assert isinstance(result, str)

    def test_is_valid_url_scheme_checking(self, normalizer):
        """Test URL validation checks allowed schemes."""
        valid_urls = [
            "http://example.com",
            "https://example.com"
        ]
        invalid_urls = [
            "ftp://example.com",
            "file:///path",
            "mailto:test@example.com"
        ]

        for url in valid_urls:
            assert normalizer.is_valid_url(url) is True

        for url in invalid_urls:
            assert normalizer.is_valid_url(url) is False

    def test_is_valid_url_same_domain_restriction(self, normalizer):
        """Test URL validation with same domain restriction."""
        base_domain = "example.com"

        same_domain_urls = [
            "http://example.com/page",
            "https://example.com/other"
        ]
        different_domain_urls = [
            "http://other.com/page",
            "https://different.org/page"
        ]

        for url in same_domain_urls:
            assert normalizer.is_valid_url(url, base_domain) is True

        for url in different_domain_urls:
            assert normalizer.is_valid_url(url, base_domain) is False

    def test_is_valid_url_domain_restrictions(self):
        """Test URL validation with allowed/blocked domains."""
        config = LinkDiscoveryConfig(
            allowed_domains={'allowed.com'},
            blocked_domains={'blocked.com'},
            same_domain_only=False
        )
        normalizer = URLNormalizer(config)

        assert normalizer.is_valid_url("http://allowed.com/page") is True
        assert normalizer.is_valid_url("http://blocked.com/page") is False
        assert normalizer.is_valid_url("http://other.com/page") is False

    def test_is_valid_url_pattern_filtering(self):
        """Test URL validation with pattern filtering."""
        config = LinkDiscoveryConfig(
            blocked_patterns=[r'\.pdf$', r'/admin/'],
            allowed_patterns=[r'/blog/']
        )
        normalizer = URLNormalizer(config)

        # Should block PDF files
        assert normalizer.is_valid_url("http://example.com/file.pdf") is False

        # Should block admin paths
        assert normalizer.is_valid_url("http://example.com/admin/panel") is False

        # Should only allow blog paths when allowed_patterns is set
        assert normalizer.is_valid_url("http://example.com/blog/post") is True
        assert normalizer.is_valid_url("http://example.com/news/article") is False


class TestLinkExtractor:
    """Test LinkExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create LinkExtractor instance for testing."""
        config = LinkDiscoveryConfig()
        return LinkExtractor(config)

    def test_extract_anchor_links_basic(self, extractor):
        """Test basic anchor link extraction."""
        html = """
        <html>
            <body>
                <a href="http://example.com/page1">Link 1</a>
                <a href="/page2">Link 2</a>
                <a href="../page3">Link 3</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        assert len(links) == 3
        assert all(isinstance(link, DiscoveredLink) for link in links)
        assert "page1" in links[0].url
        assert "page2" in links[1].url
        assert "page3" in links[2].url

    def test_extract_anchor_links_with_metadata(self, extractor):
        """Test anchor link extraction includes metadata."""
        html = """
        <html>
            <body>
                <a href="/page" title="Page Title">Link Text</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        assert len(links) == 1
        link = links[0]
        assert link.anchor_text == "Link Text"
        assert link.title == "Page Title"
        assert link.discovery_method == "html_anchor"

    def test_extract_anchor_links_filters_empty_text(self, extractor):
        """Test anchor link extraction filters empty anchor text."""
        html = """
        <html>
            <body>
                <a href="/page1">Good Link</a>
                <a href="/page2"></a>
                <a href="/page3">   </a>
                <a href="/page4">Another Good Link</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        # Should only get links with non-empty anchor text
        assert len(links) == 2
        assert all(link.anchor_text.strip() for link in links)

    def test_extract_anchor_links_skips_image_only(self, extractor):
        """Test anchor link extraction skips image-only links."""
        html = """
        <html>
            <body>
                <a href="/page1">Text Link</a>
                <a href="/page2"><img src="image.jpg" alt="Image"></a>
                <a href="/page3"><img src="image2.jpg">With Text</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        # Should skip image-only link but include text+image link
        assert len(links) == 2
        link_texts = [link.anchor_text for link in links]
        assert "Text Link" in link_texts
        assert "With Text" in link_texts

    def test_extract_anchor_links_limits_length(self, extractor):
        """Test anchor link extraction limits text length."""
        long_text = "A" * 300
        html = f"""
        <html>
            <body>
                <a href="/page">{long_text}</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        assert len(links) == 1
        assert len(links[0].anchor_text) <= extractor.config.max_anchor_text_length

    def test_extract_canonical_links(self, extractor):
        """Test canonical link extraction."""
        html = """
        <html>
            <head>
                <link rel="canonical" href="http://example.com/canonical">
            </head>
            <body></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_canonical_links(soup, "http://example.com")

        assert len(links) == 1
        link = links[0]
        assert link.url == "http://example.com/canonical"
        assert link.is_canonical is True
        assert link.discovery_method == "canonical_link"
        assert link.crawl_priority == 0.9

    def test_extract_structured_data_links_json_ld(self, extractor):
        """Test structured data link extraction from JSON-LD."""
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "http://schema.org",
                    "@type": "Article",
                    "url": "http://example.com/article",
                    "sameAs": ["http://example.com/same1", "http://example.com/same2"]
                }
                </script>
            </head>
            <body></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_structured_data_links(soup, "http://example.com")

        assert len(links) >= 1  # At least the main URL
        methods = [link.discovery_method for link in links]
        assert "json_ld" in methods

    def test_extract_structured_data_links_open_graph(self, extractor):
        """Test structured data link extraction from Open Graph."""
        html = """
        <html>
            <head>
                <meta property="og:url" content="http://example.com/og-url">
            </head>
            <body></body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_structured_data_links(soup, "http://example.com")

        assert len(links) == 1
        link = links[0]
        assert link.url == "http://example.com/og-url"
        assert link.discovery_method == "open_graph"

    def test_extract_sitemap_links(self, extractor):
        """Test sitemap link extraction from robots.txt."""
        robots_content = """
        User-agent: *
        Disallow: /admin/
        Sitemap: http://example.com/sitemap.xml
        Sitemap: http://example.com/news-sitemap.xml
        """
        links = extractor.extract_sitemap_links(robots_content, "http://example.com")

        assert len(links) == 2
        sitemap_urls = [link.url for link in links]
        assert "http://example.com/sitemap.xml" in sitemap_urls
        assert "http://example.com/news-sitemap.xml" in sitemap_urls

    def test_link_priority_calculation(self, extractor):
        """Test link priority calculation."""
        # Internal link in main content should have high priority
        html = """
        <html>
            <body>
                <article>
                    <a href="/internal">Internal Content Link</a>
                </article>
                <nav>
                    <a href="/nav">Navigation Link</a>
                </nav>
                <a href="http://external.com">External Link</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        # Find each link
        internal_link = next((l for l in links if "internal" in l.url), None)
        nav_link = next((l for l in links if "nav" in l.url), None)
        external_link = next((l for l in links if "external" in l.url), None)

        assert internal_link is not None
        assert nav_link is not None
        assert external_link is not None

        # Internal content link should have higher priority than navigation
        assert internal_link.crawl_priority > nav_link.crawl_priority

    def test_context_extraction(self, extractor):
        """Test extraction of link context."""
        html = """
        <html>
            <body>
                <p>This is some context before the <a href="/link">link text</a> and some context after.</p>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        assert len(links) == 1
        context = links[0].context
        assert "context before" in context
        assert "context after" in context
        assert "link text" in context

    def test_navigation_detection(self, extractor):
        """Test detection of navigation links."""
        html = """
        <html>
            <body>
                <nav>
                    <a href="/home">Home</a>
                </nav>
                <header>
                    <a href="/about">About</a>
                </header>
                <article>
                    <a href="/content">Content Link</a>
                </article>
                <div class="navigation">
                    <a href="/nav-item">Nav Item</a>
                </div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = extractor.extract_anchor_links(soup, "http://example.com")

        # Check navigation detection
        nav_links = [link for link in links if link.is_navigation]
        content_links = [link for link in links if not link.is_navigation]

        assert len(nav_links) >= 3  # nav, header, and classed div links
        assert len(content_links) >= 1  # article link


class TestLinkDiscovery:
    """Test LinkDiscovery main functionality."""

    @pytest.fixture
    def discovery(self):
        """Create LinkDiscovery instance for testing."""
        config = LinkDiscoveryConfig(max_depth=2, max_pages=10)
        return LinkDiscovery(config)

    @pytest.mark.asyncio
    async def test_discover_links_basic(self, discovery):
        """Test basic link discovery."""
        html = """
        <html>
            <head>
                <title>Test Page</title>
            </head>
            <body>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="http://external.com">External</a>
            </body>
        </html>
        """

        links = await discovery.discover_links(html, "http://example.com", 0)

        assert len(links) >= 2  # Internal links should be included
        assert all(link.depth == 1 for link in links)

    @pytest.mark.asyncio
    async def test_discover_links_respects_depth_limit(self, discovery):
        """Test that link discovery respects depth limits."""
        html = "<html><body><a href='/page'>Link</a></body></html>"

        # At max depth, should return no links
        links = await discovery.discover_links(html, "http://example.com", discovery.config.max_depth)
        assert len(links) == 0

        # Below max depth, should return links
        links = await discovery.discover_links(html, "http://example.com", discovery.config.max_depth - 1)
        assert len(links) > 0

    @pytest.mark.asyncio
    async def test_discover_links_deduplication(self, discovery):
        """Test link discovery deduplicates URLs."""
        html = """
        <html>
            <body>
                <a href="/page">Link 1</a>
                <a href="/page">Link 2</a>
                <a href="/page">Link 3</a>
            </body>
        </html>
        """

        links = await discovery.discover_links(html, "http://example.com", 0)

        # Should only have one link despite multiple anchors
        assert len(links) == 1

    @pytest.mark.asyncio
    async def test_discover_links_respects_page_limit(self, discovery):
        """Test link discovery respects max links per page."""
        # Create HTML with many links
        link_tags = ''.join(f'<a href="/page{i}">Page {i}</a>' for i in range(150))
        html = f"<html><body>{link_tags}</body></html>"

        links = await discovery.discover_links(html, "http://example.com", 0)

        # Should not exceed configured limit
        assert len(links) <= discovery.config.max_links_per_page

    @pytest.mark.asyncio
    async def test_discover_links_updates_statistics(self, discovery):
        """Test link discovery updates statistics."""
        html = """
        <html>
            <head>
                <link rel="canonical" href="/canonical">
            </head>
            <body>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
            </body>
        </html>
        """

        initial_count = discovery._stats['urls_discovered']
        await discovery.discover_links(html, "http://example.com", 0)

        assert discovery._stats['urls_discovered'] > initial_count
        assert len(discovery._discovered_urls) > 0

    @pytest.mark.asyncio
    async def test_crawl_recursive_breadth_first(self, discovery):
        """Test recursive crawling with breadth-first strategy."""
        discovery.config.strategy = "breadth_first"
        start_urls = ["http://example.com/start"]

        batches = []
        async for batch in discovery.crawl_recursive(start_urls, max_pages=5):
            batches.append(batch)
            # Stop after a few batches for testing
            if len(batches) >= 2:
                break

        assert len(batches) >= 1
        assert all(isinstance(batch, list) for batch in batches)

    @pytest.mark.asyncio
    async def test_crawl_recursive_depth_first(self, discovery):
        """Test recursive crawling with depth-first strategy."""
        discovery.config.strategy = "depth_first"
        start_urls = ["http://example.com/start"]

        batches = []
        async for batch in discovery.crawl_recursive(start_urls, max_pages=5):
            batches.append(batch)
            if len(batches) >= 2:
                break

        assert len(batches) >= 1

    @pytest.mark.asyncio
    async def test_crawl_recursive_respects_max_pages(self, discovery):
        """Test recursive crawling respects max pages limit."""
        # Add many URLs to queue
        for i in range(20):
            link = DiscoveredLink(
                url=f"http://example.com/page{i}",
                source_url="http://example.com"
            )
            discovery._crawl_queue.append(link)
            discovery._discovered_urls.add(link.url)

        crawled_urls = []
        async for batch in discovery.crawl_recursive([], max_pages=5):
            crawled_urls.extend(batch)

        assert len(crawled_urls) <= 5

    @pytest.mark.asyncio
    async def test_crawl_recursive_batch_processing(self, discovery):
        """Test recursive crawling processes URLs in batches."""
        discovery.config.batch_size = 3

        # Add URLs to queue
        for i in range(10):
            link = DiscoveredLink(
                url=f"http://example.com/page{i}",
                source_url="http://example.com"
            )
            discovery._crawl_queue.append(link)
            discovery._discovered_urls.add(link.url)

        batches = []
        async for batch in discovery.crawl_recursive([]):
            batches.append(batch)

        # Should have multiple batches with correct size
        assert len(batches) > 1
        for batch in batches[:-1]:  # All but last batch should be full size
            assert len(batch) <= discovery.config.batch_size

    def test_add_discovered_links(self, discovery):
        """Test adding discovered links to queue."""
        links = [
            DiscoveredLink(url="http://example.com/page1", source_url="http://example.com"),
            DiscoveredLink(url="http://example.com/page2", source_url="http://example.com"),
        ]

        initial_queue_size = len(discovery._crawl_queue)
        discovery.add_discovered_links(links)

        assert len(discovery._crawl_queue) == initial_queue_size + 2
        assert len(discovery._discovered_urls) >= 2

    def test_add_discovered_links_avoids_duplicates(self, discovery):
        """Test adding discovered links avoids duplicates."""
        link = DiscoveredLink(url="http://example.com/page", source_url="http://example.com")

        # Add same link twice
        discovery.add_discovered_links([link])
        discovery.add_discovered_links([link])

        # Should only be added once
        assert len([l for l in discovery._crawl_queue if l.url == link.url]) == 1

    def test_mark_url_failed(self, discovery):
        """Test marking URLs as failed."""
        url = "http://example.com/failed"
        error = "404 Not Found"

        initial_failed_count = discovery._stats['urls_failed']
        discovery.mark_url_failed(url, error)

        assert url in discovery._failed_urls
        assert discovery._stats['urls_failed'] == initial_failed_count + 1

    def test_get_queue_status(self, discovery):
        """Test getting queue status."""
        # Add some links
        links = [
            DiscoveredLink(url="http://example.com/page1", source_url="http://example.com"),
            DiscoveredLink(url="http://example.com/page2", source_url="http://example.com"),
        ]
        discovery.add_discovered_links(links)

        # Mark one as failed
        discovery.mark_url_failed("http://example.com/failed", "Error")

        status = discovery.get_queue_status()

        assert 'queue_size' in status
        assert 'discovered_count' in status
        assert 'failed_count' in status
        assert 'stats' in status
        assert 'config' in status

        assert status['queue_size'] == len(discovery._crawl_queue)
        assert status['discovered_count'] == len(discovery._discovered_urls)
        assert status['failed_count'] == len(discovery._failed_urls)

    def test_reset(self, discovery):
        """Test resetting discovery state."""
        # Add some data
        links = [
            DiscoveredLink(url="http://example.com/page1", source_url="http://example.com"),
        ]
        discovery.add_discovered_links(links)
        discovery.mark_url_failed("http://example.com/failed", "Error")

        # Reset
        discovery.reset()

        assert len(discovery._discovered_urls) == 0
        assert len(discovery._crawled_urls) == 0
        assert len(discovery._failed_urls) == 0
        assert len(discovery._crawl_queue) == 0
        assert discovery._stats['urls_discovered'] == 0


class TestLinkDiscoveryEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_malformed_html_handling(self):
        """Test handling of malformed HTML."""
        discovery = LinkDiscovery()

        malformed_html = """
        <html>
            <body>
                <a href="/page1">Unclosed link
                <a href="/page2">Another link</a>
                <div>Unclosed div
                    <a href="/page3">Nested link</a>
                </body>
        """  # Intentionally malformed

        # Should not crash and should extract some links
        links = await discovery.discover_links(malformed_html, "http://example.com", 0)
        assert isinstance(links, list)

    @pytest.mark.asyncio
    async def test_empty_html_handling(self):
        """Test handling of empty HTML."""
        discovery = LinkDiscovery()

        empty_htmls = ["", "<html></html>", "<html><body></body></html>"]

        for html in empty_htmls:
            links = await discovery.discover_links(html, "http://example.com", 0)
            assert isinstance(links, list)
            assert len(links) == 0

    @pytest.mark.asyncio
    async def test_invalid_base_url_handling(self):
        """Test handling of invalid base URLs."""
        discovery = LinkDiscovery()

        html = '<html><body><a href="/page">Link</a></body></html>'
        invalid_base_urls = ["not-a-url", "", "://invalid"]

        for base_url in invalid_base_urls:
            # Should not crash
            links = await discovery.discover_links(html, base_url, 0)
            assert isinstance(links, list)

    @pytest.mark.asyncio
    async def test_very_large_html_handling(self):
        """Test handling of very large HTML documents."""
        discovery = LinkDiscovery()

        # Create large HTML with many links
        large_content = "Content " * 1000
        many_links = ''.join(f'<a href="/page{i}">Link {i}</a>' for i in range(500))
        large_html = f"""
        <html>
            <body>
                <p>{large_content}</p>
                {many_links}
            </body>
        </html>
        """

        links = await discovery.discover_links(large_html, "http://example.com", 0)

        # Should handle large documents without crashing
        assert isinstance(links, list)
        # Should respect max_links_per_page limit
        assert len(links) <= discovery.config.max_links_per_page

    @pytest.mark.asyncio
    async def test_deeply_nested_links(self):
        """Test handling of deeply nested link structures."""
        discovery = LinkDiscovery()

        # Create deeply nested HTML
        nested_start = "<div>" * 50
        nested_end = "</div>" * 50
        html = f"""
        <html>
            <body>
                {nested_start}
                <a href="/deep-link">Deep Link</a>
                {nested_end}
            </body>
        </html>
        """

        links = await discovery.discover_links(html, "http://example.com", 0)

        assert len(links) == 1
        assert "deep-link" in links[0].url

    def test_unicode_urls_handling(self):
        """Test handling of Unicode URLs."""
        discovery = LinkDiscovery()

        unicode_html = """
        <html>
            <body>
                <a href="/页面">Chinese URL</a>
                <a href="/página">Spanish URL</a>
                <a href="/日本語">Japanese URL</a>
            </body>
        </html>
        """

        # Should handle Unicode URLs without crashing
        # Note: This is an async test but we'll test the synchronous parts
        soup = BeautifulSoup(unicode_html, 'html.parser')
        extractor = LinkExtractor(discovery.config)

        # Should not crash when processing Unicode URLs
        try:
            links = extractor.extract_anchor_links(soup, "http://example.com")
            assert isinstance(links, list)
        except Exception as e:
            pytest.fail(f"Unicode URL handling failed: {e}")

    @pytest.mark.asyncio
    async def test_circular_reference_handling(self):
        """Test handling of circular references in crawling."""
        discovery = LinkDiscovery()

        # Simulate circular references by adding URLs that would link back
        link1 = DiscoveredLink(url="http://example.com/page1", source_url="http://example.com")
        link2 = DiscoveredLink(url="http://example.com/page2", source_url="http://example.com/page1")
        link3 = DiscoveredLink(url="http://example.com/page1", source_url="http://example.com/page2")

        discovery.add_discovered_links([link1])
        discovery.add_discovered_links([link2])
        discovery.add_discovered_links([link3])  # This creates circular reference

        # Should handle without infinite loops
        status = discovery.get_queue_status()
        assert status['discovered_count'] == 2  # Should deduplicate page1

    def test_extreme_configuration_values(self):
        """Test handling of extreme configuration values."""
        # Test with very small limits
        small_config = LinkDiscoveryConfig(
            max_depth=0,
            max_pages=0,
            max_links_per_page=0,
            batch_size=1
        )
        discovery = LinkDiscovery(small_config)
        assert discovery.config.max_depth == 0

        # Test with very large limits
        large_config = LinkDiscoveryConfig(
            max_depth=1000,
            max_pages=1000000,
            max_links_per_page=10000,
            batch_size=1000
        )
        discovery = LinkDiscovery(large_config)
        assert discovery.config.max_depth == 1000

    @pytest.mark.asyncio
    async def test_json_ld_malformed_handling(self):
        """Test handling of malformed JSON-LD structured data."""
        discovery = LinkDiscovery()

        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "invalid": "json",
                    "missing": "quotes,
                    "broken": syntax
                }
                </script>
            </head>
            <body></body>
        </html>
        """

        # Should not crash on malformed JSON-LD
        links = await discovery.discover_links(html, "http://example.com", 0)
        assert isinstance(links, list)

    def test_robots_txt_edge_cases(self):
        """Test robots.txt parsing edge cases."""
        extractor = LinkExtractor(LinkDiscoveryConfig())

        edge_case_robots = """
        # Comments should be ignored
        User-agent: *
        SITEMAP: http://example.com/sitemap1.xml
        sitemap: http://example.com/sitemap2.xml
        Sitemap:
        Invalid line without colon
        Sitemap: not-a-valid-url
        Sitemap: http://example.com/valid.xml
        """

        links = extractor.extract_sitemap_links(edge_case_robots, "http://example.com")

        # Should extract valid sitemaps and handle edge cases gracefully
        assert isinstance(links, list)
        valid_links = [link for link in links if "http" in link.url]
        assert len(valid_links) >= 2  # At least the valid ones

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Test thread safety of concurrent operations."""
        discovery = LinkDiscovery()

        # Simulate concurrent link additions
        tasks = []
        for i in range(10):
            links = [
                DiscoveredLink(url=f"http://example.com/page{i}-{j}", source_url="http://example.com")
                for j in range(5)
            ]
            task = asyncio.create_task(asyncio.to_thread(discovery.add_discovered_links, links))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Should have all URLs without corruption
        assert len(discovery._discovered_urls) == 50
        assert len(discovery._crawl_queue) == 50
