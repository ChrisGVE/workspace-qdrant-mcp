"""Comprehensive unit tests for LinkDiscovery and RecursiveCrawler components.

Tests cover link discovery, filtering, quality assessment, and recursive crawling.
"""

import re
import time
from unittest.mock import patch

import pytest

from workspace_qdrant_mcp.web.links import (
    CrawlSession,
    DiscoveredLink,
    LinkDiscovery,
    LinkFilter,
    LinkType,
    RecursiveCrawler,
)


class TestLinkFilter:
    """Test LinkFilter configuration and defaults."""

    def test_default_filter(self):
        """Test default LinkFilter configuration."""
        filter_config = LinkFilter()

        assert filter_config.include_subdomains is True
        assert filter_config.include_external_links is False
        assert filter_config.include_subdomain_links is True
        assert filter_config.include_protocol_links is False
        assert filter_config.max_depth == 3
        assert filter_config.max_links_per_page == 100
        assert filter_config.respect_nofollow is True

        # Check default extensions
        assert 'html' in filter_config.allowed_extensions
        assert 'pdf' in filter_config.allowed_extensions
        assert 'css' in filter_config.blocked_extensions
        assert 'jpg' in filter_config.blocked_extensions

        # Check patterns were initialized
        assert len(filter_config.blocked_link_text_patterns) > 0
        assert len(filter_config.blocked_path_patterns) > 0

    def test_custom_filter(self):
        """Test custom LinkFilter configuration."""
        filter_config = LinkFilter(
            allowed_domains={'example.com'},
            blocked_domains={'spam.com'},
            include_external_links=True,
            max_depth=5,
            max_links_per_page=50
        )

        assert 'example.com' in filter_config.allowed_domains
        assert 'spam.com' in filter_config.blocked_domains
        assert filter_config.include_external_links is True
        assert filter_config.max_depth == 5
        assert filter_config.max_links_per_page == 50

    def test_pattern_initialization(self):
        """Test that default patterns are properly initialized."""
        filter_config = LinkFilter()

        # Test blocked text patterns
        login_pattern = None
        for pattern in filter_config.blocked_link_text_patterns:
            if pattern.search('login'):
                login_pattern = pattern
                break

        assert login_pattern is not None
        assert login_pattern.search('Login')  # Case insensitive

        # Test blocked path patterns
        admin_pattern = None
        for pattern in filter_config.blocked_path_patterns:
            if pattern.search('/admin/'):
                admin_pattern = pattern
                break

        assert admin_pattern is not None


class TestDiscoveredLink:
    """Test DiscoveredLink dataclass and URL parsing."""

    def test_basic_link_creation(self):
        """Test basic DiscoveredLink creation."""
        link = DiscoveredLink(
            url="https://example.com/page.html",
            text="Example Page",
            link_type=LinkType.INTERNAL,
            source_url="https://example.com",
            depth=1
        )

        assert link.url == "https://example.com/page.html"
        assert link.text == "Example Page"
        assert link.link_type == LinkType.INTERNAL
        assert link.depth == 1

        # Check parsed components
        assert link.domain == "example.com"
        assert link.path == "/page.html"
        assert link.extension == "html"

    def test_url_component_parsing(self):
        """Test URL component parsing."""
        link = DiscoveredLink(
            url="https://sub.example.com/dir/page.php?param1=value1&param2=value2",
            text="Test Link",
            link_type=LinkType.SUBDOMAIN,
            source_url="https://example.com",
            depth=1
        )

        assert link.domain == "sub.example.com"
        assert link.path == "/dir/page.php"
        assert link.extension == "php"
        assert link.query_params["param1"] == "value1"
        assert link.query_params["param2"] == "value2"

    def test_extension_extraction(self):
        """Test file extension extraction from paths."""
        test_cases = [
            ("/page.html", "html"),
            ("/document.PDF", "pdf"),  # Should be lowercase
            ("/file.tar.gz", "gz"),    # Should get last extension
            ("/no-extension", ""),
            ("/", ""),
        ]

        for path, expected_ext in test_cases:
            link = DiscoveredLink(
                url=f"https://example.com{path}",
                text="Test",
                link_type=LinkType.INTERNAL,
                source_url="https://example.com",
                depth=0
            )
            assert link.extension == expected_ext, f"Failed for path: {path}"

    def test_malformed_url_handling(self):
        """Test handling of malformed URLs."""
        # Should not raise exception
        link = DiscoveredLink(
            url="not-a-valid-url",
            text="Bad Link",
            link_type=LinkType.INVALID,
            source_url="https://example.com",
            depth=0
        )

        # Should have empty or safe defaults
        assert isinstance(link.domain, str)
        assert isinstance(link.path, str)
        assert isinstance(link.extension, str)


class TestLinkDiscovery:
    """Test LinkDiscovery functionality."""

    @pytest.fixture
    def discovery(self):
        """LinkDiscovery instance for testing."""
        return LinkDiscovery()

    @pytest.fixture
    def restrictive_discovery(self):
        """LinkDiscovery with restrictive filters."""
        filter_config = LinkFilter(
            allowed_domains={'example.com'},
            include_external_links=False,
            max_links_per_page=5
        )
        return LinkDiscovery(filter_config)

    @pytest.fixture
    def sample_html(self):
        """Sample HTML with various link types."""
        return """
        <html>
        <body>
            <h1>Test Page</h1>

            <!-- Internal links -->
            <a href="/internal-page">Internal Page</a>
            <a href="relative-page.html" title="Relative link">Relative Page</a>
            <a href="https://example.com/absolute-internal">Absolute Internal</a>

            <!-- External links -->
            <a href="https://external.com/page">External Page</a>
            <a href="https://subdomain.example.com/sub">Subdomain Page</a>

            <!-- Protocol links -->
            <a href="mailto:test@example.com">Email Link</a>
            <a href="tel:+1234567890">Phone Link</a>
            <a href="ftp://ftp.example.com">FTP Link</a>

            <!-- Fragment links -->
            <a href="#section1">Section 1</a>

            <!-- Links with attributes -->
            <a href="/page" class="nav-link" rel="nofollow">No Follow</a>
            <a href="/important" class="important" title="Important page">Important</a>

            <!-- Links with various text lengths -->
            <a href="/short">Go</a>
            <a href="/medium">Medium length link text</a>
            <a href="/long">Very long link text that exceeds normal expectations and might be filtered</a>

            <!-- Empty and bad links -->
            <a href="">Empty href</a>
            <a href="javascript:void(0)">JavaScript</a>
            <a>No href</a>

            <!-- File type links -->
            <a href="/document.pdf">PDF Document</a>
            <a href="/image.jpg">Image</a>
            <a href="/styles.css">Stylesheet</a>
            <a href="/script.js">JavaScript</a>

            <!-- Navigation-like links -->
            <a href="/home">Home</a>
            <a href="/login">Login</a>
            <a href="/admin">Admin</a>
        </body>
        </html>
        """

    def test_discover_basic_links(self, discovery, sample_html):
        """Test basic link discovery functionality."""
        base_url = "https://example.com/test-page"
        links = discovery.discover_links(sample_html, base_url)

        assert len(links) > 0

        # Check that we found various types of links
        internal_links = [l for l in links if l.link_type == LinkType.INTERNAL]
        external_links = [l for l in links if l.link_type == LinkType.EXTERNAL]
        subdomain_links = [l for l in links if l.link_type == LinkType.SUBDOMAIN]

        assert len(internal_links) > 0
        assert len(external_links) > 0  # Since default allows external links
        assert len(subdomain_links) > 0

        # Check URL resolution
        relative_link = next((l for l in links if 'relative-page.html' in l.url), None)
        assert relative_link is not None
        assert relative_link.url == "https://example.com/relative-page.html"

    def test_link_classification(self, discovery):
        """Test link classification logic."""
        base_url = "https://example.com/page"

        test_cases = [
            ("https://example.com/other", LinkType.INTERNAL),
            ("https://sub.example.com/page", LinkType.SUBDOMAIN),
            ("https://external.com/page", LinkType.EXTERNAL),
            ("mailto:test@example.com", LinkType.PROTOCOL),
            ("tel:+123456", LinkType.PROTOCOL),
            ("#section", LinkType.FRAGMENT),
            ("invalid-url", LinkType.INVALID),
        ]

        for url, expected_type in test_cases:
            link_type = discovery._classify_link(url, base_url)
            assert link_type == expected_type, f"Failed for URL: {url}"

    def test_url_normalization(self, discovery):
        """Test URL normalization."""
        test_cases = [
            ("https://example.com/page#fragment", "https://example.com/page"),
            ("https://example.com/page?utm_source=test", "https://example.com/page"),
            ("https://example.com/page/", "https://example.com/page"),
            ("https://EXAMPLE.COM/page", "https://example.com/page"),
            ("https://example.com//double//slash", "https://example.com/double/slash"),
        ]

        for original, expected in test_cases:
            normalized = discovery._normalize_url(original)
            assert normalized == expected, f"Failed to normalize {original}"

    def test_link_filtering(self, restrictive_discovery, sample_html):
        """Test link filtering with restrictive configuration."""
        base_url = "https://example.com/page"
        links = restrictive_discovery.discover_links(sample_html, base_url)

        # Should have fewer links due to restrictions
        assert len(links) <= 5  # max_links_per_page limit

        # Should not have external links
        external_links = [l for l in links if l.link_type == LinkType.EXTERNAL]
        assert len(external_links) == 0

        # All remaining links should be from allowed domain
        for link in links:
            if link.link_type in (LinkType.INTERNAL, LinkType.SUBDOMAIN):
                assert 'example.com' in link.url

    def test_nofollow_filtering(self):
        """Test nofollow link filtering."""
        filter_config = LinkFilter(respect_nofollow=True)
        discovery = LinkDiscovery(filter_config)

        html_with_nofollow = """
        <html>
        <body>
            <a href="/normal">Normal Link</a>
            <a href="/nofollow" rel="nofollow">No Follow Link</a>
            <a href="/multiple" rel="nofollow sponsored">Multiple Attributes</a>
        </body>
        </html>
        """

        links = discovery.discover_links(html_with_nofollow, "https://example.com")

        # Should only find the normal link
        assert len(links) == 1
        assert links[0].text == "Normal Link"

    def test_extension_filtering(self):
        """Test file extension filtering."""
        filter_config = LinkFilter(
            allowed_extensions={'html', 'pdf'},
            blocked_extensions={'css', 'js', 'jpg'}
        )
        discovery = LinkDiscovery(filter_config)

        html_with_files = """
        <html>
        <body>
            <a href="/page.html">HTML Page</a>
            <a href="/doc.pdf">PDF Document</a>
            <a href="/style.css">Stylesheet</a>
            <a href="/script.js">JavaScript</a>
            <a href="/image.jpg">Image</a>
            <a href="/no-extension">No Extension</a>
        </body>
        </html>
        """

        links = discovery.discover_links(html_with_files, "https://example.com")

        # Should include HTML, PDF, and no-extension links
        # Should exclude CSS, JS, and image links
        allowed_links = {l.text for l in links}
        assert "HTML Page" in allowed_links
        assert "PDF Document" in allowed_links
        assert "No Extension" in allowed_links
        assert "Stylesheet" not in allowed_links
        assert "JavaScript" not in allowed_links
        assert "Image" not in allowed_links

    def test_path_pattern_filtering(self):
        """Test path pattern filtering."""
        filter_config = LinkFilter(
            blocked_path_patterns=[
                re.compile(r'/admin/'),
                re.compile(r'/login'),
            ]
        )
        discovery = LinkDiscovery(filter_config)

        html_with_paths = """
        <html>
        <body>
            <a href="/public/page">Public Page</a>
            <a href="/admin/panel">Admin Panel</a>
            <a href="/login">Login Page</a>
            <a href="/user/login-help">Login Help</a>
        </body>
        </html>
        """

        links = discovery.discover_links(html_with_paths, "https://example.com")

        link_texts = {l.text for l in links}
        assert "Public Page" in link_texts
        assert "Login Help" in link_texts
        assert "Admin Panel" not in link_texts
        assert "Login Page" not in link_texts

    def test_link_text_filtering(self):
        """Test link text pattern filtering."""
        filter_config = LinkFilter(
            min_link_text_length=3,
            max_link_text_length=50,
            blocked_link_text_patterns=[
                re.compile(r'^(login|logout)$', re.IGNORECASE)
            ]
        )
        discovery = LinkDiscovery(filter_config)

        html_with_text = """
        <html>
        <body>
            <a href="/page1">OK</a>
            <a href="/page2">Good Link</a>
            <a href="/page3">Login</a>
            <a href="/page4">LOGOUT</a>
            <a href="/page5">""" + "Very " * 20 + """long link text</a>
        </body>
        </html>
        """

        links = discovery.discover_links(html_with_text, "https://example.com")

        link_texts = {l.text for l in links}
        assert "Good Link" in link_texts
        assert "OK" not in link_texts  # Too short
        assert "Login" not in link_texts  # Blocked pattern
        assert "LOGOUT" not in link_texts  # Blocked pattern (case insensitive)
        # Very long text should not be present

    def test_link_quality_scoring(self, discovery):
        """Test link quality scoring algorithm."""
        html_with_various_links = """
        <html>
        <body>
            <a href="/article/great-story" title="Interesting article">Great Article Story</a>
            <a href="/home">Home</a>
            <a href="/very/deep/nested/path/file">Deep Link</a>
            <a href="/doc.pdf">PDF Document</a>
            <a href="/page?param=value">Query Params</a>
            <a href="https://external.com/page">External Link</a>
        </body>
        </html>
        """

        links = discovery.discover_links(html_with_various_links, "https://example.com")

        # Find specific links
        article_link = next((l for l in links if "Article" in l.text), None)
        home_link = next((l for l in links if l.text == "Home"), None)
        deep_link = next((l for l in links if "deep" in l.url), None)
        pdf_link = next((l for l in links if "pdf" in l.url), None)

        # Article should have high quality score (good text, title attribute)
        if article_link:
            assert article_link.quality_score > 60

        # Home should have lower score (navigation penalty)
        if home_link:
            assert home_link.quality_score < 60

        # Deep link should have penalty
        if deep_link:
            assert deep_link.quality_score < 50

        # PDF should have bonus
        if pdf_link:
            assert pdf_link.quality_score > 50

    def test_link_prioritization(self, discovery, sample_html):
        """Test link prioritization based on quality scores."""
        links = discovery.discover_links(sample_html, "https://example.com")

        # Links should be sorted by priority and quality score
        for i in range(len(links) - 1):
            current_priority = links[i].priority
            current_quality = links[i].quality_score
            next_priority = links[i + 1].priority
            next_quality = links[i + 1].quality_score

            # Higher priority should come first, or equal priority with higher quality
            assert (current_priority > next_priority or
                   (current_priority == next_priority and current_quality >= next_quality))

    def test_malformed_html_handling(self, discovery):
        """Test handling of malformed HTML."""
        malformed_html = """
        <html>
        <body>
            <a href="/page1">Good Link
            <a href="/page2"">Bad Quote</a>
            <a href=/page3>Missing Quotes</a>
            <a href="/page4">Unclosed tag
            <div><a href="/page5">Nested</a></div>
        </body>
        """

        # Should not raise exception
        links = discovery.discover_links(malformed_html, "https://example.com")

        # Should still extract some valid links
        assert len(links) > 0

        # Check that valid links were extracted
        link_urls = {l.url for l in links}
        assert any('/page' in url for url in link_urls)

    def test_empty_html_handling(self, discovery):
        """Test handling of empty or invalid HTML."""
        test_cases = [
            "",  # Empty string
            "<html></html>",  # No links
            "<div>No links here</div>",  # No anchor tags
            "Not HTML at all",  # Plain text
        ]

        for html_content in test_cases:
            links = discovery.discover_links(html_content, "https://example.com")
            assert links == []

    def test_subdomain_filtering(self):
        """Test subdomain inclusion/exclusion logic."""
        # Test with subdomains allowed
        filter_allowed = LinkFilter(
            allowed_domains={'example.com'},
            include_subdomains=True
        )
        discovery_allowed = LinkDiscovery(filter_allowed)

        # Test with subdomains blocked
        filter_blocked = LinkFilter(
            allowed_domains={'example.com'},
            include_subdomains=False
        )
        discovery_blocked = LinkDiscovery(filter_blocked)

        html_subdomains = """
        <html>
        <body>
            <a href="https://example.com/page">Main Domain</a>
            <a href="https://sub.example.com/page">Subdomain</a>
            <a href="https://other.com/page">Other Domain</a>
        </body>
        </html>
        """

        # With subdomains allowed
        links_allowed = discovery_allowed.discover_links(html_subdomains, "https://example.com")
        allowed_urls = {l.url for l in links_allowed}

        assert "https://example.com/page" in allowed_urls
        assert "https://sub.example.com/page" in allowed_urls
        assert "https://other.com/page" not in allowed_urls

        # With subdomains blocked
        links_blocked = discovery_blocked.discover_links(html_subdomains, "https://example.com")
        blocked_urls = {l.url for l in links_blocked}

        assert "https://example.com/page" in blocked_urls
        assert "https://sub.example.com/page" not in blocked_urls
        assert "https://other.com/page" not in blocked_urls

    def test_max_links_per_page_limit(self):
        """Test max links per page limitation."""
        filter_config = LinkFilter(max_links_per_page=3)
        discovery = LinkDiscovery(filter_config)

        # Generate HTML with many links
        many_links_html = "<html><body>" + \
                         "".join([f'<a href="/page{i}">Page {i}</a>' for i in range(10)]) + \
                         "</body></html>"

        links = discovery.discover_links(many_links_html, "https://example.com")

        # Should be limited to max_links_per_page
        assert len(links) <= 3

    def test_link_attributes_extraction(self, discovery):
        """Test extraction of link attributes."""
        html_with_attributes = """
        <html>
        <body>
            <a href="/page" class="nav-link important" title="Navigation Link" rel="noopener noreferrer">
                Link with Attributes
            </a>
        </body>
        </html>
        """

        # Use discovery without nofollow restriction for this test
        filter_config = LinkFilter(respect_nofollow=False)
        discovery = LinkDiscovery(filter_config)

        links = discovery.discover_links(html_with_attributes, "https://example.com")

        assert len(links) == 1
        link = links[0]

        assert link.title == "Navigation Link"
        assert "nav-link" in link.class_names
        assert "important" in link.class_names
        assert "noopener" in link.rel_attributes
        assert "noreferrer" in link.rel_attributes


class TestCrawlSession:
    """Test CrawlSession functionality."""

    def test_session_initialization(self):
        """Test CrawlSession initialization."""
        session = CrawlSession(
            session_id="test-session",
            start_url="https://example.com",
            max_depth=3
        )

        assert session.session_id == "test-session"
        assert session.start_url == "https://example.com"
        assert session.max_depth == 3
        assert len(session.discovered_urls) == 0
        assert len(session.all_links) == 0

    def test_session_state_tracking(self):
        """Test session state tracking."""
        session = CrawlSession(
            session_id="test",
            start_url="https://example.com",
            max_depth=2
        )

        # Add some URLs to different sets
        session.discovered_urls.add("https://example.com/page1")
        session.discovered_urls.add("https://example.com/page2")
        session.crawled_urls.add("https://example.com/page1")
        session.failed_urls.add("https://example.com/page3")

        assert len(session.discovered_urls) == 2
        assert len(session.crawled_urls) == 1
        assert len(session.failed_urls) == 1

        # Test visit counting
        session.url_visit_counts["https://example.com/page1"] = 2
        assert session.url_visit_counts["https://example.com/page1"] == 2


class TestRecursiveCrawler:
    """Test RecursiveCrawler functionality."""

    @pytest.fixture
    def discovery(self):
        """LinkDiscovery instance for testing."""
        return LinkDiscovery()

    @pytest.fixture
    def crawler(self, discovery):
        """RecursiveCrawler instance for testing."""
        return RecursiveCrawler(discovery)

    def test_start_crawl_session(self, crawler):
        """Test starting a crawl session."""
        session = crawler.start_crawl_session("https://example.com", max_depth=2)

        assert session.start_url == "https://example.com"
        assert session.max_depth == 2
        assert len(session.discovered_urls) == 1
        assert "https://example.com" in session.discovered_urls
        assert len(session.all_links) == 1

        # Check that session is tracked
        assert session.session_id in crawler.link_discovery._active_sessions

    def test_start_crawl_session_with_custom_id(self, crawler):
        """Test starting a crawl session with custom ID."""
        custom_id = "my-custom-session"
        session = crawler.start_crawl_session("https://example.com", max_depth=1, session_id=custom_id)

        assert session.session_id == custom_id

    def test_get_next_urls_to_crawl(self, crawler):
        """Test getting next URLs to crawl."""
        session = crawler.start_crawl_session("https://example.com", max_depth=2)

        # Add some links at depth 1
        link1 = DiscoveredLink(
            url="https://example.com/page1",
            text="Page 1",
            link_type=LinkType.INTERNAL,
            source_url="https://example.com",
            depth=1,
            priority=8
        )
        link2 = DiscoveredLink(
            url="https://example.com/page2",
            text="Page 2",
            link_type=LinkType.INTERNAL,
            source_url="https://example.com",
            depth=1,
            priority=5
        )

        session.links_by_depth[1] = [link1, link2]
        session.discovered_urls.add(link1.url)
        session.discovered_urls.add(link2.url)

        # Get next URLs to crawl at depth 1
        next_urls = crawler.get_next_urls_to_crawl(session.session_id, 1)

        assert len(next_urls) == 2
        # Should be sorted by priority (higher first)
        assert next_urls[0].priority >= next_urls[1].priority

    def test_get_next_urls_beyond_max_depth(self, crawler):
        """Test getting next URLs beyond max depth."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        # Try to get URLs at depth 2 (beyond max depth)
        next_urls = crawler.get_next_urls_to_crawl(session.session_id, 2)

        assert next_urls == []

    def test_add_discovered_links(self, crawler):
        """Test adding discovered links to session."""
        session = crawler.start_crawl_session("https://example.com", max_depth=2)

        # Create some discovered links
        new_links = [
            DiscoveredLink(
                url="https://example.com/new1",
                text="New Page 1",
                link_type=LinkType.INTERNAL,
                source_url="https://example.com",
                depth=0  # Will be updated
            ),
            DiscoveredLink(
                url="https://example.com/new2",
                text="New Page 2",
                link_type=LinkType.INTERNAL,
                source_url="https://example.com",
                depth=0  # Will be updated
            )
        ]

        # Add links discovered from depth 0
        crawler.add_discovered_links(session.session_id, "https://example.com", new_links, 0)

        # Check that links were added at depth 1
        assert 1 in session.links_by_depth
        assert len(session.links_by_depth[1]) == 2

        # Check that URLs were added to discovered set
        assert "https://example.com/new1" in session.discovered_urls
        assert "https://example.com/new2" in session.discovered_urls

        # Check that depth was set correctly
        added_links = session.links_by_depth[1]
        for link in added_links:
            assert link.depth == 1

    def test_add_discovered_links_beyond_max_depth(self, crawler):
        """Test adding links beyond max depth."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        new_links = [
            DiscoveredLink(
                url="https://example.com/deep",
                text="Deep Page",
                link_type=LinkType.INTERNAL,
                source_url="https://example.com/page1",
                depth=0
            )
        ]

        # Try to add links from depth 1 (would create depth 2, beyond max)
        initial_count = len(session.all_links)
        crawler.add_discovered_links(session.session_id, "https://example.com/page1", new_links, 1)

        # Should not add any links
        assert len(session.all_links) == initial_count

    def test_add_discovered_duplicate_links(self, crawler):
        """Test adding duplicate links."""
        session = crawler.start_crawl_session("https://example.com", max_depth=2)

        # Add a URL to discovered set
        session.discovered_urls.add("https://example.com/existing")

        duplicate_links = [
            DiscoveredLink(
                url="https://example.com/existing",
                text="Existing Page",
                link_type=LinkType.INTERNAL,
                source_url="https://example.com",
                depth=0
            ),
            DiscoveredLink(
                url="https://example.com/new",
                text="New Page",
                link_type=LinkType.INTERNAL,
                source_url="https://example.com",
                depth=0
            )
        ]

        initial_count = len(session.all_links)
        crawler.add_discovered_links(session.session_id, "https://example.com", duplicate_links, 0)

        # Should only add the new link
        assert len(session.all_links) == initial_count + 1
        assert "https://example.com/new" in session.discovered_urls

    def test_mark_url_crawled(self, crawler):
        """Test marking URLs as crawled."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"

        # Mark one as successfully crawled
        crawler.mark_url_crawled(session.session_id, url1, success=True)

        # Mark one as failed
        crawler.mark_url_crawled(session.session_id, url2, success=False)

        assert url1 in session.crawled_urls
        assert url2 in session.failed_urls
        assert session.url_visit_counts[url1] == 1
        assert session.url_visit_counts[url2] == 1

    def test_handle_redirect(self, crawler):
        """Test redirect handling and loop detection."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        original_url = "https://example.com/redirect"
        redirect_url = "https://example.com/target"

        # Handle normal redirect
        result = crawler.handle_redirect(session.session_id, original_url, redirect_url)
        assert result is True
        assert original_url in session.redirect_chains
        assert redirect_url in session.redirect_chains[original_url]

    def test_redirect_loop_detection(self, crawler):
        """Test redirect loop detection."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        url_a = "https://example.com/a"
        url_b = "https://example.com/b"

        # Create redirect chain: A -> B
        crawler.handle_redirect(session.session_id, url_a, url_b)

        # Try to create loop: B -> A (should be detected)
        result = crawler.handle_redirect(session.session_id, url_b, url_a)
        assert result is False  # Loop detected

    def test_long_redirect_chain(self, crawler):
        """Test long redirect chain handling."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        original_url = "https://example.com/start"

        # Create a very long redirect chain
        current_url = original_url
        for i in range(15):  # Exceeds max chain length
            next_url = f"https://example.com/redirect{i}"
            result = crawler.handle_redirect(session.session_id, current_url, next_url)

            if i < 10:  # Within limit
                assert result is True
            else:  # Beyond limit
                assert result is False
                break

            current_url = original_url  # Keep building chain for original URL

    def test_should_crawl_url_visit_limits(self, crawler):
        """Test URL visit limit checking."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)

        url = "https://example.com/page"

        # Should allow crawling initially
        assert crawler._should_crawl_url(session, url) is True

        # Mark as visited multiple times
        session.url_visit_counts[url] = 2
        assert crawler._should_crawl_url(session, url) is True

        # Exceed visit limit
        session.url_visit_counts[url] = 5
        assert crawler._should_crawl_url(session, url) is False

    def test_get_crawl_statistics(self, crawler):
        """Test crawl statistics generation."""
        session = crawler.start_crawl_session("https://example.com", max_depth=2)

        # Add some test data
        session.total_links_found = 10
        session.unique_domains.add("example.com")
        session.unique_domains.add("sub.example.com")
        session.crawled_urls.add("https://example.com/page1")
        session.failed_urls.add("https://example.com/page2")
        session.links_by_depth[1] = ["link1", "link2"]
        session.links_by_depth[2] = ["link3"]

        stats = crawler.get_crawl_statistics(session.session_id)

        assert stats["session_id"] == session.session_id
        assert stats["start_url"] == "https://example.com"
        assert stats["max_depth"] == 2
        assert stats["total_links_discovered"] == 10
        assert stats["unique_domains"] == 2
        assert stats["urls_crawled"] == 1
        assert stats["urls_failed"] == 1
        assert stats["links_by_depth"] == {1: 2, 2: 1}
        assert "example.com" in stats["domains_discovered"]
        assert "sub.example.com" in stats["domains_discovered"]

    def test_end_crawl_session(self, crawler):
        """Test ending a crawl session."""
        session = crawler.start_crawl_session("https://example.com", max_depth=1)
        session_id = session.session_id

        # Session should be active
        assert session_id in crawler.link_discovery._active_sessions

        # End session
        ended_session = crawler.end_crawl_session(session_id)

        # Should return the session and remove from active sessions
        assert ended_session is not None
        assert ended_session.session_id == session_id
        assert session_id not in crawler.link_discovery._active_sessions

    def test_end_nonexistent_session(self, crawler):
        """Test ending a nonexistent session."""
        result = crawler.end_crawl_session("nonexistent-session")
        assert result is None

    def test_cleanup_old_sessions(self, crawler):
        """Test cleanup of old sessions."""
        # Create session with old timestamp
        old_session = crawler.start_crawl_session("https://example.com", max_depth=1)
        old_session.crawl_start_time = time.time() - (25 * 3600)  # 25 hours ago

        # Create recent session
        recent_session = crawler.start_crawl_session("https://example2.com", max_depth=1)

        # Cleanup sessions older than 24 hours
        cleaned_count = crawler.cleanup_old_sessions(max_age_hours=24)

        assert cleaned_count == 1
        assert old_session.session_id not in crawler.link_discovery._active_sessions
        assert recent_session.session_id in crawler.link_discovery._active_sessions

    def test_invalid_session_operations(self, crawler):
        """Test operations on invalid/nonexistent sessions."""
        invalid_session_id = "nonexistent"

        # Should handle gracefully
        next_urls = crawler.get_next_urls_to_crawl(invalid_session_id, 1)
        assert next_urls == []

        crawler.add_discovered_links(invalid_session_id, "https://example.com", [], 0)
        # Should not raise exception

        crawler.mark_url_crawled(invalid_session_id, "https://example.com/page")
        # Should not raise exception

        result = crawler.handle_redirect(invalid_session_id, "url1", "url2")
        assert result is False

        stats = crawler.get_crawl_statistics(invalid_session_id)
        assert stats == {}


@pytest.mark.integration
def test_full_crawl_workflow():
    """Integration test for complete crawl workflow."""
    # Setup
    link_filter = LinkFilter(
        max_depth=2,
        max_links_per_page=10,
        include_external_links=False
    )
    discovery = LinkDiscovery(link_filter)
    crawler = RecursiveCrawler(discovery)

    # Start session
    session = crawler.start_crawl_session("https://example.com", max_depth=2)

    # Simulate discovering links from start page
    start_html = """
    <html>
    <body>
        <a href="/page1">Page 1</a>
        <a href="/page2">Page 2</a>
        <a href="https://external.com/page">External</a>
    </body>
    </html>
    """

    discovered_links = discovery.discover_links(start_html, "https://example.com")
    crawler.add_discovered_links(session.session_id, "https://example.com", discovered_links, 0)

    # Get URLs to crawl at depth 1
    depth1_urls = crawler.get_next_urls_to_crawl(session.session_id, 1)
    assert len(depth1_urls) == 2  # External link should be filtered out

    # Mark first URL as crawled and add more links
    first_url = depth1_urls[0].url
    crawler.mark_url_crawled(session.session_id, first_url, success=True)

    # Simulate finding more links on first page
    page1_html = """
    <html>
    <body>
        <a href="/page3">Page 3</a>
        <a href="/page4">Page 4</a>
    </body>
    </html>
    """

    page1_links = discovery.discover_links(page1_html, first_url)
    crawler.add_discovered_links(session.session_id, first_url, page1_links, 1)

    # Check depth 2 links were added
    depth2_urls = crawler.get_next_urls_to_crawl(session.session_id, 2)
    assert len(depth2_urls) == 2

    # Get final statistics
    stats = crawler.get_crawl_statistics(session.session_id)
    assert stats["urls_crawled"] == 1
    assert stats["total_links_discovered"] >= 4

    # End session
    final_session = crawler.end_crawl_session(session.session_id)
    assert final_session is not None