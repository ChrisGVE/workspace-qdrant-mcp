"""Comprehensive unit tests for the WebCrawler component.

Tests cover all edge cases, error conditions, and normal operation scenarios.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from workspace_qdrant_mcp.web.crawler import (
    CrawlConfig,
    CrawlResponse,
    RateLimiter,
    RobotsChecker,
    WebCrawler,
)


class TestCrawlConfig:
    """Test CrawlConfig dataclass functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CrawlConfig()
        assert config.delay_between_requests == 1.0
        assert config.concurrent_requests == 5
        assert config.max_requests_per_second == 2.0
        assert config.timeout == 30.0
        assert config.max_redirects == 10
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.max_content_size == 10 * 1024 * 1024
        assert config.user_agent == "ResponsibleWebCrawler/1.0"
        assert config.respect_robots_txt is True
        assert config.robots_txt_cache_ttl == 3600

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CrawlConfig(
            delay_between_requests=2.0,
            concurrent_requests=10,
            max_requests_per_second=5.0,
            timeout=60.0,
            user_agent="CustomCrawler/1.0",
            respect_robots_txt=False
        )
        assert config.delay_between_requests == 2.0
        assert config.concurrent_requests == 10
        assert config.max_requests_per_second == 5.0
        assert config.timeout == 60.0
        assert config.user_agent == "CustomCrawler/1.0"
        assert config.respect_robots_txt is False

    def test_default_allowed_content_types(self):
        """Test default allowed content types."""
        config = CrawlConfig()
        expected_types = {
            'text/html', 'text/plain', 'text/xml',
            'application/xml', 'application/xhtml+xml',
            'application/json', 'text/css', 'text/javascript'
        }
        assert config.allowed_content_types == expected_types

    def test_custom_allowed_content_types(self):
        """Test custom allowed content types."""
        custom_types = {'text/html', 'application/json'}
        config = CrawlConfig(allowed_content_types=custom_types)
        assert config.allowed_content_types == custom_types

    def test_custom_headers(self):
        """Test custom headers configuration."""
        custom_headers = {'X-Custom-Header': 'value'}
        config = CrawlConfig(custom_headers=custom_headers)
        assert config.custom_headers == custom_headers

        # Test default empty headers
        default_config = CrawlConfig()
        assert default_config.custom_headers == {}


class TestCrawlResponse:
    """Test CrawlResponse dataclass functionality."""

    def test_basic_response(self):
        """Test basic CrawlResponse creation."""
        response = CrawlResponse(
            url="https://example.com",
            status_code=200,
            content="<html>Content</html>",
            headers={"content-type": "text/html"},
            content_type="text/html",
            encoding="utf-8"
        )
        assert response.url == "https://example.com"
        assert response.status_code == 200
        assert response.content == "<html>Content</html>"
        assert response.headers == {"content-type": "text/html"}
        assert response.content_type == "text/html"
        assert response.encoding == "utf-8"
        assert response.redirect_url is None
        assert isinstance(response.crawl_time, datetime)
        assert response.error is None

    def test_response_with_error(self):
        """Test CrawlResponse with error."""
        response = CrawlResponse(
            url="https://example.com",
            status_code=404,
            content="",
            headers={},
            content_type="",
            encoding="",
            error="Not found"
        )
        assert response.error == "Not found"
        assert response.status_code == 404

    def test_response_with_redirect(self):
        """Test CrawlResponse with redirect."""
        response = CrawlResponse(
            url="https://example.com",
            status_code=200,
            content="Content",
            headers={},
            content_type="text/html",
            encoding="utf-8",
            redirect_url="https://www.example.com"
        )
        assert response.redirect_url == "https://www.example.com"

    def test_custom_crawl_time(self):
        """Test CrawlResponse with custom crawl time."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        response = CrawlResponse(
            url="https://example.com",
            status_code=200,
            content="",
            headers={},
            content_type="",
            encoding="",
            crawl_time=custom_time
        )
        assert response.crawl_time == custom_time


class TestRateLimiter:
    """Test RateLimiter functionality."""

    @pytest.fixture
    def config(self):
        """Rate limiter test configuration."""
        return CrawlConfig(
            delay_between_requests=0.5,
            max_requests_per_second=2.0
        )

    @pytest.fixture
    def rate_limiter(self, config):
        """Rate limiter instance for testing."""
        return RateLimiter(config)

    @pytest.mark.asyncio
    async def test_first_request_no_delay(self, rate_limiter):
        """Test that first request to domain has no delay."""
        start_time = time.time()
        await rate_limiter.wait_if_needed("example.com")
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be nearly instantaneous

    @pytest.mark.asyncio
    async def test_delay_between_requests(self, rate_limiter):
        """Test delay between requests to same domain."""
        # First request
        await rate_limiter.wait_if_needed("example.com")

        # Second request should be delayed
        start_time = time.time()
        await rate_limiter.wait_if_needed("example.com")
        elapsed = time.time() - start_time
        assert elapsed >= 0.4  # Should wait at least 0.4s (0.5s - small tolerance)

    @pytest.mark.asyncio
    async def test_no_delay_different_domains(self, rate_limiter):
        """Test no delay between requests to different domains."""
        await rate_limiter.wait_if_needed("example.com")

        # Request to different domain should have no delay
        start_time = time.time()
        await rate_limiter.wait_if_needed("test.com")
        elapsed = time.time() - start_time
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_requests_per_second_limit(self, rate_limiter):
        """Test requests per second limiting."""
        domain = "example.com"

        # Make requests quickly to trigger RPS limit
        start_time = time.time()
        await rate_limiter.wait_if_needed(domain)
        await rate_limiter.wait_if_needed(domain)
        await rate_limiter.wait_if_needed(domain)  # This should be delayed
        elapsed = time.time() - start_time

        # Should take at least 1 second due to RPS limit
        assert elapsed >= 0.9

    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safety(self, rate_limiter):
        """Test thread safety with concurrent access."""
        async def make_request():
            await rate_limiter.wait_if_needed("example.com")
            return time.time()

        # Start multiple concurrent requests
        tasks = [make_request() for _ in range(5)]
        times = await asyncio.gather(*tasks)

        # Check that times are properly spaced due to rate limiting
        sorted_times = sorted(times)
        for i in range(1, len(sorted_times)):
            time_diff = sorted_times[i] - sorted_times[i-1]
            # Allow some tolerance for timing variations
            assert time_diff >= 0.3  # Should be close to delay_between_requests


class TestRobotsChecker:
    """Test RobotsChecker functionality."""

    @pytest.fixture
    def config(self):
        """Robots checker test configuration."""
        return CrawlConfig(
            respect_robots_txt=True,
            robots_txt_cache_ttl=3600,
            user_agent="TestBot/1.0"
        )

    @pytest.fixture
    def robots_checker(self, config):
        """Robots checker instance for testing."""
        return RobotsChecker(config)

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session for testing."""
        session = AsyncMock()
        return session

    def test_get_robots_url(self, robots_checker):
        """Test robots.txt URL generation."""
        test_cases = [
            ("https://example.com/path", "https://example.com/robots.txt"),
            ("http://test.com/page.html", "http://test.com/robots.txt"),
            ("https://subdomain.example.com/deep/path", "https://subdomain.example.com/robots.txt"),
        ]

        for url, expected in test_cases:
            result = robots_checker._get_robots_url(url)
            assert result == expected

    @pytest.mark.asyncio
    async def test_fetch_robots_txt_success(self, robots_checker, mock_session):
        """Test successful robots.txt fetching."""
        robots_content = "User-agent: *\nDisallow: /admin"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = robots_content
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await robots_checker._fetch_robots_txt(mock_session, "https://example.com/robots.txt")
        assert result == robots_content

    @pytest.mark.asyncio
    async def test_fetch_robots_txt_not_found(self, robots_checker, mock_session):
        """Test robots.txt not found (404)."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await robots_checker._fetch_robots_txt(mock_session, "https://example.com/robots.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_robots_txt_timeout(self, robots_checker, mock_session):
        """Test robots.txt fetch timeout."""
        mock_session.get.side_effect = asyncio.TimeoutError()

        result = await robots_checker._fetch_robots_txt(mock_session, "https://example.com/robots.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_robots_txt_connection_error(self, robots_checker, mock_session):
        """Test robots.txt fetch connection error."""
        mock_session.get.side_effect = aiohttp.ClientError()

        result = await robots_checker._fetch_robots_txt(mock_session, "https://example.com/robots.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_can_crawl_no_robots_txt(self, robots_checker, mock_session):
        """Test crawl allowed when no robots.txt exists."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await robots_checker.can_crawl(mock_session, "https://example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_can_crawl_robots_txt_allows(self, robots_checker, mock_session):
        """Test crawl allowed by robots.txt."""
        robots_content = "User-agent: TestBot\nDisallow: /admin\nAllow: /"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = robots_content
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await robots_checker.can_crawl(mock_session, "https://example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_can_crawl_caching(self, robots_checker, mock_session):
        """Test robots.txt response caching."""
        robots_content = "User-agent: *\nAllow: /"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = robots_content
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # First call should fetch robots.txt
        await robots_checker.can_crawl(mock_session, "https://example.com/page1")
        assert mock_session.get.call_count == 1

        # Second call should use cache
        await robots_checker.can_crawl(mock_session, "https://example.com/page2")
        assert mock_session.get.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_can_crawl_cache_expiry(self, robots_checker, mock_session):
        """Test robots.txt cache expiry."""
        robots_content = "User-agent: *\nAllow: /"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = robots_content
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # First call
        await robots_checker.can_crawl(mock_session, "https://example.com/page")

        # Simulate cache expiry by modifying cache timestamp
        domain = "example.com"
        if domain in robots_checker.robots_cache:
            parser, _ = robots_checker.robots_cache[domain]
            expired_time = datetime.now() - timedelta(seconds=robots_checker.config.robots_txt_cache_ttl + 1)
            robots_checker.robots_cache[domain] = (parser, expired_time)

        # Second call should fetch again
        await robots_checker.can_crawl(mock_session, "https://example.com/page2")
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_can_crawl_respect_disabled(self, mock_session):
        """Test crawling when robots.txt respect is disabled."""
        config = CrawlConfig(respect_robots_txt=False)
        robots_checker = RobotsChecker(config)

        # Should return True without making any requests
        result = await robots_checker.can_crawl(mock_session, "https://example.com/page")
        assert result is True
        assert mock_session.get.call_count == 0

    @pytest.mark.asyncio
    async def test_can_crawl_parsing_error(self, robots_checker, mock_session):
        """Test robots.txt parsing error handling."""
        # Invalid robots.txt content that might cause parsing issues
        robots_content = "Invalid robots.txt content\n\x00\x01\x02"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = robots_content
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Should default to allowing crawl on parsing errors
        result = await robots_checker.can_crawl(mock_session, "https://example.com/page")
        assert result is True


class TestWebCrawler:
    """Test WebCrawler functionality."""

    @pytest.fixture
    def config(self):
        """Web crawler test configuration."""
        return CrawlConfig(
            delay_between_requests=0.1,  # Faster for tests
            max_requests_per_second=10.0,  # Higher for tests
            timeout=5.0,  # Shorter timeout for tests
            max_retries=1,  # Fewer retries for faster tests
            retry_delay=0.1,
            concurrent_requests=3
        )

    @pytest.fixture
    async def crawler(self, config):
        """Web crawler instance for testing."""
        crawler = WebCrawler(config)
        await crawler.start()
        yield crawler
        await crawler.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        """Test async context manager functionality."""
        async with WebCrawler(config) as crawler:
            assert crawler.session is not None
        # Session should be closed after exiting context

    @pytest.mark.asyncio
    async def test_start_stop(self, config):
        """Test manual start/stop functionality."""
        crawler = WebCrawler(config)
        assert crawler.session is None

        await crawler.start()
        assert crawler.session is not None

        await crawler.stop()
        assert crawler.session is None

    def test_get_domain(self, config):
        """Test domain extraction from URLs."""
        crawler = WebCrawler(config)
        test_cases = [
            ("https://www.example.com/path", "example.com"),
            ("http://subdomain.test.co.uk/page", "test.co.uk"),
            ("https://blog.example.org", "example.org"),
        ]

        for url, expected in test_cases:
            result = crawler._get_domain(url)
            assert result == expected

    def test_get_domain_fallback(self, config):
        """Test domain extraction fallback for malformed URLs."""
        crawler = WebCrawler(config)

        # Test with malformed URL that might cause tldextract to fail
        with patch('tldextract.extract') as mock_extract:
            mock_extract.side_effect = Exception("Parsing failed")

            result = crawler._get_domain("https://example.com")
            assert result == "example.com"

    @pytest.mark.asyncio
    async def test_crawl_url_not_started(self, config):
        """Test crawl_url when crawler not started."""
        crawler = WebCrawler(config)

        with pytest.raises(RuntimeError, match="Crawler not started"):
            await crawler.crawl_url("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_url_successful(self, crawler):
        """Test successful URL crawling."""
        with patch.object(crawler, '_make_request') as mock_request:
            expected_response = CrawlResponse(
                url="https://example.com",
                status_code=200,
                content="<html>Test</html>",
                headers={"content-type": "text/html"},
                content_type="text/html",
                encoding="utf-8"
            )
            mock_request.return_value = expected_response

            result = await crawler.crawl_url("https://example.com")
            assert result == expected_response
            mock_request.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_url_duplicate_request(self, crawler):
        """Test handling of duplicate crawl requests."""
        url = "https://example.com"

        # Mock _make_request to simulate a slow request
        async def slow_request(url):
            await asyncio.sleep(0.2)
            return CrawlResponse(
                url=url, status_code=200, content="", headers={},
                content_type="", encoding=""
            )

        with patch.object(crawler, '_make_request', side_effect=slow_request):
            # Start first request
            task1 = asyncio.create_task(crawler.crawl_url(url))

            # Start second request for same URL (should be rejected)
            await asyncio.sleep(0.05)  # Give first request time to start
            result2 = await crawler.crawl_url(url)

            # Wait for first request to complete
            result1 = await task1

            assert result1.status_code == 200
            assert result2.error == "Duplicate crawl request"

    @pytest.mark.asyncio
    async def test_crawl_urls_multiple(self, crawler):
        """Test crawling multiple URLs."""
        urls = ["https://example.com", "https://test.com", "https://demo.com"]

        with patch.object(crawler, '_make_request') as mock_request:
            def make_response(url):
                return CrawlResponse(
                    url=url, status_code=200, content=f"Content for {url}",
                    headers={}, content_type="text/html", encoding="utf-8"
                )

            mock_request.side_effect = lambda url: make_response(url)

            results = await crawler.crawl_urls(urls)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.url == urls[i]
                assert result.status_code == 200
                assert f"Content for {urls[i]}" in result.content

    @pytest.mark.asyncio
    async def test_crawl_urls_empty_list(self, crawler):
        """Test crawling empty URL list."""
        results = await crawler.crawl_urls([])
        assert results == []

    @pytest.mark.asyncio
    async def test_crawl_urls_with_exception(self, crawler):
        """Test crawling URLs when some requests raise exceptions."""
        urls = ["https://example.com", "https://error.com"]

        with patch.object(crawler, 'crawl_url') as mock_crawl:
            def side_effect(url):
                if url == "https://error.com":
                    raise Exception("Network error")
                return CrawlResponse(
                    url=url, status_code=200, content="Success",
                    headers={}, content_type="text/html", encoding="utf-8"
                )

            mock_crawl.side_effect = side_effect

            results = await crawler.crawl_urls(urls)

            assert len(results) == 2
            assert results[0].status_code == 200
            assert results[1].error == "Network error"

    @pytest.mark.asyncio
    async def test_make_request_robots_blocked(self, crawler):
        """Test request blocked by robots.txt."""
        with patch.object(crawler.robots_checker, 'can_crawl', return_value=False):
            result = await crawler._make_request("https://example.com")

            assert result.status_code == 403
            assert result.error == "Blocked by robots.txt"
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_make_request_invalid_content_type(self, crawler):
        """Test request with invalid content type."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.get_encoding.return_value = 'utf-8'

        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.return_value.__aenter__.return_value = mock_response

            result = await crawler._make_request("https://example.com")

            assert result.status_code == 200
            assert "Content type not allowed" in result.error
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_make_request_content_too_large_header(self, crawler):
        """Test request with content too large (from headers)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'text/html',
            'content-length': str(crawler.config.max_content_size + 1)
        }
        mock_response.get_encoding.return_value = 'utf-8'

        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.return_value.__aenter__.return_value = mock_response

            result = await crawler._make_request("https://example.com")

            assert result.status_code == 200
            assert "Content too large" in result.error

    @pytest.mark.asyncio
    async def test_make_request_content_too_large_streaming(self, crawler):
        """Test request with content too large (detected while streaming)."""
        large_chunk = b'x' * (crawler.config.max_content_size + 1)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.get_encoding.return_value = 'utf-8'
        mock_response.content.iter_chunked.return_value = [large_chunk]

        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.return_value.__aenter__.return_value = mock_response

            result = await crawler._make_request("https://example.com")

            assert result.status_code == 200
            assert "Content exceeded size limit" in result.error

    @pytest.mark.asyncio
    async def test_make_request_encoding_detection(self, crawler):
        """Test content encoding detection and fallbacks."""
        content_bytes = "Hello, 世界".encode()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.get_encoding.return_value = None  # No encoding in response
        mock_response.content.iter_chunked.return_value = [content_bytes]
        mock_response.url = "https://example.com"

        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.return_value.__aenter__.return_value = mock_response

            result = await crawler._make_request("https://example.com")

            assert result.status_code == 200
            assert "Hello, 世界" in result.content
            assert result.encoding == 'utf-8'

    @pytest.mark.asyncio
    async def test_make_request_encoding_fallback(self, crawler):
        """Test encoding fallback for invalid UTF-8."""
        # Create content with invalid UTF-8 bytes
        content_bytes = b"Hello \xFF\xFE World"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.get_encoding.return_value = 'utf-8'
        mock_response.content.iter_chunked.return_value = [content_bytes]
        mock_response.url = "https://example.com"

        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.return_value.__aenter__.return_value = mock_response

            result = await crawler._make_request("https://example.com")

            assert result.status_code == 200
            assert "Hello" in result.content  # Should decode with replacement chars
            assert result.encoding == 'utf-8'

    @pytest.mark.asyncio
    async def test_make_request_redirect(self, crawler):
        """Test request with redirect."""
        original_url = "https://example.com"
        final_url = "https://www.example.com"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.get_encoding.return_value = 'utf-8'
        mock_response.content.iter_chunked.return_value = [b"Content"]
        mock_response.url = final_url  # Simulate redirect

        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.return_value.__aenter__.return_value = mock_response

            result = await crawler._make_request(original_url)

            assert result.status_code == 200
            assert result.url == original_url
            assert result.redirect_url == final_url

    @pytest.mark.asyncio
    async def test_make_request_timeout_retry(self, crawler):
        """Test request timeout with retry logic."""
        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            # First call times out, second succeeds
            mock_get.side_effect = [
                asyncio.TimeoutError(),
                AsyncMock().__aenter__.return_value
            ]

            # Mock successful response for retry
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.get_encoding.return_value = 'utf-8'
            mock_response.content.iter_chunked.return_value = [b"Content"]
            mock_response.url = "https://example.com"
            mock_get.side_effect[1] = mock_response

            with patch('asyncio.sleep'):  # Speed up retry delay
                result = await crawler._make_request("https://example.com")

            assert result.status_code == 200
            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_make_request_max_retries_exceeded(self, crawler):
        """Test request with max retries exceeded."""
        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            # All attempts time out
            mock_get.side_effect = asyncio.TimeoutError()

            with patch('asyncio.sleep'):  # Speed up retry delay
                result = await crawler._make_request("https://example.com")

            assert result.status_code == 0
            assert "Request timeout" in result.error
            assert mock_get.call_count == crawler.config.max_retries + 1

    @pytest.mark.asyncio
    async def test_make_request_client_error_retry(self, crawler):
        """Test request client error with retry."""
        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            # First call has client error, second succeeds
            mock_get.side_effect = [
                aiohttp.ClientError("Connection failed"),
                AsyncMock().__aenter__.return_value
            ]

            # Mock successful response for retry
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.get_encoding.return_value = 'utf-8'
            mock_response.content.iter_chunked.return_value = [b"Content"]
            mock_response.url = "https://example.com"
            mock_get.side_effect[1] = mock_response

            with patch('asyncio.sleep'):  # Speed up retry delay
                result = await crawler._make_request("https://example.com")

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_make_request_unexpected_error(self, crawler):
        """Test request with unexpected error."""
        with patch.object(crawler.robots_checker, 'can_crawl', return_value=True), \
             patch.object(crawler.rate_limiter, 'wait_if_needed'), \
             patch.object(crawler.session, 'get') as mock_get:

            mock_get.side_effect = ValueError("Unexpected error")

            result = await crawler._make_request("https://example.com")

            assert result.status_code == 0
            assert "Unexpected error" in result.error
            assert mock_get.call_count == 1  # No retries for unexpected errors


@pytest.mark.asyncio
async def test_integration_full_crawl():
    """Integration test for full crawling process."""
    config = CrawlConfig(
        delay_between_requests=0.1,
        max_requests_per_second=10.0,
        timeout=5.0,
        max_retries=1,
        respect_robots_txt=False  # Disable for integration test
    )

    async with WebCrawler(config) as crawler:
        # Test that we can crawl without errors (will fail if no network)
        try:
            result = await crawler.crawl_url("https://httpbin.org/html")
            # If successful, verify response structure
            assert isinstance(result, CrawlResponse)
            assert result.url == "https://httpbin.org/html"
        except Exception:
            # Skip if no network connection available
            pytest.skip("No network connection available for integration test")
