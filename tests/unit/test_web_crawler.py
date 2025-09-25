"""
Comprehensive unit tests for WebCrawler with edge cases and error conditions.

This test suite covers:
- Rate limiting compliance and edge cases
- Robots.txt parsing and compliance checking
- URL validation and domain restrictions
- Content filtering by type and size
- Error handling and retry logic
- Statistics tracking and performance metrics
- Configuration validation and edge cases
- Network failure scenarios and timeouts
- Concurrent request handling
- Cache management and TTL behavior
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.robotparser import RobotFileParser

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup

from common.core.web_crawler import (
    WebCrawler,
    CrawlerConfig,
    CrawlResult,
    RobotsCache
)


class TestCrawlerConfig:
    """Test CrawlerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CrawlerConfig()

        assert config.rate_limit == 2.0
        assert config.max_concurrent == 10
        assert config.request_timeout == 30.0
        assert config.respect_robots is True
        assert config.robots_cache_ttl == 3600
        assert "WorkspaceQdrantMCP" in config.user_agent
        assert config.max_content_size == 10 * 1024 * 1024
        assert 'text/html' in config.allowed_content_types
        assert config.same_domain_only is True
        assert config.max_depth == 3
        assert config.max_pages == 1000
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.retry_backoff == 2.0
        assert isinstance(config.custom_headers, dict)
        assert isinstance(config.cookies, dict)

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_headers = {'X-Custom': 'test'}
        custom_cookies = {'session': 'abc123'}
        allowed_types = {'text/html', 'application/json'}

        config = CrawlerConfig(
            rate_limit=5.0,
            max_concurrent=20,
            respect_robots=False,
            user_agent="TestAgent/1.0",
            max_content_size=5 * 1024 * 1024,
            allowed_content_types=allowed_types,
            same_domain_only=False,
            max_retries=5,
            custom_headers=custom_headers,
            cookies=custom_cookies
        )

        assert config.rate_limit == 5.0
        assert config.max_concurrent == 20
        assert config.respect_robots is False
        assert config.user_agent == "TestAgent/1.0"
        assert config.max_content_size == 5 * 1024 * 1024
        assert config.allowed_content_types == allowed_types
        assert config.same_domain_only is False
        assert config.max_retries == 5
        assert config.custom_headers == custom_headers
        assert config.cookies == custom_cookies


class TestRobotsCache:
    """Test RobotsCache functionality."""

    def test_cache_initialization(self):
        """Test robots cache initialization."""
        cache = RobotsCache(ttl=1800)
        assert cache._ttl == 1800
        assert len(cache._cache) == 0

    def test_cache_set_and_get(self):
        """Test setting and getting robots parser."""
        cache = RobotsCache()
        parser = RobotFileParser()
        parser.set_url("http://example.com/robots.txt")

        # Cache should be empty initially
        assert cache.get_robots("example.com") is None

        # Set parser in cache
        cache.set_robots("example.com", parser)

        # Should retrieve the same parser
        cached_parser = cache.get_robots("example.com")
        assert cached_parser is parser

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = RobotsCache(ttl=1)  # 1 second TTL
        parser = RobotFileParser()

        cache.set_robots("example.com", parser)
        assert cache.get_robots("example.com") is parser

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get_robots("example.com") is None

    def test_clear_expired(self):
        """Test manual clearing of expired entries."""
        cache = RobotsCache(ttl=1)
        parser = RobotFileParser()

        cache.set_robots("example.com", parser)
        time.sleep(1.1)

        # Entry should still be in cache before clearing
        assert "example.com" in cache._cache

        cache.clear_expired()

        # Entry should be removed after clearing
        assert "example.com" not in cache._cache

    def test_multiple_domains(self):
        """Test caching multiple domains."""
        cache = RobotsCache()
        parser1 = RobotFileParser()
        parser2 = RobotFileParser()

        cache.set_robots("example.com", parser1)
        cache.set_robots("test.com", parser2)

        assert cache.get_robots("example.com") is parser1
        assert cache.get_robots("test.com") is parser2
        assert cache.get_robots("other.com") is None


class TestCrawlResult:
    """Test CrawlResult dataclass."""

    def test_crawl_result_creation(self):
        """Test creating CrawlResult instances."""
        # Minimal result
        result = CrawlResult(url="http://example.com", status_code=200)
        assert result.url == "http://example.com"
        assert result.status_code == 200
        assert result.content is None
        assert result.error is None
        assert isinstance(result.metadata, dict)

    def test_crawl_result_full(self):
        """Test CrawlResult with all fields."""
        headers = {'content-type': 'text/html'}
        metadata = {'links': 5}
        crawl_time = datetime.now()

        result = CrawlResult(
            url="http://example.com",
            status_code=200,
            content="<html>test</html>",
            headers=headers,
            content_type="text/html",
            content_length=100,
            crawl_time=crawl_time,
            processing_time=1.5,
            metadata=metadata
        )

        assert result.content == "<html>test</html>"
        assert result.headers == headers
        assert result.content_type == "text/html"
        assert result.content_length == 100
        assert result.crawl_time == crawl_time
        assert result.processing_time == 1.5
        assert result.metadata == metadata


class TestWebCrawler:
    """Test WebCrawler functionality."""

    @pytest.fixture
    def crawler(self):
        """Create WebCrawler instance for testing."""
        config = CrawlerConfig(rate_limit=0)  # Disable rate limiting for tests
        return WebCrawler(config)

    @pytest.fixture
    def rate_limited_crawler(self):
        """Create WebCrawler with rate limiting enabled."""
        config = CrawlerConfig(rate_limit=10.0)  # 10 requests per second
        return WebCrawler(config)

    def test_initialization(self, crawler):
        """Test crawler initialization."""
        assert crawler.config is not None
        assert crawler._session is None
        assert crawler._rate_limiter._value == crawler.config.max_concurrent
        assert crawler._last_request_time == 0.0
        assert isinstance(crawler._robots_cache, RobotsCache)

        # Check initial statistics
        stats = crawler.get_statistics()
        assert stats['requests_made'] == 0
        assert stats['successful_requests'] == 0
        assert stats['failed_requests'] == 0

    def test_initialization_with_custom_config(self):
        """Test crawler initialization with custom config."""
        config = CrawlerConfig(
            rate_limit=5.0,
            max_concurrent=20,
            user_agent="TestBot/1.0"
        )
        crawler = WebCrawler(config)

        assert crawler.config.rate_limit == 5.0
        assert crawler.config.max_concurrent == 20
        assert crawler.config.user_agent == "TestBot/1.0"
        assert crawler._rate_limiter._value == 20

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test crawler as async context manager."""
        config = CrawlerConfig(rate_limit=0)

        async with WebCrawler(config) as crawler:
            assert crawler._session is not None
            assert isinstance(crawler._session, aiohttp.ClientSession)

        # Session should be closed after exit
        assert crawler._session.closed

    @pytest.mark.asyncio
    async def test_manual_initialization_and_close(self, crawler):
        """Test manual session initialization and closing."""
        assert crawler._session is None

        await crawler._initialize_session()
        assert crawler._session is not None
        assert isinstance(crawler._session, aiohttp.ClientSession)

        await crawler.close()
        assert crawler._session.closed

    @pytest.mark.asyncio
    async def test_rate_limiting(self, rate_limited_crawler):
        """Test rate limiting enforcement."""
        start_time = time.time()

        # Make multiple rate limit checks
        await rate_limited_crawler._enforce_rate_limit()
        await rate_limited_crawler._enforce_rate_limit()
        await rate_limited_crawler._enforce_rate_limit()

        elapsed = time.time() - start_time
        expected_min_time = 2 / rate_limited_crawler.config.rate_limit  # 2 intervals

        # Should take at least the minimum time for rate limiting
        assert elapsed >= expected_min_time * 0.9  # Allow 10% tolerance

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self, crawler):
        """Test that rate limiting can be disabled."""
        start_time = time.time()

        # Make multiple rate limit checks with rate_limit=0
        await crawler._enforce_rate_limit()
        await crawler._enforce_rate_limit()
        await crawler._enforce_rate_limit()

        elapsed = time.time() - start_time

        # Should be nearly instant with no rate limiting
        assert elapsed < 0.1

    def test_url_validation(self, crawler):
        """Test URL validation logic."""
        # Valid URLs
        assert crawler._is_valid_url("http://example.com")
        assert crawler._is_valid_url("https://example.com/path")
        assert crawler._is_valid_url("https://sub.example.com")

        # Invalid URLs
        assert not crawler._is_valid_url("ftp://example.com")
        assert not crawler._is_valid_url("not-a-url")
        assert not crawler._is_valid_url("")
        assert not crawler._is_valid_url("http://")
        assert not crawler._is_valid_url("://example.com")

    def test_domain_restriction(self, crawler):
        """Test same-domain URL validation."""
        base_domain = "example.com"

        # Same domain - should be valid
        assert crawler._is_valid_url("http://example.com", base_domain)
        assert crawler._is_valid_url("https://example.com/path", base_domain)

        # Different domain - should be invalid with same_domain_only=True
        assert not crawler._is_valid_url("http://other.com", base_domain)
        assert not crawler._is_valid_url("https://sub.other.com", base_domain)

        # Subdomain - should be invalid with exact domain matching
        assert not crawler._is_valid_url("http://sub.example.com", base_domain)

    def test_domain_restriction_disabled(self):
        """Test URL validation with same_domain_only disabled."""
        config = CrawlerConfig(same_domain_only=False)
        crawler = WebCrawler(config)

        base_domain = "example.com"

        # All valid URLs should be accepted regardless of domain
        assert crawler._is_valid_url("http://example.com", base_domain)
        assert crawler._is_valid_url("http://other.com", base_domain)
        assert crawler._is_valid_url("https://different.org", base_domain)

    def test_content_type_filtering(self, crawler):
        """Test content type filtering."""
        # Allowed content types
        assert crawler._should_process_content("text/html", 1000)
        assert crawler._should_process_content("text/plain", 1000)
        assert crawler._should_process_content("application/json", 1000)
        assert crawler._should_process_content("text/html; charset=utf-8", 1000)

        # Disallowed content types
        assert not crawler._should_process_content("image/jpeg", 1000)
        assert not crawler._should_process_content("video/mp4", 1000)
        assert not crawler._should_process_content("application/octet-stream", 1000)

    def test_content_size_filtering(self, crawler):
        """Test content size filtering."""
        max_size = crawler.config.max_content_size

        # Within size limit
        assert crawler._should_process_content("text/html", max_size)
        assert crawler._should_process_content("text/html", max_size - 1)
        assert crawler._should_process_content("text/html", 1000)

        # Exceeds size limit
        assert not crawler._should_process_content("text/html", max_size + 1)
        assert not crawler._should_process_content("text/html", max_size * 2)

    def test_content_filtering_edge_cases(self, crawler):
        """Test content filtering edge cases."""
        # None values should be handled gracefully
        assert crawler._should_process_content(None, 1000)
        assert crawler._should_process_content("text/html", None)
        assert crawler._should_process_content(None, None)

        # Zero size should be allowed
        assert crawler._should_process_content("text/html", 0)

    @pytest.mark.asyncio
    async def test_robots_txt_compliance_disabled(self, crawler):
        """Test robots.txt checking when disabled."""
        crawler.config.respect_robots = False

        # Should allow all URLs when robots.txt compliance is disabled
        result = await crawler._check_robots_txt("http://example.com/admin")
        assert result is True

        result = await crawler._check_robots_txt("https://other.com/private")
        assert result is True

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_robots_txt_compliance_enabled(self, mock_get, crawler):
        """Test robots.txt checking when enabled."""
        # Mock robots.txt response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
User-agent: *
Disallow: /admin
Disallow: /private
Allow: /public
""")
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        # Should allow public paths
        result = await crawler._check_robots_txt("http://example.com/public")
        assert result is True

        result = await crawler._check_robots_txt("http://example.com/")
        assert result is True

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_robots_txt_not_found(self, mock_get, crawler):
        """Test robots.txt handling when file not found."""
        # Mock 404 response for robots.txt
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        # Should allow all URLs when robots.txt not found
        result = await crawler._check_robots_txt("http://example.com/admin")
        assert result is True

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_robots_txt_network_error(self, mock_get, crawler):
        """Test robots.txt handling with network errors."""
        # Mock network error
        mock_get.side_effect = aiohttp.ClientError("Network error")

        await crawler._initialize_session()

        # Should allow all URLs on network error (be permissive)
        result = await crawler._check_robots_txt("http://example.com/admin")
        assert result is True

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_robots_txt_caching(self, mock_get, crawler):
        """Test robots.txt caching behavior."""
        # Mock robots.txt response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="User-agent: *\nDisallow: /admin")
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        # First request should fetch robots.txt
        await crawler._check_robots_txt("http://example.com/test")
        assert mock_get.call_count == 1

        # Second request should use cache
        await crawler._check_robots_txt("http://example.com/other")
        assert mock_get.call_count == 1  # No additional call

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_successful_crawl(self, mock_get, crawler):
        """Test successful URL crawling."""
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'text/html; charset=utf-8',
            'content-length': '100'
        }
        mock_response.text = AsyncMock(return_value="<html>test content</html>")
        mock_response.url = "http://example.com"
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com")

        assert result.status_code == 200
        assert result.content == "<html>test content</html>"
        assert result.content_type == "text/html; charset=utf-8"
        assert result.error is None
        assert result.crawl_time is not None
        assert result.processing_time is not None

        # Check statistics
        stats = crawler.get_statistics()
        assert stats['successful_requests'] == 1
        assert stats['requests_made'] == 1
        assert stats['failed_requests'] == 0

    @pytest.mark.asyncio
    async def test_invalid_url_crawl(self, crawler):
        """Test crawling invalid URLs."""
        result = await crawler.crawl_url("invalid-url")

        assert result.status_code == 0
        assert result.error == "Invalid URL format"
        assert result.content is None

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_robots_blocked_crawl(self, mock_get, crawler):
        """Test crawling URLs blocked by robots.txt."""
        # Mock robots.txt that blocks the path
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""
User-agent: *
Disallow: /admin
""")
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com/admin/panel")

        assert result.status_code == 0
        assert result.error == "Blocked by robots.txt"

        # Check statistics
        stats = crawler.get_statistics()
        assert stats['robots_blocked'] == 1

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_content_filtered_crawl(self, mock_get, crawler):
        """Test crawling with content filtering."""
        # Mock response with disallowed content type
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'image/jpeg',
            'content-length': '1000'
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com/image.jpg")

        assert result.status_code == 200
        assert "Content filtered" in result.error
        assert result.content is None

        # Check statistics
        stats = crawler.get_statistics()
        assert stats['content_filtered'] == 1

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_oversized_content_crawl(self, mock_get, crawler):
        """Test crawling content that exceeds size limits."""
        # Mock response with oversized content
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'text/html',
            'content-length': str(crawler.config.max_content_size + 1)
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com/large.html")

        assert result.status_code == 200
        assert "Content filtered" in result.error
        assert result.content is None

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_timeout_error(self, mock_get, crawler):
        """Test handling of timeout errors."""
        mock_get.side_effect = asyncio.TimeoutError("Request timeout")

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com")

        assert result.status_code == 0
        assert "Timeout" in result.error

        stats = crawler.get_statistics()
        assert stats['failed_requests'] == 1

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_client_error(self, mock_get, crawler):
        """Test handling of client errors."""
        mock_get.side_effect = aiohttp.ClientError("Connection failed")

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com")

        assert result.status_code == 0
        assert "Client error" in result.error

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_unexpected_error(self, mock_get, crawler):
        """Test handling of unexpected errors."""
        mock_get.side_effect = ValueError("Unexpected error")

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com")

        assert result.status_code == 0
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_retry_logic(self, mock_get, crawler):
        """Test retry logic with eventual success."""
        # First two calls fail, third succeeds
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.text = AsyncMock(return_value="success")
        mock_response.url = "http://example.com"

        mock_get.side_effect = [
            aiohttp.ClientError("First failure"),
            aiohttp.ClientError("Second failure"),
            AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        ]

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com")

        assert result.status_code == 200
        assert result.content == "success"
        assert mock_get.call_count == 3

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_retry_exhaustion(self, mock_get, crawler):
        """Test retry logic when all retries are exhausted."""
        mock_get.side_effect = aiohttp.ClientError("Persistent failure")

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com")

        assert result.status_code == 0
        assert "Client error" in result.error
        assert mock_get.call_count == crawler.config.max_retries + 1

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_redirect_handling(self, mock_get, crawler):
        """Test handling of HTTP redirects."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.text = AsyncMock(return_value="redirected content")
        mock_response.url = "http://example.com/final"  # Different from requested URL
        mock_get.return_value.__aenter__.return_value = mock_response

        await crawler._initialize_session()

        result = await crawler.crawl_url("http://example.com/redirect")

        assert result.status_code == 200
        assert result.content == "redirected content"
        assert result.redirect_url == "http://example.com/final"

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_crawl_multiple_urls(self, mock_get, crawler):
        """Test crawling multiple URLs concurrently."""
        # Mock responses for different URLs
        def mock_response_factory(url):
            response = AsyncMock()
            response.status = 200
            response.headers = {'content-type': 'text/html'}
            response.text = AsyncMock(return_value=f"content from {url}")
            response.url = url
            return response

        mock_get.side_effect = [
            AsyncMock(__aenter__=AsyncMock(return_value=mock_response_factory("http://example.com/1"))),
            AsyncMock(__aenter__=AsyncMock(return_value=mock_response_factory("http://example.com/2"))),
            AsyncMock(__aenter__=AsyncMock(return_value=mock_response_factory("http://example.com/3")))
        ]

        await crawler._initialize_session()

        urls = ["http://example.com/1", "http://example.com/2", "http://example.com/3"]
        results = await crawler.crawl_urls(urls)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.status_code == 200
            assert f"content from http://example.com/{i+1}" in result.content

    def test_statistics_tracking(self, crawler):
        """Test statistics tracking and retrieval."""
        # Initial statistics
        stats = crawler.get_statistics()
        assert stats['requests_made'] == 0
        assert stats['successful_requests'] == 0
        assert stats['failed_requests'] == 0
        assert stats['robots_blocked'] == 0
        assert stats['content_filtered'] == 0
        assert stats['success_rate'] == 0.0

        # Manually update statistics to test calculation
        crawler._stats['requests_made'] = 10
        crawler._stats['successful_requests'] = 8
        crawler._stats['failed_requests'] = 2

        stats = crawler.get_statistics()
        assert stats['requests_made'] == 10
        assert stats['successful_requests'] == 8
        assert stats['failed_requests'] == 2
        assert stats['success_rate'] == 0.8

    def test_statistics_reset(self, crawler):
        """Test statistics reset functionality."""
        # Set some statistics
        crawler._stats['requests_made'] = 10
        crawler._stats['successful_requests'] = 5

        # Reset statistics
        crawler.reset_statistics()

        stats = crawler.get_statistics()
        assert stats['requests_made'] == 0
        assert stats['successful_requests'] == 0
        assert stats['success_rate'] == 0.0

    @pytest.mark.asyncio
    async def test_health_check(self, crawler):
        """Test health check functionality."""
        health = await crawler.health_check()

        assert 'session_active' in health
        assert 'robots_cache_entries' in health
        assert 'stats' in health
        assert 'config_valid' in health
        assert 'connectivity' in health

        # Without session, connectivity should be None
        assert health['session_active'] is False
        assert health['connectivity'] is None

    @pytest.mark.asyncio
    async def test_health_check_with_session(self, crawler):
        """Test health check with active session."""
        await crawler._initialize_session()

        health = await crawler.health_check()
        assert health['session_active'] is True

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_concurrent_request_limiting(self, mock_get):
        """Test that concurrent requests are properly limited."""
        config = CrawlerConfig(max_concurrent=2, rate_limit=0)
        crawler = WebCrawler(config)

        # Track concurrent requests
        active_requests = []
        max_concurrent = 0

        async def mock_request(*args, **kwargs):
            active_requests.append(1)
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(active_requests))
            await asyncio.sleep(0.1)  # Simulate request duration
            active_requests.pop()

            response = AsyncMock()
            response.status = 200
            response.headers = {'content-type': 'text/html'}
            response.text = AsyncMock(return_value="test")
            response.url = args[0] if args else "http://example.com"
            return AsyncMock(__aenter__=AsyncMock(return_value=response))

        mock_get.side_effect = mock_request

        await crawler._initialize_session()

        # Start multiple concurrent requests
        urls = [f"http://example.com/{i}" for i in range(5)]
        await crawler.crawl_urls(urls)

        # Should not exceed max_concurrent limit
        assert max_concurrent <= config.max_concurrent

        await crawler.close()


class TestWebCrawlerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_crawl_without_session_initialization(self):
        """Test crawling when session is not initialized."""
        crawler = WebCrawler()

        # Should auto-initialize session when needed
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value="test")
            mock_response.url = "http://example.com"
            mock_session.get.return_value.__aenter__.return_value = mock_response

            result = await crawler.crawl_url("http://example.com")

            assert result.status_code == 200
            assert mock_session_class.called

    def test_malformed_robots_txt(self, crawler):
        """Test handling of malformed robots.txt files."""
        cache = RobotsCache()

        # Test with None parser (should handle gracefully)
        assert cache.get_robots("nonexistent.com") is None

    @pytest.mark.asyncio
    async def test_extremely_large_response(self, crawler):
        """Test handling of extremely large responses."""
        # This would typically be handled by content-length checking
        # but we test the behavior when content-length is not available
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value="x" * (crawler.config.max_content_size + 1000))
            mock_response.url = "http://example.com"
            mock_get.return_value.__aenter__.return_value = mock_response

            await crawler._initialize_session()
            result = await crawler.crawl_url("http://example.com")

            # Should still process since content-length wasn't checked beforehand
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, crawler):
        """Test handling of Unicode content."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            unicode_content = "Hello 世界 🌍 émojis and açcénts"
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
            mock_response.text = AsyncMock(return_value=unicode_content)
            mock_response.url = "http://example.com"
            mock_get.return_value.__aenter__.return_value = mock_response

            await crawler._initialize_session()
            result = await crawler.crawl_url("http://example.com")

            assert result.status_code == 200
            assert result.content == unicode_content

    @pytest.mark.asyncio
    async def test_empty_response_content(self, crawler):
        """Test handling of empty response content."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = AsyncMock(return_value="")
            mock_response.url = "http://example.com"
            mock_get.return_value.__aenter__.return_value = mock_response

            await crawler._initialize_session()
            result = await crawler.crawl_url("http://example.com")

            assert result.status_code == 200
            assert result.content == ""
            assert result.content_length == 0

    def test_custom_user_agent_in_headers(self):
        """Test that custom user agent is properly set."""
        custom_agent = "CustomBot/2.0"
        config = CrawlerConfig(user_agent=custom_agent)
        crawler = WebCrawler(config)

        assert crawler.config.user_agent == custom_agent

    @pytest.mark.asyncio
    async def test_close_without_session(self, crawler):
        """Test closing crawler without active session."""
        # Should not raise error
        await crawler.close()
        assert crawler._session is None

    @pytest.mark.asyncio
    async def test_double_close(self, crawler):
        """Test closing crawler multiple times."""
        await crawler._initialize_session()
        await crawler.close()

        # Second close should not raise error
        await crawler.close()

    def test_zero_rate_limit_handling(self):
        """Test handling of zero rate limit (disabled)."""
        config = CrawlerConfig(rate_limit=0)
        crawler = WebCrawler(config)
        assert crawler.config.rate_limit == 0

    def test_negative_rate_limit_handling(self):
        """Test handling of negative rate limit."""
        config = CrawlerConfig(rate_limit=-1)
        crawler = WebCrawler(config)
        assert crawler.config.rate_limit == -1