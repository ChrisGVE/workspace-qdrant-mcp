"""
Comprehensive unit tests for web crawling integration system.

Tests cover:
- IntegratedWebCrawler initialization and configuration
- Single URL crawling with caching integration
- Recursive crawling with depth and domain filtering
- Batch crawling with concurrency control
- Cache integration and duplicate handling
- Error handling and edge cases
- Performance monitoring and statistics
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from workspace_qdrant_mcp.web.integration import (
    CrawlResult,
    CrawlSummary,
    IntegratedCrawlConfig,
    IntegratedWebCrawler,
    crawl_site_recursive,
    crawl_url_simple,
)


class TestCrawlResult:
    """Test CrawlResult data class."""

    def test_crawl_result_creation(self):
        """Test basic CrawlResult creation."""
        result = CrawlResult(
            url="http://example.com",
            title="Example Page",
            content="Test content",
            success=True
        )

        assert result.url == "http://example.com"
        assert result.title == "Example Page"
        assert result.content == "Test content"
        assert result.success is True
        assert result.error is None
        assert result.from_cache is False
        assert result.extracted_links == []
        assert result.metadata == {}

    def test_crawl_result_with_all_fields(self):
        """Test CrawlResult with all fields populated."""
        metadata = {"quality_score": 0.85, "word_count": 100}
        links = ["http://example.com/page1", "http://example.com/page2"]

        result = CrawlResult(
            url="http://example.com",
            title="Example Page",
            content="Test content",
            extracted_links=links,
            metadata=metadata,
            success=True,
            error=None,
            from_cache=True,
            content_hash="abc123",
            duplicate_of="http://original.com"
        )

        assert result.extracted_links == links
        assert result.metadata == metadata
        assert result.from_cache is True
        assert result.content_hash == "abc123"
        assert result.duplicate_of == "http://original.com"

    def test_crawl_result_error_case(self):
        """Test CrawlResult for error scenarios."""
        result = CrawlResult(
            url="http://example.com",
            title="",
            content="",
            success=False,
            error="Connection timeout"
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.title == ""
        assert result.content == ""


class TestCrawlSummary:
    """Test CrawlSummary data class and calculations."""

    def test_crawl_summary_defaults(self):
        """Test CrawlSummary default values."""
        summary = CrawlSummary()

        assert summary.total_urls_discovered == 0
        assert summary.total_urls_crawled == 0
        assert summary.successful_crawls == 0
        assert summary.failed_crawls == 0
        assert summary.cached_hits == 0
        assert summary.duplicates_found == 0
        assert summary.total_content_size == 0
        assert summary.crawl_duration == 0.0
        assert summary.unique_domains == set()
        assert summary.error_summary == {}

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        summary = CrawlSummary()

        # No crawls
        assert summary.success_rate == 0.0

        # Some successes
        summary.total_urls_crawled = 10
        summary.successful_crawls = 8
        assert summary.success_rate == 0.8

        # All successes
        summary.successful_crawls = 10
        assert summary.success_rate == 1.0

        # No successes
        summary.successful_crawls = 0
        assert summary.success_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        summary = CrawlSummary()

        # No crawls
        assert summary.cache_hit_rate == 0.0

        # Some cache hits
        summary.total_urls_crawled = 10
        summary.cached_hits = 3
        assert summary.cache_hit_rate == 0.3

        # All cache hits
        summary.cached_hits = 10
        assert summary.cache_hit_rate == 1.0

        # No cache hits
        summary.cached_hits = 0
        assert summary.cache_hit_rate == 0.0


class TestIntegratedCrawlConfig:
    """Test IntegratedCrawlConfig."""

    def test_config_defaults(self):
        """Test configuration default values."""
        config = IntegratedCrawlConfig()

        assert config.user_agent == "IntegratedWebCrawler/1.0"
        assert config.request_timeout == 30.0
        assert config.max_retries == 3
        assert config.respect_robots is True
        assert config.rate_limit_delay == 1.0
        assert config.min_content_length == 100
        assert config.quality_threshold == 0.5
        assert config.max_depth == 3
        assert config.max_pages_per_domain == 100
        assert config.follow_external_links is False
        assert config.cache_max_size == 100 * 1024 * 1024
        assert config.cache_max_entries == 10000
        assert config.enable_persistence is True
        assert config.similarity_threshold == 0.85
        assert config.max_concurrent_requests == 5

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = IntegratedCrawlConfig(
            user_agent="CustomCrawler/2.0",
            request_timeout=60.0,
            max_retries=5,
            respect_robots=False,
            rate_limit_delay=0.5,
            min_content_length=200,
            quality_threshold=0.7,
            max_depth=5,
            max_pages_per_domain=200,
            follow_external_links=True,
            cache_max_size=200 * 1024 * 1024,
            cache_max_entries=20000,
            enable_persistence=False,
            similarity_threshold=0.9,
            max_concurrent_requests=10
        )

        assert config.user_agent == "CustomCrawler/2.0"
        assert config.request_timeout == 60.0
        assert config.max_retries == 5
        assert config.respect_robots is False
        assert config.rate_limit_delay == 0.5
        assert config.min_content_length == 200
        assert config.quality_threshold == 0.7
        assert config.max_depth == 5
        assert config.max_pages_per_domain == 200
        assert config.follow_external_links is True
        assert config.cache_max_size == 200 * 1024 * 1024
        assert config.cache_max_entries == 20000
        assert config.enable_persistence is False
        assert config.similarity_threshold == 0.9
        assert config.max_concurrent_requests == 10


class TestIntegratedWebCrawler:
    """Test IntegratedWebCrawler functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IntegratedCrawlConfig(
            rate_limit_delay=0.0,  # No delay for testing
            enable_persistence=False,  # No persistence for testing
            max_concurrent_requests=2
        )

    @pytest.fixture
    def crawler(self, config):
        """Create test crawler instance."""
        return IntegratedWebCrawler(config)

    def test_crawler_initialization(self, config):
        """Test crawler initialization."""
        crawler = IntegratedWebCrawler(config)

        assert crawler.config == config
        assert crawler.crawler is not None
        assert crawler.extractor is not None
        assert crawler.link_discovery is not None
        assert crawler.recursive_crawler is not None
        assert crawler.cache is not None
        assert crawler.request_semaphore._value == config.request_semaphore_size

    def test_crawler_initialization_default_config(self):
        """Test crawler initialization with default config."""
        crawler = IntegratedWebCrawler()
        assert isinstance(crawler.config, IntegratedCrawlConfig)

    @pytest.mark.asyncio
    async def test_crawl_url_success(self, crawler):
        """Test successful URL crawling."""
        test_url = "http://example.com"
        test_content = "<html><head><title>Test Page</title></head><body><p>Test content</p></body></html>"

        # Mock the crawler response
        mock_crawl_response = Mock()
        mock_crawl_response.success = True
        mock_crawl_response.content = test_content
        mock_crawl_response.content_type = "text/html"
        mock_crawl_response.timestamp = 1234567890.0

        # Mock the extractor response
        mock_extracted_content = Mock()
        mock_extracted_content.title = "Test Page"
        mock_extracted_content.content = "Test content"
        mock_extracted_content.quality = Mock()
        mock_extracted_content.quality.overall_score = 0.8

        with patch.object(crawler.crawler, 'crawl_url', return_value=mock_crawl_response), \
             patch.object(crawler.extractor, 'extract', return_value=mock_extracted_content), \
             patch.object(crawler.link_discovery, 'extract_links', return_value=[]):

            result = await crawler.crawl_url(test_url)

            assert result.success is True
            assert result.url == test_url
            assert result.title == "Test Page"
            assert result.content == "Test content"
            assert result.from_cache is False
            assert result.error is None

    @pytest.mark.asyncio
    async def test_crawl_url_cache_hit(self, crawler):
        """Test URL crawling with cache hit."""
        test_url = "http://example.com"

        # Mock cached entry
        mock_cached_entry = Mock()
        mock_cached_entry.content = "Cached content"
        mock_cached_entry.content_hash = "cache123"
        mock_cached_entry.metadata = {
            "title": "Cached Page",
            "links": ["http://example.com/link1"],
            "quality_score": 0.9
        }

        with patch.object(crawler.cache, 'get', return_value=mock_cached_entry):
            result = await crawler.crawl_url(test_url)

            assert result.success is True
            assert result.url == test_url
            assert result.title == "Cached Page"
            assert result.content == "Cached content"
            assert result.from_cache is True
            assert result.content_hash == "cache123"
            assert result.extracted_links == ["http://example.com/link1"]

    @pytest.mark.asyncio
    async def test_crawl_url_duplicate_reference(self, crawler):
        """Test URL crawling with duplicate reference in cache."""
        test_url = "http://example.com/duplicate"
        original_url = "http://example.com/original"

        # Mock duplicate cached entry
        mock_duplicate_entry = Mock()
        mock_duplicate_entry.metadata = {"duplicate_of": original_url}

        # Mock original cached entry
        mock_original_entry = Mock()
        mock_original_entry.content = "Original content"
        mock_original_entry.content_hash = "orig123"
        mock_original_entry.metadata = {
            "title": "Original Page",
            "links": ["http://example.com/link1"]
        }

        async def mock_cache_get(url):
            if url == test_url:
                return mock_duplicate_entry
            elif url == original_url:
                return mock_original_entry
            return None

        with patch.object(crawler.cache, 'get', side_effect=mock_cache_get):
            result = await crawler.crawl_url(test_url)

            assert result.success is True
            assert result.url == test_url
            assert result.title == "Original Page"
            assert result.content == "Original content"
            assert result.from_cache is True
            assert result.duplicate_of == original_url

    @pytest.mark.asyncio
    async def test_crawl_url_no_cache(self, crawler):
        """Test URL crawling without using cache."""
        test_url = "http://example.com"
        test_content = "<html><head><title>Test Page</title></head><body>Content</body></html>"

        # Mock responses
        mock_crawl_response = Mock()
        mock_crawl_response.success = True
        mock_crawl_response.content = test_content
        mock_crawl_response.content_type = "text/html"
        mock_crawl_response.timestamp = 1234567890.0

        mock_extracted_content = Mock()
        mock_extracted_content.title = "Test Page"
        mock_extracted_content.content = "Content"
        mock_extracted_content.quality = Mock()
        mock_extracted_content.quality.overall_score = 0.7

        with patch.object(crawler.crawler, 'crawl_url', return_value=mock_crawl_response), \
             patch.object(crawler.extractor, 'extract', return_value=mock_extracted_content), \
             patch.object(crawler.link_discovery, 'extract_links', return_value=[]):

            result = await crawler.crawl_url(test_url, use_cache=False)

            assert result.success is True
            assert result.from_cache is False

    @pytest.mark.asyncio
    async def test_crawl_url_crawler_failure(self, crawler):
        """Test URL crawling when crawler fails."""
        test_url = "http://example.com"

        # Mock failed crawler response
        mock_crawl_response = Mock()
        mock_crawl_response.success = False
        mock_crawl_response.error = "Connection failed"

        with patch.object(crawler.crawler, 'crawl_url', return_value=mock_crawl_response):
            result = await crawler.crawl_url(test_url)

            assert result.success is False
            assert result.error == "Connection failed"
            assert result.url == test_url
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_crawl_url_low_quality_content(self, crawler):
        """Test URL crawling with low quality content."""
        test_url = "http://example.com"
        test_content = "<html><body>Low quality</body></html>"

        # Mock responses
        mock_crawl_response = Mock()
        mock_crawl_response.success = True
        mock_crawl_response.content = test_content
        mock_crawl_response.content_type = "text/html"
        mock_crawl_response.timestamp = 1234567890.0

        mock_extracted_content = Mock()
        mock_extracted_content.title = "Low Quality"
        mock_extracted_content.content = "Low quality"
        mock_extracted_content.quality = Mock()
        mock_extracted_content.quality.overall_score = 0.2  # Below threshold

        with patch.object(crawler.crawler, 'crawl_url', return_value=mock_crawl_response), \
             patch.object(crawler.extractor, 'extract', return_value=mock_extracted_content), \
             patch.object(crawler.link_discovery, 'extract_links', return_value=[]):

            result = await crawler.crawl_url(test_url)

            # Should still succeed but log warning
            assert result.success is True
            assert result.metadata["quality_score"] == 0.2

    @pytest.mark.asyncio
    async def test_crawl_url_exception_handling(self, crawler):
        """Test exception handling in crawl_url."""
        test_url = "http://example.com"

        with patch.object(crawler.crawler, 'crawl_url', side_effect=Exception("Test error")):
            result = await crawler.crawl_url(test_url)

            assert result.success is False
            assert result.error == "Test error"
            assert result.url == test_url

    @pytest.mark.asyncio
    async def test_crawl_recursively_basic(self, crawler):
        """Test basic recursive crawling."""
        start_url = "http://example.com"

        # Mock crawl session
        mock_session = Mock()
        mock_session.has_pending_urls.side_effect = [True, True, False]  # Two iterations then done
        mock_session.get_next_url.side_effect = [
            "http://example.com",
            "http://example.com/page1",
            None
        ]

        # Mock successful crawl results
        async def mock_crawl_url(url):
            return CrawlResult(
                url=url,
                title=f"Page {url.split('/')[-1]}",
                content="Test content",
                extracted_links=["http://example.com/page2"],
                success=True
            )

        with patch.object(crawler.recursive_crawler, 'start_crawl_session', return_value=mock_session), \
             patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):

            results, summary = await crawler.crawl_recursively(start_url, max_depth=2)

            assert len(results) == 2
            assert summary.total_urls_crawled == 2
            assert summary.successful_crawls == 2
            assert summary.failed_crawls == 0
            assert summary.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_crawl_recursively_max_pages_limit(self, crawler):
        """Test recursive crawling with max pages limit."""
        start_url = "http://example.com"

        # Mock crawl session that would continue indefinitely
        mock_session = Mock()
        mock_session.has_pending_urls.return_value = True
        mock_session.get_next_url.side_effect = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://example.com/page3",
            "http://example.com/page4"
        ]

        # Mock successful crawl results
        async def mock_crawl_url(url):
            return CrawlResult(
                url=url,
                title="Test Page",
                content="Test content",
                success=True
            )

        with patch.object(crawler.recursive_crawler, 'start_crawl_session', return_value=mock_session), \
             patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):

            results, summary = await crawler.crawl_recursively(start_url, max_pages=2)

            # Should stop at max_pages
            assert len(results) == 2
            assert summary.total_urls_crawled == 2

    @pytest.mark.asyncio
    async def test_crawl_recursively_domain_filter(self, crawler):
        """Test recursive crawling with domain filtering."""
        start_url = "http://example.com"
        domain_filter = "example.com"

        mock_session = Mock()
        mock_session.has_pending_urls.side_effect = [True, True, True, False]
        mock_session.get_next_url.side_effect = [
            "http://example.com/page1",
            "http://other.com/page2",  # Different domain, should be skipped
            "http://example.com/page3",
            None
        ]

        # Mock successful crawl results
        async def mock_crawl_url(url):
            return CrawlResult(
                url=url,
                title="Test Page",
                content="Test content",
                success=True
            )

        with patch.object(crawler.recursive_crawler, 'start_crawl_session', return_value=mock_session), \
             patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):

            results, summary = await crawler.crawl_recursively(
                start_url,
                domain_filter=domain_filter
            )

            # Should have crawled only example.com URLs
            assert len(results) == 2
            for result in results:
                assert "example.com" in result.url

    @pytest.mark.asyncio
    async def test_crawl_recursively_with_failures(self, crawler):
        """Test recursive crawling with some failures."""
        start_url = "http://example.com"

        mock_session = Mock()
        mock_session.has_pending_urls.side_effect = [True, True, True, False]
        mock_session.get_next_url.side_effect = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://example.com/page3",
            None
        ]

        # Mock mixed success/failure results
        async def mock_crawl_url(url):
            if "page2" in url:
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    success=False,
                    error="404 Not Found"
                )
            else:
                return CrawlResult(
                    url=url,
                    title="Test Page",
                    content="Test content",
                    success=True
                )

        with patch.object(crawler.recursive_crawler, 'start_crawl_session', return_value=mock_session), \
             patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):

            results, summary = await crawler.crawl_recursively(start_url)

            assert len(results) == 3
            assert summary.successful_crawls == 2
            assert summary.failed_crawls == 1
            assert summary.error_summary.get("404 Not Found") == 1

    @pytest.mark.asyncio
    async def test_crawl_recursively_exception_handling(self, crawler):
        """Test exception handling in recursive crawling."""
        start_url = "http://example.com"

        with patch.object(crawler.recursive_crawler, 'start_crawl_session', side_effect=Exception("Test error")):
            results, summary = await crawler.crawl_recursively(start_url)

            assert results == []
            assert summary.total_urls_crawled == 0

    @pytest.mark.asyncio
    async def test_batch_crawl_success(self, crawler):
        """Test successful batch crawling."""
        urls = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://example.com/page3"
        ]

        # Mock successful crawl results
        async def mock_crawl_url(url):
            return CrawlResult(
                url=url,
                title=f"Page {url.split('/')[-1]}",
                content="Test content",
                success=True
            )

        with patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):
            results = await crawler.batch_crawl(urls)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.url == urls[i]
                assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_crawl_with_exceptions(self, crawler):
        """Test batch crawling with exceptions."""
        urls = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://example.com/page3"
        ]

        # Mock crawl_url to raise exception for page2
        async def mock_crawl_url(url):
            if "page2" in url:
                raise Exception("Network error")
            return CrawlResult(
                url=url,
                title="Test Page",
                content="Test content",
                success=True
            )

        with patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):
            results = await crawler.batch_crawl(urls)

            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[1].error == "Network error"
            assert results[2].success is True

    @pytest.mark.asyncio
    async def test_batch_crawl_concurrency_limit(self, crawler):
        """Test batch crawling respects concurrency limit."""
        urls = [f"http://example.com/page{i}" for i in range(10)]

        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0

        async def mock_crawl_url(url):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_calls -= 1
            return CrawlResult(url=url, title="", content="", success=True)

        with patch.object(crawler, 'crawl_url', side_effect=mock_crawl_url):
            await crawler.batch_crawl(urls, max_concurrent=3)

            # Should not exceed concurrency limit
            assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, crawler):
        """Test getting cache statistics."""
        # Mock cache stats
        mock_stats = Mock()
        mock_stats.total_entries = 100
        mock_stats.size_mb = 50.0
        mock_stats.hit_rate = 0.75
        mock_stats.duplicate_count = 10
        mock_stats.eviction_count = 5
        mock_stats.cleanup_count = 2
        mock_stats.average_size = 1024.0

        with patch.object(crawler.cache, 'get_stats', return_value=mock_stats), \
             patch.object(crawler.cache, 'get_duplicates', return_value={'hash1': ['url1', 'url2']}):

            stats = await crawler.get_cache_stats()

            assert stats['total_entries'] == 100
            assert stats['total_size_mb'] == 50.0
            assert stats['hit_rate'] == 0.75
            assert stats['duplicate_count'] == 10
            assert stats['eviction_count'] == 5
            assert stats['cleanup_count'] == 2
            assert stats['duplicate_groups'] == 1
            assert stats['average_entry_size'] == 1024.0

    @pytest.mark.asyncio
    async def test_clear_cache(self, crawler):
        """Test clearing cache."""
        with patch.object(crawler.cache, 'clear', return_value=50) as mock_clear:
            result = await crawler.clear_cache()
            assert result == 50
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_cache(self, crawler):
        """Test cache cleanup."""
        with patch.object(crawler.cache, 'cleanup', return_value=10) as mock_cleanup:
            result = await crawler.cleanup_cache()
            assert result == 10
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_content_hash(self, crawler):
        """Test content hash generation."""
        content = "Test content for hashing"

        with patch('workspace_qdrant_mcp.web.integration.ContentHasher') as mock_hasher:
            mock_hasher.hash_content.return_value = "test_hash_123"

            hash_result = await crawler._get_content_hash(content)

            assert hash_result == "test_hash_123"
            mock_hasher.hash_content.assert_called_once_with(content)

    @pytest.mark.asyncio
    async def test_async_context_manager(self, crawler):
        """Test crawler as async context manager."""
        with patch.object(crawler.cache, '__aenter__', return_value=crawler.cache) as mock_enter, \
             patch.object(crawler.cache, '__aexit__', return_value=None) as mock_exit:

            async with crawler as ctx_crawler:
                assert ctx_crawler is crawler

            mock_enter.assert_called_once()
            mock_exit.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions for simple usage."""

    @pytest.mark.asyncio
    async def test_crawl_url_simple(self):
        """Test simple URL crawling function."""
        test_url = "http://example.com"

        with patch('workspace_qdrant_mcp.web.integration.IntegratedWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler.__aenter__.return_value = mock_crawler
            mock_crawler.crawl_url.return_value = CrawlResult(
                url=test_url,
                title="Test Page",
                content="Test content",
                success=True
            )
            mock_crawler_class.return_value = mock_crawler

            result = await crawl_url_simple(test_url, rate_limit_delay=0.5)

            assert result.url == test_url
            assert result.success is True
            mock_crawler_class.assert_called_once()
            mock_crawler.crawl_url.assert_called_once_with(test_url)

    @pytest.mark.asyncio
    async def test_crawl_site_recursive(self):
        """Test simple recursive crawling function."""
        start_url = "http://example.com"
        expected_results = [
            CrawlResult(url="http://example.com", title="Home", content="Content", success=True),
            CrawlResult(url="http://example.com/page1", title="Page 1", content="Content", success=True)
        ]
        expected_summary = CrawlSummary(
            total_urls_crawled=2,
            successful_crawls=2,
            crawl_duration=1.5
        )

        with patch('workspace_qdrant_mcp.web.integration.IntegratedWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler.__aenter__.return_value = mock_crawler
            mock_crawler.crawl_recursively.return_value = (expected_results, expected_summary)
            mock_crawler_class.return_value = mock_crawler

            results, summary = await crawl_site_recursive(
                start_url,
                max_depth=3,
                max_pages=100,
                user_agent="TestBot/1.0"
            )

            assert results == expected_results
            assert summary == expected_summary
            mock_crawler.crawl_recursively.assert_called_once_with(
                start_url,
                max_depth=3,
                max_pages=100
            )


# Integration test for realistic workflow
class TestIntegrationWorkflow:
    """Integration tests for realistic crawling workflows."""

    @pytest.mark.asyncio
    async def test_complete_crawling_workflow(self):
        """Test a complete crawling workflow with all components."""
        config = IntegratedCrawlConfig(
            rate_limit_delay=0.0,
            enable_persistence=False,
            max_concurrent_requests=2
        )

        with patch.multiple(
            'workspace_qdrant_mcp.web.integration',
            WebCrawler=Mock,
            ContentExtractor=Mock,
            LinkDiscovery=Mock,
            RecursiveCrawler=Mock,
            ContentCache=Mock
        ):
            crawler = IntegratedWebCrawler(config)

            # Mock all component methods
            crawler.crawler.crawl_url = AsyncMock(return_value=Mock(
                success=True,
                content="<html><title>Test</title><body>Content</body></html>",
                content_type="text/html",
                timestamp=1234567890.0
            ))

            crawler.extractor.extract = Mock(return_value=Mock(
                title="Test Page",
                content="Content",
                quality=Mock(overall_score=0.8)
            ))

            crawler.link_discovery.extract_links = Mock(return_value=[])

            crawler.cache.get = AsyncMock(return_value=None)
            crawler.cache.put = AsyncMock(return_value=True)

            # Test single URL crawl
            result = await crawler.crawl_url("http://example.com")
            assert result.success is True
            assert result.title == "Test Page"

            # Test cache statistics
            crawler.cache.get_stats = Mock(return_value=Mock(
                total_entries=1,
                size_mb=1.0,
                hit_rate=0.0,
                duplicate_count=0,
                eviction_count=0,
                cleanup_count=0,
                average_size=1024.0
            ))
            crawler.cache.get_duplicates = Mock(return_value={})

            stats = await crawler.get_cache_stats()
            assert stats['total_entries'] == 1
