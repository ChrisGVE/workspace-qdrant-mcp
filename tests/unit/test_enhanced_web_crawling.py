"""
Comprehensive test suite for enhanced web crawling system.

Tests all components including content extraction, retry logic, caching, and integration.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import the modules to test
from src.python.wqm_cli.cli.parsers.advanced_retry import (
    AdvancedRetryHandler,
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    RetryReason,
)
from src.python.wqm_cli.cli.parsers.enhanced_content_extractor import (
    ContentQualityMetrics,
    EnhancedContentExtractor,
    MediaLinks,
    StructuredData,
)
from src.python.wqm_cli.cli.parsers.enhanced_web_crawler import (
    EnhancedCrawlResult,
    EnhancedWebCrawler,
)
from src.python.wqm_cli.cli.parsers.web_cache import (
    CacheConfig,
    CacheEntry,
    ContentFingerprinter,
    WebCache,
)
from src.python.wqm_cli.cli.parsers.web_crawler import SecurityConfig


class TestContentFingerprinter:
    """Test content fingerprinting functionality."""

    def setup_method(self):
        self.fingerprinter = ContentFingerprinter()

    def test_generate_fingerprint(self):
        """Test fingerprint generation."""
        content = "Hello, world!"
        fingerprint = self.fingerprinter.generate_fingerprint(content)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA256 hex length

        # Same content should produce same fingerprint
        fingerprint2 = self.fingerprinter.generate_fingerprint(content)
        assert fingerprint == fingerprint2

    def test_normalize_content_for_fingerprinting(self):
        """Test content normalization."""
        content = "  Hello,\n   World!  \n\n"
        normalized = self.fingerprinter.normalize_content_for_fingerprinting(content)

        assert normalized == "hello,\nworld!"

    def test_different_content_different_fingerprints(self):
        """Test that different content produces different fingerprints."""
        fp1 = self.fingerprinter.generate_fingerprint("content1")
        fp2 = self.fingerprinter.generate_fingerprint("content2")

        assert fp1 != fp2


class TestEnhancedContentExtractor:
    """Test enhanced content extraction."""

    def setup_method(self):
        self.extractor = EnhancedContentExtractor()

    def test_extract_basic_metadata(self):
        """Test basic metadata extraction."""
        html = '''
        <html lang="en">
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
                <meta name="author" content="Test Author">
                <meta name="keywords" content="test,page">
            </head>
            <body>
                <p>Test content</p>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html)
        metadata = result['metadata']

        assert metadata['title'] == 'Test Page'
        assert metadata['description'] == 'Test description'
        assert metadata['author'] == 'Test Author'
        assert metadata['keywords'] == 'test,page'
        assert metadata['language'] == 'en'

    def test_extract_structured_data_json_ld(self):
        """Test JSON-LD structured data extraction."""
        html = '''
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Test Article",
                    "author": "Test Author"
                }
                </script>
            </head>
            <body>
                <p>Content</p>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html)
        json_ld = result['structured_data']['json_ld']

        assert len(json_ld) == 1
        assert json_ld[0]['@type'] == 'Article'
        assert json_ld[0]['headline'] == 'Test Article'

    def test_extract_open_graph_data(self):
        """Test Open Graph metadata extraction."""
        html = '''
        <html>
            <head>
                <meta property="og:title" content="OG Title">
                <meta property="og:description" content="OG Description">
                <meta property="og:image" content="https://example.com/image.jpg">
            </head>
            <body>
                <p>Content</p>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html)
        og_data = result['structured_data']['open_graph']

        assert og_data['og:title'] == 'OG Title'
        assert og_data['og:description'] == 'OG Description'
        assert og_data['og:image'] == 'https://example.com/image.jpg'

    def test_extract_media_links(self):
        """Test media link extraction."""
        html = '''
        <html>
            <body>
                <img src="image1.jpg" alt="Image 1">
                <img src="image2.png" alt="Image 2" width="100" height="200">
                <video src="video.mp4" controls></video>
                <audio src="audio.mp3"></audio>
                <a href="document.pdf">Download PDF</a>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html, "https://example.com/")
        media = result['media_links']

        assert len(media['images']) == 2
        assert media['images'][0]['url'] == 'https://example.com/image1.jpg'
        assert media['images'][0]['alt'] == 'Image 1'

        assert len(media['videos']) == 1
        assert media['videos'][0]['url'] == 'https://example.com/video.mp4'

        assert len(media['audio']) == 1
        assert media['audio'][0]['url'] == 'https://example.com/audio.mp3'

        assert len(media['documents']) == 1
        assert media['documents'][0]['url'] == 'https://example.com/document.pdf'

    def test_content_quality_filtering(self):
        """Test content quality filtering and boilerplate removal."""
        html = '''
        <html>
            <body>
                <nav class="navigation">Navigation menu</nav>
                <aside class="sidebar">Sidebar content</aside>
                <main>
                    <article>
                        <h1>Main Article Title</h1>
                        <p>This is the main article content with substantial text that should be preserved.</p>
                        <p>Another paragraph with meaningful content about the topic.</p>
                    </article>
                </main>
                <footer>Footer content</footer>
                <div class="advertisement">Buy now! Special offer!</div>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html)
        main_content = result['main_content']

        # Should contain main content
        assert 'Main Article Title' in main_content
        assert 'substantial text' in main_content

        # Should NOT contain boilerplate
        assert 'Navigation menu' not in main_content
        assert 'Sidebar content' not in main_content
        assert 'Footer content' not in main_content
        assert 'Buy now!' not in main_content

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        html = '''
        <html>
            <body>
                <article>
                    <p>First paragraph with substantial content.</p>
                    <p>Second paragraph also with meaningful text.</p>
                    <p>Third paragraph continues the content.</p>
                </article>
                <div class="ads">Advertisement</div>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html)
        metrics = result['quality_metrics']

        assert metrics['paragraph_count'] >= 2
        assert metrics['text_length'] > 0
        assert metrics['link_density'] >= 0
        assert metrics['ad_indicators'] >= 0

    def test_text_links_extraction(self):
        """Test text link extraction."""
        html = '''
        <html>
            <body>
                <a href="page1.html">Link 1</a>
                <a href="page2.html" title="Page 2">Link 2</a>
                <a href="external.com">External Link</a>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html, "https://example.com/")
        links = result['text_links']

        assert len(links) == 3
        assert links[0]['url'] == 'https://example.com/page1.html'
        assert links[0]['text'] == 'Link 1'
        assert links[1]['title'] == 'Page 2'


class TestWebCache:
    """Test web caching functionality."""

    def setup_method(self):
        self.config = CacheConfig(enable_disk_cache=False)  # Disable disk cache for tests
        self.cache = WebCache(self.config)

    def test_cache_key_generation(self):
        """Test cache key generation."""
        url = "https://example.com/page"
        key = self.cache.get_cache_key(url)

        assert isinstance(key, str)
        assert 'GET' in key
        assert 'example.com' in key

    def test_cache_key_normalization(self):
        """Test URL normalization in cache keys."""
        url1 = "https://example.com/page"
        url2 = "https://EXAMPLE.COM/page#fragment"

        key1 = self.cache.get_cache_key(url1)
        key2 = self.cache.get_cache_key(url2)

        # Should be the same after normalization
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_cache_put_get(self):
        """Test basic cache put/get operations."""
        url = "https://example.com/page"
        content = "Test content"
        headers = {'content-type': 'text/html'}
        status_code = 200

        # Cache the response
        success = await self.cache.put(url, content, headers, status_code)
        assert success

        # Retrieve from cache
        entry = self.cache.get(url)
        assert entry is not None
        assert entry.content == content
        assert entry.status_code == status_code

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        entry = CacheEntry(
            url="https://example.com",
            content="test",
            headers={},
            status_code=200,
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl=3600  # 1 hour TTL
        )

        assert entry.is_expired()

    def test_content_fingerprinting(self):
        """Test content fingerprinting for deduplication."""
        content1 = "Same content"
        content2 = "Same content"

        duplicates1 = self.cache.find_duplicate_content(content1)
        assert len(duplicates1) == 0  # Nothing cached yet

        # Simulate cached content (would normally happen via put())
        fingerprint = self.cache.fingerprinter.generate_fingerprint(
            self.cache.fingerprinter.normalize_content_for_fingerprinting(content1)
        )
        self.cache._content_fingerprints[fingerprint] = {"https://example.com/1"}

        duplicates2 = self.cache.find_duplicate_content(content2)
        assert len(duplicates2) == 1
        assert "https://example.com/1" in duplicates2

    def test_conditional_headers(self):
        """Test conditional headers for cache validation."""
        url = "https://example.com/page"

        # No cache entry - no conditional headers
        headers = self.cache.get_conditional_headers(url)
        assert len(headers) == 0

        # Add cache entry with ETag
        entry = CacheEntry(
            url=url,
            content="test",
            headers={'etag': '"abc123"', 'last-modified': 'Wed, 21 Oct 2015 07:28:00 GMT'},
            status_code=200,
            timestamp=time.time(),
            ttl=3600
        )

        cache_key = self.cache.get_cache_key(url)
        self.cache._cache[cache_key] = entry

        headers = self.cache.get_conditional_headers(url)
        assert headers['If-None-Match'] == '"abc123"'
        assert headers['If-Modified-Since'] == 'Wed, 21 Oct 2015 07:28:00 GMT'


class TestAdvancedRetryHandler:
    """Test advanced retry logic."""

    def setup_method(self):
        self.config = RetryConfig(max_retries=3, base_delay=0.1, max_delay=1.0)
        self.retry_handler = AdvancedRetryHandler(self.config)

    def test_should_retry_on_server_errors(self):
        """Test retry logic for server errors."""
        should_retry, reason = self.retry_handler.should_retry(Exception(), status_code=500, attempt=0)
        assert should_retry
        assert reason == RetryReason.SERVER_ERROR

    def test_should_retry_on_rate_limiting(self):
        """Test retry logic for rate limiting."""
        should_retry, reason = self.retry_handler.should_retry(Exception(), status_code=429, attempt=0)
        assert should_retry
        assert reason == RetryReason.RATE_LIMITED

    def test_should_not_retry_on_client_errors(self):
        """Test no retry for client errors."""
        should_retry, reason = self.retry_handler.should_retry(Exception(), status_code=404, attempt=0)
        assert not should_retry

    def test_should_not_retry_after_max_attempts(self):
        """Test max retry limit."""
        should_retry, reason = self.retry_handler.should_retry(Exception(), status_code=500, attempt=5)
        assert not should_retry

    def test_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        delay1 = self.retry_handler.calculate_delay(0, RetryReason.SERVER_ERROR)
        delay2 = self.retry_handler.calculate_delay(1, RetryReason.SERVER_ERROR)
        delay3 = self.retry_handler.calculate_delay(2, RetryReason.SERVER_ERROR)

        # Should increase exponentially (roughly)
        assert delay2 > delay1
        assert delay3 > delay2

        # Should respect max delay
        delay_large = self.retry_handler.calculate_delay(10, RetryReason.SERVER_ERROR)
        assert delay_large <= self.config.max_delay

    def test_rate_limiting_delay(self):
        """Test special delay for rate limiting."""
        delay = self.retry_handler.calculate_delay(0, RetryReason.RATE_LIMITED)
        assert delay >= 10.0  # Minimum for rate limiting


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def setup_method(self):
        self.config = RetryConfig(circuit_failure_threshold=3, circuit_recovery_timeout=1.0)
        self.circuit_breaker = CircuitBreaker(self.config)

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        url = "https://example.com"
        assert self.circuit_breaker.can_attempt_request(url)

    def test_circuit_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        url = "https://example.com"

        # Record failures
        for _ in range(self.config.circuit_failure_threshold):
            self.circuit_breaker.record_failure(url, Exception("Server error"), 500)

        # Circuit should be open
        assert not self.circuit_breaker.can_attempt_request(url)

    def test_circuit_half_open_recovery(self):
        """Test circuit breaker half-open state."""
        url = "https://example.com"

        # Open the circuit
        for _ in range(self.config.circuit_failure_threshold):
            self.circuit_breaker.record_failure(url, Exception("Server error"), 500)

        # Wait for recovery timeout
        time.sleep(self.config.circuit_recovery_timeout + 0.1)

        # Should allow one request (half-open)
        assert self.circuit_breaker.can_attempt_request(url)

    def test_circuit_success_resets_failures(self):
        """Test successful requests reset failure count."""
        url = "https://example.com"

        # Record some failures
        self.circuit_breaker.record_failure(url, Exception("Error"), 500)
        self.circuit_breaker.record_failure(url, Exception("Error"), 500)

        # Record success
        self.circuit_breaker.record_success(url)

        # Should reset failure count
        stats = self.circuit_breaker.get_domain_stats("example.com")
        assert stats.failure_count == 0


class TestEnhancedWebCrawler:
    """Test enhanced web crawler functionality."""

    def setup_method(self):
        self.security_config = SecurityConfig()
        self.cache_config = CacheConfig(enable_disk_cache=False)
        self.retry_config = RetryConfig(max_retries=2, base_delay=0.01)

        self.crawler = EnhancedWebCrawler(
            self.security_config,
            self.cache_config,
            self.retry_config
        )

    def test_initialization(self):
        """Test crawler initialization."""
        assert self.crawler.content_extractor is not None
        assert self.crawler.cache is not None
        assert self.crawler.retry_handler is not None
        assert len(self.crawler.user_agents) > 1

    def test_user_agent_rotation(self):
        """Test user agent rotation."""
        ua1 = self.crawler._get_next_user_agent()
        ua2 = self.crawler._get_next_user_agent()

        # Should rotate
        assert ua1 != ua2

    def test_quality_score_calculation(self):
        """Test content quality score calculation."""
        metrics = {
            'text_length': 1000,
            'paragraph_count': 5,
            'link_density': 0.1,
            'ad_indicators': 0,
            'navigation_indicators': 0,
            'boilerplate_score': 0.2,
            'readability_score': 20
        }

        score = self.crawler._calculate_quality_score(metrics)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent quality

    def test_session_stats_tracking(self):
        """Test session statistics tracking."""
        stats = self.crawler.get_session_stats()

        assert 'session_id' in stats
        assert 'crawl_stats' in stats
        assert 'cache_stats' in stats
        assert 'retry_stats' in stats

        # Initial stats should be zeroed
        assert stats['crawl_stats']['urls_crawled'] == 0


class TestIntegrationEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        self.extractor = EnhancedContentExtractor()

    def test_malformed_html_handling(self):
        """Test handling of malformed HTML."""
        malformed_html = '<html><head><title>Test</head><body><p>Unclosed paragraph'

        result = self.extractor.extract_content(malformed_html)

        # Should not crash and should extract some content
        assert 'error' not in result
        assert result['main_content'] is not None

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        result = self.extractor.extract_content("")

        assert result['main_content'] == ''
        assert result['word_count'] == 0
        assert result['char_count'] == 0

    def test_very_large_content_handling(self):
        """Test handling of very large content."""
        large_content = '<p>' + 'A' * 100000 + '</p>'

        result = self.extractor.extract_content(large_content)

        # Should handle large content without issues
        assert len(result['main_content']) > 0
        assert result['word_count'] > 0

    def test_non_text_content_handling(self):
        """Test handling of non-text content."""
        binary_like = '<html><body>' + '\x00\x01\x02' * 100 + '</body></html>'

        result = self.extractor.extract_content(binary_like)

        # Should handle gracefully
        assert 'main_content' in result

    def test_deeply_nested_html(self):
        """Test handling of deeply nested HTML structures."""
        nested = '<div>' * 100 + 'Content' + '</div>' * 100
        html = f'<html><body>{nested}</body></html>'

        result = self.extractor.extract_content(html)

        assert 'Content' in result['main_content']

    def test_invalid_json_ld_handling(self):
        """Test handling of invalid JSON-LD."""
        html = '''
        <html>
            <head>
                <script type="application/ld+json">
                { invalid json content }
                </script>
            </head>
            <body><p>Content</p></body>
        </html>
        '''

        result = self.extractor.extract_content(html)

        # Should not crash on invalid JSON
        assert 'structured_data' in result
        assert len(result['structured_data']['json_ld']) == 0

    def test_special_characters_handling(self):
        """Test handling of special characters and encodings."""
        html = '''
        <html>
            <head><title>Special: àáâãäåæçèéêë</title></head>
            <body>
                <p>Unicode: 你好世界 🌍 ñáéíóú</p>
                <p>Symbols: ©®™€£¥</p>
            </body>
        </html>
        '''

        result = self.extractor.extract_content(html)

        assert 'àáâãäåæçèéêë' in result['metadata']['title']
        assert '你好世界' in result['main_content']
        assert '🌍' in result['main_content']


# Integration test helper functions
def create_mock_response(status=200, content="<html><body><p>Test content</p></body></html>", headers=None):
    """Create a mock HTTP response."""
    if headers is None:
        headers = {'content-type': 'text/html'}

    mock_response = Mock()
    mock_response.status = status
    mock_response.headers = headers
    mock_response.reason = "OK" if status == 200 else "Error"

    # Mock async iteration for content
    async def mock_iter_chunked(size):
        content_bytes = content.encode('utf-8')
        for i in range(0, len(content_bytes), size):
            yield content_bytes[i:i+size]

    mock_response.content.iter_chunked = mock_iter_chunked
    return mock_response


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
