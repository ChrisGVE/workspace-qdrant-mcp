"""
Comprehensive edge case tests for web page processing pipeline.

This test suite covers advanced edge cases, error scenarios, and performance
testing for the web crawler and parser infrastructure.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from urllib.parse import urlparse
import aiohttp
import ssl

from wqm_cli.cli.parsers import (
    WebParser,
    WebIngestionInterface,
    SecureWebCrawler,
    SecurityConfig,
    CrawlResult,
    create_secure_web_parser
)
from wqm_cli.cli.parsers.web_crawler import SecurityScanner
from wqm_cli.cli.parsers.exceptions import ParsingError


class TestAdvancedURLValidation:
    """Tests for advanced URL validation edge cases."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com', 'test.org'}
        return SecureWebCrawler(config)

    @pytest.mark.asyncio
    async def test_malformed_url_schemes(self, crawler):
        """Test various malformed URL schemes."""
        malformed_urls = [
            "htt://example.com",  # Typo in scheme
            "https//example.com",  # Missing colon
            "https:example.com",   # Missing double slash
            "://example.com",      # Missing scheme
            "http:///example.com", # Triple slash
            "",                    # Empty URL
            "   ",                 # Whitespace only
            "not-a-url",          # No scheme at all
        ]

        for url in malformed_urls:
            result = CrawlResult(url)
            is_valid = await crawler._validate_url(url, result)
            assert is_valid is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_extremely_long_urls(self, crawler):
        """Test URLs that exceed maximum length."""
        # Create URL longer than max_url_length (2048)
        base_url = "https://example.com/"
        long_path = "a" * 2100
        long_url = base_url + long_path

        result = CrawlResult(long_url)
        is_valid = await crawler._validate_url(long_url, result)

        assert is_valid is False
        assert "URL too long" in result.error

    @pytest.mark.asyncio
    async def test_suspicious_url_patterns(self, crawler):
        """Test detection of suspicious URL patterns."""
        suspicious_urls = [
            "https://example.com/script.exe",
            "https://example.com/malware.bat",
            "https://example.com/virus.scr",
            "https://example.com/download.msi",
            "https://bit.ly/shortened",  # URL shortener
        ]

        for url in suspicious_urls:
            result = CrawlResult(url)
            is_valid = await crawler._validate_url(url, result)
            # Should have security warnings
            assert len(result.security_warnings) > 0 or not is_valid

    @pytest.mark.asyncio
    async def test_unicode_and_encoded_urls(self, crawler):
        """Test URLs with Unicode and encoding issues."""
        unicode_urls = [
            "https://example.com/path%20with%20spaces",
            "https://example.com/path%2Fwith%2Fencoded",
            "https://example.com/résumé",  # Unicode characters
            "https://example.com/" + "ü" * 50,  # Many Unicode chars
            "https://example.com/%20%20%20%20%20",  # Many encoded spaces
        ]

        for url in unicode_urls:
            result = CrawlResult(url)
            is_valid = await crawler._validate_url(url, result)
            # Should either be valid or have specific encoding warnings
            if not is_valid and result.security_warnings:
                assert any("encoded" in warning.lower() for warning in result.security_warnings)


class TestAdvancedContentFetching:
    """Tests for advanced content fetching scenarios."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        return SecureWebCrawler(config)

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_network_timeout_scenarios(self, mock_get, crawler):
        """Test various network timeout scenarios."""
        # Test different timeout exceptions
        timeout_exceptions = [
            asyncio.TimeoutError(),
            aiohttp.ServerTimeoutError(),
            aiohttp.ClientConnectorError(None, OSError("Connection timeout")),
        ]

        for exception in timeout_exceptions:
            mock_get.side_effect = exception

            result = CrawlResult("https://example.com/test")
            await crawler._ensure_session()
            await crawler._fetch_content("https://example.com/test", result)

            assert result.success is False
            assert "timeout" in result.error.lower() or "connection" in result.error.lower()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_ssl_certificate_errors(self, mock_get, crawler):
        """Test SSL certificate validation errors."""
        ssl_errors = [
            aiohttp.ClientSSLError(None, ssl.SSLError("certificate verify failed")),
            aiohttp.ClientConnectorSSLError(None, ssl.SSLError("hostname mismatch")),
        ]

        for ssl_error in ssl_errors:
            mock_get.side_effect = ssl_error

            result = CrawlResult("https://example.com/ssl-test")
            await crawler._ensure_session()
            await crawler._fetch_content("https://example.com/ssl-test", result)

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_chunked_content_size_limit(self, mock_get, crawler):
        """Test content size limit with chunked responses."""
        # Create large content delivered in chunks
        large_chunk1 = b"x" * (30 * 1024 * 1024)  # 30MB
        large_chunk2 = b"y" * (25 * 1024 * 1024)  # 25MB (total > 50MB)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content.iter_chunked.return_value = [large_chunk1, large_chunk2]
        mock_get.return_value.__aenter__.return_value = mock_response

        result = CrawlResult("https://example.com/large-content")
        await crawler._ensure_session()
        await crawler._fetch_content("https://example.com/large-content", result)

        assert result.success is False
        assert "size limit exceeded" in result.error.lower()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_malformed_http_responses(self, mock_get, crawler):
        """Test handling of malformed HTTP responses."""
        malformed_scenarios = [
            # Missing content-type header
            (200, {}, "valid content"),
            # Invalid content-length
            (200, {'content-length': 'not-a-number'}, "content"),
            # Negative content-length
            (200, {'content-length': '-100'}, "content"),
        ]

        for status, headers, content in malformed_scenarios:
            mock_response = AsyncMock()
            mock_response.status = status
            mock_response.headers = headers
            mock_response.content.iter_chunked.return_value = [content.encode()]
            mock_get.return_value.__aenter__.return_value = mock_response

            result = CrawlResult("https://example.com/malformed")
            await crawler._ensure_session()
            await crawler._fetch_content("https://example.com/malformed", result)

            # Should handle gracefully without crashing
            assert result is not None

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_encoding_detection_edge_cases(self, mock_get, crawler):
        """Test content encoding detection with edge cases."""
        encoding_test_cases = [
            # Valid UTF-8 with BOM
            (b'\xef\xbb\xbf<html><body>UTF-8 content</body></html>', 'text/html; charset=utf-8'),
            # Latin-1 content
            (b'<html><body>Caf\xe9 content</body></html>', 'text/html; charset=latin-1'),
            # Binary content that looks like text
            (b'\x00\x01\x02<html>', 'text/html'),
            # Empty content
            (b'', 'text/html'),
        ]

        for content_bytes, content_type in encoding_test_cases:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': content_type}
            mock_response.content.iter_chunked.return_value = [content_bytes]
            mock_get.return_value.__aenter__.return_value = mock_response

            result = CrawlResult("https://example.com/encoding-test")
            await crawler._ensure_session()
            await crawler._fetch_content("https://example.com/encoding-test", result)

            # Should not crash and produce some result
            assert result is not None


class TestAdvancedSecurityScanning:
    """Tests for advanced security scanning scenarios."""

    @pytest.fixture
    def scanner(self):
        """Create a security scanner instance."""
        return SecurityScanner()

    @pytest.mark.asyncio
    async def test_sophisticated_malicious_patterns(self, scanner):
        """Test detection of sophisticated malicious patterns."""
        malicious_contents = [
            # Obfuscated JavaScript
            "<script>var a='ev'+'al';window[a]('malicious code');</script>",
            # Multiple script injections
            "<div>" + "<script>hack();</script>" * 20 + "</div>",
            # Hidden iframes
            "<iframe src='javascript:evil()' style='display:none'></iframe>",
            # Data URLs with JavaScript
            "<img src=\"data:text/html,<script>alert()</script>\">",
            # Event handler obfuscation
            "<div onmouseover=\"&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;:alert()\">",
            # Binary content mixed with HTML
            "<html>\x00\xff<script>malware</script></html>",
        ]

        for content in malicious_contents:
            is_safe, warnings = await scanner.scan_content(content, 'text/html')
            assert not is_safe or len(warnings) > 0

    @pytest.mark.asyncio
    async def test_content_size_security_limits(self, scanner):
        """Test security scanning with various content sizes."""
        # Very large content (potential DoS)
        huge_content = "x" * (15 * 1024 * 1024)  # 15MB
        is_safe, warnings = await scanner.scan_content(huge_content, 'text/html')

        assert any("Large content size" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_mixed_content_security_analysis(self, scanner):
        """Test security analysis of mixed content types."""
        mixed_contents = [
            # HTML with embedded CSS and JS
            """
            <html>
            <style>body { background: url('javascript:evil()'); }</style>
            <script>document.write('<img src=x onerror=alert()>');</script>
            </html>
            """,
            # XML with script injection
            """<?xml version="1.0"?>
            <root>
                <script><![CDATA[eval('malicious')]]></script>
            </root>
            """,
        ]

        for content in mixed_contents:
            is_safe, warnings = await scanner.scan_content(content, 'text/html')
            # Should detect threats in mixed content
            assert not is_safe or len(warnings) > 0


class TestAdvancedRateLimiting:
    """Tests for advanced rate limiting scenarios."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com', 'test.org', 'site.net'}
        config.request_delay = 0.5  # Shorter delay for testing
        return SecureWebCrawler(config)

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_domain(self, crawler):
        """Test rate limiting with concurrent requests to same domain."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]

        # Start all requests simultaneously
        start_time = time.time()
        tasks = []
        for url in urls:
            tasks.append(crawler._respect_rate_limit(url))

        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Should take at least 2 * request_delay (for 3 requests to same domain)
        expected_min_time = 2 * crawler.config.request_delay * 0.8  # Allow some tolerance
        assert total_time >= expected_min_time

    @pytest.mark.asyncio
    async def test_rate_limiting_multiple_domains(self, crawler):
        """Test rate limiting with multiple domains."""
        urls = [
            "https://example.com/page1",
            "https://test.org/page1",
            "https://site.net/page1"
        ]

        # Requests to different domains should not interfere
        start_time = time.time()
        tasks = []
        for url in urls:
            tasks.append(crawler._respect_rate_limit(url))

        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Should be relatively quick since different domains
        assert total_time < crawler.config.request_delay * 1.5

    @pytest.mark.asyncio
    async def test_domain_request_count_limits(self, crawler):
        """Test per-domain request count limits."""
        # Set a low limit for testing
        crawler.config.max_pages_per_domain = 2

        # First two requests should be allowed
        result1 = CrawlResult("https://example.com/page1")
        assert await crawler._validate_url("https://example.com/page1", result1)

        result2 = CrawlResult("https://example.com/page2")
        assert await crawler._validate_url("https://example.com/page2", result2)

        # Third request should be blocked
        result3 = CrawlResult("https://example.com/page3")
        assert not await crawler._validate_url("https://example.com/page3", result3)
        assert "request limit exceeded" in result3.error.lower()


class TestAdvancedRecursiveCrawling:
    """Tests for advanced recursive crawling scenarios."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        config.max_total_pages = 10
        config.max_depth = 3
        return SecureWebCrawler(config)

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.SecureWebCrawler.crawl_url')
    @patch('wqm_cli.cli.parsers.web_crawler.SecureWebCrawler._extract_links')
    async def test_circular_link_detection(self, mock_extract_links, mock_crawl_url, crawler):
        """Test detection and handling of circular links."""
        # Setup circular links: page1 -> page2 -> page1
        def create_mock_result(url, links):
            result = CrawlResult(url)
            result.success = True
            result.content = f"<html><body>Content for {url}</body></html>"
            mock_extract_links.return_value = links
            return result

        # Mock crawl results
        results = {
            "https://example.com/page1": create_mock_result(
                "https://example.com/page1",
                ["https://example.com/page2"]
            ),
            "https://example.com/page2": create_mock_result(
                "https://example.com/page2",
                ["https://example.com/page1"]  # Circular reference
            )
        }

        async def mock_crawl_side_effect(url, **kwargs):
            return results[url]

        mock_crawl_url.side_effect = mock_crawl_side_effect

        await crawler._ensure_session()

        # Should handle circular references gracefully
        crawl_results = await crawler.crawl_recursive("https://example.com/page1")

        # Should not get stuck in infinite loop
        assert len(crawl_results) <= crawler.config.max_total_pages

        # Should visit each unique URL only once
        unique_urls = set(result.url for result in crawl_results)
        assert len(unique_urls) == len([r for r in crawl_results if r.success])

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.SecureWebCrawler.crawl_url')
    async def test_depth_limit_enforcement(self, mock_crawl_url, crawler):
        """Test that depth limits are properly enforced."""
        # Mock a deep link hierarchy
        def create_deep_result(depth):
            result = CrawlResult(f"https://example.com/depth{depth}")
            result.success = True
            result.content = f"<html><body>Depth {depth}</body></html>"
            return result

        mock_crawl_url.return_value = create_deep_result(1)

        await crawler._ensure_session()

        # Override _extract_links to simulate deep hierarchy
        async def mock_extract_links(content, base_url):
            current_depth = int(base_url.split('depth')[1])
            if current_depth < 5:  # Create deeper links than max_depth
                return [f"https://example.com/depth{current_depth + 1}"]
            return []

        crawler._extract_links = mock_extract_links

        results = await crawler.crawl_recursive("https://example.com/depth0", max_depth=2)

        # Should respect depth limit
        max_found_depth = 0
        for result in results:
            if 'depth' in result.url:
                depth = int(result.url.split('depth')[1])
                max_found_depth = max(max_found_depth, depth)

        assert max_found_depth <= 2


class TestLargeWebsiteHandling:
    """Tests for handling large websites and performance edge cases."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'bigsite.com'}
        config.max_total_pages = 100
        return SecureWebCrawler(config)

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.SecureWebCrawler.crawl_url')
    async def test_large_number_of_pages(self, mock_crawl_url, crawler):
        """Test crawling a large number of pages."""
        # Simulate a site with many pages
        def create_page_result(page_num):
            result = CrawlResult(f"https://bigsite.com/page{page_num}")
            result.success = True
            result.content = f"<html><body>Page {page_num} content</body></html>"
            return result

        page_counter = 0
        def mock_crawl_side_effect(url, **kwargs):
            nonlocal page_counter
            page_counter += 1
            return create_page_result(page_counter)

        mock_crawl_url.side_effect = mock_crawl_side_effect

        await crawler._ensure_session()

        # Override link extraction to generate many links
        async def mock_extract_many_links(content, base_url):
            # Generate links up to the limit
            links = []
            for i in range(min(20, crawler.config.max_total_pages)):
                links.append(f"https://bigsite.com/page{i}")
            return links

        crawler._extract_links = mock_extract_many_links

        results = await crawler.crawl_recursive("https://bigsite.com/start", max_pages=50)

        # Should respect page limits
        assert len(results) <= 50
        assert all(isinstance(result, CrawlResult) for result in results)

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_content(self, crawler):
        """Test memory usage patterns with large content."""
        # This is more of a smoke test - ensuring no memory leaks
        large_html = "<html><body>" + ("Large content block. " * 10000) + "</body></html>"

        # Process multiple large documents
        for i in range(5):
            result = CrawlResult(f"https://bigsite.com/large{i}")
            result.content = large_html
            result.success = True

            # Process content
            try:
                await crawler._process_content(result)
            except Exception as e:
                # Should handle large content gracefully
                assert "memory" not in str(e).lower() or "size" in str(e).lower()


class TestRobotsTxtEdgeCases:
    """Tests for robots.txt handling edge cases."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        config.respect_robots_txt = True
        return SecureWebCrawler(config)

    @pytest.mark.asyncio
    @patch('urllib.robotparser.RobotFileParser.read')
    async def test_malformed_robots_txt(self, mock_robots_read, crawler):
        """Test handling of malformed robots.txt files."""
        # Simulate robots.txt read failure
        mock_robots_read.side_effect = Exception("Malformed robots.txt")

        # Should allow crawling when robots.txt is malformed
        can_crawl = await crawler._check_robots_txt("https://example.com/test")
        assert can_crawl is True  # Should err on the side of allowing

    @pytest.mark.asyncio
    @patch('urllib.robotparser.RobotFileParser.read')
    @patch('urllib.robotparser.RobotFileParser.can_fetch')
    async def test_robots_txt_with_custom_user_agent(self, mock_can_fetch, mock_robots_read, crawler):
        """Test robots.txt checking with custom user agent."""
        mock_robots_read.return_value = None
        mock_can_fetch.return_value = False

        # Should respect robots.txt for our specific user agent
        can_crawl = await crawler._check_robots_txt("https://example.com/blocked")
        assert can_crawl is False

    @pytest.mark.asyncio
    async def test_robots_txt_caching(self, crawler):
        """Test that robots.txt responses are properly cached."""
        # First call should fetch robots.txt
        url1 = "https://example.com/page1"
        result1 = await crawler._check_robots_txt(url1)

        # Second call to same domain should use cache
        url2 = "https://example.com/page2"
        result2 = await crawler._check_robots_txt(url2)

        # Both should return same result (using cache)
        assert result1 == result2

        # Should have cached the robots.txt
        robots_url = "https://example.com/robots.txt"
        assert robots_url in crawler.robots_cache or len(crawler.robots_cache) > 0


class TestWebParserIntegration:
    """Integration tests for WebParser with edge cases."""

    @pytest.fixture
    def parser(self):
        """Create a test parser instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        return WebParser(config)

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.SecureWebCrawler')
    async def test_parse_with_mixed_success_failure(self, mock_crawler_class, parser):
        """Test parsing when some pages succeed and others fail."""
        # Mix of successful and failed results
        results = []

        # Successful result
        success_result = CrawlResult("https://example.com/success")
        success_result.success = True
        success_result.content = "<html><body>Success content</body></html>"
        success_result.metadata = {'parsed_content': 'Success content'}
        results.append(success_result)

        # Failed result
        failed_result = CrawlResult("https://example.com/failed")
        failed_result.success = False
        failed_result.error = "Connection timeout"
        results.append(failed_result)

        mock_crawler = AsyncMock()
        mock_crawler.crawl_recursive.return_value = results
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

        # Should succeed with partial results
        result = await parser.parse(
            "https://example.com/start",
            crawl_depth=1,
            max_pages=2
        )

        assert result.content == 'Success content'
        assert result.additional_metadata['pages_crawled'] == 1
        assert result.additional_metadata['total_attempts'] == 2

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.SecureWebCrawler')
    async def test_parse_with_security_warnings(self, mock_crawler_class, parser):
        """Test parsing with security warnings present."""
        result_with_warnings = CrawlResult("https://example.com/suspicious")
        result_with_warnings.success = True
        result_with_warnings.content = "<html><body>Content</body></html>"
        result_with_warnings.security_warnings = [
            "Script tag detected",
            "Suspicious pattern found"
        ]
        result_with_warnings.metadata = {'parsed_content': 'Content'}

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = result_with_warnings
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

        result = await parser.parse("https://example.com/suspicious")

        assert result.content == 'Content'
        assert len(result.additional_metadata['security_warnings']) == 2
        assert result.parsing_info['security_status'] == 'warnings_found'

    @pytest.mark.asyncio
    async def test_parse_options_validation(self, parser):
        """Test validation of parsing options."""
        # Test with various option combinations
        valid_options = [
            {'crawl_depth': 2, 'max_pages': 10},
            {'domain_allowlist': ['example.com', 'test.org']},
            {'request_delay': 2.0, 'respect_robots_txt': False},
            {'enable_security_scan': False, 'quarantine_suspicious': False},
        ]

        for options in valid_options:
            # Should not raise exception with valid options
            config = parser._create_config_from_options(**options)
            assert isinstance(config, SecurityConfig)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])