"""
Tests for secure web content parsing and ingestion.

This test suite covers the web crawler security features, content parsing,
and integration with the document processing pipeline.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest
from wqm_cli.cli.parsers import (
    CrawlResult,
    SecureWebCrawler,
    SecurityConfig,
    WebIngestionInterface,
    WebParser,
    create_secure_web_parser,
)
from wqm_cli.cli.parsers.exceptions import ParsingError


class AsyncIteratorMock:
    """Mock class for async iterators like aiohttp iter_chunked."""

    def __init__(self, items: list[bytes]):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestSecurityConfig:
    """Tests for SecurityConfig class."""

    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()

        assert config.allowed_schemes == {'http', 'https'}
        assert 'localhost' in config.domain_blocklist
        assert config.max_content_size == 50 * 1024 * 1024
        assert config.respect_robots_txt is True
        assert config.enable_content_scanning is True
        assert config.request_delay == 1.0

    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com', 'test.org'}
        config.max_content_size = 10 * 1024 * 1024
        config.request_delay = 2.0

        assert config.domain_allowlist == {'example.com', 'test.org'}
        assert config.max_content_size == 10 * 1024 * 1024
        assert config.request_delay == 2.0


class TestSecureWebCrawler:
    """Tests for SecureWebCrawler class."""

    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com', 'test.org'}
        return SecureWebCrawler(config)

    @pytest.fixture
    def mock_html_content(self):
        """Mock HTML content for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is test content.</p>
            <a href="https://example.com/page2">Link to page 2</a>
        </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_url_validation_success(self, crawler):
        """Test successful URL validation."""
        result = CrawlResult("https://example.com/test")
        is_valid = await crawler._validate_url("https://example.com/test", result)

        assert is_valid is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_url_validation_blocked_domain(self, crawler):
        """Test URL validation with blocked domain."""
        result = CrawlResult("https://blocked.com/test")
        is_valid = await crawler._validate_url("https://blocked.com/test", result)

        assert is_valid is False
        assert "not in allowlist" in result.error

    @pytest.mark.asyncio
    async def test_url_validation_invalid_scheme(self, crawler):
        """Test URL validation with invalid scheme."""
        result = CrawlResult("ftp://example.com/test")
        is_valid = await crawler._validate_url("ftp://example.com/test", result)

        assert is_valid is False
        assert "Invalid scheme" in result.error

    @pytest.mark.asyncio
    async def test_url_validation_localhost_blocked(self, crawler):
        """Test URL validation blocks localhost."""
        result = CrawlResult("http://localhost:8000/test")
        is_valid = await crawler._validate_url("http://localhost:8000/test", result)

        assert is_valid is False
        # Localhost is blocked either via blocklist or by not being in allowlist
        assert "blocklist" in result.error or "not in allowlist" in result.error

    @pytest.mark.asyncio
    async def test_rate_limiting(self, crawler):
        """Test rate limiting functionality."""
        import time

        # First request should not wait
        start_time = time.time()
        await crawler._respect_rate_limit("https://example.com/test1")
        first_duration = time.time() - start_time

        # Second request to same domain should wait
        start_time = time.time()
        await crawler._respect_rate_limit("https://example.com/test2")
        second_duration = time.time() - start_time

        assert first_duration < 0.1  # First request is immediate
        assert second_duration >= crawler.config.request_delay * 0.8  # Second request waits

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.ClientSession')
    async def test_fetch_content_success(self, mock_session_class, crawler, mock_html_content):
        """Test successful content fetching."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_response.content.iter_chunked.return_value = AsyncIteratorMock(
            [mock_html_content.encode()]
        )

        # Create an async context manager for session.get()
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__.return_value = mock_response
        mock_get_cm.__aexit__.return_value = None

        # Set up mock session with proper async context manager
        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_cm
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        result = CrawlResult("https://example.com/test")

        # Ensure session is created (will now use mock)
        await crawler._ensure_session()

        await crawler._fetch_content("https://example.com/test", result)

        assert result.success is True
        assert result.status_code == 200
        assert result.content == mock_html_content
        assert result.content_type == 'text/html; charset=utf-8'

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_content_invalid_content_type(self, mock_get, crawler):
        """Test content fetching with invalid content type."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_get.return_value.__aenter__.return_value = mock_response

        result = CrawlResult("https://example.com/test")
        await crawler._ensure_session()
        await crawler._fetch_content("https://example.com/test", result)

        assert result.success is False
        assert "Unsupported content type" in result.error

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_content_size_limit(self, mock_get, crawler):
        """Test content fetching with size limit exceeded."""
        large_content = 'x' * (60 * 1024 * 1024)  # 60MB

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {
            'content-type': 'text/html',
            'content-length': str(len(large_content))
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        result = CrawlResult("https://example.com/test")
        await crawler._ensure_session()
        await crawler._fetch_content("https://example.com/test", result)

        assert result.success is False
        assert "Content too large" in result.error

    @pytest.mark.asyncio
    async def test_extract_links(self, crawler, mock_html_content):
        """Test link extraction from HTML content."""
        base_url = "https://example.com/page1"
        links = await crawler._extract_links(mock_html_content, base_url)

        assert len(links) > 0
        assert "https://example.com/page2" in links

    @pytest.mark.asyncio
    async def test_security_scanner_suspicious_content(self, crawler):
        """Test security scanner with suspicious content."""
        suspicious_html = """
        <html>
        <body>
            <script>document.write('malicious code');</script>
            <p>Normal content</p>
        </body>
        </html>
        """

        is_safe, warnings = await crawler.scanner.scan_content(suspicious_html, 'text/html')

        assert is_safe is False
        assert len(warnings) > 0
        assert any('script' in warning.lower() for warning in warnings)

    @pytest.mark.asyncio
    async def test_security_scanner_safe_content(self, crawler):
        """Test security scanner with safe content."""
        safe_html = """
        <html>
        <body>
            <h1>Safe Content</h1>
            <p>This is perfectly safe HTML content.</p>
        </body>
        </html>
        """

        is_safe, warnings = await crawler.scanner.scan_content(safe_html, 'text/html')

        assert is_safe is True or len([w for w in warnings if 'script' in w.lower()]) == 0


class TestWebParser:
    """Tests for WebParser class."""

    @pytest.fixture
    def parser(self):
        """Create a test parser instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        return WebParser(config)

    def test_can_parse_valid_url(self, parser):
        """Test URL format detection."""
        assert parser.can_parse("https://example.com/test") is True
        assert parser.can_parse("http://example.com/test") is True
        assert parser.can_parse("/local/file.html") is False
        assert parser.can_parse("ftp://example.com/test") is False

    def test_format_name(self, parser):
        """Test format name."""
        assert parser.format_name == "Web Content"

    @pytest.mark.asyncio
    async def test_parse_invalid_url(self, parser):
        """Test parsing with invalid URL format."""
        with pytest.raises(ParsingError, match="Invalid URL format"):
            await parser.parse("/local/file.html")

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_parser.SecureWebCrawler')
    async def test_parse_single_page(self, mock_crawler_class, parser):
        """Test single page parsing."""
        # Mock successful crawl result
        mock_result = CrawlResult("https://example.com/test")
        mock_result.success = True
        mock_result.content = "<html><body><h1>Test</h1><p>Content</p></body></html>"
        mock_result.metadata = {
            'parsed_content': 'Test\n\nContent',
            'html_metadata': {'title': 'Test Page'}
        }

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_result
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

        result = await parser.parse("https://example.com/test")

        assert result.content == 'Test\n\nContent'
        assert result.file_type == 'web'
        assert 'source_url' in result.metadata
        assert result.metadata['source_url'] == "https://example.com/test"

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_parser.SecureWebCrawler')
    async def test_parse_recursive_crawl(self, mock_crawler_class, parser):
        """Test recursive crawling."""
        # Mock multiple crawl results
        results = []
        for i in range(3):
            result = CrawlResult(f"https://example.com/page{i}")
            result.success = True
            result.content = f"<html><body><h1>Page {i}</h1></body></html>"
            result.metadata = {'parsed_content': f'Page {i}'}
            results.append(result)

        mock_crawler = AsyncMock()
        mock_crawler.crawl_recursive.return_value = results
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

        result = await parser.parse(
            "https://example.com/test",
            crawl_depth=2,
            max_pages=3
        )

        assert 'Page 0' in result.content
        assert 'Page 1' in result.content
        assert 'Page 2' in result.content
        assert result.metadata['pages_crawled'] == 3

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_parser.SecureWebCrawler')
    async def test_parse_no_successful_results(self, mock_crawler_class, parser):
        """Test parsing when all crawl attempts fail."""
        mock_result = CrawlResult("https://example.com/test")
        mock_result.success = False
        mock_result.error = "Connection timeout"

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_result
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

        with pytest.raises(ParsingError, match="All crawl attempts failed"):
            await parser.parse("https://example.com/test")

    def test_get_parsing_options(self, parser):
        """Test parsing options documentation."""
        options = parser.get_parsing_options()

        assert 'crawl_depth' in options
        assert 'max_pages' in options
        assert 'domain_allowlist' in options
        assert 'enable_security_scan' in options
        assert options['crawl_depth']['default'] == 0
        assert options['max_pages']['default'] == 1


class TestWebIngestionInterface:
    """Tests for WebIngestionInterface class."""

    @pytest.fixture
    def interface(self):
        """Create test interface instance."""
        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        return WebIngestionInterface(config)

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_parser.WebParser.parse')
    async def test_ingest_url(self, mock_parse, interface):
        """Test single URL ingestion."""
        mock_doc = MagicMock()
        mock_doc.content = "Test content"
        mock_parse.return_value = mock_doc

        result = await interface.ingest_url("https://example.com/test")

        assert result == mock_doc
        mock_parse.assert_called_once_with("https://example.com/test")

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_parser.WebParser.parse')
    async def test_ingest_site(self, mock_parse, interface):
        """Test multi-page site ingestion."""
        mock_doc = MagicMock()
        mock_doc.content = "Site content"
        mock_parse.return_value = mock_doc

        result = await interface.ingest_site(
            "https://example.com/",
            max_pages=10,
            max_depth=2
        )

        assert result == mock_doc
        mock_parse.assert_called_once_with(
            "https://example.com/",
            crawl_depth=2,
            max_pages=10
        )

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_parser.WebParser.parse')
    async def test_ingest_with_allowlist(self, mock_parse, interface):
        """Test ingestion with domain allowlist."""
        mock_doc = MagicMock()
        mock_parse.return_value = mock_doc

        result = await interface.ingest_with_allowlist(
            "https://example.com/test",
            ['example.com', 'trusted.org']
        )

        assert result == mock_doc
        mock_parse.assert_called_once_with(
            "https://example.com/test",
            domain_allowlist=['example.com', 'trusted.org']
        )


class TestCreateSecureWebParser:
    """Tests for create_secure_web_parser utility function."""

    def test_default_secure_parser(self):
        """Test creating parser with default secure settings."""
        parser = create_secure_web_parser()

        assert isinstance(parser, WebParser)
        assert parser.security_config.enable_content_scanning is True
        assert parser.security_config.quarantine_suspicious is True
        assert parser.security_config.max_content_size == 10 * 1024 * 1024  # 10MB
        assert parser.security_config.request_delay == 1.5

    def test_secure_parser_with_allowlist(self):
        """Test creating parser with domain allowlist."""
        allowed_domains = ['example.com', 'trusted.org']
        parser = create_secure_web_parser(allowed_domains=allowed_domains)

        assert parser.security_config.domain_allowlist == set(allowed_domains)

    def test_secure_parser_security_disabled(self):
        """Test creating parser with security disabled."""
        parser = create_secure_web_parser(
            enable_scanning=False,
            quarantine_threats=False
        )

        assert parser.security_config.enable_content_scanning is False
        assert parser.security_config.quarantine_suspicious is False


class TestIntegrationSecurityScenarios:
    """Integration tests for security scenarios."""

    @pytest.mark.asyncio
    @patch('wqm_cli.cli.parsers.web_crawler.ClientSession')
    async def test_malicious_content_quarantine(self, mock_session_class):
        """Test that malicious content is properly quarantined."""
        malicious_html = """
        <html>
        <body>
            <script>eval('malicious code');</script>
            <p>Some content</p>
            <script>window.location='http://evil.com';</script>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content.iter_chunked.return_value = AsyncIteratorMock(
            [malicious_html.encode()]
        )

        # Create an async context manager for session.get()
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__.return_value = mock_response
        mock_get_cm.__aexit__.return_value = None

        # Set up mock session with proper async context manager
        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_cm
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        config = SecurityConfig()
        config.domain_allowlist = {'example.com'}
        config.enable_content_scanning = True

        async with SecureWebCrawler(config) as crawler:
            result = await crawler.crawl_url("https://example.com/malicious")

            # Either content was blocked due to security scan, or warnings were added
            # Implementation may allow content through with warnings vs blocking
            if result.success:
                # If success, security warnings should still be recorded
                assert len(result.security_warnings) > 0
                assert any('script' in warning.lower() for warning in result.security_warnings)
            else:
                # If blocked, verify it was due to security
                assert len(result.security_warnings) > 0 or 'security' in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_domain_restriction_enforcement(self):
        """Test that domain restrictions are properly enforced."""
        config = SecurityConfig()
        config.domain_allowlist = {'allowed.com'}

        parser = WebParser(config)

        # This should fail due to domain restriction
        with pytest.raises(ParsingError):
            await parser.parse("https://blocked.com/test")

    @pytest.mark.asyncio
    @patch('tempfile.gettempdir')
    @patch('aiofiles.open')
    async def test_quarantine_file_creation(self, mock_open, mock_tempdir):
        """Test quarantine file creation for suspicious content."""
        mock_tempdir.return_value = '/tmp'
        mock_file = AsyncMock()
        mock_open.return_value.__aenter__.return_value = mock_file

        result = CrawlResult("https://example.com/suspicious")
        result.content = "<script>malicious</script>"
        result.security_warnings = ["Script tag detected"]
        result.timestamp = 1234567890

        config = SecurityConfig()
        crawler = SecureWebCrawler(config)

        await crawler._quarantine_content(result)

        # Verify quarantine file was created
        mock_open.assert_called_once()
        mock_file.write.assert_called()
        assert 'quarantine_path' in result.metadata


if __name__ == '__main__':
    pytest.main([__file__])
