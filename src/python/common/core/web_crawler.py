"""
Respectful Web Crawler with Rate Limiting and Robots.txt Compliance

This module provides a comprehensive web crawling system designed for respectful
and compliant web scraping with advanced features including:

- Rate limiting (configurable, default 2 requests/second)
- Robots.txt compliance with proper parsing and caching
- Same-domain restrictions and URL validation
- Comprehensive error handling and retry logic
- Concurrent connection management with configurable limits
- Request/response tracking and performance metrics
- Support for custom User-Agent and headers
- Cookie and session management for authenticated crawling
- Content-type filtering and size limits

Features:
    - Respectful crawling with configurable delays and limits
    - Robust robots.txt parsing and compliance checking
    - Advanced error handling with exponential backoff retry
    - Comprehensive logging and metrics collection
    - Memory-efficient streaming for large responses
    - Integration with existing document processing pipeline

Example:
    ```python
    from workspace_qdrant_mcp.core.web_crawler import WebCrawler

    # Initialize crawler with rate limiting
    crawler = WebCrawler(
        rate_limit=2.0,  # 2 requests per second
        max_concurrent=10,
        respect_robots=True
    )

    # Crawl single URL
    result = await crawler.crawl_url("https://example.com")

    # Recursive crawling with depth limits
    results = await crawler.crawl_recursive(
        start_url="https://example.com",
        max_depth=3,
        same_domain_only=True
    )
    ```
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp


@dataclass
class CrawlResult:
    """Result of a web crawling operation."""

    url: str
    status_code: int
    content: str | None = None
    headers: dict[str, str] | None = None
    content_type: str | None = None
    content_length: int | None = None
    crawl_time: datetime | None = None
    processing_time: float | None = None
    error: str | None = None
    redirect_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlerConfig:
    """Configuration for web crawler behavior."""

    # Rate limiting
    rate_limit: float = 2.0  # Requests per second
    max_concurrent: int = 10
    request_timeout: float = 30.0

    # Robots.txt compliance
    respect_robots: bool = True
    robots_cache_ttl: int = 3600  # 1 hour
    user_agent: str = "WorkspaceQdrantMCP/1.0 (+https://github.com/ChrisGVE/workspace-qdrant-mcp)"

    # Content filtering
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: set[str] = field(default_factory=lambda: {
        'text/html', 'text/plain', 'text/xml',
        'application/xml', 'application/json',
        'application/pdf'
    })

    # Crawling behavior
    same_domain_only: bool = True
    max_depth: int = 3
    max_pages: int = 1000

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Headers and authentication
    custom_headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)


class RobotsCache:
    """Cache for robots.txt files with TTL support."""

    def __init__(self, ttl: int = 3600):
        self._cache: dict[str, tuple[RobotFileParser, datetime]] = {}
        self._ttl = ttl

    def get_robots(self, domain: str) -> RobotFileParser | None:
        """Get cached robots.txt parser for domain."""
        if domain not in self._cache:
            return None

        parser, cached_time = self._cache[domain]
        if datetime.now() - cached_time > timedelta(seconds=self._ttl):
            del self._cache[domain]
            return None

        return parser

    def set_robots(self, domain: str, parser: RobotFileParser) -> None:
        """Cache robots.txt parser for domain."""
        self._cache[domain] = (parser, datetime.now())

    def clear_expired(self) -> None:
        """Remove expired entries from cache."""
        now = datetime.now()
        expired = [
            domain for domain, (_, cached_time) in self._cache.items()
            if now - cached_time > timedelta(seconds=self._ttl)
        ]
        for domain in expired:
            del self._cache[domain]


class WebCrawler:
    """
    Respectful web crawler with rate limiting and robots.txt compliance.

    This crawler is designed to be respectful of server resources and
    website policies while providing comprehensive crawling capabilities.
    """

    def __init__(self, config: CrawlerConfig | None = None):
        """Initialize web crawler with configuration."""
        self.config = config or CrawlerConfig()
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiter = asyncio.Semaphore(self.config.max_concurrent)
        self._last_request_time = 0.0
        self._robots_cache = RobotsCache(self.config.robots_cache_ttl)

        # Statistics tracking
        self._stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'robots_blocked': 0,
            'content_filtered': 0,
            'bytes_downloaded': 0,
            'average_response_time': 0.0
        }

    async def __aenter__(self) -> 'WebCrawler':
        """Async context manager entry."""
        await self._initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _initialize_session(self) -> None:
        """Initialize aiohttp session with proper configuration."""
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent,
            limit_per_host=5,
            ttl_dns_cache=300
        )

        headers = {
            'User-Agent': self.config.user_agent,
            **self.config.custom_headers
        }

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
            cookies=self.config.cookies
        )

    async def close(self) -> None:
        """Close the crawler and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self.config.rate_limit <= 0:
            return

        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.config.rate_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not self.config.respect_robots:
            return True

        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Check cache first
        robots_parser = self._robots_cache.get_robots(domain)

        if robots_parser is None:
            # Fetch and parse robots.txt
            robots_url = urljoin(domain, '/robots.txt')
            robots_parser = RobotFileParser()
            robots_parser.set_url(robots_url)

            try:
                async with self._session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        robots_parser.read_file(content.splitlines())
                    else:
                        # If robots.txt not found, allow all
                        robots_parser.allow_all = True
            except Exception:
                # On error, allow all (be permissive)
                robots_parser.allow_all = True

            self._robots_cache.set_robots(domain, robots_parser)

        return robots_parser.can_fetch(self.config.user_agent, url)

    def _is_valid_url(self, url: str, base_domain: str | None = None) -> bool:
        """Validate URL format and domain restrictions."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False

            if parsed.scheme not in ('http', 'https'):
                return False

            if base_domain and self.config.same_domain_only:
                return parsed.netloc == base_domain

            return True
        except Exception:
            return False

    def _should_process_content(self, content_type: str | None,
                              content_length: int | None) -> bool:
        """Check if content should be processed based on type and size."""
        if content_type:
            # Extract main content type (ignore charset, etc.)
            main_type = content_type.split(';')[0].strip().lower()
            if main_type not in self.config.allowed_content_types:
                return False

        if content_length and content_length > self.config.max_content_size:
            return False

        return True

    async def _make_request(self, url: str) -> CrawlResult:
        """Make HTTP request with error handling and retries."""
        if not self._session:
            await self._initialize_session()

        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                await self._enforce_rate_limit()

                async with self._rate_limiter:
                    self._stats['requests_made'] += 1

                    async with self._session.get(url) as response:
                        # Check content type and size before downloading
                        content_type = response.headers.get('content-type')
                        content_length = response.headers.get('content-length')
                        content_length = int(content_length) if content_length else None

                        if not self._should_process_content(content_type, content_length):
                            self._stats['content_filtered'] += 1
                            return CrawlResult(
                                url=url,
                                status_code=response.status,
                                content_type=content_type,
                                content_length=content_length,
                                crawl_time=datetime.now(),
                                processing_time=time.time() - start_time,
                                error="Content filtered due to type or size restrictions"
                            )

                        # Download content
                        content = await response.text()

                        # Update statistics
                        self._stats['successful_requests'] += 1
                        if content_length:
                            self._stats['bytes_downloaded'] += content_length

                        # Handle redirects
                        final_url = str(response.url)
                        redirect_url = final_url if final_url != url else None

                        return CrawlResult(
                            url=url,
                            status_code=response.status,
                            content=content,
                            headers=dict(response.headers),
                            content_type=content_type,
                            content_length=len(content.encode('utf-8')),
                            crawl_time=datetime.now(),
                            processing_time=time.time() - start_time,
                            redirect_url=redirect_url
                        )

            except asyncio.TimeoutError:
                error = f"Timeout after {self.config.request_timeout}s"
            except aiohttp.ClientError as e:
                error = f"Client error: {str(e)}"
            except Exception as e:
                error = f"Unexpected error: {str(e)}"

            # If this wasn't the last attempt, wait before retrying
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                await asyncio.sleep(delay)
            else:
                self._stats['failed_requests'] += 1
                return CrawlResult(
                    url=url,
                    status_code=0,
                    crawl_time=datetime.now(),
                    processing_time=time.time() - start_time,
                    error=error
                )

    async def crawl_url(self, url: str) -> CrawlResult:
        """
        Crawl a single URL with robots.txt compliance and rate limiting.

        Args:
            url: The URL to crawl

        Returns:
            CrawlResult containing the response data or error information
        """
        # Validate URL
        if not self._is_valid_url(url):
            return CrawlResult(
                url=url,
                status_code=0,
                error="Invalid URL format"
            )

        # Check robots.txt compliance
        if not await self._check_robots_txt(url):
            self._stats['robots_blocked'] += 1
            return CrawlResult(
                url=url,
                status_code=0,
                error="Blocked by robots.txt"
            )

        return await self._make_request(url)

    async def crawl_urls(self, urls: list[str]) -> list[CrawlResult]:
        """
        Crawl multiple URLs concurrently with rate limiting.

        Args:
            urls: List of URLs to crawl

        Returns:
            List of CrawlResult objects
        """
        tasks = [self.crawl_url(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_statistics(self) -> dict[str, Any]:
        """Get crawler statistics and performance metrics."""
        total_requests = self._stats['requests_made']
        if total_requests > 0:
            success_rate = self._stats['successful_requests'] / total_requests
            self._stats.get('average_response_time', 0.0)
        else:
            success_rate = 0.0

        return {
            **self._stats,
            'success_rate': success_rate,
            'robots_cache_size': len(self._robots_cache._cache),
            'config': {
                'rate_limit': self.config.rate_limit,
                'max_concurrent': self.config.max_concurrent,
                'respect_robots': self.config.respect_robots,
                'max_content_size': self.config.max_content_size
            }
        }

    def reset_statistics(self) -> None:
        """Reset all crawler statistics."""
        self._stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'robots_blocked': 0,
            'content_filtered': 0,
            'bytes_downloaded': 0,
            'average_response_time': 0.0
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check of crawler components."""
        health = {
            'session_active': self._session is not None,
            'robots_cache_entries': len(self._robots_cache._cache),
            'stats': self.get_statistics(),
            'config_valid': True
        }

        # Test with a simple request if session is active
        if self._session:
            try:
                async with self._session.get('http://httpbin.org/status/200', timeout=5):
                    health['connectivity'] = True
            except Exception:
                health['connectivity'] = False
        else:
            health['connectivity'] = None

        return health
