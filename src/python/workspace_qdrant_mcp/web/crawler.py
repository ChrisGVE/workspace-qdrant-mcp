"""Web crawler with respectful rate limiting and robots.txt compliance.

This module implements a responsible web crawler that respects robots.txt,
implements proper rate limiting, and handles various error conditions gracefully.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import tldextract
from loguru import logger


@dataclass
class CrawlResponse:
    """Response from a web crawl request."""

    url: str
    status_code: int
    content: str
    headers: Dict[str, str]
    content_type: str
    encoding: str
    redirect_url: Optional[str] = None
    crawl_time: datetime = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.crawl_time is None:
            self.crawl_time = datetime.now()


@dataclass
class CrawlConfig:
    """Configuration for web crawler behavior."""

    # Rate limiting
    delay_between_requests: float = 1.0
    concurrent_requests: int = 5
    max_requests_per_second: float = 2.0

    # Request settings
    timeout: float = 30.0
    max_redirects: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0

    # Content settings
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: Set[str] = None

    # User agent and headers
    user_agent: str = "ResponsibleWebCrawler/1.0"
    custom_headers: Dict[str, str] = None

    # Robots.txt compliance
    respect_robots_txt: bool = True
    robots_txt_cache_ttl: int = 3600  # 1 hour

    def __post_init__(self):
        if self.allowed_content_types is None:
            self.allowed_content_types = {
                'text/html', 'text/plain', 'text/xml',
                'application/xml', 'application/xhtml+xml',
                'application/json', 'text/css', 'text/javascript'
            }

        if self.custom_headers is None:
            self.custom_headers = {}


class RateLimiter:
    """Rate limiter for respectful crawling."""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.last_request_times: Dict[str, float] = {}
        self.request_counts: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, domain: str) -> None:
        """Wait if necessary to respect rate limits for a domain."""
        async with self._lock:
            current_time = time.time()

            # Check delay between requests
            if domain in self.last_request_times:
                time_since_last = current_time - self.last_request_times[domain]
                if time_since_last < self.config.delay_between_requests:
                    wait_time = self.config.delay_between_requests - time_since_last
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
                    await asyncio.sleep(wait_time)
                    current_time = time.time()

            # Check requests per second limit
            if domain not in self.request_counts:
                self.request_counts[domain] = []

            # Remove old request times (older than 1 second)
            cutoff_time = current_time - 1.0
            self.request_counts[domain] = [
                t for t in self.request_counts[domain] if t > cutoff_time
            ]

            # Check if we're exceeding requests per second
            if len(self.request_counts[domain]) >= self.config.max_requests_per_second:
                wait_time = 1.0 - (current_time - self.request_counts[domain][0])
                if wait_time > 0:
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for RPS limit on {domain}")
                    await asyncio.sleep(wait_time)
                    current_time = time.time()

            # Record this request
            self.last_request_times[domain] = current_time
            self.request_counts[domain].append(current_time)


class RobotsChecker:
    """Robots.txt compliance checker with caching."""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.robots_cache: Dict[str, Tuple[RobotFileParser, datetime]] = {}
        self._lock = asyncio.Lock()

    def _get_robots_url(self, url: str) -> str:
        """Get robots.txt URL for a given URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    async def _fetch_robots_txt(self, session: aiohttp.ClientSession, robots_url: str) -> Optional[str]:
        """Fetch robots.txt content."""
        try:
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with session.get(robots_url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"Fetched robots.txt from {robots_url}")
                    return content
                else:
                    logger.debug(f"Robots.txt not found at {robots_url} (status: {response.status})")
                    return None
        except Exception as e:
            logger.debug(f"Error fetching robots.txt from {robots_url}: {e}")
            return None

    async def can_crawl(self, session: aiohttp.ClientSession, url: str) -> bool:
        """Check if URL can be crawled according to robots.txt."""
        if not self.config.respect_robots_txt:
            return True

        try:
            async with self._lock:
                robots_url = self._get_robots_url(url)
                domain = urlparse(url).netloc
                current_time = datetime.now()

                # Check cache
                if domain in self.robots_cache:
                    robots_parser, fetch_time = self.robots_cache[domain]
                    if current_time - fetch_time < timedelta(seconds=self.config.robots_txt_cache_ttl):
                        return robots_parser.can_fetch(self.config.user_agent, url)

                # Fetch and parse robots.txt
                robots_content = await self._fetch_robots_txt(session, robots_url)
                robots_parser = RobotFileParser()

                if robots_content:
                    robots_parser.set_url(robots_url)
                    robots_parser.read()
                    # Parse the content manually since set_url doesn't actually fetch
                    for line in robots_content.split('\n'):
                        robots_parser.errcode = 200
                        break
                    # Store content for parsing
                    robots_parser._RobotFileParser__entries = []
                    try:
                        # Use the content directly
                        import io
                        robots_parser.read_file = lambda: io.StringIO(robots_content)
                        robots_parser.read()
                    except Exception:
                        # Fallback: assume allowed if parsing fails
                        logger.warning(f"Failed to parse robots.txt for {domain}, allowing crawl")
                        return True
                else:
                    # No robots.txt found, allow crawling
                    robots_parser.allow_all = True

                # Cache the result
                self.robots_cache[domain] = (robots_parser, current_time)

                return robots_parser.can_fetch(self.config.user_agent, url)

        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}, allowing crawl")
            return True


class WebCrawler:
    """Respectful web crawler with rate limiting and robots.txt compliance."""

    def __init__(self, config: Optional[CrawlConfig] = None):
        self.config = config or CrawlConfig()
        self.rate_limiter = RateLimiter(self.config)
        self.robots_checker = RobotsChecker(self.config)
        self.session: Optional[aiohttp.ClientSession] = None
        self._active_crawls: Set[str] = set()
        self._semaphore = asyncio.Semaphore(self.config.concurrent_requests)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Initialize the crawler session."""
        if self.session is None:
            headers = {
                'User-Agent': self.config.user_agent,
                **self.config.custom_headers
            }

            connector = aiohttp.TCPConnector(
                limit=self.config.concurrent_requests * 2,
                limit_per_host=self.config.concurrent_requests,
                ttl_dns_cache=300,
                use_dns_cache=True
            )

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            self.session = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout
            )
            logger.info("Web crawler session started")

    async def stop(self):
        """Close the crawler session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Web crawler session stopped")

    def _get_domain(self, url: str) -> str:
        """Get domain from URL using tldextract for better parsing."""
        try:
            extracted = tldextract.extract(url)
            return f"{extracted.domain}.{extracted.suffix}"
        except Exception:
            # Fallback to basic parsing
            return urlparse(url).netloc

    async def _make_request(self, url: str) -> CrawlResponse:
        """Make a single HTTP request with retries and error handling."""
        if not self.session:
            raise RuntimeError("Crawler not started. Use async with or call start() first.")

        domain = self._get_domain(url)
        retries = 0

        while retries <= self.config.max_retries:
            try:
                # Rate limiting
                await self.rate_limiter.wait_if_needed(domain)

                # Robots.txt check
                if not await self.robots_checker.can_crawl(self.session, url):
                    return CrawlResponse(
                        url=url,
                        status_code=403,
                        content="",
                        headers={},
                        content_type="",
                        encoding="",
                        error="Blocked by robots.txt"
                    )

                logger.debug(f"Crawling URL: {url} (attempt {retries + 1})")

                async with self.session.get(url, max_redirects=self.config.max_redirects) as response:
                    # Check content type
                    content_type = response.headers.get('content-type', '').split(';')[0].lower()
                    if content_type not in self.config.allowed_content_types:
                        return CrawlResponse(
                            url=url,
                            status_code=response.status,
                            content="",
                            headers=dict(response.headers),
                            content_type=content_type,
                            encoding="",
                            error=f"Content type not allowed: {content_type}"
                        )

                    # Check content size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.config.max_content_size:
                        return CrawlResponse(
                            url=url,
                            status_code=response.status,
                            content="",
                            headers=dict(response.headers),
                            content_type=content_type,
                            encoding="",
                            error=f"Content too large: {content_length} bytes"
                        )

                    # Read content with size limit
                    content_chunks = []
                    content_size = 0

                    async for chunk in response.content.iter_chunked(8192):
                        content_size += len(chunk)
                        if content_size > self.config.max_content_size:
                            return CrawlResponse(
                                url=url,
                                status_code=response.status,
                                content="",
                                headers=dict(response.headers),
                                content_type=content_type,
                                encoding="",
                                error=f"Content exceeded size limit: {content_size} bytes"
                            )
                        content_chunks.append(chunk)

                    content_bytes = b''.join(content_chunks)

                    # Determine encoding
                    encoding = response.get_encoding()
                    try:
                        content = content_bytes.decode(encoding or 'utf-8')
                    except UnicodeDecodeError:
                        # Try common encodings
                        for enc in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                content = content_bytes.decode(enc)
                                encoding = enc
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            content = content_bytes.decode('utf-8', errors='replace')
                            encoding = 'utf-8'

                    # Determine redirect URL
                    redirect_url = None
                    if str(response.url) != url:
                        redirect_url = str(response.url)

                    return CrawlResponse(
                        url=url,
                        status_code=response.status,
                        content=content,
                        headers=dict(response.headers),
                        content_type=content_type,
                        encoding=encoding,
                        redirect_url=redirect_url
                    )

            except asyncio.TimeoutError:
                retries += 1
                error_msg = f"Request timeout for {url}"
                if retries > self.config.max_retries:
                    logger.warning(f"{error_msg} (max retries exceeded)")
                    return CrawlResponse(
                        url=url,
                        status_code=0,
                        content="",
                        headers={},
                        content_type="",
                        encoding="",
                        error=error_msg
                    )
                else:
                    logger.debug(f"{error_msg} (retry {retries}/{self.config.max_retries})")
                    await asyncio.sleep(self.config.retry_delay * retries)

            except aiohttp.ClientError as e:
                retries += 1
                error_msg = f"Client error for {url}: {e}"
                if retries > self.config.max_retries:
                    logger.warning(f"{error_msg} (max retries exceeded)")
                    return CrawlResponse(
                        url=url,
                        status_code=0,
                        content="",
                        headers={},
                        content_type="",
                        encoding="",
                        error=error_msg
                    )
                else:
                    logger.debug(f"{error_msg} (retry {retries}/{self.config.max_retries})")
                    await asyncio.sleep(self.config.retry_delay * retries)

            except Exception as e:
                error_msg = f"Unexpected error for {url}: {e}"
                logger.error(error_msg)
                return CrawlResponse(
                    url=url,
                    status_code=0,
                    content="",
                    headers={},
                    content_type="",
                    encoding="",
                    error=error_msg
                )

        # Should not reach here
        return CrawlResponse(
            url=url,
            status_code=0,
            content="",
            headers={},
            content_type="",
            encoding="",
            error="Max retries exceeded"
        )

    async def crawl_url(self, url: str) -> CrawlResponse:
        """Crawl a single URL with rate limiting and compliance checks."""
        async with self._semaphore:
            if url in self._active_crawls:
                logger.warning(f"URL {url} is already being crawled, skipping")
                return CrawlResponse(
                    url=url,
                    status_code=0,
                    content="",
                    headers={},
                    content_type="",
                    encoding="",
                    error="Duplicate crawl request"
                )

            self._active_crawls.add(url)
            try:
                return await self._make_request(url)
            finally:
                self._active_crawls.discard(url)

    async def crawl_urls(self, urls: List[str]) -> List[CrawlResponse]:
        """Crawl multiple URLs concurrently with rate limiting."""
        if not urls:
            return []

        logger.info(f"Starting crawl of {len(urls)} URLs")
        tasks = [self.crawl_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error crawling {urls[i]}: {result}")
                responses.append(CrawlResponse(
                    url=urls[i],
                    status_code=0,
                    content="",
                    headers={},
                    content_type="",
                    encoding="",
                    error=str(result)
                ))
            else:
                responses.append(result)

        logger.info(f"Completed crawl of {len(urls)} URLs")
        return responses