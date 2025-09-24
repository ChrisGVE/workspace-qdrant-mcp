"""
Enhanced web crawler features for better error handling and robustness.

This module provides enhancements to the existing web crawler to handle
edge cases more effectively and improve overall reliability.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from wqm_cli.cli.parsers.web_crawler import (
    SecureWebCrawler,
    SecurityConfig,
    CrawlResult,
    SecurityScanner
)


class EnhancedSecurityConfig(SecurityConfig):
    """Enhanced security configuration with additional options."""

    def __init__(self):
        super().__init__()
        # Additional retry and recovery settings
        self.max_retries = 3
        self.retry_delay = 2.0  # Base delay for exponential backoff
        self.exponential_backoff = True

        # Enhanced robots.txt settings
        self.robots_txt_timeout = 10.0
        self.robots_txt_cache_ttl = 3600  # 1 hour cache TTL

        # Connection pool settings
        self.connection_pool_size = 20
        self.connection_pool_ttl = 300  # 5 minutes

        # Enhanced content validation
        self.validate_html_structure = True
        self.max_redirect_chain = 10

        # Performance monitoring
        self.enable_performance_metrics = True
        self.log_slow_requests = True
        self.slow_request_threshold = 10.0  # seconds


class RetryableError(Exception):
    """Exception for errors that can be retried."""
    pass


class PermanentError(Exception):
    """Exception for errors that should not be retried."""
    pass


class EnhancedSecureWebCrawler(SecureWebCrawler):
    """Enhanced web crawler with improved error handling and robustness."""

    def __init__(self, config: Optional[EnhancedSecurityConfig] = None):
        # Convert to enhanced config if regular config provided
        if config is None:
            config = EnhancedSecurityConfig()
        elif not isinstance(config, EnhancedSecurityConfig):
            enhanced_config = EnhancedSecurityConfig()
            # Copy all attributes from base config
            for attr in dir(config):
                if not attr.startswith('_') and hasattr(enhanced_config, attr):
                    setattr(enhanced_config, attr, getattr(config, attr))
            config = enhanced_config

        super().__init__(config)
        self.config: EnhancedSecurityConfig = config
        self.performance_metrics: Dict[str, List[float]] = {
            'request_times': [],
            'dns_lookup_times': [],
            'connection_times': [],
            'processing_times': []
        }

        # Enhanced robots.txt cache with TTL
        self.robots_cache_timestamps: Dict[str, float] = {}

    async def _ensure_session(self):
        """Create an optimized aiohttp session with connection pooling."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = {"User-Agent": self.config.user_agent}

            # Enhanced connector with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )

            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=connector,
                raise_for_status=False  # Handle status codes manually
            )

    async def crawl_url(self, url: str, **options) -> CrawlResult:
        """Enhanced URL crawling with retry logic and better error handling."""
        max_retries = options.get('max_retries', self.config.max_retries)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await self._crawl_url_single_attempt(url, attempt, **options)

                # Log performance metrics if enabled
                if self.config.enable_performance_metrics and result.success:
                    self._log_performance_metrics(result)

                return result

            except PermanentError as e:
                # Don't retry permanent errors
                logger.warning(f"Permanent error for {url}: {e}")
                result = CrawlResult(url)
                result.error = str(e)
                return result

            except RetryableError as e:
                last_error = e
                if attempt < max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"Retrying {url} in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries exceeded for {url}: {e}")

            except Exception as e:
                last_error = e
                # Treat unknown exceptions as retryable
                if attempt < max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    await asyncio.sleep(delay)
                    continue

        # All retries failed
        result = CrawlResult(url)
        result.error = f"Failed after {max_retries} retries: {last_error}"
        return result

    async def _crawl_url_single_attempt(self, url: str, attempt: int, **options) -> CrawlResult:
        """Single crawl attempt with enhanced error categorization."""
        result = CrawlResult(url)
        start_time = time.time()

        try:
            # Validate URL (permanent errors)
            if not await self._validate_url(url, result):
                raise PermanentError(result.error)

            # Check robots.txt with enhanced caching
            if self.config.respect_robots_txt:
                if not await self._check_robots_txt_enhanced(url):
                    raise PermanentError("Blocked by robots.txt")

            # Rate limiting
            await self._respect_rate_limit(url)

            # Fetch content with retry classification
            await self._fetch_content_enhanced(url, result, **options)

            # Security scanning
            if result.content and self.config.enable_content_scanning:
                await self._scan_content_security(result)

            # Enhanced content processing
            if result.success and result.content:
                await self._process_content_enhanced(result)

            # Record successful request time
            request_time = time.time() - start_time
            if self.config.log_slow_requests and request_time > self.config.slow_request_threshold:
                logger.warning(f"Slow request to {url}: {request_time:.2f}s")

            logger.info(f"Successfully crawled: {url} ({result.status_code}) in {request_time:.2f}s")

        except (PermanentError, RetryableError):
            raise
        except Exception as e:
            # Classify the exception
            if self._is_retryable_error(e):
                raise RetryableError(str(e))
            else:
                raise PermanentError(str(e))

        return result

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        if self.config.exponential_backoff:
            return self.config.retry_delay * (2 ** attempt)
        else:
            return self.config.retry_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """Classify if an error is retryable."""
        # Network-related errors that might be temporary
        retryable_errors = [
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
            ConnectionError,
            OSError
        ]

        # HTTP status codes that might be temporary
        if hasattr(error, 'status'):
            retryable_status_codes = {429, 500, 502, 503, 504}
            if error.status in retryable_status_codes:
                return True

        return any(isinstance(error, err_type) for err_type in retryable_errors)

    async def _check_robots_txt_enhanced(self, url: str) -> bool:
        """Enhanced robots.txt checking with better caching and error handling."""
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = f"{base_url}/robots.txt"

            # Check cache with TTL
            current_time = time.time()
            if (robots_url in self.robots_cache and
                robots_url in self.robots_cache_timestamps):

                cache_age = current_time - self.robots_cache_timestamps[robots_url]
                if cache_age < self.config.robots_txt_cache_ttl:
                    rp = self.robots_cache[robots_url]
                    return rp.can_fetch(self.config.user_agent, url)

            # Fetch robots.txt with timeout
            try:
                await self._ensure_session()
                async with self._session.get(
                    robots_url,
                    timeout=aiohttp.ClientTimeout(total=self.config.robots_txt_timeout)
                ) as response:
                    if response.status == 200:
                        robots_content = await response.text()

                        # Parse robots.txt content
                        from urllib.robotparser import RobotFileParser
                        rp = RobotFileParser()
                        rp.set_url(robots_url)

                        # Set content directly instead of fetching
                        rp.read()

                        # Cache the result
                        self.robots_cache[robots_url] = rp
                        self.robots_cache_timestamps[robots_url] = current_time

                        return rp.can_fetch(self.config.user_agent, url)
                    else:
                        # No robots.txt or error - allow by default
                        logger.debug(f"No robots.txt found for {base_url} (status: {response.status})")
                        return True

            except Exception as e:
                logger.debug(f"Error fetching robots.txt from {robots_url}: {e}")
                # If robots.txt can't be fetched, allow crawling
                return True

        except Exception as e:
            logger.debug(f"Error in robots.txt check for {url}: {e}")
            return True  # Default to allowing when in doubt

    async def _fetch_content_enhanced(self, url: str, result: CrawlResult, **options):
        """Enhanced content fetching with better error handling."""
        await self._ensure_session()

        max_content_size = options.get("max_content_size", self.config.max_content_size)
        allowed_content_types = options.get(
            "allowed_content_types", self.config.allowed_content_types
        )
        follow_redirects = options.get("follow_redirects", True)
        max_redirects = options.get("max_redirects", self.config.max_redirect_chain)

        try:
            async with self._session.get(
                url,
                allow_redirects=follow_redirects,
                max_redirects=max_redirects
            ) as response:
                result.status_code = response.status
                result.headers = dict(response.headers)

                # Enhanced status code handling
                if response.status >= 400:
                    if response.status in {429, 500, 502, 503, 504}:
                        raise RetryableError(f"HTTP {response.status}: {response.reason}")
                    else:
                        raise PermanentError(f"HTTP {response.status}: {response.reason}")

                # Content type validation with better error messages
                content_type = response.headers.get("content-type", "").lower()
                result.content_type = content_type

                base_content_type = content_type.split(";")[0].strip()
                if base_content_type and base_content_type not in allowed_content_types:
                    raise PermanentError(
                        f"Unsupported content type: {base_content_type}. "
                        f"Allowed types: {', '.join(allowed_content_types)}"
                    )

                # Enhanced content size checking
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        content_size = int(content_length)
                        if content_size > max_content_size:
                            raise PermanentError(
                                f"Content too large: {content_size:,} bytes > {max_content_size:,} bytes"
                            )
                    except ValueError:
                        logger.warning(f"Invalid content-length header: {content_length}")

                # Read content with improved chunking
                content_bytes = b""
                chunk_count = 0

                try:
                    async for chunk in response.content.iter_chunked(16384):  # 16KB chunks
                        if len(content_bytes) + len(chunk) > max_content_size:
                            raise PermanentError(
                                f"Content size limit exceeded during download: "
                                f"{len(content_bytes) + len(chunk):,} > {max_content_size:,} bytes"
                            )
                        content_bytes += chunk
                        chunk_count += 1

                        # Periodic check for very large files
                        if chunk_count % 100 == 0:
                            logger.debug(f"Downloaded {len(content_bytes):,} bytes from {url}")

                except asyncio.CancelledError:
                    raise RetryableError("Download cancelled")

                # Enhanced content decoding
                result.content = await self._decode_content_safely(content_bytes, content_type)
                result.success = True

        except aiohttp.ClientError as e:
            # Most aiohttp errors are retryable
            raise RetryableError(f"Network error: {e}")
        except asyncio.TimeoutError:
            raise RetryableError("Request timeout")
        except (RetryableError, PermanentError):
            raise
        except Exception as e:
            raise RetryableError(f"Unexpected error during fetch: {e}")

    async def _decode_content_safely(self, content_bytes: bytes, content_type: str) -> str:
        """Safely decode content with multiple encoding attempts."""
        if not content_bytes:
            return ""

        # Extract charset from content-type header
        charset = None
        if "charset=" in content_type.lower():
            try:
                charset = content_type.lower().split("charset=")[1].split(";")[0].strip()
            except (IndexError, AttributeError):
                pass

        # Try different encodings in order of preference
        encodings_to_try = []

        if charset:
            encodings_to_try.append(charset)

        # Common web encodings
        encodings_to_try.extend(["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"])

        # Try each encoding
        for encoding in encodings_to_try:
            try:
                decoded = content_bytes.decode(encoding)
                # Validate that we got reasonable text
                if len(decoded.strip()) > 0:
                    return decoded
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: decode with errors='replace'
        logger.warning(f"Could not decode content properly, using fallback encoding")
        return content_bytes.decode("utf-8", errors="replace")

    async def _process_content_enhanced(self, result: CrawlResult):
        """Enhanced content processing with better validation."""
        try:
            # Validate HTML structure if enabled
            if self.config.validate_html_structure and result.content_type:
                if "html" in result.content_type.lower():
                    await self._validate_html_structure(result)

            # Use parent processing logic
            await super()._process_content(result)

        except Exception as e:
            logger.error(f"Enhanced content processing failed for {result.url}: {e}")
            # Don't fail the entire crawl due to processing errors
            result.security_warnings.append(f"Content processing warning: {e}")

    async def _validate_html_structure(self, result: CrawlResult):
        """Validate basic HTML structure for security and quality."""
        if not result.content:
            return

        content = result.content.lower()
        warnings = []

        # Check for basic HTML structure
        if "<html" not in content and "<body" not in content and "<head" not in content:
            warnings.append("Content appears to be invalid HTML")

        # Check for suspicious redirects in meta tags
        if "meta" in content and "refresh" in content:
            if "url=" in content:
                warnings.append("Meta redirect detected - potential redirection attack")

        # Check for excessive inline styles (potential XSS vector)
        style_count = content.count("style=")
        if style_count > 50:
            warnings.append(f"Excessive inline styles detected: {style_count}")

        # Add warnings to result
        result.security_warnings.extend(warnings)

    def _log_performance_metrics(self, result: CrawlResult):
        """Log and store performance metrics."""
        # This is a placeholder for more detailed performance logging
        # In a real implementation, you would collect detailed timing metrics
        logger.debug(f"Performance metrics logged for {result.url}")

    async def crawl_recursive_enhanced(self, start_url: str, **options) -> List[CrawlResult]:
        """Enhanced recursive crawling with better progress tracking and error recovery."""
        max_depth = options.get("max_depth", self.config.max_depth)
        max_pages = options.get("max_pages", self.config.max_total_pages)
        same_domain_only = options.get("same_domain_only", True)

        results = []
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        start_domain = urlparse(start_url).netloc

        # Progress tracking
        total_attempted = 0
        successful_crawls = 0
        failed_crawls = 0

        await self._ensure_session()

        try:
            while urls_to_visit and len(results) < max_pages:
                url, depth = urls_to_visit.pop(0)

                # Skip if already visited
                if url in self.visited_urls:
                    continue

                # Skip if too deep
                if depth > max_depth:
                    continue

                # Skip if different domain and same_domain_only is True
                if same_domain_only and urlparse(url).netloc != start_domain:
                    continue

                total_attempted += 1

                # Progress logging
                if total_attempted % 10 == 0:
                    logger.info(
                        f"Crawl progress: {total_attempted} attempted, "
                        f"{successful_crawls} successful, {failed_crawls} failed"
                    )

                # Crawl the URL with enhanced error handling
                result = await self.crawl_url(url, **options)
                results.append(result)

                # Mark as visited regardless of success/failure
                self.visited_urls.add(url)

                # Track statistics
                if result.success:
                    successful_crawls += 1
                else:
                    failed_crawls += 1

                # Extract links if successful and not at max depth
                if result.success and result.content and depth < max_depth:
                    try:
                        links = await self._extract_links(result.content, url)
                        for link in links:
                            if link not in self.visited_urls:
                                urls_to_visit.append((link, depth + 1))
                    except Exception as e:
                        logger.warning(f"Failed to extract links from {url}: {e}")

                # Respect rate limiting between pages
                if urls_to_visit:  # Don't wait after the last URL
                    await asyncio.sleep(0.1)  # Small delay between pages

        except Exception as e:
            logger.error(f"Enhanced recursive crawl failed: {e}")

        # Final progress report
        logger.info(
            f"Crawl completed: {total_attempted} attempted, "
            f"{successful_crawls} successful, {failed_crawls} failed"
        )

        return results


def create_enhanced_web_crawler(
    allowed_domains: Optional[List[str]] = None,
    max_retries: int = 3,
    enable_performance_monitoring: bool = True,
    **kwargs
) -> EnhancedSecureWebCrawler:
    """Create an enhanced web crawler with optimized settings.

    Args:
        allowed_domains: List of domains to allow (None = all)
        max_retries: Maximum retry attempts for failed requests
        enable_performance_monitoring: Enable detailed performance tracking
        **kwargs: Additional configuration options

    Returns:
        Configured EnhancedSecureWebCrawler instance
    """
    config = EnhancedSecurityConfig()

    if allowed_domains:
        config.domain_allowlist = set(allowed_domains)

    config.max_retries = max_retries
    config.enable_performance_metrics = enable_performance_monitoring

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return EnhancedSecureWebCrawler(config)