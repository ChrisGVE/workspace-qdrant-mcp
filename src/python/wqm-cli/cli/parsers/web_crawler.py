from common.observability import get_logger

logger = get_logger(__name__)
"""
Secure web content crawler with comprehensive security hardening.

This module provides secure web crawling capabilities with malware protection,
access controls, rate limiting, and content validation. It implements multiple
security layers to protect against malicious content and ensure safe operation.
"""

import asyncio
import hashlib
import logging
import mimetypes
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

try:
    import aiofiles
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

from .exceptions import ParsingError
from .html_parser import HtmlParser

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Configuration for web crawler security settings."""

    def __init__(self):
        # URL security
        self.allowed_schemes = {"http", "https"}
        self.blocked_schemes = {"file", "ftp", "data", "javascript"}
        self.max_url_length = 2048

        # Domain control
        self.domain_allowlist: Set[str] = set()
        self.domain_blocklist: Set[str] = {"localhost", "127.0.0.1", "0.0.0.0"}

        # Content security
        self.max_content_size = 50 * 1024 * 1024  # 50MB
        self.max_pages_per_domain = 1000
        self.allowed_content_types = {
            "text/html",
            "text/plain",
            "application/xhtml+xml",
            "text/xml",
        }

        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.max_concurrent_requests = 5
        self.max_retries = 3
        self.timeout_seconds = 30

        # Crawl limits
        self.max_depth = 3
        self.max_total_pages = 500
        self.respect_robots_txt = True

        # Security scanning
        self.enable_content_scanning = True
        self.quarantine_suspicious = True

        # User agent
        self.user_agent = "WorkspaceQdrant-SecureCrawler/1.0 (+security-enabled)"


class CrawlResult:
    """Result of web crawling operation."""

    def __init__(self, url: str, success: bool = False):
        self.url = url
        self.success = success
        self.content: Optional[str] = None
        self.content_type: Optional[str] = None
        self.status_code: Optional[int] = None
        self.headers: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.security_warnings: List[str] = []
        self.file_path: Optional[Path] = None
        self.timestamp = time.time()
        self.error: Optional[str] = None


class SecurityScanner:
    """Security scanner for web content."""

    def __init__(self):
        self.suspicious_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers
            re.compile(r"eval\s*\(", re.IGNORECASE),
            re.compile(r"document\.write", re.IGNORECASE),
            re.compile(r"window\.location", re.IGNORECASE),
        ]

        self.malicious_extensions = {
            ".exe",
            ".bat",
            ".cmd",
            ".scr",
            ".pif",
            ".com",
            ".dll",
            ".msi",
            ".jar",
            ".js",
            ".vbs",
            ".ps1",
        }

    async def scan_content(
        self, content: str, content_type: str
    ) -> Tuple[bool, List[str]]:
        """Scan content for security threats.

        Returns:
            Tuple of (is_safe, warnings)
        """
        warnings = []

        # Check content type
        if content_type and not self._is_safe_content_type(content_type):
            warnings.append(f"Potentially unsafe content type: {content_type}")

        # Scan for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                warnings.append(
                    f"Suspicious pattern detected: {pattern.pattern[:50]}..."
                )

        # Check content size
        if len(content) > 10 * 1024 * 1024:  # 10MB
            warnings.append(f"Large content size: {len(content)} bytes")

        # Additional heuristics
        if content.count("<script") > 10:
            warnings.append("High number of script tags detected")

        if re.search(r"\x00|\xff", content):
            warnings.append("Binary content detected in text")

        # Determine safety
        critical_warnings = [
            w
            for w in warnings
            if any(
                keyword in w.lower() for keyword in ["script", "javascript", "binary"]
            )
        ]

        is_safe = len(critical_warnings) == 0
        return is_safe, warnings

    def _is_safe_content_type(self, content_type: str) -> bool:
        """Check if content type is considered safe."""
        safe_types = {
            "text/html",
            "text/plain",
            "text/xml",
            "application/xhtml+xml",
            "application/xml",
        }

        base_type = content_type.split(";")[0].strip().lower()
        return base_type in safe_types

    async def scan_url(self, url: str) -> Tuple[bool, List[str]]:
        """Scan URL for security issues."""
        warnings = []
        parsed = urlparse(url)

        # Check for suspicious domains
        if any(
            suspicious in parsed.netloc.lower()
            for suspicious in ["bit.ly", "tinyurl", "short"]
        ):
            warnings.append("Shortened URL detected")

        # Check for suspicious paths
        if any(ext in parsed.path.lower() for ext in self.malicious_extensions):
            warnings.append(f"Suspicious file extension in URL: {parsed.path}")

        # Check for encoded characters
        if "%" in url and re.search(r"%[0-9a-fA-F]{2}", url):
            decoded_count = url.count("%")
            if decoded_count > 5:
                warnings.append(f"High number of encoded characters: {decoded_count}")

        is_safe = len(warnings) == 0
        return is_safe, warnings


class SecureWebCrawler:
    """Secure web crawler with comprehensive security hardening."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "Web crawling requires 'aiohttp' and 'aiofiles'. "
                "Install with: pip install aiohttp aiofiles"
            )

        if not BS4_AVAILABLE:
            raise RuntimeError(
                "Web crawling requires 'beautifulsoup4'. "
                "Install with: pip install beautifulsoup4"
            )

        self.config = config or SecurityConfig()
        self.scanner = SecurityScanner()
        self.html_parser = HtmlParser()

        # Tracking
        self.visited_urls: Set[str] = set()
        self.domain_request_counts: Dict[str, int] = {}
        self.last_request_times: Dict[str, float] = {}
        self.robots_cache: Dict[str, RobotFileParser] = {}

        # Session will be created when needed
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None:
            timeout = ClientTimeout(total=self.config.timeout_seconds)
            headers = {"User-Agent": self.config.user_agent}

            self._session = ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(
                    limit=self.config.max_concurrent_requests,
                    ssl=True,  # Always verify SSL
                ),
            )

    async def close(self):
        """Close the crawler and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def crawl_url(self, url: str, **options) -> CrawlResult:
        """Crawl a single URL with security checks.

        Args:
            url: URL to crawl
            **options: Crawling options
                - max_content_size: Override max content size
                - allowed_content_types: Override allowed content types
                - follow_redirects: Whether to follow redirects (default: True)

        Returns:
            CrawlResult with content and metadata
        """
        result = CrawlResult(url)

        try:
            # Validate URL
            if not await self._validate_url(url, result):
                return result

            # Check robots.txt
            if self.config.respect_robots_txt:
                if not await self._check_robots_txt(url):
                    result.error = "Blocked by robots.txt"
                    logger.warning(f"Access blocked by robots.txt: {url}")
                    return result

            # Rate limiting
            await self._respect_rate_limit(url)

            # Fetch content
            await self._fetch_content(url, result, **options)

            # Security scanning
            if result.content and self.config.enable_content_scanning:
                await self._scan_content_security(result)

            # Parse if successful
            if result.success and result.content:
                await self._process_content(result)

            logger.info(f"Successfully crawled: {url} ({result.status_code})")

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"Failed to crawl {url}: {e}")

        return result

    async def crawl_recursive(self, start_url: str, **options) -> List[CrawlResult]:
        """Crawl recursively from a starting URL.

        Args:
            start_url: Starting URL for recursive crawl
            **options: Crawling options
                - max_depth: Maximum crawl depth (default: config.max_depth)
                - max_pages: Maximum pages to crawl (default: config.max_total_pages)
                - same_domain_only: Only crawl same domain (default: True)

        Returns:
            List of CrawlResults
        """
        max_depth = options.get("max_depth", self.config.max_depth)
        max_pages = options.get("max_pages", self.config.max_total_pages)
        same_domain_only = options.get("same_domain_only", True)

        results = []
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        start_domain = urlparse(start_url).netloc

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

                # Crawl the URL
                result = await self.crawl_url(url, **options)
                results.append(result)

                # Mark as visited
                self.visited_urls.add(url)

                # Extract links if successful
                if result.success and result.content and depth < max_depth:
                    links = await self._extract_links(result.content, url)
                    for link in links:
                        if link not in self.visited_urls:
                            urls_to_visit.append((link, depth + 1))

        except Exception as e:
            logger.error(f"Recursive crawl failed: {e}")

        return results

    async def _validate_url(self, url: str, result: CrawlResult) -> bool:
        """Validate URL for security and format."""
        try:
            # Basic format check
            if len(url) > self.config.max_url_length:
                result.error = (
                    f"URL too long: {len(url)} > {self.config.max_url_length}"
                )
                return False

            parsed = urlparse(url)

            # Scheme validation
            if parsed.scheme not in self.config.allowed_schemes:
                result.error = f"Invalid scheme: {parsed.scheme}"
                return False

            if parsed.scheme in self.config.blocked_schemes:
                result.error = f"Blocked scheme: {parsed.scheme}"
                return False

            # Domain validation
            if not parsed.netloc:
                result.error = "Missing domain"
                return False

            domain = parsed.netloc.lower()

            # Check allowlist
            if (
                self.config.domain_allowlist
                and domain not in self.config.domain_allowlist
            ):
                result.error = f"Domain not in allowlist: {domain}"
                return False

            # Check blocklist
            if domain in self.config.domain_blocklist:
                result.error = f"Domain in blocklist: {domain}"
                return False

            # Check per-domain limits
            if (
                self.domain_request_counts.get(domain, 0)
                >= self.config.max_pages_per_domain
            ):
                result.error = f"Domain request limit exceeded: {domain}"
                return False

            # URL security scan
            is_safe, warnings = await self.scanner.scan_url(url)
            if not is_safe and warnings:
                result.security_warnings.extend(warnings)
                if self.config.quarantine_suspicious:
                    result.error = (
                        f"URL failed security scan: {'; '.join(warnings[:2])}"
                    )
                    return False

            return True

        except Exception as e:
            result.error = f"URL validation failed: {e}"
            return False

    async def _check_robots_txt(self, url: str) -> bool:
        """Check robots.txt for crawl permission."""
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = urljoin(base_url, "/robots.txt")

            # Check cache first
            if robots_url in self.robots_cache:
                rp = self.robots_cache[robots_url]
            else:
                # Fetch and parse robots.txt
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    # Use a simple synchronous approach for robots.txt
                    rp.read()
                    self.robots_cache[robots_url] = rp
                except Exception:
                    # If robots.txt can't be fetched, assume allowed
                    return True

            return rp.can_fetch(self.config.user_agent, url)

        except Exception:
            # If anything fails, err on the side of caution but allow
            return True

    async def _respect_rate_limit(self, url: str):
        """Implement rate limiting between requests."""
        domain = urlparse(url).netloc
        current_time = time.time()

        # Check if we need to wait
        last_request = self.last_request_times.get(domain, 0)
        time_since_last = current_time - last_request

        if time_since_last < self.config.request_delay:
            wait_time = self.config.request_delay - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
            await asyncio.sleep(wait_time)

        # Update counters
        self.last_request_times[domain] = time.time()
        self.domain_request_counts[domain] = (
            self.domain_request_counts.get(domain, 0) + 1
        )

    async def _fetch_content(self, url: str, result: CrawlResult, **options):
        """Fetch content from URL with security checks."""
        await self._ensure_session()

        max_content_size = options.get("max_content_size", self.config.max_content_size)
        allowed_content_types = options.get(
            "allowed_content_types", self.config.allowed_content_types
        )
        follow_redirects = options.get("follow_redirects", True)

        try:
            async with self._session.get(
                url,
                allow_redirects=follow_redirects,
                max_redirects=5 if follow_redirects else 0,
            ) as response:
                result.status_code = response.status
                result.headers = dict(response.headers)

                # Check response code
                if response.status >= 400:
                    result.error = f"HTTP {response.status}: {response.reason}"
                    return

                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                result.content_type = content_type

                base_content_type = content_type.split(";")[0].strip()
                if base_content_type and base_content_type not in allowed_content_types:
                    result.error = f"Unsupported content type: {base_content_type}"
                    return

                # Check content length
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_content_size:
                    result.error = (
                        f"Content too large: {content_length} > {max_content_size}"
                    )
                    return

                # Read content with size limit
                content_bytes = b""
                async for chunk in response.content.iter_chunked(8192):
                    if len(content_bytes) + len(chunk) > max_content_size:
                        result.error = f"Content size limit exceeded during download"
                        return
                    content_bytes += chunk

                # Decode content
                try:
                    result.content = content_bytes.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    result.content = content_bytes.decode("latin1", errors="replace")

                result.success = True

        except asyncio.TimeoutError:
            result.error = "Request timeout"
        except Exception as e:
            result.error = f"Fetch failed: {e}"

    async def _scan_content_security(self, result: CrawlResult):
        """Scan content for security threats."""
        if not result.content:
            return

        is_safe, warnings = await self.scanner.scan_content(
            result.content, result.content_type or ""
        )

        result.security_warnings.extend(warnings)

        if not is_safe and self.config.quarantine_suspicious:
            # Move content to quarantine
            result.error = f"Content failed security scan: {'; '.join(warnings[:2])}"
            result.success = False

            # Save to quarantine for analysis
            await self._quarantine_content(result)

    async def _quarantine_content(self, result: CrawlResult):
        """Save suspicious content to quarantine directory."""
        try:
            quarantine_dir = Path(tempfile.gettempdir()) / "qdrant_quarantine"
            quarantine_dir.mkdir(exist_ok=True)

            # Create unique filename
            url_hash = hashlib.sha256(result.url.encode()).hexdigest()[:12]
            timestamp = int(result.timestamp)
            filename = f"quarantine_{timestamp}_{url_hash}.html"

            quarantine_path = quarantine_dir / filename

            # Save content and metadata
            metadata = {
                "url": result.url,
                "timestamp": result.timestamp,
                "security_warnings": result.security_warnings,
                "content_type": result.content_type,
                "status_code": result.status_code,
            }

            async with aiofiles.open(quarantine_path, "w", encoding="utf-8") as f:
                await f.write(f"<!-- QUARANTINED CONTENT\n{metadata}\n-->\n")
                await f.write(result.content or "")

            logger.warning(f"Content quarantined: {quarantine_path}")
            result.metadata["quarantine_path"] = str(quarantine_path)

        except Exception as e:
            logger.error(f"Failed to quarantine content: {e}")

    async def _process_content(self, result: CrawlResult):
        """Process crawled content using HTML parser."""
        try:
            # Create temporary file for HTML parser
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(result.content or "")
                temp_path = Path(f.name)

            try:
                # Use existing HTML parser
                parsed_doc = await self.html_parser.parse(temp_path)

                # Add parsed content to result
                result.metadata.update(
                    {
                        "parsed_content": parsed_doc.content,
                        "html_metadata": parsed_doc.additional_metadata,
                        "parsing_info": parsed_doc.parsing_info,
                    }
                )

                result.file_path = temp_path

            finally:
                # Clean up temp file (or keep it if quarantined)
                if "quarantine_path" not in result.metadata:
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Failed to process content for {result.url}: {e}")
            result.security_warnings.append(f"Content processing failed: {e}")

    async def _extract_links(self, content: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        links = []

        try:
            soup = BeautifulSoup(content, "lxml")

            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"].strip()
                if not href or href.startswith("#"):
                    continue

                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)

                # Basic validation
                parsed = urlparse(absolute_url)
                if parsed.scheme in self.config.allowed_schemes:
                    links.append(absolute_url)

        except Exception as e:
            logger.error(f"Failed to extract links: {e}")

        return links[:50]  # Limit number of links to prevent explosion


def create_security_config(
    domain_allowlist: Optional[List[str]] = None,
    max_pages: int = 100,
    max_depth: int = 2,
    request_delay: float = 1.0,
) -> SecurityConfig:
    """Create a security configuration with common settings.

    Args:
        domain_allowlist: List of allowed domains (None means all allowed)
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        request_delay: Delay between requests in seconds

    Returns:
        Configured SecurityConfig instance
    """
    config = SecurityConfig()

    if domain_allowlist:
        config.domain_allowlist = set(domain_allowlist)

    config.max_total_pages = max_pages
    config.max_depth = max_depth
    config.request_delay = request_delay

    return config
