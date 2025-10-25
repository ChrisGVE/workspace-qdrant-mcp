"""
Enhanced web crawler that integrates advanced features with the existing secure crawler.

This module extends the existing SecureWebCrawler with enhanced content processing,
advanced retry logic, caching, and better integration with the document processing pipeline.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from .advanced_retry import AdvancedRetryHandler, RetryConfig
from .enhanced_content_extractor import EnhancedContentExtractor
from .web_cache import CacheConfig, WebCache
from .web_crawler import CrawlResult, SecureWebCrawler, SecurityConfig


class EnhancedCrawlResult(CrawlResult):
    """Enhanced crawl result with additional metadata and content analysis."""

    def __init__(self, url: str, success: bool = False):
        super().__init__(url, success)
        # Additional fields for enhanced crawler
        self.extracted_content: dict[str, Any] = {}
        self.content_quality_score: float = 0.0
        self.structured_data: dict[str, Any] = {}
        self.media_links: dict[str, Any] = {}
        self.text_links: list[dict[str, str]] = []
        self.processing_time: float = 0.0
        self.cache_hit: bool = False
        self.retry_attempts: int = 0
        self.deduplication_key: str | None = None


class EnhancedWebCrawler(SecureWebCrawler):
    """Enhanced web crawler with advanced content processing and reliability features."""

    def __init__(
        self,
        security_config: SecurityConfig | None = None,
        cache_config: CacheConfig | None = None,
        retry_config: RetryConfig | None = None
    ):
        super().__init__(security_config)

        # Enhanced components
        self.content_extractor = EnhancedContentExtractor()
        self.cache = WebCache(cache_config)
        self.retry_handler = AdvancedRetryHandler(retry_config)

        # Session management
        self.crawl_session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        self.session_stats = {
            'urls_crawled': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retry_attempts': 0,
            'content_extracted': 0,
            'duplicates_found': 0,
            'processing_time_total': 0.0
        }

        # User agent rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "WorkspaceQdrant-EnhancedCrawler/1.0 (+https://github.com/workspace-qdrant-mcp)"
        ]
        self.current_ua_index = 0

    async def close(self):
        """Close the crawler and save cache."""
        if self.cache.config.enable_disk_cache:
            await self.cache._save_to_disk()
        await super().close()

    def _get_next_user_agent(self) -> str:
        """Get next user agent for rotation."""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua

    async def crawl_url(self, url: str, **options) -> EnhancedCrawlResult:
        """Enhanced crawl with caching, retry logic, and content processing."""
        start_time = time.time()
        result = EnhancedCrawlResult(url)

        try:
            # Check cache first
            cached_entry = self.cache.get(url)
            if cached_entry:
                result.content = cached_entry.content
                result.headers = cached_entry.headers
                result.status_code = cached_entry.status_code
                result.content_type = cached_entry.headers.get('content-type', '')
                result.success = True
                result.cache_hit = True
                self.session_stats['cache_hits'] += 1

                logger.debug(f"Cache hit for {url}")
            else:
                # Cache miss - crawl with retry logic
                self.session_stats['cache_misses'] += 1
                await self._crawl_with_retry(url, result, **options)

            # Process content if successful
            if result.success and result.content:
                await self._process_enhanced_content(url, result)

            # Check for duplicates
            await self._check_content_duplication(result)

            result.processing_time = time.time() - start_time
            self.session_stats['processing_time_total'] += result.processing_time
            self.session_stats['urls_crawled'] += 1

            if result.extracted_content:
                self.session_stats['content_extracted'] += 1

            logger.info(f"Enhanced crawl completed: {url} (cache_hit: {result.cache_hit}, time: {result.processing_time:.2f}s)")

        except Exception as e:
            result.error = str(e)
            result.success = False
            result.processing_time = time.time() - start_time
            logger.error(f"Enhanced crawl failed: {url} - {e}")

        return result

    async def _crawl_with_retry(self, url: str, result: EnhancedCrawlResult, **options):
        """Crawl URL with advanced retry logic."""
        await self._ensure_session()

        # Update session with rotated user agent
        self._session._default_headers['User-Agent'] = self._get_next_user_agent()

        async def fetch_operation():
            # Validate URL
            if not await self._validate_url(url, result):
                raise Exception(result.error)

            # Check robots.txt
            if self.config.respect_robots_txt:
                if not await self._check_robots_txt(url):
                    raise Exception("Blocked by robots.txt")

            # Rate limiting
            await self._respect_rate_limit(url)

            # Get conditional headers for cache validation
            conditional_headers = self.cache.get_conditional_headers(url)

            # Fetch content
            await self._fetch_content_with_cache_validation(url, result, conditional_headers, **options)

            # Security scanning
            if result.content and self.config.enable_content_scanning:
                await self._scan_content_security(result)

            return result

        try:
            await self.retry_handler.execute_with_retry(fetch_operation, url)
            self.session_stats['retry_attempts'] += result.retry_attempts
        except Exception as e:
            result.error = str(e)
            result.success = False

    async def _fetch_content_with_cache_validation(
        self,
        url: str,
        result: EnhancedCrawlResult,
        conditional_headers: dict[str, str],
        **options
    ):
        """Fetch content with cache validation support."""
        max_content_size = options.get("max_content_size", self.config.max_content_size)
        allowed_content_types = options.get(
            "allowed_content_types", self.config.allowed_content_types
        )
        follow_redirects = options.get("follow_redirects", True)

        try:
            # Merge conditional headers with default headers
            headers = dict(self._session._default_headers)
            headers.update(conditional_headers)

            async with self._session.get(
                url,
                headers=headers,
                allow_redirects=follow_redirects,
                max_redirects=5 if follow_redirects else 0,
            ) as response:
                result.status_code = response.status
                result.headers = dict(response.headers)

                # Handle 304 Not Modified
                if response.status == 304:
                    cached_entry = self.cache.handle_304_response(url)
                    if cached_entry:
                        result.content = cached_entry.content
                        result.content_type = cached_entry.headers.get('content-type', '')
                        result.success = True
                        result.cache_hit = True
                        return

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
                    result.error = f"Content too large: {content_length} > {max_content_size}"
                    return

                # Read content with size limit
                content_bytes = b""
                async for chunk in response.content.iter_chunked(8192):
                    if len(content_bytes) + len(chunk) > max_content_size:
                        result.error = "Content size limit exceeded during download"
                        return
                    content_bytes += chunk

                # Decode content
                try:
                    result.content = content_bytes.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    result.content = content_bytes.decode("latin1", errors="replace")

                # Cache the response
                await self.cache.put(
                    url,
                    result.content,
                    result.headers,
                    result.status_code,
                    request_headers=headers
                )

                result.success = True

        except asyncio.TimeoutError:
            result.error = "Request timeout"
            result.retry_attempts += 1
        except Exception as e:
            result.error = f"Fetch failed: {e}"
            result.retry_attempts += 1

    async def _process_enhanced_content(self, url: str, result: EnhancedCrawlResult):
        """Process content with enhanced extraction."""
        try:
            # Extract enhanced content
            extracted = self.content_extractor.extract_content(result.content, url)

            result.extracted_content = extracted
            result.structured_data = extracted.get('structured_data', {})
            result.media_links = extracted.get('media_links', {})
            result.text_links = extracted.get('text_links', [])

            # Calculate content quality score
            quality_metrics = extracted.get('quality_metrics', {})
            result.content_quality_score = self._calculate_quality_score(quality_metrics)

            # Update main content with enhanced extraction
            main_content = extracted.get('main_content', '')
            if main_content and len(main_content) > len(result.content or '') * 0.1:
                # Only replace if enhanced content is substantial
                result.content = main_content

            # Store metadata
            metadata = extracted.get('metadata', {})
            result.metadata.update(metadata)

        except Exception as e:
            logger.warning(f"Enhanced content processing failed for {url}: {e}")

    def _calculate_quality_score(self, metrics: dict[str, Any]) -> float:
        """Calculate content quality score (0-1)."""
        try:
            score = 0.0

            # Text length score (normalized)
            text_length = metrics.get('text_length', 0)
            score += min(text_length / 5000, 0.3)  # Max 0.3 for length

            # Paragraph count score
            paragraph_count = metrics.get('paragraph_count', 0)
            score += min(paragraph_count / 10, 0.2)  # Max 0.2 for paragraphs

            # Link density penalty (lower is better)
            link_density = metrics.get('link_density', 1.0)
            score += max(0.0, 0.2 - link_density)  # Max 0.2 for low link density

            # Penalties for low-quality indicators
            ad_indicators = metrics.get('ad_indicators', 0)
            nav_indicators = metrics.get('navigation_indicators', 0)
            boilerplate_score = metrics.get('boilerplate_score', 1.0)

            penalties = (ad_indicators * 0.05) + (nav_indicators * 0.03) + (boilerplate_score * 0.2)
            score = max(0.0, score - penalties)

            # Readability bonus
            readability = metrics.get('readability_score', 0)
            if 15 <= readability <= 25:  # Good sentence length range
                score += 0.1

            return min(1.0, score)

        except Exception:
            return 0.0

    async def _check_content_duplication(self, result: EnhancedCrawlResult):
        """Check for content duplication."""
        if not result.content or not self.cache.config.enable_content_fingerprinting:
            return

        try:
            # Find duplicate content
            duplicates = self.cache.find_duplicate_content(result.content)
            if duplicates:
                result.deduplication_key = self.cache.fingerprinter.generate_fingerprint(
                    self.cache.fingerprinter.normalize_content_for_fingerprinting(result.content)
                )

                # Don't count current URL as duplicate
                other_duplicates = [url for url in duplicates if url != result.url]
                if other_duplicates:
                    result.metadata['duplicate_urls'] = other_duplicates
                    self.session_stats['duplicates_found'] += 1
                    logger.info(f"Found {len(other_duplicates)} duplicate(s) for {result.url}")

        except Exception as e:
            logger.warning(f"Duplication check failed for {result.url}: {e}")

    async def crawl_recursive(self, start_url: str, **options) -> list[EnhancedCrawlResult]:
        """Enhanced recursive crawling with session management."""
        logger.info(f"Starting enhanced recursive crawl (session: {self.crawl_session_id})")
        start_time = time.time()

        try:
            # Reset session stats
            self.session_stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.session_stats.items()}

            max_depth = options.get("max_depth", self.config.max_depth)
            max_pages = options.get("max_pages", self.config.max_total_pages)
            same_domain_only = options.get("same_domain_only", True)

            results = []
            urls_to_visit = [(start_url, 0)]  # (url, depth)
            start_domain = urlparse(start_url).netloc

            await self._ensure_session()

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

                # Extract links if successful and not at max depth
                if result.success and result.text_links and depth < max_depth:
                    for link_info in result.text_links:
                        link_url = link_info.get('url')
                        if link_url and link_url not in self.visited_urls:
                            # Basic URL validation
                            parsed = urlparse(link_url)
                            if parsed.scheme in {'http', 'https'}:
                                urls_to_visit.append((link_url, depth + 1))

                # Progress logging
                if len(results) % 10 == 0:
                    logger.info(f"Crawled {len(results)} pages, {len(urls_to_visit)} remaining")

            total_time = time.time() - start_time

            # Log session summary
            logger.info(f"Enhanced crawl session completed: {self.crawl_session_id}")
            logger.info(f"Results: {len(results)} pages, {total_time:.2f}s total")
            logger.info(f"Cache: {self.session_stats['cache_hits']} hits, {self.session_stats['cache_misses']} misses")
            logger.info(f"Retries: {self.session_stats['retry_attempts']} total attempts")
            logger.info(f"Duplicates: {self.session_stats['duplicates_found']} found")

            return results

        except Exception as e:
            logger.error(f"Enhanced recursive crawl failed: {e}")
            return []

    def get_session_stats(self) -> dict[str, Any]:
        """Get comprehensive session statistics."""
        cache_stats = self.cache.get_cache_stats()
        retry_stats = self.retry_handler.get_retry_stats()

        return {
            'session_id': self.crawl_session_id,
            'crawl_stats': self.session_stats.copy(),
            'cache_stats': cache_stats,
            'retry_stats': retry_stats,
            'visited_urls_count': len(self.visited_urls),
            'domain_request_counts': self.domain_request_counts.copy()
        }

    async def save_session_report(self, output_path: Path | None = None) -> Path:
        """Save comprehensive session report."""
        if output_path is None:
            output_path = Path(f"crawl_session_{self.crawl_session_id}_report.json")

        stats = self.get_session_stats()

        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            logger.info(f"Session report saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save session report: {e}")
            raise
