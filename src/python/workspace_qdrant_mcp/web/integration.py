"""
Web crawling system integration module.

This module integrates all web crawling components:
- WebCrawler for respectful HTTP requests
- ContentExtractor for content parsing and quality assessment
- LinkDiscovery and RecursiveCrawler for intelligent navigation
- ContentCache for duplicate detection and storage optimization

Provides a high-level interface for complete web crawling workflows.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from .cache import CacheConfig, ContentCache
from .crawler import CrawlConfig, WebCrawler
from .extractor import ContentExtractor, ExtractionStrategy
from .links import CrawlSession, LinkDiscovery, RecursiveCrawler

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result from a web crawling operation."""
    url: str
    title: str
    content: str
    extracted_links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    from_cache: bool = False
    content_hash: Optional[str] = None
    duplicate_of: Optional[str] = None


@dataclass
class CrawlSummary:
    """Summary of a crawling session."""
    total_urls_discovered: int = 0
    total_urls_crawled: int = 0
    successful_crawls: int = 0
    failed_crawls: int = 0
    cached_hits: int = 0
    duplicates_found: int = 0
    total_content_size: int = 0
    crawl_duration: float = 0.0
    unique_domains: Set[str] = field(default_factory=set)
    error_summary: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of crawling."""
        if self.total_urls_crawled == 0:
            return 0.0
        return self.successful_crawls / self.total_urls_crawled

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_urls_crawled == 0:
            return 0.0
        return self.cached_hits / self.total_urls_crawled


@dataclass
class IntegratedCrawlConfig:
    """Configuration for integrated web crawling system."""
    # Crawler configuration
    user_agent: str = "IntegratedWebCrawler/1.0"
    request_timeout: float = 30.0
    max_retries: int = 3
    respect_robots: bool = True
    rate_limit_delay: float = 1.0

    # Content extraction configuration
    extraction_strategies: List[ExtractionStrategy] = field(
        default_factory=lambda: [
            ExtractionStrategy.BEAUTIFUL_SOUP,
            ExtractionStrategy.LXML,
            ExtractionStrategy.REGEX,
            ExtractionStrategy.SIMPLE_TEXT
        ]
    )
    min_content_length: int = 100
    quality_threshold: float = 0.5

    # Link discovery configuration
    max_depth: int = 3
    max_pages_per_domain: int = 100
    follow_external_links: bool = False
    exclude_patterns: List[str] = field(default_factory=list)

    # Caching configuration
    cache_max_size: int = 100 * 1024 * 1024  # 100MB
    cache_max_entries: int = 10000
    enable_persistence: bool = True
    similarity_threshold: float = 0.85

    # Performance configuration
    max_concurrent_requests: int = 5
    request_semaphore_size: int = 10


class IntegratedWebCrawler:
    """
    Integrated web crawling system combining all components.

    Provides a unified interface for:
    - Respectful web crawling with rate limiting
    - Content extraction and quality assessment
    - Intelligent link discovery and navigation
    - Content caching with duplicate detection
    """

    def __init__(self, config: Optional[IntegratedCrawlConfig] = None):
        self.config = config or IntegratedCrawlConfig()

        # Initialize components
        self.crawler = WebCrawler(CrawlConfig(
            user_agent=self.config.user_agent,
            request_timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            respect_robots=self.config.respect_robots,
            rate_limit_delay=self.config.rate_limit_delay
        ))

        self.extractor = ContentExtractor(
            strategies=self.config.extraction_strategies,
            min_content_length=self.config.min_content_length
        )

        self.link_discovery = LinkDiscovery()

        self.recursive_crawler = RecursiveCrawler(
            crawler=self.crawler,
            link_discovery=self.link_discovery
        )

        self.cache = ContentCache(CacheConfig(
            max_size=self.config.cache_max_size,
            max_entries=self.config.cache_max_entries,
            enable_persistence=self.config.enable_persistence,
            similarity_threshold=self.config.similarity_threshold
        ))

        # Concurrency control
        self.request_semaphore = asyncio.Semaphore(self.config.request_semaphore_size)

        logger.info("IntegratedWebCrawler initialized with comprehensive configuration")

    async def crawl_url(self, url: str, use_cache: bool = True) -> CrawlResult:
        """
        Crawl a single URL with full integration.

        Args:
            url: URL to crawl
            use_cache: Whether to use caching

        Returns:
            CrawlResult with all extracted information
        """
        try:
            # Check cache first if enabled
            if use_cache:
                cached_entry = await self.cache.get(url)
                if cached_entry:
                    logger.debug(f"Cache hit for URL: {url}")

                    # Handle duplicate references
                    if "duplicate_of" in cached_entry.metadata:
                        original_url = cached_entry.metadata["duplicate_of"]
                        original_entry = await self.cache.get(original_url)
                        if original_entry:
                            return CrawlResult(
                                url=url,
                                title=original_entry.metadata.get("title", ""),
                                content=original_entry.content,
                                extracted_links=original_entry.metadata.get("links", []),
                                metadata=original_entry.metadata,
                                success=True,
                                from_cache=True,
                                content_hash=original_entry.content_hash,
                                duplicate_of=original_url
                            )

                    # Return cached content
                    return CrawlResult(
                        url=url,
                        title=cached_entry.metadata.get("title", ""),
                        content=cached_entry.content,
                        extracted_links=cached_entry.metadata.get("links", []),
                        metadata=cached_entry.metadata,
                        success=True,
                        from_cache=True,
                        content_hash=cached_entry.content_hash
                    )

            # Crawl the URL
            async with self.request_semaphore:
                crawl_response = await self.crawler.crawl_url(url)

            if not crawl_response.success:
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    success=False,
                    error=crawl_response.error
                )

            # Extract content
            extracted_content = self.extractor.extract(crawl_response.content, url)

            # Check content quality
            if extracted_content.quality.overall_score < self.config.quality_threshold:
                logger.warning(f"Low quality content for URL: {url} (score: {extracted_content.quality.overall_score:.2f})")

            # Extract links
            discovered_links = self.link_discovery.extract_links(
                crawl_response.content,
                url
            )
            link_urls = [link.url for link in discovered_links]

            # Prepare metadata
            metadata = {
                "title": extracted_content.title,
                "links": link_urls,
                "content_type": crawl_response.content_type,
                "quality_score": extracted_content.quality.overall_score,
                "word_count": len(extracted_content.content.split()),
                "link_count": len(link_urls),
                "crawl_timestamp": crawl_response.timestamp
            }

            # Cache the content if enabled
            if use_cache:
                await self.cache.put(
                    url,
                    extracted_content.content,
                    crawl_response.content_type,
                    metadata
                )

            return CrawlResult(
                url=url,
                title=extracted_content.title,
                content=extracted_content.content,
                extracted_links=link_urls,
                metadata=metadata,
                success=True,
                content_hash=await self._get_content_hash(extracted_content.content)
            )

        except Exception as e:
            logger.error(f"Error crawling URL {url}: {e}")
            return CrawlResult(
                url=url,
                title="",
                content="",
                success=False,
                error=str(e)
            )

    async def crawl_recursively(
        self,
        start_url: str,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        domain_filter: Optional[str] = None
    ) -> Tuple[List[CrawlResult], CrawlSummary]:
        """
        Perform recursive crawling starting from a URL.

        Args:
            start_url: Starting URL for crawling
            max_depth: Maximum crawling depth (None for config default)
            max_pages: Maximum pages to crawl (None for unlimited)
            domain_filter: Domain to restrict crawling to

        Returns:
            Tuple of (crawl_results, crawl_summary)
        """
        import time
        start_time = time.time()

        max_depth = max_depth or self.config.max_depth
        results: List[CrawlResult] = []
        summary = CrawlSummary()

        try:
            # Start recursive crawl session
            session = self.recursive_crawler.start_crawl_session(start_url, max_depth)
            crawled_count = 0

            while session.has_pending_urls():
                if max_pages and crawled_count >= max_pages:
                    logger.info(f"Reached maximum page limit: {max_pages}")
                    break

                # Get next URL to crawl
                next_url = session.get_next_url()
                if not next_url:
                    break

                # Apply domain filter if specified
                if domain_filter:
                    if urlparse(next_url).netloc != domain_filter:
                        session.mark_crawled(next_url, False)
                        continue

                summary.total_urls_discovered += 1

                # Crawl the URL
                result = await self.crawl_url(next_url)
                results.append(result)
                crawled_count += 1
                summary.total_urls_crawled += 1

                if result.success:
                    summary.successful_crawls += 1
                    summary.total_content_size += len(result.content)

                    # Track unique domains
                    domain = urlparse(next_url).netloc
                    summary.unique_domains.add(domain)

                    if result.from_cache:
                        summary.cached_hits += 1

                    if result.duplicate_of:
                        summary.duplicates_found += 1

                    # Mark as successfully crawled and add discovered links
                    session.mark_crawled(next_url, True)

                    # Add discovered links to session
                    for link_url in result.extracted_links:
                        session.add_discovered_url(link_url)

                else:
                    summary.failed_crawls += 1
                    session.mark_crawled(next_url, False)

                    # Track error types
                    error_type = result.error or "Unknown error"
                    summary.error_summary[error_type] = summary.error_summary.get(error_type, 0) + 1

                # Respect rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)

            summary.crawl_duration = time.time() - start_time

            logger.info(f"Recursive crawl completed: {summary.successful_crawls}/{summary.total_urls_crawled} successful, "
                       f"{summary.cache_hit_rate:.1%} cache hit rate, {summary.crawl_duration:.1f}s duration")

            return results, summary

        except Exception as e:
            logger.error(f"Error in recursive crawling: {e}")
            summary.crawl_duration = time.time() - start_time
            return results, summary

    async def batch_crawl(self, urls: List[str], max_concurrent: Optional[int] = None) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent requests (None for config default)

        Returns:
            List of CrawlResults
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_url(url)

        logger.info(f"Starting batch crawl of {len(urls)} URLs with max_concurrent={max_concurrent}")

        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed CrawlResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(CrawlResult(
                    url=urls[i],
                    title="",
                    content="",
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        successful = sum(1 for r in final_results if r.success)
        logger.info(f"Batch crawl completed: {successful}/{len(urls)} successful")

        return final_results

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.cache.get_stats()
        duplicates = self.cache.get_duplicates()

        return {
            "total_entries": stats.total_entries,
            "total_size_mb": stats.size_mb,
            "hit_rate": stats.hit_rate,
            "duplicate_count": stats.duplicate_count,
            "eviction_count": stats.eviction_count,
            "cleanup_count": stats.cleanup_count,
            "duplicate_groups": len(duplicates),
            "average_entry_size": stats.average_size
        }

    async def clear_cache(self) -> int:
        """Clear all cached content."""
        return await self.cache.clear()

    async def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        return await self.cache.cleanup()

    async def _get_content_hash(self, content: str) -> str:
        """Get hash of content for result tracking."""
        from .cache import ContentHasher
        return ContentHasher.hash_content(content)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.cache.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cache.__aexit__(exc_type, exc_val, exc_tb)


# Convenience function for simple crawling
async def crawl_url_simple(url: str, **config_kwargs) -> CrawlResult:
    """
    Simple function to crawl a single URL with minimal configuration.

    Args:
        url: URL to crawl
        **config_kwargs: Configuration parameters to override

    Returns:
        CrawlResult with extracted information
    """
    config = IntegratedCrawlConfig(**config_kwargs)

    async with IntegratedWebCrawler(config) as crawler:
        return await crawler.crawl_url(url)


# Convenience function for recursive crawling
async def crawl_site_recursive(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    **config_kwargs
) -> Tuple[List[CrawlResult], CrawlSummary]:
    """
    Simple function to recursively crawl a site.

    Args:
        start_url: Starting URL
        max_depth: Maximum depth to crawl
        max_pages: Maximum pages to crawl
        **config_kwargs: Configuration parameters to override

    Returns:
        Tuple of (results, summary)
    """
    config = IntegratedCrawlConfig(**config_kwargs)

    async with IntegratedWebCrawler(config) as crawler:
        return await crawler.crawl_recursively(
            start_url,
            max_depth=max_depth,
            max_pages=max_pages
        )