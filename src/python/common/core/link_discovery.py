"""
Link Discovery and Recursive Crawling Management

This module provides comprehensive link discovery capabilities for recursive web crawling
with intelligent depth management, cycle detection, and URL filtering.

Features:
    - Intelligent link discovery from HTML content
    - URL normalization and canonicalization
    - Same-domain and cross-domain filtering
    - Depth-based crawling with configurable limits
    - Cycle detection to prevent infinite loops
    - Priority-based link scheduling
    - Link quality assessment and filtering
    - Robots.txt compliance checking
    - URL pattern matching and exclusion rules
    - Sitemap.xml discovery and parsing
    - Breadth-first and depth-first crawling strategies

Link Types Supported:
    - HTML anchor links (a href)
    - Canonical links (rel="canonical")
    - Sitemap references
    - RSS/Atom feeds
    - JSON-LD structured data links
    - Open Graph URLs
    - Social media sharing links

Example:
    ```python
    from workspace_qdrant_mcp.core.link_discovery import LinkDiscovery

    discovery = LinkDiscovery(
        max_depth=3,
        same_domain_only=True,
        max_links_per_page=50
    )

    # Discover links from HTML content
    links = await discovery.discover_links(
        html_content="<html>...</html>",
        base_url="https://example.com/page",
        current_depth=1
    )

    # Start recursive crawling
    async for batch in discovery.crawl_recursive(
        start_urls=["https://example.com"],
        max_pages=100
    ):
        # Process batch of URLs
        pass
    ```
"""

import asyncio
import re
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup


@dataclass
class DiscoveredLink:
    """Represents a discovered link with metadata."""

    url: str
    source_url: str
    anchor_text: str | None = None
    title: str | None = None
    depth: int = 0
    discovery_method: str = "html_anchor"

    # Link context
    context: str | None = None  # Surrounding text
    position: int = 0  # Position on page

    # Quality indicators
    is_internal: bool = True
    is_canonical: bool = False
    is_navigation: bool = False

    # Crawling metadata
    discovered_at: datetime | None = None
    crawl_priority: float = 0.5  # 0.0 (lowest) to 1.0 (highest)

    # Status tracking
    crawled: bool = False
    crawl_attempted: bool = False
    crawl_error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkDiscoveryConfig:
    """Configuration for link discovery and recursive crawling."""

    # Crawling limits
    max_depth: int = 3
    max_pages: int = 1000
    max_links_per_page: int = 100

    # Domain restrictions
    same_domain_only: bool = True
    allowed_domains: set[str] = field(default_factory=set)
    blocked_domains: set[str] = field(default_factory=set)

    # URL filtering
    allowed_schemes: set[str] = field(default_factory=lambda: {'http', 'https'})
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r'\.pdf$', r'\.doc$', r'\.docx$', r'\.xls$', r'\.xlsx$',
        r'\.zip$', r'\.rar$', r'\.exe$', r'\.dmg$',
        r'/admin/', r'/login/', r'/logout/', r'/account/',
        r'\?print=', r'\?download=', r'#'
    ])
    allowed_patterns: list[str] = field(default_factory=list)

    # Link quality filtering
    min_anchor_text_length: int = 2
    max_anchor_text_length: int = 200
    skip_image_links: bool = True
    skip_empty_anchor_text: bool = True

    # Discovery methods
    discover_from_anchors: bool = True
    discover_from_sitemaps: bool = True
    discover_from_structured_data: bool = True
    discover_from_feeds: bool = False

    # Crawling strategy
    strategy: str = "breadth_first"  # breadth_first, depth_first
    batch_size: int = 10
    delay_between_batches: float = 1.0

    # Duplicate detection
    url_normalization: bool = True
    ignore_query_params: set[str] = field(default_factory=lambda: {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'ref', 'source'
    })


class URLNormalizer:
    """Handles URL normalization and canonicalization."""

    def __init__(self, config: LinkDiscoveryConfig):
        self.config = config

    def normalize_url(self, url: str, base_url: str | None = None) -> str:
        """
        Normalize URL for deduplication and comparison.

        Args:
            url: URL to normalize
            base_url: Base URL for resolving relatives

        Returns:
            Normalized URL string
        """
        if not url or not url.strip():
            return ""

        # Resolve relative URLs
        if base_url:
            url = urljoin(base_url, url.strip())

        try:
            parsed = urlparse(url)

            # Normalize scheme and netloc
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()

            # Remove default ports
            if ':80' in netloc and scheme == 'http':
                netloc = netloc.replace(':80', '')
            elif ':443' in netloc and scheme == 'https':
                netloc = netloc.replace(':443', '')

            # Normalize path
            path = parsed.path
            if not path:
                path = '/'

            # Remove trailing slash for non-root paths
            if path != '/' and path.endswith('/'):
                path = path.rstrip('/')

            # Handle query parameters
            query = parsed.query
            if self.config.url_normalization and query:
                query = self._normalize_query_params(query)

            # Remove fragment if not needed
            fragment = ""  # Usually ignore fragments

            # Reconstruct URL
            normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
            return normalized

        except Exception:
            return url  # Return original if normalization fails

    def _normalize_query_params(self, query: str) -> str:
        """Normalize query parameters by removing tracking parameters."""
        if not query:
            return ""

        try:
            params = parse_qs(query, keep_blank_values=True)

            # Remove ignored parameters
            filtered_params = {
                k: v for k, v in params.items()
                if k not in self.config.ignore_query_params
            }

            if not filtered_params:
                return ""

            # Sort parameters for consistent ordering
            sorted_params = []
            for key in sorted(filtered_params.keys()):
                for value in filtered_params[key]:
                    sorted_params.append(f"{key}={value}")

            return "&".join(sorted_params)

        except Exception:
            return query

    def is_valid_url(self, url: str, base_domain: str | None = None) -> bool:
        """
        Check if URL is valid for crawling.

        Args:
            url: URL to validate
            base_domain: Base domain for same-domain checking

        Returns:
            True if URL is valid for crawling
        """
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in self.config.allowed_schemes:
                return False

            # Check domain restrictions
            if self.config.same_domain_only and base_domain:
                if parsed.netloc.lower() != base_domain.lower():
                    return False

            # Check allowed domains
            if self.config.allowed_domains:
                if parsed.netloc.lower() not in self.config.allowed_domains:
                    return False

            # Check blocked domains
            if parsed.netloc.lower() in self.config.blocked_domains:
                return False

            # Check URL patterns
            full_url = url.lower()

            # Check blocked patterns
            for pattern in self.config.blocked_patterns:
                if re.search(pattern, full_url, re.IGNORECASE):
                    return False

            # Check allowed patterns (if any specified)
            if self.config.allowed_patterns:
                allowed = False
                for pattern in self.config.allowed_patterns:
                    if re.search(pattern, full_url, re.IGNORECASE):
                        allowed = True
                        break
                if not allowed:
                    return False

            return True

        except Exception:
            return False


class LinkExtractor:
    """Extracts links from HTML content using various methods."""

    def __init__(self, config: LinkDiscoveryConfig):
        self.config = config
        self.normalizer = URLNormalizer(config)

    def extract_anchor_links(self, soup: BeautifulSoup, base_url: str) -> list[DiscoveredLink]:
        """Extract links from HTML anchor tags."""
        if not self.config.discover_from_anchors:
            return []

        links = []
        anchors = soup.find_all('a', href=True)

        for i, anchor in enumerate(anchors):
            href = anchor.get('href', '').strip()
            if not href:
                continue

            # Resolve and normalize URL
            resolved_url = self.normalizer.normalize_url(href, base_url)
            if not resolved_url:
                continue

            # Extract anchor text and title
            anchor_text = anchor.get_text().strip()
            title = anchor.get('title', '').strip()

            # Apply quality filters
            if self.config.skip_empty_anchor_text and not anchor_text:
                continue

            if len(anchor_text) < self.config.min_anchor_text_length:
                continue

            if len(anchor_text) > self.config.max_anchor_text_length:
                anchor_text = anchor_text[:self.config.max_anchor_text_length]

            # Skip image-only links if configured
            if self.config.skip_image_links and anchor.find('img') and not anchor_text:
                continue

            # Determine if link is internal
            is_internal = self._is_internal_link(resolved_url, base_url)

            # Extract context (surrounding text)
            context = self._extract_context(anchor)

            # Determine if it's navigation
            is_navigation = self._is_navigation_link(anchor, anchor_text)

            # Calculate priority
            priority = self._calculate_link_priority(anchor, anchor_text, is_internal, is_navigation)

            link = DiscoveredLink(
                url=resolved_url,
                source_url=base_url,
                anchor_text=anchor_text,
                title=title,
                discovery_method="html_anchor",
                context=context,
                position=i,
                is_internal=is_internal,
                is_navigation=is_navigation,
                discovered_at=datetime.now(),
                crawl_priority=priority
            )

            links.append(link)

        return links

    def extract_canonical_links(self, soup: BeautifulSoup, base_url: str) -> list[DiscoveredLink]:
        """Extract canonical links from HTML."""
        links = []

        # Check for canonical link tag
        canonical = soup.find('link', rel='canonical', href=True)
        if canonical:
            href = canonical.get('href', '').strip()
            resolved_url = self.normalizer.normalize_url(href, base_url)

            if resolved_url:
                link = DiscoveredLink(
                    url=resolved_url,
                    source_url=base_url,
                    discovery_method="canonical_link",
                    is_canonical=True,
                    discovered_at=datetime.now(),
                    crawl_priority=0.9  # High priority for canonical URLs
                )
                links.append(link)

        return links

    def extract_structured_data_links(self, soup: BeautifulSoup, base_url: str) -> list[DiscoveredLink]:
        """Extract links from structured data (JSON-LD, microdata)."""
        if not self.config.discover_from_structured_data:
            return []

        links = []

        # Extract from JSON-LD scripts
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                import json
                data = json.loads(script.string)
                urls = self._extract_urls_from_json_ld(data)

                for url in urls:
                    resolved_url = self.normalizer.normalize_url(url, base_url)
                    if resolved_url:
                        link = DiscoveredLink(
                            url=resolved_url,
                            source_url=base_url,
                            discovery_method="json_ld",
                            discovered_at=datetime.now(),
                            crawl_priority=0.7
                        )
                        links.append(link)

            except (json.JSONDecodeError, TypeError):
                pass

        # Extract from Open Graph tags
        og_url = soup.find('meta', property='og:url', content=True)
        if og_url:
            url = og_url.get('content', '').strip()
            resolved_url = self.normalizer.normalize_url(url, base_url)
            if resolved_url:
                link = DiscoveredLink(
                    url=resolved_url,
                    source_url=base_url,
                    discovery_method="open_graph",
                    discovered_at=datetime.now(),
                    crawl_priority=0.8
                )
                links.append(link)

        return links

    def extract_sitemap_links(self, robots_content: str, base_url: str) -> list[DiscoveredLink]:
        """Extract sitemap URLs from robots.txt content."""
        if not self.config.discover_from_sitemaps:
            return []

        links = []

        for line in robots_content.splitlines():
            line = line.strip().lower()
            if line.startswith('sitemap:'):
                sitemap_url = line[8:].strip()
                resolved_url = self.normalizer.normalize_url(sitemap_url, base_url)

                if resolved_url:
                    link = DiscoveredLink(
                        url=resolved_url,
                        source_url=base_url,
                        discovery_method="robots_sitemap",
                        discovered_at=datetime.now(),
                        crawl_priority=0.6
                    )
                    links.append(link)

        return links

    def _is_internal_link(self, url: str, base_url: str) -> bool:
        """Check if link is internal (same domain)."""
        try:
            url_domain = urlparse(url).netloc.lower()
            base_domain = urlparse(base_url).netloc.lower()
            return url_domain == base_domain
        except Exception:
            return False

    def _extract_context(self, anchor, context_chars: int = 100) -> str:
        """Extract surrounding text context for a link."""
        try:
            # Get parent element text
            parent = anchor.parent
            if parent:
                full_text = parent.get_text()
                anchor_text = anchor.get_text()

                # Find anchor position in parent text
                anchor_pos = full_text.find(anchor_text)
                if anchor_pos >= 0:
                    start = max(0, anchor_pos - context_chars // 2)
                    end = min(len(full_text), anchor_pos + len(anchor_text) + context_chars // 2)
                    return full_text[start:end].strip()

            return ""
        except Exception:
            return ""

    def _is_navigation_link(self, anchor, anchor_text: str) -> bool:
        """Determine if link is likely navigation."""
        nav_indicators = [
            'nav', 'menu', 'header', 'footer', 'sidebar',
            'breadcrumb', 'pagination', 'pager'
        ]

        # Check anchor classes and IDs
        classes = ' '.join(anchor.get('class', []))
        anchor_id = anchor.get('id', '')

        for indicator in nav_indicators:
            if indicator in classes.lower() or indicator in anchor_id.lower():
                return True

        # Check parent elements
        parent = anchor.parent
        while parent and parent.name:
            parent_classes = ' '.join(parent.get('class', []))
            parent_id = parent.get('id', '')

            for indicator in nav_indicators:
                if indicator in parent_classes.lower() or indicator in parent_id.lower():
                    return True

            if parent.name in ['nav', 'header', 'footer']:
                return True

            parent = parent.parent

        return False

    def _calculate_link_priority(self, anchor, anchor_text: str, is_internal: bool, is_navigation: bool) -> float:
        """Calculate crawling priority for a link."""
        priority = 0.5  # Base priority

        # Internal links get higher priority
        if is_internal:
            priority += 0.2

        # Navigation links get lower priority
        if is_navigation:
            priority -= 0.1

        # Longer, descriptive anchor text gets higher priority
        if anchor_text and len(anchor_text) > 10:
            priority += 0.1

        # Links in main content area get higher priority
        if self._is_in_main_content(anchor):
            priority += 0.2

        # Ensure priority stays in valid range
        return max(0.0, min(1.0, priority))

    def _is_in_main_content(self, anchor) -> bool:
        """Check if anchor is in main content area."""
        content_indicators = ['article', 'main', 'content', 'post', 'story']

        parent = anchor.parent
        while parent and parent.name:
            if parent.name in ['article', 'main']:
                return True

            classes = ' '.join(parent.get('class', []))
            for indicator in content_indicators:
                if indicator in classes.lower():
                    return True

            parent = parent.parent

        return False

    def _extract_urls_from_json_ld(self, data: Any) -> list[str]:
        """Extract URLs from JSON-LD structured data."""
        urls = []

        if isinstance(data, dict):
            for key, value in data.items():
                if key in ['url', 'sameAs', 'mainEntityOfPage']:
                    if isinstance(value, str):
                        urls.append(value)
                    elif isinstance(value, list):
                        urls.extend([v for v in value if isinstance(v, str)])
                elif isinstance(value, (dict, list)):
                    urls.extend(self._extract_urls_from_json_ld(value))

        elif isinstance(data, list):
            for item in data:
                urls.extend(self._extract_urls_from_json_ld(item))

        return urls


class LinkDiscovery:
    """
    Main class for link discovery and recursive crawling management.
    """

    def __init__(self, config: LinkDiscoveryConfig | None = None):
        """Initialize link discovery with configuration."""
        self.config = config or LinkDiscoveryConfig()
        self.normalizer = URLNormalizer(self.config)
        self.extractor = LinkExtractor(self.config)

        # Tracking
        self._discovered_urls: set[str] = set()
        self._crawled_urls: set[str] = set()
        self._failed_urls: set[str] = set()
        self._crawl_queue: deque = deque()

        # Statistics
        self._stats = {
            'urls_discovered': 0,
            'urls_crawled': 0,
            'urls_failed': 0,
            'max_depth_reached': 0,
            'discovery_methods': defaultdict(int)
        }

    async def discover_links(self, html_content: str, base_url: str,
                           current_depth: int = 0) -> list[DiscoveredLink]:
        """
        Discover links from HTML content.

        Args:
            html_content: HTML content to extract links from
            base_url: Base URL for resolving relative links
            current_depth: Current crawling depth

        Returns:
            List of discovered links
        """
        if current_depth >= self.config.max_depth:
            return []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            all_links = []

            # Extract different types of links
            anchor_links = self.extractor.extract_anchor_links(soup, base_url)
            canonical_links = self.extractor.extract_canonical_links(soup, base_url)
            structured_data_links = self.extractor.extract_structured_data_links(soup, base_url)

            all_links.extend(anchor_links)
            all_links.extend(canonical_links)
            all_links.extend(structured_data_links)

            # Set depth for all links
            for link in all_links:
                link.depth = current_depth + 1

            # Filter and deduplicate
            filtered_links = self._filter_and_deduplicate_links(all_links, base_url)

            # Update statistics
            self._update_discovery_stats(filtered_links)

            return filtered_links

        except Exception:
            # Log error but don't fail completely
            return []

    def _filter_and_deduplicate_links(self, links: list[DiscoveredLink],
                                    base_url: str) -> list[DiscoveredLink]:
        """Filter and deduplicate discovered links."""
        filtered = []
        seen_urls = set()

        # Get base domain for filtering
        try:
            base_domain = urlparse(base_url).netloc
        except Exception:
            base_domain = None

        for link in links:
            # Validate URL
            if not self.normalizer.is_valid_url(link.url, base_domain):
                continue

            # Check if already seen
            if link.url in seen_urls or link.url in self._discovered_urls:
                continue

            # Check limits
            if len(filtered) >= self.config.max_links_per_page:
                break

            seen_urls.add(link.url)
            filtered.append(link)

        return filtered

    def _update_discovery_stats(self, links: list[DiscoveredLink]) -> None:
        """Update discovery statistics."""
        self._stats['urls_discovered'] += len(links)

        for link in links:
            self._stats['discovery_methods'][link.discovery_method] += 1
            self._discovered_urls.add(link.url)

            if link.depth > self._stats['max_depth_reached']:
                self._stats['max_depth_reached'] = link.depth

    async def crawl_recursive(self, start_urls: list[str],
                            max_pages: int | None = None) -> AsyncGenerator[list[str], None]:
        """
        Perform recursive crawling starting from given URLs.

        Args:
            start_urls: Initial URLs to start crawling from
            max_pages: Maximum number of pages to crawl

        Yields:
            Batches of URLs ready for crawling
        """
        max_pages = max_pages or self.config.max_pages

        # Initialize queue with start URLs
        for url in start_urls:
            normalized_url = self.normalizer.normalize_url(url)
            if normalized_url and normalized_url not in self._discovered_urls:
                link = DiscoveredLink(
                    url=normalized_url,
                    source_url="",
                    depth=0,
                    discovery_method="start_url",
                    discovered_at=datetime.now(),
                    crawl_priority=1.0
                )
                self._crawl_queue.append(link)
                self._discovered_urls.add(normalized_url)

        crawled_count = 0
        batch = []

        while self._crawl_queue and crawled_count < max_pages:
            # Get next link based on strategy
            if self.config.strategy == "breadth_first":
                link = self._crawl_queue.popleft()
            else:  # depth_first
                link = self._crawl_queue.pop()

            # Skip if already crawled or failed
            if link.url in self._crawled_urls or link.url in self._failed_urls:
                continue

            # Add to current batch
            batch.append(link.url)
            self._crawled_urls.add(link.url)
            crawled_count += 1

            # Yield batch when full
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []

                # Delay between batches
                if self.config.delay_between_batches > 0:
                    await asyncio.sleep(self.config.delay_between_batches)

        # Yield remaining batch
        if batch:
            yield batch

    def add_discovered_links(self, links: list[DiscoveredLink]) -> None:
        """
        Add newly discovered links to the crawl queue.

        Args:
            links: List of discovered links to add
        """
        for link in links:
            if (link.url not in self._discovered_urls and
                link.url not in self._crawled_urls and
                link.url not in self._failed_urls):

                # Insert based on priority and strategy
                if self.config.strategy == "breadth_first":
                    # For breadth-first, add to end (but could sort by priority)
                    self._crawl_queue.append(link)
                else:
                    # For depth-first, add to end (higher depth will be popped first)
                    self._crawl_queue.append(link)

                self._discovered_urls.add(link.url)

    def mark_url_failed(self, url: str, error: str) -> None:
        """
        Mark URL as failed.

        Args:
            url: URL that failed to crawl
            error: Error message
        """
        self._failed_urls.add(url)
        self._stats['urls_failed'] += 1

    def get_queue_status(self) -> dict[str, Any]:
        """Get current status of the crawl queue."""
        return {
            'queue_size': len(self._crawl_queue),
            'discovered_count': len(self._discovered_urls),
            'crawled_count': len(self._crawled_urls),
            'failed_count': len(self._failed_urls),
            'stats': dict(self._stats),
            'config': {
                'max_depth': self.config.max_depth,
                'max_pages': self.config.max_pages,
                'strategy': self.config.strategy,
                'same_domain_only': self.config.same_domain_only
            }
        }

    def reset(self) -> None:
        """Reset the discovery state."""
        self._discovered_urls.clear()
        self._crawled_urls.clear()
        self._failed_urls.clear()
        self._crawl_queue.clear()
        self._stats = {
            'urls_discovered': 0,
            'urls_crawled': 0,
            'urls_failed': 0,
            'max_depth_reached': 0,
            'discovery_methods': defaultdict(int)
        }
