"""Link discovery and recursive crawling with intelligent depth management.

This module provides comprehensive link discovery, filtering, and recursive
crawling capabilities with loop detection and intelligent depth management.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Pattern
from urllib.parse import urljoin, urlparse, urlunparse

import tldextract
from loguru import logger


class LinkType(Enum):
    """Types of discovered links."""
    INTERNAL = "internal"  # Same domain
    EXTERNAL = "external"  # Different domain
    SUBDOMAIN = "subdomain"  # Same domain, different subdomain
    PROTOCOL = "protocol"  # mailto:, tel:, etc.
    FRAGMENT = "fragment"  # #anchor links
    INVALID = "invalid"  # Malformed URLs


@dataclass
class LinkFilter:
    """Configuration for link filtering."""

    # Domain restrictions
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=set)
    include_subdomains: bool = True

    # Path patterns
    allowed_path_patterns: List[Pattern] = field(default_factory=list)
    blocked_path_patterns: List[Pattern] = field(default_factory=list)

    # File types
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        'html', 'htm', 'php', 'asp', 'aspx', 'jsp', 'pdf', 'doc', 'docx'
    })
    blocked_extensions: Set[str] = field(default_factory=lambda: {
        'css', 'js', 'jpg', 'jpeg', 'png', 'gif', 'ico', 'svg',
        'mp3', 'mp4', 'avi', 'mov', 'zip', 'tar', 'gz'
    })

    # Link types to include
    include_external_links: bool = False
    include_subdomain_links: bool = True
    include_protocol_links: bool = False

    # Content-based filtering
    min_link_text_length: int = 1
    max_link_text_length: int = 200
    blocked_link_text_patterns: List[Pattern] = field(default_factory=list)

    # Crawling behavior
    max_depth: int = 3
    max_links_per_page: int = 100
    respect_nofollow: bool = True

    def __post_init__(self):
        """Initialize pattern lists with default blocked patterns."""
        if not self.blocked_link_text_patterns:
            self.blocked_link_text_patterns = [
                re.compile(r'^\s*$'),  # Empty text
                re.compile(r'^(login|logout|sign\s*in|sign\s*out)$', re.IGNORECASE),
                re.compile(r'^(register|signup|sign\s*up)$', re.IGNORECASE),
                re.compile(r'^(advertisement|ad|sponsored)$', re.IGNORECASE),
                re.compile(r'^(cookie|privacy\s*policy|terms)$', re.IGNORECASE),
            ]

        if not self.blocked_path_patterns:
            self.blocked_path_patterns = [
                re.compile(r'/admin/'),
                re.compile(r'/wp-admin/'),
                re.compile(r'/login'),
                re.compile(r'/logout'),
                re.compile(r'/cgi-bin/'),
                re.compile(r'/\.'),  # Hidden directories
                re.compile(r'/(css|js|images|img|assets)/', re.IGNORECASE),
            ]


@dataclass
class DiscoveredLink:
    """Represents a discovered link with metadata."""

    url: str
    text: str
    link_type: LinkType
    source_url: str
    depth: int

    # Link attributes
    title: str = ""
    rel_attributes: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    # Analysis data
    domain: str = ""
    path: str = ""
    extension: str = ""
    query_params: Dict[str, str] = field(default_factory=dict)

    # Quality metrics
    quality_score: float = 0.0
    priority: int = 1  # 1-10, higher is better

    # Discovery metadata
    discovery_time: float = 0.0
    parent_page_title: str = ""

    def __post_init__(self):
        """Parse URL components after initialization."""
        self._parse_url_components()

    def _parse_url_components(self):
        """Parse and store URL components."""
        try:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc
            self.path = parsed.path
            self.extension = self._extract_extension(self.path)

            # Parse query parameters
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        self.query_params[key] = value
        except Exception as e:
            logger.debug(f"Failed to parse URL components for {self.url}: {e}")


    def _extract_extension(self, path: str) -> str:
        """Extract file extension from path."""
        if '.' in path:
            return path.split('.')[-1].lower()
        return ""


@dataclass
class CrawlSession:
    """Tracks state for a recursive crawling session."""

    session_id: str
    start_url: str
    max_depth: int

    # State tracking
    discovered_urls: Set[str] = field(default_factory=set)
    crawled_urls: Set[str] = field(default_factory=set)
    failed_urls: Set[str] = field(default_factory=set)

    # Link storage
    all_links: List[DiscoveredLink] = field(default_factory=list)
    links_by_depth: Dict[int, List[DiscoveredLink]] = field(default_factory=dict)

    # Statistics
    total_links_found: int = 0
    unique_domains: Set[str] = field(default_factory=set)
    crawl_start_time: float = 0.0

    # Loop detection
    url_visit_counts: Dict[str, int] = field(default_factory=dict)
    redirect_chains: Dict[str, List[str]] = field(default_factory=dict)


class LinkDiscovery:
    """Advanced link discovery and recursive crawling system."""

    def __init__(self, link_filter: Optional[LinkFilter] = None):
        """Initialize link discovery system."""
        self.link_filter = link_filter or LinkFilter()
        self._active_sessions: Dict[str, CrawlSession] = {}

        # URL normalization patterns
        self._normalization_patterns = [
            (re.compile(r'#.*$'), ''),  # Remove fragments
            (re.compile(r'\?utm_.*$'), ''),  # Remove UTM parameters
            (re.compile(r'/$'), ''),  # Remove trailing slash for consistency
            (re.compile(r'/+'), '/'),  # Collapse multiple slashes
        ]

    def discover_links(self, html_content: str, base_url: str,
                      page_title: str = "") -> List[DiscoveredLink]:
        """Discover and analyze links in HTML content."""
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception:
            logger.warning(f"Failed to parse HTML for link discovery from {base_url}")
            return []

        links = []

        # Find all anchor tags
        anchor_tags = soup.find_all('a', href=True)

        # Limit the number of links processed per page
        if len(anchor_tags) > self.link_filter.max_links_per_page:
            anchor_tags = anchor_tags[:self.link_filter.max_links_per_page]
            logger.debug(f"Limited link processing to {self.link_filter.max_links_per_page} links")

        for anchor in anchor_tags:
            link = self._process_anchor_tag(anchor, base_url, page_title)
            if link and self._should_include_link(link):
                links.append(link)

        # Score and prioritize links
        self._score_links(links)

        # Sort by priority and quality
        links.sort(key=lambda x: (x.priority, x.quality_score), reverse=True)

        logger.debug(f"Discovered {len(links)} valid links from {base_url}")
        return links

    def _process_anchor_tag(self, anchor, base_url: str, page_title: str) -> Optional[DiscoveredLink]:
        """Process a single anchor tag into a DiscoveredLink."""
        try:
            href = anchor.get('href', '').strip()
            if not href:
                return None

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            normalized_url = self._normalize_url(absolute_url)

            # Extract link text
            link_text = anchor.get_text(strip=True)
            if not link_text:
                link_text = anchor.get('title', '')

            # Skip if text is too short or too long
            if (len(link_text) < self.link_filter.min_link_text_length or
                len(link_text) > self.link_filter.max_link_text_length):
                return None

            # Check for nofollow attribute
            rel_attrs = anchor.get('rel', [])
            if isinstance(rel_attrs, str):
                rel_attrs = rel_attrs.split()

            if self.link_filter.respect_nofollow and 'nofollow' in rel_attrs:
                return None

            # Determine link type
            link_type = self._classify_link(normalized_url, base_url)

            # Create discovered link
            link = DiscoveredLink(
                url=normalized_url,
                text=link_text,
                link_type=link_type,
                source_url=base_url,
                depth=0,  # Will be set by crawler
                title=anchor.get('title', ''),
                rel_attributes=rel_attrs,
                class_names=anchor.get('class', []),
                parent_page_title=page_title,
                discovery_time=time.time()
            )

            return link

        except Exception as e:
            logger.debug(f"Error processing anchor tag: {e}")
            return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistency."""
        normalized = url

        for pattern, replacement in self._normalization_patterns:
            normalized = pattern.sub(replacement, normalized)

        # Ensure proper encoding
        try:
            parsed = urlparse(normalized)
            # Reconstruct URL to ensure proper encoding
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),  # Lowercase domain
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
        except Exception:
            pass  # Return original if parsing fails

        return normalized

    def _classify_link(self, url: str, base_url: str) -> LinkType:
        """Classify the type of link."""
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)

            # Protocol links (mailto, tel, ftp, etc.)
            if parsed_url.scheme not in ('http', 'https'):
                return LinkType.PROTOCOL

            # Fragment links
            if not parsed_url.netloc and parsed_url.fragment:
                return LinkType.FRAGMENT

            # Invalid URLs
            if not parsed_url.netloc:
                return LinkType.INVALID

            # Extract domains using tldextract for better handling
            url_extract = tldextract.extract(url)
            base_extract = tldextract.extract(base_url)

            url_domain = f"{url_extract.domain}.{url_extract.suffix}"
            base_domain = f"{base_extract.domain}.{base_extract.suffix}"

            # Same domain
            if url_domain == base_domain:
                # Check if same subdomain
                if url_extract.subdomain == base_extract.subdomain:
                    return LinkType.INTERNAL
                else:
                    return LinkType.SUBDOMAIN
            else:
                return LinkType.EXTERNAL

        except Exception as e:
            logger.debug(f"Error classifying link {url}: {e}")
            return LinkType.INVALID

    def _should_include_link(self, link: DiscoveredLink) -> bool:
        """Determine if a link should be included based on filters."""

        # Check link type filters
        if link.link_type == LinkType.EXTERNAL and not self.link_filter.include_external_links:
            return False

        if link.link_type == LinkType.SUBDOMAIN and not self.link_filter.include_subdomain_links:
            return False

        if link.link_type == LinkType.PROTOCOL and not self.link_filter.include_protocol_links:
            return False

        if link.link_type in (LinkType.FRAGMENT, LinkType.INVALID):
            return False

        # Check domain filters
        if self.link_filter.allowed_domains:
            if link.domain not in self.link_filter.allowed_domains:
                # Check if subdomain is allowed
                if not self._is_subdomain_allowed(link.domain):
                    return False

        if link.domain in self.link_filter.blocked_domains:
            return False

        # Check extension filters
        if link.extension:
            if self.link_filter.allowed_extensions and link.extension not in self.link_filter.allowed_extensions:
                return False

            if link.extension in self.link_filter.blocked_extensions:
                return False

        # Check path patterns
        for pattern in self.link_filter.blocked_path_patterns:
            if pattern.search(link.path):
                return False

        if self.link_filter.allowed_path_patterns:
            allowed = False
            for pattern in self.link_filter.allowed_path_patterns:
                if pattern.search(link.path):
                    allowed = True
                    break
            if not allowed:
                return False

        # Check link text patterns
        for pattern in self.link_filter.blocked_link_text_patterns:
            if pattern.search(link.text):
                return False

        return True

    def _is_subdomain_allowed(self, domain: str) -> bool:
        """Check if a subdomain of an allowed domain should be included."""
        if not self.link_filter.include_subdomains:
            return False

        try:
            extracted = tldextract.extract(f"http://{domain}")
            base_domain = f"{extracted.domain}.{extracted.suffix}"
            return base_domain in self.link_filter.allowed_domains
        except Exception:
            return False

    def _score_links(self, links: List[DiscoveredLink]) -> None:
        """Score links based on quality indicators."""
        for link in links:
            score = 50.0  # Base score

            # Link text quality
            text_length = len(link.text)
            if 5 <= text_length <= 50:
                score += 10
            elif text_length > 100:
                score -= 10

            # Descriptive text bonus
            if any(word in link.text.lower() for word in ['article', 'post', 'news', 'story', 'blog']):
                score += 15

            # Navigation penalty
            if any(word in link.text.lower() for word in ['home', 'back', 'previous', 'next', 'menu']):
                score -= 10

            # Internal links bonus
            if link.link_type == LinkType.INTERNAL:
                score += 20
            elif link.link_type == LinkType.SUBDOMAIN:
                score += 10

            # Path quality
            path_segments = [seg for seg in link.path.split('/') if seg]
            if len(path_segments) <= 3:  # Not too deep
                score += 5
            elif len(path_segments) > 6:  # Very deep
                score -= 10

            # File extension bonus
            if link.extension in ['html', 'htm', 'php']:
                score += 5
            elif link.extension in ['pdf', 'doc', 'docx']:
                score += 10

            # Title attribute bonus
            if link.title:
                score += 5

            # Query parameters penalty (often dynamic/tracking)
            if link.query_params:
                score -= 5

            link.quality_score = max(0, min(100, score))

            # Set priority based on quality score
            if link.quality_score >= 80:
                link.priority = 9
            elif link.quality_score >= 60:
                link.priority = 7
            elif link.quality_score >= 40:
                link.priority = 5
            elif link.quality_score >= 20:
                link.priority = 3
            else:
                link.priority = 1


class RecursiveCrawler:
    """Manages recursive crawling with loop detection and depth management."""

    def __init__(self, link_discovery: LinkDiscovery):
        """Initialize recursive crawler."""
        self.link_discovery = link_discovery
        self._max_visits_per_url = 3
        self._max_redirect_chain = 10

    def start_crawl_session(self, start_url: str, max_depth: int, session_id: Optional[str] = None) -> CrawlSession:
        """Start a new crawl session."""
        import time
        import hashlib

        if not session_id:
            session_id = hashlib.md5(f"{start_url}:{time.time()}".encode()).hexdigest()

        session = CrawlSession(
            session_id=session_id,
            start_url=start_url,
            max_depth=max_depth,
            crawl_start_time=time.time()
        )

        # Initialize with start URL at depth 0
        start_link = DiscoveredLink(
            url=start_url,
            text="Start URL",
            link_type=LinkType.INTERNAL,
            source_url="",
            depth=0
        )

        session.discovered_urls.add(start_url)
        session.all_links.append(start_link)
        session.links_by_depth[0] = [start_link]

        self.link_discovery._active_sessions[session_id] = session

        logger.info(f"Started crawl session {session_id} from {start_url} (max depth: {max_depth})")
        return session

    def get_next_urls_to_crawl(self, session_id: str, current_depth: int) -> List[DiscoveredLink]:
        """Get the next batch of URLs to crawl at the current depth."""
        session = self.link_discovery._active_sessions.get(session_id)
        if not session:
            return []

        if current_depth > session.max_depth:
            return []

        # Get uncrawled links at current depth
        uncrawled_links = []

        if current_depth in session.links_by_depth:
            for link in session.links_by_depth[current_depth]:
                if (link.url not in session.crawled_urls and
                    link.url not in session.failed_urls and
                    self._should_crawl_url(session, link.url)):
                    uncrawled_links.append(link)

        # Sort by priority
        uncrawled_links.sort(key=lambda x: (x.priority, x.quality_score), reverse=True)

        logger.debug(f"Found {len(uncrawled_links)} URLs to crawl at depth {current_depth}")
        return uncrawled_links

    def add_discovered_links(self, session_id: str, source_url: str,
                           discovered_links: List[DiscoveredLink], current_depth: int) -> None:
        """Add newly discovered links to the crawl session."""
        session = self.link_discovery._active_sessions.get(session_id)
        if not session:
            return

        next_depth = current_depth + 1
        if next_depth > session.max_depth:
            return

        new_links = []

        for link in discovered_links:
            # Skip if already discovered
            if link.url in session.discovered_urls:
                continue

            # Set depth
            link.depth = next_depth
            link.source_url = source_url

            # Add to session
            session.discovered_urls.add(link.url)
            session.all_links.append(link)

            if next_depth not in session.links_by_depth:
                session.links_by_depth[next_depth] = []
            session.links_by_depth[next_depth].append(link)

            # Update statistics
            session.total_links_found += 1
            if link.domain:
                session.unique_domains.add(link.domain)

            new_links.append(link)

        logger.debug(f"Added {len(new_links)} new links at depth {next_depth} from {source_url}")

    def mark_url_crawled(self, session_id: str, url: str, success: bool = True) -> None:
        """Mark a URL as crawled (successfully or failed)."""
        session = self.link_discovery._active_sessions.get(session_id)
        if not session:
            return

        if success:
            session.crawled_urls.add(url)
        else:
            session.failed_urls.add(url)

        # Update visit count
        session.url_visit_counts[url] = session.url_visit_counts.get(url, 0) + 1

    def handle_redirect(self, session_id: str, original_url: str, redirect_url: str) -> bool:
        """Handle URL redirects and detect redirect loops."""
        session = self.link_discovery._active_sessions.get(session_id)
        if not session:
            return False

        # Initialize redirect chain if needed
        if original_url not in session.redirect_chains:
            session.redirect_chains[original_url] = []

        redirect_chain = session.redirect_chains[original_url]

        # Check for redirect loops
        if redirect_url in redirect_chain:
            logger.warning(f"Redirect loop detected: {original_url} -> {redirect_url}")
            return False

        # Check redirect chain length
        if len(redirect_chain) >= self._max_redirect_chain:
            logger.warning(f"Redirect chain too long for {original_url}")
            return False

        # Add to chain
        redirect_chain.append(redirect_url)

        return True

    def _should_crawl_url(self, session: CrawlSession, url: str) -> bool:
        """Determine if a URL should be crawled based on visit history."""

        # Check visit count
        visit_count = session.url_visit_counts.get(url, 0)
        if visit_count >= self._max_visits_per_url:
            logger.debug(f"Skipping {url} - visited {visit_count} times already")
            return False

        return True

    def get_crawl_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get crawling statistics for a session."""
        session = self.link_discovery._active_sessions.get(session_id)
        if not session:
            return {}

        current_time = time.time()
        elapsed_time = current_time - session.crawl_start_time

        stats = {
            "session_id": session_id,
            "start_url": session.start_url,
            "max_depth": session.max_depth,
            "elapsed_time": elapsed_time,
            "total_links_discovered": session.total_links_found,
            "unique_domains": len(session.unique_domains),
            "urls_crawled": len(session.crawled_urls),
            "urls_failed": len(session.failed_urls),
            "urls_pending": len(session.discovered_urls) - len(session.crawled_urls) - len(session.failed_urls),
            "links_by_depth": {depth: len(links) for depth, links in session.links_by_depth.items()},
            "domains_discovered": list(session.unique_domains),
        }

        return stats

    def end_crawl_session(self, session_id: str) -> Optional[CrawlSession]:
        """End a crawl session and return final results."""
        session = self.link_discovery._active_sessions.pop(session_id, None)
        if session:
            logger.info(f"Ended crawl session {session_id}")
            logger.info(f"Total links discovered: {session.total_links_found}")
            logger.info(f"URLs crawled: {len(session.crawled_urls)}")
            logger.info(f"Unique domains: {len(session.unique_domains)}")

        return session

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old crawl sessions."""
        import time

        cutoff_time = time.time() - (max_age_hours * 3600)
        sessions_to_remove = []

        for session_id, session in self.link_discovery._active_sessions.items():
            if session.crawl_start_time < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.link_discovery._active_sessions[session_id]

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old crawl sessions")

        return len(sessions_to_remove)


# Import time module for timestamp generation
import time