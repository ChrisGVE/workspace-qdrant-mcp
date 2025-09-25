"""
Web Crawling and External Content System

Comprehensive web crawling system with respectful crawling practices,
rate limiting, content extraction, and seamless integration with the
workspace-qdrant-mcp document processing pipeline.

This module provides:
1. Respectful web crawling with robots.txt compliance
2. Rate limiting (2 requests/second default)
3. Single page and recursive crawling with depth limits
4. Content extraction and quality filtering
5. Concurrent connection management with limits
6. Error handling with exponential backoff retry logic
7. HTML parsing with content cleaning
8. Metadata extraction (title, description, keywords)
9. Integration with existing document processing pipeline
"""

import asyncio
import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse, urlunparse
import aiohttp
import aiofiles
from bs4 import BeautifulSoup, Comment
import hashlib
import json
import yaml
from collections import defaultdict, deque
import asyncio
import signal
import sys


class CrawlMode(Enum):
    """Crawling modes supported by the system."""
    SINGLE_PAGE = auto()
    RECURSIVE = auto()
    SITEMAP = auto()


class ContentQuality(Enum):
    """Content quality levels for filtering."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    SPAM = auto()


class CrawlStatus(Enum):
    """Status of individual crawl operations."""
    PENDING = auto()
    IN_PROGRESS = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    RATE_LIMITED = auto()
    ROBOTS_BLOCKED = auto()


@dataclass
class CrawlConfig:
    """Configuration for web crawling operations."""
    max_depth: int = 3
    max_pages: int = 1000
    rate_limit_delay: float = 0.5  # 2 requests/second
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    follow_external_links: bool = False
    respect_robots_txt: bool = True
    user_agent: str = "workspace-qdrant-mcp-crawler/1.0"
    allowed_content_types: Set[str] = field(default_factory=lambda: {
        'text/html', 'application/xhtml+xml', 'text/plain'
    })
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    content_quality_threshold: ContentQuality = ContentQuality.MEDIUM
    extract_images: bool = False
    extract_links: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    cookie_jar: Optional[str] = None
    proxy_url: Optional[str] = None


@dataclass
class ExtractedContent:
    """Extracted content from a web page."""
    url: str
    title: str
    description: str
    keywords: List[str]
    text_content: str
    html_content: str
    metadata: Dict[str, Any]
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    content_hash: str = ""
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    content_quality: ContentQuality = ContentQuality.MEDIUM
    language: Optional[str] = None
    word_count: int = 0
    reading_time_minutes: int = 0


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    url: str
    status: CrawlStatus
    content: Optional[ExtractedContent] = None
    error_message: Optional[str] = None
    response_code: Optional[int] = None
    crawl_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    retry_count: int = 0
    redirected_from: Optional[str] = None


@dataclass
class CrawlSession:
    """Manages state for a crawling session."""
    session_id: str
    start_url: str
    config: CrawlConfig
    mode: CrawlMode
    visited_urls: Set[str] = field(default_factory=set)
    pending_urls: deque = field(default_factory=deque)
    failed_urls: Set[str] = field(default_factory=set)
    results: List[CrawlResult] = field(default_factory=list)
    robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)
    rate_limiter_last_request: Dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    session_stats: Dict[str, int] = field(default_factory=lambda: {
        'pages_crawled': 0, 'pages_failed': 0, 'pages_skipped': 0,
        'bytes_downloaded': 0, 'total_requests': 0
    })


class RobotsTxtParser:
    """Handles robots.txt parsing and compliance checking."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)
        self.last_cache_clean = datetime.now()

    async def can_fetch(self, session: aiohttp.ClientSession, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, '/robots.txt')

            # Clean old cache entries periodically
            await self._clean_cache()

            # Check cache first
            if robots_url in self.cache:
                cache_entry = self.cache[robots_url]
                if datetime.now() - cache_entry['timestamp'] < self.cache_ttl:
                    rp = cache_entry['parser']
                    return rp.can_fetch(self.user_agent, url)

            # Fetch and parse robots.txt
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)

            try:
                async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        rp.set_url(robots_url)
                        rp.feed(robots_content)
                    else:
                        # If robots.txt doesn't exist, allow crawling
                        rp.feed("")

                # Cache the parser
                self.cache[robots_url] = {
                    'parser': rp,
                    'timestamp': datetime.now()
                }

                return rp.can_fetch(self.user_agent, url)

            except Exception:
                # If we can't fetch robots.txt, allow crawling
                return True

        except Exception:
            # On any error, allow crawling (fail open)
            return True

    async def _clean_cache(self):
        """Clean expired entries from cache."""
        if datetime.now() - self.last_cache_clean > timedelta(hours=1):
            current_time = datetime.now()
            expired_keys = [
                key for key, value in self.cache.items()
                if current_time - value['timestamp'] > self.cache_ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            self.last_cache_clean = current_time


class RateLimiter:
    """Implements rate limiting for web requests."""

    def __init__(self, requests_per_second: float = 2.0):
        self.delay = 1.0 / requests_per_second
        self.domain_last_request = {}
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, url: str):
        """Wait if necessary to comply with rate limiting."""
        domain = urlparse(url).netloc

        async with self._lock:
            now = time.time()
            last_request = self.domain_last_request.get(domain, 0)
            time_since_last = now - last_request

            if time_since_last < self.delay:
                wait_time = self.delay - time_since_last
                await asyncio.sleep(wait_time)

            self.domain_last_request[domain] = time.time()


class ContentExtractor:
    """Extracts and processes content from HTML pages."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Patterns for content quality assessment
        self.spam_indicators = [
            r'click here', r'buy now', r'limited time', r'act now',
            r'congratulations', r'you have won', r'free money',
            r'viagra', r'casino', r'poker', r'lottery'
        ]

        self.quality_indicators = [
            r'article', r'blog', r'news', r'research', r'study',
            r'analysis', r'review', r'guide', r'tutorial'
        ]

    def extract_content(self, html: str, url: str) -> ExtractedContent:
        """Extract structured content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        self._remove_unwanted_elements(soup)

        # Extract basic metadata
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        keywords = self._extract_keywords(soup)
        language = self._extract_language(soup)

        # Extract main content
        text_content = self._extract_text_content(soup)

        # Extract links and images
        links = self._extract_links(soup, url) if url else []
        images = self._extract_images(soup, url) if url else []

        # Calculate content metrics
        word_count = len(text_content.split())
        reading_time = max(1, word_count // 200)  # Assume 200 words per minute

        # Assess content quality
        quality = self._assess_content_quality(text_content, html)

        # Generate content hash
        content_hash = hashlib.sha256(text_content.encode()).hexdigest()[:16]

        # Extract additional metadata
        metadata = self._extract_additional_metadata(soup)

        return ExtractedContent(
            url=url,
            title=title,
            description=description,
            keywords=keywords,
            text_content=text_content,
            html_content=str(soup),
            metadata=metadata,
            links=links,
            images=images,
            content_hash=content_hash,
            content_quality=quality,
            language=language,
            word_count=word_count,
            reading_time_minutes=reading_time
        )

    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted elements from HTML."""
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'advertisement', 'ads', 'popup', 'cookie-notice'
        ]

        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove elements with ad-related classes/ids
        ad_patterns = ['ad', 'ads', 'advertisement', 'sponsor', 'promo', 'popup']
        for pattern in ad_patterns:
            for element in soup.find_all(class_=re.compile(pattern, re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(pattern, re.I)):
                element.decompose()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try multiple title sources
        title_sources = [
            soup.find('meta', property='og:title'),
            soup.find('meta', name='twitter:title'),
            soup.find('title'),
            soup.find('h1')
        ]

        for source in title_sources:
            if source:
                if hasattr(source, 'get') and source.get('content'):
                    title = source.get('content').strip()
                elif hasattr(source, 'get_text'):
                    title = source.get_text().strip()
                elif hasattr(source, 'string') and source.string:
                    title = source.string.strip()

                if title and len(title) > 5:  # Reasonable title length
                    return title[:200]  # Limit title length

        return "Untitled Page"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        description_sources = [
            soup.find('meta', name='description'),
            soup.find('meta', property='og:description'),
            soup.find('meta', name='twitter:description')
        ]

        for source in description_sources:
            if source and source.get('content'):
                desc = source.get('content').strip()
                if desc and len(desc) > 10:
                    return desc[:500]  # Limit description length

        # Fallback: use first paragraph
        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text().strip()
            if len(text) > 10:
                return text[:500]

        return ""

    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract keywords from page."""
        keywords = []

        # Meta keywords
        meta_keywords = soup.find('meta', name='keywords')
        if meta_keywords and meta_keywords.get('content'):
            keywords.extend([
                kw.strip() for kw in meta_keywords.get('content').split(',')
                if kw.strip()
            ])

        # Extract from headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = heading.get_text().strip()
            if heading_text:
                # Simple keyword extraction from headings
                words = re.findall(r'\b[a-zA-Z]{3,}\b', heading_text.lower())
                keywords.extend(words[:3])  # Limit per heading

        # Remove duplicates and limit
        return list(dict.fromkeys(keywords))[:20]

    def _extract_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page language."""
        lang_sources = [
            soup.find('html', lang=True),
            soup.find('meta', {'http-equiv': 'content-language'}),
            soup.find('meta', name='language')
        ]

        for source in lang_sources:
            if source:
                if hasattr(source, 'get') and source.get('lang'):
                    return source.get('lang')[:5]  # e.g., 'en-US'
                elif hasattr(source, 'get') and source.get('content'):
                    return source.get('content')[:5]

        return None

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Try to find main content area
        main_content_selectors = [
            'main', '[role="main"]', 'article', '.content', '#content',
            '.main-content', '.post-content', '.entry-content'
        ]

        main_content = None
        for selector in main_content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Use body if no main content area found
        if not main_content:
            main_content = soup.find('body') or soup

        # Extract text with some structure preservation
        text_parts = []

        for element in main_content.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text().strip()
            if text and len(text) > 5:  # Skip very short text
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)
                text_parts.append(text)

        content = '\n\n'.join(text_parts)

        # Clean up the final content
        content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
        content = content.strip()

        return content

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from page."""
        links = []

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)
                # Basic URL validation
                if absolute_url.startswith(('http://', 'https://')):
                    links.append(absolute_url)

        return list(dict.fromkeys(links))  # Remove duplicates

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from page."""
        images = []

        for img in soup.find_all('img', src=True):
            src = img.get('src')
            if src:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, src)
                if absolute_url.startswith(('http://', 'https://')):
                    images.append(absolute_url)

        return list(dict.fromkeys(images))  # Remove duplicates

    def _assess_content_quality(self, text_content: str, html_content: str) -> ContentQuality:
        """Assess the quality of extracted content."""
        if not text_content or len(text_content) < 100:
            return ContentQuality.LOW

        text_lower = text_content.lower()
        html_lower = html_content.lower()

        # Check for spam indicators
        spam_score = sum(
            len(re.findall(pattern, text_lower, re.I))
            for pattern in self.spam_indicators
        )

        if spam_score > 3:
            return ContentQuality.SPAM

        # Check for quality indicators
        quality_score = sum(
            len(re.findall(pattern, text_lower, re.I))
            for pattern in self.quality_indicators
        )

        # Additional quality metrics
        word_count = len(text_content.split())
        sentence_count = len(re.split(r'[.!?]+', text_content))
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Check content structure
        has_headings = bool(re.search(r'<h[1-6]', html_lower))
        has_paragraphs = text_content.count('\n\n') > 2

        # Scoring
        total_score = 0
        total_score += quality_score * 2
        total_score += 1 if word_count > 300 else 0
        total_score += 1 if 10 < avg_sentence_length < 25 else 0
        total_score += 1 if has_headings else 0
        total_score += 1 if has_paragraphs else 0
        total_score -= spam_score

        if total_score >= 5:
            return ContentQuality.HIGH
        elif total_score >= 2:
            return ContentQuality.MEDIUM
        else:
            return ContentQuality.LOW

    def _extract_additional_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract additional metadata from page."""
        metadata = {}

        # Open Graph metadata
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for tag in og_tags:
            prop = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if prop and content:
                metadata[f'og_{prop}'] = content

        # Twitter Card metadata
        twitter_tags = soup.find_all('meta', name=lambda x: x and x.startswith('twitter:'))
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content', '')
            if name and content:
                metadata[f'twitter_{name}'] = content

        # Schema.org structured data
        try:
            schema_scripts = soup.find_all('script', type='application/ld+json')
            for script in schema_scripts:
                if script.string:
                    try:
                        schema_data = json.loads(script.string)
                        metadata['schema_org'] = schema_data
                        break  # Use first valid schema data
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        # Author information
        author_meta = soup.find('meta', name='author')
        if author_meta and author_meta.get('content'):
            metadata['author'] = author_meta.get('content')

        # Publication date
        date_selectors = [
            'meta[name="date"]', 'meta[property="article:published_time"]',
            'meta[name="publish_date"]', 'time[datetime]'
        ]

        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_value = date_elem.get('content') or date_elem.get('datetime')
                if date_value:
                    metadata['publication_date'] = date_value
                    break

        return metadata


class WebCrawler:
    """Main web crawler with comprehensive features."""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.robots_parser = RobotsTxtParser(config.user_agent)
        self.rate_limiter = RateLimiter(1.0 / config.rate_limit_delay)
        self.content_extractor = ContentExtractor()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._setup_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_session()

    async def _setup_session(self):
        """Setup aiohttp session with configuration."""
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)

        headers = {
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        headers.update(self.config.custom_headers)

        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            limit_per_host=min(self.config.max_concurrent_requests, 5),
            ttl_dns_cache=300,
            use_dns_cache=True
        )

        cookie_jar = None
        if self.config.cookie_jar:
            cookie_jar = aiohttp.CookieJar()

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=connector,
            cookie_jar=cookie_jar
        )

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    async def _cleanup_session(self):
        """Cleanup aiohttp session."""
        if self._session:
            await self._session.close()

    async def crawl_single_page(self, url: str) -> CrawlResult:
        """Crawl a single page and extract content."""
        session = CrawlSession(
            session_id=f"single_{int(time.time())}",
            start_url=url,
            config=self.config,
            mode=CrawlMode.SINGLE_PAGE
        )

        result = await self._crawl_url(session, url)
        return result

    async def crawl_recursive(self, start_url: str) -> List[CrawlResult]:
        """Crawl website recursively with depth and domain limits."""
        session = CrawlSession(
            session_id=f"recursive_{int(time.time())}",
            start_url=start_url,
            config=self.config,
            mode=CrawlMode.RECURSIVE
        )

        # Add start URL to pending queue
        session.pending_urls.append((start_url, 0))  # (url, depth)
        results = []

        while session.pending_urls and len(results) < self.config.max_pages:
            # Get next batch of URLs to process
            batch_urls = []
            batch_size = min(self.config.max_concurrent_requests, len(session.pending_urls))

            for _ in range(batch_size):
                if session.pending_urls:
                    batch_urls.append(session.pending_urls.popleft())

            if not batch_urls:
                break

            # Process batch concurrently
            tasks = []
            for url, depth in batch_urls:
                if url not in session.visited_urls:
                    task = self._process_url_with_depth(session, url, depth)
                    tasks.append(task)

            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, CrawlResult):
                        results.append(result)
                        session.results.append(result)

        self.logger.info(f"Recursive crawl completed: {len(results)} pages crawled")
        return results

    async def _process_url_with_depth(self, session: CrawlSession, url: str, depth: int) -> CrawlResult:
        """Process URL and add discovered links if within depth limit."""
        result = await self._crawl_url(session, url)
        session.visited_urls.add(url)

        # Add discovered links to queue if within depth limit
        if (depth < self.config.max_depth and
            result.status == CrawlStatus.SUCCESS and
            result.content and
            result.content.links):

            base_domain = urlparse(session.start_url).netloc

            for link_url in result.content.links:
                # Check domain restrictions
                if not self.config.follow_external_links:
                    link_domain = urlparse(link_url).netloc
                    if link_domain != base_domain:
                        continue

                # Add to pending queue if not already processed
                if (link_url not in session.visited_urls and
                    link_url not in [u for u, d in session.pending_urls]):
                    session.pending_urls.append((link_url, depth + 1))

        return result

    async def _crawl_url(self, session: CrawlSession, url: str) -> CrawlResult:
        """Crawl a single URL with error handling and retries."""
        start_time = time.time()

        for attempt in range(self.config.retry_attempts):
            try:
                # Rate limiting
                await self.rate_limiter.wait_if_needed(url)

                # Check robots.txt compliance
                if self.config.respect_robots_txt:
                    if not await self.robots_parser.can_fetch(self._session, url):
                        return CrawlResult(
                            url=url,
                            status=CrawlStatus.ROBOTS_BLOCKED,
                            error_message="Blocked by robots.txt",
                            processing_time_seconds=time.time() - start_time
                        )

                # Make request with semaphore for concurrency control
                async with self._semaphore:
                    result = await self._fetch_and_extract(session, url)
                    result.retry_count = attempt
                    result.processing_time_seconds = time.time() - start_time
                    return result

            except asyncio.TimeoutError:
                error_msg = f"Request timeout after {self.config.request_timeout}s"
                self.logger.warning(f"Timeout for {url}: {error_msg}")

                if attempt == self.config.retry_attempts - 1:
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.FAILED,
                        error_message=error_msg,
                        retry_count=attempt,
                        processing_time_seconds=time.time() - start_time
                    )

            except Exception as e:
                error_msg = f"Error crawling {url}: {str(e)}"
                self.logger.error(error_msg)

                if attempt == self.config.retry_attempts - 1:
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.FAILED,
                        error_message=error_msg,
                        retry_count=attempt,
                        processing_time_seconds=time.time() - start_time
                    )

            # Exponential backoff for retries
            if attempt < self.config.retry_attempts - 1:
                wait_time = self.config.retry_backoff_factor ** attempt
                await asyncio.sleep(wait_time)

        return CrawlResult(
            url=url,
            status=CrawlStatus.FAILED,
            error_message="Max retries exceeded",
            retry_count=self.config.retry_attempts,
            processing_time_seconds=time.time() - start_time
        )

    async def _fetch_and_extract(self, session: CrawlSession, url: str) -> CrawlResult:
        """Fetch URL and extract content."""
        try:
            async with self._session.get(url) as response:
                # Update session stats
                session.session_stats['total_requests'] += 1

                # Check response status
                if response.status >= 400:
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.FAILED,
                        response_code=response.status,
                        error_message=f"HTTP {response.status}"
                    )

                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in self.config.allowed_content_types):
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.SKIPPED,
                        response_code=response.status,
                        error_message=f"Unsupported content type: {content_type}"
                    )

                # Check content size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.config.max_content_size:
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.SKIPPED,
                        response_code=response.status,
                        error_message=f"Content too large: {content_length} bytes"
                    )

                # Read content with size limit
                content = await response.text()

                if len(content) > self.config.max_content_size:
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.SKIPPED,
                        response_code=response.status,
                        error_message=f"Content too large: {len(content)} characters"
                    )

                # Update stats
                session.session_stats['bytes_downloaded'] += len(content)

                # Extract content
                extracted_content = self.content_extractor.extract_content(content, url)

                # Check content quality threshold
                if extracted_content.content_quality.value < self.config.content_quality_threshold.value:
                    return CrawlResult(
                        url=url,
                        status=CrawlStatus.SKIPPED,
                        response_code=response.status,
                        error_message=f"Content quality below threshold: {extracted_content.content_quality.name}"
                    )

                # Check for redirects
                redirected_from = None
                if response.history:
                    redirected_from = str(response.history[0].url)

                session.session_stats['pages_crawled'] += 1

                return CrawlResult(
                    url=url,
                    status=CrawlStatus.SUCCESS,
                    content=extracted_content,
                    response_code=response.status,
                    redirected_from=redirected_from
                )

        except Exception as e:
            session.session_stats['pages_failed'] += 1
            raise


class WebCrawlingSystem:
    """High-level interface for web crawling operations."""

    def __init__(self, config: Optional[CrawlConfig] = None):
        self.config = config or CrawlConfig()
        self.logger = logging.getLogger(__name__)
        self.crawler = WebCrawler(self.config)

    async def crawl_url(self, url: str) -> CrawlResult:
        """Crawl a single URL."""
        self.logger.info(f"Starting single page crawl: {url}")

        async with self.crawler:
            result = await self.crawler.crawl_single_page(url)

        self.logger.info(f"Single page crawl completed: {result.status.name}")
        return result

    async def crawl_website(self, start_url: str, max_pages: int = None, max_depth: int = None) -> List[CrawlResult]:
        """Crawl website recursively."""
        if max_pages:
            self.config.max_pages = max_pages
        if max_depth:
            self.config.max_depth = max_depth

        self.logger.info(f"Starting recursive crawl: {start_url} (max_pages={self.config.max_pages}, max_depth={self.config.max_depth})")

        async with self.crawler:
            results = await self.crawler.crawl_recursive(start_url)

        success_count = sum(1 for r in results if r.status == CrawlStatus.SUCCESS)
        self.logger.info(f"Recursive crawl completed: {success_count}/{len(results)} pages successful")
        return results

    async def crawl_urls_batch(self, urls: List[str]) -> List[CrawlResult]:
        """Crawl multiple URLs in parallel."""
        self.logger.info(f"Starting batch crawl: {len(urls)} URLs")

        async with self.crawler:
            tasks = []
            for url in urls:
                task = self.crawler.crawl_single_page(url)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and convert to results
            crawl_results = []
            for i, result in enumerate(results):
                if isinstance(result, CrawlResult):
                    crawl_results.append(result)
                else:
                    # Create failed result for exceptions
                    crawl_results.append(CrawlResult(
                        url=urls[i],
                        status=CrawlStatus.FAILED,
                        error_message=str(result)
                    ))

        success_count = sum(1 for r in crawl_results if r.status == CrawlStatus.SUCCESS)
        self.logger.info(f"Batch crawl completed: {success_count}/{len(crawl_results)} pages successful")
        return crawl_results

    def export_results(self, results: List[CrawlResult], format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export crawl results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not output_path:
            output_path = f"crawl_results_{timestamp}.{format}"

        if format.lower() == 'json':
            self._export_json(results, output_path)
        elif format.lower() == 'yaml':
            self._export_yaml(results, output_path)
        elif format.lower() == 'csv':
            self._export_csv(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Results exported to: {output_path}")
        return output_path

    def _export_json(self, results: List[CrawlResult], output_path: str):
        """Export results to JSON."""
        data = []
        for result in results:
            result_dict = {
                'url': result.url,
                'status': result.status.name,
                'response_code': result.response_code,
                'error_message': result.error_message,
                'crawl_timestamp': result.crawl_timestamp.isoformat(),
                'processing_time_seconds': result.processing_time_seconds,
                'retry_count': result.retry_count,
                'redirected_from': result.redirected_from
            }

            if result.content:
                result_dict['content'] = {
                    'title': result.content.title,
                    'description': result.content.description,
                    'keywords': result.content.keywords,
                    'text_content': result.content.text_content[:1000] + '...' if len(result.content.text_content) > 1000 else result.content.text_content,
                    'content_hash': result.content.content_hash,
                    'content_quality': result.content.content_quality.name,
                    'language': result.content.language,
                    'word_count': result.content.word_count,
                    'reading_time_minutes': result.content.reading_time_minutes,
                    'links_count': len(result.content.links),
                    'images_count': len(result.content.images),
                    'metadata': result.content.metadata
                }

            data.append(result_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_yaml(self, results: List[CrawlResult], output_path: str):
        """Export results to YAML."""
        # Similar to JSON but use YAML format
        data = []
        for result in results:
            result_dict = {
                'url': result.url,
                'status': result.status.name,
                'title': result.content.title if result.content else None,
                'word_count': result.content.word_count if result.content else 0,
                'content_quality': result.content.content_quality.name if result.content else None
            }
            data.append(result_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _export_csv(self, results: List[CrawlResult], output_path: str):
        """Export results to CSV."""
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'URL', 'Status', 'Response Code', 'Title', 'Word Count',
                'Content Quality', 'Language', 'Error Message', 'Processing Time'
            ])

            # Data rows
            for result in results:
                writer.writerow([
                    result.url,
                    result.status.name,
                    result.response_code or '',
                    result.content.title if result.content else '',
                    result.content.word_count if result.content else 0,
                    result.content.content_quality.name if result.content else '',
                    result.content.language if result.content else '',
                    result.error_message or '',
                    result.processing_time_seconds
                ])

    def get_config(self) -> CrawlConfig:
        """Get current crawling configuration."""
        return self.config

    def update_config(self, **kwargs):
        """Update crawling configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

        # Recreate crawler with new config
        self.crawler = WebCrawler(self.config)


def create_default_config() -> CrawlConfig:
    """Create default crawling configuration."""
    return CrawlConfig(
        max_depth=2,
        max_pages=100,
        rate_limit_delay=0.5,  # 2 requests per second
        max_concurrent_requests=5,
        request_timeout=30,
        retry_attempts=3,
        follow_external_links=False,
        respect_robots_txt=True,
        content_quality_threshold=ContentQuality.MEDIUM
    )


async def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Web Crawling System")
    parser.add_argument('url', help='URL to crawl')
    parser.add_argument('--recursive', action='store_true', help='Crawl recursively')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages to crawl')
    parser.add_argument('--max-depth', type=int, default=2, help='Maximum crawl depth')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['json', 'yaml', 'csv'], default='json', help='Export format')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = create_default_config()
    config.max_pages = args.max_pages
    config.max_depth = args.max_depth

    # Create crawler
    crawler_system = WebCrawlingSystem(config)

    try:
        if args.recursive:
            results = await crawler_system.crawl_website(args.url)
        else:
            result = await crawler_system.crawl_url(args.url)
            results = [result]

        # Export results
        output_path = crawler_system.export_results(
            results,
            format=args.format,
            output_path=args.output
        )

        print(f"Crawling completed. Results saved to: {output_path}")

        # Print summary
        success_count = sum(1 for r in results if r.status == CrawlStatus.SUCCESS)
        print(f"Summary: {success_count}/{len(results)} pages successfully crawled")

    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during crawling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())