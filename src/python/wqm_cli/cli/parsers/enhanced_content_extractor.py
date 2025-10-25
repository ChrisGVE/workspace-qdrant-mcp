"""
Enhanced content extraction with quality filtering and structured data parsing.

This module provides advanced content extraction capabilities for web crawling,
including boilerplate removal, structured data extraction, and media cataloging.
"""

import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from loguru import logger


class ContentQualityMetrics:
    """Metrics for evaluating content quality."""

    def __init__(self):
        self.text_length = 0
        self.paragraph_count = 0
        self.link_density = 0.0
        self.ad_indicators = 0
        self.navigation_indicators = 0
        self.boilerplate_score = 0.0
        self.readability_score = 0.0


class StructuredData:
    """Container for structured data extracted from web pages."""

    def __init__(self):
        self.json_ld: list[dict[str, Any]] = []
        self.microdata: dict[str, Any] = {}
        self.open_graph: dict[str, str] = {}
        self.twitter_cards: dict[str, str] = {}
        self.schema_org: dict[str, Any] = {}


class MediaLinks:
    """Container for media links found in web pages."""

    def __init__(self):
        self.images: list[dict[str, str]] = []
        self.videos: list[dict[str, str]] = []
        self.audio: list[dict[str, str]] = []
        self.documents: list[dict[str, str]] = []


class EnhancedContentExtractor:
    """Enhanced content extractor with quality filtering and structured data parsing."""

    def __init__(self):
        if not BS4_AVAILABLE:
            raise RuntimeError("Enhanced content extraction requires 'beautifulsoup4'")

        # Patterns for identifying low-quality content
        self.boilerplate_selectors = [
            'nav', 'header', 'footer', 'aside', '.sidebar', '.navigation',
            '.menu', '.breadcrumb', '.pagination', '.social-share',
            '.advertisement', '.ad', '.banner', '.popup', '.modal'
        ]

        self.ad_keywords = {
            'advertisement', 'sponsored', 'promo', 'banner', 'popup',
            'subscribe', 'newsletter', 'marketing', 'promotion'
        }

        self.navigation_keywords = {
            'navigation', 'menu', 'breadcrumb', 'sitemap', 'archive',
            'category', 'tag', 'filter', 'sort', 'previous', 'next'
        }

        # Content quality thresholds
        self.min_paragraph_length = 50
        self.max_link_density = 0.3
        self.min_text_content_ratio = 0.1

    def extract_content(self, html: str, base_url: str = "") -> dict[str, Any]:
        """
        Extract enhanced content from HTML.

        Args:
            html: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Dictionary containing extracted content, metadata, and structured data
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract basic metadata
            metadata = self._extract_metadata(soup)

            # Extract structured data
            structured_data = self._extract_structured_data(soup)

            # Extract media links
            media_links = self._extract_media_links(soup, base_url)

            # Extract and filter main content
            main_content = self._extract_main_content(soup)
            quality_metrics = self._evaluate_content_quality(main_content, soup)

            # Extract all text links
            text_links = self._extract_text_links(soup, base_url)

            return {
                'main_content': main_content,
                'metadata': metadata,
                'structured_data': structured_data.__dict__,
                'media_links': media_links.__dict__,
                'text_links': text_links,
                'quality_metrics': quality_metrics.__dict__,
                'word_count': len(main_content.split()),
                'char_count': len(main_content)
            }

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {
                'main_content': '',
                'metadata': {},
                'structured_data': {},
                'media_links': {},
                'text_links': [],
                'quality_metrics': {},
                'word_count': 0,
                'char_count': 0,
                'error': str(e)
            }

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract basic page metadata."""
        metadata = {}

        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property') or tag.get('http-equiv')
            content = tag.get('content')

            if name and content:
                name = name.lower()
                if name in ['description', 'keywords', 'author', 'robots', 'viewport']:
                    metadata[name] = content
                elif name.startswith('og:'):
                    metadata[name] = content
                elif name.startswith('twitter:'):
                    metadata[name] = content

        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')

        return metadata

    def _extract_structured_data(self, soup: BeautifulSoup) -> StructuredData:
        """Extract structured data from the page."""
        structured = StructuredData()

        # JSON-LD extraction
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured.json_ld.append(data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse JSON-LD: {e}")

        # Open Graph
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for tag in og_tags:
            prop = tag.get('property')
            content = tag.get('content')
            if prop and content:
                structured.open_graph[prop] = content

        # Twitter Cards
        twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            name = tag.get('name')
            content = tag.get('content')
            if name and content:
                structured.twitter_cards[name] = content

        # Microdata extraction (basic)
        microdata_items = soup.find_all(attrs={'itemscope': True})
        for item in microdata_items:
            item_type = item.get('itemtype')
            if item_type:
                properties = {}
                prop_elements = item.find_all(attrs={'itemprop': True})
                for prop_elem in prop_elements:
                    prop_name = prop_elem.get('itemprop')
                    prop_value = (
                        prop_elem.get('content') or
                        prop_elem.get('href') or
                        prop_elem.get_text().strip()
                    )
                    if prop_name and prop_value:
                        properties[prop_name] = prop_value

                if properties:
                    structured.microdata[item_type] = properties

        return structured

    def _extract_media_links(self, soup: BeautifulSoup, base_url: str) -> MediaLinks:
        """Extract media links from the page."""
        media = MediaLinks()

        # Images
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            if src:
                full_url = urljoin(base_url, src)
                media.images.append({
                    'url': full_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', '')
                })

        # Videos
        video_tags = soup.find_all('video')
        for video in video_tags:
            src = video.get('src')
            if src:
                full_url = urljoin(base_url, src)
                media.videos.append({
                    'url': full_url,
                    'type': video.get('type', ''),
                    'controls': video.has_attr('controls'),
                    'autoplay': video.has_attr('autoplay')
                })

            # Source tags within video
            sources = video.find_all('source')
            for source in sources:
                src = source.get('src')
                if src:
                    full_url = urljoin(base_url, src)
                    media.videos.append({
                        'url': full_url,
                        'type': source.get('type', ''),
                        'controls': video.has_attr('controls'),
                        'autoplay': video.has_attr('autoplay')
                    })

        # Audio
        audio_tags = soup.find_all('audio')
        for audio in audio_tags:
            src = audio.get('src')
            if src:
                full_url = urljoin(base_url, src)
                media.audio.append({
                    'url': full_url,
                    'type': audio.get('type', ''),
                    'controls': audio.has_attr('controls'),
                    'autoplay': audio.has_attr('autoplay')
                })

        # Document links
        doc_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                if any(parsed.path.lower().endswith(ext) for ext in doc_extensions):
                    media.documents.append({
                        'url': full_url,
                        'text': link.get_text().strip(),
                        'title': link.get('title', '')
                    })

        return media

    def _extract_text_links(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        """Extract all text links from the page."""
        links = []
        link_tags = soup.find_all('a', href=True)

        for link in link_tags:
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                text = link.get_text().strip()
                if text:  # Only include links with text
                    links.append({
                        'url': full_url,
                        'text': text,
                        'title': link.get('title', '')
                    })

        return links

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content with boilerplate removal."""
        # Remove boilerplate elements
        for selector in self.boilerplate_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Remove script and style tags
        for tag_name in ['script', 'style', 'noscript']:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Try to find main content container
        main_containers = [
            'main', 'article', '.content', '.post', '.entry',
            '#content', '#main', '.main-content', '.article-body'
        ]

        main_content = None
        for selector in main_containers:
            container = soup.select_one(selector)
            if container:
                main_content = container
                break

        # If no main container found, use body
        if main_content is None:
            main_content = soup.find('body') or soup

        # Extract text content
        text_content = []

        # Process paragraphs and headers primarily
        for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = element.get_text().strip()
            if len(text) >= self.min_paragraph_length:
                # Check if it's not likely to be boilerplate
                if not self._is_likely_boilerplate(text, element):
                    text_content.append(text)

        # If we don't have enough content, be less strict
        if len('\n'.join(text_content)) < 500:
            text_content = []
            for element in main_content.find_all(['p', 'div', 'span']):
                text = element.get_text().strip()
                if len(text) >= 20:  # Lower threshold
                    if not self._is_likely_boilerplate(text, element):
                        text_content.append(text)

        return '\n\n'.join(text_content)

    def _is_likely_boilerplate(self, text: str, element: Tag) -> bool:
        """Check if text/element is likely boilerplate."""
        text_lower = text.lower()

        # Check for ad keywords
        ad_score = sum(1 for keyword in self.ad_keywords if keyword in text_lower)
        nav_score = sum(1 for keyword in self.navigation_keywords if keyword in text_lower)

        # Check element classes/ids
        element_classes = ' '.join(element.get('class', [])).lower()
        element_id = (element.get('id') or '').lower()

        boilerplate_indicators = [
            'nav', 'menu', 'sidebar', 'footer', 'header', 'ad',
            'advertisement', 'sponsored', 'social', 'share'
        ]

        class_score = sum(1 for indicator in boilerplate_indicators
                         if indicator in element_classes or indicator in element_id)

        # Very short text is often boilerplate
        if len(text) < 30:
            return True

        # High concentration of links
        links = element.find_all('a')
        if len(links) > 0 and len(text) / len(links) < 50:  # Less than 50 chars per link
            return True

        # High scores indicate boilerplate
        total_score = ad_score + nav_score + class_score
        return total_score >= 2

    def _evaluate_content_quality(self, content: str, soup: BeautifulSoup) -> ContentQualityMetrics:
        """Evaluate the quality of extracted content."""
        metrics = ContentQualityMetrics()

        # Basic metrics
        metrics.text_length = len(content)
        metrics.paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

        # Link density (links per character)
        total_text = soup.get_text()
        links = soup.find_all('a')
        total_link_text = sum(len(link.get_text()) for link in links)

        if len(total_text) > 0:
            metrics.link_density = total_link_text / len(total_text)

        # Ad indicators
        all_text = soup.get_text().lower()
        metrics.ad_indicators = sum(1 for keyword in self.ad_keywords if keyword in all_text)

        # Navigation indicators
        metrics.navigation_indicators = sum(1 for keyword in self.navigation_keywords if keyword in all_text)

        # Boilerplate score (0-1, lower is better)
        boilerplate_elements = len(soup.select(','.join(self.boilerplate_selectors)))
        total_elements = len(soup.find_all())
        if total_elements > 0:
            metrics.boilerplate_score = boilerplate_elements / total_elements

        # Simple readability score (average sentence length)
        sentences = re.split(r'[.!?]+', content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            metrics.readability_score = sum(sentence_lengths) / len(sentence_lengths)

        return metrics
