"""Advanced content extraction with multiple parsing strategies and quality assessment.

This module provides comprehensive content extraction capabilities with fallback
strategies, content validation, and quality assessment for web crawling.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString
from lxml import etree, html
from loguru import logger


class ExtractionStrategy(Enum):
    """Content extraction strategies in order of preference."""
    BEAUTIFULSOUP = "beautifulsoup"
    LXML = "lxml"
    REGEX = "regex"
    SIMPLE_TEXT = "simple_text"


@dataclass
class ContentQuality:
    """Quality assessment of extracted content."""

    # Quality scores (0-100)
    overall_score: float = 0.0
    text_density: float = 0.0  # Ratio of text to markup
    structure_score: float = 0.0  # Presence of semantic elements
    readability_score: float = 0.0  # Based on sentence structure

    # Content metrics
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    heading_count: int = 0
    link_count: int = 0

    # Quality indicators
    has_title: bool = False
    has_meta_description: bool = False
    has_structured_content: bool = False
    has_navigation: bool = False
    language_detected: Optional[str] = None

    # Issues detected
    issues: List[str] = field(default_factory=list)


@dataclass
class ExtractedContent:
    """Comprehensive extracted content from a web page."""

    # Basic content
    title: str = ""
    content: str = ""
    cleaned_content: str = ""
    summary: str = ""

    # Metadata
    meta_description: str = ""
    meta_keywords: List[str] = field(default_factory=list)
    author: str = ""
    publish_date: Optional[str] = None
    language: str = ""

    # Structured content
    headings: List[Tuple[int, str]] = field(default_factory=list)  # (level, text)
    paragraphs: List[str] = field(default_factory=list)
    links: List[Tuple[str, str]] = field(default_factory=list)  # (url, text)
    images: List[Dict[str, str]] = field(default_factory=list)

    # Quality and metadata
    quality: ContentQuality = field(default_factory=ContentQuality)
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.BEAUTIFULSOUP
    processing_time: float = 0.0

    # Raw data for debugging
    raw_html: str = ""


class ContentExtractor:
    """Advanced content extractor with multiple strategies and quality assessment."""

    def __init__(self):
        self.strategies = [
            ExtractionStrategy.BEAUTIFULSOUP,
            ExtractionStrategy.LXML,
            ExtractionStrategy.REGEX,
            ExtractionStrategy.SIMPLE_TEXT
        ]

        # Content removal patterns
        self.noise_selectors = {
            'ads': ['[class*="ad"]', '[id*="ad"]', '.advertisement', '.sponsor'],
            'navigation': ['nav', '.navigation', '.nav-menu', '.breadcrumb'],
            'social': ['.social', '.share', '.sharing', '[class*="social"]'],
            'comments': ['.comments', '.comment', '#comments', '.disqus'],
            'footer': ['footer', '.footer'],
            'sidebar': ['.sidebar', '.side-bar', '.widget'],
            'popup': ['.popup', '.modal', '.overlay'],
            'tracking': ['script[src*="analytics"]', 'script[src*="tracking"]']
        }

        # Content quality indicators
        self.quality_selectors = {
            'article': ['article', '.article', '.post', '.content', 'main'],
            'headings': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
            'paragraphs': ['p'],
            'structured': ['section', 'header', 'main', 'aside', 'footer'],
            'semantic': ['time', 'address', 'blockquote', 'cite']
        }

    def extract(self, html: str, url: str = "") -> ExtractedContent:
        """Extract content using multiple strategies with quality assessment."""
        import time
        start_time = time.time()

        if not html.strip():
            return ExtractedContent(processing_time=time.time() - start_time)

        result = None

        # Try extraction strategies in order
        for strategy in self.strategies:
            try:
                if strategy == ExtractionStrategy.BEAUTIFULSOUP:
                    result = self._extract_with_beautifulsoup(html, url)
                elif strategy == ExtractionStrategy.LXML:
                    result = self._extract_with_lxml(html, url)
                elif strategy == ExtractionStrategy.REGEX:
                    result = self._extract_with_regex(html, url)
                elif strategy == ExtractionStrategy.SIMPLE_TEXT:
                    result = self._extract_simple_text(html, url)

                if result and result.content.strip():
                    result.extraction_strategy = strategy
                    break

            except Exception as e:
                logger.debug(f"Extraction failed with {strategy.value}: {e}")
                continue

        if not result:
            result = ExtractedContent()

        # Store raw HTML for debugging
        result.raw_html = html[:10000]  # Limit size

        # Assess content quality
        result.quality = self._assess_quality(result, html)

        result.processing_time = time.time() - start_time
        return result

    def _extract_with_beautifulsoup(self, html: str, url: str) -> ExtractedContent:
        """Extract content using BeautifulSoup with advanced cleaning."""
        soup = BeautifulSoup(html, 'lxml')
        result = ExtractedContent()

        # Remove noise elements
        self._remove_noise_elements(soup)

        # Extract basic metadata
        result.title = self._extract_title(soup)
        result.meta_description = self._extract_meta_description(soup)
        result.meta_keywords = self._extract_meta_keywords(soup)
        result.author = self._extract_author(soup)
        result.publish_date = self._extract_publish_date(soup)
        result.language = self._extract_language(soup)

        # Find main content area
        main_content = self._find_main_content(soup)
        if not main_content:
            main_content = soup

        # Extract structured content
        result.headings = self._extract_headings(main_content)
        result.paragraphs = self._extract_paragraphs(main_content)
        result.links = self._extract_links(main_content, url)
        result.images = self._extract_images(main_content, url)

        # Extract and clean main content
        result.content = self._extract_text_content(main_content)
        result.cleaned_content = self._clean_text(result.content)
        result.summary = self._generate_summary(result.cleaned_content)

        return result

    def _extract_with_lxml(self, html: str, url: str) -> ExtractedContent:
        """Extract content using lxml for faster processing."""
        try:
            doc = html.fromstring(html)
        except Exception:
            # Fallback to etree parsing
            try:
                parser = etree.HTMLParser()
                doc = etree.fromstring(html.encode('utf-8'), parser)
            except Exception as e:
                raise ValueError(f"Failed to parse HTML with lxml: {e}")

        result = ExtractedContent()

        # Extract basic metadata
        result.title = self._extract_title_lxml(doc)
        result.meta_description = self._extract_meta_description_lxml(doc)
        result.language = self._extract_language_lxml(doc)

        # Remove script and style elements
        etree.strip_elements(doc, 'script', 'style', with_tail=False)

        # Extract text content
        result.content = self._extract_text_lxml(doc)
        result.cleaned_content = self._clean_text(result.content)
        result.summary = self._generate_summary(result.cleaned_content)

        # Extract links
        result.links = self._extract_links_lxml(doc, url)

        return result

    def _extract_with_regex(self, html: str, url: str) -> ExtractedContent:
        """Extract content using regex patterns as fallback."""
        result = ExtractedContent()

        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            result.title = self._clean_text(title_match.group(1))

        # Extract meta description
        meta_desc = re.search(
            r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
            html, re.IGNORECASE
        )
        if meta_desc:
            result.meta_description = meta_desc.group(1)

        # Remove script and style tags
        html_clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html_clean = re.sub(r'<style[^>]*>.*?</style>', '', html_clean, flags=re.IGNORECASE | re.DOTALL)

        # Remove HTML tags
        text_content = re.sub(r'<[^>]+>', '', html_clean)

        result.content = text_content
        result.cleaned_content = self._clean_text(text_content)
        result.summary = self._generate_summary(result.cleaned_content)

        return result

    def _extract_simple_text(self, html: str, url: str) -> ExtractedContent:
        """Simple text extraction as last resort."""
        result = ExtractedContent()

        # Very basic HTML tag removal
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)

        result.content = text
        result.cleaned_content = self._clean_text(text)
        result.summary = self._generate_summary(result.cleaned_content)

        return result

    def _remove_noise_elements(self, soup: BeautifulSoup) -> None:
        """Remove noise elements from soup."""
        for noise_type, selectors in self.noise_selectors.items():
            for selector in selectors:
                try:
                    for element in soup.select(selector):
                        element.decompose()
                except Exception:
                    continue

    def _find_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content area of the page."""
        # Try article and main semantic elements first
        for selector in ['article', 'main', '[role="main"]']:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content

        # Try common content containers
        content_selectors = [
            '.content', '.post', '.article', '.entry',
            '#content', '#main', '#primary', '.main'
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                text_length = len(main_content.get_text().strip())
                if text_length > 100:  # Minimum content length
                    return main_content

        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title with fallbacks."""
        # Try title tag first
        title_tag = soup.find('title')
        if title_tag:
            title = self._clean_text(title_tag.get_text())
            if title:
                return title

        # Try h1 tags
        h1_tag = soup.find('h1')
        if h1_tag:
            title = self._clean_text(h1_tag.get_text())
            if title:
                return title

        # Try meta property og:title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return self._clean_text(og_title['content'])

        return ""

    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        # Try standard meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return self._clean_text(meta_desc['content'])

        # Try Open Graph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return self._clean_text(og_desc['content'])

        return ""

    def _extract_meta_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract meta keywords."""
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            keywords = meta_keywords['content'].split(',')
            return [self._clean_text(kw) for kw in keywords if self._clean_text(kw)]
        return []

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author information."""
        # Try meta author tag
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            return self._clean_text(meta_author['content'])

        # Try common author selectors
        author_selectors = ['.author', '.by-author', '[rel="author"]', '.byline']
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                author_text = self._clean_text(author_elem.get_text())
                if author_text:
                    return author_text

        return ""

    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date."""
        # Try time elements with datetime attribute
        time_elem = soup.find('time', attrs={'datetime': True})
        if time_elem:
            return time_elem['datetime']

        # Try meta publication date tags
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'meta[name="date"]'
        ]

        for selector in date_selectors:
            meta_elem = soup.select_one(selector)
            if meta_elem and meta_elem.get('content'):
                return meta_elem['content']

        return None

    def _extract_language(self, soup: BeautifulSoup) -> str:
        """Extract page language."""
        # Try html lang attribute
        html_elem = soup.find('html')
        if html_elem and html_elem.get('lang'):
            return html_elem['lang']

        # Try meta language tags
        meta_lang = soup.find('meta', attrs={'http-equiv': 'content-language'})
        if meta_lang and meta_lang.get('content'):
            return meta_lang['content']

        return ""

    def _extract_headings(self, soup: BeautifulSoup) -> List[Tuple[int, str]]:
        """Extract headings with their levels."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                text = self._clean_text(heading.get_text())
                if text:
                    headings.append((level, text))
        return headings

    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraph content."""
        paragraphs = []
        for p in soup.find_all('p'):
            text = self._clean_text(p.get_text())
            if text and len(text) > 20:  # Minimum paragraph length
                paragraphs.append(text)
        return paragraphs

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
        """Extract links with their text."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = self._clean_text(link.get_text())

            # Convert relative URLs to absolute
            if base_url:
                href = urljoin(base_url, href)

            if href and text:
                links.append((href, text))
        return links

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract image information."""
        images = []
        for img in soup.find_all('img'):
            img_info = {}

            # Extract src
            src = img.get('src') or img.get('data-src')
            if src:
                if base_url:
                    src = urljoin(base_url, src)
                img_info['src'] = src

            # Extract alt text
            if img.get('alt'):
                img_info['alt'] = self._clean_text(img['alt'])

            # Extract title
            if img.get('title'):
                img_info['title'] = self._clean_text(img['title'])

            if img_info:
                images.append(img_info)

        return images

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from soup."""
        # Get all text content
        text_parts = []

        for element in soup.descendants:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    text_parts.append(text)

        return ' '.join(text_parts)

    def _extract_title_lxml(self, doc) -> str:
        """Extract title using lxml."""
        title_elements = doc.xpath('//title/text()')
        if title_elements:
            return self._clean_text(title_elements[0])

        h1_elements = doc.xpath('//h1/text()')
        if h1_elements:
            return self._clean_text(h1_elements[0])

        return ""

    def _extract_meta_description_lxml(self, doc) -> str:
        """Extract meta description using lxml."""
        desc_elements = doc.xpath('//meta[@name="description"]/@content')
        if desc_elements:
            return self._clean_text(desc_elements[0])
        return ""

    def _extract_language_lxml(self, doc) -> str:
        """Extract language using lxml."""
        lang_elements = doc.xpath('//html/@lang')
        if lang_elements:
            return lang_elements[0]
        return ""

    def _extract_text_lxml(self, doc) -> str:
        """Extract text content using lxml."""
        text_content = doc.text_content()
        return self._clean_text(text_content)

    def _extract_links_lxml(self, doc, base_url: str) -> List[Tuple[str, str]]:
        """Extract links using lxml."""
        links = []
        for link in doc.xpath('//a[@href]'):
            href = link.get('href')
            text = self._clean_text(link.text_content())

            if base_url and href:
                href = urljoin(base_url, href)

            if href and text:
                links.append((href, text))

        return links

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Decode HTML entities
        from html import unescape
        text = unescape(text)

        return text

    def _generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary."""
        if not text or len(text) < 100:
            return text

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if not sentences:
            return text[:200]

        # Take first few sentences or up to certain length
        summary_sentences = []
        total_length = 0

        for sentence in sentences[:max_sentences]:
            if total_length + len(sentence) > 500:  # Max summary length
                break
            summary_sentences.append(sentence)
            total_length += len(sentence)

        return '. '.join(summary_sentences) + '.' if summary_sentences else text[:200]

    def _assess_quality(self, content: ExtractedContent, html: str) -> ContentQuality:
        """Assess the quality of extracted content."""
        quality = ContentQuality()

        # Calculate basic metrics
        quality.word_count = len(content.cleaned_content.split())
        quality.sentence_count = len(re.findall(r'[.!?]+', content.cleaned_content))
        quality.paragraph_count = len(content.paragraphs)
        quality.heading_count = len(content.headings)
        quality.link_count = len(content.links)

        # Quality indicators
        quality.has_title = bool(content.title.strip())
        quality.has_meta_description = bool(content.meta_description.strip())
        quality.has_structured_content = bool(content.headings and content.paragraphs)
        quality.language_detected = content.language

        # Calculate quality scores
        quality.text_density = self._calculate_text_density(content.cleaned_content, html)
        quality.structure_score = self._calculate_structure_score(content)
        quality.readability_score = self._calculate_readability_score(content.cleaned_content)

        # Overall quality score (weighted average)
        quality.overall_score = (
            quality.text_density * 0.3 +
            quality.structure_score * 0.4 +
            quality.readability_score * 0.3
        )

        # Detect quality issues
        quality.issues = self._detect_quality_issues(content, quality)

        return quality

    def _calculate_text_density(self, text: str, html: str) -> float:
        """Calculate text density (text vs markup ratio)."""
        if not html or not text:
            return 0.0

        text_length = len(text.strip())
        html_length = len(html)

        if html_length == 0:
            return 0.0

        density = min(text_length / html_length, 1.0) * 100
        return density

    def _calculate_structure_score(self, content: ExtractedContent) -> float:
        """Calculate content structure quality score."""
        score = 0.0

        # Title presence and quality
        if content.title:
            score += 20
            if 10 <= len(content.title.split()) <= 15:  # Good title length
                score += 10

        # Meta description
        if content.meta_description:
            score += 15

        # Headings structure
        if content.headings:
            score += 20
            # Bonus for hierarchical headings
            heading_levels = [level for level, _ in content.headings]
            if len(set(heading_levels)) > 1:
                score += 10

        # Paragraph content
        if content.paragraphs:
            score += 15
            if len(content.paragraphs) >= 3:
                score += 10

        # Language detection
        if content.language:
            score += 10

        return min(score, 100.0)

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score."""
        if not text:
            return 0.0

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)

        # Average characters per word
        avg_chars_per_word = sum(len(word) for word in words) / len(words)

        # Simple readability score (inverse of complexity)
        # Good readability: 15-20 words per sentence, 4-6 chars per word
        sentence_score = max(0, 100 - abs(avg_words_per_sentence - 17.5) * 2)
        word_score = max(0, 100 - abs(avg_chars_per_word - 5) * 10)

        return (sentence_score + word_score) / 2

    def _detect_quality_issues(self, content: ExtractedContent, quality: ContentQuality) -> List[str]:
        """Detect content quality issues."""
        issues = []

        # Content length issues
        if quality.word_count < 50:
            issues.append("Very short content")
        elif quality.word_count < 100:
            issues.append("Short content")

        # Structure issues
        if not content.title:
            issues.append("Missing title")
        if not content.meta_description:
            issues.append("Missing meta description")
        if not content.headings:
            issues.append("No headings found")
        if quality.paragraph_count < 2:
            issues.append("Insufficient paragraph structure")

        # Content quality issues
        if quality.text_density < 20:
            issues.append("Low text density (too much markup)")
        if quality.readability_score < 30:
            issues.append("Poor readability")

        # Language issues
        if not content.language:
            issues.append("Language not detected")

        return issues