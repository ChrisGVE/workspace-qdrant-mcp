"""
Advanced Content Extraction with Multiple Parsing Strategies

This module provides comprehensive content extraction capabilities from web pages
and documents with multiple parsing strategies, quality assessment, and metadata collection.

Features:
    - Multiple parsing strategies (BeautifulSoup, readability, custom)
    - Content quality assessment and scoring
    - Metadata extraction (title, description, keywords, author)
    - Text cleaning and normalization
    - Language detection
    - Content structure analysis
    - Link extraction and validation
    - Image and media metadata collection
    - Fallback mechanisms for failed extractions
    - Performance optimization for large documents

Parsing Strategies:
    1. BeautifulSoup - Traditional HTML parsing with tag-based extraction
    2. Readability - Focus on main content extraction (article-like content)
    3. Custom - Configurable rules-based extraction
    4. Hybrid - Combination of multiple strategies with ranking

Example:
    ```python
    from workspace_qdrant_mcp.core.content_extractor import ContentExtractor

    extractor = ContentExtractor()

    # Extract content with quality assessment
    result = await extractor.extract_content(
        html_content="<html>...</html>",
        url="https://example.com",
        strategy="hybrid"
    )

    print(f"Quality score: {result.quality_score}")
    print(f"Main content: {result.main_content}")
    print(f"Metadata: {result.metadata}")
    ```
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment


@dataclass
class ContentExtractionResult:
    """Result of content extraction operation."""

    url: str
    main_content: str | None = None
    title: str | None = None
    description: str | None = None
    keywords: list[str] = field(default_factory=list)
    author: str | None = None
    publish_date: datetime | None = None
    language: str | None = None

    # Quality metrics
    quality_score: float = 0.0
    content_length: int = 0
    word_count: int = 0
    paragraph_count: int = 0

    # Extracted elements
    links: list[dict[str, str]] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    headings: list[dict[str, str]] = field(default_factory=list)

    # Processing info
    extraction_strategy: str = "unknown"
    processing_time: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Raw data for fallback
    raw_html: str | None = None
    raw_text: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionConfig:
    """Configuration for content extraction."""

    # Strategy selection
    preferred_strategy: str = "hybrid"
    fallback_strategies: list[str] = field(default_factory=lambda: ["beautifulsoup", "custom"])

    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 1000000  # 1MB
    min_word_count: int = 10

    # Quality thresholds
    min_quality_score: float = 0.3
    quality_weights: dict[str, float] = field(default_factory=lambda: {
        "content_length": 0.2,
        "paragraph_density": 0.25,
        "link_ratio": 0.15,
        "structural_quality": 0.25,
        "language_quality": 0.15
    })

    # Extraction options
    extract_links: bool = True
    extract_images: bool = True
    extract_headings: bool = True
    extract_metadata: bool = True
    preserve_formatting: bool = False

    # Text processing
    clean_whitespace: bool = True
    remove_scripts: bool = True
    remove_styles: bool = True
    remove_comments: bool = True

    # Language detection
    detect_language: bool = True
    supported_languages: set[str] = field(default_factory=lambda: {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'
    })


class ContentExtractor:
    """
    Advanced content extraction with multiple parsing strategies and quality assessment.
    """

    def __init__(self, config: ExtractionConfig | None = None):
        """Initialize content extractor with configuration."""
        self.config = config or ExtractionConfig()

        # Common selectors for content extraction
        self._content_selectors = [
            'article', '[role="main"]', 'main', '.content', '.post-content',
            '.entry-content', '.article-content', '.story-body', '.post-body',
            '#content', '#main-content', '.main-content'
        ]

        # Selectors for elements to remove
        self._noise_selectors = [
            'script', 'style', 'nav', 'header', 'footer', '.sidebar',
            '.advertisement', '.ads', '.comments', '.social-share',
            '.related-articles', '.newsletter-signup', '[role="banner"]',
            '[role="complementary"]', '[role="contentinfo"]'
        ]

        # Metadata selectors
        self._metadata_selectors = {
            'title': ['title', 'h1', '.title', '.headline', '[property="og:title"]'],
            'description': ['meta[name="description"]', '[property="og:description"]'],
            'keywords': ['meta[name="keywords"]'],
            'author': ['meta[name="author"]', '.author', '.byline', '[rel="author"]'],
            'publish_date': [
                'meta[property="article:published_time"]',
                'time[datetime]', '.date', '.published'
            ]
        }

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean HTML by removing unwanted elements."""
        # Remove comments
        if self.config.remove_comments:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

        # Remove scripts and styles
        if self.config.remove_scripts:
            for tag in soup(['script', 'noscript']):
                tag.extract()

        if self.config.remove_styles:
            for tag in soup(['style', 'link[rel="stylesheet"]']):
                tag.extract()

        # Remove noise elements
        for selector in self._noise_selectors:
            for element in soup.select(selector):
                element.extract()

        return soup

    def _extract_text_beautifulsoup(self, soup: BeautifulSoup) -> str:
        """Extract text using BeautifulSoup strategy."""
        # Try content selectors first
        for selector in self._content_selectors:
            content_elements = soup.select(selector)
            if content_elements:
                # Use the largest content element
                largest = max(content_elements, key=lambda x: len(x.get_text()))
                return self._clean_text(largest.get_text())

        # Fallback to body content
        body = soup.find('body')
        if body:
            return self._clean_text(body.get_text())

        # Ultimate fallback
        return self._clean_text(soup.get_text())

    def _extract_text_readability(self, soup: BeautifulSoup) -> str:
        """Extract text using readability-inspired strategy."""
        # Score paragraphs based on content quality
        paragraphs = soup.find_all(['p', 'div'])
        scored_paragraphs = []

        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) < 20:  # Skip very short paragraphs
                continue

            score = 0

            # Length score (favor medium-length paragraphs)
            length = len(text)
            if 50 <= length <= 500:
                score += 2
            elif 20 <= length < 50 or 500 < length <= 1000:
                score += 1

            # Word count score
            words = text.split()
            if 10 <= len(words) <= 100:
                score += 1

            # Punctuation score (proper sentences)
            if text.endswith(('.', '!', '?')):
                score += 1

            # Link density penalty
            links = p.find_all('a')
            link_text_length = sum(len(link.get_text()) for link in links)
            if link_text_length > 0:
                link_density = link_text_length / length
                if link_density > 0.3:
                    score -= 2
                elif link_density > 0.1:
                    score -= 1

            if score > 0:
                scored_paragraphs.append((score, text))

        # Sort by score and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        top_paragraphs = scored_paragraphs[:20]  # Limit to top 20

        return '\n\n'.join(text for _, text in top_paragraphs)

    def _extract_text_custom(self, soup: BeautifulSoup) -> str:
        """Extract text using custom rules-based strategy."""
        content_parts = []

        # Extract title/headline
        title = soup.find(['h1', 'h2']) or soup.find(class_=re.compile(r'title|headline'))
        if title:
            content_parts.append(title.get_text().strip())

        # Extract main content using multiple strategies
        main_content = None

        # Strategy 1: Look for semantic content containers
        for tag_name in ['article', 'main']:
            element = soup.find(tag_name)
            if element:
                main_content = element
                break

        # Strategy 2: Look for content-specific classes
        if not main_content:
            for class_pattern in [r'content', r'post', r'article', r'story']:
                element = soup.find(class_=re.compile(class_pattern))
                if element:
                    main_content = element
                    break

        # Strategy 3: Find the div with most paragraph content
        if not main_content:
            divs = soup.find_all('div')
            if divs:
                div_scores = []
                for div in divs:
                    p_count = len(div.find_all('p'))
                    text_length = len(div.get_text())
                    score = p_count * 10 + text_length * 0.1
                    div_scores.append((score, div))

                if div_scores:
                    div_scores.sort(key=lambda x: x[0], reverse=True)
                    main_content = div_scores[0][1]

        if main_content:
            content_parts.append(self._clean_text(main_content.get_text()))

        return '\n\n'.join(filter(None, content_parts))

    def _extract_text_hybrid(self, soup: BeautifulSoup) -> str:
        """Extract text using hybrid strategy combining multiple approaches."""
        results = {}

        # Try all strategies
        results['beautifulsoup'] = self._extract_text_beautifulsoup(soup)
        results['readability'] = self._extract_text_readability(soup)
        results['custom'] = self._extract_text_custom(soup)

        # Score each result
        scored_results = []
        for strategy, content in results.items():
            if content:
                score = self._calculate_content_quality_score(content, soup)
                scored_results.append((score, strategy, content))

        if scored_results:
            # Return the highest-scoring result
            scored_results.sort(key=lambda x: x[0], reverse=True)
            return scored_results[0][2]

        # Fallback to simple text extraction
        return self._clean_text(soup.get_text())

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Clean whitespace
        if self.config.clean_whitespace:
            # Replace multiple whitespace with single space
            text = re.sub(r'\s+', ' ', text)

            # Clean up line breaks
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

            # Strip leading/trailing whitespace
            text = text.strip()

        return text

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}

        if not self.config.extract_metadata:
            return metadata

        # Title extraction
        title = None
        for selector in self._metadata_selectors['title']:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    title = element.get('content')
                else:
                    title = element.get_text().strip()
                if title:
                    break
        metadata['title'] = title

        # Description extraction
        description = None
        for selector in self._metadata_selectors['description']:
            element = soup.select_one(selector)
            if element:
                description = element.get('content')
                if description:
                    break
        metadata['description'] = description

        # Keywords extraction
        keywords = []
        keywords_element = soup.select_one('meta[name="keywords"]')
        if keywords_element:
            keywords_text = keywords_element.get('content', '')
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        metadata['keywords'] = keywords

        # Author extraction
        author = None
        for selector in self._metadata_selectors['author']:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    author = element.get('content')
                else:
                    author = element.get_text().strip()
                if author:
                    break
        metadata['author'] = author

        # Publish date extraction
        publish_date = None
        for selector in self._metadata_selectors['publish_date']:
            element = soup.select_one(selector)
            if element:
                date_text = element.get('content') or element.get('datetime') or element.get_text()
                if date_text:
                    try:
                        # Simple date parsing - could be enhanced
                        from dateutil.parser import parse
                        publish_date = parse(date_text.strip())
                        break
                    except Exception:
                        pass
        metadata['publish_date'] = publish_date

        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        """Extract and resolve links from HTML."""
        if not self.config.extract_links:
            return []

        links = []
        for anchor in soup.find_all('a', href=True):
            href = anchor['href'].strip()
            text = anchor.get_text().strip()
            title = anchor.get('title', '')

            # Resolve relative URLs
            try:
                if href.startswith("../"):
                    absolute_url = urljoin(urljoin(base_url, href), "..")
                else:
                    absolute_url = urljoin(base_url, href)
                links.append({
                    'url': absolute_url,
                    'text': text,
                    'title': title,
                    'rel': anchor.get('rel', [])
                })
            except Exception:
                pass

        return links

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        """Extract and resolve images from HTML."""
        if not self.config.extract_images:
            return []

        images = []
        for img in soup.find_all('img', src=True):
            src = img['src'].strip()
            alt = img.get('alt', '')
            title = img.get('title', '')

            # Resolve relative URLs
            try:
                absolute_url = urljoin(base_url, src)
                images.append({
                    'url': absolute_url,
                    'alt': alt,
                    'title': title,
                    'width': img.get('width'),
                    'height': img.get('height')
                })
            except Exception:
                pass

        return images

    def _extract_headings(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """Extract heading structure from HTML."""
        if not self.config.extract_headings:
            return []

        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text().strip()
            if text:
                headings.append({
                    'level': int(heading.name[1]),
                    'text': text,
                    'id': heading.get('id')
                })

        return headings

    def _detect_language(self, text: str) -> str | None:
        """Detect language of extracted text."""
        if not self.config.detect_language or not text:
            return None

        try:
            # Simple language detection based on character patterns
            # This is a basic implementation - could be enhanced with proper language detection library

            # Check for common language patterns
            if re.search(r'[а-яё]', text.lower()):
                return 'ru'
            elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
                return 'ja'
            elif re.search(r'[一-龯]', text):
                return 'zh'
            elif re.search(r'[한-힣]', text):
                return 'ko'
            elif re.search(r'[àáâãäåçèéêëìíîïñòóôõöøùúûüýÿ]', text.lower()):
                # Romance languages
                if 'ñ' in text.lower():
                    return 'es'
                elif 'ç' in text.lower():
                    return 'fr'
                else:
                    return 'fr'  # Default to French for other romance language chars
            else:
                return 'en'  # Default to English
        except Exception:
            return None

    def _calculate_content_quality_score(self, content: str, soup: BeautifulSoup) -> float:
        """Calculate quality score for extracted content."""
        if not content:
            return 0.0

        score = 0.0
        weights = self.config.quality_weights

        # Content length score
        length = len(content)
        if length > 500:
            score += weights.get('content_length', 0.2) * min(1.0, length / 2000)

        # Paragraph density score
        paragraphs = content.split('\n\n')
        paragraph_count = len([p for p in paragraphs if len(p.strip()) > 50])
        if paragraph_count > 0:
            avg_paragraph_length = length / paragraph_count
            if 100 <= avg_paragraph_length <= 800:
                score += weights.get('paragraph_density', 0.25)

        # Link ratio score (lower is better for main content)
        links = soup.find_all('a')
        link_text_length = sum(len(link.get_text()) for link in links)
        if length > 0:
            link_ratio = link_text_length / length
            if link_ratio < 0.1:
                score += weights.get('link_ratio', 0.15)
            elif link_ratio < 0.2:
                score += weights.get('link_ratio', 0.15) * 0.5

        # Structural quality score
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        paragraphs_in_soup = soup.find_all('p')
        if headings and paragraphs_in_soup:
            structure_score = min(1.0, len(headings) / 10 + len(paragraphs_in_soup) / 20)
            score += weights.get('structural_quality', 0.25) * structure_score

        # Language quality score
        words = content.split()
        if len(words) > 10:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 4 <= avg_word_length <= 8:  # Reasonable average word length
                score += weights.get('language_quality', 0.15)

        return min(1.0, score)

    async def extract_content(self, html_content: str, url: str,
                            strategy: str | None = None) -> ContentExtractionResult:
        """
        Extract content from HTML using specified strategy.

        Args:
            html_content: Raw HTML content to extract from
            url: URL of the content for link resolution
            strategy: Extraction strategy to use (default: config.preferred_strategy)

        Returns:
            ContentExtractionResult with extracted content and metadata
        """
        import time
        start_time = time.time()

        result = ContentExtractionResult(url=url)
        extraction_strategy = strategy or self.config.preferred_strategy
        explicit_strategy = strategy is not None

        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            result.raw_html = html_content[:10000]  # Store first 10KB for fallback

            # Clean HTML
            cleaned_soup = self._clean_html(soup)

            # Extract content based on strategy
            strategies = {
                'beautifulsoup': self._extract_text_beautifulsoup,
                'readability': self._extract_text_readability,
                'custom': self._extract_text_custom,
                'hybrid': self._extract_text_hybrid
            }

            if extraction_strategy in strategies:
                result.main_content = strategies[extraction_strategy](cleaned_soup)
                result.extraction_strategy = extraction_strategy
            else:
                # Fallback to default strategy
                result.main_content = self._extract_text_beautifulsoup(cleaned_soup)
                result.extraction_strategy = 'beautifulsoup'
                result.warnings.append(f"Unknown strategy '{extraction_strategy}', used beautifulsoup")

            # If extraction failed, try fallback strategies
            if not result.main_content or len(result.main_content) < self.config.min_content_length:
                for fallback_strategy in self.config.fallback_strategies:
                    if fallback_strategy in strategies and fallback_strategy != extraction_strategy:
                        fallback_content = strategies[fallback_strategy](cleaned_soup)
                        if fallback_content and len(fallback_content) >= self.config.min_content_length:
                            result.main_content = fallback_content
                            result.extraction_strategy = f"{extraction_strategy} -> {fallback_strategy}"
                            break
                if not result.main_content:
                    result.main_content = soup.get_text()
                    result.extraction_strategy = "fallback"
                elif not explicit_strategy and len(result.main_content) < self.config.min_content_length:
                    result.main_content = soup.get_text()
                    result.extraction_strategy = "fallback"

            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            result.title = metadata.get('title')
            result.description = metadata.get('description')
            result.keywords = metadata.get('keywords', [])
            result.author = metadata.get('author')
            result.publish_date = metadata.get('publish_date')

            # Extract additional elements
            result.links = self._extract_links(soup, url)
            result.images = self._extract_images(soup, url)
            result.headings = self._extract_headings(soup)

            # Calculate content metrics
            if result.main_content:
                result.content_length = len(result.main_content)
                result.word_count = len(result.main_content.split())
                result.paragraph_count = len([p for p in result.main_content.split('\n\n') if p.strip()])
                result.language = self._detect_language(result.main_content)

            # Calculate quality score
            result.quality_score = self._calculate_content_quality_score(result.main_content or "", soup)

            # Store raw text for fallback
            result.raw_text = soup.get_text()[:10000]

            # Additional metadata
            result.metadata = {
                'links_count': len(result.links),
                'images_count': len(result.images),
                'headings_count': len(result.headings),
                'html_size': len(html_content),
                **metadata
            }

        except Exception as e:
            error_msg = f"Content extraction failed: {str(e)}"
            result.errors.append(error_msg)

            # Fallback to simple text extraction
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                result.main_content = soup.get_text()
                result.extraction_strategy = "fallback"
            except Exception:
                result.main_content = ""
                result.errors.append("Fallback extraction also failed")

        finally:
            result.processing_time = time.time() - start_time

        return result

    def validate_content(self, result: ContentExtractionResult) -> bool:
        """
        Validate extracted content against quality thresholds.

        Args:
            result: ContentExtractionResult to validate

        Returns:
            True if content meets quality standards, False otherwise
        """
        if not result.main_content:
            return False

        # Check minimum content length
        if result.content_length < self.config.min_content_length:
            return False

        # Check maximum content length
        if result.content_length > self.config.max_content_length:
            return False

        # Check minimum word count
        if result.word_count < self.config.min_word_count:
            return False

        # Check quality score
        if result.quality_score < self.config.min_quality_score:
            return False

        return True

    def get_content_summary(self, result: ContentExtractionResult) -> dict[str, Any]:
        """
        Generate summary of extracted content.

        Args:
            result: ContentExtractionResult to summarize

        Returns:
            Dictionary with content summary information
        """
        return {
            'url': result.url,
            'title': result.title,
            'content_length': result.content_length,
            'word_count': result.word_count,
            'paragraph_count': result.paragraph_count,
            'quality_score': result.quality_score,
            'language': result.language,
            'extraction_strategy': result.extraction_strategy,
            'links_count': len(result.links),
            'images_count': len(result.images),
            'headings_count': len(result.headings),
            'processing_time': result.processing_time,
            'has_errors': len(result.errors) > 0,
            'has_warnings': len(result.warnings) > 0
        }
