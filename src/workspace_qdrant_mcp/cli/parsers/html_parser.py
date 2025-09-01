"""
HTML and web content parser for extracting text content and metadata.

This module provides functionality to parse HTML files and web content,
extracting clean text content and metadata from HTML head section while
removing scripts, styles, navigation, and other non-content elements.
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

try:
    from bs4 import BeautifulSoup, Comment
    import chardet
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from .base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class HtmlParser(DocumentParser):
    """
    Parser for HTML files and web content.
    
    Extracts clean text content from HTML documents while preserving
    metadata from the HTML head section. Removes scripts, styles,
    navigation, ads, and other non-content elements to focus on
    readable content.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """HTML file extensions."""
        return ['.html', '.htm', '.xhtml']

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return 'HTML Web Content'

    def _check_availability(self) -> None:
        """Check if required libraries are available."""
        if not BS4_AVAILABLE:
            raise RuntimeError(
                "HTML parsing requires 'beautifulsoup4' and 'chardet'. "
                "Install with: pip install beautifulsoup4 chardet"
            )

    async def parse(self, file_path: str | Path, **options: Any) -> ParsedDocument:
        """
        Parse HTML file and extract text content.

        Args:
            file_path: Path to HTML file
            **options: Parsing options
                - remove_navigation: bool = True - Remove nav elements and menus
                - remove_ads: bool = True - Remove advertisement content
                - remove_scripts_styles: bool = True - Remove script and style tags
                - preserve_links: bool = False - Preserve link text with URLs
                - preserve_headings: bool = True - Preserve heading structure
                - extract_metadata: bool = True - Extract HTML head metadata
                - encoding: str = None - Force specific encoding (auto-detect if None)
                - content_selectors: list[str] = None - CSS selectors for main content
                - remove_selectors: list[str] = None - CSS selectors for elements to remove

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            RuntimeError: If required libraries are not installed
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        self._check_availability()
        self.validate_file(file_path)
        
        file_path = Path(file_path)
        
        try:
            # Parse options
            remove_navigation = options.get('remove_navigation', True)
            remove_ads = options.get('remove_ads', True)
            remove_scripts_styles = options.get('remove_scripts_styles', True)
            preserve_links = options.get('preserve_links', False)
            preserve_headings = options.get('preserve_headings', True)
            extract_metadata = options.get('extract_metadata', True)
            encoding = options.get('encoding', None)
            content_selectors = options.get('content_selectors', None)
            remove_selectors = options.get('remove_selectors', None)
            
            # Read and decode HTML file
            html_content = await self._read_html_file(file_path, encoding)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract metadata if requested
            metadata = {}
            if extract_metadata:
                metadata = await self._extract_metadata(soup)
            
            # Clean and extract text content
            text_content = await self._extract_text_content(
                soup,
                remove_navigation=remove_navigation,
                remove_ads=remove_ads,
                remove_scripts_styles=remove_scripts_styles,
                preserve_links=preserve_links,
                preserve_headings=preserve_headings,
                content_selectors=content_selectors,
                remove_selectors=remove_selectors
            )
            
            # Parsing information
            parsing_info = {
                "original_length": len(html_content),
                "content_length": len(text_content),
                "encoding_detected": self._detect_encoding(html_content),
                "encoding_used": encoding or self._detect_encoding(html_content),
                "remove_navigation": remove_navigation,
                "remove_ads": remove_ads,
                "preserve_links": preserve_links,
                "preserve_headings": preserve_headings
            }
            
            logger.info(f"Successfully parsed HTML: {file_path.name} "
                       f"({parsing_info['original_length']:,} -> "
                       f"{parsing_info['content_length']:,} chars)")
            
            return ParsedDocument.create(
                content=text_content,
                file_path=file_path,
                file_type='html',
                additional_metadata=metadata,
                parsing_info=parsing_info
            )
            
        except Exception as e:
            logger.error(f"Failed to parse HTML {file_path}: {e}")
            raise RuntimeError(f"HTML parsing failed: {e}") from e

    async def _read_html_file(self, file_path: Path, encoding: str | None = None) -> str:
        """Read HTML file with proper encoding detection."""
        try:
            # Read raw bytes first
            raw_content = file_path.read_bytes()
            
            if encoding:
                # Use specified encoding
                return raw_content.decode(encoding, errors='replace')
            
            # Auto-detect encoding
            detected_encoding = self._detect_encoding(raw_content)
            
            try:
                return raw_content.decode(detected_encoding, errors='replace')
            except (UnicodeDecodeError, LookupError):
                # Fallback to UTF-8 with error replacement
                logger.warning(f"Failed to decode with {detected_encoding}, using UTF-8 fallback")
                return raw_content.decode('utf-8', errors='replace')
                
        except Exception as e:
            logger.error(f"Failed to read HTML file {file_path}: {e}")
            raise RuntimeError(f"Could not read HTML file: {e}") from e

    def _detect_encoding(self, content: bytes) -> str:
        """Detect encoding of HTML content."""
        try:
            # Try chardet detection
            detected = chardet.detect(content)
            if detected and detected.get('encoding'):
                confidence = detected.get('confidence', 0)
                if confidence > 0.7:  # High confidence threshold
                    return detected['encoding']
            
            # Look for HTML meta charset
            content_str = content[:2048].decode('utf-8', errors='ignore').lower()
            
            # Look for HTML5 charset declaration
            charset_match = re.search(r'<meta\s+charset\s*=\s*["\']?([^"\'\s>]+)', content_str)
            if charset_match:
                return charset_match.group(1)
            
            # Look for older HTTP-EQUIV charset
            http_equiv_match = re.search(
                r'<meta\s+http-equiv\s*=\s*["\']?content-type["\']?\s+'
                r'content\s*=\s*["\']?[^"\']*charset\s*=\s*([^"\'\s;>]+)',
                content_str
            )
            if http_equiv_match:
                return http_equiv_match.group(1)
                
        except Exception:
            pass
        
        # Default fallback
        return 'utf-8'

    async def _extract_metadata(self, soup: "BeautifulSoup") -> dict[str, str | int | float | bool]:
        """Extract metadata from HTML head section."""
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            metadata['title'] = title_tag.get_text().strip()
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            content = meta.get('content', '').strip()
            
            if not content:
                continue
                
            if name in ['description', 'keywords', 'author']:
                metadata[name] = content
            elif name == 'generator':
                metadata['generator'] = content
            elif name == 'viewport':
                metadata['viewport'] = content
            elif name in ['robots', 'googlebot']:
                metadata[f'{name}_directive'] = content
            
            # Open Graph metadata
            property_attr = meta.get('property', '').lower()
            if property_attr.startswith('og:'):
                og_key = property_attr[3:]  # Remove 'og:' prefix
                if og_key in ['title', 'description', 'type', 'url', 'image', 'site_name']:
                    metadata[f'og_{og_key}'] = content
            
            # Twitter Card metadata
            elif name.startswith('twitter:'):
                twitter_key = name[8:]  # Remove 'twitter:' prefix
                if twitter_key in ['card', 'title', 'description', 'image', 'site']:
                    metadata[f'twitter_{twitter_key}'] = content
        
        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        # Count structural elements
        metadata['heading_count'] = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        metadata['paragraph_count'] = len(soup.find_all('p'))
        metadata['link_count'] = len(soup.find_all('a', href=True))
        metadata['image_count'] = len(soup.find_all('img'))
        metadata['list_count'] = len(soup.find_all(['ul', 'ol']))
        metadata['table_count'] = len(soup.find_all('table'))
        
        # Check for specific content types
        metadata['has_forms'] = len(soup.find_all('form')) > 0
        metadata['has_videos'] = len(soup.find_all('video')) > 0
        metadata['has_audio'] = len(soup.find_all('audio')) > 0
        metadata['has_iframes'] = len(soup.find_all('iframe')) > 0
        
        return metadata

    async def _extract_text_content(
        self,
        soup: "BeautifulSoup",
        remove_navigation: bool = True,
        remove_ads: bool = True,
        remove_scripts_styles: bool = True,
        preserve_links: bool = False,
        preserve_headings: bool = True,
        content_selectors: list[str] | None = None,
        remove_selectors: list[str] | None = None
    ) -> str:
        """Extract clean text content from HTML."""
        
        # Make a copy to avoid modifying the original
        content_soup = BeautifulSoup(str(soup), 'lxml')
        
        # Remove comments
        for comment in content_soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove scripts, styles, and other non-content tags
        if remove_scripts_styles:
            for tag in content_soup(['script', 'style', 'noscript', 'meta', 'link']):
                tag.extract()
        
        # Remove navigation elements
        if remove_navigation:
            nav_selectors = [
                'nav', 'navigation', '[role="navigation"]',
                '.nav', '.navigation', '.navbar', '.menu', '.sidebar',
                '#nav', '#navigation', '#navbar', '#menu', '#sidebar',
                'header nav', 'footer nav', '.breadcrumb', '.breadcrumbs'
            ]
            for selector in nav_selectors:
                for element in content_soup.select(selector):
                    element.extract()
        
        # Remove advertisement and promotional content
        if remove_ads:
            ad_selectors = [
                '.ad', '.ads', '.advertisement', '.promo', '.promotion',
                '#ad', '#ads', '#advertisement', '#promo', '#promotion',
                '[class*="ad-"]', '[id*="ad-"]', '[class*="ads-"]',
                '.sponsored', '.banner', '.popup', '.modal',
                '[class*="sponsor"]', '[id*="sponsor"]'
            ]
            for selector in ad_selectors:
                for element in content_soup.select(selector):
                    element.extract()
        
        # Remove additional unwanted elements
        unwanted_selectors = [
            'footer', 'aside', '.footer', '.aside',
            '.social', '.share', '.sharing', '.comments',
            '.cookie', '.gdpr', '.privacy-notice',
            '.related', '.recommended', '.suggested'
        ]
        for selector in unwanted_selectors:
            for element in content_soup.select(selector):
                element.extract()
        
        # Apply custom remove selectors
        if remove_selectors:
            for selector in remove_selectors:
                for element in content_soup.select(selector):
                    element.extract()
        
        # Focus on main content if selectors provided
        if content_selectors:
            main_content_elements = []
            for selector in content_selectors:
                elements = content_soup.select(selector)
                main_content_elements.extend(elements)
            
            if main_content_elements:
                # Create new soup with only selected content
                new_soup = BeautifulSoup('<div></div>', 'lxml')
                container = new_soup.find('div')
                for element in main_content_elements:
                    container.append(element)
                content_soup = new_soup
        else:
            # Try to find main content automatically
            main_content = (
                content_soup.find('main') or
                content_soup.find('article') or
                content_soup.find('[role="main"]') or
                content_soup.find('#main') or
                content_soup.find('#content') or
                content_soup.find('.main') or
                content_soup.find('.content')
            )
            if main_content:
                # Create new soup with just the main content
                new_soup = BeautifulSoup('<div></div>', 'lxml')
                container = new_soup.find('div')
                container.append(main_content)
                content_soup = new_soup
        
        # Process links if preservation is requested
        if preserve_links:
            for link in content_soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text().strip()
                if text and href:
                    link.string = f"{text} ({href})"
        
        # Extract text content
        text_parts = []
        
        if preserve_headings:
            # Process elements in document order, preserving structure
            for element in content_soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section', 'li']):
                text = element.get_text().strip()
                if not text:
                    continue
                
                # Add heading markers
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    level = int(element.name[1])
                    text = f"{'#' * level} {text}"
                
                text_parts.append(text)
        else:
            # Simple text extraction
            text = content_soup.get_text()
            text_parts = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Join and clean up text
        full_text = '\n\n'.join(text_parts)
        
        # Clean up excessive whitespace
        full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)  # Multiple newlines to double
        full_text = re.sub(r'[ \t]+', ' ', full_text)  # Multiple spaces to single
        
        return full_text.strip()

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for HTML files."""
        return {
            'remove_navigation': {
                'type': bool,
                'default': True,
                'description': 'Remove navigation elements and menus'
            },
            'remove_ads': {
                'type': bool,
                'default': True,
                'description': 'Remove advertisement and promotional content'
            },
            'remove_scripts_styles': {
                'type': bool,
                'default': True,
                'description': 'Remove script and style tags'
            },
            'preserve_links': {
                'type': bool,
                'default': False,
                'description': 'Preserve link text with URLs in parentheses'
            },
            'preserve_headings': {
                'type': bool,
                'default': True,
                'description': 'Preserve heading structure with markdown-style markers'
            },
            'extract_metadata': {
                'type': bool,
                'default': True,
                'description': 'Extract metadata from HTML head section'
            },
            'encoding': {
                'type': str,
                'default': None,
                'description': 'Force specific encoding (auto-detect if None)'
            },
            'content_selectors': {
                'type': list,
                'default': None,
                'description': 'CSS selectors for main content (e.g., ["main", "article"])'
            },
            'remove_selectors': {
                'type': list,
                'default': None,
                'description': 'CSS selectors for elements to remove (e.g., [".ads", "#popup"])'
            }
        }