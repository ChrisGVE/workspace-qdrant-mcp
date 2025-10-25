from loguru import logger

"""
Web content parser that integrates secure crawling with document parsing.

This module provides a document parser interface for web content,
integrating the secure web crawler with the existing document processing pipeline.
"""

import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base import DocumentParser, ParsedDocument
from .exceptions import ParsingError
from .web_crawler import CrawlResult, SecureWebCrawler, SecurityConfig

# logger imported from loguru


class WebParser(DocumentParser):
    """
    Parser for web content using secure crawling.

    This parser fetches web content securely and processes it using
    the existing HTML parser infrastructure with comprehensive security
    hardening and malware protection.
    """

    def __init__(self, security_config: SecurityConfig | None = None):
        """Initialize web parser with security configuration."""
        self.security_config = security_config or SecurityConfig()
        self._crawler: SecureWebCrawler | None = None

    @property
    def supported_extensions(self) -> list[str]:
        """Supported URL patterns (not file extensions)."""
        return [".html", ".htm"]  # For compatibility, but we handle URLs

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "Web Content"

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if input is a web URL."""
        try:
            url_str = str(file_path)
            parsed = urlparse(url_str)
            return parsed.scheme in {"http", "https"}
        except Exception:
            return False

    async def parse(
        self, file_path: str | Path, **options: Any
    ) -> ParsedDocument:
        """
        Parse web content from URL.

        Args:
            file_path: URL to crawl (treated as file_path for interface compatibility)
            **options: Parsing options
                - crawl_depth: int = 0 - Crawl depth (0 = single page)
                - max_pages: int = 1 - Maximum pages to crawl
                - same_domain_only: bool = True - Stay within same domain
                - domain_allowlist: List[str] = None - Allowed domains
                - max_content_size: int = 50MB - Maximum content size per page
                - request_delay: float = 1.0 - Delay between requests
                - respect_robots_txt: bool = True - Respect robots.txt
                - enable_security_scan: bool = True - Enable malware scanning
                - quarantine_suspicious: bool = True - Quarantine suspicious content

        Returns:
            ParsedDocument with web content

        Raises:
            ParsingError: If crawling or parsing fails
        """
        url = str(file_path)

        if not self.can_parse(url):
            raise ParsingError(f"Invalid URL format: {url}")

        try:
            # Configure security settings from options
            config = self._create_config_from_options(**options)

            # Initialize crawler
            async with SecureWebCrawler(config) as crawler:
                # Determine crawling strategy
                crawl_depth = options.get("crawl_depth", 0)
                max_pages = options.get("max_pages", 1)

                if crawl_depth > 0 or max_pages > 1:
                    # Recursive crawling
                    results = await crawler.crawl_recursive(
                        url,
                        max_depth=crawl_depth,
                        max_pages=max_pages,
                        same_domain_only=options.get("same_domain_only", True),
                    )
                else:
                    # Single page crawling
                    result = await crawler.crawl_url(url)
                    results = [result]

                # Process results
                return await self._process_crawl_results(url, results, **options)

        except Exception as e:
            logger.error(f"Web parsing failed for {url}: {e}")
            raise ParsingError(f"Failed to parse web content: {e}") from e

    def _create_config_from_options(self, **options) -> SecurityConfig:
        """Create security config from parsing options."""
        config = SecurityConfig()

        # Apply options
        if "domain_allowlist" in options and options["domain_allowlist"]:
            config.domain_allowlist = set(options["domain_allowlist"])

        if "max_content_size" in options:
            config.max_content_size = options["max_content_size"]

        if "request_delay" in options:
            config.request_delay = options["request_delay"]

        if "respect_robots_txt" in options:
            config.respect_robots_txt = options["respect_robots_txt"]

        if "enable_security_scan" in options:
            config.enable_content_scanning = options["enable_security_scan"]

        if "quarantine_suspicious" in options:
            config.quarantine_suspicious = options["quarantine_suspicious"]

        # Override with instance config if provided
        if self.security_config:
            if self.security_config.domain_allowlist:
                config.domain_allowlist = self.security_config.domain_allowlist
            if self.security_config.domain_blocklist:
                config.domain_blocklist = self.security_config.domain_blocklist

        return config

    async def _process_crawl_results(
        self, original_url: str, results: list[CrawlResult], **options
    ) -> ParsedDocument:
        """Process crawl results into a ParsedDocument."""

        if not results:
            raise ParsingError("No content retrieved from web crawling")

        # Collect successful results
        successful_results = [r for r in results if r.success and r.content]

        if not successful_results:
            # Collect error messages
            errors = [r.error or "Unknown error" for r in results if r.error]
            security_warnings = []
            for r in results:
                security_warnings.extend(r.security_warnings)

            error_msg = f"All crawl attempts failed. Errors: {'; '.join(errors[:3])}"
            if security_warnings:
                error_msg += f" Security warnings: {'; '.join(security_warnings[:2])}"

            raise ParsingError(error_msg)

        # Combine content from successful results
        combined_content = []
        combined_metadata = {
            "source_url": original_url,
            "pages_crawled": len(successful_results),
            "total_attempts": len(results),
            "crawl_timestamp": max(r.timestamp for r in successful_results),
            "security_warnings": [],
            "page_metadata": [],
        }

        for i, result in enumerate(successful_results):
            # Add content with page separator
            if i > 0:
                combined_content.append(f"\n\n--- Page {i + 1}: {result.url} ---\n")

            # Use parsed content if available, otherwise raw content
            if "parsed_content" in result.metadata:
                combined_content.append(result.metadata["parsed_content"])
            else:
                combined_content.append(result.content or "")

            # Collect metadata
            page_meta = {
                "url": result.url,
                "status_code": result.status_code,
                "content_type": result.content_type,
                "timestamp": result.timestamp,
            }

            # Add HTML metadata if available
            if "html_metadata" in result.metadata:
                page_meta["html_metadata"] = result.metadata["html_metadata"]

            combined_metadata["page_metadata"].append(page_meta)

            # Collect security warnings
            if result.security_warnings:
                combined_metadata["security_warnings"].extend(result.security_warnings)

        # Join all content
        full_content = "".join(combined_content).strip()

        # Parsing information
        parsing_info = {
            "parser": "WebParser",
            "pages_processed": len(successful_results),
            "total_content_length": len(full_content),
            "security_scan_enabled": self.security_config.enable_content_scanning,
            "crawl_depth": options.get("crawl_depth", 0),
            "same_domain_only": options.get("same_domain_only", True),
        }

        # Add security summary
        if combined_metadata["security_warnings"]:
            parsing_info["security_warnings_count"] = len(
                combined_metadata["security_warnings"]
            )
            parsing_info["security_status"] = "warnings_found"
        else:
            parsing_info["security_status"] = "clean"

        logger.info(
            f"Successfully processed {len(successful_results)} web pages from {original_url}"
        )

        # Create temporary file path for compatibility
        temp_path = Path(tempfile.mktemp(suffix=".html"))

        return ParsedDocument.create(
            content=full_content,
            file_path=temp_path,
            file_type="web",
            additional_metadata=combined_metadata,
            parsing_info=parsing_info,
        )

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for web content."""
        return {
            "crawl_depth": {
                "type": int,
                "default": 0,
                "description": "Maximum crawl depth (0 = single page only)",
            },
            "max_pages": {
                "type": int,
                "default": 1,
                "description": "Maximum number of pages to crawl",
            },
            "same_domain_only": {
                "type": bool,
                "default": True,
                "description": "Only crawl pages from the same domain",
            },
            "domain_allowlist": {
                "type": list,
                "default": None,
                "description": "List of allowed domains (empty = all allowed)",
            },
            "max_content_size": {
                "type": int,
                "default": 52428800,  # 50MB
                "description": "Maximum content size per page in bytes",
            },
            "request_delay": {
                "type": float,
                "default": 1.0,
                "description": "Delay between requests in seconds (respectful crawling)",
            },
            "respect_robots_txt": {
                "type": bool,
                "default": True,
                "description": "Respect robots.txt directives",
            },
            "enable_security_scan": {
                "type": bool,
                "default": True,
                "description": "Enable malware and security scanning of content",
            },
            "quarantine_suspicious": {
                "type": bool,
                "default": True,
                "description": "Quarantine suspicious content instead of processing",
            },
        }


class WebIngestionInterface:
    """
    High-level interface for web content ingestion.

    Provides convenient methods for common web ingestion scenarios
    with built-in security best practices.
    """

    def __init__(self, security_config: SecurityConfig | None = None):
        self.parser = WebParser(security_config)

    async def ingest_url(self, url: str, **options) -> ParsedDocument:
        """Ingest content from a single URL.

        Args:
            url: URL to ingest
            **options: Parsing options (see WebParser.get_parsing_options())

        Returns:
            ParsedDocument with web content
        """
        return await self.parser.parse(url, **options)

    async def ingest_site(
        self, start_url: str, max_pages: int = 10, max_depth: int = 2, **options
    ) -> ParsedDocument:
        """Ingest content from multiple pages of a website.

        Args:
            start_url: Starting URL for crawling
            max_pages: Maximum pages to crawl
            max_depth: Maximum crawl depth
            **options: Additional parsing options

        Returns:
            ParsedDocument with combined content from all pages
        """
        return await self.parser.parse(
            start_url, crawl_depth=max_depth, max_pages=max_pages, **options
        )

    async def ingest_with_allowlist(
        self, url: str, allowed_domains: list[str], **options
    ) -> ParsedDocument:
        """Ingest content with strict domain restrictions.

        Args:
            url: URL to ingest
            allowed_domains: List of allowed domains
            **options: Additional parsing options

        Returns:
            ParsedDocument with web content
        """
        return await self.parser.parse(url, domain_allowlist=allowed_domains, **options)


def create_secure_web_parser(
    allowed_domains: list[str] | None = None,
    enable_scanning: bool = True,
    quarantine_threats: bool = True,
) -> WebParser:
    """Create a web parser with security-focused defaults.

    Args:
        allowed_domains: List of domains to allow (None = all)
        enable_scanning: Enable malware scanning
        quarantine_threats: Quarantine suspicious content

    Returns:
        Configured WebParser instance
    """
    config = SecurityConfig()

    if allowed_domains:
        config.domain_allowlist = set(allowed_domains)

    config.enable_content_scanning = enable_scanning
    config.quarantine_suspicious = quarantine_threats

    # Security-focused defaults
    config.max_content_size = 10 * 1024 * 1024  # 10MB limit
    config.max_total_pages = 50  # Conservative limit
    config.max_depth = 2  # Shallow crawling
    config.request_delay = 1.5  # Respectful crawling

    return WebParser(config)
