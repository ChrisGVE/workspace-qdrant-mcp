"""Web interface for memory curation and external content crawling."""

# Import web crawling modules directly to avoid server dependencies in tests
from . import cache, crawler, extractor, links, integration

def create_web_app():
    """Lazy import to avoid circular dependencies."""
    from .server import create_web_app as _create_web_app
    return _create_web_app()

# Export main classes for easy access
from .integration import IntegratedWebCrawler, IntegratedCrawlConfig, crawl_url_simple, crawl_site_recursive
from .cache import ContentCache, CacheConfig
from .crawler import WebCrawler, CrawlConfig
from .extractor import ContentExtractor
from .links import LinkDiscovery, RecursiveCrawler

__all__ = [
    "create_web_app",
    # Modules
    "cache", "crawler", "extractor", "links", "integration",
    # Main classes
    "IntegratedWebCrawler", "IntegratedCrawlConfig",
    "ContentCache", "CacheConfig",
    "WebCrawler", "CrawlConfig",
    "ContentExtractor",
    "LinkDiscovery", "RecursiveCrawler",
    # Convenience functions
    "crawl_url_simple", "crawl_site_recursive"
]
