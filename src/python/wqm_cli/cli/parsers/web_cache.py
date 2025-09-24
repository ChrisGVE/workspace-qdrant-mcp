"""
Web content caching system with TTL, ETag, and content fingerprinting.

This module provides comprehensive caching capabilities for web crawling,
including response caching, content deduplication, and HTTP cache validation.
"""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse, urlunparse

import aiofiles
from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    url: str
    content: str
    headers: Dict[str, str]
    status_code: int
    timestamp: float
    ttl: float  # Time to live in seconds
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    content_hash: Optional[str] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() > (self.timestamp + self.ttl)

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()

    def can_revalidate(self) -> bool:
        """Check if entry can be revalidated with server."""
        return self.etag is not None or self.last_modified is not None


@dataclass
class CacheConfig:
    """Configuration for web cache."""
    # Cache size limits
    max_entries: int = 10000
    max_memory_mb: int = 500  # Maximum memory usage in MB
    max_content_size: int = 10 * 1024 * 1024  # 10MB per entry

    # Default TTL values (in seconds)
    default_ttl: float = 3600.0  # 1 hour
    min_ttl: float = 300.0  # 5 minutes
    max_ttl: float = 86400.0  # 24 hours

    # Cache persistence
    enable_disk_cache: bool = True
    cache_directory: Optional[Path] = None
    cache_filename: str = "web_cache.pkl"

    # Content fingerprinting
    enable_content_fingerprinting: bool = True
    fingerprint_algorithm: str = "sha256"

    # Cleanup settings
    cleanup_interval: float = 3600.0  # Clean up every hour
    max_age_for_cleanup: float = 7 * 24 * 3600  # 7 days


class ContentFingerprinter:
    """Generate content fingerprints for deduplication."""

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm

    def generate_fingerprint(self, content: str) -> str:
        """Generate content fingerprint."""
        hasher = hashlib.new(self.algorithm)
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()

    def normalize_content_for_fingerprinting(self, content: str) -> str:
        """Normalize content before fingerprinting."""
        # Remove whitespace variations and normalize
        lines = [line.strip() for line in content.split('\n')]
        normalized = '\n'.join(line for line in lines if line)
        return normalized.lower()


class WebCache:
    """Web content cache with TTL, ETag support, and content fingerprinting."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.fingerprinter = ContentFingerprinter(self.config.fingerprint_algorithm)

        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._content_fingerprints: Dict[str, Set[str]] = {}  # fingerprint -> urls
        self._last_cleanup = time.time()

        # Initialize cache directory
        if self.config.enable_disk_cache:
            if self.config.cache_directory is None:
                self.config.cache_directory = Path.home() / ".workspace-qdrant" / "cache"
            self.config.cache_directory.mkdir(parents=True, exist_ok=True)

        # Load from disk if available
        if self.config.enable_disk_cache:
            self._load_from_disk()

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent caching."""
        parsed = urlparse(url)
        # Remove fragment and normalize query parameters
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized

    def _calculate_ttl_from_headers(self, headers: Dict[str, str]) -> float:
        """Calculate TTL from HTTP headers."""
        # Check Cache-Control header
        cache_control = headers.get('cache-control', '').lower()
        if 'no-cache' in cache_control or 'no-store' in cache_control:
            return 0.0

        # Extract max-age from Cache-Control
        if 'max-age=' in cache_control:
            try:
                max_age_part = [part for part in cache_control.split(',') if 'max-age=' in part][0]
                max_age = int(max_age_part.split('=')[1].strip())
                return min(max(max_age, self.config.min_ttl), self.config.max_ttl)
            except (ValueError, IndexError):
                pass

        # Check Expires header
        expires = headers.get('expires')
        if expires:
            try:
                # Simple parsing - would need proper HTTP date parsing in production
                # For now, use default TTL
                pass
            except Exception:
                pass

        return self.config.default_ttl

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        return time.time() - self._last_cleanup > self.config.cleanup_interval

    async def _cleanup_cache(self) -> None:
        """Clean up expired and old entries."""
        current_time = time.time()
        urls_to_remove = []

        # Find expired entries
        for url, entry in self._cache.items():
            if (entry.is_expired() or
                current_time - entry.timestamp > self.config.max_age_for_cleanup):
                urls_to_remove.append(url)

        # Remove expired entries
        for url in urls_to_remove:
            self._remove_entry(url)

        # If still over limits, remove least recently used entries
        if len(self._cache) > self.config.max_entries:
            # Sort by last access time (oldest first)
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_access)
            excess_count = len(self._cache) - self.config.max_entries

            for url, _ in sorted_entries[:excess_count]:
                self._remove_entry(url)

        self._last_cleanup = current_time

        if urls_to_remove:
            logger.info(f"Cache cleanup: removed {len(urls_to_remove)} expired entries")

    def _remove_entry(self, url: str) -> None:
        """Remove cache entry and update fingerprint index."""
        if url in self._cache:
            entry = self._cache[url]

            # Remove from fingerprint index
            if entry.content_hash:
                if entry.content_hash in self._content_fingerprints:
                    self._content_fingerprints[entry.content_hash].discard(url)
                    if not self._content_fingerprints[entry.content_hash]:
                        del self._content_fingerprints[entry.content_hash]

            del self._cache[url]

    def _add_to_fingerprint_index(self, url: str, content_hash: str) -> None:
        """Add URL to content fingerprint index."""
        if content_hash not in self._content_fingerprints:
            self._content_fingerprints[content_hash] = set()
        self._content_fingerprints[content_hash].add(url)

    def get_cache_key(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None) -> str:
        """Generate cache key for request."""
        normalized_url = self._normalize_url(url)

        # Include relevant headers in cache key
        cache_headers = []
        if headers:
            relevant_headers = {'accept', 'accept-encoding', 'accept-language'}
            for header, value in headers.items():
                if header.lower() in relevant_headers:
                    cache_headers.append(f"{header.lower()}:{value}")

        key_parts = [method.upper(), normalized_url] + sorted(cache_headers)
        cache_key = "|".join(key_parts)
        return cache_key

    def get(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None) -> Optional[CacheEntry]:
        """Get cached response."""
        cache_key = self.get_cache_key(url, method, headers)

        if cache_key in self._cache:
            entry = self._cache[cache_key]

            if entry.is_expired():
                self._remove_entry(cache_key)
                return None

            entry.update_access()
            return entry

        return None

    async def put(
        self,
        url: str,
        content: str,
        headers: Dict[str, str],
        status_code: int,
        method: str = "GET",
        request_headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Store response in cache."""
        # Check content size limits
        if len(content) > self.config.max_content_size:
            logger.warning(f"Content too large for cache: {len(content)} bytes > {self.config.max_content_size}")
            return False

        # Calculate TTL
        ttl = self._calculate_ttl_from_headers(headers)
        if ttl <= 0:
            return False  # Not cacheable

        # Generate cache key
        cache_key = self.get_cache_key(url, method, request_headers)

        # Generate content fingerprint
        content_hash = None
        if self.config.enable_content_fingerprinting:
            normalized_content = self.fingerprinter.normalize_content_for_fingerprinting(content)
            content_hash = self.fingerprinter.generate_fingerprint(normalized_content)

        # Create cache entry
        entry = CacheEntry(
            url=url,
            content=content,
            headers=headers.copy(),
            status_code=status_code,
            timestamp=time.time(),
            ttl=ttl,
            etag=headers.get('etag'),
            last_modified=headers.get('last-modified'),
            content_hash=content_hash
        )

        # Store in cache
        self._cache[cache_key] = entry

        # Update fingerprint index
        if content_hash:
            self._add_to_fingerprint_index(cache_key, content_hash)

        # Cleanup if needed
        if self._should_cleanup():
            await self._cleanup_cache()

        # Save to disk if enabled
        if self.config.enable_disk_cache:
            await self._save_to_disk()

        logger.debug(f"Cached response for {url} (TTL: {ttl}s)")
        return True

    def get_conditional_headers(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get conditional headers for cache validation."""
        entry = self.get(url, method, headers)
        if not entry or not entry.can_revalidate():
            return {}

        conditional_headers = {}
        if entry.etag:
            conditional_headers['If-None-Match'] = entry.etag
        if entry.last_modified:
            conditional_headers['If-Modified-Since'] = entry.last_modified

        return conditional_headers

    def handle_304_response(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None) -> Optional[CacheEntry]:
        """Handle 304 Not Modified response by refreshing cache entry."""
        cache_key = self.get_cache_key(url, method, headers)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            # Refresh timestamp and TTL
            entry.timestamp = time.time()
            entry.ttl = self._calculate_ttl_from_headers(entry.headers)
            entry.update_access()
            logger.debug(f"Refreshed cache entry for {url}")
            return entry
        return None

    def find_duplicate_content(self, content: str) -> List[str]:
        """Find URLs with duplicate content."""
        if not self.config.enable_content_fingerprinting:
            return []

        normalized_content = self.fingerprinter.normalize_content_for_fingerprinting(content)
        content_hash = self.fingerprinter.generate_fingerprint(normalized_content)

        if content_hash in self._content_fingerprints:
            return list(self._content_fingerprints[content_hash])

        return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(len(entry.content) for entry in self._cache.values())

        return {
            'total_entries': len(self._cache),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'fingerprint_entries': len(self._content_fingerprints),
            'expired_entries': sum(1 for entry in self._cache.values() if entry.is_expired()),
            'hit_rate': self._calculate_hit_rate(),
            'memory_usage_mb': total_size / (1024 * 1024),
            'config': {
                'max_entries': self.config.max_entries,
                'max_memory_mb': self.config.max_memory_mb,
                'default_ttl': self.config.default_ttl
            }
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self._cache:
            return 0.0

        total_accesses = sum(entry.access_count for entry in self._cache.values())
        if total_accesses == 0:
            return 0.0

        # This is a simplified calculation - in practice, you'd track hits vs misses
        return min(1.0, total_accesses / len(self._cache))

    async def _save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.config.cache_directory:
            return

        cache_file = self.config.cache_directory / self.config.cache_filename

        try:
            # Create a serializable version of the cache
            cache_data = {
                'cache': self._cache,
                'fingerprints': self._content_fingerprints,
                'saved_at': time.time()
            }

            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(pickle.dumps(cache_data))

        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.config.cache_directory:
            return

        cache_file = self.config.cache_directory / self.config.cache_filename

        if not cache_file.exists():
            return

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self._cache = cache_data.get('cache', {})
            self._content_fingerprints = cache_data.get('fingerprints', {})

            # Remove expired entries after loading
            current_time = time.time()
            urls_to_remove = [
                url for url, entry in self._cache.items()
                if entry.is_expired() or current_time - entry.timestamp > self.config.max_age_for_cleanup
            ]

            for url in urls_to_remove:
                self._remove_entry(url)

            logger.info(f"Loaded {len(self._cache)} cache entries from disk")

        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            self._cache = {}
            self._content_fingerprints = {}

    def clear_cache(self, url_pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally matching a pattern."""
        if url_pattern is None:
            # Clear all
            count = len(self._cache)
            self._cache.clear()
            self._content_fingerprints.clear()
        else:
            # Clear matching entries
            urls_to_remove = [url for url in self._cache.keys() if url_pattern in url]
            count = len(urls_to_remove)
            for url in urls_to_remove:
                self._remove_entry(url)

        logger.info(f"Cleared {count} cache entries")
        return count