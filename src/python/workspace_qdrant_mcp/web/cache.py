"""
Content caching system with duplicate detection and cleanup.

This module provides comprehensive content caching with:
- Fast duplicate detection using content hashes
- LRU eviction with size and time-based limits
- Persistent storage with compression
- Content similarity detection
- Cache statistics and analytics
- Cleanup and maintenance operations
"""

import asyncio
import gzip
import hashlib
import json
import logging
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached content entry."""
    url: str
    content: str
    content_hash: str
    content_type: str
    timestamp: float
    size: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        if self.size <= 0:
            self.size = len(self.content.encode('utf-8'))


@dataclass
class CacheStats:
    """Cache statistics and analytics."""
    total_entries: int = 0
    total_size: int = 0
    hit_count: int = 0
    miss_count: int = 0
    duplicate_count: int = 0
    eviction_count: int = 0
    cleanup_count: int = 0
    oldest_entry: float = 0.0
    newest_entry: float = 0.0
    average_size: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def size_mb(self) -> float:
        """Get total cache size in MB."""
        return self.total_size / (1024 * 1024)


@dataclass
class CacheConfig:
    """Configuration for content cache."""
    max_size: int = 100 * 1024 * 1024  # 100MB default
    max_entries: int = 10000
    max_age_seconds: int = 24 * 60 * 60  # 24 hours
    enable_compression: bool = True
    enable_persistence: bool = True
    cache_dir: Optional[Path] = None
    similarity_threshold: float = 0.85
    cleanup_interval: int = 3600  # 1 hour
    enable_analytics: bool = True

    def __post_init__(self):
        """Post-initialization processing."""
        if self.cache_dir is None:
            self.cache_dir = Path.cwd() / ".cache" / "web_content"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


class ContentHasher:
    """Fast content hashing for duplicate detection."""

    @staticmethod
    def hash_content(content: str, algorithm: str = "xxhash") -> str:
        """Generate content hash using specified algorithm."""
        try:
            if algorithm == "xxhash" and XXHASH_AVAILABLE:
                return xxhash.xxh64(content.encode('utf-8')).hexdigest()
            elif algorithm == "sha256":
                return hashlib.sha256(content.encode('utf-8')).hexdigest()
            elif algorithm == "md5":
                return hashlib.md5(content.encode('utf-8')).hexdigest()
            else:
                # Fallback to SHA256
                return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Hash generation failed: {e}")
            # Ultimate fallback
            return hashlib.md5(content.encode('utf-8')).hexdigest()

    @staticmethod
    def similarity_hash(content: str, ngram_size: int = 3) -> str:
        """Generate similarity hash for near-duplicate detection."""
        try:
            # Normalize content
            normalized = ' '.join(content.lower().split())
            if len(normalized) < ngram_size:
                return ContentHasher.hash_content(normalized)

            # Generate n-grams
            ngrams = set()
            for i in range(len(normalized) - ngram_size + 1):
                ngrams.add(normalized[i:i + ngram_size])

            # Hash the sorted n-grams
            ngram_str = ''.join(sorted(ngrams))
            return ContentHasher.hash_content(ngram_str)
        except Exception as e:
            logger.warning(f"Similarity hash generation failed: {e}")
            return ContentHasher.hash_content(content)


class DuplicateDetector:
    """Detects duplicate and near-duplicate content."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.content_hashes: Dict[str, List[str]] = defaultdict(list)
        self.similarity_hashes: Dict[str, List[str]] = defaultdict(list)

    def add_content(self, url: str, content: str) -> Tuple[bool, Optional[str]]:
        """
        Add content for duplicate detection.

        Returns:
            Tuple of (is_duplicate, original_url)
        """
        try:
            content_hash = ContentHasher.hash_content(content)

            # Check for exact duplicates
            if content_hash in self.content_hashes:
                original_url = self.content_hashes[content_hash][0]
                logger.debug(f"Exact duplicate found: {url} -> {original_url}")
                return True, original_url

            # Check for near-duplicates
            similarity_hash = ContentHasher.similarity_hash(content)
            if similarity_hash in self.similarity_hashes:
                # Perform detailed similarity check
                for existing_url in self.similarity_hashes[similarity_hash]:
                    if self._calculate_similarity(content, existing_url) > self.similarity_threshold:
                        logger.debug(f"Near-duplicate found: {url} -> {existing_url}")
                        return True, existing_url

            # Not a duplicate, add to tracking
            self.content_hashes[content_hash].append(url)
            self.similarity_hashes[similarity_hash].append(url)
            return False, None

        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            return False, None

    def _calculate_similarity(self, content1: str, url2: str) -> float:
        """Calculate content similarity (simplified implementation)."""
        try:
            # This is a simplified similarity calculation
            # In production, you might use more sophisticated algorithms
            words1 = set(content1.lower().split())
            # For this implementation, we'll use the similarity hash match as high similarity
            return 0.9  # Assume high similarity if similarity hashes match
        except Exception:
            return 0.0

    def remove_content(self, url: str):
        """Remove content from duplicate tracking."""
        try:
            # Remove from all tracking structures
            for hash_dict in [self.content_hashes, self.similarity_hashes]:
                for hash_key, urls in list(hash_dict.items()):
                    if url in urls:
                        urls.remove(url)
                        if not urls:
                            del hash_dict[hash_key]
        except Exception as e:
            logger.error(f"Error removing content from duplicate tracking: {e}")

    def get_duplicates(self) -> Dict[str, List[str]]:
        """Get all duplicate content groups."""
        duplicates = {}
        for content_hash, urls in self.content_hashes.items():
            if len(urls) > 1:
                duplicates[content_hash] = urls
        return duplicates


class ContentCache:
    """High-performance content cache with duplicate detection."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.duplicate_detector = DuplicateDetector(self.config.similarity_threshold)
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        # Initialize cache directory
        if self.config.enable_persistence:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ContentCache initialized with max_size={self.config.max_size}, max_entries={self.config.max_entries}")

    async def get(self, url: str) -> Optional[CacheEntry]:
        """Get content from cache."""
        async with self._lock:
            try:
                if url in self.cache:
                    entry = self.cache[url]
                    # Update access statistics
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    # Move to end (most recently used)
                    self.cache.move_to_end(url)
                    self.stats.hit_count += 1
                    logger.debug(f"Cache hit for URL: {url}")
                    return entry
                else:
                    self.stats.miss_count += 1
                    logger.debug(f"Cache miss for URL: {url}")
                    return None
            except Exception as e:
                logger.error(f"Error getting from cache: {e}")
                return None

    async def put(self, url: str, content: str, content_type: str = "text/html",
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put content in cache with duplicate detection."""
        async with self._lock:
            try:
                # Check for duplicates
                is_duplicate, original_url = self.duplicate_detector.add_content(url, content)
                if is_duplicate and original_url:
                    self.stats.duplicate_count += 1
                    logger.debug(f"Duplicate content detected: {url} -> {original_url}")
                    # Create a lightweight reference entry
                    entry = CacheEntry(
                        url=url,
                        content="",  # Don't store duplicate content
                        content_hash="",
                        content_type=content_type,
                        timestamp=time.time(),
                        size=0,
                        metadata={"duplicate_of": original_url, **(metadata or {})}
                    )
                    self.cache[url] = entry
                    return True

                # Generate content hash
                content_hash = ContentHasher.hash_content(content)

                # Create cache entry
                entry = CacheEntry(
                    url=url,
                    content=content,
                    content_hash=content_hash,
                    content_type=content_type,
                    timestamp=time.time(),
                    size=len(content.encode('utf-8')),
                    metadata=metadata or {}
                )

                # Check if we need to evict entries
                await self._ensure_capacity(entry.size)

                # Add to cache
                self.cache[url] = entry
                self.stats.total_entries += 1
                self.stats.total_size += entry.size

                # Update statistics
                if self.stats.oldest_entry == 0 or entry.timestamp < self.stats.oldest_entry:
                    self.stats.oldest_entry = entry.timestamp
                if entry.timestamp > self.stats.newest_entry:
                    self.stats.newest_entry = entry.timestamp

                self._update_average_size()

                # Save to disk if persistence is enabled
                if self.config.enable_persistence:
                    await self._save_entry_to_disk(entry)

                logger.debug(f"Content cached for URL: {url} (size: {entry.size} bytes)")
                return True

            except Exception as e:
                logger.error(f"Error putting content in cache: {e}")
                return False

    async def remove(self, url: str) -> bool:
        """Remove content from cache."""
        async with self._lock:
            try:
                if url in self.cache:
                    entry = self.cache.pop(url)
                    self.stats.total_entries -= 1
                    self.stats.total_size -= entry.size
                    self.duplicate_detector.remove_content(url)
                    self._update_average_size()

                    # Remove from disk
                    if self.config.enable_persistence:
                        await self._remove_entry_from_disk(entry)

                    logger.debug(f"Content removed from cache: {url}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Error removing from cache: {e}")
                return False

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            try:
                count = len(self.cache)
                self.cache.clear()
                self.duplicate_detector = DuplicateDetector(self.config.similarity_threshold)
                self.stats = CacheStats()

                # Clear disk cache
                if self.config.enable_persistence:
                    await self._clear_disk_cache()

                logger.info(f"Cache cleared: {count} entries removed")
                return count
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return 0

    async def cleanup(self) -> int:
        """Clean up expired entries."""
        async with self._lock:
            try:
                current_time = time.time()
                expired_urls = []

                for url, entry in self.cache.items():
                    age = current_time - entry.timestamp
                    if age > self.config.max_age_seconds:
                        expired_urls.append(url)

                # Remove expired entries
                for url in expired_urls:
                    await self.remove(url)

                self.stats.cleanup_count += len(expired_urls)
                logger.info(f"Cache cleanup completed: {len(expired_urls)} expired entries removed")
                return len(expired_urls)

            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
                return 0

    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        try:
            # Check size limit
            while (self.stats.total_size + new_entry_size > self.config.max_size or
                   len(self.cache) >= self.config.max_entries):
                if not self.cache:
                    break

                # Remove least recently used entry
                url, entry = self.cache.popitem(last=False)
                self.stats.total_size -= entry.size
                self.stats.total_entries -= 1
                self.stats.eviction_count += 1
                self.duplicate_detector.remove_content(url)

                if self.config.enable_persistence:
                    await self._remove_entry_from_disk(entry)

                logger.debug(f"Evicted entry: {url}")
        except Exception as e:
            logger.error(f"Error ensuring cache capacity: {e}")

    def _update_average_size(self):
        """Update average entry size statistic."""
        if self.stats.total_entries > 0:
            self.stats.average_size = self.stats.total_size / self.stats.total_entries
        else:
            self.stats.average_size = 0.0

    async def _save_entry_to_disk(self, entry: CacheEntry):
        """Save cache entry to persistent storage."""
        try:
            if not self.config.enable_persistence:
                return

            # Create filename from URL hash
            url_hash = ContentHasher.hash_content(entry.url, "md5")
            filename = f"{url_hash}.json"
            filepath = self.config.cache_dir / filename

            # Prepare entry data
            entry_data = {
                "url": entry.url,
                "content": entry.content,
                "content_hash": entry.content_hash,
                "content_type": entry.content_type,
                "timestamp": entry.timestamp,
                "size": entry.size,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "metadata": entry.metadata
            }

            # Save with optional compression
            if self.config.enable_compression:
                compressed_data = gzip.compress(json.dumps(entry_data).encode('utf-8'))
                with open(filepath.with_suffix('.json.gz'), 'wb') as f:
                    f.write(compressed_data)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(entry_data, f)

        except Exception as e:
            logger.error(f"Error saving entry to disk: {e}")

    async def _remove_entry_from_disk(self, entry: CacheEntry):
        """Remove cache entry from persistent storage."""
        try:
            if not self.config.enable_persistence:
                return

            url_hash = ContentHasher.hash_content(entry.url, "md5")
            filename = f"{url_hash}.json"

            # Try both compressed and uncompressed versions
            for filepath in [
                self.config.cache_dir / f"{filename}.gz",
                self.config.cache_dir / filename
            ]:
                if filepath.exists():
                    filepath.unlink()
                    break

        except Exception as e:
            logger.error(f"Error removing entry from disk: {e}")

    async def _clear_disk_cache(self):
        """Clear all cached files from disk."""
        try:
            if not self.config.enable_persistence or not self.config.cache_dir.exists():
                return

            for file_path in self.config.cache_dir.glob("*.json*"):
                file_path.unlink()

        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

    async def load_from_disk(self) -> int:
        """Load cache entries from persistent storage."""
        try:
            if not self.config.enable_persistence or not self.config.cache_dir.exists():
                return 0

            loaded_count = 0
            current_time = time.time()

            for file_path in self.config.cache_dir.glob("*.json*"):
                try:
                    # Load entry data
                    if file_path.suffix == '.gz':
                        with open(file_path, 'rb') as f:
                            entry_data = json.loads(gzip.decompress(f.read()).decode('utf-8'))
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            entry_data = json.load(f)

                    # Check if entry is still valid
                    age = current_time - entry_data.get('timestamp', 0)
                    if age > self.config.max_age_seconds:
                        file_path.unlink()
                        continue

                    # Create cache entry
                    entry = CacheEntry(
                        url=entry_data['url'],
                        content=entry_data['content'],
                        content_hash=entry_data['content_hash'],
                        content_type=entry_data['content_type'],
                        timestamp=entry_data['timestamp'],
                        size=entry_data['size'],
                        access_count=entry_data.get('access_count', 0),
                        last_accessed=entry_data.get('last_accessed', entry_data['timestamp']),
                        metadata=entry_data.get('metadata', {})
                    )

                    # Add to cache
                    self.cache[entry.url] = entry
                    self.stats.total_entries += 1
                    self.stats.total_size += entry.size
                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Error loading cache entry from {file_path}: {e}")
                    # Remove corrupted file
                    try:
                        file_path.unlink()
                    except:
                        pass

            self._update_average_size()
            logger.info(f"Loaded {loaded_count} cache entries from disk")
            return loaded_count

        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")
            return 0

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self.stats

    def get_duplicates(self) -> Dict[str, List[str]]:
        """Get all duplicate content groups."""
        return self.duplicate_detector.get_duplicates()

    async def start_cleanup_task(self):
        """Start automatic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.cleanup_interval)
                    await self.cleanup()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Automatic cleanup task started")

    async def stop_cleanup_task(self):
        """Stop automatic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Automatic cleanup task stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.load_from_disk()
        if self.config.cleanup_interval > 0:
            await self.start_cleanup_task()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_cleanup_task()