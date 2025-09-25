"""
Comprehensive unit tests for web content caching system.

Tests cover:
- ContentHasher and duplicate detection
- Cache operations (get, put, remove, clear)
- LRU eviction and capacity management
- Persistent storage and compression
- Cache statistics and analytics
- Cleanup and maintenance operations
- Edge cases and error conditions
"""

import asyncio
import gzip
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from workspace_qdrant_mcp.web.cache import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    ContentCache,
    ContentHasher,
    DuplicateDetector,
)


class TestContentHasher:
    """Test content hashing functionality."""

    def test_hash_content_xxhash(self):
        """Test content hashing with xxhash."""
        content = "This is test content for hashing"
        hash1 = ContentHasher.hash_content(content, "xxhash")
        hash2 = ContentHasher.hash_content(content, "xxhash")

        assert hash1 == hash2
        assert len(hash1) > 0
        assert isinstance(hash1, str)

    def test_hash_content_sha256(self):
        """Test content hashing with SHA256."""
        content = "This is test content for hashing"
        hash1 = ContentHasher.hash_content(content, "sha256")
        hash2 = ContentHasher.hash_content(content, "sha256")

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_hash_content_md5(self):
        """Test content hashing with MD5."""
        content = "This is test content for hashing"
        hash1 = ContentHasher.hash_content(content, "md5")
        hash2 = ContentHasher.hash_content(content, "md5")

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_hash_content_fallback(self):
        """Test fallback to SHA256 for unknown algorithm."""
        content = "This is test content for hashing"
        hash_unknown = ContentHasher.hash_content(content, "unknown")
        hash_sha256 = ContentHasher.hash_content(content, "sha256")

        assert hash_unknown == hash_sha256

    def test_hash_content_different_content(self):
        """Test that different content produces different hashes."""
        content1 = "This is content one"
        content2 = "This is content two"

        hash1 = ContentHasher.hash_content(content1)
        hash2 = ContentHasher.hash_content(content2)

        assert hash1 != hash2

    def test_hash_content_empty_string(self):
        """Test hashing empty string."""
        hash_empty = ContentHasher.hash_content("")
        assert len(hash_empty) > 0

    def test_hash_content_unicode(self):
        """Test hashing unicode content."""
        content = "Ñice ünïcödé tëxt 🌟"
        hash1 = ContentHasher.hash_content(content)
        hash2 = ContentHasher.hash_content(content)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_similarity_hash_basic(self):
        """Test basic similarity hash generation."""
        content = "This is some content for similarity testing"
        hash1 = ContentHasher.similarity_hash(content)
        hash2 = ContentHasher.similarity_hash(content)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_similarity_hash_similar_content(self):
        """Test similarity hash with similar content."""
        content1 = "This is some content for similarity testing"
        content2 = "This is some content for similarity testing with extra"

        hash1 = ContentHasher.similarity_hash(content1)
        hash2 = ContentHasher.similarity_hash(content2)

        # Similar content might have same similarity hash
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)

    def test_similarity_hash_short_content(self):
        """Test similarity hash with very short content."""
        content = "ab"
        hash_result = ContentHasher.similarity_hash(content, ngram_size=3)

        assert len(hash_result) > 0

    def test_similarity_hash_empty_content(self):
        """Test similarity hash with empty content."""
        hash_result = ContentHasher.similarity_hash("")
        assert len(hash_result) > 0

    def test_similarity_hash_normalization(self):
        """Test that similarity hash normalizes content."""
        content1 = "This   IS   Some   CONTENT"
        content2 = "this is some content"

        hash1 = ContentHasher.similarity_hash(content1)
        hash2 = ContentHasher.similarity_hash(content2)

        assert hash1 == hash2

    @patch('workspace_qdrant_mcp.web.cache.logger')
    def test_hash_content_exception_handling(self, mock_logger):
        """Test exception handling in hash generation."""
        with patch('hashlib.sha256', side_effect=Exception("Hash error")):
            with patch('hashlib.md5') as mock_md5:
                mock_md5.return_value.hexdigest.return_value = "fallback_hash"
                result = ContentHasher.hash_content("test")
                assert result == "fallback_hash"
                mock_logger.warning.assert_called()

    @patch('workspace_qdrant_mcp.web.cache.logger')
    def test_similarity_hash_exception_handling(self, mock_logger):
        """Test exception handling in similarity hash generation."""
        with patch.object(ContentHasher, 'hash_content', side_effect=Exception("Test error")):
            # Should still work with fallback
            result = ContentHasher.similarity_hash("test content")
            mock_logger.warning.assert_called()


class TestDuplicateDetector:
    """Test duplicate detection functionality."""

    def test_exact_duplicate_detection(self):
        """Test detection of exact duplicate content."""
        detector = DuplicateDetector()
        content = "This is test content"

        # First addition should not be duplicate
        is_dup1, orig_url1 = detector.add_content("http://example.com/1", content)
        assert not is_dup1
        assert orig_url1 is None

        # Second addition with same content should be duplicate
        is_dup2, orig_url2 = detector.add_content("http://example.com/2", content)
        assert is_dup2
        assert orig_url2 == "http://example.com/1"

    def test_different_content_not_duplicate(self):
        """Test that different content is not marked as duplicate."""
        detector = DuplicateDetector()

        is_dup1, _ = detector.add_content("http://example.com/1", "Content one")
        is_dup2, _ = detector.add_content("http://example.com/2", "Content two")

        assert not is_dup1
        assert not is_dup2

    def test_similarity_threshold(self):
        """Test similarity threshold configuration."""
        detector = DuplicateDetector(similarity_threshold=0.5)
        assert detector.similarity_threshold == 0.5

    def test_remove_content(self):
        """Test removing content from duplicate tracking."""
        detector = DuplicateDetector()
        content = "Test content for removal"

        # Add content
        detector.add_content("http://example.com/1", content)

        # Verify it's tracked
        assert len(detector.content_hashes) > 0

        # Remove content
        detector.remove_content("http://example.com/1")

        # Verify removal affects tracking
        all_urls = []
        for urls in detector.content_hashes.values():
            all_urls.extend(urls)
        assert "http://example.com/1" not in all_urls

    def test_remove_nonexistent_content(self):
        """Test removing non-existent content doesn't cause errors."""
        detector = DuplicateDetector()

        # Should not raise exception
        detector.remove_content("http://nonexistent.com")

    def test_get_duplicates_empty(self):
        """Test getting duplicates when none exist."""
        detector = DuplicateDetector()
        duplicates = detector.get_duplicates()
        assert duplicates == {}

    def test_get_duplicates_with_duplicates(self):
        """Test getting duplicates when they exist."""
        detector = DuplicateDetector()
        content = "Duplicate content"

        detector.add_content("http://example.com/1", content)
        detector.add_content("http://example.com/2", content)

        duplicates = detector.get_duplicates()
        assert len(duplicates) > 0

        # Find the duplicate group
        for hash_key, urls in duplicates.items():
            if "http://example.com/1" in urls:
                assert "http://example.com/2" in urls
                assert len(urls) == 2
                break
        else:
            pytest.fail("Expected duplicate group not found")

    @patch('workspace_qdrant_mcp.web.cache.logger')
    def test_add_content_exception_handling(self, mock_logger):
        """Test exception handling in add_content."""
        detector = DuplicateDetector()

        with patch.object(ContentHasher, 'hash_content', side_effect=Exception("Test error")):
            is_dup, orig_url = detector.add_content("http://example.com", "content")
            assert not is_dup
            assert orig_url is None
            mock_logger.error.assert_called()

    @patch('workspace_qdrant_mcp.web.cache.logger')
    def test_remove_content_exception_handling(self, mock_logger):
        """Test exception handling in remove_content."""
        detector = DuplicateDetector()

        # Add some content first
        detector.add_content("http://example.com", "test")

        # Force an exception during removal
        with patch.dict(detector.content_hashes, {"key": Mock(side_effect=Exception("Test error"))}):
            detector.remove_content("http://example.com")
            mock_logger.error.assert_called()

    def test_calculate_similarity_fallback(self):
        """Test similarity calculation fallback behavior."""
        detector = DuplicateDetector()

        # Test the private method directly
        similarity = detector._calculate_similarity("content1", "http://example.com")
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(
            url="http://example.com",
            content="Test content",
            content_hash="abc123",
            content_type="text/html",
            timestamp=1234567890.0,
            size=100
        )

        assert entry.url == "http://example.com"
        assert entry.content == "Test content"
        assert entry.content_hash == "abc123"
        assert entry.content_type == "text/html"
        assert entry.timestamp == 1234567890.0
        assert entry.size == 100
        assert entry.access_count == 0
        assert entry.last_accessed > 0
        assert isinstance(entry.metadata, dict)

    def test_cache_entry_auto_size_calculation(self):
        """Test automatic size calculation when size <= 0."""
        content = "Test content with unicode: üñîçödé"
        entry = CacheEntry(
            url="http://example.com",
            content=content,
            content_hash="abc123",
            content_type="text/html",
            timestamp=1234567890.0,
            size=0  # Should trigger auto-calculation
        )

        expected_size = len(content.encode('utf-8'))
        assert entry.size == expected_size

    def test_cache_entry_with_metadata(self):
        """Test cache entry with metadata."""
        metadata = {"custom": "value", "number": 42}
        entry = CacheEntry(
            url="http://example.com",
            content="Test content",
            content_hash="abc123",
            content_type="text/html",
            timestamp=1234567890.0,
            size=100,
            metadata=metadata
        )

        assert entry.metadata == metadata


class TestCacheStats:
    """Test cache statistics functionality."""

    def test_cache_stats_defaults(self):
        """Test cache statistics default values."""
        stats = CacheStats()

        assert stats.total_entries == 0
        assert stats.total_size == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.duplicate_count == 0
        assert stats.eviction_count == 0
        assert stats.cleanup_count == 0
        assert stats.oldest_entry == 0.0
        assert stats.newest_entry == 0.0
        assert stats.average_size == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats()

        # No hits or misses
        assert stats.hit_rate == 0.0

        # Some hits and misses
        stats.hit_count = 7
        stats.miss_count = 3
        assert stats.hit_rate == 0.7

        # Only hits
        stats.miss_count = 0
        assert stats.hit_rate == 1.0

        # Only misses
        stats.hit_count = 0
        stats.miss_count = 5
        assert stats.hit_rate == 0.0

    def test_size_mb_calculation(self):
        """Test size in MB calculation."""
        stats = CacheStats()

        stats.total_size = 1024 * 1024  # 1MB
        assert stats.size_mb == 1.0

        stats.total_size = 5 * 1024 * 1024  # 5MB
        assert stats.size_mb == 5.0

        stats.total_size = 512 * 1024  # 0.5MB
        assert stats.size_mb == 0.5


class TestCacheConfig:
    """Test cache configuration functionality."""

    def test_cache_config_defaults(self):
        """Test cache configuration default values."""
        config = CacheConfig()

        assert config.max_size == 100 * 1024 * 1024  # 100MB
        assert config.max_entries == 10000
        assert config.max_age_seconds == 24 * 60 * 60  # 24 hours
        assert config.enable_compression is True
        assert config.enable_persistence is True
        assert config.similarity_threshold == 0.85
        assert config.cleanup_interval == 3600
        assert config.enable_analytics is True
        assert isinstance(config.cache_dir, Path)

    def test_cache_config_custom_values(self):
        """Test cache configuration with custom values."""
        config = CacheConfig(
            max_size=50 * 1024 * 1024,
            max_entries=5000,
            max_age_seconds=12 * 60 * 60,
            enable_compression=False,
            enable_persistence=False,
            similarity_threshold=0.9,
            cleanup_interval=1800,
            enable_analytics=False
        )

        assert config.max_size == 50 * 1024 * 1024
        assert config.max_entries == 5000
        assert config.max_age_seconds == 12 * 60 * 60
        assert config.enable_compression is False
        assert config.enable_persistence is False
        assert config.similarity_threshold == 0.9
        assert config.cleanup_interval == 1800
        assert config.enable_analytics is False

    def test_cache_config_cache_dir_string(self):
        """Test cache directory from string path."""
        config = CacheConfig(cache_dir="/tmp/test_cache")
        assert isinstance(config.cache_dir, Path)
        assert str(config.cache_dir) == "/tmp/test_cache"


class TestContentCache:
    """Test content cache functionality."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(
            max_size=1024,  # Small size for testing
            max_entries=3,
            max_age_seconds=3600,
            enable_persistence=False,  # Disable for most tests
            cleanup_interval=0  # Disable automatic cleanup
        )

    @pytest.fixture
    def cache(self, cache_config):
        """Create test cache instance."""
        return ContentCache(cache_config)

    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_config):
        """Test cache initialization."""
        cache = ContentCache(cache_config)
        assert cache.config == cache_config
        assert len(cache.cache) == 0
        assert isinstance(cache.duplicate_detector, DuplicateDetector)
        assert isinstance(cache.stats, CacheStats)

    @pytest.mark.asyncio
    async def test_cache_put_and_get(self, cache):
        """Test basic put and get operations."""
        url = "http://example.com/test"
        content = "Test content for caching"
        content_type = "text/html"

        # Put content
        result = await cache.put(url, content, content_type)
        assert result is True

        # Get content
        entry = await cache.get(url)
        assert entry is not None
        assert entry.url == url
        assert entry.content == content
        assert entry.content_type == content_type
        assert entry.access_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        entry = await cache.get("http://nonexistent.com")
        assert entry is None
        assert cache.stats.miss_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_updates_stats(self, cache):
        """Test that cache hits update statistics and access info."""
        url = "http://example.com/test"
        content = "Test content"

        # Put and get content
        await cache.put(url, content)
        entry1 = await cache.get(url)
        entry2 = await cache.get(url)

        assert entry1.access_count == 1
        assert entry2.access_count == 2
        assert entry2.last_accessed > entry1.last_accessed
        assert cache.stats.hit_count == 2

    @pytest.mark.asyncio
    async def test_cache_lru_behavior(self, cache):
        """Test LRU (Least Recently Used) behavior."""
        # Add three entries (cache max is 3)
        await cache.put("http://example.com/1", "content1")
        await cache.put("http://example.com/2", "content2")
        await cache.put("http://example.com/3", "content3")

        # Access first entry to make it most recently used
        await cache.get("http://example.com/1")

        # Add fourth entry, should evict entry 2 (least recently used)
        await cache.put("http://example.com/4", "content4")

        # Check that entry 2 was evicted
        entry2 = await cache.get("http://example.com/2")
        assert entry2 is None

        # Check that entry 1 is still there (was accessed recently)
        entry1 = await cache.get("http://example.com/1")
        assert entry1 is not None

    @pytest.mark.asyncio
    async def test_cache_size_eviction(self, cache):
        """Test eviction based on cache size limit."""
        # Add content that exceeds size limit
        large_content = "x" * 500  # 500 bytes
        await cache.put("http://example.com/1", large_content)
        await cache.put("http://example.com/2", large_content)

        # This should trigger eviction due to size limit (1024 bytes)
        await cache.put("http://example.com/3", large_content)

        assert cache.stats.eviction_count > 0

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, cache):
        """Test duplicate content detection."""
        content = "Duplicate test content"

        # Add same content with different URLs
        result1 = await cache.put("http://example.com/1", content)
        result2 = await cache.put("http://example.com/2", content)

        assert result1 is True
        assert result2 is True
        assert cache.stats.duplicate_count == 1

        # Check that duplicate entry references original
        entry2 = await cache.get("http://example.com/2")
        assert entry2 is not None
        assert "duplicate_of" in entry2.metadata

    @pytest.mark.asyncio
    async def test_cache_remove(self, cache):
        """Test removing entries from cache."""
        url = "http://example.com/test"
        content = "Test content"

        # Add and then remove
        await cache.put(url, content)
        initial_size = cache.stats.total_size
        initial_entries = cache.stats.total_entries

        result = await cache.remove(url)
        assert result is True
        assert cache.stats.total_size < initial_size
        assert cache.stats.total_entries < initial_entries

        # Verify entry is gone
        entry = await cache.get(url)
        assert entry is None

    @pytest.mark.asyncio
    async def test_cache_remove_nonexistent(self, cache):
        """Test removing non-existent entry."""
        result = await cache.remove("http://nonexistent.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test clearing all cache entries."""
        # Add some entries
        await cache.put("http://example.com/1", "content1")
        await cache.put("http://example.com/2", "content2")

        # Clear cache
        count = await cache.clear()
        assert count == 2
        assert len(cache.cache) == 0
        assert cache.stats.total_entries == 0
        assert cache.stats.total_size == 0

    @pytest.mark.asyncio
    async def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Use very short max age for testing
        config = CacheConfig(max_age_seconds=1, enable_persistence=False)
        cache = ContentCache(config)

        # Add entry
        await cache.put("http://example.com/test", "content")

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Run cleanup
        cleaned = await cache.cleanup()
        assert cleaned == 1
        assert len(cache.cache) == 0

    @pytest.mark.asyncio
    async def test_cache_with_metadata(self, cache):
        """Test caching with custom metadata."""
        url = "http://example.com/test"
        content = "Test content"
        metadata = {"custom": "value", "tags": ["test", "cache"]}

        await cache.put(url, content, metadata=metadata)
        entry = await cache.get(url)

        assert entry.metadata == metadata

    @pytest.mark.asyncio
    async def test_get_stats(self, cache):
        """Test getting cache statistics."""
        stats = cache.get_stats()
        assert isinstance(stats, CacheStats)

        # Add some content and check stats update
        await cache.put("http://example.com/test", "content")
        updated_stats = cache.get_stats()
        assert updated_stats.total_entries == 1

    @pytest.mark.asyncio
    async def test_get_duplicates(self, cache):
        """Test getting duplicate content information."""
        content = "Duplicate content"

        # Add duplicates
        await cache.put("http://example.com/1", content)
        await cache.put("http://example.com/2", content)

        duplicates = cache.get_duplicates()
        assert len(duplicates) > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_put(self, cache):
        """Test error handling in put operation."""
        with patch.object(cache.duplicate_detector, 'add_content', side_effect=Exception("Test error")):
            result = await cache.put("http://example.com", "content")
            assert result is False

    @pytest.mark.asyncio
    async def test_error_handling_in_get(self, cache):
        """Test error handling in get operation."""
        # Add entry first
        await cache.put("http://example.com", "content")

        # Force error during get
        with patch.dict(cache.cache, {"http://example.com": Mock(side_effect=Exception("Test error"))}):
            entry = await cache.get("http://example.com")
            assert entry is None

    @pytest.mark.asyncio
    async def test_error_handling_in_remove(self, cache):
        """Test error handling in remove operation."""
        await cache.put("http://example.com", "content")

        with patch.object(cache.cache, 'pop', side_effect=Exception("Test error")):
            result = await cache.remove("http://example.com")
            assert result is False

    @pytest.mark.asyncio
    async def test_error_handling_in_clear(self, cache):
        """Test error handling in clear operation."""
        with patch.object(cache.cache, 'clear', side_effect=Exception("Test error")):
            count = await cache.clear()
            assert count == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_cleanup(self, cache):
        """Test error handling in cleanup operation."""
        with patch.object(cache.cache, 'items', side_effect=Exception("Test error")):
            count = await cache.cleanup()
            assert count == 0

    @pytest.mark.asyncio
    async def test_ensure_capacity_empty_cache(self, cache):
        """Test ensure_capacity with empty cache."""
        # Should not raise exception
        await cache._ensure_capacity(100)

    @pytest.mark.asyncio
    async def test_update_average_size_zero_entries(self, cache):
        """Test average size calculation with zero entries."""
        cache._update_average_size()
        assert cache.stats.average_size == 0.0

    @pytest.mark.asyncio
    async def test_update_average_size_with_entries(self, cache):
        """Test average size calculation with entries."""
        await cache.put("http://example.com", "test content")
        assert cache.stats.average_size > 0


class TestContentCachePersistence:
    """Test persistent storage functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def persistent_config(self, temp_cache_dir):
        """Create configuration with persistence enabled."""
        return CacheConfig(
            cache_dir=temp_cache_dir,
            enable_persistence=True,
            enable_compression=True,
            cleanup_interval=0
        )

    @pytest.fixture
    def persistent_cache(self, persistent_config):
        """Create cache with persistence enabled."""
        return ContentCache(persistent_config)

    @pytest.mark.asyncio
    async def test_save_entry_to_disk(self, persistent_cache):
        """Test saving cache entry to disk."""
        url = "http://example.com/test"
        content = "Test content for persistence"

        await persistent_cache.put(url, content)

        # Check that file was created
        cache_files = list(persistent_cache.config.cache_dir.glob("*.json*"))
        assert len(cache_files) > 0

    @pytest.mark.asyncio
    async def test_load_from_disk(self, persistent_cache):
        """Test loading cache entries from disk."""
        url = "http://example.com/test"
        content = "Test content for persistence"

        # Save entry
        await persistent_cache.put(url, content)

        # Create new cache instance (simulating restart)
        new_cache = ContentCache(persistent_cache.config)
        loaded_count = await new_cache.load_from_disk()

        assert loaded_count == 1
        entry = await new_cache.get(url)
        assert entry is not None
        assert entry.content == content

    @pytest.mark.asyncio
    async def test_remove_entry_from_disk(self, persistent_cache):
        """Test removing cache entry from disk."""
        url = "http://example.com/test"
        content = "Test content"

        # Add and remove entry
        await persistent_cache.put(url, content)
        initial_files = len(list(persistent_cache.config.cache_dir.glob("*.json*")))

        await persistent_cache.remove(url)
        final_files = len(list(persistent_cache.config.cache_dir.glob("*.json*")))

        assert final_files < initial_files

    @pytest.mark.asyncio
    async def test_clear_disk_cache(self, persistent_cache):
        """Test clearing all cached files from disk."""
        # Add some entries
        await persistent_cache.put("http://example.com/1", "content1")
        await persistent_cache.put("http://example.com/2", "content2")

        # Clear cache
        await persistent_cache.clear()

        # Check that no cache files remain
        cache_files = list(persistent_cache.config.cache_dir.glob("*.json*"))
        assert len(cache_files) == 0

    @pytest.mark.asyncio
    async def test_load_expired_entries_cleaned(self, temp_cache_dir):
        """Test that expired entries are cleaned during load."""
        # Create cache with very short max age
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            enable_persistence=True,
            max_age_seconds=1
        )
        cache = ContentCache(config)

        # Add entry
        await cache.put("http://example.com/test", "content")

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Load from disk - should clean expired entries
        new_cache = ContentCache(config)
        loaded_count = await new_cache.load_from_disk()

        assert loaded_count == 0

    @pytest.mark.asyncio
    async def test_persistence_disabled(self):
        """Test behavior when persistence is disabled."""
        config = CacheConfig(enable_persistence=False)
        cache = ContentCache(config)

        await cache.put("http://example.com/test", "content")

        # Should not create any files
        cache_files = list(config.cache_dir.glob("*.json*")) if config.cache_dir.exists() else []
        assert len(cache_files) == 0

    @pytest.mark.asyncio
    async def test_compression_enabled(self, persistent_cache):
        """Test content compression when enabled."""
        url = "http://example.com/test"
        content = "Test content for compression testing"

        await persistent_cache.put(url, content)

        # Check for compressed files
        compressed_files = list(persistent_cache.config.cache_dir.glob("*.gz"))
        assert len(compressed_files) > 0

    @pytest.mark.asyncio
    async def test_load_corrupted_file_handling(self, persistent_cache):
        """Test handling of corrupted cache files during load."""
        # Create a corrupted cache file
        corrupted_file = persistent_cache.config.cache_dir / "corrupted.json"
        corrupted_file.write_text("invalid json content")

        # Should handle gracefully
        loaded_count = await persistent_cache.load_from_disk()
        assert loaded_count == 0

        # Corrupted file should be removed
        assert not corrupted_file.exists()

    @pytest.mark.asyncio
    async def test_save_entry_error_handling(self, persistent_cache):
        """Test error handling during entry saving."""
        with patch('builtins.open', side_effect=PermissionError("Cannot write")):
            # Should not raise exception
            await persistent_cache.put("http://example.com/test", "content")

    @pytest.mark.asyncio
    async def test_remove_entry_error_handling(self, persistent_cache):
        """Test error handling during entry removal from disk."""
        await persistent_cache.put("http://example.com/test", "content")

        with patch.object(Path, 'unlink', side_effect=PermissionError("Cannot delete")):
            # Should not raise exception
            await persistent_cache.remove("http://example.com/test")

    @pytest.mark.asyncio
    async def test_clear_disk_error_handling(self, persistent_cache):
        """Test error handling during disk cache clearing."""
        await persistent_cache.put("http://example.com/test", "content")

        with patch.object(Path, 'glob', side_effect=PermissionError("Cannot list")):
            # Should not raise exception
            await persistent_cache.clear()


class TestContentCacheCleanup:
    """Test automatic cleanup functionality."""

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self):
        """Test starting automatic cleanup task."""
        config = CacheConfig(cleanup_interval=0.1)  # Very short interval
        cache = ContentCache(config)

        await cache.start_cleanup_task()
        assert cache._cleanup_task is not None
        assert not cache._cleanup_task.done()

        await cache.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self):
        """Test stopping automatic cleanup task."""
        config = CacheConfig(cleanup_interval=0.1)
        cache = ContentCache(config)

        await cache.start_cleanup_task()
        await cache.stop_cleanup_task()

        assert cache._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_start_cleanup_task_already_running(self):
        """Test starting cleanup task when already running."""
        config = CacheConfig(cleanup_interval=0.1)
        cache = ContentCache(config)

        await cache.start_cleanup_task()
        first_task = cache._cleanup_task

        # Starting again should not create new task
        await cache.start_cleanup_task()
        assert cache._cleanup_task is first_task

        await cache.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_cleanup_task_error_handling(self):
        """Test error handling in cleanup task."""
        config = CacheConfig(cleanup_interval=0.1)
        cache = ContentCache(config)

        # Mock cleanup to raise exception
        with patch.object(cache, 'cleanup', side_effect=Exception("Cleanup error")):
            await cache.start_cleanup_task()
            await asyncio.sleep(0.2)  # Let task run and handle error
            await cache.stop_cleanup_task()

        # Task should have handled the error and continued running

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test cache as async context manager."""
        config = CacheConfig(
            enable_persistence=False,
            cleanup_interval=0.1
        )

        async with ContentCache(config) as cache:
            await cache.put("http://example.com/test", "content")
            entry = await cache.get("http://example.com/test")
            assert entry is not None

        # Cleanup task should be stopped after exiting context

    @pytest.mark.asyncio
    async def test_async_context_manager_with_persistence(self, temp_cache_dir):
        """Test async context manager with persistence."""
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            enable_persistence=True,
            cleanup_interval=0.1
        )

        async with ContentCache(config) as cache:
            await cache.put("http://example.com/test", "content")

        # Should have loaded from disk and started cleanup task


# Integration test for complete workflow
class TestContentCacheIntegration:
    """Integration tests for complete cache workflow."""

    @pytest.mark.asyncio
    async def test_complete_cache_lifecycle(self):
        """Test complete cache lifecycle with all features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=Path(temp_dir),
                enable_persistence=True,
                enable_compression=True,
                max_entries=5,
                max_age_seconds=3600,
                cleanup_interval=0
            )

            async with ContentCache(config) as cache:
                # Add various types of content
                await cache.put("http://example.com/page1", "Unique content 1", "text/html")
                await cache.put("http://example.com/page2", "Unique content 2", "text/html")
                await cache.put("http://example.com/page3", "Unique content 1", "text/html")  # Duplicate

                # Verify duplicate detection
                assert cache.stats.duplicate_count == 1

                # Access content to update LRU
                entry1 = await cache.get("http://example.com/page1")
                assert entry1.access_count == 1

                # Add more content to trigger eviction
                for i in range(4, 8):
                    await cache.put(f"http://example.com/page{i}", f"Content {i}", "text/html")

                # Verify eviction occurred
                assert cache.stats.eviction_count > 0

                # Check final state
                stats = cache.get_stats()
                assert stats.total_entries <= config.max_entries
                assert stats.hit_rate > 0

                # Verify persistence
                cache_files = list(config.cache_dir.glob("*.json*"))
                assert len(cache_files) > 0

            # Test loading from persistence
            async with ContentCache(config) as new_cache:
                loaded_count = await new_cache.load_from_disk()
                assert loaded_count > 0