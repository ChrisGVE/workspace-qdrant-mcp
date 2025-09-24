"""
Comprehensive unit tests for advanced file filtering system.

Tests cover all filtering criteria, edge cases, performance scenarios,
and error handling for the AdvancedFileFilter system.

Test Categories:
    - Pattern matching (regex, glob patterns)
    - Content-based filtering with streaming
    - Size-based filtering with edge cases
    - MIME type detection and filtering
    - Performance optimization and caching
    - Error handling and recovery
    - Statistics and monitoring
    - Batch processing scenarios
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from src.python.common.core.advanced_file_filters import (
    AdvancedFileFilter,
    FilterConfig,
    FilterStatistics,
    PatternCache,
    create_code_file_filter,
    create_document_filter
)


class TestFilterConfig:
    """Test FilterConfig dataclass functionality."""

    def test_filter_config_default_initialization(self):
        """Test default filter configuration initialization."""
        config = FilterConfig()

        assert config.include_patterns == []
        assert config.exclude_patterns == []
        assert config.min_file_size is None
        assert config.max_file_size is None
        assert config.content_filters == []
        assert config.content_sample_size == 4096
        assert config.allowed_mime_types == set()
        assert config.blocked_mime_types == set()
        assert config.regex_cache_size == 1000
        assert config.enable_content_filtering is True
        assert config.enable_mime_type_detection is True
        assert config.max_concurrent_content_checks == 10
        assert config.case_sensitive_patterns is False
        assert config.follow_symlinks is True
        assert config.check_file_permissions is True

    def test_filter_config_custom_initialization(self):
        """Test custom filter configuration initialization."""
        config = FilterConfig(
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["__pycache__/*"],
            min_file_size=1024,
            max_file_size=1024*1024,
            content_filters=["python", "javascript"],
            content_sample_size=8192,
            allowed_mime_types={"text/x-python", "text/javascript"},
            blocked_mime_types={"application/octet-stream"},
            regex_cache_size=500,
            enable_content_filtering=False,
            enable_mime_type_detection=False,
            max_concurrent_content_checks=5,
            case_sensitive_patterns=True,
            follow_symlinks=False,
            check_file_permissions=False
        )

        assert config.include_patterns == ["*.py", "*.js"]
        assert config.exclude_patterns == ["__pycache__/*"]
        assert config.min_file_size == 1024
        assert config.max_file_size == 1024*1024
        assert config.content_filters == ["python", "javascript"]
        assert config.content_sample_size == 8192
        assert config.allowed_mime_types == {"text/x-python", "text/javascript"}
        assert config.blocked_mime_types == {"application/octet-stream"}
        assert config.regex_cache_size == 500
        assert config.enable_content_filtering is False
        assert config.enable_mime_type_detection is False
        assert config.max_concurrent_content_checks == 5
        assert config.case_sensitive_patterns is True
        assert config.follow_symlinks is False
        assert config.check_file_permissions is False

    def test_filter_config_validation(self):
        """Test filter configuration validation."""
        # Test invalid content sample size
        config = FilterConfig(content_sample_size=0)
        assert config.content_sample_size == 4096  # Should be corrected to default

        config = FilterConfig(content_sample_size=-100)
        assert config.content_sample_size == 4096  # Should be corrected to default

        # Test invalid size limits
        with pytest.raises(ValueError, match="max_file_size cannot be less than min_file_size"):
            FilterConfig(min_file_size=1024, max_file_size=512)


class TestFilterStatistics:
    """Test FilterStatistics functionality."""

    def test_filter_statistics_initialization(self):
        """Test filter statistics initialization."""
        stats = FilterStatistics()

        assert stats.total_files_checked == 0
        assert stats.files_accepted == 0
        assert stats.files_rejected == 0
        assert stats.patterns_matched == 0
        assert stats.content_checks_performed == 0
        assert stats.size_rejections == 0
        assert stats.mime_type_rejections == 0
        assert stats.regex_cache_hits == 0
        assert stats.regex_cache_misses == 0
        assert stats.avg_processing_time_ms == 0.0
        assert stats.peak_memory_usage_mb == 0.0
        assert len(stats.recent_processing_times) == 0

    def test_processing_time_tracking(self):
        """Test processing time tracking functionality."""
        stats = FilterStatistics()

        # Add some processing times
        times = [10.5, 15.2, 8.7, 12.3, 9.1]
        for time_ms in times:
            stats.add_processing_time(time_ms)

        assert len(stats.recent_processing_times) == 5
        assert stats.avg_processing_time_ms == sum(times) / len(times)

        # Test rolling window behavior (maxlen=1000)
        for i in range(1000):
            stats.add_processing_time(1.0)

        assert len(stats.recent_processing_times) == 1000
        assert stats.avg_processing_time_ms == 1.0

    def test_derived_statistics(self):
        """Test derived statistics calculations."""
        stats = FilterStatistics()

        # Test rejection rate
        stats.total_files_checked = 100
        stats.files_rejected = 25
        stats.files_accepted = 75

        assert stats.rejection_rate == 25.0

        # Test edge case: no files checked
        stats.total_files_checked = 0
        assert stats.rejection_rate == 0.0

        # Test throughput calculation
        stats.add_processing_time(10.0)  # 10ms per file
        expected_throughput = 1000.0 / 10.0  # 100 files/sec
        assert abs(stats.throughput_files_per_sec - expected_throughput) < 0.01

    def test_statistics_serialization(self):
        """Test statistics to_dict conversion."""
        stats = FilterStatistics()
        stats.total_files_checked = 100
        stats.files_accepted = 80
        stats.files_rejected = 20
        stats.add_processing_time(5.0)

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["total_files_checked"] == 100
        assert stats_dict["files_accepted"] == 80
        assert stats_dict["files_rejected"] == 20
        assert stats_dict["rejection_rate_percent"] == 20.0
        assert abs(stats_dict["throughput_files_per_sec"] - 200.0) < 0.01

    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        stats = FilterStatistics()

        # Set some values
        stats.total_files_checked = 100
        stats.files_accepted = 80
        stats.files_rejected = 20
        stats.add_processing_time(5.0)
        stats.patterns_matched = 50

        # Reset
        stats.reset()

        # Verify all values are reset
        assert stats.total_files_checked == 0
        assert stats.files_accepted == 0
        assert stats.files_rejected == 0
        assert stats.patterns_matched == 0
        assert stats.avg_processing_time_ms == 0.0
        assert len(stats.recent_processing_times) == 0


class TestPatternCache:
    """Test PatternCache functionality."""

    def test_pattern_cache_initialization(self):
        """Test pattern cache initialization."""
        cache = PatternCache(max_size=100)

        assert cache.max_size == 100
        assert len(cache.patterns) == 0
        assert len(cache.access_order) == 0
        assert len(cache.access_counts) == 0

    def test_pattern_caching(self):
        """Test pattern compilation and caching."""
        cache = PatternCache(max_size=10)

        # Test pattern compilation and caching
        pattern1 = cache.get_pattern(r".*\.py$")
        pattern2 = cache.get_pattern(r".*\.py$")  # Should be same cached instance

        assert pattern1 is pattern2  # Same compiled pattern object
        assert len(cache.patterns) == 1
        assert cache.access_counts[r".*\.py$:True"] == 2

    def test_case_sensitivity(self):
        """Test case-sensitive pattern caching."""
        cache = PatternCache(max_size=10)

        pattern_sensitive = cache.get_pattern("TEST", case_sensitive=True)
        pattern_insensitive = cache.get_pattern("TEST", case_sensitive=False)

        assert pattern_sensitive is not pattern_insensitive
        assert len(cache.patterns) == 2

        # Test different behavior
        assert pattern_sensitive.match("TEST")
        assert not pattern_sensitive.match("test")
        assert pattern_insensitive.match("TEST")
        assert pattern_insensitive.match("test")

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = PatternCache(max_size=3)

        # Fill cache
        patterns = []
        for i in range(3):
            pattern = cache.get_pattern(f"pattern{i}")
            patterns.append(pattern)

        assert len(cache.patterns) == 3

        # Add one more - should trigger eviction
        new_pattern = cache.get_pattern("pattern3")

        # Cache should still have max_size entries
        assert len(cache.patterns) <= cache.max_size

        # Access an existing pattern to test LRU behavior
        cache.get_pattern("pattern1")  # Should move to end of LRU

        # Add another new pattern
        cache.get_pattern("pattern4")

        assert len(cache.patterns) <= cache.max_size

    def test_invalid_pattern_handling(self):
        """Test handling of invalid regex patterns."""
        cache = PatternCache(max_size=10)

        # Test invalid regex
        pattern = cache.get_pattern("[invalid regex")

        # Should return a pattern that never matches
        assert not pattern.match("anything")
        assert not pattern.search("anything")

    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        cache = PatternCache(max_size=10)

        # Add some patterns with different access counts
        cache.get_pattern("pattern1")
        cache.get_pattern("pattern2")
        cache.get_pattern("pattern1")  # Access again
        cache.get_pattern("pattern1")  # Access again

        stats = cache.get_stats()

        assert stats["cached_patterns"] == 2
        assert stats["max_size"] == 10
        assert stats["total_accesses"] == 4
        assert len(stats["most_used_patterns"]) == 2

        # Most used should be pattern1
        most_used = stats["most_used_patterns"][0]
        assert "pattern1" in most_used[0]
        assert most_used[1] == 3

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        cache = PatternCache(max_size=10)

        # Add some patterns
        cache.get_pattern("pattern1")
        cache.get_pattern("pattern2")

        assert len(cache.patterns) == 2
        assert len(cache.access_order) > 0
        assert len(cache.access_counts) == 2

        # Clear cache
        cache.clear()

        assert len(cache.patterns) == 0
        assert len(cache.access_order) == 0
        assert len(cache.access_counts) == 0


class TestAdvancedFileFilterBasic:
    """Test basic AdvancedFileFilter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_files = {}

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_file(self, name: str, content: str = "test content", size: int = None) -> Path:
        """Create a test file with specified content or size."""
        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if size is not None:
            # Create file of specific size
            content = "A" * size

        file_path.write_text(content, encoding='utf-8')
        self.test_files[name] = file_path
        return file_path

    def test_filter_initialization_with_dict_config(self):
        """Test filter initialization with dictionary configuration."""
        config_dict = {
            "include_patterns": ["*.py"],
            "exclude_patterns": ["__pycache__/*"],
            "max_file_size": 1024 * 1024
        }

        filter_manager = AdvancedFileFilter(config_dict)

        assert filter_manager.config.include_patterns == ["*.py"]
        assert filter_manager.config.exclude_patterns == ["__pycache__/*"]
        assert filter_manager.config.max_file_size == 1024 * 1024

    def test_filter_initialization_with_config_object(self):
        """Test filter initialization with FilterConfig object."""
        config = FilterConfig(
            include_patterns=["*.js"],
            exclude_patterns=["node_modules/*"],
            min_file_size=512
        )

        filter_manager = AdvancedFileFilter(config)

        assert filter_manager.config.include_patterns == ["*.js"]
        assert filter_manager.config.exclude_patterns == ["node_modules/*"]
        assert filter_manager.config.min_file_size == 512

    @pytest.mark.asyncio
    async def test_basic_file_acceptance(self):
        """Test basic file acceptance scenario."""
        # Create test file
        test_file = self.create_test_file("test.py", "print('hello')")

        # Create filter that accepts Python files
        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(test_file)

        assert should_process is True
        assert "accepted" in reason
        assert filter_manager.statistics.files_accepted == 1
        assert filter_manager.statistics.total_files_checked == 1

    @pytest.mark.asyncio
    async def test_basic_file_rejection(self):
        """Test basic file rejection scenario."""
        # Create test file
        test_file = self.create_test_file("test.txt", "plain text")

        # Create filter that only accepts Python files
        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(test_file)

        assert should_process is False
        assert "no_include_pattern_match" in reason
        assert filter_manager.statistics.files_rejected == 1
        assert filter_manager.statistics.total_files_checked == 1

    @pytest.mark.asyncio
    async def test_nonexistent_file_handling(self):
        """Test handling of non-existent files."""
        nonexistent_file = self.temp_dir / "does_not_exist.py"

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(nonexistent_file)

        assert should_process is False
        assert "file_not_found" in reason

    @pytest.mark.asyncio
    async def test_directory_handling(self):
        """Test handling of directories (should be rejected)."""
        test_dir = self.temp_dir / "test_directory"
        test_dir.mkdir()

        config = FilterConfig(include_patterns=["*"])
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(test_dir)

        assert should_process is False
        assert "not_a_file" in reason


class TestAdvancedFileFilterPatterns:
    """Test pattern-based filtering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_file(self, name: str, content: str = "test") -> Path:
        """Create a test file."""
        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.asyncio
    async def test_include_patterns(self):
        """Test include pattern functionality."""
        # Create test files
        py_file = self.create_test_file("script.py", "print('hello')")
        js_file = self.create_test_file("script.js", "console.log('hello')")
        txt_file = self.create_test_file("readme.txt", "documentation")

        # Filter for Python and JavaScript files
        config = FilterConfig(include_patterns=[r".*\.py$", r".*\.js$"])
        filter_manager = AdvancedFileFilter(config)

        # Test Python file (should be accepted)
        should_process, reason = await filter_manager.should_process_file(py_file)
        assert should_process is True

        # Test JavaScript file (should be accepted)
        should_process, reason = await filter_manager.should_process_file(js_file)
        assert should_process is True

        # Test text file (should be rejected)
        should_process, reason = await filter_manager.should_process_file(txt_file)
        assert should_process is False
        assert "no_include_pattern_match" in reason

    @pytest.mark.asyncio
    async def test_exclude_patterns(self):
        """Test exclude pattern functionality."""
        # Create test files
        main_file = self.create_test_file("main.py", "main code")
        cache_file = self.create_test_file("__pycache__/module.pyc", "cached bytecode")
        test_file = self.create_test_file("test.py", "test code")

        # Filter that excludes cache files
        config = FilterConfig(
            include_patterns=[r".*\.py$", r".*\.pyc$"],
            exclude_patterns=[r".*__pycache__.*"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Test main file (should be accepted)
        should_process, reason = await filter_manager.should_process_file(main_file)
        assert should_process is True

        # Test cached file (should be rejected by exclude pattern)
        should_process, reason = await filter_manager.should_process_file(cache_file)
        assert should_process is False
        assert "excluded_by_pattern" in reason

        # Test regular file (should be accepted)
        should_process, reason = await filter_manager.should_process_file(test_file)
        assert should_process is True

    @pytest.mark.asyncio
    async def test_no_include_patterns(self):
        """Test behavior when no include patterns are specified."""
        # Create test files
        any_file = self.create_test_file("any_file.xyz", "any content")

        # Filter with only exclude patterns (should accept all except excluded)
        config = FilterConfig(exclude_patterns=[r".*\.tmp$"])
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(any_file)
        assert should_process is True
        assert "no_include_patterns" in reason

    @pytest.mark.asyncio
    async def test_case_sensitive_patterns(self):
        """Test case-sensitive pattern matching."""
        # Create test files
        upper_file = self.create_test_file("FILE.PY", "upper case extension")
        lower_file = self.create_test_file("file.py", "lower case extension")

        # Case-sensitive filter
        config = FilterConfig(
            include_patterns=[r".*\.py$"],
            case_sensitive_patterns=True
        )
        filter_manager = AdvancedFileFilter(config)

        # Upper case should be rejected
        should_process, reason = await filter_manager.should_process_file(upper_file)
        assert should_process is False

        # Lower case should be accepted
        should_process, reason = await filter_manager.should_process_file(lower_file)
        assert should_process is True

    @pytest.mark.asyncio
    async def test_case_insensitive_patterns(self):
        """Test case-insensitive pattern matching."""
        # Create test files
        upper_file = self.create_test_file("FILE.PY", "upper case extension")
        lower_file = self.create_test_file("file.py", "lower case extension")
        mixed_file = self.create_test_file("File.Py", "mixed case extension")

        # Case-insensitive filter (default)
        config = FilterConfig(
            include_patterns=[r".*\.py$"],
            case_sensitive_patterns=False
        )
        filter_manager = AdvancedFileFilter(config)

        # All should be accepted
        for test_file in [upper_file, lower_file, mixed_file]:
            should_process, reason = await filter_manager.should_process_file(test_file)
            assert should_process is True

    @pytest.mark.asyncio
    async def test_complex_regex_patterns(self):
        """Test complex regex pattern matching."""
        # Create test files
        test_files = [
            self.create_test_file("test_module.py", "test code"),
            self.create_test_file("module_test.py", "test code"),
            self.create_test_file("test.py", "test code"),
            self.create_test_file("module.py", "main code"),
            self.create_test_file("integration_test.js", "test code"),
        ]

        # Filter for test files (any file with 'test' in the name)
        config = FilterConfig(include_patterns=[r".*test.*\.(py|js)$"])
        filter_manager = AdvancedFileFilter(config)

        expected_results = [True, True, True, False, True]

        for test_file, expected in zip(test_files, expected_results):
            should_process, reason = await filter_manager.should_process_file(test_file)
            assert should_process == expected, f"File {test_file.name} should be {expected}"

    @pytest.mark.asyncio
    async def test_pattern_statistics_tracking(self):
        """Test pattern matching statistics tracking."""
        # Create test files
        matching_file = self.create_test_file("test.py", "python code")
        non_matching_file = self.create_test_file("test.txt", "text content")

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        # Process matching file
        await filter_manager.should_process_file(matching_file)
        assert filter_manager.statistics.patterns_matched >= 1

        # Process non-matching file
        await filter_manager.should_process_file(non_matching_file)

        # Check statistics
        stats = filter_manager.get_statistics()
        assert stats.total_files_checked == 2
        assert stats.files_accepted == 1
        assert stats.files_rejected == 1


class TestAdvancedFileFilterSizeFiltering:
    """Test size-based filtering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_file(self, name: str, size: int) -> Path:
        """Create a test file of specific size."""
        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file of specific size
        content = "A" * size
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.asyncio
    async def test_max_file_size_filtering(self):
        """Test maximum file size filtering."""
        # Create files of different sizes
        small_file = self.create_test_file("small.txt", 1024)  # 1KB
        large_file = self.create_test_file("large.txt", 2048)  # 2KB

        # Filter with 1.5KB limit
        config = FilterConfig(
            max_file_size=1536,  # 1.5KB
            include_patterns=["*.txt"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Small file should be accepted
        should_process, reason = await filter_manager.should_process_file(small_file)
        assert should_process is True

        # Large file should be rejected
        should_process, reason = await filter_manager.should_process_file(large_file)
        assert should_process is False
        assert "size_limit_exceeded" in reason
        assert filter_manager.statistics.size_rejections == 1

    @pytest.mark.asyncio
    async def test_min_file_size_filtering(self):
        """Test minimum file size filtering."""
        # Create files of different sizes
        tiny_file = self.create_test_file("tiny.txt", 512)   # 512B
        normal_file = self.create_test_file("normal.txt", 2048)  # 2KB

        # Filter with 1KB minimum
        config = FilterConfig(
            min_file_size=1024,  # 1KB
            include_patterns=["*.txt"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Tiny file should be rejected
        should_process, reason = await filter_manager.should_process_file(tiny_file)
        assert should_process is False
        assert "size_limit_exceeded" in reason
        assert filter_manager.statistics.size_rejections == 1

        # Normal file should be accepted
        should_process, reason = await filter_manager.should_process_file(normal_file)
        assert should_process is True

    @pytest.mark.asyncio
    async def test_size_range_filtering(self):
        """Test file size range filtering."""
        # Create files of various sizes
        too_small = self.create_test_file("too_small.txt", 256)   # 256B
        just_right = self.create_test_file("just_right.txt", 1024)  # 1KB
        too_large = self.create_test_file("too_large.txt", 4096)   # 4KB

        # Filter with size range 512B - 2KB
        config = FilterConfig(
            min_file_size=512,
            max_file_size=2048,
            include_patterns=["*.txt"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Too small should be rejected
        should_process, reason = await filter_manager.should_process_file(too_small)
        assert should_process is False
        assert "size_limit_exceeded" in reason

        # Just right should be accepted
        should_process, reason = await filter_manager.should_process_file(just_right)
        assert should_process is True

        # Too large should be rejected
        should_process, reason = await filter_manager.should_process_file(too_large)
        assert should_process is False
        assert "size_limit_exceeded" in reason

        # Should have 2 size rejections
        assert filter_manager.statistics.size_rejections == 2

    @pytest.mark.asyncio
    async def test_zero_size_file_handling(self):
        """Test handling of zero-size files."""
        # Create empty file
        empty_file = self.create_test_file("empty.txt", 0)

        # Filter that allows empty files
        config1 = FilterConfig(include_patterns=["*.txt"])
        filter_manager1 = AdvancedFileFilter(config1)

        should_process, reason = await filter_manager1.should_process_file(empty_file)
        assert should_process is True

        # Filter that requires minimum size
        config2 = FilterConfig(
            min_file_size=1,
            include_patterns=["*.txt"]
        )
        filter_manager2 = AdvancedFileFilter(config2)

        should_process, reason = await filter_manager2.should_process_file(empty_file)
        assert should_process is False
        assert "size_limit_exceeded" in reason

    @pytest.mark.asyncio
    async def test_file_size_check_error_handling(self):
        """Test error handling during file size checks."""
        # Create a file that exists
        test_file = self.create_test_file("test.txt", 1024)

        # Mock stat() to raise an exception
        with patch.object(Path, 'stat', side_effect=OSError("Permission denied")):
            config = FilterConfig(include_patterns=["*.txt"])
            filter_manager = AdvancedFileFilter(config)

            should_process, reason = await filter_manager.should_process_file(test_file)
            assert should_process is False
            assert "size_check_failed" in reason


class TestAdvancedFileFilterContentFiltering:
    """Test content-based filtering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_file(self, name: str, content: str) -> Path:
        """Create a test file with specific content."""
        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.asyncio
    async def test_content_filtering_matching(self):
        """Test content filtering with matching content."""
        # Create files with different content
        python_file = self.create_test_file("script.py", "import os\nprint('hello world')")
        javascript_file = self.create_test_file("app.js", "function main() { console.log('hello'); }")
        text_file = self.create_test_file("readme.txt", "This is documentation text")

        # Filter for files containing programming keywords
        config = FilterConfig(
            content_filters=["import", "function"],
            enable_content_filtering=True,
            include_patterns=["*"]  # Accept all file types for content check
        )
        filter_manager = AdvancedFileFilter(config)

        # Python file should be accepted (contains 'import')
        should_process, reason = await filter_manager.should_process_file(python_file)
        assert should_process is True
        assert "matched_content" in reason

        # JavaScript file should be accepted (contains 'function')
        should_process, reason = await filter_manager.should_process_file(javascript_file)
        assert should_process is True
        assert "matched_content" in reason

        # Text file should be rejected (no matching content)
        should_process, reason = await filter_manager.should_process_file(text_file)
        assert should_process is False
        assert "no_content_match" in reason

        # Verify content check statistics
        assert filter_manager.statistics.content_checks_performed == 3

    @pytest.mark.asyncio
    async def test_content_filtering_disabled(self):
        """Test behavior when content filtering is disabled."""
        python_file = self.create_test_file("script.py", "import os\nprint('hello')")

        # Content filtering disabled
        config = FilterConfig(
            content_filters=["import", "function"],
            enable_content_filtering=False,  # Disabled
            include_patterns=["*.py"]
        )
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(python_file)
        assert should_process is True
        assert "accepted" in reason

        # No content checks should have been performed
        assert filter_manager.statistics.content_checks_performed == 0

    @pytest.mark.asyncio
    async def test_empty_content_handling(self):
        """Test handling of files with empty content."""
        empty_file = self.create_test_file("empty.py", "")

        config = FilterConfig(
            content_filters=["import"],
            enable_content_filtering=True,
            include_patterns=["*.py"]
        )
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(empty_file)
        assert should_process is False
        assert "empty_content" in reason

    @pytest.mark.asyncio
    async def test_content_sample_size_limiting(self):
        """Test content sampling with size limits."""
        # Create file with large content
        large_content = "import os\n" + "# comment line\n" * 1000
        large_file = self.create_test_file("large.py", large_content)

        # Small sample size
        config = FilterConfig(
            content_filters=["import"],
            content_sample_size=100,  # Small sample
            enable_content_filtering=True,
            include_patterns=["*.py"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Should still find 'import' in the sample
        should_process, reason = await filter_manager.should_process_file(large_file)
        assert should_process is True
        assert "matched_content" in reason

    @pytest.mark.asyncio
    async def test_content_filtering_case_insensitive(self):
        """Test case-insensitive content filtering."""
        mixed_case_file = self.create_test_file("mixed.py", "IMPORT os\nPrint('hello')")

        config = FilterConfig(
            content_filters=["import", "print"],  # lowercase filters
            enable_content_filtering=True,
            include_patterns=["*.py"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Should match despite case differences (content is converted to lowercase)
        should_process, reason = await filter_manager.should_process_file(mixed_case_file)
        assert should_process is True
        assert "matched_content" in reason

    @pytest.mark.asyncio
    async def test_content_filtering_error_handling(self):
        """Test error handling during content filtering."""
        # Create a file
        test_file = self.create_test_file("test.py", "import os")

        # Mock file reading to raise an exception
        original_read = filter_manager._read_content_sample if 'filter_manager' in locals() else None

        config = FilterConfig(
            content_filters=["import"],
            enable_content_filtering=True,
            include_patterns=["*.py"]
        )
        filter_manager = AdvancedFileFilter(config)

        with patch.object(filter_manager, '_read_content_sample', side_effect=Exception("Read error")):
            should_process, reason = await filter_manager.should_process_file(test_file)
            # Should fall back to accepting the file when content check fails
            assert should_process is True
            assert "content_check_error" in reason

    @pytest.mark.asyncio
    async def test_binary_file_content_filtering(self):
        """Test content filtering with binary files."""
        # Create a binary file (will cause encoding errors)
        binary_file = self.temp_dir / "binary.dat"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')

        config = FilterConfig(
            content_filters=["import"],
            enable_content_filtering=True,
            include_patterns=["*.dat"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Should handle gracefully (content reading uses 'ignore' errors)
        should_process, reason = await filter_manager.should_process_file(binary_file)
        assert should_process is False  # No matching content
        assert "no_content_match" in reason or "empty_content" in reason


class TestAdvancedFileFilterMimeTypes:
    """Test MIME type filtering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_file(self, name: str, content: str = "test") -> Path:
        """Create a test file."""
        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.asyncio
    async def test_allowed_mime_types(self):
        """Test allowed MIME types filtering."""
        # Create files of different types
        python_file = self.create_test_file("script.py", "print('hello')")
        text_file = self.create_test_file("readme.txt", "documentation")
        html_file = self.create_test_file("index.html", "<html></html>")

        # Allow only Python and text files
        config = FilterConfig(
            allowed_mime_types={"text/x-python", "text/plain"},
            enable_mime_type_detection=True,
            include_patterns=["*"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Python file should be accepted
        should_process, reason = await filter_manager.should_process_file(python_file)
        assert should_process is True

        # Text file should be accepted
        should_process, reason = await filter_manager.should_process_file(text_file)
        assert should_process is True

        # HTML file should be rejected
        should_process, reason = await filter_manager.should_process_file(html_file)
        assert should_process is False
        assert "mime_type_blocked" in reason
        assert filter_manager.statistics.mime_type_rejections == 1

    @pytest.mark.asyncio
    async def test_blocked_mime_types(self):
        """Test blocked MIME types filtering."""
        # Create files of different types
        python_file = self.create_test_file("script.py", "print('hello')")
        text_file = self.create_test_file("readme.txt", "documentation")

        # Block text files
        config = FilterConfig(
            blocked_mime_types={"text/plain"},
            enable_mime_type_detection=True,
            include_patterns=["*"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Python file should be accepted
        should_process, reason = await filter_manager.should_process_file(python_file)
        assert should_process is True

        # Text file should be rejected
        should_process, reason = await filter_manager.should_process_file(text_file)
        assert should_process is False
        assert "mime_type_blocked" in reason

    @pytest.mark.asyncio
    async def test_mime_type_detection_disabled(self):
        """Test behavior when MIME type detection is disabled."""
        html_file = self.create_test_file("index.html", "<html></html>")

        config = FilterConfig(
            blocked_mime_types={"text/html"},
            enable_mime_type_detection=False,  # Disabled
            include_patterns=["*.html"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Should be accepted because MIME type checking is disabled
        should_process, reason = await filter_manager.should_process_file(html_file)
        assert should_process is True
        assert filter_manager.statistics.mime_type_rejections == 0

    @pytest.mark.asyncio
    async def test_unknown_mime_type_handling(self):
        """Test handling of files with unknown MIME types."""
        # Create file with unusual extension
        weird_file = self.create_test_file("file.xyz123", "unknown content")

        config = FilterConfig(
            allowed_mime_types={"text/plain"},
            enable_mime_type_detection=True,
            include_patterns=["*"]
        )
        filter_manager = AdvancedFileFilter(config)

        # Should be rejected (unknown type not in allowed list)
        should_process, reason = await filter_manager.should_process_file(weird_file)
        assert should_process is False
        assert "mime_type_blocked" in reason


class TestAdvancedFileFilterPerformance:
    """Test performance and optimization features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_files(self, count: int, prefix: str = "test") -> List[Path]:
        """Create multiple test files."""
        files = []
        for i in range(count):
            file_path = self.create_test_file(f"{prefix}_{i}.py", f"# Test file {i}\nprint('hello {i}')")
            files.append(file_path)
        return files

    def create_test_file(self, name: str, content: str) -> Path:
        """Create a single test file."""
        file_path = self.temp_dir / name
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.asyncio
    async def test_pattern_cache_performance(self):
        """Test pattern cache performance benefits."""
        # Create many files
        test_files = self.create_test_files(50, "performance_test")

        # Filter with pattern caching
        config = FilterConfig(
            include_patterns=[r".*\.py$"],
            regex_cache_size=100
        )
        filter_manager = AdvancedFileFilter(config)

        start_time = time.perf_counter()

        # Process all files
        for test_file in test_files:
            await filter_manager.should_process_file(test_file)

        processing_time = time.perf_counter() - start_time

        # Should have good cache hit rate
        cache_stats = filter_manager.pattern_cache.get_stats()
        assert cache_stats["cached_patterns"] > 0
        assert filter_manager.statistics.regex_cache_hits > 0

        # Should be reasonably fast
        assert processing_time < 5.0  # Should process 50 files in under 5 seconds

    @pytest.mark.asyncio
    async def test_concurrent_content_filtering(self):
        """Test concurrent content filtering performance."""
        # Create files with different content
        files_with_content = []
        for i in range(10):
            content = f"import module_{i}\nprint('test {i}')" if i % 2 == 0 else f"no imports here {i}"
            file_path = self.create_test_file(f"concurrent_{i}.py", content)
            files_with_content.append(file_path)

        config = FilterConfig(
            content_filters=["import"],
            enable_content_filtering=True,
            max_concurrent_content_checks=5,
            include_patterns=["*.py"]
        )
        filter_manager = AdvancedFileFilter(config)

        start_time = time.perf_counter()

        # Process all files concurrently
        tasks = [
            filter_manager.should_process_file(file_path)
            for file_path in files_with_content
        ]
        results = await asyncio.gather(*tasks)

        processing_time = time.perf_counter() - start_time

        # Should have processed files with content filtering
        accepted_count = sum(1 for result, _ in results if result)
        assert accepted_count == 5  # Half the files have "import"
        assert filter_manager.statistics.content_checks_performed == 10

        # Should be reasonably fast with concurrency
        assert processing_time < 2.0

    @pytest.mark.asyncio
    async def test_throughput_statistics(self):
        """Test throughput statistics calculation."""
        test_files = self.create_test_files(20, "throughput_test")

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        # Process files and measure throughput
        start_time = time.perf_counter()
        for test_file in test_files:
            await filter_manager.should_process_file(test_file)
        total_time = time.perf_counter() - start_time

        stats = filter_manager.get_statistics()

        # Check throughput calculation
        assert stats.throughput_files_per_sec > 0
        assert stats.avg_processing_time_ms > 0

        # Throughput should be reasonable
        expected_throughput = len(test_files) / total_time
        assert abs(stats.throughput_files_per_sec - expected_throughput) < expected_throughput * 0.5

    def test_memory_efficiency_with_large_cache(self):
        """Test memory efficiency with large pattern cache."""
        # Create filter with large cache
        config = FilterConfig(regex_cache_size=10000)
        filter_manager = AdvancedFileFilter(config)

        # Add many patterns to cache
        patterns = [f".*\\.{ext}$" for ext in ["py", "js", "ts", "java", "cpp", "c", "h", "hpp"]]
        patterns.extend([f"test_{i}_.*\\.py$" for i in range(100)])

        # Pre-compile patterns
        for pattern in patterns:
            filter_manager.pattern_cache.get_pattern(pattern)

        cache_stats = filter_manager.pattern_cache.get_stats()
        assert cache_stats["cached_patterns"] > 100

        # Should handle large cache efficiently
        assert len(filter_manager.pattern_cache.patterns) <= filter_manager.config.regex_cache_size

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        test_files = self.create_test_files(10, "monitoring_test")

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        # Process files
        for test_file in test_files:
            await filter_manager.should_process_file(test_file)

        # Get performance report
        report = filter_manager.get_performance_report()

        assert "filtering_statistics" in report
        assert "pattern_cache_statistics" in report
        assert "configuration" in report
        assert "performance_insights" in report

        # Check insights
        insights = report["performance_insights"]
        assert "cache_hit_rate" in insights
        assert "processing_efficiency" in insights
        assert insights["cache_hit_rate"] >= 0
        assert insights["processing_efficiency"] >= 0


class TestConvenienceFilters:
    """Test convenience filter creation functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def create_test_file(self, name: str, content: str = "test") -> Path:
        """Create a test file."""
        file_path = self.temp_dir / name
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.asyncio
    async def test_code_file_filter(self):
        """Test code file filter convenience function."""
        # Create various file types
        python_file = self.create_test_file("script.py", "import os")
        javascript_file = self.create_test_file("app.js", "function main() {}")
        java_file = self.create_test_file("Main.java", "class Main {}")
        image_file = self.create_test_file("image.jpg", "binary data")
        cache_file = self.create_test_file("__pycache__/cached.pyc", "bytecode")

        filter_manager = create_code_file_filter()

        # Code files should be accepted
        should_process, _ = await filter_manager.should_process_file(python_file)
        assert should_process is True

        should_process, _ = await filter_manager.should_process_file(javascript_file)
        assert should_process is True

        should_process, _ = await filter_manager.should_process_file(java_file)
        assert should_process is True

        # Non-code files should be rejected
        should_process, _ = await filter_manager.should_process_file(image_file)
        assert should_process is False

        # Cache files should be excluded
        should_process, _ = await filter_manager.should_process_file(cache_file)
        assert should_process is False

    @pytest.mark.asyncio
    async def test_document_filter(self):
        """Test document filter convenience function."""
        # Create various document types
        pdf_file = self.create_test_file("document.pdf", "PDF content")
        markdown_file = self.create_test_file("readme.md", "# Documentation")
        text_file = self.create_test_file("notes.txt", "Plain text")
        code_file = self.create_test_file("script.py", "import os")
        temp_file = self.create_test_file("temp.tmp", "temporary data")

        filter_manager = create_document_filter()

        # Document files should be accepted
        should_process, _ = await filter_manager.should_process_file(pdf_file)
        assert should_process is True

        should_process, _ = await filter_manager.should_process_file(markdown_file)
        assert should_process is True

        should_process, _ = await filter_manager.should_process_file(text_file)
        assert should_process is True

        # Non-document files should be rejected
        should_process, _ = await filter_manager.should_process_file(code_file)
        assert should_process is False

        # Temporary files should be excluded
        should_process, _ = await filter_manager.should_process_file(temp_file)
        assert should_process is False


class TestAdvancedFileFilterEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    @pytest.mark.asyncio
    async def test_symlink_handling(self):
        """Test symlink handling with different configurations."""
        # Create a regular file and a symlink
        regular_file = self.temp_dir / "regular.txt"
        regular_file.write_text("regular content")

        symlink_file = self.temp_dir / "symlink.txt"
        try:
            symlink_file.symlink_to(regular_file)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this system")

        # Filter that follows symlinks
        config1 = FilterConfig(
            include_patterns=["*.txt"],
            follow_symlinks=True
        )
        filter_manager1 = AdvancedFileFilter(config1)

        should_process, _ = await filter_manager1.should_process_file(symlink_file)
        assert should_process is True

        # Filter that doesn't follow symlinks
        config2 = FilterConfig(
            include_patterns=["*.txt"],
            follow_symlinks=False
        )
        filter_manager2 = AdvancedFileFilter(config2)

        should_process, reason = await filter_manager2.should_process_file(symlink_file)
        assert should_process is False
        assert "symlink_blocked" in reason

    @pytest.mark.asyncio
    async def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a file
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test content")

        # Mock permission check to raise PermissionError
        with patch.object(Path, 'stat', side_effect=PermissionError("Access denied")):
            config = FilterConfig(
                include_patterns=["*.txt"],
                check_file_permissions=True
            )
            filter_manager = AdvancedFileFilter(config)

            should_process, reason = await filter_manager.should_process_file(test_file)
            assert should_process is False
            assert "permission_denied" in reason

    @pytest.mark.asyncio
    async def test_permission_check_disabled(self):
        """Test behavior when permission checking is disabled."""
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test content")

        # Disable permission checking
        config = FilterConfig(
            include_patterns=["*.txt"],
            check_file_permissions=False
        )
        filter_manager = AdvancedFileFilter(config)

        # Should not check permissions
        should_process, _ = await filter_manager.should_process_file(test_file)
        assert should_process is True

    @pytest.mark.asyncio
    async def test_filter_with_no_criteria(self):
        """Test filter behavior with no filtering criteria."""
        test_file = self.temp_dir / "any_file.xyz"
        test_file.write_text("any content")

        # Empty configuration (no patterns, no size limits, etc.)
        config = FilterConfig()
        filter_manager = AdvancedFileFilter(config)

        should_process, reason = await filter_manager.should_process_file(test_file)
        assert should_process is True
        assert "no_include_patterns" in reason or "accepted" in reason

    @pytest.mark.asyncio
    async def test_invalid_file_path_handling(self):
        """Test handling of invalid file paths."""
        invalid_paths = [
            "",  # Empty string
            "/dev/null/invalid",  # Invalid path structure
            "\0invalid",  # Null byte in path
        ]

        config = FilterConfig(include_patterns=["*"])
        filter_manager = AdvancedFileFilter(config)

        for invalid_path in invalid_paths:
            try:
                should_process, reason = await filter_manager.should_process_file(invalid_path)
                # Should handle gracefully
                assert should_process is False
            except Exception as e:
                # Some platforms may raise exceptions for invalid paths
                assert isinstance(e, (ValueError, OSError))

    @pytest.mark.asyncio
    async def test_unicode_file_names(self):
        """Test handling of Unicode file names."""
        # Create files with Unicode names
        unicode_file = self.temp_dir / "测试文件.py"
        unicode_file.write_text("print('hello unicode')", encoding='utf-8')

        emoji_file = self.temp_dir / "🐍_script.py"
        emoji_file.write_text("print('hello emoji')", encoding='utf-8')

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        # Should handle Unicode filenames
        should_process, _ = await filter_manager.should_process_file(unicode_file)
        assert should_process is True

        should_process, _ = await filter_manager.should_process_file(emoji_file)
        assert should_process is True

    @pytest.mark.asyncio
    async def test_extremely_long_file_paths(self):
        """Test handling of extremely long file paths."""
        # Create a very long path
        long_dir = self.temp_dir
        for i in range(10):  # Create nested directories
            long_dir = long_dir / f"very_long_directory_name_{i}"
        long_dir.mkdir(parents=True, exist_ok=True)

        long_file = long_dir / ("very_long_filename_" * 10 + ".py")
        try:
            long_file.write_text("test content")
        except OSError:
            pytest.skip("System doesn't support very long paths")

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        # Should handle long paths
        should_process, _ = await filter_manager.should_process_file(long_file)
        assert should_process is True

    @pytest.mark.asyncio
    async def test_concurrent_filter_updates(self):
        """Test thread-safety of filter configuration updates."""
        test_file = self.temp_dir / "test.py"
        test_file.write_text("import os")

        config = FilterConfig(include_patterns=[r".*\.py$"])
        filter_manager = AdvancedFileFilter(config)

        async def process_files():
            """Process files while config is being updated."""
            results = []
            for _ in range(10):
                result = await filter_manager.should_process_file(test_file)
                results.append(result)
                await asyncio.sleep(0.01)  # Small delay
            return results

        async def update_config():
            """Update configuration during processing."""
            await asyncio.sleep(0.05)  # Let some processing happen first
            new_config = FilterConfig(include_patterns=[r".*\.txt$"])
            filter_manager.update_config(new_config)

        # Run both operations concurrently
        process_task = asyncio.create_task(process_files())
        update_task = asyncio.create_task(update_config())

        results, _ = await asyncio.gather(process_task, update_task)

        # Should not crash and should have processed files
        assert len(results) == 10
        assert all(isinstance(result, tuple) for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])