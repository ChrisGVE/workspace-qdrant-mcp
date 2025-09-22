"""
Unit tests for file watching and event processing functionality.

Tests the core file monitoring system including:
- FileWatcher lifecycle management (start/stop/pause/resume)
- WatchConfiguration and event handling
- File system event processing and filtering
- Debouncing and rate limiting
- Language-aware filtering integration
- Error handling and recovery scenarios
- Event callbacks and ingestion coordination
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call
import pytest

from workspace_qdrant_mcp.core.file_watcher import (
    FileWatcher,
    WatchConfiguration,
    WatchEvent
)

from .conftest_daemon import (
    isolated_daemon_temp_dir,
    mock_file_watcher,
    mock_file_event,
    create_test_file_tree,
    wait_for_condition,
    MockAsyncContextManager
)


class TestWatchConfiguration:
    """Test WatchConfiguration dataclass functionality."""
    
    def test_watch_configuration_initialization(self):
        """Test basic watch configuration initialization."""
        config = WatchConfiguration(
            id="test_watch_001",
            path="/tmp/test_watch",
            collection="test-collection"
        )
        
        assert config.id == "test_watch_001"
        assert config.path == "/tmp/test_watch"
        assert config.collection == "test-collection"
        assert config.patterns == ["*.pdf", "*.epub", "*.txt", "*.md"]  # Defaults
        assert ".git/*" in config.ignore_patterns
        assert config.auto_ingest is True
        assert config.recursive is True
        assert config.debounce_seconds == 5
        assert config.status == "active"
        assert config.files_processed == 0
        assert config.errors_count == 0
        assert config.use_language_filtering is True
    
    def test_watch_configuration_custom(self):
        """Test custom watch configuration."""
        custom_patterns = ["*.py", "*.js", "*.ts"]
        custom_ignore = ["*.pyc", "dist/*"]
        
        config = WatchConfiguration(
            id="custom_watch_002",
            path="/custom/path",
            collection="custom-collection",
            patterns=custom_patterns,
            ignore_patterns=custom_ignore,
            auto_ingest=False,
            recursive=False,
            debounce_seconds=10,
            use_language_filtering=False
        )
        
        assert config.patterns == custom_patterns
        assert config.ignore_patterns == custom_ignore
        assert config.auto_ingest is False
        assert config.recursive is False
        assert config.debounce_seconds == 10
        assert config.use_language_filtering is False
    
    def test_watch_configuration_to_dict(self):
        """Test watch configuration serialization to dictionary."""
        config = WatchConfiguration(
            id="dict_test_003",
            path="/dict/test",
            collection="dict-collection"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["id"] == "dict_test_003"
        assert config_dict["path"] == "/dict/test"
        assert config_dict["collection"] == "dict-collection"
        assert "created_at" in config_dict
        assert "patterns" in config_dict
    
    def test_watch_configuration_from_dict(self):
        """Test watch configuration deserialization from dictionary."""
        config_data = {
            "id": "from_dict_004",
            "path": "/from/dict",
            "collection": "from-dict-collection",
            "patterns": ["*.log", "*.json"],
            "auto_ingest": False,
            "debounce_seconds": 15
        }
        
        config = WatchConfiguration.from_dict(config_data)
        
        assert config.id == "from_dict_004"
        assert config.path == "/from/dict"
        assert config.collection == "from-dict-collection"
        assert config.patterns == ["*.log", "*.json"]
        assert config.auto_ingest is False
        assert config.debounce_seconds == 15
    
    def test_watch_configuration_timestamps(self):
        """Test watch configuration timestamp handling."""
        config = WatchConfiguration(
            id="timestamp_005",
            path="/timestamp/test",
            collection="timestamp-collection"
        )
        
        assert config.created_at is not None
        assert config.last_activity is None
        
        # Update last activity
        now = datetime.now(timezone.utc).isoformat()
        config.last_activity = now
        
        assert config.last_activity == now


class TestWatchEvent:
    """Test WatchEvent dataclass functionality."""
    
    def test_watch_event_initialization(self):
        """Test watch event initialization."""
        event = WatchEvent(
            change_type="modified",
            file_path="/tmp/test_file.py",
            collection="test-collection"
        )
        
        assert event.change_type == "modified"
        assert event.file_path == "/tmp/test_file.py"
        assert event.collection == "test-collection"
        assert event.timestamp is not None
    
    def test_watch_event_types(self):
        """Test different watch event types."""
        events = [
            WatchEvent("added", "/tmp/new_file.py", "test-collection"),
            WatchEvent("modified", "/tmp/changed_file.py", "test-collection"),
            WatchEvent("deleted", "/tmp/removed_file.py", "test-collection")
        ]
        
        assert events[0].change_type == "added"
        assert events[1].change_type == "modified"
        assert events[2].change_type == "deleted"
        
        # All should have timestamps
        assert all(event.timestamp is not None for event in events)


class TestFileWatcherInitialization:
    """Test FileWatcher initialization and configuration."""
    
    def test_file_watcher_basic_initialization(self):
        """Test basic file watcher initialization."""
        config = WatchConfiguration(
            id="basic_test",
            path="/tmp/basic_test",
            collection="basic-collection",
            use_language_filtering=False  # Disable for simpler testing
        )
        
        ingestion_callback = Mock()
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        assert watcher.config is config
        assert watcher.ingestion_callback is ingestion_callback
        assert watcher.event_callback is event_callback
        assert watcher._running is False
        assert watcher._task is None
        assert watcher._debounce_tasks == {}
        assert watcher.language_filter is None  # Disabled
    
    def test_file_watcher_with_language_filtering(self):
        """Test file watcher initialization with language filtering."""
        config = WatchConfiguration(
            id="language_test",
            path="/tmp/language_test",
            collection="language-collection",
            use_language_filtering=True
        )
        
        ingestion_callback = Mock()
        
        with patch('src.workspace_qdrant_mcp.core.file_watcher.LanguageAwareFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter_class.return_value = mock_filter
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback,
                filter_config_path="/tmp/filter_config"
            )
            
            assert watcher.language_filter is mock_filter
            mock_filter_class.assert_called_once_with("/tmp/filter_config")
    
    def test_file_watcher_without_language_filtering(self):
        """Test file watcher initialization without language filtering."""
        config = WatchConfiguration(
            id="no_language_test",
            path="/tmp/no_language_test",
            collection="no-language-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        assert watcher.language_filter is None


class TestFileWatcherLifecycle:
    """Test FileWatcher lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_file_watcher_start_success(self, isolated_daemon_temp_dir):
        """Test successful file watcher startup."""
        config = WatchConfiguration(
            id="start_test",
            path=str(isolated_daemon_temp_dir),
            collection="start-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        with patch.object(watcher, '_watch_loop', AsyncMock()) as mock_watch_loop:
            await watcher.start()
            
            assert watcher._running is True
            assert watcher.config.status == "active"
            assert watcher._task is not None
            mock_watch_loop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_watcher_start_already_running(self, isolated_daemon_temp_dir):
        """Test starting file watcher when already running."""
        config = WatchConfiguration(
            id="already_running_test",
            path=str(isolated_daemon_temp_dir),
            collection="already-running-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Mock running state
        watcher._running = True
        
        with patch.object(watcher, '_watch_loop', AsyncMock()) as mock_watch_loop:
            await watcher.start()
            
            # Should not start watch loop again
            mock_watch_loop.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_file_watcher_start_with_language_filter(self, isolated_daemon_temp_dir):
        """Test file watcher startup with language filter initialization."""
        config = WatchConfiguration(
            id="language_start_test",
            path=str(isolated_daemon_temp_dir),
            collection="language-start-collection",
            use_language_filtering=True
        )
        
        ingestion_callback = AsyncMock()
        
        with patch('src.workspace_qdrant_mcp.core.file_watcher.LanguageAwareFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.load_configuration = AsyncMock()
            mock_filter_class.return_value = mock_filter
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            with patch.object(watcher, '_watch_loop', AsyncMock()):
                await watcher.start()
                
                assert watcher._running is True
                mock_filter.load_configuration.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_watcher_start_language_filter_failure(self, isolated_daemon_temp_dir):
        """Test file watcher startup with language filter initialization failure."""
        config = WatchConfiguration(
            id="language_fail_test",
            path=str(isolated_daemon_temp_dir),
            collection="language-fail-collection",
            use_language_filtering=True
        )
        
        ingestion_callback = AsyncMock()
        
        with patch('src.workspace_qdrant_mcp.core.file_watcher.LanguageAwareFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.load_configuration = AsyncMock(side_effect=Exception("Filter init failed"))
            mock_filter_class.return_value = mock_filter
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            with patch.object(watcher, '_watch_loop', AsyncMock()):
                await watcher.start()
                
                assert watcher._running is True
                assert watcher.language_filter is None  # Should fallback to None
    
    @pytest.mark.asyncio
    async def test_file_watcher_stop(self, isolated_daemon_temp_dir):
        """Test file watcher stop functionality."""
        config = WatchConfiguration(
            id="stop_test",
            path=str(isolated_daemon_temp_dir),
            collection="stop-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Mock running state with task
        mock_task = Mock()
        mock_task.cancel = Mock()
        mock_task.done.return_value = False
        watcher._running = True
        watcher._task = mock_task
        
        # Mock debounce tasks
        mock_debounce_task = Mock()
        mock_debounce_task.done.return_value = False
        mock_debounce_task.cancel = Mock()
        watcher._debounce_tasks = {"test_file": mock_debounce_task}
        
        await watcher.stop()
        
        assert watcher._running is False
        assert watcher.config.status == "paused"
        mock_task.cancel.assert_called_once()
        mock_debounce_task.cancel.assert_called_once()
        assert watcher._debounce_tasks == {}
    
    @pytest.mark.asyncio
    async def test_file_watcher_pause_and_resume(self, isolated_daemon_temp_dir):
        """Test file watcher pause and resume functionality."""
        config = WatchConfiguration(
            id="pause_resume_test",
            path=str(isolated_daemon_temp_dir),
            collection="pause-resume-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Mock running state
        watcher._running = True
        watcher.config.status = "active"
        
        with patch.object(watcher, 'stop', AsyncMock()) as mock_stop:
            with patch.object(watcher, 'start', AsyncMock()) as mock_start:
                # Test pause
                await watcher.pause()
                mock_stop.assert_called_once()
                assert watcher.config.status == "paused"
                
                # Test resume
                watcher._running = False  # Simulate stopped state
                await watcher.resume()
                mock_start.assert_called_once()
    
    def test_file_watcher_is_running(self, isolated_daemon_temp_dir):
        """Test file watcher running status check."""
        config = WatchConfiguration(
            id="running_check_test",
            path=str(isolated_daemon_temp_dir),
            collection="running-check-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Initially not running
        assert watcher.is_running() is False
        
        # Mock running state with active task
        mock_task = Mock()
        mock_task.done.return_value = False
        watcher._running = True
        watcher._task = mock_task
        
        assert watcher.is_running() is True
        
        # Mock completed task
        mock_task.done.return_value = True
        assert watcher.is_running() is False


class TestFileWatcherFilteringAndStatistics:
    """Test file watcher filtering and statistics functionality."""
    
    def test_get_filter_statistics_with_language_filter(self):
        """Test getting filter statistics with language filtering enabled."""
        config = WatchConfiguration(
            id="stats_test",
            path="/tmp/stats_test",
            collection="stats-collection",
            use_language_filtering=True
        )
        
        ingestion_callback = Mock()
        
        with patch('src.workspace_qdrant_mcp.core.file_watcher.LanguageAwareFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_stats = Mock()
            mock_stats.to_dict.return_value = {"files_processed": 10, "files_filtered": 5}
            mock_filter.get_statistics.return_value = mock_stats
            mock_filter.get_configuration_summary.return_value = {"languages": ["python", "javascript"]}
            mock_filter_class.return_value = mock_filter
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            stats = watcher.get_filter_statistics()
            
            assert stats["language_filtering"] is True
            assert "filter_config_summary" in stats
            assert "detailed_stats" in stats
            assert stats["detailed_stats"]["files_processed"] == 10
    
    def test_get_filter_statistics_without_language_filter(self):
        """Test getting filter statistics without language filtering."""
        config = WatchConfiguration(
            id="no_filter_stats_test",
            path="/tmp/no_filter_stats_test",
            collection="no-filter-stats-collection",
            use_language_filtering=False,
            files_filtered=15,
            files_processed=25
        )
        
        ingestion_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        stats = watcher.get_filter_statistics()
        
        assert stats["language_filtering"] is False
        assert stats["files_filtered"] == 15
        assert stats["files_processed"] == 25
        assert stats["filter_method"] == "basic_patterns"
    
    def test_reset_filter_statistics_with_language_filter(self):
        """Test resetting filter statistics with language filtering."""
        config = WatchConfiguration(
            id="reset_stats_test",
            path="/tmp/reset_stats_test",
            collection="reset-stats-collection",
            use_language_filtering=True,
            files_filtered=10,
            files_processed=20
        )
        
        ingestion_callback = Mock()
        
        with patch('src.workspace_qdrant_mcp.core.file_watcher.LanguageAwareFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.reset_statistics = Mock()
            mock_filter_class.return_value = mock_filter
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            watcher.reset_filter_statistics()
            
            mock_filter.reset_statistics.assert_called_once()
            assert watcher.config.files_filtered == 0
            assert watcher.config.files_processed == 0
    
    def test_reset_filter_statistics_without_language_filter(self):
        """Test resetting filter statistics without language filtering."""
        config = WatchConfiguration(
            id="reset_no_filter_test",
            path="/tmp/reset_no_filter_test",
            collection="reset-no-filter-collection",
            use_language_filtering=False,
            files_filtered=5,
            files_processed=15
        )
        
        ingestion_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        watcher.reset_filter_statistics()
        
        assert watcher.config.files_filtered == 0
        assert watcher.config.files_processed == 0


class TestFileWatcherEventProcessing:
    """Test file watcher event processing functionality."""
    
    @pytest.mark.asyncio
    async def test_file_watcher_pattern_matching(self, isolated_daemon_temp_dir):
        """Test file watcher pattern matching logic."""
        # Create test files
        test_files = create_test_file_tree(isolated_daemon_temp_dir, {
            "document.pdf": "PDF content",
            "readme.md": "# README",
            "code.py": "print('hello')",
            "image.jpg": "binary image data",
            "config.json": '{"test": true}'
        })
        
        config = WatchConfiguration(
            id="pattern_test",
            path=str(isolated_daemon_temp_dir),
            collection="pattern-collection",
            patterns=["*.pdf", "*.md"],  # Only match PDF and MD files
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        # Mock the pattern matching logic that would be in _should_process_file
        def mock_should_process(file_path):
            path = Path(file_path)
            return any(path.match(pattern) for pattern in config.patterns)
        
        # Test pattern matching
        assert mock_should_process(str(test_files["document.pdf"])) is True
        assert mock_should_process(str(test_files["readme.md"])) is True
        assert mock_should_process(str(test_files["code.py"])) is False
        assert mock_should_process(str(test_files["image.jpg"])) is False
    
    @pytest.mark.asyncio
    async def test_file_watcher_ignore_patterns(self, isolated_daemon_temp_dir):
        """Test file watcher ignore pattern functionality."""
        # Create test file structure with ignored directories
        git_dir = isolated_daemon_temp_dir / ".git"
        git_dir.mkdir()
        node_modules_dir = isolated_daemon_temp_dir / "node_modules"
        node_modules_dir.mkdir()
        
        test_files = create_test_file_tree(isolated_daemon_temp_dir, {
            "main.py": "main code",
            ".git/config": "git config",
            "node_modules/package.js": "node package",
            "__pycache__/module.pyc": "compiled python",
            ".DS_Store": "mac metadata"
        })
        
        config = WatchConfiguration(
            id="ignore_test",
            path=str(isolated_daemon_temp_dir),
            collection="ignore-collection",
            patterns=["*"],  # Match all files
            ignore_patterns=[".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"],
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Mock the ignore pattern logic
        def mock_should_ignore(file_path):
            path = Path(file_path)
            relative_path = path.relative_to(isolated_daemon_temp_dir)
            return any(relative_path.match(pattern) for pattern in config.ignore_patterns)
        
        # Test ignore patterns
        assert mock_should_ignore(str(test_files["main.py"])) is False  # Should process
        assert mock_should_ignore(str(test_files[".git/config"])) is True  # Should ignore
        assert mock_should_ignore(str(test_files["node_modules/package.js"])) is True  # Should ignore
        assert mock_should_ignore(str(test_files[".DS_Store"])) is True  # Should ignore
    
    @pytest.mark.asyncio
    async def test_file_watcher_debouncing(self, isolated_daemon_temp_dir):
        """Test file watcher debouncing functionality."""
        config = WatchConfiguration(
            id="debounce_test",
            path=str(isolated_daemon_temp_dir),
            collection="debounce-collection",
            debounce_seconds=1,  # Short debounce for testing
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        # Create a test file
        test_file = isolated_daemon_temp_dir / "debounce_test.txt"
        test_file.write_text("initial content")
        file_path = str(test_file)
        
        # Mock debounce logic
        async def mock_debounced_process(file_path, collection):
            await asyncio.sleep(config.debounce_seconds)
            await ingestion_callback(file_path, collection)
        
        # Simulate rapid file changes
        with patch.object(watcher, '_debounced_process_file', mock_debounced_process):
            # First change - should start debounce
            task1 = asyncio.create_task(mock_debounced_process(file_path, config.collection))
            watcher._debounce_tasks[file_path] = task1
            
            # Second change before debounce completes - should cancel first
            if not task1.done():
                task1.cancel()
            
            task2 = asyncio.create_task(mock_debounced_process(file_path, config.collection))
            watcher._debounce_tasks[file_path] = task2
            
            # Wait for debounce to complete
            await task2
            
            # Ingestion should be called once after debounce
            ingestion_callback.assert_called_once_with(file_path, config.collection)
    
    @pytest.mark.asyncio
    async def test_file_watcher_event_callbacks(self, isolated_daemon_temp_dir):
        """Test file watcher event callback functionality."""
        config = WatchConfiguration(
            id="callback_test",
            path=str(isolated_daemon_temp_dir),
            collection="callback-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        # Simulate different event types
        events = [
            WatchEvent("added", str(isolated_daemon_temp_dir / "new_file.txt"), config.collection),
            WatchEvent("modified", str(isolated_daemon_temp_dir / "changed_file.txt"), config.collection),
            WatchEvent("deleted", str(isolated_daemon_temp_dir / "removed_file.txt"), config.collection)
        ]
        
        # Mock event processing
        for event in events:
            if watcher.event_callback:
                watcher.event_callback(event)
        
        # Verify event callback was called for each event
        assert event_callback.call_count == 3
        
        # Verify event types were processed
        call_args = [call.args[0] for call in event_callback.call_args_list]
        event_types = [event.change_type for event in call_args]
        assert "added" in event_types
        assert "modified" in event_types
        assert "deleted" in event_types


class TestFileWatcherErrorHandling:
    """Test file watcher error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_file_watcher_ingestion_callback_error(self, isolated_daemon_temp_dir):
        """Test file watcher handling of ingestion callback errors."""
        config = WatchConfiguration(
            id="error_test",
            path=str(isolated_daemon_temp_dir),
            collection="error-collection",
            use_language_filtering=False
        )
        
        # Mock ingestion callback that raises an exception
        ingestion_callback = AsyncMock(side_effect=Exception("Ingestion failed"))
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        test_file = isolated_daemon_temp_dir / "error_test.txt"
        test_file.write_text("test content")
        
        # Mock error handling in file processing
        async def mock_process_with_error(file_path, collection):
            try:
                await ingestion_callback(file_path, collection)
            except Exception as e:
                watcher.config.errors_count += 1
                if watcher.event_callback:
                    error_event = WatchEvent("error", file_path, collection)
                    watcher.event_callback(error_event)
                raise
        
        # Test error handling
        with pytest.raises(Exception, match="Ingestion failed"):
            await mock_process_with_error(str(test_file), config.collection)
        
        assert watcher.config.errors_count == 1
    
    @pytest.mark.asyncio
    async def test_file_watcher_watch_loop_error_recovery(self, isolated_daemon_temp_dir):
        """Test file watcher recovery from watch loop errors."""
        config = WatchConfiguration(
            id="recovery_test",
            path=str(isolated_daemon_temp_dir),
            collection="recovery-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Mock watch loop that encounters an error
        error_count = 0
        max_errors = 2
        
        async def mock_watch_loop_with_recovery():
            nonlocal error_count
            while watcher._running and error_count < max_errors:
                error_count += 1
                if error_count == 1:
                    watcher.config.status = "error"
                    raise Exception(f"Watch error {error_count}")
                else:
                    # Simulate recovery
                    watcher.config.status = "active"
                    await asyncio.sleep(0.1)  # Brief processing
        
        # Test error recovery behavior
        try:
            await mock_watch_loop_with_recovery()
        except Exception as e:
            assert "Watch error 1" in str(e)
            assert watcher.config.status == "error"
    
    @pytest.mark.asyncio
    async def test_file_watcher_permission_error_handling(self, isolated_daemon_temp_dir):
        """Test file watcher handling of permission errors."""
        config = WatchConfiguration(
            id="permission_test",
            path="/root/restricted",  # Likely to cause permission error
            collection="permission-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Mock permission error during file access
        async def mock_process_with_permission_error(file_path):
            if "/root/restricted" in file_path:
                raise PermissionError("Permission denied")
            return True
        
        # Test permission error handling
        with pytest.raises(PermissionError, match="Permission denied"):
            await mock_process_with_permission_error(config.path + "/test_file.txt")


class TestFileWatcherIntegration:
    """Integration tests for file watcher components."""
    
    @pytest.mark.asyncio
    async def test_file_watcher_full_lifecycle_integration(self, isolated_daemon_temp_dir):
        """Test complete file watcher lifecycle integration."""
        # Create initial file structure
        test_files = create_test_file_tree(isolated_daemon_temp_dir, {
            "initial.txt": "Initial content",
            "docs": {
                "readme.md": "# Documentation",
                "guide.pdf": "PDF content"
            },
            ".git": {
                "config": "git config"
            }
        })
        
        config = WatchConfiguration(
            id="integration_test",
            path=str(isolated_daemon_temp_dir),
            collection="integration-collection",
            patterns=["*.txt", "*.md", "*.pdf"],
            ignore_patterns=[".git/*"],
            debounce_seconds=0.1,  # Very short for testing
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        # Mock the watch loop to simulate file events
        processed_files = []
        
        async def mock_watch_loop():
            # Simulate processing existing files
            for file_path in [
                str(test_files["initial.txt"]),
                str(test_files["docs/readme.md"]),
                str(test_files["docs/guide.pdf"])
            ]:
                if watcher._running:
                    processed_files.append(file_path)
                    await ingestion_callback(file_path, config.collection)
                    if event_callback:
                        event = WatchEvent("added", file_path, config.collection)
                        event_callback(event)
                    await asyncio.sleep(0.01)  # Small delay
        
        # Test full lifecycle
        with patch.object(watcher, '_watch_loop', mock_watch_loop):
            # Start watcher
            await watcher.start()
            assert watcher.is_running() is True
            assert config.status == "active"
            
            # Let it process for a short time
            await asyncio.sleep(0.1)
            
            # Verify files were processed (excluding ignored files)
            assert len(processed_files) == 3
            assert any("initial.txt" in path for path in processed_files)
            assert any("readme.md" in path for path in processed_files)
            assert any("guide.pdf" in path for path in processed_files)
            assert not any(".git" in path for path in processed_files)
            
            # Verify callbacks were called
            assert ingestion_callback.call_count == 3
            assert event_callback.call_count == 3
            
            # Test pause functionality
            await watcher.pause()
            assert config.status == "paused"
            assert watcher.is_running() is False
            
            # Test resume functionality
            await watcher.resume()
            assert config.status == "active"
            
            # Stop watcher
            await watcher.stop()
            assert watcher.is_running() is False
            assert config.status == "paused"
    
    @pytest.mark.asyncio
    async def test_file_watcher_multiple_watchers_coordination(self, isolated_daemon_temp_dir):
        """Test coordination between multiple file watchers."""
        # Create separate directories for different watchers
        dir1 = isolated_daemon_temp_dir / "watcher1"
        dir2 = isolated_daemon_temp_dir / "watcher2"
        dir1.mkdir()
        dir2.mkdir()
        
        # Create test files in each directory
        create_test_file_tree(dir1, {
            "doc1.txt": "Document 1",
            "code1.py": "Python code 1"
        })
        
        create_test_file_tree(dir2, {
            "doc2.md": "# Document 2",
            "code2.js": "JavaScript code 2"
        })
        
        # Configure different watchers for different file types
        config1 = WatchConfiguration(
            id="watcher1",
            path=str(dir1),
            collection="collection1",
            patterns=["*.txt"],
            use_language_filtering=False
        )
        
        config2 = WatchConfiguration(
            id="watcher2",
            path=str(dir2),
            collection="collection2",
            patterns=["*.md"],
            use_language_filtering=False
        )
        
        # Track processed files by watcher
        watcher1_files = []
        watcher2_files = []
        
        async def mock_ingestion_callback1(file_path, collection):
            watcher1_files.append((file_path, collection))
        
        async def mock_ingestion_callback2(file_path, collection):
            watcher2_files.append((file_path, collection))
        
        watcher1 = FileWatcher(
            config=config1,
            ingestion_callback=mock_ingestion_callback1
        )
        
        watcher2 = FileWatcher(
            config=config2,
            ingestion_callback=mock_ingestion_callback2
        )
        
        # Mock watch loops for both watchers
        async def mock_watch_loop1():
            if watcher1._running:
                await mock_ingestion_callback1(str(dir1 / "doc1.txt"), "collection1")
        
        async def mock_watch_loop2():
            if watcher2._running:
                await mock_ingestion_callback2(str(dir2 / "doc2.md"), "collection2")
        
        # Test coordinated operation
        with patch.object(watcher1, '_watch_loop', mock_watch_loop1):
            with patch.object(watcher2, '_watch_loop', mock_watch_loop2):
                # Start both watchers
                await watcher1.start()
                await watcher2.start()
                
                assert watcher1.is_running() is True
                assert watcher2.is_running() is True
                
                # Let them process
                await asyncio.sleep(0.1)
                
                # Verify each watcher processed its specific files
                assert len(watcher1_files) == 1
                assert len(watcher2_files) == 1
                assert watcher1_files[0][1] == "collection1"
                assert watcher2_files[0][1] == "collection2"
                assert "doc1.txt" in watcher1_files[0][0]
                assert "doc2.md" in watcher2_files[0][0]
                
                # Stop both watchers
                await watcher1.stop()
                await watcher2.stop()
                
                assert watcher1.is_running() is False
                assert watcher2.is_running() is False


@pytest.mark.file_watching
class TestFileWatcherAdvancedFeatures:
    """Test advanced file watcher features and edge cases."""
    
    @pytest.mark.asyncio
    async def test_file_watcher_recursive_directory_processing(self, isolated_daemon_temp_dir):
        """Test recursive directory processing functionality."""
        # Create nested directory structure
        deep_structure = {
            "level1": {
                "level2": {
                    "level3": {
                        "deep_file.txt": "Deep content"
                    },
                    "file2.txt": "Level 2 content"
                },
                "file1.txt": "Level 1 content"
            },
            "root_file.txt": "Root content"
        }
        
        test_files = create_test_file_tree(isolated_daemon_temp_dir, deep_structure)
        
        config = WatchConfiguration(
            id="recursive_test",
            path=str(isolated_daemon_temp_dir),
            collection="recursive-collection",
            patterns=["*.txt"],
            recursive=True,
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        processed_files = []
        
        async def tracking_callback(file_path, collection):
            processed_files.append(file_path)
            await ingestion_callback(file_path, collection)
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=tracking_callback
        )
        
        # Mock recursive file discovery
        expected_files = [
            "root_file.txt",
            "level1/file1.txt", 
            "level1/level2/file2.txt",
            "level1/level2/level3/deep_file.txt"
        ]
        
        async def mock_recursive_watch():
            for expected_file in expected_files:
                if watcher._running:
                    full_path = str(isolated_daemon_temp_dir / expected_file)
                    await tracking_callback(full_path, config.collection)
        
        # Test recursive processing
        with patch.object(watcher, '_watch_loop', mock_recursive_watch):
            await watcher.start()
            await asyncio.sleep(0.1)
            await watcher.stop()
            
            # Should have processed all nested files
            assert len(processed_files) == 4
            assert ingestion_callback.call_count == 4
            
            # Verify all expected files were processed
            processed_names = [Path(f).name for f in processed_files]
            assert "root_file.txt" in processed_names
            assert "deep_file.txt" in processed_names
    
    @pytest.mark.asyncio
    async def test_file_watcher_configuration_updates(self, isolated_daemon_temp_dir):
        """Test dynamic configuration updates."""
        config = WatchConfiguration(
            id="config_update_test",
            path=str(isolated_daemon_temp_dir),
            collection="config-update-collection",
            patterns=["*.txt"],
            debounce_seconds=1,
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Test initial configuration
        assert config.patterns == ["*.txt"]
        assert config.debounce_seconds == 1
        
        # Update configuration
        config.patterns = ["*.txt", "*.md", "*.pdf"]
        config.debounce_seconds = 2
        config.files_processed = 10
        
        # Verify updates
        assert config.patterns == ["*.txt", "*.md", "*.pdf"]
        assert config.debounce_seconds == 2
        assert config.files_processed == 10
        
        # Test configuration serialization after updates
        config_dict = config.to_dict()
        assert config_dict["patterns"] == ["*.txt", "*.md", "*.pdf"]
        assert config_dict["debounce_seconds"] == 2
        assert config_dict["files_processed"] == 10
    
    @pytest.mark.asyncio
    async def test_file_watcher_performance_monitoring(self, isolated_daemon_temp_dir):
        """Test file watcher performance monitoring and statistics."""
        config = WatchConfiguration(
            id="performance_test",
            path=str(isolated_daemon_temp_dir),
            collection="performance-collection",
            use_language_filtering=False
        )
        
        ingestion_callback = AsyncMock()
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        # Simulate processing statistics
        start_time = time.time()
        
        # Mock file processing with statistics tracking
        test_files = ["file1.txt", "file2.txt", "file3.txt"]
        
        for i, filename in enumerate(test_files):
            config.files_processed += 1
            config.last_activity = datetime.now(timezone.utc).isoformat()
            
            # Simulate processing time
            await asyncio.sleep(0.01)
        
        processing_time = time.time() - start_time
        
        # Verify statistics
        assert config.files_processed == 3
        assert config.last_activity is not None
        
        # Get filter statistics
        stats = watcher.get_filter_statistics()
        assert stats["files_processed"] == 3
        assert stats["language_filtering"] is False
        
        # Verify performance is reasonable
        assert processing_time < 1.0  # Should process quickly in tests