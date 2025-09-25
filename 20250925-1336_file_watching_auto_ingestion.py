#!/usr/bin/env python3
"""
File Watching and Auto-Ingestion System for workspace-qdrant-mcp

This module implements a comprehensive file watching system with:
- Real-time file monitoring with platform-specific optimizations
- Priority-based processing queue (MCP active → current project → background)
- Incremental updates with content hash tracking
- Intelligent debouncing for LSP projects
- Cross-platform compatibility and performance optimization

Author: Claude Code
Created: 2025-09-25T13:36:35+02:00
Task: 261 - Build File Watching and Auto-Ingestion System
"""

import asyncio
import hashlib
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import yaml
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """Priority levels for file processing."""
    CRITICAL = 1    # MCP active operations
    HIGH = 2        # Current project files
    MEDIUM = 3      # Background library folders
    LOW = 4         # Background projects


class FileChangeType(Enum):
    """Types of file changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileWatchConfig:
    """Configuration for file watching behavior."""
    debounce_delay: float = 0.5  # seconds
    max_queue_size: int = 10000
    batch_size: int = 100
    hash_algorithm: str = "sha256"
    excluded_patterns: List[str] = field(default_factory=lambda: [
        "*.tmp", "*.log", "*.cache", "__pycache__", ".git", "node_modules",
        "*.pyc", "*.pyo", "*.lock", ".DS_Store", "Thumbs.db"
    ])
    included_extensions: Set[str] = field(default_factory=lambda: {
        ".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".cpp",
        ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".kt", ".scala",
        ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".xml", ".html",
        ".css", ".scss", ".less", ".sql", ".dockerfile", ".sh", ".bat"
    })
    max_file_size_mb: int = 50


@dataclass
class FileChangeEvent:
    """Represents a file change event with metadata."""
    path: Path
    change_type: FileChangeType
    timestamp: float
    priority: ProcessingPriority
    content_hash: Optional[str] = None
    file_size: Optional[int] = None
    project_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize additional fields after creation."""
        if self.path.exists() and self.path.is_file():
            try:
                stat_info = self.path.stat()
                self.file_size = stat_info.st_size
                if self.file_size <= self.get_max_hash_size():
                    self.content_hash = self._calculate_hash()
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not access file {self.path}: {e}")

    def _calculate_hash(self) -> str:
        """Calculate content hash for the file."""
        try:
            hasher = hashlib.sha256()
            with open(self.path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not hash file {self.path}: {e}")
            return None

    def get_max_hash_size(self) -> int:
        """Get maximum file size for hashing (50MB)."""
        return 50 * 1024 * 1024

    def should_process(self, config: FileWatchConfig) -> bool:
        """Determine if this file should be processed."""
        # Check file size
        if self.file_size and self.file_size > config.max_file_size_mb * 1024 * 1024:
            return False

        # Check extension
        if self.path.suffix.lower() not in config.included_extensions:
            return False

        # Check excluded patterns
        for pattern in config.excluded_patterns:
            if self.path.match(pattern):
                return False

        return True


class PriorityQueue:
    """Thread-safe priority queue for file change events."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {priority: deque() for priority in ProcessingPriority}
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self._total_size = 0

    def put(self, event: FileChangeEvent, timeout: Optional[float] = None) -> bool:
        """Add event to priority queue with timeout."""
        with self.not_full:
            if timeout is None:
                while self._total_size >= self.max_size:
                    self.not_full.wait()
            else:
                end_time = time.time() + timeout
                while self._total_size >= self.max_size:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return False
                    self.not_full.wait(remaining)

            self.queues[event.priority].append(event)
            self._total_size += 1
            self.not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[FileChangeEvent]:
        """Get highest priority event from queue."""
        with self.not_empty:
            if timeout is None:
                while self._total_size == 0:
                    self.not_empty.wait()
            else:
                end_time = time.time() + timeout
                while self._total_size == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    self.not_empty.wait(remaining)

            # Get from highest priority queue with items
            for priority in ProcessingPriority:
                if self.queues[priority]:
                    event = self.queues[priority].popleft()
                    self._total_size -= 1
                    self.not_full.notify()
                    return event

            return None

    def size(self) -> int:
        """Get total queue size."""
        with self.lock:
            return self._total_size

    def clear(self):
        """Clear all queues."""
        with self.lock:
            for queue in self.queues.values():
                queue.clear()
            self._total_size = 0
            self.not_full.notify_all()


class DebouncingManager:
    """Manages debouncing of file change events."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.pending_events: Dict[Path, float] = {}
        self.lock = threading.RLock()
        self.timer_threads: Dict[Path, threading.Timer] = {}

    def should_process(self, event: FileChangeEvent) -> bool:
        """Check if event should be processed immediately or debounced."""
        with self.lock:
            path = event.path
            now = time.time()

            # Cancel existing timer for this path
            if path in self.timer_threads:
                self.timer_threads[path].cancel()

            # Update last seen time
            self.pending_events[path] = now

            # For critical priority, process immediately
            if event.priority == ProcessingPriority.CRITICAL:
                return True

            # For LSP project files, use debouncing
            if self._is_lsp_project(event.project_path):
                return False  # Will be processed after debounce delay

            return True

    def _is_lsp_project(self, project_path: Optional[Path]) -> bool:
        """Check if project has LSP configuration."""
        if not project_path or not project_path.exists():
            return False

        # Check for common LSP config files
        lsp_indicators = [
            ".vscode/settings.json",
            "pyrightconfig.json",
            "pyproject.toml",
            "tsconfig.json",
            "package.json",
            "Cargo.toml",
            "go.mod"
        ]

        for indicator in lsp_indicators:
            if (project_path / indicator).exists():
                return True

        return False

    def schedule_debounced_processing(self, event: FileChangeEvent, callback: Callable):
        """Schedule debounced processing for an event."""
        with self.lock:
            path = event.path

            def process_after_delay():
                with self.lock:
                    # Check if this is still the latest event for this path
                    if (path in self.pending_events and
                        time.time() - self.pending_events[path] >= self.delay):
                        callback(event)
                        self.pending_events.pop(path, None)
                        self.timer_threads.pop(path, None)

            timer = threading.Timer(self.delay, process_after_delay)
            self.timer_threads[path] = timer
            timer.start()

    def cleanup_expired(self):
        """Clean up expired pending events."""
        with self.lock:
            now = time.time()
            expired_paths = [
                path for path, timestamp in self.pending_events.items()
                if now - timestamp > self.delay * 2
            ]

            for path in expired_paths:
                self.pending_events.pop(path, None)
                if path in self.timer_threads:
                    self.timer_threads[path].cancel()
                    self.timer_threads.pop(path, None)


class ContentHashTracker:
    """Tracks content hashes for incremental updates."""

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or Path.cwd() / ".file_watch_cache.json"
        self.hashes: Dict[str, str] = {}
        self.lock = threading.RLock()
        self.load_cache()

    def load_cache(self):
        """Load hash cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.hashes = json.load(f)
                logger.info(f"Loaded {len(self.hashes)} cached file hashes")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load hash cache: {e}")
                self.hashes = {}

    def save_cache(self):
        """Save hash cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.hashes, f, indent=2)
        except OSError as e:
            logger.warning(f"Could not save hash cache: {e}")

    def has_changed(self, event: FileChangeEvent) -> bool:
        """Check if file content has actually changed."""
        if not event.content_hash:
            return True  # Assume changed if we can't calculate hash

        with self.lock:
            path_key = str(event.path.resolve())
            cached_hash = self.hashes.get(path_key)

            if cached_hash != event.content_hash:
                self.hashes[path_key] = event.content_hash
                return True

            return False

    def remove_file(self, path: Path):
        """Remove file from hash cache."""
        with self.lock:
            path_key = str(path.resolve())
            self.hashes.pop(path_key, None)

    def cleanup_missing_files(self):
        """Remove non-existent files from cache."""
        with self.lock:
            existing_paths = []
            for path_str in self.hashes.keys():
                if Path(path_str).exists():
                    existing_paths.append(path_str)

            removed_count = len(self.hashes) - len(existing_paths)
            self.hashes = {path: self.hashes[path] for path in existing_paths}

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} missing files from cache")


class ProjectDetector:
    """Detects project boundaries and determines priorities."""

    def __init__(self):
        self.project_roots: Dict[Path, ProcessingPriority] = {}
        self.current_project: Optional[Path] = None
        self.mcp_active_paths: Set[Path] = set()

    def detect_project_root(self, file_path: Path) -> Optional[Path]:
        """Detect the project root for a given file."""
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:
            # Check for common project indicators
            indicators = [
                ".git", "pyproject.toml", "package.json", "Cargo.toml",
                "go.mod", "pom.xml", "build.gradle", "CMakeLists.txt",
                "Makefile", "requirements.txt", "setup.py"
            ]

            for indicator in indicators:
                if (current / indicator).exists():
                    return current

            current = current.parent

        return None

    def determine_priority(self, file_path: Path) -> ProcessingPriority:
        """Determine processing priority for a file."""
        file_path = file_path.resolve()

        # Check if file is in MCP active path
        for active_path in self.mcp_active_paths:
            try:
                active_resolved = active_path.resolve()
                if file_path.is_relative_to(active_resolved):
                    return ProcessingPriority.CRITICAL
            except (ValueError, AttributeError):
                continue

        # Check if file is in current project
        if self.current_project:
            try:
                current_resolved = self.current_project.resolve()
                if file_path.is_relative_to(current_resolved):
                    return ProcessingPriority.HIGH
            except (ValueError, AttributeError):
                pass

        # Check configured project priorities
        for project_root, priority in self.project_roots.items():
            try:
                project_resolved = project_root.resolve()
                if file_path.is_relative_to(project_resolved):
                    return priority
            except (ValueError, AttributeError):
                continue

        return ProcessingPriority.LOW

    def set_current_project(self, project_path: Path):
        """Set the current active project."""
        self.current_project = project_path.resolve()
        logger.info(f"Set current project to: {self.current_project}")

    def add_mcp_active_path(self, path: Path):
        """Add path to MCP active monitoring."""
        self.mcp_active_paths.add(path.resolve())

    def remove_mcp_active_path(self, path: Path):
        """Remove path from MCP active monitoring."""
        self.mcp_active_paths.discard(path.resolve())


class FileWatchEventHandler(FileSystemEventHandler):
    """Handles file system events from watchdog."""

    def __init__(self, file_watcher: 'FileWatchingSystem'):
        self.file_watcher = file_watcher

    def on_any_event(self, event: FileSystemEvent):
        """Handle any file system event."""
        if event.is_directory:
            return

        try:
            path = Path(event.src_path)
            change_type = self._get_change_type(event.event_type)

            if change_type:
                self.file_watcher._handle_file_change(path, change_type)

        except Exception as e:
            logger.error(f"Error handling file event: {e}")

    def _get_change_type(self, event_type: str) -> Optional[FileChangeType]:
        """Convert watchdog event type to our change type."""
        mapping = {
            'created': FileChangeType.CREATED,
            'modified': FileChangeType.MODIFIED,
            'deleted': FileChangeType.DELETED,
            'moved': FileChangeType.MOVED
        }
        return mapping.get(event_type)


class FileWatchingSystem:
    """Main file watching and auto-ingestion system."""

    def __init__(self, config: Optional[FileWatchConfig] = None):
        self.config = config or FileWatchConfig()
        self.priority_queue = PriorityQueue(self.config.max_queue_size)
        self.debouncer = DebouncingManager(self.config.debounce_delay)
        self.hash_tracker = ContentHashTracker()
        self.project_detector = ProjectDetector()

        # Watchdog components
        self.observer = Observer()
        self.event_handler = FileWatchEventHandler(self)
        self.watched_paths: Set[Path] = set()

        # Processing
        self.processor_executor = ThreadPoolExecutor(max_workers=4)
        self.processing_callbacks: List[Callable[[FileChangeEvent], None]] = []
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

    def add_processing_callback(self, callback: Callable[[FileChangeEvent], None]):
        """Add a callback for processing file change events."""
        self.processing_callbacks.append(callback)

    def add_watch_path(self, path: Path, recursive: bool = True,
                      priority: ProcessingPriority = ProcessingPriority.MEDIUM):
        """Add a path to watch for file changes."""
        path = path.resolve()

        if path in self.watched_paths:
            logger.info(f"Path already being watched: {path}")
            return

        try:
            self.observer.schedule(
                self.event_handler,
                str(path),
                recursive=recursive
            )

            self.watched_paths.add(path)
            self.project_detector.project_roots[path] = priority

            logger.info(f"Added watch path: {path} (priority: {priority.name})")

        except OSError as e:
            logger.error(f"Failed to add watch path {path}: {e}")

    def remove_watch_path(self, path: Path):
        """Remove a path from watching."""
        path = path.resolve()

        if path not in self.watched_paths:
            return

        # Find and unschedule the watch
        for watch in self.observer.emitters:
            if Path(watch.watch.path) == path:
                self.observer.unschedule(watch.watch)
                break

        self.watched_paths.discard(path)
        self.project_detector.project_roots.pop(path, None)

        logger.info(f"Removed watch path: {path}")

    def start(self):
        """Start the file watching system."""
        if self.is_running:
            logger.warning("File watching system is already running")
            return

        self.is_running = True
        self.observer.start()

        # Start worker thread for processing events
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()

        logger.info("File watching system started")

    def stop(self):
        """Stop the file watching system."""
        if not self.is_running:
            return

        self.is_running = False
        self.observer.stop()
        self.observer.join()

        # Save hash cache before stopping
        self.hash_tracker.save_cache()

        logger.info("File watching system stopped")

    def _handle_file_change(self, path: Path, change_type: FileChangeType):
        """Handle a file change event."""
        try:
            # Determine project and priority
            project_path = self.project_detector.detect_project_root(path)
            priority = self.project_detector.determine_priority(path)

            # Create event
            event = FileChangeEvent(
                path=path,
                change_type=change_type,
                timestamp=time.time(),
                priority=priority,
                project_path=project_path
            )

            # Check if we should process this file
            if not event.should_process(self.config):
                return

            # Check for content changes (skip for deletions)
            if change_type != FileChangeType.DELETED:
                if not self.hash_tracker.has_changed(event):
                    logger.debug(f"No content change detected for {path}")
                    return
            else:
                self.hash_tracker.remove_file(path)

            # Handle debouncing
            if self.debouncer.should_process(event):
                self._queue_event(event)
            else:
                self.debouncer.schedule_debounced_processing(
                    event, self._queue_event
                )

        except Exception as e:
            logger.error(f"Error handling file change for {path}: {e}")

    def _queue_event(self, event: FileChangeEvent):
        """Queue an event for processing."""
        try:
            if not self.priority_queue.put(event, timeout=1.0):
                logger.warning(f"Failed to queue event - queue full: {event.path}")
        except Exception as e:
            logger.error(f"Error queuing event: {e}")

    def _process_events(self):
        """Worker thread for processing queued events."""
        logger.info("Event processing worker started")

        while self.is_running:
            try:
                event = self.priority_queue.get(timeout=1.0)
                if event:
                    self._process_single_event(event)

            except Exception as e:
                logger.error(f"Error in event processing worker: {e}")

        logger.info("Event processing worker stopped")

    def _process_single_event(self, event: FileChangeEvent):
        """Process a single file change event."""
        try:
            logger.debug(f"Processing {event.change_type.value}: {event.path}")

            # Call all registered callbacks
            for callback in self.processing_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in processing callback: {e}")

        except Exception as e:
            logger.error(f"Error processing event {event.path}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "watched_paths": len(self.watched_paths),
            "queue_size": self.priority_queue.size(),
            "cached_hashes": len(self.hash_tracker.hashes),
            "is_running": self.is_running,
            "current_project": str(self.project_detector.current_project)
                             if self.project_detector.current_project else None,
            "mcp_active_paths": [str(p) for p in self.project_detector.mcp_active_paths]
        }

    def cleanup(self):
        """Perform cleanup operations."""
        self.hash_tracker.cleanup_missing_files()
        self.debouncer.cleanup_expired()
        self.hash_tracker.save_cache()


# Example usage and integration
def example_callback(event: FileChangeEvent):
    """Example callback for processing file events."""
    print(f"Processing {event.change_type.value}: {event.path}")
    print(f"Priority: {event.priority.name}")
    print(f"Content hash: {event.content_hash}")
    print(f"Project: {event.project_path}")
    print("---")


def main():
    """Example usage of the file watching system."""
    # Create and configure the system
    config = FileWatchConfig(
        debounce_delay=0.5,
        max_queue_size=5000,
        batch_size=50
    )

    watcher = FileWatchingSystem(config)

    # Add processing callback
    watcher.add_processing_callback(example_callback)

    # Set current project (highest priority)
    current_project = Path.cwd()
    watcher.project_detector.set_current_project(current_project)

    # Add watch paths with different priorities
    watcher.add_watch_path(current_project, recursive=True, priority=ProcessingPriority.HIGH)

    # Add library paths with lower priority
    library_paths = [
        Path.home() / "Documents" / "code",
        Path.home() / "Projects"
    ]

    for lib_path in library_paths:
        if lib_path.exists():
            watcher.add_watch_path(lib_path, recursive=True, priority=ProcessingPriority.MEDIUM)

    try:
        # Start watching
        watcher.start()

        print("File watching started. Press Ctrl+C to stop...")
        print(f"Stats: {watcher.get_stats()}")

        # Keep running
        while True:
            time.sleep(10)
            stats = watcher.get_stats()
            print(f"Queue size: {stats['queue_size']}, Cached: {stats['cached_hashes']}")

    except KeyboardInterrupt:
        print("\nStopping file watcher...")
    finally:
        watcher.stop()
        watcher.cleanup()


if __name__ == "__main__":
    main()