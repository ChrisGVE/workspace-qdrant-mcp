
"""
Watch state synchronization and persistence system.

This module provides real-time synchronization between configuration changes
and active watchers, with file locking, event notifications, and atomic operations
to ensure consistency across concurrent access.
"""

import asyncio
import fcntl
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set
from weakref import WeakSet

logger = logging.getLogger(__name__)


@dataclass
class ConfigChangeEvent:
    """Event representing a configuration change."""
    
    event_type: str  # added, modified, removed, status_changed
    watch_id: str
    old_config: Optional[Dict[str, Any]] = None
    new_config: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "unknown"  # mcp_tool, recovery, health_monitor, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type,
            "watch_id": self.watch_id,
            "old_config": self.old_config,
            "new_config": self.new_config,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class FileLockManager:
    """Manages file locking for configuration files."""
    
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.lock_file = config_file.with_suffix('.lock')
        self._lock_fd: Optional[int] = None
        self._lock_timeout = 30  # seconds
    
    @asynccontextmanager
    async def acquire_lock(self) -> AsyncGenerator[None, None]:
        """Acquire exclusive lock on configuration file."""
        try:
            await self._acquire_lock()
            yield
        finally:
            await self._release_lock()
    
    async def _acquire_lock(self) -> None:
        """Acquire file lock with timeout."""
        lock_acquired = False
        start_time = time.time()
        
        while not lock_acquired and (time.time() - start_time) < self._lock_timeout:
            try:
                # Create lock file
                self._lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                
                # Write lock information
                lock_info = {
                    "pid": os.getpid(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config_file": str(self.config_file)
                }
                
                os.write(self._lock_fd, json.dumps(lock_info).encode())
                lock_acquired = True
                
                logger.debug(f"Acquired config file lock: {self.lock_file}")
                
            except FileExistsError:
                # Lock file exists, check if it's stale
                if await self._is_stale_lock():
                    logger.warning("Removing stale lock file")
                    try:
                        self.lock_file.unlink()
                    except FileNotFoundError:
                        pass  # Already removed
                else:
                    # Wait and retry
                    await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error acquiring lock: {e}")
                await asyncio.sleep(0.1)
        
        if not lock_acquired:
            raise TimeoutError(f"Could not acquire lock on {self.config_file} within {self._lock_timeout} seconds")
    
    async def _release_lock(self) -> None:
        """Release file lock."""
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
                self._lock_fd = None
            except Exception as e:
                logger.error(f"Error closing lock file descriptor: {e}")
        
        try:
            self.lock_file.unlink()
            logger.debug(f"Released config file lock: {self.lock_file}")
        except FileNotFoundError:
            pass  # Already removed
        except Exception as e:
            logger.error(f"Error removing lock file: {e}")
    
    async def _is_stale_lock(self) -> bool:
        """Check if lock file is stale (process no longer exists)."""
        try:
            if not self.lock_file.exists():
                return False
            
            # Read lock information
            with open(self.lock_file, 'r') as f:
                lock_info = json.load(f)
            
            pid = lock_info.get("pid")
            if not pid:
                return True  # Invalid lock file
            
            # Check if process still exists
            try:
                os.kill(pid, 0)  # Send null signal to test if process exists
                return False  # Process exists, lock is not stale
            except OSError:
                return True  # Process doesn't exist, lock is stale
        
        except Exception as e:
            logger.warning(f"Error checking stale lock: {e}")
            return True  # Assume stale if we can't determine


class WatchEventNotifier:
    """Event notification system for watch configuration changes."""
    
    def __init__(self):
        self.subscribers: WeakSet[Callable[[ConfigChangeEvent], None]] = WeakSet()
        self.event_history: List[ConfigChangeEvent] = []
        self.max_history_size = 1000
        self._notification_queue = asyncio.Queue()
        self._notification_task: Optional[asyncio.Task] = None
        self._running = False
    
    def subscribe(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Subscribe to configuration change events."""
        self.subscribers.add(callback)
        logger.debug(f"Added event subscriber: {callback}")
    
    def unsubscribe(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Unsubscribe from configuration change events."""
        self.subscribers.discard(callback)
        logger.debug(f"Removed event subscriber: {callback}")
    
    async def notify(self, event: ConfigChangeEvent) -> None:
        """Notify all subscribers of a configuration change."""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Queue for async notification
        await self._notification_queue.put(event)
    
    async def start_notifications(self) -> None:
        """Start the notification processing task."""
        if self._running:
            return
        
        self._running = True
        self._notification_task = asyncio.create_task(self._notification_loop())
        logger.info("Started watch event notifications")
    
    async def stop_notifications(self) -> None:
        """Stop the notification processing task."""
        self._running = False
        
        if self._notification_task:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped watch event notifications")
    
    async def _notification_loop(self) -> None:
        """Main notification processing loop."""
        while self._running:
            try:
                # Wait for events with timeout to allow graceful shutdown
                event = await asyncio.wait_for(self._notification_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue  # Normal timeout, check if still running
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _process_event(self, event: ConfigChangeEvent) -> None:
        """Process and distribute an event to subscribers."""
        logger.debug(f"Processing config change event: {event.event_type} for {event.watch_id}")
        
        # Notify all subscribers
        for subscriber in list(self.subscribers):  # Create list to avoid modification during iteration
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber {subscriber}: {e}")
    
    def get_event_history(self, watch_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history with optional filtering."""
        events = self.event_history
        
        if watch_id:
            events = [e for e in events if e.watch_id == watch_id]
        
        # Return most recent events first
        recent_events = events[-limit:]
        recent_events.reverse()
        
        return [event.to_dict() for event in recent_events]


class SynchronizedWatchConfigManager:
    """Enhanced configuration manager with synchronization and event notifications."""
    
    def __init__(self, config_file: Optional[Path] = None, project_dir: Optional[Path] = None):
        from .watch_config import PersistentWatchConfigManager, WatchConfigFile
        
        # Initialize base config manager
        self.base_manager = PersistentWatchConfigManager(config_file, project_dir)
        self.config_file = self.base_manager.config_file
        
        # Initialize synchronization components
        self.lock_manager = FileLockManager(self.config_file)
        self.event_notifier = WatchEventNotifier()
        
        # Cache for in-memory configuration
        self._config_cache: Optional[WatchConfigFile] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_timeout = 5.0  # seconds
        
        logger.info(f"Initialized synchronized config manager: {self.config_file}")
    
    async def initialize(self) -> None:
        """Initialize the synchronized config manager."""
        await self.event_notifier.start_notifications()
        # Load initial configuration into cache
        await self.load_config()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.event_notifier.stop_notifications()
    
    async def load_config(self, force_reload: bool = False) -> Any:
        """Load configuration with caching and synchronization."""
        current_time = time.time()
        
        # Check if we can use cached config
        if (not force_reload and 
            self._config_cache is not None and 
            self._cache_timestamp is not None and 
            (current_time - self._cache_timestamp) < self._cache_timeout):
            return self._config_cache
        
        # Load configuration with file lock
        async with self.lock_manager.acquire_lock():
            config = await self.base_manager.load_config()
            
            # Update cache
            self._config_cache = config
            self._cache_timestamp = current_time
            
            logger.debug(f"Loaded configuration with {len(config.watches)} watches")
            return config
    
    async def save_config(self, config: Any) -> bool:
        """Save configuration with atomic operations and event notifications."""
        async with self.lock_manager.acquire_lock():
            # Save configuration
            success = await self.base_manager.save_config(config)
            
            if success:
                # Update cache
                self._config_cache = config
                self._cache_timestamp = time.time()
                logger.debug("Configuration saved and cache updated")
            
            return success
    
    async def add_watch_config(self, watch_config: Any, source: str = "mcp_tool") -> bool:
        """Add watch configuration with event notification."""
        # Get current config for comparison
        current_config = await self.load_config()
        existing_watch = None
        for watch in current_config.watches:
            if watch.id == watch_config.id:
                existing_watch = watch
                break
        
        # Perform the add operation
        success = await self.base_manager.add_watch_config(watch_config)
        
        if success:
            # Clear cache to force reload on next access
            self._config_cache = None
            
            # Notify subscribers
            event = ConfigChangeEvent(
                event_type="added" if not existing_watch else "modified",
                watch_id=watch_config.id,
                old_config=existing_watch.to_dict() if existing_watch else None,
                new_config=watch_config.to_dict(),
                source=source
            )
            
            await self.event_notifier.notify(event)
            logger.info(f"Watch config {'added' if not existing_watch else 'modified'}: {watch_config.id}")
        
        return success
    
    async def remove_watch_config(self, watch_id: str, source: str = "mcp_tool") -> bool:
        """Remove watch configuration with event notification."""
        # Get current config to capture removed watch
        current_config = await self.load_config()
        removed_watch = None
        for watch in current_config.watches:
            if watch.id == watch_id:
                removed_watch = watch
                break
        
        if not removed_watch:
            return False  # Watch doesn't exist
        
        # Perform the remove operation
        success = await self.base_manager.remove_watch_config(watch_id)
        
        if success:
            # Clear cache
            self._config_cache = None
            
            # Notify subscribers
            event = ConfigChangeEvent(
                event_type="removed",
                watch_id=watch_id,
                old_config=removed_watch.to_dict(),
                new_config=None,
                source=source
            )
            
            await self.event_notifier.notify(event)
            logger.info(f"Watch config removed: {watch_id}")
        
        return success
    
    async def update_watch_config(self, watch_config: Any, source: str = "mcp_tool") -> bool:
        """Update watch configuration with event notification."""
        # Get current config for comparison
        current_config = await self.load_config()
        old_watch = None
        for watch in current_config.watches:
            if watch.id == watch_config.id:
                old_watch = watch
                break
        
        if not old_watch:
            return False  # Watch doesn't exist
        
        # Perform the update operation
        success = await self.base_manager.update_watch_config(watch_config)
        
        if success:
            # Clear cache
            self._config_cache = None
            
            # Notify subscribers
            event = ConfigChangeEvent(
                event_type="modified",
                watch_id=watch_config.id,
                old_config=old_watch.to_dict(),
                new_config=watch_config.to_dict(),
                source=source
            )
            
            await self.event_notifier.notify(event)
            logger.info(f"Watch config updated: {watch_config.id}")
        
        return success
    
    async def update_watch_status(self, watch_id: str, status: str, source: str = "system") -> bool:
        """Update watch status with event notification."""
        watch_config = await self.get_watch_config(watch_id)
        if not watch_config:
            return False
        
        old_status = watch_config.status
        if old_status == status:
            return True  # No change needed
        
        # Update status
        watch_config.status = status
        success = await self.update_watch_config(watch_config, source=source)
        
        if success:
            # Send additional status change event
            event = ConfigChangeEvent(
                event_type="status_changed",
                watch_id=watch_id,
                old_config={"status": old_status},
                new_config={"status": status},
                source=source
            )
            
            await self.event_notifier.notify(event)
            logger.info(f"Watch status changed: {watch_id} ({old_status} -> {status})")
        
        return success
    
    # Delegate remaining methods to base manager
    async def list_watch_configs(self, active_only: bool = False) -> List[Any]:
        """List watch configurations."""
        return await self.base_manager.list_watch_configs(active_only)
    
    async def get_watch_config(self, watch_id: str) -> Optional[Any]:
        """Get specific watch configuration."""
        return await self.base_manager.get_watch_config(watch_id)
    
    async def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all configurations."""
        return await self.base_manager.validate_all_configs()
    
    def get_config_file_path(self) -> Path:
        """Get configuration file path."""
        return self.base_manager.get_config_file_path()
    
    async def backup_config(self, backup_path: Optional[Path] = None) -> bool:
        """Create configuration backup."""
        return await self.base_manager.backup_config(backup_path)
    
    # Event system methods
    def subscribe_to_changes(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Subscribe to configuration change events."""
        self.event_notifier.subscribe(callback)
    
    def unsubscribe_from_changes(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Unsubscribe from configuration change events."""
        self.event_notifier.unsubscribe(callback)
    
    def get_change_history(self, watch_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self.event_notifier.get_event_history(watch_id, limit)
    
    async def force_sync(self) -> None:
        """Force synchronization of all cached data."""
        self._config_cache = None
        self._cache_timestamp = None
        await self.load_config(force_reload=True)
        logger.info("Forced configuration synchronization")