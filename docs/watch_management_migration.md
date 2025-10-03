# Watch Management Migration Guide

**Date:** October 3, 2025
**Version:** v0.2.1dev1
**Migration:** gRPC-based FileWatcher → SQLite-based Watch Management

## Overview

This guide documents the architectural migration from gRPC-based file watching to SQLite-based watch management in workspace-qdrant-mcp. The new architecture provides better performance, simpler deployment, and improved crash resistance.

## Architecture Comparison

### Before: gRPC Architecture

```
┌─────────────┐                    ┌──────────────┐
│   CLI/MCP   │───── gRPC ────────▶│ Rust Daemon  │
│             │   AddWatch/        │              │
│             │   RemoveWatch      │              │
└─────────────┘                    └──────────────┘
      │                                    │
      │                                    │
      ▼                                    ▼
┌─────────────┐                    ┌──────────────┐
│ FileWatcher │                    │ File Watcher │
│  (Python)   │                    │   (Rust)     │
│  watchdog   │                    │              │
└─────────────┘                    └──────────────┘
```

**Issues:**
- Network overhead from gRPC calls
- Multiple file watchers (Python + Rust)
- Complex coordination between processes
- No crash recovery for watch configurations
- gRPC server adds deployment complexity

### After: SQLite Architecture

```
┌─────────────┐
│   CLI/MCP   │
│             │
└──────┬──────┘
       │
       │ Direct SQLite writes
       ▼
┌─────────────────────────────────┐
│      SQLite Database            │
│    (watch_folders table)        │
│    WAL mode, ACID guarantees    │
└──────┬──────────────────────────┘
       │
       │ Polls for changes
       ▼
┌──────────────┐
│ Rust Daemon  │
│              │
│ Single File  │
│   Watcher    │
└──────────────┘
```

**Benefits:**
- No network overhead (direct SQLite access)
- Single high-performance Rust file watcher
- Simple architecture (no gRPC coordination)
- Crash recovery via SQLite WAL mode
- ACID guarantees for configuration changes
- Easier deployment (no gRPC server needed)

## Key Components

### WatchFolderConfig Dataclass

The `WatchFolderConfig` dataclass defines watch folder configuration:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

@dataclass
class WatchFolderConfig:
    """Configuration for persistent watch folders."""

    # Required fields
    watch_id: str                      # Unique identifier for this watch
    path: str                          # Directory to watch
    collection: str                    # Target Qdrant collection
    patterns: List[str]                # File patterns to match (e.g., ["*.py", "*.md"])
    ignore_patterns: List[str]         # Patterns to exclude (e.g., ["*.pyc", "__pycache__/*"])

    # Optional fields with defaults
    auto_ingest: bool = True           # Automatically ingest matching files
    recursive: bool = True             # Watch subdirectories
    recursive_depth: int = 10          # Maximum recursion depth
    debounce_seconds: float = 2.0      # Wait time before processing changes
    enabled: bool = True               # Enable/disable this watch

    # Auto-populated fields
    created_at: datetime = None        # Creation timestamp (auto-set)
    updated_at: datetime = None        # Last update timestamp (auto-set)
    last_scan: Optional[datetime] = None  # Last scan timestamp
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
```

### SQLiteStateManager API

The `SQLiteStateManager` class provides the API for managing watch folders:

```python
from workspace_qdrant_mcp.core.sqlite_state_manager import (
    SQLiteStateManager,
    WatchFolderConfig
)

class SQLiteStateManager:
    """Manages state persistence in SQLite database."""

    async def initialize(self) -> None:
        """Initialize database and create tables."""

    async def save_watch_folder_config(
        self,
        watch_id: str,
        config: WatchFolderConfig
    ) -> None:
        """Save or update watch folder configuration."""

    async def get_watch_folder_config(
        self,
        watch_id: str
    ) -> Optional[WatchFolderConfig]:
        """Retrieve watch folder configuration by ID."""

    async def list_watch_folders(
        self,
        enabled_only: bool = False
    ) -> List[WatchFolderConfig]:
        """List all watch folder configurations."""

    async def remove_watch_folder_config(
        self,
        watch_id: str
    ) -> bool:
        """Remove watch folder configuration."""

    async def cleanup_stale_watches(
        self,
        max_age_days: int = 30
    ) -> int:
        """Remove watch configurations for non-existent paths."""
```

## Migration Examples

### Example 1: Add Watch Folder

**Before (gRPC):**
```python
# Using gRPC client
from workspace_qdrant_mcp.grpc.watch_client import WatchClient

client = WatchClient()
await client.add_watch(
    path="/path/to/project",
    patterns=["*.py"],
    collection="my-project"
)
```

**After (SQLite):**
```python
# Using SQLiteStateManager
from workspace_qdrant_mcp.core.sqlite_state_manager import (
    SQLiteStateManager,
    WatchFolderConfig
)

state_manager = SQLiteStateManager()
await state_manager.initialize()

watch_config = WatchFolderConfig(
    watch_id="my-project-py",
    path="/path/to/project",
    collection="my-project",
    patterns=["*.py"],
    ignore_patterns=["*.pyc", "__pycache__/*"]
)

await state_manager.save_watch_folder_config(
    watch_config.watch_id,
    watch_config
)
```

### Example 2: List Watch Folders

**Before (gRPC):**
```python
client = WatchClient()
watches = await client.list_watches()
```

**After (SQLite):**
```python
state_manager = SQLiteStateManager()
await state_manager.initialize()

# List all watches
all_watches = await state_manager.list_watch_folders()

# List only enabled watches
active_watches = await state_manager.list_watch_folders(enabled_only=True)
```

### Example 3: Remove Watch Folder

**Before (gRPC):**
```python
client = WatchClient()
await client.remove_watch(watch_id="my-project-py")
```

**After (SQLite):**
```python
state_manager = SQLiteStateManager()
await state_manager.initialize()

await state_manager.remove_watch_folder_config("my-project-py")
```

### Example 4: Temporarily Disable Watch

**Before (gRPC):**
```python
# Required removing and re-adding
client = WatchClient()
await client.remove_watch(watch_id="my-project-py")
# ... later ...
await client.add_watch(...)
```

**After (SQLite):**
```python
# Simply toggle enabled flag
state_manager = SQLiteStateManager()
await state_manager.initialize()

config = await state_manager.get_watch_folder_config("my-project-py")
config.enabled = False
await state_manager.save_watch_folder_config(config.watch_id, config)

# Re-enable later
config.enabled = True
await state_manager.save_watch_folder_config(config.watch_id, config)
```

## Database Schema

### watch_folders Table

```sql
CREATE TABLE watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    collection TEXT NOT NULL,
    patterns TEXT NOT NULL,           -- JSON array: ["*.py", "*.md"]
    ignore_patterns TEXT NOT NULL,    -- JSON array: ["*.pyc", "__pycache__/*"]
    auto_ingest INTEGER DEFAULT 1,    -- Boolean (0/1)
    recursive INTEGER DEFAULT 1,      -- Boolean (0/1)
    recursive_depth INTEGER DEFAULT 10,
    debounce_seconds REAL DEFAULT 2.0,
    enabled INTEGER DEFAULT 1,        -- Boolean (0/1)
    created_at TEXT NOT NULL,         -- ISO 8601 timestamp
    updated_at TEXT NOT NULL,         -- ISO 8601 timestamp
    last_scan TEXT,                   -- ISO 8601 timestamp
    metadata TEXT                     -- JSON object
);

CREATE INDEX idx_watch_folders_enabled ON watch_folders(enabled);
CREATE INDEX idx_watch_folders_path ON watch_folders(path);
```

### Data Types

- **TEXT fields:** Stored as JSON when containing arrays/objects
- **INTEGER booleans:** 0 = False, 1 = True
- **REAL:** Floating-point numbers
- **Timestamps:** ISO 8601 format in UTC

## How the Rust Daemon Detects Changes

The Rust daemon uses a polling mechanism to detect configuration changes:

1. **Periodic Polling:** Every 5 seconds (configurable), the daemon queries the `watch_folders` table
2. **Change Detection:** Compares `updated_at` timestamps with last known state
3. **Configuration Update:** When changes detected:
   - Parse JSON patterns and metadata
   - Update internal watch configuration
   - Reconfigure file system watchers
4. **File Events:** When files change:
   - Check against enabled watch configurations
   - Match against patterns/ignore_patterns
   - Add matching files to `ingestion_queue` table

## Performance Characteristics

### SQLite WAL Mode

The database uses Write-Ahead Logging (WAL) mode for:
- **Concurrent Access:** Multiple readers don't block writers
- **Crash Recovery:** Uncommitted transactions are rolled back on restart
- **Better Performance:** Writes are appended to WAL file, not main database
- **ACID Guarantees:** All operations are atomic, consistent, isolated, durable

### Polling Overhead

- **Database Queries:** ~1ms per poll (5-second interval = 0.02% CPU)
- **Memory:** Watch configurations cached in Rust daemon
- **File System:** Only modified watches trigger watcher reconfiguration

### Comparison to gRPC

| Metric | gRPC | SQLite |
|--------|------|--------|
| Network overhead | TCP/gRPC | None (local file) |
| Latency per operation | 5-20ms | <1ms |
| Concurrent access | gRPC locks | WAL concurrent reads |
| Crash recovery | Manual state rebuild | Automatic (WAL) |
| Deployment complexity | gRPC server + client | SQLite file only |

## Migration Checklist

- [x] Remove Python FileWatcher components
- [x] Remove gRPC watch service (server and client)
- [x] Implement SQLiteStateManager with watch_folders table
- [x] Update Rust daemon to poll SQLite for configuration
- [x] Implement SQLite-based file watching in Rust
- [x] Update CLI commands to use SQLiteStateManager
- [x] Update tests to use SQLite architecture
- [x] Document new architecture in CLAUDE.md
- [x] Create this migration guide

## Troubleshooting

### Watch Not Triggering

1. **Check if watch is enabled:**
   ```python
   config = await state_manager.get_watch_folder_config("my-watch-id")
   print(f"Enabled: {config.enabled}")
   ```

2. **Verify path exists:**
   ```python
   from pathlib import Path
   print(f"Path exists: {Path(config.path).exists()}")
   ```

3. **Check patterns:**
   ```python
   print(f"Patterns: {config.patterns}")
   print(f"Ignore patterns: {config.ignore_patterns}")
   ```

4. **Verify Rust daemon is running:**
   ```bash
   uv run wqm service status
   ```

### Configuration Not Updating

1. **Check database connection:**
   ```python
   state_manager = SQLiteStateManager()
   await state_manager.initialize()
   # Should not raise errors
   ```

2. **Verify updated_at timestamp changed:**
   ```python
   config = await state_manager.get_watch_folder_config("my-watch-id")
   print(f"Last updated: {config.updated_at}")
   ```

3. **Check Rust daemon logs:**
   ```bash
   uv run wqm service logs
   ```

### Database Corruption

If SQLite database becomes corrupted:

1. **Stop daemon:**
   ```bash
   uv run wqm service stop
   ```

2. **Backup database:**
   ```bash
   cp ~/.config/workspace-qdrant/state.db ~/.config/workspace-qdrant/state.db.backup
   ```

3. **Check integrity:**
   ```bash
   sqlite3 ~/.config/workspace-qdrant/state.db "PRAGMA integrity_check;"
   ```

4. **Recover or reinitialize:**
   ```python
   # Reinitialize will recreate tables
   state_manager = SQLiteStateManager()
   await state_manager.initialize()
   ```

## Additional Resources

- **CLAUDE.md:** Project documentation with SQLite architecture
- **sqlite_state_manager.py:** Implementation source code
- **tests/test_sqlite_state_manager_comprehensive.py:** Comprehensive test suite
- **First Principles:** FIRST-PRINCIPLES.md for architectural decisions

## Summary

The migration from gRPC to SQLite-based watch management provides:

1. **Simpler Architecture:** Direct database access, no network layer
2. **Better Performance:** Sub-millisecond operations, no gRPC overhead
3. **Improved Reliability:** WAL mode crash recovery, ACID guarantees
4. **Easier Deployment:** No gRPC server to configure or manage
5. **More Features:** Enable/disable watches, metadata storage, cleanup utilities

The new architecture maintains all previous functionality while providing better performance and reliability characteristics.
