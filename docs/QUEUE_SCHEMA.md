# Unified Queue Schema

This document describes the unified ingestion queue schema that consolidates all queue types into a single table with explicit item_type and operation discriminators.

## Overview

The unified queue replaces multiple legacy queue tables:
- `ingestion_queue` (file-based)
- `content_ingestion_queue` (text/scratchbook)
- `processing_queue` (legacy)
- `missing_metadata_queue`
- `cumulative_deletions_queue`

## Table Structure

### `unified_queue` Table

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `queue_id` | TEXT | PRIMARY KEY, NOT NULL | UUID-style unique identifier |
| `item_type` | TEXT | NOT NULL, CHECK enum | Type discriminator (see Item Types) |
| `op` | TEXT | NOT NULL, CHECK enum | Operation type (see Operations) |
| `tenant_id` | TEXT | NOT NULL | Project identifier (git remote URL hash or path hash) |
| `collection` | TEXT | NOT NULL | Target Qdrant collection name |
| `priority` | INTEGER | NOT NULL, CHECK 0-10 | Processing priority (higher = first) |
| `status` | TEXT | NOT NULL, CHECK enum | Processing status |
| `created_at` | TEXT | NOT NULL | ISO 8601 timestamp |
| `updated_at` | TEXT | NOT NULL | ISO 8601 timestamp |
| `lease_until` | TEXT | NULL | Lease expiry for distributed processing |
| `worker_id` | TEXT | NULL | Identifier of processing worker |
| `idempotency_key` | TEXT | NOT NULL, UNIQUE | Duplicate prevention key |
| `payload_json` | TEXT | NOT NULL | Type-specific payload (JSON) |
| `retry_count` | INTEGER | NOT NULL, DEFAULT 0 | Current retry attempt |
| `max_retries` | INTEGER | NOT NULL, DEFAULT 3 | Maximum retry attempts |
| `error_message` | TEXT | NULL | Last error description |
| `last_error_at` | TEXT | NULL | Timestamp of last error |
| `branch` | TEXT | DEFAULT 'main' | Git branch (for file items) |
| `metadata` | TEXT | DEFAULT '{}' | Extensibility metadata (JSON) |

## Item Types

| Type | Description | Valid Operations |
|------|-------------|------------------|
| `content` | Direct text content (scratchbook, notes, clipboard) | ingest, update, delete |
| `file` | Single file ingestion with path reference | ingest, update, delete |
| `folder` | Folder scan operation (generates child file items) | ingest, delete, scan |
| `project` | Project initialization/scan (top-level container) | ingest, delete, scan |
| `library` | Library documentation ingestion | ingest, update, delete |
| `delete_tenant` | Tenant-wide deletion operation | delete only |
| `delete_document` | Single document deletion by ID | delete only |
| `rename` | File/folder rename tracking | update only |

## Operations

| Operation | Description |
|-----------|-------------|
| `ingest` | Initial ingestion or re-ingestion of content |
| `update` | Update existing content (typically delete + reingest) |
| `delete` | Remove content from vector database |
| `scan` | Scan directory/project without immediate ingestion |

## Item Type and Operation Compatibility Matrix

```
item_type        | ingest | update | delete | scan
-----------------|--------|--------|--------|------
content          |   Y    |   Y    |   Y    |  N
file             |   Y    |   Y    |   Y    |  N
folder           |   Y    |   N    |   Y    |  Y
project          |   Y    |   N    |   Y    |  Y
library          |   Y    |   Y    |   Y    |  N
delete_tenant    |   N    |   N    |   Y    |  N
delete_document  |   N    |   N    |   Y    |  N
rename           |   N    |   Y    |   N    |  N
```

## Status Values

| Status | Description |
|--------|-------------|
| `pending` | Ready to be picked up by processor |
| `in_progress` | Currently being processed (lease acquired) |
| `done` | Successfully completed (can be cleaned up) |
| `failed` | Max retries exceeded |

## Processing Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Enqueue                                  │
│  MCP/CLI → generate idempotency key → INSERT with pending  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dequeue                                  │
│  SELECT WHERE status='pending' ORDER BY priority DESC,     │
│  created_at ASC LIMIT batch_size                           │
│  → UPDATE status='in_progress', lease_until, worker_id     │
└─────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│    Success           │   │    Failure           │
│  UPDATE status='done'│   │  retry_count < max?  │
└──────────────────────┘   └──────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
       ┌──────────────────────┐         ┌──────────────────────┐
       │  Retry               │         │  Failed              │
       │  UPDATE retry_count++│         │  UPDATE status=      │
       │  status='pending'    │         │  'failed'            │
       │  (with backoff)      │         └──────────────────────┘
       └──────────────────────┘
```

## Payload JSON Schemas

### `content` Item

```json
{
  "content": "The actual text content",
  "source_type": "scratchbook|mcp|clipboard",
  "main_tag": "optional-primary-tag",
  "full_tag": "optional/hierarchical/tag"
}
```

### `file` Item

```json
{
  "file_path": "/absolute/path/to/file",
  "file_type": "code|document|config|...",
  "file_hash": "sha256:...",
  "size_bytes": 12345
}
```

### `folder` Item

```json
{
  "folder_path": "/absolute/path/to/folder",
  "recursive": true,
  "recursive_depth": 10,
  "patterns": ["*.py", "*.rs"],
  "ignore_patterns": ["*.pyc", "__pycache__/*"]
}
```

### `project` Item

```json
{
  "project_root": "/absolute/path",
  "git_remote": "https://github.com/...",
  "project_type": "rust|python|mixed"
}
```

### `library` Item

```json
{
  "library_name": "qdrant-client",
  "library_version": "1.15.0",
  "source_url": "https://..."
}
```

### `delete_tenant` Item

```json
{
  "tenant_id_to_delete": "...",
  "reason": "project_removed|user_request"
}
```

### `delete_document` Item

```json
{
  "document_id": "uuid-or-path",
  "point_ids": ["uuid1", "uuid2"]
}
```

### `rename` Item

```json
{
  "old_path": "/old/path",
  "new_path": "/new/path",
  "is_folder": false
}
```

## Idempotency Key Generation

Format: `{item_type}:{collection}:{identifier_hash}`

The identifier_hash is derived from:
- For `content`: SHA256 of content text (first 16 hex chars)
- For `file`: SHA256 of absolute path (first 16 hex chars)
- For `folder`: SHA256 of absolute path (first 16 hex chars)
- For `project`: SHA256 of project root path (first 16 hex chars)
- For `library`: SHA256 of `{library_name}:{version}` (first 16 hex chars)
- For `delete_*`: SHA256 of target identifier (first 16 hex chars)
- For `rename`: SHA256 of `{old_path}:{new_path}` (first 16 hex chars)

## Indexes

| Index Name | Columns | Purpose |
|------------|---------|---------|
| `idx_unified_queue_dequeue` | status, priority DESC, created_at | Fast priority-based dequeue |
| `idx_unified_queue_idempotency` | idempotency_key | Duplicate prevention |
| `idx_unified_queue_lease_expiry` | lease_until (partial: in_progress) | Stale lease detection |
| `idx_unified_queue_collection_tenant` | collection, tenant_id | Per-project queries |
| `idx_unified_queue_item_type` | item_type, status | Type distribution analysis |
| `idx_unified_queue_failed` | status, last_error_at (partial: failed) | Failed item monitoring |
| `idx_unified_queue_worker` | worker_id, status (partial: in_progress) | Worker tracking |

## Priority Guidelines

| Priority | Use Case |
|----------|----------|
| 10 | Immediate MCP request (user waiting) |
| 9 | High-priority CLI operation |
| 8 | Content/scratchbook from MCP |
| 7 | Active project file changes |
| 5 | Normal file watcher events (default) |
| 3 | Background folder scans |
| 1 | Library documentation (low priority) |
| 0 | Deferred/bulk operations |

## Lease Management

Leases prevent duplicate processing in distributed scenarios:

1. **Lease Acquisition**: When dequeuing, set `lease_until` to `now() + lease_duration`
2. **Lease Duration**: Default 5 minutes, configurable per item_type
3. **Lease Renewal**: Long-running operations should renew periodically
4. **Stale Detection**: Items with `status='in_progress'` and `lease_until < now()` are stale
5. **Requeue on Startup**: Daemon startup should requeue all stale items back to `pending`

## Migration from Legacy Queues

See `src/python/common/core/schema/unified_queue_migration.sql` for migration scripts.

Key migration steps:
1. Create unified_queue table
2. Migrate ingestion_queue items (map to file item_type)
3. Migrate content_ingestion_queue items (map to content item_type)
4. Mark legacy tables as deprecated
5. After verification period, drop legacy tables
