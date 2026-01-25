-- Unified Queue Schema (Task 22)
--
-- This schema unifies all ingestion queue types into a single table with
-- explicit item_type and operation enums. It replaces:
--   - ingestion_queue (file-based)
--   - content_ingestion_queue (text/scratchbook)
--   - processing_queue (legacy)
--   - missing_metadata_queue
--   - cumulative_deletions_queue
--
-- Design Principles:
-- 1. Single table for all queue item types with discriminator column
-- 2. Explicit status tracking with lease-based distributed processing
-- 3. Idempotency keys for duplicate prevention
-- 4. Comprehensive retry management with exponential backoff
-- 5. Performance indexes for priority-based dequeue and monitoring

-- =============================================================================
-- Item Type Enum Values
-- =============================================================================
-- content        : Direct text content (scratchbook, notes, clipboard)
-- file           : Single file ingestion with path reference
-- folder         : Folder scan operation (generates child file items)
-- project        : Project initialization/scan (top-level container)
-- library        : Library documentation ingestion
-- delete_tenant  : Tenant-wide deletion operation
-- delete_document: Single document deletion by ID
-- rename         : File/folder rename tracking

-- =============================================================================
-- Operation Enum Values
-- =============================================================================
-- ingest : Initial ingestion or re-ingestion of content
-- update : Update existing content (delete + reingest)
-- delete : Remove content from vector database
-- scan   : Scan directory/project without immediate ingestion

-- =============================================================================
-- Main Unified Queue Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS unified_queue (
    -- Primary identifier: UUID-style unique queue item ID
    queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),

    -- Item type discriminator with strict validation
    -- Determines how payload_json should be interpreted
    item_type TEXT NOT NULL CHECK (item_type IN (
        'content', 'file', 'folder', 'project', 'library',
        'delete_tenant', 'delete_document', 'rename'
    )),

    -- Operation type with strict validation
    -- Not all operations are valid for all item_types (see compatibility matrix below)
    op TEXT NOT NULL CHECK (op IN ('ingest', 'update', 'delete', 'scan')),

    -- Tenant and collection routing
    -- tenant_id: Project identifier (from git remote URL or path hash)
    -- collection: Target Qdrant collection name
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,

    -- Priority-based processing (0=lowest/background, 10=highest/MCP immediate)
    -- Default 5 for normal file watcher items
    -- 8+ for MCP direct operations
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 0 AND priority <= 10),

    -- Processing status with lease management
    -- pending     : Ready to be picked up by processor
    -- in_progress : Currently being processed (lease acquired)
    -- done        : Successfully completed (can be cleaned up)
    -- failed      : Max retries exceeded (moved to dead letter conceptually)
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),

    -- Timestamp tracking
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- Lease management for distributed processing
    -- lease_until: Timestamp when the lease expires (NULL = no lease)
    -- If status='in_progress' and lease_until < now(), item can be reclaimed
    lease_until TEXT,

    -- Worker tracking
    -- worker_id: Identifier of the processor that acquired the lease
    worker_id TEXT,

    -- Idempotency key for duplicate prevention
    -- Format: {item_type}:{collection}:{identifier_hash}
    -- Must be unique to prevent duplicate processing
    idempotency_key TEXT NOT NULL UNIQUE,

    -- Type-specific payload (JSON)
    -- Structure varies by item_type (see payload schemas below)
    payload_json TEXT NOT NULL DEFAULT '{}',

    -- Retry management
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,

    -- Error tracking
    error_message TEXT,
    last_error_at TEXT,

    -- Branch tracking (optional, for file-based items)
    branch TEXT DEFAULT 'main',

    -- Metadata for extensibility (JSON)
    metadata TEXT DEFAULT '{}'
);

-- =============================================================================
-- Performance Indexes
-- =============================================================================

-- Primary dequeue index: pending items by priority (DESC) and age (ASC)
-- This is the most critical index for queue performance
CREATE INDEX IF NOT EXISTS idx_unified_queue_dequeue
    ON unified_queue(status, priority DESC, created_at ASC)
    WHERE status = 'pending';

-- Idempotency lookup (already unique constraint, but explicit index)
CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_queue_idempotency
    ON unified_queue(idempotency_key);

-- Lease expiry check for stale in_progress items
-- Partial index only on in_progress items for efficiency
CREATE INDEX IF NOT EXISTS idx_unified_queue_lease_expiry
    ON unified_queue(lease_until)
    WHERE status = 'in_progress';

-- Collection and tenant filtering for statistics and debugging
CREATE INDEX IF NOT EXISTS idx_unified_queue_collection_tenant
    ON unified_queue(collection, tenant_id);

-- Item type distribution analysis
CREATE INDEX IF NOT EXISTS idx_unified_queue_item_type
    ON unified_queue(item_type, status);

-- Failed items for monitoring and manual review
CREATE INDEX IF NOT EXISTS idx_unified_queue_failed
    ON unified_queue(status, last_error_at DESC)
    WHERE status = 'failed';

-- Worker tracking for debugging stuck items
CREATE INDEX IF NOT EXISTS idx_unified_queue_worker
    ON unified_queue(worker_id, status)
    WHERE status = 'in_progress';

-- =============================================================================
-- Monitoring Views
-- =============================================================================

-- Overall queue statistics
CREATE VIEW IF NOT EXISTS v_queue_stats AS
SELECT
    status,
    item_type,
    COUNT(*) as count,
    AVG(priority) as avg_priority,
    MIN(created_at) as oldest_item,
    MAX(created_at) as newest_item,
    AVG(retry_count) as avg_retries
FROM unified_queue
GROUP BY status, item_type;

-- Per-collection queue depth
CREATE VIEW IF NOT EXISTS v_queue_by_collection AS
SELECT
    collection,
    tenant_id,
    status,
    COUNT(*) as count,
    SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) as high_priority_count,
    MIN(created_at) as oldest_item
FROM unified_queue
WHERE status IN ('pending', 'in_progress')
GROUP BY collection, tenant_id, status;

-- Stale in_progress items (potential stuck jobs)
CREATE VIEW IF NOT EXISTS v_stale_items AS
SELECT
    queue_id,
    item_type,
    op,
    collection,
    tenant_id,
    worker_id,
    lease_until,
    created_at,
    updated_at,
    retry_count
FROM unified_queue
WHERE status = 'in_progress'
  AND lease_until < strftime('%Y-%m-%dT%H:%M:%fZ', 'now');

-- Item type distribution for pattern analysis
CREATE VIEW IF NOT EXISTS v_item_type_distribution AS
SELECT
    item_type,
    op,
    status,
    COUNT(*) as count,
    AVG(CAST((julianday('now') - julianday(created_at)) * 86400 AS INTEGER)) as avg_age_seconds
FROM unified_queue
GROUP BY item_type, op, status;

-- =============================================================================
-- Triggers
-- =============================================================================

-- Auto-update updated_at timestamp on row modification
CREATE TRIGGER IF NOT EXISTS tr_unified_queue_updated_at
AFTER UPDATE ON unified_queue
FOR EACH ROW
BEGIN
    UPDATE unified_queue
    SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    WHERE queue_id = NEW.queue_id;
END;

-- =============================================================================
-- Payload JSON Schemas (documented, not enforced in SQLite)
-- =============================================================================
--
-- item_type='content':
--   {
--     "content": "...",           -- The actual text content
--     "source_type": "scratchbook|mcp|clipboard",
--     "main_tag": "...",          -- Primary categorization tag
--     "full_tag": "..."           -- Full hierarchical tag
--   }
--
-- item_type='file':
--   {
--     "file_path": "/absolute/path/to/file",
--     "file_type": "code|document|config|...",
--     "file_hash": "sha256:...",  -- For change detection
--     "size_bytes": 12345
--   }
--
-- item_type='folder':
--   {
--     "folder_path": "/absolute/path/to/folder",
--     "recursive": true,
--     "recursive_depth": 10,
--     "patterns": ["*.py", "*.rs"],
--     "ignore_patterns": ["*.pyc", "__pycache__/*"]
--   }
--
-- item_type='project':
--   {
--     "project_root": "/absolute/path",
--     "git_remote": "https://github.com/...",
--     "project_type": "rust|python|mixed"
--   }
--
-- item_type='library':
--   {
--     "library_name": "qdrant-client",
--     "library_version": "1.15.0",
--     "source_url": "https://..."
--   }
--
-- item_type='delete_tenant':
--   {
--     "tenant_id_to_delete": "...",
--     "reason": "project_removed|user_request"
--   }
--
-- item_type='delete_document':
--   {
--     "document_id": "uuid-or-path",
--     "point_ids": ["uuid1", "uuid2"]  -- Optional specific point IDs
--   }
--
-- item_type='rename':
--   {
--     "old_path": "/old/path",
--     "new_path": "/new/path",
--     "is_folder": false
--   }
--
-- =============================================================================
-- Item Type and Operation Compatibility Matrix
-- =============================================================================
--
-- item_type        | ingest | update | delete | scan
-- -----------------|--------|--------|--------|------
-- content          |   Y    |   Y    |   Y    |  N
-- file             |   Y    |   Y    |   Y    |  N
-- folder           |   Y    |   N    |   Y    |  Y
-- project          |   Y    |   N    |   Y    |  Y
-- library          |   Y    |   Y    |   Y    |  N
-- delete_tenant    |   N    |   N    |   Y    |  N
-- delete_document  |   N    |   N    |   Y    |  N
-- rename           |   N    |   Y    |   N    |  N
--
-- Note: Invalid combinations should be rejected at enqueue time by application code

-- =============================================================================
-- Migration Support
-- =============================================================================

-- Index for identifying items by legacy queue source (if tracking migration)
CREATE INDEX IF NOT EXISTS idx_unified_queue_migration_source
    ON unified_queue(json_extract(metadata, '$.migrated_from'))
    WHERE json_extract(metadata, '$.migrated_from') IS NOT NULL;
