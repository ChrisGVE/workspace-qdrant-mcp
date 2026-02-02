-- Legacy ingestion_queue schema
-- Note: This is a legacy table from Phase 1. New code should use unified_queue.
-- Maintained for backward compatibility with existing tests.

CREATE TABLE IF NOT EXISTS ingestion_queue (
    file_absolute_path TEXT PRIMARY KEY NOT NULL,
    collection_name TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    branch TEXT DEFAULT 'main',
    operation TEXT NOT NULL DEFAULT 'ingest'
        CHECK (operation IN ('ingest', 'update', 'delete')),
    priority INTEGER NOT NULL DEFAULT 5
        CHECK (priority >= 0 AND priority <= 10),
    queued_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_from TEXT,  -- Added by migration, timestamp for exponential backoff
    error_message_id TEXT,
    collection_type TEXT  -- 'code', 'notes', 'lib', 'memory'
);

-- Index for efficient dequeue operations (priority DESC, queued_timestamp ASC)
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_dequeue
    ON ingestion_queue(priority DESC, queued_timestamp ASC);

-- Index for filtering by tenant
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_tenant
    ON ingestion_queue(tenant_id);

-- Index for retry_from filtering (skip items with future retry timestamps)
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_retry_from
    ON ingestion_queue(retry_from);

-- Index for collection type filtering
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection_type
    ON ingestion_queue(collection_type);

-- Messages table for error tracking and failed operations
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_type TEXT NOT NULL,
    error_message TEXT,
    file_path TEXT,
    collection_name TEXT,
    created_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Index for message cleanup (purge old messages)
CREATE INDEX IF NOT EXISTS idx_messages_timestamp
    ON messages(created_timestamp);
