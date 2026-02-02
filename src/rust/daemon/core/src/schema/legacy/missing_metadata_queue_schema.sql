-- Legacy missing_metadata_queue schema
-- Note: This is a legacy table from Phase 1. New code should use unified_queue.
-- Maintained for backward compatibility with existing tests.

CREATE TABLE IF NOT EXISTS missing_metadata_queue (
    queue_id TEXT PRIMARY KEY NOT NULL,
    file_absolute_path TEXT NOT NULL UNIQUE,
    collection_name TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    branch TEXT DEFAULT 'main',
    operation TEXT NOT NULL DEFAULT 'ingest'
        CHECK (operation IN ('ingest', 'update', 'delete')),
    priority INTEGER NOT NULL DEFAULT 5
        CHECK (priority >= 0 AND priority <= 10),
    missing_tools TEXT NOT NULL,  -- JSON array of missing tool descriptors
    queued_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_check_timestamp TEXT,  -- When tools were last checked
    metadata TEXT  -- Additional JSON metadata
);

-- Index for efficient retrieval (priority DESC, queued_timestamp ASC)
CREATE INDEX IF NOT EXISTS idx_missing_metadata_queue_priority
    ON missing_metadata_queue(priority DESC, queued_timestamp ASC);

-- Index for filtering by tenant
CREATE INDEX IF NOT EXISTS idx_missing_metadata_queue_tenant
    ON missing_metadata_queue(tenant_id);
