-- Migration: Convert retry_from from foreign key to timestamp
-- Schema Version: 4.1
-- Purpose: Enable exponential backoff retry scheduling in queue processor
--
-- This migration changes retry_from from a TEXT foreign key (referencing file_absolute_path)
-- to a TEXT timestamp field (RFC3339 format) that stores when an item should be retried.
--
-- Note: SQLite doesn't support DROP FOREIGN KEY, so we need to recreate the table

-- =============================================================================
-- BACKUP AND RECREATE ingestion_queue TABLE
-- =============================================================================

-- Step 1: Create new table with updated schema
CREATE TABLE IF NOT EXISTS ingestion_queue_new (
    -- Primary identifier: absolute file path ensures uniqueness per file
    file_absolute_path TEXT PRIMARY KEY NOT NULL,

    -- Collection and tenant information
    collection_name TEXT NOT NULL,
    tenant_id TEXT DEFAULT 'default',
    branch TEXT DEFAULT 'main',

    -- Operation type with validation
    operation TEXT NOT NULL CHECK (operation IN ('ingest', 'update', 'delete')),

    -- Priority-based processing (0=lowest, 10=highest)
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 0 AND 10),

    -- Timestamp tracking
    queued_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Retry management
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_from TEXT,  -- RFC3339 timestamp for retry scheduling (was: foreign key reference)

    -- Error tracking reference
    error_message_id INTEGER,

    -- Foreign key constraint (removed retry_from foreign key)
    FOREIGN KEY (error_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

-- Step 2: Copy data from old table (retry_from values will be NULL as they were file paths)
INSERT INTO ingestion_queue_new (
    file_absolute_path, collection_name, tenant_id, branch,
    operation, priority, queued_timestamp, retry_count, error_message_id
)
SELECT
    file_absolute_path, collection_name, tenant_id, branch,
    operation, priority, queued_timestamp, retry_count, error_message_id
FROM ingestion_queue;

-- Step 3: Drop old table
DROP TABLE ingestion_queue;

-- Step 4: Rename new table to original name
ALTER TABLE ingestion_queue_new RENAME TO ingestion_queue;

-- =============================================================================
-- RECREATE INDEXES
-- =============================================================================

-- Performance indexes for queue operations
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_priority_time
    ON ingestion_queue(priority DESC, queued_timestamp ASC);

-- New index for retry_from timestamp filtering
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_retry_timestamp
    ON ingestion_queue(retry_from) WHERE retry_from IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection
    ON ingestion_queue(collection_name, tenant_id, branch);

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_operation
    ON ingestion_queue(operation);

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_error
    ON ingestion_queue(error_message_id) WHERE error_message_id IS NOT NULL;

-- =============================================================================
-- VERIFICATION QUERY (for testing)
-- =============================================================================
-- SELECT sql FROM sqlite_master WHERE name = 'ingestion_queue';
