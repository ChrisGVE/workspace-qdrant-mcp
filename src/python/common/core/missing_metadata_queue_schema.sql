-- Missing Metadata Queue Schema Extension
-- Schema Version: 4.1
-- Purpose: Tracks files that cannot be processed due to missing tools (LSP, Tree-sitter, etc.)
--          Enables retry when tools become available

-- =============================================================================
-- MISSING METADATA QUEUE TABLE (Task 352, Subtask 4)
-- =============================================================================

-- Queue for items waiting for tool availability
-- Items are moved here from ingestion_queue when required tools are unavailable
CREATE TABLE IF NOT EXISTS missing_metadata_queue (
    -- Primary identifier: matches original queue_id format
    queue_id TEXT PRIMARY KEY NOT NULL,

    -- File and collection information (preserved from original queue item)
    file_absolute_path TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    branch TEXT NOT NULL,

    -- Operation type with validation
    operation TEXT NOT NULL CHECK (operation IN ('ingest', 'update', 'delete')),

    -- Priority preserved from original queue (0=lowest, 10=highest)
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 0 AND 10),

    -- Missing tools information (JSON array of MissingTool objects)
    -- Example: [{"LspServer": {"language": "rust"}}, {"TreeSitterParser": {"language": "python"}}]
    missing_tools TEXT NOT NULL,  -- JSON array

    -- Timestamp tracking
    queued_timestamp TEXT NOT NULL,  -- ISO 8601 format
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_check_timestamp TEXT,  -- ISO 8601 format - last time we checked tool availability

    -- Optional metadata (preserved from original queue item)
    metadata TEXT,  -- JSON

    -- Ensure uniqueness per file/tenant/branch combination
    UNIQUE(file_absolute_path, tenant_id, branch)
);

-- Performance indexes for missing metadata queue operations
CREATE INDEX IF NOT EXISTS idx_missing_metadata_priority
    ON missing_metadata_queue(priority DESC, queued_timestamp ASC);

CREATE INDEX IF NOT EXISTS idx_missing_metadata_tenant
    ON missing_metadata_queue(tenant_id, branch);

CREATE INDEX IF NOT EXISTS idx_missing_metadata_file_path
    ON missing_metadata_queue(file_absolute_path);

CREATE INDEX IF NOT EXISTS idx_missing_metadata_retry_count
    ON missing_metadata_queue(retry_count);

CREATE INDEX IF NOT EXISTS idx_missing_metadata_last_check
    ON missing_metadata_queue(last_check_timestamp)
    WHERE last_check_timestamp IS NOT NULL;

-- =============================================================================
-- MISSING METADATA STATISTICS VIEW
-- =============================================================================

-- Convenience view for monitoring missing metadata queue
CREATE VIEW IF NOT EXISTS missing_metadata_statistics AS
SELECT
    COUNT(*) AS total_items,
    SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) AS urgent_items,
    SUM(CASE WHEN priority >= 5 AND priority < 8 THEN 1 ELSE 0 END) AS high_priority_items,
    SUM(CASE WHEN priority >= 3 AND priority < 5 THEN 1 ELSE 0 END) AS normal_priority_items,
    SUM(CASE WHEN priority < 3 THEN 1 ELSE 0 END) AS low_priority_items,
    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) AS retry_items,
    COUNT(DISTINCT collection_name) AS unique_collections,
    COUNT(DISTINCT tenant_id) AS unique_tenants,
    MIN(queued_timestamp) AS oldest_item,
    MAX(queued_timestamp) AS newest_item
FROM missing_metadata_queue;

-- =============================================================================
-- MISSING TOOLS SUMMARY VIEW
-- =============================================================================

-- Aggregated view of missing tools to identify most common blockers
-- Note: This requires JSON extraction which may not work in all SQLite versions
-- For full functionality, use SQLite 3.38.0+ with JSON1 extension
CREATE VIEW IF NOT EXISTS missing_tools_summary AS
SELECT
    collection_name,
    COUNT(*) AS affected_files,
    MIN(queued_timestamp) AS oldest_affected,
    AVG(retry_count) AS avg_retry_count
FROM missing_metadata_queue
GROUP BY collection_name
ORDER BY affected_files DESC;

-- =============================================================================
-- NOTES
-- =============================================================================

-- This schema extends the main queue_schema.sql with missing metadata tracking.
-- Items flow: ingestion_queue -> (missing tools detected) -> missing_metadata_queue -> (tools available) -> ingestion_queue
--
-- The queue_id is typically generated as: sha256(file_absolute_path || tenant_id || branch || timestamp)
-- This ensures uniqueness while allowing easy correlation with original queue items.
--
-- The UNIQUE constraint on (file_absolute_path, tenant_id, branch) prevents duplicate entries
-- for the same file, which could occur if:
-- 1. File is re-added to ingestion_queue while already in missing_metadata_queue
-- 2. Multiple processes try to move the same file simultaneously
--
-- Retry mechanism (to be implemented in separate task):
-- - Periodic background job checks items in missing_metadata_queue
-- - For each item, check if missing tools are now available
-- - If available, move item back to ingestion_queue
-- - Update retry_count and last_check_timestamp
-- - Implement exponential backoff for retry intervals
