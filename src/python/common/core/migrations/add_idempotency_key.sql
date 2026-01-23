-- Migration: Add idempotency_key and status columns to ingestion_queue
-- Task 434: Implement queue idempotency with deterministic keys
-- Schema Version: 5.0
--
-- Purpose: Prevent duplicate ingestion on retries by tracking unique operation keys
-- and processing status (pending, in_progress, done, failed)

-- Add status column with default 'pending'
-- Note: CHECK constraint added via application-level validation for SQLite compatibility
ALTER TABLE ingestion_queue ADD COLUMN status TEXT DEFAULT 'pending';

-- Add idempotency_key column for deterministic deduplication
-- Key format depends on content type:
--   Files: sha256(file_path + mtime + operation)
--   Strings: sha256(content_hash + collection + operation)
--   URLs: sha256(url + operation)
ALTER TABLE ingestion_queue ADD COLUMN idempotency_key TEXT;

-- Add completion timestamp for retention-based cleanup
ALTER TABLE ingestion_queue ADD COLUMN completed_at TIMESTAMP;

-- Add content_hash for string/URL content (null for files which use mtime)
ALTER TABLE ingestion_queue ADD COLUMN content_hash TEXT;

-- Create index on idempotency_key for fast duplicate lookups
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_idempotency_key
    ON ingestion_queue(idempotency_key) WHERE idempotency_key IS NOT NULL;

-- Create index on status for efficient filtering
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_status
    ON ingestion_queue(status);

-- Create index on completed_at for retention cleanup
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_completed_at
    ON ingestion_queue(completed_at) WHERE completed_at IS NOT NULL;

-- Create composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_status_priority
    ON ingestion_queue(status, priority DESC, queued_timestamp ASC);

-- View for monitoring idempotency statistics
CREATE VIEW IF NOT EXISTS idempotency_stats AS
SELECT
    status,
    COUNT(*) AS count,
    COUNT(DISTINCT idempotency_key) AS unique_operations,
    MIN(queued_timestamp) AS oldest,
    MAX(queued_timestamp) AS newest,
    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) AS retried_items
FROM ingestion_queue
GROUP BY status;
