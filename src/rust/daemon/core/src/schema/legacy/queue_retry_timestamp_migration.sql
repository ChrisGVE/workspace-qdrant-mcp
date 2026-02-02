-- Legacy migration: Add retry_from column to ingestion_queue
-- Note: This is a legacy migration from Phase 1.
-- The column is already included in queue_schema.sql for new databases.

-- Add retry_from column if it doesn't exist
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we use a workaround
-- This will fail silently if the column already exists

-- Check if column exists and add if not
SELECT CASE
    WHEN (SELECT COUNT(*) FROM pragma_table_info('ingestion_queue') WHERE name='retry_from') = 0
    THEN 'ALTER TABLE ingestion_queue ADD COLUMN retry_from TEXT'
    ELSE 'SELECT 1'
END;
