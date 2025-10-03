-- Migration: Add collection_type field to ingestion_queue table
-- Version: 4.1
-- Date: 2025-10-03
-- Purpose: Add collection type classification to queue items for type-specific processing

-- Add collection_type column (nullable for backward compatibility)
ALTER TABLE ingestion_queue ADD COLUMN collection_type VARCHAR;

-- Create index for collection_type queries
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection_type
    ON ingestion_queue(collection_type) WHERE collection_type IS NOT NULL;

-- Migration complete - collection_type field added successfully
-- Note: Existing rows will have NULL collection_type until migration script populates them
