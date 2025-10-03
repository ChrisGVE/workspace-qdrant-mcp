-- Enhanced Error Messages Schema
-- Schema Enhancement for messages table
-- Purpose: Add severity levels, categorization, and acknowledgment tracking
--          for comprehensive error message management

-- =============================================================================
-- ENHANCED MESSAGES TABLE
-- =============================================================================

-- Enhanced messages table with severity, category, and acknowledgment support
-- Replaces the basic error tracking with a comprehensive message management system
CREATE TABLE IF NOT EXISTS messages_enhanced (
    -- Primary identifier
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Timestamp when error/message occurred
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Severity level with validation
    severity TEXT NOT NULL CHECK (severity IN ('error', 'warning', 'info')) DEFAULT 'error',

    -- Category for error classification
    category TEXT NOT NULL CHECK (
        category IN (
            'file_corrupt',
            'tool_missing',
            'network',
            'metadata_invalid',
            'processing_failed',
            'parse_error',
            'permission_denied',
            'resource_exhausted',
            'timeout',
            'unknown'
        )
    ),

    -- Human-readable error message
    message TEXT NOT NULL,

    -- Context as JSON with file_path, collection, tenant_id, etc.
    -- Examples:
    --   {"file_path": "/path/to/file", "collection": "my-project", "tenant_id": "default"}
    --   {"stack_trace": "...", "line": 42, "column": 10}
    context TEXT,  -- JSON

    -- Acknowledgment tracking
    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),  -- Boolean as INTEGER
    acknowledged_at TIMESTAMP,
    acknowledged_by TEXT,

    -- Retry tracking (preserved from original schema)
    retry_count INTEGER NOT NULL DEFAULT 0
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Index for severity-based filtering
CREATE INDEX IF NOT EXISTS idx_messages_enhanced_severity
    ON messages_enhanced(severity);

-- Index for category-based filtering
CREATE INDEX IF NOT EXISTS idx_messages_enhanced_category
    ON messages_enhanced(category);

-- Index for timestamp-based queries (most recent errors)
CREATE INDEX IF NOT EXISTS idx_messages_enhanced_timestamp
    ON messages_enhanced(timestamp DESC);

-- Index for unacknowledged messages
CREATE INDEX IF NOT EXISTS idx_messages_enhanced_acknowledged
    ON messages_enhanced(acknowledged) WHERE acknowledged = 0;

-- Compound index for severity + acknowledged filtering (common query pattern)
CREATE INDEX IF NOT EXISTS idx_messages_enhanced_severity_ack
    ON messages_enhanced(severity, acknowledged);

-- Index for retry count (finding messages with retries)
CREATE INDEX IF NOT EXISTS idx_messages_enhanced_retry_count
    ON messages_enhanced(retry_count) WHERE retry_count > 0;

-- =============================================================================
-- CONVENIENCE VIEWS
-- =============================================================================

-- View for unacknowledged errors by severity
CREATE VIEW IF NOT EXISTS unacknowledged_messages AS
SELECT
    id,
    timestamp,
    severity,
    category,
    message,
    context,
    retry_count,
    -- Extract file_path from JSON context if available
    json_extract(context, '$.file_path') as file_path,
    json_extract(context, '$.collection') as collection,
    json_extract(context, '$.tenant_id') as tenant_id
FROM messages_enhanced
WHERE acknowledged = 0
ORDER BY
    CASE severity
        WHEN 'error' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'info' THEN 3
    END,
    timestamp DESC;

-- View for error statistics by category and severity
CREATE VIEW IF NOT EXISTS message_statistics AS
SELECT
    severity,
    category,
    COUNT(*) as total_count,
    SUM(CASE WHEN acknowledged = 0 THEN 1 ELSE 0 END) as unacknowledged_count,
    MAX(timestamp) as last_occurrence,
    AVG(retry_count) as avg_retry_count
FROM messages_enhanced
WHERE timestamp >= datetime('now', '-7 days')
GROUP BY severity, category
ORDER BY severity, total_count DESC;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Trigger to automatically set acknowledged_at when acknowledged changes to 1
CREATE TRIGGER IF NOT EXISTS set_acknowledged_timestamp
AFTER UPDATE OF acknowledged ON messages_enhanced
FOR EACH ROW
WHEN NEW.acknowledged = 1 AND OLD.acknowledged = 0
BEGIN
    UPDATE messages_enhanced
    SET acknowledged_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- Trigger to validate acknowledged_by is set when acknowledging
CREATE TRIGGER IF NOT EXISTS validate_acknowledged_by
BEFORE UPDATE OF acknowledged ON messages_enhanced
FOR EACH ROW
WHEN NEW.acknowledged = 1 AND NEW.acknowledged_by IS NULL
BEGIN
    SELECT RAISE(ABORT, 'acknowledged_by must be set when acknowledging a message');
END;

-- =============================================================================
-- MIGRATION COMPATIBILITY
-- =============================================================================

-- This schema is designed to replace the existing messages table
-- Migration mapping from old schema:
--   error_type → category (mapped)
--   error_message → message (direct)
--   error_details → context (JSON transformation)
--   occurred_timestamp → timestamp (direct)
--   file_path → context.file_path (JSON field)
--   collection_name → context.collection (JSON field)
--   retry_count → retry_count (direct)
--
-- New fields initialized as:
--   severity → 'error' (default)
--   acknowledged → 0 (default)
--   acknowledged_at → NULL
--   acknowledged_by → NULL
