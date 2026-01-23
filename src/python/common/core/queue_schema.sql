-- Enhanced SQLite Schema for Priority-Based Ingestion Queue System
-- Schema Version: 4.1
-- Purpose: Implements robust queue management with tenant isolation, collection metadata,
--          and comprehensive error tracking for the workspace-qdrant-mcp daemon
--
-- Table Creation Order (for foreign key constraints):
--   1. messages (referenced by ingestion_queue.error_message_id)
--   2. collection_metadata
--   3. ingestion_queue

-- =============================================================================
-- ERROR TRACKING TABLE (Subtask 344.3) - Must be created FIRST
-- ingestion_queue has a foreign key reference to this table
-- =============================================================================

-- Comprehensive error tracking for queue operations
CREATE TABLE IF NOT EXISTS messages (
    -- Auto-incrementing primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Error categorization
    error_type TEXT NOT NULL,

    -- Error description
    error_message TEXT NOT NULL,

    -- Additional error context (JSON)
    -- Examples:
    --   {"stack_trace": "...", "context": {...}}
    --   {"http_status": 500, "upstream_error": "..."}
    error_details TEXT,  -- JSON

    -- Timestamp
    occurred_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Related file and collection for correlation
    file_path TEXT,
    collection_name TEXT,

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,

    -- Created timestamp for purge operations
    created_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for error analysis and troubleshooting
CREATE INDEX IF NOT EXISTS idx_messages_error_type
    ON messages(error_type);

CREATE INDEX IF NOT EXISTS idx_messages_occurred_timestamp
    ON messages(occurred_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_messages_file_path
    ON messages(file_path) WHERE file_path IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_collection
    ON messages(collection_name) WHERE collection_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_retry_count
    ON messages(retry_count);

-- =============================================================================
-- COLLECTION METADATA TABLE (Subtask 344.2)
-- =============================================================================

-- Tracks collection types and configurations for intelligent queue management
CREATE TABLE IF NOT EXISTS collection_metadata (
    -- Primary identifier: collection name
    collection_name TEXT PRIMARY KEY NOT NULL,

    -- Collection type with validation
    collection_type TEXT NOT NULL CHECK (
        collection_type IN ('non-watched', 'watched-dynamic', 'watched-cumulative', 'project')
    ),

    -- Timestamp tracking
    created_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Type-specific configuration (JSON)
    -- Examples:
    --   non-watched: {"max_items": 1000, "ttl_days": 30}
    --   watched-dynamic: {"watch_path": "/path", "patterns": ["*.py"]}
    --   watched-cumulative: {"watch_path": "/path", "retention": "permanent"}
    --   project: {"project_root": "/path", "lsp_enabled": true}
    configuration TEXT,  -- JSON

    -- Multi-tenant support
    tenant_id TEXT DEFAULT 'default',
    branch TEXT DEFAULT 'main'
);

-- Indexes for collection metadata queries
CREATE INDEX IF NOT EXISTS idx_collection_metadata_type
    ON collection_metadata(collection_type);

CREATE INDEX IF NOT EXISTS idx_collection_metadata_tenant
    ON collection_metadata(tenant_id, branch);

CREATE INDEX IF NOT EXISTS idx_collection_metadata_updated
    ON collection_metadata(last_updated);

-- =============================================================================
-- CORE QUEUE TABLE (Subtask 344.1)
-- =============================================================================

-- Main ingestion queue table with file-based primary key
-- Supports multi-tenant isolation, branch-specific processing, and retry chains
CREATE TABLE IF NOT EXISTS ingestion_queue (
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
    retry_from TEXT,  -- Timestamp for exponential backoff (ISO 8601 format)

    -- Error tracking reference
    error_message_id INTEGER,

    -- Collection type classification (system, library, project, global)
    collection_type VARCHAR,

    -- Foreign key constraints
    -- Note: retry_from is NOT a FK - it's a timestamp for backoff scheduling
    FOREIGN KEY (error_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

-- Performance indexes for queue operations
CREATE INDEX IF NOT EXISTS idx_ingestion_queue_priority_time
    ON ingestion_queue(priority DESC, queued_timestamp ASC);

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_retry_from
    ON ingestion_queue(retry_from) WHERE retry_from IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection
    ON ingestion_queue(collection_name, tenant_id, branch);

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_operation
    ON ingestion_queue(operation);

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_error
    ON ingestion_queue(error_message_id) WHERE error_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection_type
    ON ingestion_queue(collection_type) WHERE collection_type IS NOT NULL;

-- =============================================================================
-- QUEUE STATISTICS VIEW
-- =============================================================================

-- Convenience view for queue monitoring
CREATE VIEW IF NOT EXISTS queue_statistics AS
SELECT
    COUNT(*) AS total_items,
    SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) AS urgent_items,
    SUM(CASE WHEN priority >= 5 AND priority < 8 THEN 1 ELSE 0 END) AS high_priority_items,
    SUM(CASE WHEN priority >= 3 AND priority < 5 THEN 1 ELSE 0 END) AS normal_priority_items,
    SUM(CASE WHEN priority < 3 THEN 1 ELSE 0 END) AS low_priority_items,
    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) AS retry_items,
    SUM(CASE WHEN error_message_id IS NOT NULL THEN 1 ELSE 0 END) AS items_with_errors,
    COUNT(DISTINCT collection_name) AS unique_collections,
    COUNT(DISTINCT tenant_id) AS unique_tenants,
    MIN(queued_timestamp) AS oldest_item,
    MAX(queued_timestamp) AS newest_item
FROM ingestion_queue;

-- =============================================================================
-- ERROR SUMMARY VIEW
-- =============================================================================

-- Aggregated error statistics for monitoring
CREATE VIEW IF NOT EXISTS error_summary AS
SELECT
    error_type,
    COUNT(*) AS occurrence_count,
    MAX(occurred_timestamp) AS last_occurrence,
    AVG(retry_count) AS avg_retry_count,
    COUNT(DISTINCT file_path) AS affected_files
FROM messages
WHERE occurred_timestamp >= datetime('now', '-7 days')
GROUP BY error_type
ORDER BY occurrence_count DESC;

-- =============================================================================
-- COLLECTION QUEUE SUMMARY VIEW
-- =============================================================================

-- Per-collection queue statistics
CREATE VIEW IF NOT EXISTS collection_queue_summary AS
SELECT
    iq.collection_name,
    iq.tenant_id,
    iq.branch,
    cm.collection_type,
    COUNT(*) AS queued_items,
    AVG(iq.priority) AS avg_priority,
    MIN(iq.queued_timestamp) AS oldest_queued,
    SUM(CASE WHEN iq.retry_count > 0 THEN 1 ELSE 0 END) AS items_with_retries
FROM ingestion_queue iq
LEFT JOIN collection_metadata cm ON iq.collection_name = cm.collection_name
GROUP BY iq.collection_name, iq.tenant_id, iq.branch, cm.collection_type;

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- =============================================================================

-- Auto-update collection_metadata.last_updated on modification
CREATE TRIGGER IF NOT EXISTS update_collection_metadata_timestamp
AFTER UPDATE ON collection_metadata
BEGIN
    UPDATE collection_metadata
    SET last_updated = CURRENT_TIMESTAMP
    WHERE collection_name = NEW.collection_name;
END;

-- =============================================================================
-- SAMPLE DATA INSERTION FUNCTIONS (for testing/development)
-- =============================================================================

-- Note: These are example INSERT statements for testing
-- In production, use parameterized queries from Python code

-- Example: Insert a high-priority ingestion job
-- INSERT INTO ingestion_queue (file_absolute_path, collection_name, operation, priority)
-- VALUES ('/path/to/file.py', 'my-project-code', 'ingest', 8);

-- Example: Register a project collection
-- INSERT INTO collection_metadata (collection_name, collection_type, configuration)
-- VALUES (
--     'my-project-code',
--     'project',
--     '{"project_root": "/path/to/project", "lsp_enabled": true, "languages": ["python", "javascript"]}'
-- );

-- Example: Log an error
-- INSERT INTO messages (error_type, error_message, error_details, file_path, collection_name)
-- VALUES (
--     'PARSE_ERROR',
--     'Failed to parse Python file',
--     '{"exception": "SyntaxError", "line": 42}',
--     '/path/to/file.py',
--     'my-project-code'
-- );

-- =============================================================================
-- SCHEMA VERSION TRACKING
-- =============================================================================

-- Update schema_version table to track this queue schema version
-- INSERT INTO schema_version (version) VALUES (4);
