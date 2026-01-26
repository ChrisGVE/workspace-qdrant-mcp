-- Migration 004: Add active_projects table for fairness scheduler
-- Version: 14
-- Task: 36 (code-audit-r2)
-- Description: Track currently active projects for queue priority scheduling
--
-- This table is used by:
-- 1. Unified queue processor - updates items_processed_count and last_activity_at
-- 2. Watch folder scanner - registers project activity
-- 3. Garbage collector - removes stale projects (inactive > 24 hours)
-- 4. Fairness scheduler - determines which projects get processing priority

-- =============================================================================
-- Table: active_projects
-- =============================================================================
-- Tracks project activity for intelligent queue scheduling. Projects with recent
-- activity get higher priority in the processing queue.
--
-- Fields:
--   project_id: Unique identifier (typically tenant_id or normalized path)
--   tenant_id: Tenant identifier for multi-tenant isolation
--   last_activity_at: Timestamp of most recent activity (query/ingest/watch event)
--   items_processed_count: Running count of processed queue items
--   items_in_queue: Current number of items pending in queue for this project
--   watch_enabled: Whether file watching is active (0/1)
--   watch_folder_id: Reference to watch_folders table (nullable)
--   created_at: When project was first registered
--   updated_at: When record was last modified
--   metadata: JSON object for extensible project metadata

CREATE TABLE IF NOT EXISTS active_projects (
    project_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    last_activity_at TEXT NOT NULL DEFAULT (datetime('now')),
    items_processed_count INTEGER NOT NULL DEFAULT 0,
    items_in_queue INTEGER NOT NULL DEFAULT 0,
    watch_enabled INTEGER NOT NULL DEFAULT 0 CHECK (watch_enabled IN (0, 1)),
    watch_folder_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT,

    -- Foreign key to watch_folders (optional relationship)
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id) ON DELETE SET NULL
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for finding recently active projects (fairness scheduler)
CREATE INDEX IF NOT EXISTS idx_active_projects_last_activity
    ON active_projects(last_activity_at DESC);

-- Composite index for watch-related queries
CREATE INDEX IF NOT EXISTS idx_active_projects_watch
    ON active_projects(watch_enabled, watch_folder_id);

-- Index for tenant-based lookups
CREATE INDEX IF NOT EXISTS idx_active_projects_tenant
    ON active_projects(tenant_id);

-- =============================================================================
-- Trigger: Auto-update updated_at on modification
-- =============================================================================

-- Note: Trigger omitted from migration file due to parser limitations with
-- embedded semicolons in BEGIN...END blocks. The trigger is added inline
-- in the _migrate_schema method after loading these statements.

-- =============================================================================
-- View: v_active_projects_stats
-- =============================================================================
-- Monitoring view for dashboard and admin commands

CREATE VIEW IF NOT EXISTS v_active_projects_stats AS
SELECT
    COUNT(*) as total_projects,
    SUM(CASE WHEN watch_enabled = 1 THEN 1 ELSE 0 END) as watched_projects,
    SUM(items_processed_count) as total_items_processed,
    SUM(items_in_queue) as total_items_in_queue,
    MAX(last_activity_at) as most_recent_activity,
    MIN(last_activity_at) as oldest_activity,
    COUNT(CASE WHEN datetime(last_activity_at) > datetime('now', '-1 hour') THEN 1 END) as active_last_hour,
    COUNT(CASE WHEN datetime(last_activity_at) > datetime('now', '-24 hours') THEN 1 END) as active_last_24h
FROM active_projects;

-- =============================================================================
-- View: v_stale_projects
-- =============================================================================
-- View for garbage collection - projects inactive for more than 24 hours

CREATE VIEW IF NOT EXISTS v_stale_projects AS
SELECT
    project_id,
    tenant_id,
    last_activity_at,
    items_processed_count,
    created_at,
    julianday('now') - julianday(last_activity_at) as days_inactive
FROM active_projects
WHERE datetime(last_activity_at) < datetime('now', '-24 hours')
ORDER BY last_activity_at ASC;

-- =============================================================================
-- Update schema version
-- =============================================================================

INSERT OR REPLACE INTO schema_version (version) VALUES (14);
