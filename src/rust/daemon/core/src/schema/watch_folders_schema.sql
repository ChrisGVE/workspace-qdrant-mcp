-- watch_folders_schema.sql
-- Unified watch folders table for projects, libraries, and submodules
-- Per WORKSPACE_QDRANT_MCP.md spec v1.6.3
--
-- This table consolidates:
--   - registered_projects (merged)
--   - project_submodules (merged)
--   - watch_configurations (merged)
--   - library_watches (merged)

CREATE TABLE IF NOT EXISTS watch_folders (
    -- Primary identification
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,              -- Absolute filesystem path
    collection TEXT NOT NULL                -- "projects" or "libraries"
        CHECK (collection IN ('projects', 'libraries')),
    tenant_id TEXT NOT NULL,                -- project_id (projects) or library name (libraries)

    -- Hierarchy (for submodules)
    parent_watch_id TEXT,                   -- NULL for top-level, references parent for submodules
    submodule_path TEXT,                    -- Relative path within parent (NULL if not submodule)

    -- Project-specific (NULL for libraries)
    git_remote_url TEXT,                    -- Normalized remote URL
    remote_hash TEXT,                       -- sha256(remote_url)[:12] for grouping duplicates
    disambiguation_path TEXT,               -- Path suffix for clone disambiguation
    is_active INTEGER DEFAULT 0             -- Activity flag (inherited by subprojects)
        CHECK (is_active IN (0, 1)),
    last_activity_at TEXT,                  -- Synced across parent and all subprojects

    -- Library-specific (NULL for projects)
    library_mode TEXT                       -- "sync" or "incremental"
        CHECK (library_mode IS NULL OR library_mode IN ('sync', 'incremental')),

    -- Shared configuration
    follow_symlinks INTEGER DEFAULT 0
        CHECK (follow_symlinks IN (0, 1)),
    enabled INTEGER DEFAULT 1
        CHECK (enabled IN (0, 1)),
    cleanup_on_disable INTEGER DEFAULT 0    -- Remove content when disabled
        CHECK (cleanup_on_disable IN (0, 1)),

    -- Timestamps
    created_at TEXT NOT NULL,               -- ISO 8601 format
    updated_at TEXT NOT NULL,               -- ISO 8601 format
    last_scan TEXT,                         -- NULL if never scanned

    -- Foreign key for submodule hierarchy
    FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
);

-- Index for finding duplicates (same remote, different paths)
CREATE INDEX IF NOT EXISTS idx_watch_remote_hash ON watch_folders(remote_hash);

-- Index for active project lookups (used in queue priority calculation)
CREATE INDEX IF NOT EXISTS idx_watch_active ON watch_folders(is_active) WHERE is_active = 1;

-- Index for daemon polling (find recently updated watches)
CREATE INDEX IF NOT EXISTS idx_watch_updated ON watch_folders(updated_at);

-- Index for enabled watches only
CREATE INDEX IF NOT EXISTS idx_watch_enabled ON watch_folders(enabled) WHERE enabled = 1;

-- Index for subproject hierarchy (find children of a parent)
CREATE INDEX IF NOT EXISTS idx_watch_parent ON watch_folders(parent_watch_id);

-- Index for collection + tenant queries
CREATE INDEX IF NOT EXISTS idx_watch_collection_tenant ON watch_folders(collection, tenant_id);

-- Index for path lookups (common query pattern)
CREATE INDEX IF NOT EXISTS idx_watch_path ON watch_folders(path);

-- ============================================================================
-- Activity Inheritance SQL
-- ============================================================================
-- When any member of a project group (parent or submodule) is activated,
-- ALL members of the group inherit the activity state.
--
-- Usage: Execute with :watch_id parameter to activate a project and all
-- its related watches (parent, siblings, children).
--
-- Example query to activate a project and all its related watches:
--
-- UPDATE watch_folders
-- SET is_active = 1, last_activity_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
-- WHERE watch_id = :watch_id
--    OR parent_watch_id = :watch_id
--    OR watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id)
--    OR parent_watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id);
--
-- ============================================================================
