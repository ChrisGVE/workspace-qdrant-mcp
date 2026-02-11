//! Daemon State Management Module
//!
//! This module handles SQLite-based persistence for the daemon's watch folders
//! configuration. Per ADR-003 and the 3-table SQLite compliance requirement,
//! only three tables are allowed: schema_version, unified_queue, and watch_folders.
//!
//! The daemon is the sole owner of the SQLite database schema. This module
//! initializes the spec-required tables via SchemaManager.

use std::collections::HashMap;
use std::path::Path;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{Row, SqlitePool, sqlite::{SqlitePoolOptions, SqliteConnectOptions}};
use tracing::{info, warn};

use crate::schema_version::{SchemaManager, SchemaError};

/// Daemon state management errors
#[derive(thiserror::Error, Debug)]
pub enum DaemonStateError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Schema migration error: {0}")]
    Schema(#[from] SchemaError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("State error: {0}")]
    State(String),
}

/// Result type for daemon state operations
pub type DaemonStateResult<T> = Result<T, DaemonStateError>;


/// Watch folder record matching spec-defined watch_folders table
/// Per WORKSPACE_QDRANT_MCP.md v1.6.4, this unified table consolidates
/// project and library watching configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchFolderRecord {
    /// Unique watch identifier (PRIMARY KEY)
    pub watch_id: String,
    /// Absolute filesystem path
    pub path: String,
    /// Collection type: "projects" or "libraries"
    pub collection: String,
    /// Tenant identifier (project_id for projects, library name for libraries)
    pub tenant_id: String,

    // Hierarchy (for submodules)
    /// Parent watch ID (None for top-level)
    pub parent_watch_id: Option<String>,
    /// Relative path within parent (None if not submodule)
    pub submodule_path: Option<String>,

    // Project-specific fields (None for libraries)
    /// Normalized git remote URL
    pub git_remote_url: Option<String>,
    /// SHA256 hash of remote URL (first 12 chars)
    pub remote_hash: Option<String>,
    /// Path suffix for clone disambiguation
    pub disambiguation_path: Option<String>,
    /// Activity flag - inherited by all subprojects
    pub is_active: bool,
    /// Last activity timestamp
    pub last_activity_at: Option<DateTime<Utc>>,
    /// Whether this watch folder is paused (events buffered, not processed)
    pub is_paused: bool,
    /// Timestamp when pause began, None if not paused
    pub pause_start_time: Option<DateTime<Utc>>,
    /// Whether this folder is archived (no watching/ingesting, but remains searchable)
    pub is_archived: bool,

    // Library-specific fields (None for projects)
    /// Library mode: "sync" or "incremental"
    pub library_mode: Option<String>,

    // Shared configuration
    /// Whether to follow symlinks during watching
    pub follow_symlinks: bool,
    /// Whether this watch is enabled
    pub enabled: bool,
    /// Whether to remove content from Qdrant when watch is disabled
    pub cleanup_on_disable: bool,

    // Timestamps
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Last scan timestamp (None if never scanned)
    pub last_scan: Option<DateTime<Utc>>,
}

impl WatchFolderRecord {
    /// Check if this is a submodule (has a parent)
    pub fn is_submodule(&self) -> bool {
        self.parent_watch_id.is_some()
    }

    /// Check if this is a library watch
    pub fn is_library(&self) -> bool {
        self.collection == "libraries"
    }

    /// Check if this is a project watch
    pub fn is_project(&self) -> bool {
        self.collection == "projects"
    }
}

/// Daemon state manager for SQLite persistence
pub struct DaemonStateManager {
    pool: SqlitePool,
}

impl DaemonStateManager {
    /// Create a new daemon state manager
    pub async fn new<P: AsRef<Path>>(database_path: P) -> DaemonStateResult<Self> {
        info!("Initializing daemon state manager with database: {}", 
            database_path.as_ref().display());

        let connect_options = SqliteConnectOptions::new()
            .filename(database_path.as_ref())
            .create_if_missing(true);
        
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await?;

        Ok(Self { pool })
    }

    /// Create a daemon state manager from an existing pool
    ///
    /// Use this when you already have a database pool (e.g., in gRPC services)
    /// and don't want to create a new connection.
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Initialize the database schema
    ///
    /// Per ADR-003, the daemon owns the SQLite database and is responsible for:
    /// Running schema migrations for spec-required tables (schema_version,
    /// unified_queue, watch_folders) - these are the ONLY allowed tables per
    /// the 3-table SQLite compliance requirement.
    pub async fn initialize(&self) -> DaemonStateResult<()> {
        info!("Initializing daemon state database schema");

        // Run schema migrations for spec-required tables
        // This creates schema_version, unified_queue, and watch_folders tables
        let schema_manager = SchemaManager::new(self.pool.clone());
        schema_manager.run_migrations().await?;

        let version = schema_manager.get_current_version().await?.unwrap_or(0);
        info!("Schema at version {}", version);

        info!("Daemon state database schema initialized successfully");
        Ok(())
    }

    /// Close the database connection
    pub async fn close(&self) -> DaemonStateResult<()> {
        info!("Closing daemon state manager");
        self.pool.close().await;
        Ok(())
    }

    // ========================================================================
    // Watch Folders methods (spec-defined watch_folders table)
    // ========================================================================

    /// Store or update a watch folder record
    pub async fn store_watch_folder(&self, record: &WatchFolderRecord) -> DaemonStateResult<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO watch_folders (
                watch_id, path, collection, tenant_id,
                parent_watch_id, submodule_path,
                git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                is_paused, pause_start_time, is_archived,
                library_mode,
                follow_symlinks, enabled, cleanup_on_disable,
                created_at, updated_at, last_scan
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
            "#,
        )
        .bind(&record.watch_id)
        .bind(&record.path)
        .bind(&record.collection)
        .bind(&record.tenant_id)
        .bind(&record.parent_watch_id)
        .bind(&record.submodule_path)
        .bind(&record.git_remote_url)
        .bind(&record.remote_hash)
        .bind(&record.disambiguation_path)
        .bind(record.is_active as i32)
        .bind(record.last_activity_at.map(|dt| dt.to_rfc3339()))
        .bind(record.is_paused as i32)
        .bind(record.pause_start_time.map(|dt| dt.to_rfc3339()))
        .bind(record.is_archived as i32)
        .bind(&record.library_mode)
        .bind(record.follow_symlinks as i32)
        .bind(record.enabled as i32)
        .bind(record.cleanup_on_disable as i32)
        .bind(record.created_at.to_rfc3339())
        .bind(record.updated_at.to_rfc3339())
        .bind(record.last_scan.map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get a watch folder by ID
    pub async fn get_watch_folder(&self, watch_id: &str) -> DaemonStateResult<Option<WatchFolderRecord>> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders WHERE watch_id = ?1
            "#,
        )
        .bind(watch_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(self.row_to_watch_folder(&row)?))
        } else {
            Ok(None)
        }
    }

    /// List watch folders with optional collection filter
    pub async fn list_watch_folders(&self, collection_filter: Option<&str>, enabled_only: bool) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let base_query = r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
        "#;

        let query = match (collection_filter, enabled_only) {
            (Some(_), true) => format!("{} WHERE collection = ?1 AND enabled = 1 ORDER BY created_at", base_query),
            (Some(_), false) => format!("{} WHERE collection = ?1 ORDER BY created_at", base_query),
            (None, true) => format!("{} WHERE enabled = 1 ORDER BY created_at", base_query),
            (None, false) => format!("{} ORDER BY created_at", base_query),
        };

        let rows = if let Some(collection) = collection_filter {
            sqlx::query(&query)
                .bind(collection)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query(&query)
                .fetch_all(&self.pool)
                .await?
        };

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }

        Ok(results)
    }

    /// List active project watch folders
    pub async fn list_active_projects(&self) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
            WHERE collection = 'projects' AND is_active = 1 AND enabled = 1
            ORDER BY last_activity_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }

        Ok(results)
    }

    /// Activate a project and all descendant watches (recursive)
    /// Uses WITH RECURSIVE to traverse the full parent_watch_id hierarchy
    pub async fn activate_project_group(&self, watch_id: &str) -> DaemonStateResult<u64> {
        let result = sqlx::query(
            r#"
            WITH RECURSIVE project_group AS (
                SELECT watch_id FROM watch_folders WHERE watch_id = ?1
                UNION
                SELECT wf.watch_id FROM watch_folders wf
                JOIN project_group pg ON wf.parent_watch_id = pg.watch_id
            )
            UPDATE watch_folders
            SET is_active = 1,
                last_activity_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id IN (SELECT watch_id FROM project_group)
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Deactivate a project and all descendant watches (recursive)
    /// Uses WITH RECURSIVE to traverse the full parent_watch_id hierarchy
    pub async fn deactivate_project_group(&self, watch_id: &str) -> DaemonStateResult<u64> {
        let result = sqlx::query(
            r#"
            WITH RECURSIVE project_group AS (
                SELECT watch_id FROM watch_folders WHERE watch_id = ?1
                UNION
                SELECT wf.watch_id FROM watch_folders wf
                JOIN project_group pg ON wf.parent_watch_id = pg.watch_id
            )
            UPDATE watch_folders
            SET is_active = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id IN (SELECT watch_id FROM project_group)
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Get watch folder by tenant_id (project_id for projects, library_name for libraries)
    /// Returns the first matching top-level watch (not a submodule)
    pub async fn get_watch_folder_by_tenant_id(
        &self,
        tenant_id: &str,
        collection: &str,
    ) -> DaemonStateResult<Option<WatchFolderRecord>> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
            WHERE tenant_id = ?1 AND collection = ?2 AND parent_watch_id IS NULL
            LIMIT 1
            "#,
        )
        .bind(tenant_id)
        .bind(collection)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(self.row_to_watch_folder(&row)?)),
            None => Ok(None),
        }
    }

    /// Activate a project by tenant_id (project_id)
    /// Finds the watch folder and activates the entire project group
    /// Returns (rows_affected, watch_id used)
    pub async fn activate_project_by_tenant_id(
        &self,
        tenant_id: &str,
    ) -> DaemonStateResult<(u64, Option<String>)> {
        // Find the main watch folder for this tenant
        let watch_folder = self.get_watch_folder_by_tenant_id(tenant_id, "projects").await?;

        match watch_folder {
            Some(folder) => {
                let affected = self.activate_project_group(&folder.watch_id).await?;
                Ok((affected, Some(folder.watch_id)))
            }
            None => Ok((0, None)),
        }
    }

    /// Deactivate a project by tenant_id (project_id)
    /// Finds the watch folder and deactivates the entire project group
    /// Returns (rows_affected, watch_id used)
    pub async fn deactivate_project_by_tenant_id(
        &self,
        tenant_id: &str,
    ) -> DaemonStateResult<(u64, Option<String>)> {
        // Find the main watch folder for this tenant
        let watch_folder = self.get_watch_folder_by_tenant_id(tenant_id, "projects").await?;

        match watch_folder {
            Some(folder) => {
                let affected = self.deactivate_project_group(&folder.watch_id).await?;
                Ok((affected, Some(folder.watch_id)))
            }
            None => Ok((0, None)),
        }
    }

    /// Update last_activity_at for a project and all descendant watches (heartbeat)
    /// Uses WITH RECURSIVE to traverse the full parent_watch_id hierarchy
    pub async fn heartbeat_project_group(&self, watch_id: &str) -> DaemonStateResult<u64> {
        let result = sqlx::query(
            r#"
            WITH RECURSIVE project_group AS (
                SELECT watch_id FROM watch_folders WHERE watch_id = ?1
                UNION
                SELECT wf.watch_id FROM watch_folders wf
                JOIN project_group pg ON wf.parent_watch_id = pg.watch_id
            )
            UPDATE watch_folders
            SET last_activity_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id IN (SELECT watch_id FROM project_group)
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Update heartbeat by tenant_id (project_id)
    /// Finds the watch folder and updates last_activity_at for the entire project group
    /// Returns (rows_affected, watch_id used)
    pub async fn heartbeat_project_by_tenant_id(
        &self,
        tenant_id: &str,
    ) -> DaemonStateResult<(u64, Option<String>)> {
        // Find the main watch folder for this tenant
        let watch_folder = self.get_watch_folder_by_tenant_id(tenant_id, "projects").await?;

        match watch_folder {
            Some(folder) => {
                let affected = self.heartbeat_project_group(&folder.watch_id).await?;
                Ok((affected, Some(folder.watch_id)))
            }
            None => Ok((0, None)),
        }
    }

    /// Deactivate projects that have been inactive for longer than the timeout.
    ///
    /// Queries for active projects whose `last_activity_at` is older than
    /// `timeout_secs` seconds ago. For each timed-out project, deactivates
    /// the entire project group (including submodules) via `deactivate_project_group`.
    ///
    /// Returns the number of project groups deactivated.
    pub async fn deactivate_inactive_projects(&self, timeout_secs: i64) -> DaemonStateResult<u64> {
        // Find active parent projects (not submodules) whose last activity exceeds timeout
        let stale_watches: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT watch_id FROM watch_folders
            WHERE is_active = 1
              AND collection = 'projects'
              AND parent_watch_id IS NULL
              AND last_activity_at IS NOT NULL
              AND (julianday('now') - julianday(last_activity_at)) * 86400 > ?1
            "#,
        )
        .bind(timeout_secs)
        .fetch_all(&self.pool)
        .await?;

        if stale_watches.is_empty() {
            return Ok(0);
        }

        info!(
            "Inactivity timeout: deactivating {} project(s) inactive for >{}s",
            stale_watches.len(), timeout_secs
        );

        let mut deactivated = 0u64;
        for watch_id in &stale_watches {
            match self.deactivate_project_group(watch_id).await {
                Ok(affected) => {
                    info!("Deactivated project group {} ({} watch folders)", watch_id, affected);
                    deactivated += 1;
                }
                Err(e) => {
                    warn!("Failed to deactivate project group {}: {}", watch_id, e);
                }
            }
        }

        Ok(deactivated)
    }

    /// Update watch folder enabled status
    pub async fn set_watch_folder_enabled(&self, watch_id: &str, enabled: bool) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET enabled = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?2
            "#,
        )
        .bind(enabled as i32)
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Update last scan timestamp for a watch folder
    pub async fn update_watch_folder_last_scan(&self, watch_id: &str) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?1
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Remove watch folder
    pub async fn remove_watch_folder(&self, watch_id: &str) -> DaemonStateResult<bool> {
        let result = sqlx::query("DELETE FROM watch_folders WHERE watch_id = ?1")
            .bind(watch_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    // ========================================================================
    // Pause/Resume methods (Task 543)
    // ========================================================================

    /// Pause all enabled watch folders
    /// Returns the number of watch folders paused
    pub async fn pause_all_watchers(&self) -> DaemonStateResult<u64> {
        use super::watch_folders_schema::PAUSE_ALL_WATCHERS_SQL;
        let result = sqlx::query(PAUSE_ALL_WATCHERS_SQL)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    /// Resume all enabled watch folders
    /// Returns the number of watch folders resumed
    pub async fn resume_all_watchers(&self) -> DaemonStateResult<u64> {
        use super::watch_folders_schema::RESUME_ALL_WATCHERS_SQL;
        let result = sqlx::query(RESUME_ALL_WATCHERS_SQL)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    /// Get list of watch folder IDs that are currently paused
    pub async fn get_paused_watch_ids(&self) -> DaemonStateResult<Vec<String>> {
        let rows: Vec<sqlx::sqlite::SqliteRow> = sqlx::query(
            "SELECT watch_id FROM watch_folders WHERE is_paused = 1 AND enabled = 1"
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| r.try_get::<String, _>("watch_id").unwrap_or_default()).collect())
    }

    /// Check if any enabled watch folder is paused.
    /// Returns true if at least one enabled watch has is_paused=1.
    pub async fn any_watchers_paused(&self) -> DaemonStateResult<bool> {
        let count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1 AND enabled = 1"
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(count > 0)
    }

    // ========================================================================
    // Project Disambiguation methods (Task 3)
    // ========================================================================

    /// Find existing watch folders (clones) with the same remote_hash
    ///
    /// Used for duplicate detection when registering a new project.
    /// Returns all top-level watch folders (parent_watch_id IS NULL) with matching remote_hash.
    pub async fn find_clones_by_remote_hash(
        &self,
        remote_hash: &str,
    ) -> DaemonStateResult<Vec<WatchFolderRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id,
                   parent_watch_id, submodule_path,
                   git_remote_url, remote_hash, disambiguation_path, is_active, last_activity_at,
                   is_paused, pause_start_time, is_archived,
                   library_mode,
                   follow_symlinks, enabled, cleanup_on_disable,
                   created_at, updated_at, last_scan
            FROM watch_folders
            WHERE remote_hash = ?1 AND parent_watch_id IS NULL AND collection = 'projects'
            ORDER BY created_at ASC
            "#,
        )
        .bind(remote_hash)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            results.push(self.row_to_watch_folder(&row)?);
        }

        Ok(results)
    }

    /// Update disambiguation_path and tenant_id for a watch folder
    ///
    /// Used when disambiguating clones. Updates the project_id (tenant_id)
    /// to include the disambiguation component.
    pub async fn update_project_disambiguation(
        &self,
        watch_id: &str,
        new_tenant_id: &str,
        disambiguation_path: &str,
    ) -> DaemonStateResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET tenant_id = ?1,
                disambiguation_path = ?2,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id = ?3
            "#,
        )
        .bind(new_tenant_id)
        .bind(disambiguation_path)
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Register a project with automatic duplicate detection and disambiguation
    ///
    /// This method handles the full disambiguation workflow:
    /// 1. Check for existing clones with the same remote_hash
    /// 2. If duplicates found, compute disambiguation paths for ALL clones
    /// 3. Update existing clones with new disambiguation paths
    /// 4. Store the new project with its disambiguation path
    ///
    /// Returns (WatchFolderRecord, Vec<(old_tenant_id, new_tenant_id)>) where the second
    /// element contains alias mappings for any updated existing projects.
    pub async fn register_project_with_disambiguation(
        &self,
        mut record: WatchFolderRecord,
    ) -> DaemonStateResult<(WatchFolderRecord, Vec<(String, String)>)> {
        use crate::project_disambiguation::{DisambiguationPathComputer, ProjectIdCalculator};
        use std::path::PathBuf;

        let mut aliases: Vec<(String, String)> = Vec::new();

        // Only handle disambiguation for projects with git remotes
        if let Some(ref remote_hash) = record.remote_hash {
            // Find existing clones with the same remote
            let existing_clones = self.find_clones_by_remote_hash(remote_hash).await?;

            if !existing_clones.is_empty() {
                // Collect all paths including the new one
                let mut all_paths: Vec<PathBuf> = existing_clones
                    .iter()
                    .map(|c| PathBuf::from(&c.path))
                    .collect();
                all_paths.push(PathBuf::from(&record.path));

                // Recompute disambiguation for all clones
                let disambig_map = DisambiguationPathComputer::recompute_all(&all_paths);

                let calculator = ProjectIdCalculator::new();

                // Update existing clones with new disambiguation paths
                for clone in &existing_clones {
                    let clone_path = PathBuf::from(&clone.path);
                    if let Some(new_disambig) = disambig_map.get(&clone_path) {
                        // Skip if disambiguation hasn't changed
                        let current_disambig = clone.disambiguation_path.as_deref().unwrap_or("");
                        if new_disambig == current_disambig {
                            continue;
                        }

                        // Calculate new tenant_id with disambiguation
                        let new_tenant_id = calculator.calculate(
                            &clone_path,
                            clone.git_remote_url.as_deref(),
                            if new_disambig.is_empty() { None } else { Some(new_disambig) },
                        );

                        // Record the alias (old_id -> new_id)
                        if clone.tenant_id != new_tenant_id {
                            aliases.push((clone.tenant_id.clone(), new_tenant_id.clone()));
                        }

                        // Update the existing clone
                        self.update_project_disambiguation(
                            &clone.watch_id,
                            &new_tenant_id,
                            new_disambig,
                        )
                        .await?;
                    }
                }

                // Set disambiguation path for the new project
                let new_path = PathBuf::from(&record.path);
                if let Some(new_disambig) = disambig_map.get(&new_path) {
                    record.disambiguation_path = if new_disambig.is_empty() {
                        None
                    } else {
                        Some(new_disambig.clone())
                    };

                    // Recalculate tenant_id with disambiguation
                    record.tenant_id = calculator.calculate(
                        &new_path,
                        record.git_remote_url.as_deref(),
                        record.disambiguation_path.as_deref(),
                    );
                }
            }
        }

        // Store the new project
        self.store_watch_folder(&record).await?;

        Ok((record, aliases))
    }

    /// Check if a path is already registered as a project
    pub async fn is_path_registered(&self, path: &str) -> DaemonStateResult<bool> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folders WHERE path = ?1 AND collection = 'projects'"
        )
        .bind(path)
        .fetch_one(&self.pool)
        .await?;

        Ok(count > 0)
    }

    /// Helper to convert a database row to WatchFolderRecord
    fn row_to_watch_folder(&self, row: &sqlx::sqlite::SqliteRow) -> DaemonStateResult<WatchFolderRecord> {
        let parse_datetime = |s: &str| -> DateTime<Utc> {
            DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now())
        };

        let parse_optional_datetime = |s: Option<String>| -> Option<DateTime<Utc>> {
            s.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc)))
        };

        Ok(WatchFolderRecord {
            watch_id: row.try_get("watch_id")?,
            path: row.try_get("path")?,
            collection: row.try_get("collection")?,
            tenant_id: row.try_get("tenant_id")?,
            parent_watch_id: row.try_get("parent_watch_id")?,
            submodule_path: row.try_get("submodule_path")?,
            git_remote_url: row.try_get("git_remote_url")?,
            remote_hash: row.try_get("remote_hash")?,
            disambiguation_path: row.try_get("disambiguation_path")?,
            is_active: row.try_get::<i32, _>("is_active")? != 0,
            last_activity_at: parse_optional_datetime(row.try_get("last_activity_at")?),
            is_paused: row.try_get::<i32, _>("is_paused").unwrap_or(0) != 0,
            pause_start_time: parse_optional_datetime(row.try_get("pause_start_time").unwrap_or(None)),
            is_archived: row.try_get::<i32, _>("is_archived").unwrap_or(0) != 0,
            library_mode: row.try_get("library_mode")?,
            follow_symlinks: row.try_get::<i32, _>("follow_symlinks")? != 0,
            enabled: row.try_get::<i32, _>("enabled")? != 0,
            cleanup_on_disable: row.try_get::<i32, _>("cleanup_on_disable")? != 0,
            created_at: parse_datetime(&row.try_get::<String, _>("created_at")?),
            updated_at: parse_datetime(&row.try_get::<String, _>("updated_at")?),
            last_scan: parse_optional_datetime(row.try_get("last_scan")?),
        })
    }

    /// Get database statistics (watch folders only)
    pub async fn get_stats(&self) -> DaemonStateResult<HashMap<String, JsonValue>> {
        let mut stats = HashMap::new();

        // Watch folder counts by enabled/disabled status
        let watch_rows = sqlx::query(
            "SELECT CASE WHEN enabled = 1 THEN 'enabled' ELSE 'disabled' END as status, COUNT(*) as count FROM watch_folders GROUP BY enabled"
        )
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();

        let mut watch_stats = HashMap::new();
        for row in watch_rows {
            let status: String = row.try_get("status").unwrap_or_default();
            let count: i64 = row.try_get("count").unwrap_or(0);
            watch_stats.insert(status, JsonValue::Number(count.into()));
        }
        stats.insert("watch_counts".to_string(), JsonValue::Object(watch_stats.into_iter().collect()));

        // Watch folder counts by collection type
        let collection_rows = sqlx::query(
            "SELECT collection, COUNT(*) as count FROM watch_folders GROUP BY collection"
        )
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();

        let mut collection_stats = HashMap::new();
        for row in collection_rows {
            let collection: String = row.try_get("collection").unwrap_or_default();
            let count: i64 = row.try_get("count").unwrap_or(0);
            collection_stats.insert(collection, JsonValue::Number(count.into()));
        }
        stats.insert("collection_counts".to_string(), JsonValue::Object(collection_stats.into_iter().collect()));

        // Active projects count
        let active_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folders WHERE is_active = 1 AND collection = 'projects'"
        )
            .fetch_one(&self.pool)
            .await
            .unwrap_or(0);
        stats.insert("active_projects".to_string(), JsonValue::Number(active_count.into()));

        Ok(stats)
    }

    /// Get a reference to the underlying SQLite pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

impl Clone for DaemonStateManager {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

/// Poll the database for pause state and sync to a shared AtomicBool.
///
/// This function queries whether any enabled watch folder is paused and sets
/// the provided `pause_flag` accordingly. Used by the daemon to detect
/// CLI-driven pause/resume changes that bypass the gRPC endpoint.
///
/// Returns `true` if the flag value changed.
pub async fn poll_pause_state(
    pool: &SqlitePool,
    pause_flag: &std::sync::atomic::AtomicBool,
) -> DaemonStateResult<bool> {
    let count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1 AND enabled = 1"
    )
    .fetch_one(pool)
    .await?;
    let db_paused = count > 0;
    let previous = pause_flag.swap(db_paused, std::sync::atomic::Ordering::SeqCst);
    Ok(previous != db_paused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_daemon_state_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("daemon_test.db");
        
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();
        
        assert!(db_path.exists());
    }

    #[tokio::test]
    async fn test_watch_folder_crud() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("watch_folder_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create a watch folder record
        let record = WatchFolderRecord {
            watch_id: "test-watch-001".to_string(),
            path: "/projects/my-project".to_string(),
            collection: "projects".to_string(),
            tenant_id: "my-project-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some("https://github.com/user/repo.git".to_string()),
            remote_hash: Some("abc123def456".to_string()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        // Store
        manager.store_watch_folder(&record).await.unwrap();

        // Retrieve
        let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.path, "/projects/my-project");
        assert_eq!(retrieved.tenant_id, "my-project-tenant");
        assert!(!retrieved.is_active);
        assert!(retrieved.enabled);

        // Activate project
        let updated = manager.activate_project_group("test-watch-001").await.unwrap();
        assert_eq!(updated, 1);

        let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap().unwrap();
        assert!(retrieved.is_active);

        // List active projects
        let active = manager.list_active_projects().await.unwrap();
        assert_eq!(active.len(), 1);

        // Deactivate
        manager.deactivate_project_group("test-watch-001").await.unwrap();
        let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap().unwrap();
        assert!(!retrieved.is_active);

        // Delete
        let deleted = manager.remove_watch_folder("test-watch-001").await.unwrap();
        assert!(deleted);

        let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_watch_folder_with_submodule() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("watch_submodule_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create parent project
        let parent = WatchFolderRecord {
            watch_id: "parent-001".to_string(),
            path: "/projects/parent".to_string(),
            collection: "projects".to_string(),
            tenant_id: "parent-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some("https://github.com/user/parent.git".to_string()),
            remote_hash: Some("parent12hash".to_string()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&parent).await.unwrap();

        // Create submodule
        let submodule = WatchFolderRecord {
            watch_id: "submodule-001".to_string(),
            path: "/projects/parent/libs/sub".to_string(),
            collection: "projects".to_string(),
            tenant_id: "submodule-tenant".to_string(),
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some("libs/sub".to_string()),
            git_remote_url: Some("https://github.com/user/sub.git".to_string()),
            remote_hash: Some("sub123hash".to_string()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&submodule).await.unwrap();

        // Activate parent should activate submodule too
        let updated = manager.activate_project_group("parent-001").await.unwrap();
        assert_eq!(updated, 2); // Both parent and submodule

        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
        assert!(parent_record.is_active);
        assert!(submodule_record.is_active);

        // Deactivate submodule only affects the submodule (recursive goes DOWN, not UP)
        manager.deactivate_project_group("submodule-001").await.unwrap();
        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
        assert!(parent_record.is_active); // Parent stays active
        assert!(!submodule_record.is_active);

        // Deactivate from parent deactivates entire group
        manager.deactivate_project_group("parent-001").await.unwrap();
        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        assert!(!parent_record.is_active);
    }

    #[tokio::test]
    async fn test_watch_folder_library_config() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("watch_library_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create a library watch folder
        let record = WatchFolderRecord {
            watch_id: "lib-001".to_string(),
            path: "/libraries/my-docs".to_string(),
            collection: "libraries".to_string(),
            tenant_id: "my-docs".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: Some("sync".to_string()),
            follow_symlinks: true,
            enabled: true,
            cleanup_on_disable: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        manager.store_watch_folder(&record).await.unwrap();

        // Retrieve and verify library-specific fields
        let retrieved = manager.get_watch_folder("lib-001").await.unwrap().unwrap();
        assert_eq!(retrieved.collection, "libraries");
        assert_eq!(retrieved.library_mode, Some("sync".to_string()));
        assert!(retrieved.follow_symlinks);
        assert!(retrieved.cleanup_on_disable);
    }

    #[tokio::test]
    async fn test_watch_folder_collection_filter() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("watch_filter_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create project watches
        for i in 1..=3 {
            let record = WatchFolderRecord {
                watch_id: format!("project-{}", i),
                path: format!("/projects/proj{}", i),
                collection: "projects".to_string(),
                tenant_id: format!("proj{}-tenant", i),
                parent_watch_id: None,
                submodule_path: None,
                git_remote_url: None,
                remote_hash: None,
                disambiguation_path: None,
                is_active: false,
                last_activity_at: None,
                is_paused: false,
                pause_start_time: None,
                is_archived: false,
                library_mode: None,
                follow_symlinks: false,
                enabled: true,
                cleanup_on_disable: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                last_scan: None,
            };
            manager.store_watch_folder(&record).await.unwrap();
        }

        // Create library watches
        for i in 1..=2 {
            let record = WatchFolderRecord {
                watch_id: format!("library-{}", i),
                path: format!("/libraries/lib{}", i),
                collection: "libraries".to_string(),
                tenant_id: format!("lib{}", i),
                parent_watch_id: None,
                submodule_path: None,
                git_remote_url: None,
                remote_hash: None,
                disambiguation_path: None,
                is_active: false,
                last_activity_at: None,
                is_paused: false,
                pause_start_time: None,
                is_archived: false,
                library_mode: None,
                follow_symlinks: false,
                enabled: true,
                cleanup_on_disable: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                last_scan: None,
            };
            manager.store_watch_folder(&record).await.unwrap();
        }

        // Test filter by collection
        let projects = manager.list_watch_folders(Some("projects"), false).await.unwrap();
        assert_eq!(projects.len(), 3);

        let libraries = manager.list_watch_folders(Some("libraries"), false).await.unwrap();
        assert_eq!(libraries.len(), 2);

        // Test no filter
        let all = manager.list_watch_folders(None, false).await.unwrap();
        assert_eq!(all.len(), 5);
    }

    #[tokio::test]
    async fn test_watch_folder_enabled_toggle() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("watch_enabled_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create enabled watch folder
        let record = WatchFolderRecord {
            watch_id: "toggle-test".to_string(),
            path: "/projects/toggle".to_string(),
            collection: "projects".to_string(),
            tenant_id: "toggle-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();

        // Verify initially enabled
        let retrieved = manager.get_watch_folder("toggle-test").await.unwrap().unwrap();
        assert!(retrieved.enabled);

        // Disable
        let updated = manager.set_watch_folder_enabled("toggle-test", false).await.unwrap();
        assert!(updated);
        let retrieved = manager.get_watch_folder("toggle-test").await.unwrap().unwrap();
        assert!(!retrieved.enabled);

        // List enabled only should not include disabled watch
        let enabled_only = manager.list_watch_folders(Some("projects"), true).await.unwrap();
        assert_eq!(enabled_only.len(), 0);

        // Re-enable
        let updated = manager.set_watch_folder_enabled("toggle-test", true).await.unwrap();
        assert!(updated);
        let retrieved = manager.get_watch_folder("toggle-test").await.unwrap().unwrap();
        assert!(retrieved.enabled);

        // List enabled only should now include watch
        let enabled_only = manager.list_watch_folders(Some("projects"), true).await.unwrap();
        assert_eq!(enabled_only.len(), 1);
    }

    #[tokio::test]
    async fn test_watch_folder_last_scan_update() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("watch_scan_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create watch folder
        let record = WatchFolderRecord {
            watch_id: "scan-test".to_string(),
            path: "/projects/scan".to_string(),
            collection: "projects".to_string(),
            tenant_id: "scan-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();

        // Verify no last_scan initially
        let retrieved = manager.get_watch_folder("scan-test").await.unwrap().unwrap();
        assert!(retrieved.last_scan.is_none());

        // Update last_scan
        let updated = manager.update_watch_folder_last_scan("scan-test").await.unwrap();
        assert!(updated);

        // Verify last_scan is now set
        let retrieved = manager.get_watch_folder("scan-test").await.unwrap().unwrap();
        assert!(retrieved.last_scan.is_some());
    }

    #[tokio::test]
    async fn test_get_watch_folder_by_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_tenant_lookup.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create a project watch folder
        let record = WatchFolderRecord {
            watch_id: "watch-001".to_string(),
            path: "/projects/myproject".to_string(),
            collection: "projects".to_string(),
            tenant_id: "abc123def456".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some("https://github.com/user/myproject.git".to_string()),
            remote_hash: Some("abc123hash".to_string()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();

        // Look up by tenant_id
        let found = manager.get_watch_folder_by_tenant_id("abc123def456", "projects")
            .await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().watch_id, "watch-001");

        // Look up non-existent tenant_id
        let not_found = manager.get_watch_folder_by_tenant_id("nonexistent", "projects")
            .await.unwrap();
        assert!(not_found.is_none());

        // Look up wrong collection
        let wrong_collection = manager.get_watch_folder_by_tenant_id("abc123def456", "libraries")
            .await.unwrap();
        assert!(wrong_collection.is_none());
    }

    #[tokio::test]
    async fn test_activate_project_by_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_tenant_activate.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create parent project
        let parent = WatchFolderRecord {
            watch_id: "parent-001".to_string(),
            path: "/projects/parent".to_string(),
            collection: "projects".to_string(),
            tenant_id: "parent-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&parent).await.unwrap();

        // Create submodule
        let submodule = WatchFolderRecord {
            watch_id: "submodule-001".to_string(),
            path: "/projects/parent/sub".to_string(),
            collection: "projects".to_string(),
            tenant_id: "sub-tenant".to_string(),
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some("sub".to_string()),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&submodule).await.unwrap();

        // Activate by tenant_id
        let (affected, watch_id) = manager.activate_project_by_tenant_id("parent-tenant")
            .await.unwrap();

        assert_eq!(affected, 2); // Parent and submodule
        assert_eq!(watch_id, Some("parent-001".to_string()));

        // Verify both are active
        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
        assert!(parent_record.is_active);
        assert!(submodule_record.is_active);
        assert!(parent_record.last_activity_at.is_some());
        assert!(submodule_record.last_activity_at.is_some());
    }

    #[tokio::test]
    async fn test_deactivate_project_by_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_tenant_deactivate.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create parent project (active)
        let parent = WatchFolderRecord {
            watch_id: "parent-001".to_string(),
            path: "/projects/parent".to_string(),
            collection: "projects".to_string(),
            tenant_id: "parent-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: Some(Utc::now()),
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&parent).await.unwrap();

        // Create submodule (active)
        let submodule = WatchFolderRecord {
            watch_id: "submodule-001".to_string(),
            path: "/projects/parent/sub".to_string(),
            collection: "projects".to_string(),
            tenant_id: "sub-tenant".to_string(),
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some("sub".to_string()),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: Some(Utc::now()),
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&submodule).await.unwrap();

        // Deactivate by tenant_id
        let (affected, watch_id) = manager.deactivate_project_by_tenant_id("parent-tenant")
            .await.unwrap();

        assert_eq!(affected, 2); // Parent and submodule
        assert_eq!(watch_id, Some("parent-001".to_string()));

        // Verify both are inactive
        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
        assert!(!parent_record.is_active);
        assert!(!submodule_record.is_active);
    }

    #[tokio::test]
    async fn test_heartbeat_project_group() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_heartbeat.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create parent project
        let parent = WatchFolderRecord {
            watch_id: "parent-001".to_string(),
            path: "/projects/parent".to_string(),
            collection: "projects".to_string(),
            tenant_id: "parent-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: None, // No activity yet
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&parent).await.unwrap();

        // Create submodule
        let submodule = WatchFolderRecord {
            watch_id: "submodule-001".to_string(),
            path: "/projects/parent/sub".to_string(),
            collection: "projects".to_string(),
            tenant_id: "sub-tenant".to_string(),
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some("sub".to_string()),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: None, // No activity yet
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&submodule).await.unwrap();

        // Send heartbeat
        let affected = manager.heartbeat_project_group("parent-001").await.unwrap();
        assert_eq!(affected, 2); // Parent and submodule

        // Verify both have activity timestamps
        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
        assert!(parent_record.last_activity_at.is_some());
        assert!(submodule_record.last_activity_at.is_some());
    }

    #[tokio::test]
    async fn test_heartbeat_project_by_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_heartbeat_tenant.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create parent project
        let parent = WatchFolderRecord {
            watch_id: "parent-001".to_string(),
            path: "/projects/parent".to_string(),
            collection: "projects".to_string(),
            tenant_id: "parent-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&parent).await.unwrap();

        // Create submodule
        let submodule = WatchFolderRecord {
            watch_id: "submodule-001".to_string(),
            path: "/projects/parent/sub".to_string(),
            collection: "projects".to_string(),
            tenant_id: "sub-tenant".to_string(),
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some("sub".to_string()),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&submodule).await.unwrap();

        // Heartbeat by tenant_id
        let (affected, watch_id) = manager.heartbeat_project_by_tenant_id("parent-tenant")
            .await.unwrap();

        assert_eq!(affected, 2); // Parent and submodule
        assert_eq!(watch_id, Some("parent-001".to_string()));

        // Verify both have activity timestamps
        let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
        let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
        assert!(parent_record.last_activity_at.is_some());
        assert!(submodule_record.last_activity_at.is_some());
    }

    #[tokio::test]
    async fn test_recursive_activity_inheritance_3_levels() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_3level_recursive.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create 3-level hierarchy: root -> mid -> leaf
        let root = WatchFolderRecord {
            watch_id: "root-001".to_string(),
            path: "/projects/root".to_string(),
            collection: "projects".to_string(),
            tenant_id: "root-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&root).await.unwrap();

        let mid = WatchFolderRecord {
            watch_id: "mid-001".to_string(),
            path: "/projects/root/libs/mid".to_string(),
            collection: "projects".to_string(),
            tenant_id: "mid-tenant".to_string(),
            parent_watch_id: Some("root-001".to_string()),
            submodule_path: Some("libs/mid".to_string()),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&mid).await.unwrap();

        let leaf = WatchFolderRecord {
            watch_id: "leaf-001".to_string(),
            path: "/projects/root/libs/mid/deps/leaf".to_string(),
            collection: "projects".to_string(),
            tenant_id: "leaf-tenant".to_string(),
            parent_watch_id: Some("mid-001".to_string()),
            submodule_path: Some("deps/leaf".to_string()),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&leaf).await.unwrap();

        // Activate from root should activate all 3 levels
        let affected = manager.activate_project_group("root-001").await.unwrap();
        assert_eq!(affected, 3);

        let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
        let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
        let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
        assert!(root_r.is_active);
        assert!(mid_r.is_active);
        assert!(leaf_r.is_active);
        assert!(root_r.last_activity_at.is_some());
        assert!(mid_r.last_activity_at.is_some());
        assert!(leaf_r.last_activity_at.is_some());

        // Heartbeat from root should touch all 3 levels
        let hb = manager.heartbeat_project_group("root-001").await.unwrap();
        assert_eq!(hb, 3);

        // Deactivate from root should deactivate all 3 levels
        let deact = manager.deactivate_project_group("root-001").await.unwrap();
        assert_eq!(deact, 3);

        let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
        let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
        let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
        assert!(!root_r.is_active);
        assert!(!mid_r.is_active);
        assert!(!leaf_r.is_active);

        // Activate from mid should only activate mid and leaf (not root)
        let affected = manager.activate_project_group("mid-001").await.unwrap();
        assert_eq!(affected, 2);

        let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
        let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
        let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
        assert!(!root_r.is_active); // Root stays inactive
        assert!(mid_r.is_active);
        assert!(leaf_r.is_active);
    }

    #[tokio::test]
    async fn test_activate_nonexistent_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_nonexistent.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Activate non-existent tenant should return 0 affected
        let (affected, watch_id) = manager.activate_project_by_tenant_id("nonexistent")
            .await.unwrap();

        assert_eq!(affected, 0);
        assert!(watch_id.is_none());
    }

    #[tokio::test]
    async fn test_with_pool_constructor() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_with_pool.db");

        // Create manager with new() to get the pool set up
        let manager1 = DaemonStateManager::new(&db_path).await.unwrap();
        manager1.initialize().await.unwrap();

        // Create a watch folder
        let record = WatchFolderRecord {
            watch_id: "test-001".to_string(),
            path: "/projects/test".to_string(),
            collection: "projects".to_string(),
            tenant_id: "test-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager1.store_watch_folder(&record).await.unwrap();

        // Create another manager using with_pool - simulate sharing pool
        // Note: This simulates the gRPC service scenario
        let connect_options = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(false);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await
            .unwrap();

        let manager2 = DaemonStateManager::with_pool(pool);

        // Should be able to read the data
        let found = manager2.get_watch_folder("test-001").await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().tenant_id, "test-tenant");
    }

    // ========================================================================
    // Disambiguation Tests (Task 3)
    // ========================================================================

    #[tokio::test]
    async fn test_find_clones_by_remote_hash() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_find_clones.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create two projects with the same remote_hash (same repo, different paths)
        let remote_hash = "abc123hashxyz";

        let record1 = WatchFolderRecord {
            watch_id: "clone-001".to_string(),
            path: "/home/user/work/project".to_string(),
            collection: "projects".to_string(),
            tenant_id: "project-tenant-1".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some("https://github.com/user/repo.git".to_string()),
            remote_hash: Some(remote_hash.to_string()),
            disambiguation_path: Some("work/project".to_string()),
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        let record2 = WatchFolderRecord {
            watch_id: "clone-002".to_string(),
            path: "/home/user/personal/project".to_string(),
            collection: "projects".to_string(),
            tenant_id: "project-tenant-2".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some("https://github.com/user/repo.git".to_string()),
            remote_hash: Some(remote_hash.to_string()),
            disambiguation_path: Some("personal/project".to_string()),
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        manager.store_watch_folder(&record1).await.unwrap();
        manager.store_watch_folder(&record2).await.unwrap();

        // Find clones by remote_hash
        let clones = manager.find_clones_by_remote_hash(remote_hash).await.unwrap();
        assert_eq!(clones.len(), 2);

        // Verify different tenant_ids
        let tenant_ids: Vec<_> = clones.iter().map(|c| &c.tenant_id).collect();
        assert!(tenant_ids.contains(&&"project-tenant-1".to_string()));
        assert!(tenant_ids.contains(&&"project-tenant-2".to_string()));

        // Search for non-existent hash
        let no_clones = manager.find_clones_by_remote_hash("nonexistent").await.unwrap();
        assert!(no_clones.is_empty());
    }

    #[tokio::test]
    async fn test_register_project_with_disambiguation_first_clone() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_disambig_first.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        use crate::project_disambiguation::ProjectIdCalculator;
        let calculator = ProjectIdCalculator::new();

        // Register first clone (no disambiguation needed)
        let remote_hash = calculator.calculate_remote_hash("https://github.com/user/repo.git");
        let tenant_id = calculator.calculate(
            std::path::Path::new("/home/user/work/project"),
            Some("https://github.com/user/repo.git"),
            None,
        );

        let record = WatchFolderRecord {
            watch_id: "first-clone".to_string(),
            path: "/home/user/work/project".to_string(),
            collection: "projects".to_string(),
            tenant_id: tenant_id.clone(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some("https://github.com/user/repo.git".to_string()),
            remote_hash: Some(remote_hash.clone()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        let (result, aliases) = manager.register_project_with_disambiguation(record).await.unwrap();

        // First clone should have no aliases
        assert!(aliases.is_empty());

        // First clone should not need disambiguation
        assert!(result.disambiguation_path.is_none() || result.disambiguation_path.as_deref() == Some(""));
    }

    #[tokio::test]
    async fn test_register_project_with_disambiguation_second_clone() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_disambig_second.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        use crate::project_disambiguation::ProjectIdCalculator;
        let calculator = ProjectIdCalculator::new();

        let git_remote = "https://github.com/user/repo.git";
        let remote_hash = calculator.calculate_remote_hash(git_remote);

        // Register first clone
        let first_record = WatchFolderRecord {
            watch_id: "first-clone".to_string(),
            path: "/home/user/work/project".to_string(),
            collection: "projects".to_string(),
            tenant_id: calculator.calculate(
                std::path::Path::new("/home/user/work/project"),
                Some(git_remote),
                None,
            ),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some(git_remote.to_string()),
            remote_hash: Some(remote_hash.clone()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        let (first_result, _) = manager.register_project_with_disambiguation(first_record).await.unwrap();
        let original_tenant_id = first_result.tenant_id.clone();

        // Register second clone (should trigger disambiguation for both)
        let second_record = WatchFolderRecord {
            watch_id: "second-clone".to_string(),
            path: "/home/user/personal/project".to_string(),
            collection: "projects".to_string(),
            tenant_id: calculator.calculate(
                std::path::Path::new("/home/user/personal/project"),
                Some(git_remote),
                None,
            ),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: Some(git_remote.to_string()),
            remote_hash: Some(remote_hash.clone()),
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        let (second_result, aliases) = manager.register_project_with_disambiguation(second_record).await.unwrap();

        // Second clone should have disambiguation path
        assert!(second_result.disambiguation_path.is_some());
        let second_disambig = second_result.disambiguation_path.as_ref().unwrap();
        assert!(second_disambig.contains("personal"), "Expected 'personal' in disambiguation path: {}", second_disambig);

        // Verify first clone was updated with disambiguation
        let updated_first = manager.get_watch_folder("first-clone").await.unwrap().unwrap();
        assert!(updated_first.disambiguation_path.is_some());
        let first_disambig = updated_first.disambiguation_path.as_ref().unwrap();
        assert!(first_disambig.contains("work"), "Expected 'work' in disambiguation path: {}", first_disambig);

        // Verify tenant_ids are now different
        assert_ne!(updated_first.tenant_id, second_result.tenant_id,
            "Both clones should have different tenant_ids");

        // Verify alias was created for the first clone
        assert!(!aliases.is_empty(), "Should have created alias for first clone");
        let (old_id, new_id) = &aliases[0];
        assert_eq!(old_id, &original_tenant_id, "Alias should map from original tenant_id");
        assert_eq!(new_id, &updated_first.tenant_id, "Alias should map to new tenant_id");
    }

    #[tokio::test]
    async fn test_is_path_registered() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_path_registered.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let record = WatchFolderRecord {
            watch_id: "test-project".to_string(),
            path: "/home/user/myproject".to_string(),
            collection: "projects".to_string(),
            tenant_id: "test-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };

        // Path should not be registered initially
        assert!(!manager.is_path_registered("/home/user/myproject").await.unwrap());

        // Register the project
        manager.store_watch_folder(&record).await.unwrap();

        // Path should now be registered
        assert!(manager.is_path_registered("/home/user/myproject").await.unwrap());

        // Different path should not be registered
        assert!(!manager.is_path_registered("/home/user/other").await.unwrap());
    }

    #[tokio::test]
    async fn test_any_watchers_paused() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("pause_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // No watchers at all - should return false
        assert!(!manager.any_watchers_paused().await.unwrap());

        // Add an enabled, non-paused watcher
        let record = WatchFolderRecord {
            watch_id: "pause-test-001".to_string(),
            path: "/projects/pause-test".to_string(),
            collection: "projects".to_string(),
            tenant_id: "pause-test-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();

        // Not paused yet
        assert!(!manager.any_watchers_paused().await.unwrap());

        // Pause all watchers
        let paused = manager.pause_all_watchers().await.unwrap();
        assert_eq!(paused, 1);
        assert!(manager.any_watchers_paused().await.unwrap());

        // Resume
        let resumed = manager.resume_all_watchers().await.unwrap();
        assert_eq!(resumed, 1);
        assert!(!manager.any_watchers_paused().await.unwrap());
    }

    #[tokio::test]
    async fn test_poll_pause_state() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("poll_pause_test.db");

        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        let flag = std::sync::atomic::AtomicBool::new(false);

        // No change when no watchers
        let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
        assert!(!changed);
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));

        // Add enabled watcher
        let record = WatchFolderRecord {
            watch_id: "poll-test-001".to_string(),
            path: "/projects/poll-test".to_string(),
            collection: "projects".to_string(),
            tenant_id: "poll-test-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();

        // Still no change (not paused)
        let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
        assert!(!changed);

        // Pause via DB
        manager.pause_all_watchers().await.unwrap();

        // Flag should change from false -> true
        let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
        assert!(changed);
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));

        // Poll again - no change (already true)
        let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
        assert!(!changed);

        // Resume via DB
        manager.resume_all_watchers().await.unwrap();

        // Flag should change from true -> false
        let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
        assert!(changed);
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_deactivate_inactive_projects() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_inactivity.db");
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create an active project with old last_activity_at
        let old_project = WatchFolderRecord {
            watch_id: "old-project".to_string(),
            path: "/projects/old".to_string(),
            collection: "projects".to_string(),
            tenant_id: "old-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: Some(Utc::now() - chrono::Duration::hours(13)),
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&old_project).await.unwrap();

        // Create an active project with recent activity
        let recent_project = WatchFolderRecord {
            watch_id: "recent-project".to_string(),
            path: "/projects/recent".to_string(),
            collection: "projects".to_string(),
            tenant_id: "recent-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: Some(Utc::now() - chrono::Duration::minutes(30)),
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&recent_project).await.unwrap();

        // 12-hour timeout (43200 seconds)
        let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
        assert_eq!(deactivated, 1, "Only the 13h-old project should be deactivated");

        // Verify old project is deactivated
        let old = manager.get_watch_folder("old-project").await.unwrap().unwrap();
        assert!(!old.is_active);

        // Verify recent project is still active
        let recent = manager.get_watch_folder("recent-project").await.unwrap().unwrap();
        assert!(recent.is_active);
    }

    #[tokio::test]
    async fn test_deactivate_inactive_skips_null_activity() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_inactivity_null.db");
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create active project with NULL last_activity_at (never had a session)
        let null_project = WatchFolderRecord {
            watch_id: "null-project".to_string(),
            path: "/projects/null".to_string(),
            collection: "projects".to_string(),
            tenant_id: "null-tenant".to_string(),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&null_project).await.unwrap();

        // Should not deactivate projects with NULL last_activity_at
        let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
        assert_eq!(deactivated, 0);

        let record = manager.get_watch_folder("null-project").await.unwrap().unwrap();
        assert!(record.is_active);
    }
}
