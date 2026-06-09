//! Phase 2: Database pool creation, schema migrations, and startup reconciliation.
//!
//! Initializes the SQLite connection pool, runs schema migrations (ADR-003),
//! sets up the search and graph databases, and performs startup reconciliation
//! to clean stale state and validate watch folders.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{info, warn};

use workspace_qdrant_core::{
    config::Config, poll_pause_state, queue_config::QueueConnectionConfig, SchemaManager,
    SearchDbManager,
};

/// Concrete graph store type used throughout the daemon.
pub type ConcreteGraphStore =
    workspace_qdrant_core::graph::SharedGraphStore<workspace_qdrant_core::graph::SqliteGraphStore>;

/// Result of database initialization containing all database handles.
pub struct DatabaseHandles {
    /// Main SQLite pool for queue operations and state management.
    pub queue_pool: SqlitePool,
    /// FTS5 search database manager.
    pub search_db: Arc<SearchDbManager>,
    /// Graph store (if initialization succeeded).
    pub graph_store: Option<ConcreteGraphStore>,
    /// Shared pause flag initialized from database state.
    pub pause_flag: Arc<std::sync::atomic::AtomicBool>,
}

/// Create the SQLite connection pool and run schema migrations.
pub async fn create_pool_and_migrate(
    config: &Config,
) -> Result<SqlitePool, Box<dyn std::error::Error>> {
    let db_path = config.database_path.clone().unwrap_or_else(|| {
        wqm_common::paths::get_database_path()
            .unwrap_or_else(|_| PathBuf::from("/tmp/workspace-qdrant-state.db"))
    });

    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    // Create SQLite connection pool for queue operations and ProjectService
    let queue_config = QueueConnectionConfig::with_database_path(&db_path);
    let queue_pool = queue_config
        .create_pool()
        .await
        .map_err(|e| format!("Failed to create queue database pool: {}", e))?;

    info!("Queue database pool created at: {}", db_path.display());

    // Run schema migrations (ADR-003: daemon owns database and schema)
    info!("Running database schema migrations...");
    let schema_manager = SchemaManager::new(queue_pool.clone());
    schema_manager
        .run_migrations()
        .await
        .map_err(|e| format!("Failed to run schema migrations: {}", e))?;
    info!("Schema migrations complete");

    Ok(queue_pool)
}

/// Initialize the FTS5 search database (Task 45).
pub async fn init_search_db(
    queue_pool: &SqlitePool,
) -> Result<Arc<SearchDbManager>, Box<dyn std::error::Error>> {
    let db_path = get_state_db_path(queue_pool);
    let search_db_path = workspace_qdrant_core::search_db::search_db_path_from_state(&db_path);
    info!(
        "Initializing search database at: {}",
        search_db_path.display()
    );
    let search_db = SearchDbManager::new(&search_db_path)
        .await
        .map_err(|e| format!("Failed to initialize search database: {}", e))?;
    let search_db_version = search_db
        .get_schema_version()
        .await
        .map_err(|e| format!("Failed to read search DB version: {}", e))?;
    info!(
        "Search database initialized at version {:?}",
        search_db_version
    );
    Ok(Arc::new(search_db))
}

/// Initialize the graph database for code relationship tracking (graph-rag).
pub async fn init_graph_db(queue_pool: &SqlitePool) -> Option<ConcreteGraphStore> {
    let db_path = get_state_db_path(queue_pool);
    let graph_db_path = db_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join(workspace_qdrant_core::graph::GRAPH_DB_FILENAME);
    info!(
        "Initializing graph database at: {}",
        graph_db_path.display()
    );
    match workspace_qdrant_core::graph::GraphDbManager::new(&graph_db_path).await {
        Ok(manager) => {
            info!(
                "Graph database initialized at version {}",
                workspace_qdrant_core::graph::GRAPH_SCHEMA_VERSION
            );
            let store = workspace_qdrant_core::graph::SqliteGraphStore::new(manager.pool().clone());
            Some(workspace_qdrant_core::graph::SharedGraphStore::new(store))
        }
        Err(e) => {
            warn!(
                "Graph database initialization failed: {} (graph features disabled)",
                e
            );
            None
        }
    }
}

/// Run the **fast** half of startup reconciliation.
///
/// Only SQL-only cleanup and watch-folder path validation run here — all
/// operations are either set-based or at worst O(watch_folders) and
/// complete in milliseconds even on large projects. The slow ignore-rule
/// reconciliation is deferred to `spawn_background_reconciliation` so
/// that gRPC readiness does not regress on projects with many files
/// (issue #59).
pub async fn run_reconciliation(queue_pool: &SqlitePool) {
    info!("Running startup reconciliation (fast path)...");

    let queue_manager =
        workspace_qdrant_core::queue_operations::QueueManager::new(queue_pool.clone());
    match workspace_qdrant_core::startup::clean_stale_state(queue_pool, &queue_manager).await {
        Ok(stats) => {
            if stats.has_changes() {
                info!(
                    "Stale state cleanup: {} items reset, {} old items cleaned, \
                     {} Delete(s) enqueued for missing files, \
                     {} stale tracked files removed, {} orphan chunks removed",
                    stats.items_reset,
                    stats.items_cleaned,
                    stats.deletes_enqueued,
                    stats.tracked_files_removed,
                    stats.orphan_chunks_removed
                );
            } else {
                info!("Stale state cleanup: no stale state found");
            }
        }
        Err(e) => warn!("Stale state cleanup failed (non-fatal): {}", e),
    }

    match workspace_qdrant_core::startup::validate_watch_folders(queue_pool).await {
        Ok(stats) => {
            if stats.folders_deactivated > 0 {
                warn!(
                    "Watch folder validation: {} of {} folders deactivated (paths no longer exist)",
                    stats.folders_deactivated, stats.folders_checked
                );
            } else {
                info!(
                    "Watch folder validation: all {} folders valid",
                    stats.folders_checked
                );
            }
        }
        Err(e) => warn!("Watch folder validation failed (non-fatal): {}", e),
    }

    info!("Startup reconciliation (fast path) complete");
}

/// Spawn the **slow** half of startup reconciliation in the background.
///
/// Ignore-rule reconciliation walks the filesystem and enqueues `file/add`
/// / `file/delete` items for any drift, which on large projects (e.g.
/// 90K+ files) can take several minutes. Calling this from the main
/// startup path delayed gRPC readiness for that entire window. Running
/// it on a spawned task lets gRPC come up immediately while the reconcile
/// proceeds concurrently (issue #59).
pub fn spawn_background_reconciliation(queue_pool: SqlitePool) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let qm = Arc::new(workspace_qdrant_core::queue_operations::QueueManager::new(
            queue_pool.clone(),
        ));
        let started = std::time::Instant::now();
        info!("[startup-bg] Starting background ignore reconciliation...");
        match workspace_qdrant_core::startup::reconcile_all_ignore_rules(&queue_pool, &qm).await {
            Ok(stats) if stats.stale_deleted > 0 || stats.missing_added > 0 => {
                info!(
                    "[startup-bg] Ignore reconciliation complete in {:?}: \
                     {} stale deleted, {} missing added",
                    started.elapsed(),
                    stats.stale_deleted,
                    stats.missing_added
                );
            }
            Ok(_) => info!(
                "[startup-bg] Ignore reconciliation complete in {:?}: index consistent",
                started.elapsed()
            ),
            Err(e) => warn!(
                "[startup-bg] Ignore reconciliation failed after {:?}: {}",
                started.elapsed(),
                e
            ),
        }

        // Purge documents for branches deleted from git. The file watcher
        // excludes `.git/`, so branch deletions are never observed live — this
        // reconciliation is the canonical purge path (otherwise a deleted branch
        // leaves its indexed documents orphaned forever). Runs after the ignore
        // reconcile in the same background task. Non-fatal.
        match workspace_qdrant_core::startup::reconciliation::branch_prune::prune_orphaned_branches(
            &queue_pool,
            &qm,
        )
        .await
        {
            Ok(s) if s.branches_pruned > 0 => info!(
                "[startup-bg] Branch prune complete: {} orphaned branch(es), \
                 {} file delete(s) enqueued",
                s.branches_pruned, s.files_enqueued
            ),
            Ok(_) => {}
            Err(e) => warn!("[startup-bg] Branch prune failed: {}", e),
        }
    })
}

/// Initialize the shared pause flag from database state (Task 543.16).
pub async fn init_pause_flag(queue_pool: &SqlitePool) -> Arc<std::sync::atomic::AtomicBool> {
    let pause_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if let Ok(changed) = poll_pause_state(queue_pool, &pause_flag).await {
        if pause_flag.load(std::sync::atomic::Ordering::SeqCst) {
            info!("Restored pause state from database: watchers are PAUSED");
        } else if changed {
            info!("Pause state synced from database: watchers are ACTIVE");
        }
    }
    pause_flag
}

/// Initialize all database handles in one call.
pub async fn initialize_all(
    config: &Config,
) -> Result<DatabaseHandles, Box<dyn std::error::Error>> {
    let queue_pool = create_pool_and_migrate(config).await?;
    let search_db = init_search_db(&queue_pool).await?;
    let graph_store = init_graph_db(&queue_pool).await;
    run_reconciliation(&queue_pool).await;
    let pause_flag = init_pause_flag(&queue_pool).await;

    Ok(DatabaseHandles {
        queue_pool,
        search_db,
        graph_store,
        pause_flag,
    })
}

// -- private helpers --

/// Derive the state.db path (best-effort: re-read from environment).
fn get_state_db_path(_pool: &SqlitePool) -> PathBuf {
    if let Ok(override_path) = std::env::var("WQM_DATABASE_PATH") {
        return PathBuf::from(override_path);
    }
    wqm_common::paths::get_database_path()
        .unwrap_or_else(|_| PathBuf::from("/tmp/workspace-qdrant-state.db"))
}
