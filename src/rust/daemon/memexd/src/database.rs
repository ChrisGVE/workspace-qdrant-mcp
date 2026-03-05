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
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(format!("{}/.workspace-qdrant/state.db", home))
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

/// Run startup reconciliation: clean stale state and validate watch folders (Task 512).
pub async fn run_reconciliation(queue_pool: &SqlitePool) {
    info!("Running startup reconciliation...");

    match workspace_qdrant_core::startup::clean_stale_state(queue_pool).await {
        Ok(stats) => {
            if stats.has_changes() {
                info!(
                    "Stale state cleanup: {} items reset, {} old items cleaned, \
                     {} stale tracked files removed, {} orphan chunks removed",
                    stats.items_reset,
                    stats.items_cleaned,
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

    info!("Startup reconciliation complete");
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
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    if let Ok(override_path) = std::env::var("WQM_DATABASE_PATH") {
        return PathBuf::from(override_path);
    }
    PathBuf::from(format!("{}/.workspace-qdrant/state.db", home))
}
