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

/// Type-erased graph store handle used by the ingestion/query path
/// (queue processor, `ProcessingContext`). The concrete backend (SQLite CTE
/// or LadybugDB) is selected at runtime from `config.graph.backend`.
pub type ConcreteGraphStore = Arc<dyn workspace_qdrant_core::graph::GraphStore>;

/// SQLite-only graph handle for the gRPC `GraphService`.
///
/// The graph analytics handlers (PageRank, community detection, betweenness)
/// and `migrate` operate directly on a `SqlitePool` (`store.pool()`), so they
/// are inherently SQLite-bound. When the LadybugDB backend is selected this is
/// `None` and `GraphService` is not registered — analytics and migration are a
/// SQLite-only feature by design.
pub type GraphServiceStore =
    workspace_qdrant_core::graph::SharedGraphStore<workspace_qdrant_core::graph::SqliteGraphStore>;

/// Result of database initialization containing all database handles.
pub struct DatabaseHandles {
    /// Main SQLite pool for queue operations and state management.
    pub queue_pool: SqlitePool,
    /// FTS5 search database manager.
    pub search_db: Arc<SearchDbManager>,
    /// Type-erased graph store for the queue processor (backend-selectable).
    pub graph_store: Option<ConcreteGraphStore>,
    /// SQLite-typed graph store for `GraphService` (None when backend is ladybug).
    pub graph_sqlite: Option<GraphServiceStore>,
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

/// Result of graph backend initialization.
///
/// `processing` is the type-erased handle used by the queue processor and is
/// populated for every successfully-initialized backend. `service` is the
/// SQLite-typed handle for `GraphService`; it is only `Some` when the SQLite
/// backend is active.
#[derive(Default)]
pub struct GraphStores {
    pub processing: Option<ConcreteGraphStore>,
    pub service: Option<GraphServiceStore>,
}

/// Initialize the graph database for code relationship tracking (graph-rag).
///
/// The backend is selected at runtime from `config.graph.backend`:
/// - `sqlite` (default): recursive-CTE store; also drives `GraphService`.
/// - `ladybug`: LadybugDB (Kuzu) store, only available when compiled with the
///   `ladybug` feature. Analytics/migration (`GraphService`) are unavailable.
///
/// Returns empty handles (graph features disabled) on initialization failure.
pub async fn init_graph_db(config: &Config, queue_pool: &SqlitePool) -> GraphStores {
    use workspace_qdrant_core::graph::{factory, GraphBackend};

    let db_path = get_state_db_path(queue_pool);
    let db_dir = config.graph.db_dir.clone().unwrap_or_else(|| {
        db_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf()
    });

    // Fail fast with a clear message if the configured backend is not
    // compiled in (e.g. `ladybug` without the feature flag).
    if let Err(e) = factory::validate_backend(&config.graph.backend) {
        warn!("Graph backend unavailable: {} (graph features disabled)", e);
        return GraphStores::default();
    }

    match config.graph.backend {
        GraphBackend::Sqlite => {
            match factory::create_sqlite_graph_store_with_rag(&db_dir, &config.graph_rag).await {
                Ok(store) => {
                    info!(
                        "Graph database initialized (SQLite backend, version {})",
                        workspace_qdrant_core::graph::GRAPH_SCHEMA_VERSION
                    );
                    GraphStores {
                        processing: Some(Arc::new(store.clone())),
                        service: Some(store),
                    }
                }
                Err(e) => {
                    warn!(
                        "Graph database initialization failed: {} (graph features disabled)",
                        e
                    );
                    GraphStores::default()
                }
            }
        }
        GraphBackend::Ladybug => {
            init_ladybug_graph_db(&db_dir, &config.graph, &config.graph_rag).await
        }
    }
}

/// Initialize the LadybugDB graph backend (only compiled with `ladybug`).
#[cfg(feature = "ladybug")]
async fn init_ladybug_graph_db(
    db_dir: &std::path::Path,
    graph_config: &workspace_qdrant_core::graph::GraphConfig,
    graph_rag: &workspace_qdrant_core::config::GraphRagConfig,
) -> GraphStores {
    match workspace_qdrant_core::graph::factory::create_ladybug_graph_store(
        db_dir,
        graph_config,
        graph_rag,
    )
    .await
    {
        Ok(store) => {
            info!("Graph database initialized (LadybugDB backend); GraphService analytics/migration are SQLite-only and disabled");
            GraphStores {
                processing: Some(Arc::new(store)),
                service: None,
            }
        }
        Err(e) => {
            warn!(
                "LadybugDB graph initialization failed: {} (graph features disabled)",
                e
            );
            GraphStores::default()
        }
    }
}

/// Fallback when the `ladybug` feature is not compiled in. `validate_backend`
/// already rejects the ladybug backend before this is reached, so this only
/// guards against logic drift.
#[cfg(not(feature = "ladybug"))]
async fn init_ladybug_graph_db(
    _db_dir: &std::path::Path,
    _graph_config: &workspace_qdrant_core::graph::GraphConfig,
    _graph_rag: &workspace_qdrant_core::config::GraphRagConfig,
) -> GraphStores {
    warn!("LadybugDB backend requested but the 'ladybug' feature is not enabled (graph features disabled)");
    GraphStores::default()
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

    // Recompute workspace and git org groups from scratch so that groups
    // stay consistent after projects are added or removed between runs.
    let group_stats = workspace_qdrant_core::startup::reconcile_project_groups(queue_pool).await;
    if group_stats.workspace_groups > 0 || group_stats.git_org_groups > 0 {
        info!(
            "Project groups reconciled: {} workspace groups, {} git org groups",
            group_stats.workspace_groups, group_stats.git_org_groups
        );
    }

    info!("Startup reconciliation (fast path) complete");
}

/// Spawn the **slow** half of startup reconciliation in the background.
///
/// Three phases:
/// 1. Ignore-rule reconciliation: walk filesystem, enqueue `file/add` /
///    `file/delete` for any ignore drift (issue #59).
/// 2. FTS orphan prune: remove search.db FTS entries whose `file_id` is no
///    longer in `tracked_files`, so `grep`/`search` stop returning ghosts of
///    files deleted from disk (issue #130).
/// 3. Full-scan reconciliation: enqueue a `rebuild=true` scan for each
///    enabled project so that missing/changed/deleted files are discovered
///    without relying on `last_scan` mtime pruning.
///
/// Running on a spawned task lets gRPC come up immediately — a backlog of
/// thousands of FTS orphan deletes must not gate readiness (issue #59).
pub fn spawn_background_reconciliation(
    queue_pool: SqlitePool,
    search_db: Arc<SearchDbManager>,
) -> tokio::task::JoinHandle<()> {
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

        // #130: prune search.db FTS rows orphaned from `tracked_files`.
        let fts_started = std::time::Instant::now();
        match workspace_qdrant_core::startup::prune_orphan_fts_entries(&queue_pool, &search_db)
            .await
        {
            Ok(removed) if removed > 0 => info!(
                "[startup-bg] FTS orphan prune complete in {:?}: {} orphaned file(s) removed",
                fts_started.elapsed(),
                removed
            ),
            Ok(_) => info!(
                "[startup-bg] FTS orphan prune complete in {:?}: no orphans",
                fts_started.elapsed()
            ),
            Err(e) => warn!(
                "[startup-bg] FTS orphan prune failed after {:?}: {}",
                fts_started.elapsed(),
                e
            ),
        }

        enqueue_startup_scans(&queue_pool, &qm).await;
        trigger_semantic_upgrade(&queue_pool, &qm).await;
    })
}

/// Enqueue a rebuild scan for each enabled project watch folder.
///
/// This ensures that files added/modified/deleted while the daemon was
/// down are discovered on the next startup. The scan uses `rebuild=true`
/// to bypass `last_scan` mtime pruning. The `file/update` path compares
/// hashes against tracked_files and skips unchanged files cheaply.
async fn enqueue_startup_scans(
    pool: &SqlitePool,
    qm: &Arc<workspace_qdrant_core::queue_operations::QueueManager>,
) {
    let folders: Vec<(String, String, String)> = match sqlx::query_as(
        "SELECT watch_id, tenant_id, path FROM watch_folders \
         WHERE enabled = 1 AND collection = 'projects'",
    )
    .fetch_all(pool)
    .await
    {
        Ok(f) => f,
        Err(e) => {
            warn!(
                "[startup-bg] Failed to query watch_folders for reconciliation scans: {}",
                e
            );
            return;
        }
    };

    let mut enqueued = 0u32;
    for (_watch_id, tenant_id, path) in &folders {
        let payload = serde_json::json!({
            "project_root": path,
            "rebuild": true,
        });
        let payload_json = payload.to_string();

        match qm
            .enqueue_unified(
                workspace_qdrant_core::unified_queue_schema::ItemType::Tenant,
                workspace_qdrant_core::unified_queue_schema::QueueOperation::Scan,
                tenant_id,
                "projects",
                &payload_json,
                None,
                None,
            )
            .await
        {
            Ok((_, true)) => enqueued += 1,
            Ok((_, false)) => {}
            Err(e) => {
                warn!(
                    "[startup-bg] Failed to enqueue reconciliation scan for {}: {}",
                    tenant_id, e
                );
            }
        }
    }

    if enqueued > 0 {
        info!(
            "[startup-bg] Enqueued {} reconciliation scans (rebuild=true)",
            enqueued
        );
    }
}

/// One-time migration: reset treesitter_status for files that were incorrectly
/// marked 'done' by the old text-only chunking path, then trigger capability
/// upgrade to re-process them with proper semantic chunking.
///
/// Uses a lightweight `startup_migrations` table to ensure this runs only once.
async fn trigger_semantic_upgrade(
    pool: &SqlitePool,
    qm: &Arc<workspace_qdrant_core::queue_operations::QueueManager>,
) {
    let migration_key = "semantic_chunking_upgrade_v1";

    let _ = sqlx::query(
        "CREATE TABLE IF NOT EXISTS startup_migrations (\
         key TEXT PRIMARY KEY, applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))",
    )
    .execute(pool)
    .await;

    let already_done: bool =
        sqlx::query_scalar("SELECT COUNT(*) > 0 FROM startup_migrations WHERE key = ?1")
            .bind(migration_key)
            .fetch_one(pool)
            .await
            .unwrap_or(false);

    if already_done {
        return;
    }

    let reset = sqlx::query(
        "UPDATE tracked_files SET treesitter_status = 'none' \
         WHERE treesitter_status = 'done'",
    )
    .execute(pool)
    .await;

    let reset_count = match reset {
        Ok(r) => r.rows_affected(),
        Err(e) => {
            warn!("[startup-bg] Failed to reset treesitter_status: {}", e);
            return;
        }
    };

    if reset_count > 0 {
        info!(
            "[startup-bg] Reset treesitter_status for {} files (semantic chunking upgrade)",
            reset_count
        );

        let tenants: Vec<String> =
            sqlx::query_scalar("SELECT DISTINCT tenant_id FROM watch_folders WHERE enabled = 1")
                .fetch_all(pool)
                .await
                .unwrap_or_default();

        let mut total_enqueued = 0u32;
        for tenant_id in &tenants {
            total_enqueued +=
                workspace_qdrant_core::strategies::capability_upgrade::trigger_capability_upgrade(
                    pool,
                    qm,
                    tenant_id,
                    workspace_qdrant_core::tracked_files_schema::UpgradeReason::GrammarAvailable,
                    None,
                )
                .await;
        }

        info!(
            "[startup-bg] Semantic chunking upgrade: {} files enqueued for re-processing",
            total_enqueued
        );
    }

    let _ = sqlx::query("INSERT OR IGNORE INTO startup_migrations (key) VALUES (?1)")
        .bind(migration_key)
        .execute(pool)
        .await;
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
    let graph_stores = init_graph_db(config, &queue_pool).await;
    run_reconciliation(&queue_pool).await;
    let pause_flag = init_pause_flag(&queue_pool).await;

    Ok(DatabaseHandles {
        queue_pool,
        search_db,
        graph_store: graph_stores.processing,
        graph_sqlite: graph_stores.service,
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
