//! Graph database manager and schema migrations.
//!
//! Manages `graph.db` — a dedicated SQLite database for code relationship
//! storage, separate from `state.db` to avoid lock contention with queue ops.

use std::path::{Path, PathBuf};
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Current schema version for graph.db.
pub const GRAPH_SCHEMA_VERSION: i32 = 5;

/// Default graph database filename.
pub const GRAPH_DB_FILENAME: &str = "graph.db";

/// Errors from graph database operations.
#[derive(Error, Debug)]
pub enum GraphDbError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Schema migration error: {0}")]
    Migration(String),

    #[error(
        "Downgrade not supported: database version {db_version} > code version {code_version}"
    )]
    DowngradeNotSupported { db_version: i32, code_version: i32 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Node not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Lock timeout on \"{op}\" after {waited:?}")]
    LockTimeout { op: String, waited: Duration },

    /// A Rust panic was caught at the synchronous lbug binding layer.
    ///
    /// SEC-03 scope: this guard traps Rust panics originating in the
    /// C++/lbug binding layer only. It CANNOT catch C++ exceptions,
    /// `abort()`, `std::terminate()`, OOM, or OS signals (SIGSEGV/SIGABRT).
    /// True C++-fault isolation requires process isolation (DEF-7).
    /// This is best-effort containment, not a complete fault barrier.
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for graph database operations.
pub type GraphDbResult<T> = Result<T, GraphDbError>;

/// Graph database manager.
///
/// Manages a separate SQLite database (`graph.db`) with independent schema
/// versioning, WAL mode, and foreign keys.
pub struct GraphDbManager {
    pool: SqlitePool,
    path: PathBuf,
}

impl GraphDbManager {
    /// Create a new graph database manager.
    ///
    /// Opens (or creates) the database, enables WAL mode and foreign keys,
    /// then runs any pending schema migrations.
    pub async fn new<P: AsRef<Path>>(database_path: P) -> GraphDbResult<Self> {
        let path = database_path.as_ref().to_path_buf();
        info!("Initializing graph database: {}", path.display());

        let connect_options = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await?;

        // Verify WAL mode
        let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await?;
        if journal_mode.to_lowercase() != "wal" {
            warn!("Expected WAL journal mode, got '{}'", journal_mode);
        } else {
            debug!("WAL mode confirmed for graph.db");
        }

        let manager = Self { pool, path };
        manager.run_migrations().await?;

        Ok(manager)
    }

    /// Create a manager from an existing pool (for tests).
    pub fn with_pool(pool: SqlitePool, path: PathBuf) -> Self {
        Self { pool, path }
    }

    /// Get a reference to the connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Get the database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Close the database connection pool.
    pub async fn close(&self) {
        info!("Closing graph database: {}", self.path.display());
        self.pool.close().await;
    }

    /// Run pending schema migrations.
    async fn run_migrations(&self) -> GraphDbResult<()> {
        // Create schema version table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS graph_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )",
        )
        .execute(&self.pool)
        .await?;

        let current: Option<i32> =
            sqlx::query_scalar("SELECT MAX(version) FROM graph_schema_version")
                .fetch_optional(&self.pool)
                .await?
                .flatten();
        let current = current.unwrap_or(0);

        info!(
            "Graph schema version: {}, target: {}",
            current, GRAPH_SCHEMA_VERSION
        );

        if current > GRAPH_SCHEMA_VERSION {
            return Err(GraphDbError::DowngradeNotSupported {
                db_version: current,
                code_version: GRAPH_SCHEMA_VERSION,
            });
        }

        if current == GRAPH_SCHEMA_VERSION {
            debug!("Graph schema is up to date");
            return Ok(());
        }

        for version in (current + 1)..=GRAPH_SCHEMA_VERSION {
            info!("Running graph migration to version {}", version);
            self.run_migration(version).await?;
            sqlx::query("INSERT INTO graph_schema_version (version) VALUES (?1)")
                .bind(version)
                .execute(&self.pool)
                .await?;
        }

        info!(
            "Graph schema migrations complete. Now at version {}",
            GRAPH_SCHEMA_VERSION
        );
        Ok(())
    }

    async fn run_migration(&self, version: i32) -> GraphDbResult<()> {
        match version {
            1 => self.migrate_v1().await,
            2 => self.migrate_v2().await,
            3 => self.migrate_v3().await,
            4 => self.migrate_v4().await,
            5 => self.migrate_v5().await,
            _ => Err(GraphDbError::Migration(format!(
                "Unknown graph migration version: {}",
                version
            ))),
        }
    }

    async fn migrate_v1(&self) -> GraphDbResult<()> {
        info!("Graph migration v1: creating nodes and edges tables");

        let mut tx = self.pool.begin().await?;

        // Nodes table
        sqlx::query(
            "CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                language TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query("CREATE INDEX idx_nodes_tenant ON graph_nodes(tenant_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_nodes_file ON graph_nodes(tenant_id, file_path)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_nodes_symbol ON graph_nodes(tenant_id, symbol_name)")
            .execute(&mut *tx)
            .await?;

        // Edges table
        sqlx::query(
            "CREATE TABLE graph_edges (
                edge_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id),
                FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id)
            )",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query("CREATE INDEX idx_edges_tenant ON graph_edges(tenant_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_edges_source ON graph_edges(source_node_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_edges_target ON graph_edges(target_node_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_edges_source_file ON graph_edges(tenant_id, source_file)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX idx_edges_type ON graph_edges(edge_type)")
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(())
    }

    /// Migration v2: add `branches` column to `graph_nodes` and `branch`
    /// column to `graph_edges` for branch-scoped graph queries.
    async fn migrate_v2(&self) -> GraphDbResult<()> {
        info!("Graph migration v2: adding branches/branch columns");

        let mut tx = self.pool.begin().await?;

        // Add branches JSON array to nodes (defaults to ["main"])
        sqlx::query(
            r#"ALTER TABLE graph_nodes ADD COLUMN branches TEXT NOT NULL DEFAULT '["main"]'"#,
        )
        .execute(&mut *tx)
        .await?;

        // Add scalar branch to edges (NULL = globally inferred)
        sqlx::query("ALTER TABLE graph_edges ADD COLUMN branch TEXT")
            .execute(&mut *tx)
            .await?;

        // Index for branch-scoped node queries
        sqlx::query("CREATE INDEX idx_nodes_branches ON graph_nodes(tenant_id, branches)")
            .execute(&mut *tx)
            .await?;

        // Index for branch-scoped edge queries
        sqlx::query("CREATE INDEX idx_edges_branch ON graph_edges(tenant_id, branch)")
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(())
    }

    /// Migration v3: composite indexes for cross-boundary and type-filtered queries.
    async fn migrate_v3(&self) -> GraphDbResult<()> {
        info!("Graph migration v3: adding composite indexes for narrative/concept traversal");

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_edges_type_source \
             ON graph_edges(edge_type, source_node_id)",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_edges_type_target \
             ON graph_edges(edge_type, target_node_id)",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_nodes_type_tenant \
             ON graph_nodes(symbol_type, tenant_id)",
        )
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }

    /// Migration v4: node-driven composite indexes for cross-boundary traversal.
    ///
    /// The bidirectional cross-boundary CTE drives from the recursive node set
    /// into `graph_edges`. Without `(source_node_id, edge_type)` /
    /// `(target_node_id, edge_type)` the planner scans all edges of a type per
    /// recursion step, which is O(degree^2) on a high-degree ConceptNode. These
    /// indexes let each step look up a node's edges directly.
    async fn migrate_v4(&self) -> GraphDbResult<()> {
        info!("Graph migration v4: adding node-driven edge indexes for cross-boundary traversal");

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_edges_source_type              ON graph_edges(source_node_id, edge_type)",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_edges_target_type              ON graph_edges(target_node_id, edge_type)",
        )
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }

    /// Migration v5: add Qdrant point-id link columns to `graph_nodes`.
    ///
    /// `qdrant_point_id` stores the Qdrant point UUID for chunk-derived nodes so
    /// that the graph layer can locate the corresponding embedding vector without
    /// re-computing the point ID from scratch. `point_id_state` tracks whether
    /// the link has been established ("linked") or is still absent ("none").
    ///
    /// A partial index on `qdrant_point_id` (WHERE NOT NULL) makes lookups by
    /// point ID fast without bloating the index for the majority of nodes that
    /// have no Qdrant link (concept nodes, stubs, file nodes).
    async fn migrate_v5(&self) -> GraphDbResult<()> {
        info!("Graph migration v5: adding qdrant_point_id link columns");

        let mut tx = self.pool.begin().await?;

        sqlx::query("ALTER TABLE graph_nodes ADD COLUMN qdrant_point_id TEXT")
            .execute(&mut *tx)
            .await?;

        sqlx::query(
            "ALTER TABLE graph_nodes ADD COLUMN point_id_state TEXT NOT NULL DEFAULT 'none'",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_nodes_point_id \
             ON graph_nodes(qdrant_point_id) WHERE qdrant_point_id IS NOT NULL",
        )
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }
}

impl Clone for GraphDbManager {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            path: self.path.clone(),
        }
    }
}
