//! Schema Version Table and Migration System
//!
//! This module implements the schema version tracking and migration system.
//! Per docs/specs/12-configuration.md, the daemon is the sole owner of
//! the SQLite database schema. It creates all tables on startup and handles
//! all schema version upgrades.
//!
//! Other components (MCP Server, CLI) must NOT create tables or run migrations.

pub mod migration;

pub mod fk_guard;
mod v01;
mod v02;
mod v03;
mod v04;
mod v05;
mod v06;
mod v07;
mod v08;
mod v09;
mod v10;
mod v11;
mod v12;
mod v13;
mod v14;
mod v15;
mod v16;
mod v17;
mod v18;
mod v19;
mod v20;
mod v21;
mod v22;
mod v23;
mod v24;
mod v25;
mod v26;
mod v27;
mod v28;
mod v29;
mod v30;
mod v31;
mod v32;
mod v33;
mod v34;
mod v35;
mod v36;
pub mod v37;
mod v38;
pub mod v39;
pub mod v40;
mod v41;
mod v42;
mod v43;
mod v44;
mod v45;
mod v46;
mod v47;
mod v48;

use sqlx::{pool::PoolConnection, sqlite::SqliteRow, Executor, Row, Sqlite, SqlitePool};
use thiserror::Error;
use tracing::{debug, info, warn};

pub use self::migration::{Migration, MigrationRegistry};

// ============================================================================
// ForeignKeysGuard — RAII wrapper that disables FK checks for a connection
// and restores the original state on drop (even on panic).
//
// Convention for migrations v35+: each migration owns its own transaction
// (BEGIN IMMEDIATE / COMMIT / ROLLBACK). The ForeignKeysGuard must be
// acquired before opening the transaction so the PRAGMA change takes effect
// outside the transaction scope as required by SQLite.
// ============================================================================

/// RAII guard that sets `PRAGMA foreign_keys = OFF` on a dedicated SQLite
/// connection and restores the original value when dropped — even on panic.
///
/// The guard takes ownership of the connection for the duration of the DDL
/// window. Use `conn_mut()` to access the connection while the guard is held,
/// and `into_conn()` to reclaim it when done (which also restores the pragma
/// asynchronously via an explicit async call, bypassing Drop).
///
/// Drop provides the panic-safety guarantee: it runs a synchronous restore
/// via `tokio::task::block_in_place` so the pragma is always restored even
/// when the async success path is not reached.
pub struct ForeignKeysGuard {
    /// The connection on which FK checks were disabled. Wrapped in Option so
    /// Drop can take ownership when it needs to run the restore.
    conn: Option<PoolConnection<Sqlite>>,
    /// The FK pragma value captured before disabling (0 or 1).
    original_value: i32,
}

impl ForeignKeysGuard {
    /// Capture the current `PRAGMA foreign_keys` value on `conn`, set it to
    /// `OFF`, take ownership of the connection, and return a guard that
    /// restores the original value when dropped.
    pub async fn disable(
        mut conn: PoolConnection<Sqlite>,
    ) -> Result<ForeignKeysGuard, SchemaError> {
        let original_value: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&mut *conn)
            .await?;

        conn.execute("PRAGMA foreign_keys = OFF").await?;
        debug!(
            "ForeignKeysGuard: disabled FK checks (was {})",
            original_value
        );

        Ok(ForeignKeysGuard {
            conn: Some(conn),
            original_value,
        })
    }

    /// Borrow the underlying connection mutably for DDL operations.
    pub fn conn_mut(&mut self) -> &mut PoolConnection<Sqlite> {
        self.conn
            .as_mut()
            .expect("ForeignKeysGuard: connection already consumed")
    }

    /// Restore FK checks asynchronously and return the owned connection.
    /// Prefer this over relying on Drop for the success path; Drop is the
    /// panic-safety fallback only.
    pub async fn restore(mut self) -> Result<PoolConnection<Sqlite>, SchemaError> {
        let mut conn = self
            .conn
            .take()
            .expect("ForeignKeysGuard: connection already consumed");

        let pragma = if self.original_value != 0 {
            "PRAGMA foreign_keys = ON"
        } else {
            "PRAGMA foreign_keys = OFF"
        };
        conn.execute(pragma).await?;
        debug!(
            "ForeignKeysGuard: restored FK checks to {}",
            self.original_value
        );

        // self.conn is now None — Drop will skip the restore.
        Ok(conn)
    }
}

impl Drop for ForeignKeysGuard {
    /// Panic-safety fallback: if the guard is dropped without `restore()` being
    /// called (e.g. on panic or early return on the error path), restore the
    /// FK pragma synchronously so it is never left disabled.
    fn drop(&mut self) {
        if let Some(mut conn) = self.conn.take() {
            let pragma = if self.original_value != 0 {
                "PRAGMA foreign_keys = ON"
            } else {
                "PRAGMA foreign_keys = OFF"
            };
            warn!(
                "ForeignKeysGuard::drop: restoring FK checks to {} via block_in_place \
                 (guard was not explicitly restored — check for panics or early returns)",
                self.original_value
            );
            // block_in_place yields the current worker thread to run a blocking
            // operation inside an async Tokio context.
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    if let Err(e) = conn.execute(pragma).await {
                        warn!("ForeignKeysGuard::drop: failed to restore FK pragma: {}", e);
                    }
                });
            });
        }
    }
}

/// Current schema version - increment when adding new migrations
pub const CURRENT_SCHEMA_VERSION: i32 = 48;

/// Errors that can occur during schema operations
#[derive(Error, Debug)]
pub enum SchemaError {
    #[error("SQLite error: {0}")]
    SqlxError(#[from] sqlx::Error),

    #[error("Migration error: {0}")]
    MigrationError(String),

    #[error("Schema version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: i32, found: i32 },

    #[error(
        "Downgrade not supported: database version {db_version} > code version {code_version}"
    )]
    DowngradeNotSupported { db_version: i32, code_version: i32 },
}

/// SQL to create the schema_version table
pub const CREATE_SCHEMA_VERSION_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// Schema version entry
#[derive(Debug, Clone)]
pub struct SchemaVersionEntry {
    pub version: i32,
    pub applied_at: String,
}

/// Schema manager handles version tracking and migrations
pub struct SchemaManager {
    pool: SqlitePool,
}

impl SchemaManager {
    /// Create a new schema manager
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Initialize the schema version table
    pub async fn initialize(&self) -> Result<(), SchemaError> {
        debug!("Initializing schema version table");
        sqlx::query(CREATE_SCHEMA_VERSION_SQL)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Get the current schema version from the database.
    /// Returns None if no version is set (fresh database).
    pub async fn get_current_version(&self) -> Result<Option<i32>, SchemaError> {
        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')"
        )
        .fetch_one(&self.pool)
        .await?;

        if !table_exists {
            return Ok(None);
        }

        let result: Option<i32> = sqlx::query_scalar("SELECT MAX(version) FROM schema_version")
            .fetch_optional(&self.pool)
            .await?
            .flatten();

        Ok(result)
    }

    /// Record a migration as applied
    pub async fn record_migration(&self, version: i32) -> Result<(), SchemaError> {
        sqlx::query("INSERT INTO schema_version (version) VALUES (?1)")
            .bind(version)
            .execute(&self.pool)
            .await?;
        info!("Recorded migration to schema version {}", version);
        Ok(())
    }

    /// Get all applied migrations
    pub async fn get_migration_history(&self) -> Result<Vec<SchemaVersionEntry>, SchemaError> {
        let rows: Vec<SqliteRow> =
            sqlx::query("SELECT version, applied_at FROM schema_version ORDER BY version ASC")
                .fetch_all(&self.pool)
                .await?;

        let mut result = Vec::new();
        for row in rows {
            result.push(SchemaVersionEntry {
                version: row.get("version"),
                applied_at: row.get("applied_at"),
            });
        }

        Ok(result)
    }

    /// Check if migrations are needed and run them
    pub async fn run_migrations(&self) -> Result<(), SchemaError> {
        self.initialize().await?;

        let current = self.get_current_version().await?.unwrap_or(0);
        info!(
            "Current schema version: {}, target: {}",
            current, CURRENT_SCHEMA_VERSION
        );

        if current > CURRENT_SCHEMA_VERSION {
            return Err(SchemaError::DowngradeNotSupported {
                db_version: current,
                code_version: CURRENT_SCHEMA_VERSION,
            });
        }

        if current == CURRENT_SCHEMA_VERSION {
            debug!("Schema is up to date");
            return Ok(());
        }

        let registry = Self::build_registry();

        for version in (current + 1)..=CURRENT_SCHEMA_VERSION {
            info!("Running migration to version {}", version);
            self.run_migration_from_registry(&registry, version).await?;
            self.record_migration(version).await?;
        }

        info!(
            "Schema migrations complete. Now at version {}",
            CURRENT_SCHEMA_VERSION
        );
        Ok(())
    }

    /// Run the migration chain only up to and including `target` (test helper).
    ///
    /// Mirrors [`run_migrations`](Self::run_migrations) but stops at `target`
    /// instead of [`CURRENT_SCHEMA_VERSION`]. Used by version-specific migration
    /// tests that need to assert the schema state *at* a historic version (e.g.
    /// the v40 `tracked_files` shape) without the later migrations (v48) having
    /// already rewritten it.
    pub async fn run_migrations_through(&self, target: i32) -> Result<(), SchemaError> {
        self.initialize().await?;
        let current = self.get_current_version().await?.unwrap_or(0);
        if current >= target {
            return Ok(());
        }
        let registry = Self::build_registry();
        for version in (current + 1)..=target {
            self.run_migration_from_registry(&registry, version).await?;
            self.record_migration(version).await?;
        }
        Ok(())
    }

    /// Run a specific migration by version number (used by tests).
    pub async fn run_migration(&self, version: i32) -> Result<(), SchemaError> {
        let registry = Self::build_registry();
        self.run_migration_from_registry(&registry, version).await
    }

    /// Build the migration registry with all migrations.
    fn build_registry() -> MigrationRegistry {
        let mut registry = MigrationRegistry::new();
        registry.register(Box::new(v01::V01Migration));
        registry.register(Box::new(v02::V02Migration));
        registry.register(Box::new(v03::V03Migration));
        registry.register(Box::new(v04::V04Migration));
        registry.register(Box::new(v05::V05Migration));
        registry.register(Box::new(v06::V06Migration));
        registry.register(Box::new(v07::V07Migration));
        registry.register(Box::new(v08::V08Migration));
        registry.register(Box::new(v09::V09Migration));
        registry.register(Box::new(v10::V10Migration));
        registry.register(Box::new(v11::V11Migration));
        registry.register(Box::new(v12::V12Migration));
        registry.register(Box::new(v13::V13Migration));
        registry.register(Box::new(v14::V14Migration));
        registry.register(Box::new(v15::V15Migration));
        registry.register(Box::new(v16::V16Migration));
        registry.register(Box::new(v17::V17Migration));
        registry.register(Box::new(v18::V18Migration));
        registry.register(Box::new(v19::V19Migration));
        registry.register(Box::new(v20::V20Migration));
        registry.register(Box::new(v21::V21Migration));
        registry.register(Box::new(v22::V22Migration));
        registry.register(Box::new(v23::V23Migration));
        registry.register(Box::new(v24::V24Migration));
        registry.register(Box::new(v25::V25Migration));
        registry.register(Box::new(v26::V26Migration));
        registry.register(Box::new(v27::V27Migration));
        registry.register(Box::new(v28::V28Migration));
        registry.register(Box::new(v29::V29Migration));
        registry.register(Box::new(v30::V30Migration));
        registry.register(Box::new(v31::V31Migration));
        registry.register(Box::new(v32::V32Migration));
        registry.register(Box::new(v33::V33Migration));
        registry.register(Box::new(v34::V34Migration));
        registry.register(Box::new(v35::V35Migration));
        registry.register(Box::new(v36::V36Migration));
        registry.register(Box::new(v37::V37Migration));
        registry.register(Box::new(v38::V38Migration));
        registry.register(Box::new(v39::V39Migration));
        registry.register(Box::new(v40::V40Migration));
        registry.register(Box::new(v41::V41Migration));
        registry.register(Box::new(v42::V42Migration));
        registry.register(Box::new(v43::V43Migration));
        registry.register(Box::new(v44::V44Migration));
        registry.register(Box::new(v45::V45Migration));
        registry.register(Box::new(v46::V46Migration));
        registry.register(Box::new(v47::V47Migration));
        registry.register(Box::new(v48::V48Migration));
        registry
    }

    /// Run a single migration from the registry.
    ///
    /// Transaction ownership convention (v35+): each migration that performs
    /// table reconstruction is responsible for wrapping its own work in
    /// `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK`. This function does NOT add
    /// a wrapping transaction — doing so would prevent migrations from using
    /// `PRAGMA foreign_keys = OFF`, which SQLite requires to be set outside
    /// an active transaction. Migrations v1–v34 predate this convention and
    /// rely on SQLite's implicit per-statement transactions; they must not be
    /// retrofitted (no migration effort policy).
    async fn run_migration_from_registry(
        &self,
        registry: &MigrationRegistry,
        version: i32,
    ) -> Result<(), SchemaError> {
        match registry.get(version) {
            Some(migration) => migration.up(&self.pool).await,
            None => Err(SchemaError::MigrationError(format!(
                "Unknown migration version: {}",
                version
            ))),
        }
    }
}

/// Check if schema is initialized (for graceful degradation by MCP/CLI)
pub async fn is_schema_initialized(pool: &SqlitePool) -> bool {
    let result: Result<bool, _> = sqlx::query_scalar(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'",
    )
    .fetch_optional(pool)
    .await
    .map(|opt: Option<i32>| opt.is_some());

    result.unwrap_or(false)
}

/// Get schema version (for health checks)
pub async fn get_schema_version(pool: &SqlitePool) -> Option<i32> {
    let manager = SchemaManager::new(pool.clone());
    manager.get_current_version().await.ok().flatten()
}

#[cfg(test)]
mod tests;
