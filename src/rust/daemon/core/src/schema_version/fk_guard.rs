//! Dedicated-connection FK guard for migrations that need `PRAGMA foreign_keys = OFF`.
//!
//! Opens a fresh `SqliteConnection` (never pooled) so that if any step fails,
//! the connection is simply dropped — no pool connection is left with FK checks
//! disabled.

use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{ConnectOptions, Executor, SqliteConnection};
use tracing::debug;

use super::SchemaError;

/// A dedicated (non-pooled) SQLite connection with FK checks disabled.
///
/// Call [`open`] to create, use [`conn`] for DDL, then [`restore`] on
/// success. If dropped without `restore`, the connection closes without
/// re-enabling FK — safe because it was never part of a pool.
pub struct DedicatedFkConn {
    conn: SqliteConnection,
}

impl DedicatedFkConn {
    pub async fn open(opts: SqliteConnectOptions) -> Result<Self, SchemaError> {
        let mut conn = opts.connect().await?;
        conn.execute("PRAGMA foreign_keys = OFF").await?;
        debug!("fk_guard: disabled FK checks on dedicated connection");
        Ok(Self { conn })
    }

    pub fn conn(&mut self) -> &mut SqliteConnection {
        &mut self.conn
    }

    pub async fn restore(mut self) -> Result<(), SchemaError> {
        self.conn.execute("PRAGMA foreign_keys = ON").await?;
        debug!("fk_guard: re-enabled FK checks on dedicated connection");
        Ok(())
    }
}
