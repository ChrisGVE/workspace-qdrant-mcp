//! Recover cascade for the auxiliary databases (search.db and graph.db).
//!
//! Located at `src/rust/daemon/core/src/recover/search_graph.rs`, sibling of
//! `mod.rs`. state.db is the primary plane (handled in `mod.rs`); these two
//! satellite databases hold their own tenant_id-keyed rows in separate
//! connection pools and so are reconciled separately.
//!
//! - **search.db** holds `file_metadata` (one row per indexed file). Its
//!   `tenant_id` is renamed on a tenancy flip, and its **absolute**
//!   `file_path` is prefix-rewritten on a path move. `relative_path` is
//!   watch-root-relative and never changes.
//! - **graph.db** holds `graph_nodes` / `graph_edges`. Both carry `tenant_id`
//!   (renamed on a flip). Their `file_path` / `source_file` are stored
//!   *relative* to the project root, so a path move needs no rewrite there.
//!
//! Each function takes the satellite pool directly so the module stays pure
//! and unit-testable against an in-memory database; the caller (the gRPC
//! recover handler) owns pool selection.

use sqlx::SqlitePool;

use crate::schema_version::SchemaError;

use super::escape_like;

/// Rename `old_tenant_id` -> `new_tenant_id` in search.db `file_metadata`.
/// Returns rows updated. A missing table (search.db not provisioned) is treated
/// as zero rows rather than an error, mirroring the optional-table handling the
/// existing tenant rename uses for `tracked_files`.
pub async fn rename_tenant_search_db(
    pool: &SqlitePool,
    old_tenant_id: &str,
    new_tenant_id: &str,
) -> Result<i64, SchemaError> {
    if !table_exists(pool, "file_metadata").await? {
        return Ok(0);
    }
    let r = sqlx::query("UPDATE file_metadata SET tenant_id = ?1 WHERE tenant_id = ?2")
        .bind(new_tenant_id)
        .bind(old_tenant_id)
        .execute(pool)
        .await?;
    Ok(r.rows_affected() as i64)
}

/// Prefix-rewrite the absolute `file_metadata.file_path` for a tenant on a path
/// move (old_prefix -> new_prefix). `relative_path` is left untouched.
pub async fn rewrite_paths_search_db(
    pool: &SqlitePool,
    tenant_id: &str,
    old_prefix: &str,
    new_prefix: &str,
) -> Result<i64, SchemaError> {
    if !table_exists(pool, "file_metadata").await? {
        return Ok(0);
    }
    let like = format!("{}%", escape_like(old_prefix));
    let r = sqlx::query(
        "UPDATE file_metadata \
         SET file_path = ?1 || substr(file_path, ?2) \
         WHERE tenant_id = ?3 AND file_path LIKE ?4 ESCAPE '\\'",
    )
    .bind(new_prefix)
    .bind((old_prefix.len() + 1) as i64)
    .bind(tenant_id)
    .bind(&like)
    .execute(pool)
    .await?;
    Ok(r.rows_affected() as i64)
}

/// Rename `old_tenant_id` -> `new_tenant_id` across graph.db
/// (`graph_nodes` + `graph_edges`). Returns total rows updated. Both tables'
/// path columns are project-relative, so a path move needs no rewrite here.
pub async fn rename_tenant_graph_db(
    pool: &SqlitePool,
    old_tenant_id: &str,
    new_tenant_id: &str,
) -> Result<i64, SchemaError> {
    let mut total = 0i64;
    for table in ["graph_nodes", "graph_edges"] {
        if !table_exists(pool, table).await? {
            continue;
        }
        let sql = format!("UPDATE {table} SET tenant_id = ?1 WHERE tenant_id = ?2");
        let r = sqlx::query(&sql)
            .bind(new_tenant_id)
            .bind(old_tenant_id)
            .execute(pool)
            .await?;
        total += r.rows_affected() as i64;
    }
    Ok(total)
}

/// Whether `name` is a table in the connected database.
async fn table_exists(pool: &SqlitePool, name: &str) -> Result<bool, SchemaError> {
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
    )
    .bind(name)
    .fetch_one(pool)
    .await?;
    Ok(exists)
}
