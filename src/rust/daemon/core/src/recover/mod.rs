//! Manual reconcile / recover cascade (issue #140).
//!
//! Located at `src/rust/daemon/core/src/recover/mod.rs`. This module owns the
//! *complete* tenant-keyed cascade that the recover command (CLI `wqm project
//! recover` / `wqm library recover`, and the MCP admin surface) relies on. It
//! is the deliberate, manual counterpart to the automatic registration-time
//! reconciliation in `grpc::services::project_service::reconcile` (#138/#139),
//! which covers only a narrow subset of tables.
//!
//! # Two independent kinds of drift
//!
//! A project's identity has two parts that can drift apart from what is stored:
//!
//! - **Path move** — the directory moved on disk but its identity (git remote)
//!   is unchanged. Per-file rows in SQLite are stored *relative* to the watch
//!   root (`tracked_files.relative_path`, the graph store's relative
//!   `file_path`), so the only SQLite path that moves is the watch root itself
//!   (`watch_folders.path`). The transient `unified_queue.file_path` is the one
//!   per-row *absolute* path and is rewritten by prefix. In Qdrant the
//!   per-chunk `file_path` / `absolute_path` payload keys are absolute and are
//!   rewritten old->new prefix; `relative_path` is left untouched.
//! - **Tenancy flip** — the directory's `tenant_id` changed (gained or lost a
//!   git remote, or a clone-disambiguation suffix changed). Every tenant_id
//!   keyed row across SQLite and every tenant-keyed Qdrant collection must be
//!   re-keyed old->new.
//!
//! A single recover call may do both at once (a project that moved *and*
//! flipped tenancy).
//!
//! # Cascade coverage (closes the #140 boundary)
//!
//! [`TENANT_KEYED_TABLES`] is the authoritative list of state.db tables that
//! carry a `tenant_id` column, enumerated from the schema. `tracked_files` is
//! deliberately absent: post-v37 it has no `tenant_id` column and is keyed by
//! `watch_folder_id` (a stable surrogate), so a tenancy flip needs no change
//! there. The search.db (`file_metadata`) and graph.db (`graph_nodes`,
//! `graph_edges`) live in their own connection pools and are handled by
//! [`rename_tenant_search_db`] / [`rewrite_paths_search_db`] and
//! [`rename_tenant_graph_db`]; their relative path columns never need rewriting.
//!
//! Qdrant collections are tenant-keyed by the `tenant_id` payload field; the
//! caller enqueues the existing cascade-rename for every collection in
//! [`crate::recover::QDRANT_TENANT_COLLECTIONS`] and, for a path move, rewrites
//! the absolute path payload keys via the storage client.

use sqlx::{Sqlite, SqlitePool, Transaction};

use crate::schema_version::SchemaError;

mod search_graph;
pub use search_graph::{rename_tenant_graph_db, rename_tenant_search_db, rewrite_paths_search_db};

#[cfg(test)]
mod tests;

/// Every state.db table that carries a `tenant_id` column, with the column name.
///
/// This is the authoritative cascade list for a tenancy flip in the main
/// database. Enumerated from the schema (see module docs for why
/// `tracked_files` is excluded). Adding a tenant-keyed table to the schema
/// means adding it here too — the test `tenant_keyed_tables_all_exist` guards
/// against the list referencing a table that is not in a freshly migrated DB.
pub const TENANT_KEYED_TABLES: &[&str] = &[
    "watch_folders",
    "unified_queue",
    "keywords",
    "tags",
    "keyword_baskets",
    "canonical_tags",
    "tag_hierarchy_edges",
    "rules_mirror",
    "symbol_cooccurrence",
    "project_groups",
    "project_embeddings",
    "processing_timings",
];

/// Qdrant collections keyed by the `tenant_id` payload field.
///
/// On a tenancy flip the caller enqueues a cascade-rename for each of these so
/// the `tenant_id` payload is rewritten old->new on every point. The previous
/// registration-time reconcile covered only `projects` and `rules`; recover
/// covers the full set, including `scratchpad` and per-tenant media.
pub const QDRANT_TENANT_COLLECTIONS: &[&str] = &["projects", "rules", "scratchpad", "images"];

/// Outcome of a SQLite-plane tenancy rename: how many rows changed, per table.
#[derive(Debug, Default, Clone)]
pub struct CascadeCounts {
    /// Total rows updated across all tables.
    pub total_rows: i64,
    /// Per-table `(table_name, rows_updated)` for reporting / dry-run.
    pub per_table: Vec<(&'static str, i64)>,
}

impl CascadeCounts {
    fn record(&mut self, table: &'static str, rows: i64) {
        self.total_rows += rows;
        self.per_table.push((table, rows));
    }
}

/// Count the tenant_id-keyed rows that a flip would rename, per table, WITHOUT
/// writing. Used by `--dry-run`.
pub async fn count_tenant_rows(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<CascadeCounts, SchemaError> {
    let mut counts = CascadeCounts::default();
    for &table in TENANT_KEYED_TABLES {
        let present: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        )
        .bind(table)
        .fetch_one(pool)
        .await?;
        if !present {
            counts.record(table, 0);
            continue;
        }
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE tenant_id = ?1");
        let rows: i64 = sqlx::query_scalar(&sql)
            .bind(tenant_id)
            .fetch_one(pool)
            .await?;
        counts.record(table, rows);
    }
    Ok(counts)
}

/// Rename `old_tenant_id` -> `new_tenant_id` across every tenant-keyed state.db
/// table, in one transaction. Returns per-table counts.
///
/// `tracked_files` is intentionally not touched (no tenant_id column post-v37;
/// it is keyed by the stable `watch_folder_id`).
pub async fn rename_tenant_state_db(
    pool: &SqlitePool,
    old_tenant_id: &str,
    new_tenant_id: &str,
) -> Result<CascadeCounts, SchemaError> {
    let mut tx = pool.begin().await?;
    let counts = rename_tenant_in_tx(&mut tx, old_tenant_id, new_tenant_id).await?;
    tx.commit().await?;
    Ok(counts)
}

/// Tenancy-rename body shared by the standalone call and a larger transaction.
async fn rename_tenant_in_tx(
    tx: &mut Transaction<'_, Sqlite>,
    old_tenant_id: &str,
    new_tenant_id: &str,
) -> Result<CascadeCounts, SchemaError> {
    let mut counts = CascadeCounts::default();
    for &table in TENANT_KEYED_TABLES {
        // A table may be absent in a partially-provisioned database (e.g. a
        // grouping table created lazily on first use). Skip it rather than
        // failing the whole rename — it has no rows for this tenant anyway.
        if !table_exists_tx(tx, table).await? {
            counts.record(table, 0);
            continue;
        }
        let sql = format!("UPDATE {table} SET tenant_id = ?1 WHERE tenant_id = ?2");
        let result = sqlx::query(&sql)
            .bind(new_tenant_id)
            .bind(old_tenant_id)
            .execute(&mut **tx)
            .await?;
        counts.record(table, result.rows_affected() as i64);
    }
    Ok(counts)
}

/// Whether `name` is a table in the transaction's database.
async fn table_exists_tx(
    tx: &mut Transaction<'_, Sqlite>,
    name: &str,
) -> Result<bool, SchemaError> {
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
    )
    .bind(name)
    .fetch_one(&mut **tx)
    .await?;
    Ok(exists)
}

/// Rewrite the stored watch-root path of a project/library, plus the one
/// per-row absolute path column in state.db (`unified_queue.file_path`).
///
/// `tracked_files` stores per-file paths relative to the watch root and so is
/// unaffected by a root move. Returns the total number of rows updated.
///
/// `old_prefix` / `new_prefix` are the old and new absolute watch-root paths.
/// `unified_queue.file_path` values are rewritten only where they begin with
/// `old_prefix`; SQLite's `substr`/`replace` would over-match a bare `replace`,
/// so the rewrite is anchored with a `LIKE 'old_prefix%'` guard and a
/// length-based `substr` splice to replace only the leading prefix.
pub async fn repoint_path_state_db(
    pool: &SqlitePool,
    tenant_id: &str,
    old_prefix: &str,
    new_prefix: &str,
) -> Result<i64, SchemaError> {
    let mut tx = pool.begin().await?;
    let rows = repoint_path_in_tx(&mut tx, tenant_id, old_prefix, new_prefix).await?;
    tx.commit().await?;
    Ok(rows)
}

async fn repoint_path_in_tx(
    tx: &mut Transaction<'_, Sqlite>,
    tenant_id: &str,
    old_prefix: &str,
    new_prefix: &str,
) -> Result<i64, SchemaError> {
    let mut total = 0i64;

    // watch_folders.path is the watch root itself: an exact swap old -> new.
    let r = sqlx::query("UPDATE watch_folders SET path = ?1 WHERE tenant_id = ?2 AND path = ?3")
        .bind(new_prefix)
        .bind(tenant_id)
        .bind(old_prefix)
        .execute(&mut **tx)
        .await?;
    total += r.rows_affected() as i64;

    // unified_queue.file_path is per-row and absolute: splice the leading
    // prefix only. `?1 || substr(file_path, len(old_prefix)+1)` keeps the
    // remainder of each path intact.
    let like = format!("{}%", escape_like(old_prefix));
    let r = sqlx::query(
        "UPDATE unified_queue \
         SET file_path = ?1 || substr(file_path, ?2) \
         WHERE tenant_id = ?3 \
           AND file_path IS NOT NULL \
           AND file_path LIKE ?4 ESCAPE '\\'",
    )
    .bind(new_prefix)
    .bind((old_prefix.len() + 1) as i64)
    .bind(tenant_id)
    .bind(&like)
    .execute(&mut **tx)
    .await?;
    total += r.rows_affected() as i64;

    Ok(total)
}

/// Count rows a path re-point would change in state.db, WITHOUT writing.
pub async fn count_repoint_rows(
    pool: &SqlitePool,
    tenant_id: &str,
    old_prefix: &str,
) -> Result<i64, SchemaError> {
    let mut total = 0i64;

    let wf: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM watch_folders WHERE tenant_id = ?1 AND path = ?2")
            .bind(tenant_id)
            .bind(old_prefix)
            .fetch_one(pool)
            .await?;
    total += wf;

    let like = format!("{}%", escape_like(old_prefix));
    let uq: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue \
         WHERE tenant_id = ?1 AND file_path IS NOT NULL AND file_path LIKE ?2 ESCAPE '\\'",
    )
    .bind(tenant_id)
    .bind(&like)
    .fetch_one(pool)
    .await?;
    total += uq;

    Ok(total)
}

/// Escape SQLite `LIKE` wildcards (`%`, `_`) and the escape char itself so a
/// path prefix is matched literally. Pairs with `ESCAPE '\'` in the query.
pub(super) fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}
