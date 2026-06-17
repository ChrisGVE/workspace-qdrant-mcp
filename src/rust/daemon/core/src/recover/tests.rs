//! Tests for the recover cascade (#140).
//!
//! Located at `src/rust/daemon/core/src/recover/tests.rs`. Each test builds a
//! migrated in-memory state.db (or a hand-built search.db / graph.db), seeds
//! tenant-keyed rows and paths, runs a cascade function, and asserts the
//! before/after counts. No filesystem is touched.

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;

use super::*;
use crate::schema_version::SchemaManager;

/// A fully migrated in-memory state.db.
async fn state_db() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    SchemaManager::new(pool.clone())
        .run_migrations()
        .await
        .unwrap();
    pool
}

/// Insert a project watch_folders row plus one tenant-keyed row in a few of the
/// satellite tables, so a rename has something to move in each.
async fn seed_tenant(pool: &SqlitePool, tenant: &str, path: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, \
         is_active, follow_symlinks, cleanup_on_disable, created_at, updated_at) \
         VALUES (?1, ?2, 'projects', ?3, 1, 1, 0, 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
    )
    .bind(format!("wf-{tenant}"))
    .bind(path)
    .bind(tenant)
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO keywords (keyword, tenant_id, collection, doc_id, score, created_at) \
         VALUES ('kw', ?1, 'projects', 'doc1', 1.0, '2026-01-01T00:00:00Z')",
    )
    .bind(tenant)
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO symbol_cooccurrence \
         (symbol_a, symbol_b, tenant_id, collection, cooccurrence_count, updated_at) \
         VALUES ('a', 'b', ?1, 'projects', 3, '2026-01-01T00:00:00Z')",
    )
    .bind(tenant)
    .execute(pool)
    .await
    .unwrap();
}

#[tokio::test]
async fn tenant_keyed_tables_all_exist() {
    // Guards the cascade list against referencing a table absent from a freshly
    // migrated database (which would make a rename silently error).
    let pool = state_db().await;
    for &table in TENANT_KEYED_TABLES {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        )
        .bind(table)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            exists,
            "cascade table {table} must exist in a migrated state.db"
        );

        // And each must actually have a tenant_id column.
        let has_col: bool = sqlx::query_scalar(&format!(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('{table}') WHERE name='tenant_id')"
        ))
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            has_col,
            "cascade table {table} must have a tenant_id column"
        );
    }
}

#[tokio::test]
async fn rename_moves_every_seeded_table() {
    let pool = state_db().await;
    seed_tenant(&pool, "old_tenant", "/old/path").await;

    let counts = rename_tenant_state_db(&pool, "old_tenant", "new_tenant")
        .await
        .unwrap();

    // watch_folders + keywords + symbol_cooccurrence each had 1 row.
    assert_eq!(counts.total_rows, 3, "per_table: {:?}", counts.per_table);

    let old_left: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM watch_folders WHERE tenant_id = 'old_tenant'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(old_left, 0, "no rows may remain under the old tenant_id");

    let kw: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM keywords WHERE tenant_id = 'new_tenant'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(kw, 1);
}

#[tokio::test]
async fn rename_is_idempotent_when_old_id_absent() {
    let pool = state_db().await;
    seed_tenant(&pool, "current", "/p").await;

    // Renaming a tenant that does not exist changes nothing.
    let counts = rename_tenant_state_db(&pool, "ghost", "whatever")
        .await
        .unwrap();
    assert_eq!(counts.total_rows, 0);
}

#[tokio::test]
async fn count_tenant_rows_matches_actual_rename() {
    let pool = state_db().await;
    seed_tenant(&pool, "t", "/p").await;

    let planned = count_tenant_rows(&pool, "t").await.unwrap();
    let applied = rename_tenant_state_db(&pool, "t", "t2").await.unwrap();

    assert_eq!(planned.total_rows, applied.total_rows);
    assert_eq!(planned.total_rows, 3);
}

#[tokio::test]
async fn repoint_swaps_watch_root_and_splices_queue_paths() {
    let pool = state_db().await;
    seed_tenant(&pool, "t", "/old/proj").await;

    // Two queue rows under the moving prefix and one unrelated row that must
    // not be touched.
    for fp in ["/old/proj/src/a.rs", "/old/proj/src/b.rs"] {
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, tenant_id, branch, collection, item_type, op, file_path, \
              idempotency_key, retry_count, created_at, updated_at) \
             VALUES (?1, 't', 'main', 'projects', 'file', 'add', ?2, ?1, 0, \
                     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        )
        .bind(format!("q-{fp}"))
        .bind(fp)
        .execute(&pool)
        .await
        .unwrap();
    }
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, tenant_id, branch, collection, item_type, op, file_path, \
          idempotency_key, retry_count, created_at, updated_at) \
         VALUES ('q-other', 't', 'main', 'projects', 'file', 'add', \
                 '/elsewhere/x.rs', 'q-other', 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let planned = count_repoint_rows(&pool, "t", "/old/proj").await.unwrap();
    let rows = repoint_path_state_db(&pool, "t", "/old/proj", "/new/home")
        .await
        .unwrap();

    // watch_folders (1) + the two prefixed queue rows (2); the "/elsewhere" row
    // does not match the prefix and must not be counted or rewritten.
    assert_eq!(planned, rows);
    assert_eq!(rows, 3);

    let other: String =
        sqlx::query_scalar("SELECT file_path FROM unified_queue WHERE queue_id = 'q-other'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(other, "/elsewhere/x.rs", "non-matching path is untouched");

    let new_root: String =
        sqlx::query_scalar("SELECT path FROM watch_folders WHERE tenant_id = 't'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(new_root, "/new/home");

    let a: String = sqlx::query_scalar(
        "SELECT file_path FROM unified_queue WHERE queue_id = 'q-/old/proj/src/a.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(a, "/new/home/src/a.rs");
}

#[tokio::test]
async fn escape_like_neutralizes_wildcards() {
    assert_eq!(escape_like("/a_b/100%"), "/a\\_b/100\\%");
    assert_eq!(escape_like("/plain/path"), "/plain/path");
}

// --- search.db / graph.db satellites ---------------------------------------

async fn search_db() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::query(
        "CREATE TABLE file_metadata (file_id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, \
         branch TEXT, file_path TEXT NOT NULL, relative_path TEXT)",
    )
    .execute(&pool)
    .await
    .unwrap();
    pool
}

#[tokio::test]
async fn search_db_rename_and_repoint() {
    let pool = search_db().await;
    sqlx::query(
        "INSERT INTO file_metadata (file_id, tenant_id, file_path, relative_path) \
         VALUES (1, 'old', '/old/proj/src/a.rs', 'src/a.rs')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let renamed = rename_tenant_search_db(&pool, "old", "new").await.unwrap();
    assert_eq!(renamed, 1);

    let repointed = rewrite_paths_search_db(&pool, "new", "/old/proj", "/new/home")
        .await
        .unwrap();
    assert_eq!(repointed, 1);

    let (fp, rel): (String, String) =
        sqlx::query_as("SELECT file_path, relative_path FROM file_metadata WHERE file_id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(fp, "/new/home/src/a.rs", "absolute path is rewritten");
    assert_eq!(rel, "src/a.rs", "relative path is left untouched");
}

#[tokio::test]
async fn search_db_missing_table_is_zero_not_error() {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let n = rename_tenant_search_db(&pool, "a", "b").await.unwrap();
    assert_eq!(n, 0);
    let m = rewrite_paths_search_db(&pool, "a", "/x", "/y")
        .await
        .unwrap();
    assert_eq!(m, 0);
}

async fn graph_db() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::query(
        "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, \
         file_path TEXT NOT NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "CREATE TABLE graph_edges (edge_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, \
         source_file TEXT NOT NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    pool
}

#[tokio::test]
async fn graph_db_rename_covers_nodes_and_edges() {
    let pool = graph_db().await;
    sqlx::query("INSERT INTO graph_nodes VALUES ('n1', 'old', 'src/main.rs')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO graph_edges VALUES ('e1', 'old', 'src/main.rs')")
        .execute(&pool)
        .await
        .unwrap();

    let n = rename_tenant_graph_db(&pool, "old", "new").await.unwrap();
    assert_eq!(n, 2, "both a node and an edge are re-keyed");

    let node_path: String =
        sqlx::query_scalar("SELECT file_path FROM graph_nodes WHERE node_id = 'n1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(node_path, "src/main.rs", "relative graph path is untouched");
}
