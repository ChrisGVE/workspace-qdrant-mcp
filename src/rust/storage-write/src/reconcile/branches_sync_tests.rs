//! Tests for AC-F15.2: re-sync store.db.branches from state.db.project_locations.
//!
//! Each test builds an in-memory (temp-file) state.db with the minimal schema and
//! a store.db via the standard fixture.

use super::*;
use crate::blob::test_support::{fixture, TENANT};
use crate::connection::open_store_write;
use crate::schema::ddl_statements;
use tempfile::TempDir;

/// Minimal state.db schema for branches_sync tests.
const STATE_DDL: &[&str] = &[
    "CREATE TABLE projects (\
        project_id INTEGER PRIMARY KEY AUTOINCREMENT, \
        name TEXT NOT NULL, \
        tenant_id TEXT NOT NULL UNIQUE, \
        db_path TEXT NOT NULL, \
        content_key_version INTEGER NOT NULL DEFAULT 3, \
        created_at TEXT NOT NULL, \
        updated_at TEXT NOT NULL)",
    "CREATE TABLE project_locations (\
        location_id INTEGER PRIMARY KEY AUTOINCREMENT, \
        project_id INTEGER NOT NULL REFERENCES projects(project_id), \
        location TEXT NOT NULL, \
        branch_name TEXT NOT NULL, \
        branch_id TEXT NOT NULL UNIQUE, \
        active INTEGER NOT NULL DEFAULT 1, \
        sync_state TEXT NOT NULL DEFAULT 'pending', \
        last_synced TEXT, \
        created_at TEXT NOT NULL, \
        updated_at TEXT NOT NULL)",
];

/// Build a fresh in-memory state.db with the minimal schema.
async fn open_state_db() -> (sqlx::SqlitePool, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("state.db");
    let pool = open_store_write(&path).await.expect("open state");
    for stmt in STATE_DDL {
        sqlx::query(stmt).execute(&pool).await.expect("state ddl");
    }
    (pool, dir)
}

/// Insert a project + location row into state.db.
async fn insert_location(
    state_pool: &sqlx::SqlitePool,
    tenant_id: &str,
    branch_id: &str,
    branch_name: &str,
    location: &str,
) {
    // Upsert the project row.
    sqlx::query(
        "INSERT INTO projects(name, tenant_id, db_path, created_at, updated_at) \
         VALUES (?, ?, '/tmp/store.db', '2026-01-01', '2026-01-01') \
         ON CONFLICT(tenant_id) DO NOTHING",
    )
    .bind(tenant_id)
    .bind(tenant_id)
    .execute(state_pool)
    .await
    .expect("project upsert");

    let project_id: i64 = sqlx::query_scalar("SELECT project_id FROM projects WHERE tenant_id=?")
        .bind(tenant_id)
        .fetch_one(state_pool)
        .await
        .expect("project_id");

    sqlx::query(
        "INSERT INTO project_locations(project_id, location, branch_name, branch_id, \
         created_at, updated_at) \
         VALUES (?, ?, ?, ?, '2026-01-01', '2026-01-01')",
    )
    .bind(project_id)
    .bind(location)
    .bind(branch_name)
    .bind(branch_id)
    .execute(state_pool)
    .await
    .expect("location insert");
}

/// Count branches in store.db.
async fn branch_count(pool: &sqlx::SqlitePool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM branches")
        .fetch_one(pool)
        .await
        .expect("branch count")
}

// AC-F15.2: branch present in state.db but absent from store.db is inserted.
#[tokio::test]
async fn sync_inserts_missing_branch() {
    let fx = fixture("branch-existing").await;
    let (state_pool, _dir) = open_state_db().await;

    insert_location(&state_pool, TENANT, "branch-new", "feature/x", "/repo").await;

    let inserted = run_branches_sync(&fx.pool, &state_pool, TENANT)
        .await
        .expect("sync");

    assert_eq!(inserted, 1, "one new branch must be inserted");
    let count = branch_count(&fx.pool).await;
    // fixture already has "branch-existing" + the new one = 2.
    assert_eq!(count, 2);
}

// AC-F15.2: branch already in store.db is not duplicated (ON CONFLICT DO NOTHING).
#[tokio::test]
async fn sync_skips_existing_branch() {
    let fx = fixture("branch-a").await;
    let (state_pool, _dir) = open_state_db().await;

    // state.db has the same branch_id that the fixture already inserted.
    insert_location(&state_pool, TENANT, "branch-a", "main", "/repo").await;

    let inserted = run_branches_sync(&fx.pool, &state_pool, TENANT)
        .await
        .expect("sync");

    assert_eq!(inserted, 0, "existing branch must not be re-inserted");
    assert_eq!(branch_count(&fx.pool).await, 1, "no duplicate branches");
}

// AC-F15.2: state.db wins -- missing branch is added (additive-only).
#[tokio::test]
async fn sync_additive_only_does_not_delete_extra_store_branches() {
    let fx = fixture("branch-a").await;
    let (state_pool, _dir) = open_state_db().await;

    // state.db has no rows for this tenant -- store.db has "branch-a".
    // sync must be a no-op (additive only: does not delete the extra branch).
    let inserted = run_branches_sync(&fx.pool, &state_pool, TENANT)
        .await
        .expect("sync");

    assert_eq!(inserted, 0);
    assert_eq!(
        branch_count(&fx.pool).await,
        1,
        "additive-only: store branch not deleted even if absent from state.db"
    );
}

// Multiple locations for the same tenant all get synced.
#[tokio::test]
async fn sync_multiple_locations() {
    // Start with a store that has no branches (fresh schema, no fixture branch).
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("store.db");
    let store_pool = open_store_write(&path).await.expect("open store");
    for stmt in ddl_statements() {
        sqlx::query(stmt)
            .execute(&store_pool)
            .await
            .expect("store ddl");
    }
    sqlx::query("INSERT INTO store_meta(tenant_id) VALUES (?)")
        .bind(TENANT)
        .execute(&store_pool)
        .await
        .expect("store_meta");

    let (state_pool, _state_dir) = open_state_db().await;
    insert_location(&state_pool, TENANT, "br-1", "main", "/repo1").await;
    insert_location(&state_pool, TENANT, "br-2", "feat/x", "/repo2").await;
    insert_location(&state_pool, TENANT, "br-3", "feat/y", "/repo3").await;

    let inserted = run_branches_sync(&store_pool, &state_pool, TENANT)
        .await
        .expect("sync multi");

    assert_eq!(inserted, 3, "all three locations must be inserted");
    assert_eq!(branch_count(&store_pool).await, 3);
}
