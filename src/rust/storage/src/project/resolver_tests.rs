//! Tests for `ProjectRegistry` (AC-F10.4, AC-F10.10).
//!
//! File: `wqm-storage/src/project/resolver_tests.rs`
//! Context: sibling test module for `resolver.rs`, split out to keep the main
//!   file under the 500-line budget (arch §9 / coding.md §VIII).

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::PathBuf;
use std::str::FromStr;
use tempfile::NamedTempFile;

use super::{most_specific_match, LocationRow, ProjectRegistry, SearchScope};

/// Create the state.db schema (projects + project_locations + project_groups).
async fn create_schema(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS projects (
            project_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            tenant_id   TEXT NOT NULL UNIQUE,
            db_path     TEXT NOT NULL,
            content_key_version INTEGER NOT NULL DEFAULT 3,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .expect("create projects");

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS project_locations (
            location_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER NOT NULL REFERENCES projects(project_id),
            location    TEXT NOT NULL,
            branch_name TEXT NOT NULL,
            branch_id   TEXT NOT NULL UNIQUE,
            active      INTEGER NOT NULL DEFAULT 1,
            sync_state  TEXT NOT NULL DEFAULT 'current'
                            CHECK (sync_state IN ('pending','indexing','current','error')),
            last_synced TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .expect("create project_locations");

    // project_groups table (schema v24): groups related tenants for scope=group.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS project_groups (
            group_id   TEXT NOT NULL,
            tenant_id  TEXT NOT NULL,
            group_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (group_id, tenant_id)
        )",
    )
    .execute(pool)
    .await
    .expect("create project_groups");
}

// ---------------------------------------------------------------------------
// Test DB bootstrap (writable, not via wqm-storage-write)
// ---------------------------------------------------------------------------

async fn create_writable_pool(path: &std::path::Path) -> SqlitePool {
    let url = format!("sqlite://{}", path.display());
    let opts = SqliteConnectOptions::from_str(&url)
        .expect("url")
        .create_if_missing(true)
        .pragma("foreign_keys", "ON")
        .pragma("journal_mode", "WAL");
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
        .expect("writable pool")
}

/// Seed a minimal state.db with `projects` and `project_locations` tables.
async fn seed_state_db(pool: &SqlitePool, rows: &[(&str, &str, &str, &str, &str)]) {
    // rows: (name, tenant_id, db_path, location, branch_name)
    create_schema(pool).await;

    for (name, tenant_id, db_path, location, branch_name) in rows {
        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01')",
        )
        .bind(name)
        .bind(tenant_id)
        .bind(db_path)
        .execute(pool)
        .await
        .expect("insert project");

        let branch_id = format!("bid-{tenant_id}-{branch_name}");
        sqlx::query(
            "INSERT INTO project_locations
             (project_id, location, branch_name, branch_id, active, created_at, updated_at)
             VALUES (
               (SELECT project_id FROM projects WHERE tenant_id = ?1),
               ?2, ?3, ?4, 1, '2026-01-01', '2026-01-01'
             )",
        )
        .bind(tenant_id)
        .bind(location)
        .bind(branch_name)
        .bind(&branch_id)
        .execute(pool)
        .await
        .expect("insert project_location");
    }
}

// ---------------------------------------------------------------------------
// most_specific_match unit tests (pure logic, no I/O)
// ---------------------------------------------------------------------------

fn make_row(location: &str, tenant_id: &str) -> LocationRow {
    LocationRow {
        location: location.to_string(),
        branch_id: format!("bid-{tenant_id}"),
        tenant_id: tenant_id.to_string(),
        db_path: format!("/data/{tenant_id}/store.db"),
    }
}

// AC-F10.4: most-specific root wins (submodule beats container).
#[test]
fn t_most_specific_submodule_beats_container() {
    let rows = vec![make_row("/a", "container"), make_row("/a/b", "submodule")];
    let cwd = PathBuf::from("/a/b/src/main.rs");
    let binding = most_specific_match(&cwd, &rows).expect("should match");
    assert_eq!(binding.tenant_id.as_str(), "submodule");
}

// AC-F10.4: container matches when CWD is outside any submodule.
#[test]
fn t_most_specific_falls_back_to_container() {
    let rows = vec![make_row("/a", "container"), make_row("/a/b", "submodule")];
    let cwd = PathBuf::from("/a/other/file.rs");
    let binding = most_specific_match(&cwd, &rows).expect("should match");
    assert_eq!(binding.tenant_id.as_str(), "container");
}

// AC-F10.4: unregistered CWD returns None (never defaults to a tenant).
#[test]
fn t_unregistered_cwd_returns_none() {
    let rows = vec![make_row("/registered", "proj")];
    let cwd = PathBuf::from("/totally/unrelated/path");
    assert!(most_specific_match(&cwd, &rows).is_none());
}

// CWD exactly at the root is a match.
#[test]
fn t_cwd_equal_to_root_matches() {
    let rows = vec![make_row("/home/user/project", "proj")];
    let cwd = PathBuf::from("/home/user/project");
    let binding = most_specific_match(&cwd, &rows).expect("exact root should match");
    assert_eq!(binding.tenant_id.as_str(), "proj");
}

// Empty rows always returns None.
#[test]
fn t_empty_registry_returns_none() {
    let rows: Vec<LocationRow> = vec![];
    let cwd = PathBuf::from("/anywhere");
    assert!(most_specific_match(&cwd, &rows).is_none());
}

// ---------------------------------------------------------------------------
// ProjectRegistry integration tests (async, real temp DB)
// ---------------------------------------------------------------------------

// AC-F10.10: type exists and resolves a registered project's CWD to tenant_id.
#[tokio::test]
async fn t_f10_10_registry_resolves_cwd_to_tenant() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        seed_state_db(
            &w_pool,
            &[(
                "MyProj",
                "tenant-abc",
                "/tmp/proj/store.db",
                "/home/user/myproj",
                "main",
            )],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path())
        .await
        .expect("open registry");

    // A path inside the registered root resolves to the project.
    let binding = registry
        .resolve_project("/home/user/myproj/src/lib.rs")
        .await
        .expect("query ok")
        .expect("should resolve");

    assert_eq!(binding.tenant_id.as_str(), "tenant-abc");
    assert_eq!(binding.branch_id.as_str(), "bid-tenant-abc-main");
    assert_eq!(
        binding.db_path,
        std::path::PathBuf::from("/tmp/proj/store.db")
    );
}

// AC-F10.2 / SEC-3: unregistered CWD returns None (never falls through).
#[tokio::test]
async fn t_f10_02_unregistered_cwd_returns_none() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        seed_state_db(
            &w_pool,
            &[(
                "Proj",
                "t1",
                "/data/t1/store.db",
                "/registered/path",
                "main",
            )],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path())
        .await
        .expect("open registry");

    let result = registry
        .resolve_project("/completely/unrelated")
        .await
        .expect("query ok");

    assert!(
        result.is_none(),
        "unregistered CWD must return None — never an all-tenant fall-through (SEC-3)"
    );
}

// AC-F10.4: nested project (submodule beats container).
#[tokio::test]
async fn t_f10_04_nested_project_most_specific_wins() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        // Insert container first, then submodule (order should not matter).
        create_schema(&w_pool).await;
        seed_state_db(
            &w_pool,
            &[
                (
                    "Container",
                    "container-t",
                    "/d/c/store.db",
                    "/work/project",
                    "main",
                ),
                (
                    "Submodule",
                    "submodule-t",
                    "/d/s/store.db",
                    "/work/project/sub",
                    "main",
                ),
            ],
        )
        .await;
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.unwrap();

    // CWD inside the submodule -> submodule wins.
    let b = registry
        .resolve_project("/work/project/sub/src/main.rs")
        .await
        .unwrap()
        .expect("should match");
    assert_eq!(b.tenant_id.as_str(), "submodule-t");

    // CWD inside container but outside submodule -> container wins.
    let b2 = registry
        .resolve_project("/work/project/src/lib.rs")
        .await
        .unwrap()
        .expect("should match");
    assert_eq!(b2.tenant_id.as_str(), "container-t");
}

// list_project_branches returns entries from project_locations.
#[tokio::test]
async fn t_list_project_branches_returns_entries() {
    let tmp = NamedTempFile::new().expect("tempfile");

    {
        let w_pool = create_writable_pool(tmp.path()).await;
        seed_state_db(
            &w_pool,
            &[(
                "Proj",
                "tenant-xyz",
                "/data/xyz/store.db",
                "/mypath",
                "main",
            )],
        )
        .await;
        // Add a second branch location for same tenant.
        sqlx::query(
            "INSERT INTO project_locations
             (project_id, location, branch_name, branch_id, active, sync_state, created_at, updated_at)
             VALUES (
               (SELECT project_id FROM projects WHERE tenant_id = 'tenant-xyz'),
               '/mypath', 'develop', 'bid-develop', 1, 'indexing', '2026-01-01', '2026-01-01'
             )",
        )
        .execute(&w_pool)
        .await
        .unwrap();
        w_pool.close().await;
    }

    let registry = ProjectRegistry::open(tmp.path()).await.unwrap();
    let branches = registry
        .list_project_branches("tenant-xyz")
        .await
        .expect("query ok");

    assert_eq!(branches.len(), 2);
    let names: Vec<&str> = branches.iter().map(|b| b.branch_name.as_str()).collect();
    assert!(names.contains(&"develop"), "develop branch present");
    assert!(names.contains(&"main"), "main branch present");
    // Check sync_state is carried through.
    let dev = branches
        .iter()
        .find(|b| b.branch_name == "develop")
        .unwrap();
    assert_eq!(dev.sync_state, "indexing");
}

// ---------------------------------------------------------------------------
// enumerate_by_scope tests (AC-F17.2, AC-F17 scope=group)
// ---------------------------------------------------------------------------

/// Seed three projects: proj-a and proj-b share "grp-1", proj-c is independent.
async fn seed_three_projects_with_group(pool: &SqlitePool) {
    create_schema(pool).await;

    for (name, tid, dbp, loc, bn) in &[
        ("ProjA", "t-a", "/d/a/store.db", "/loc/a", "main"),
        ("ProjB", "t-b", "/d/b/store.db", "/loc/b", "main"),
        ("ProjC", "t-c", "/d/c/store.db", "/loc/c", "main"),
    ] {
        sqlx::query(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01')",
        )
        .bind(name)
        .bind(tid)
        .bind(dbp)
        .execute(pool)
        .await
        .unwrap();

        let bid = format!("bid-{tid}");
        sqlx::query(
            "INSERT INTO project_locations
             (project_id, location, branch_name, branch_id, active, created_at, updated_at)
             VALUES (
               (SELECT project_id FROM projects WHERE tenant_id = ?1),
               ?2, ?3, ?4, 1, '2026-01-01', '2026-01-01'
             )",
        )
        .bind(tid)
        .bind(loc)
        .bind(bn)
        .bind(&bid)
        .execute(pool)
        .await
        .unwrap();
    }

    // Only proj-a and proj-b share group grp-1; proj-c is isolated.
    for tid in &["t-a", "t-b"] {
        sqlx::query(
            "INSERT INTO project_groups (group_id, tenant_id, group_type, confidence, created_at)
             VALUES ('grp-1', ?1, 'workspace', 1.0, '2026-01-01')",
        )
        .bind(*tid)
        .execute(pool)
        .await
        .unwrap();
    }
}

// scope=project returns only the binding's project.
#[tokio::test]
async fn t_f17_enumerate_scope_project_returns_one() {
    let tmp = NamedTempFile::new().unwrap();
    let pool = create_writable_pool(tmp.path()).await;
    seed_three_projects_with_group(&pool).await;
    pool.close().await;

    let registry = ProjectRegistry::open(tmp.path()).await.unwrap();
    let bindings = registry
        .enumerate_by_scope(SearchScope::Project, "t-a")
        .await
        .unwrap();

    assert_eq!(
        bindings.len(),
        1,
        "scope=project must return exactly one binding"
    );
    assert_eq!(bindings[0].tenant_id.as_str(), "t-a");
}

// scope=group returns only the group members (t-a and t-b), not the outsider
// (t-c). This is the AC-F17 scope=group contract.
#[tokio::test]
async fn t_f17_enumerate_scope_group_excludes_non_member() {
    let tmp = NamedTempFile::new().unwrap();
    let pool = create_writable_pool(tmp.path()).await;
    seed_three_projects_with_group(&pool).await;
    pool.close().await;

    let registry = ProjectRegistry::open(tmp.path()).await.unwrap();
    let bindings = registry
        .enumerate_by_scope(SearchScope::Group, "t-a")
        .await
        .unwrap();

    let tids: Vec<&str> = bindings.iter().map(|b| b.tenant_id.as_str()).collect();
    assert!(tids.contains(&"t-a"), "group member t-a must be included");
    assert!(tids.contains(&"t-b"), "group member t-b must be included");
    assert!(
        !tids.contains(&"t-c"),
        "non-member t-c must NOT appear in scope=group results"
    );
    assert_eq!(bindings.len(), 2, "exactly 2 group members");
}

// scope=all returns every active project.
#[tokio::test]
async fn t_f17_enumerate_scope_all_returns_all_projects() {
    let tmp = NamedTempFile::new().unwrap();
    let pool = create_writable_pool(tmp.path()).await;
    seed_three_projects_with_group(&pool).await;
    pool.close().await;

    let registry = ProjectRegistry::open(tmp.path()).await.unwrap();
    let bindings = registry
        .enumerate_by_scope(SearchScope::All, "t-a")
        .await
        .unwrap();

    assert_eq!(bindings.len(), 3, "scope=all must include all 3 projects");
}

// scope=group falls back to the single project when the tenant has no group.
#[tokio::test]
async fn t_f17_enumerate_scope_group_no_group_falls_back_to_project() {
    let tmp = NamedTempFile::new().unwrap();
    let pool = create_writable_pool(tmp.path()).await;
    seed_three_projects_with_group(&pool).await;
    pool.close().await;

    // t-c has no group membership.
    let registry = ProjectRegistry::open(tmp.path()).await.unwrap();
    let bindings = registry
        .enumerate_by_scope(SearchScope::Group, "t-c")
        .await
        .unwrap();

    assert_eq!(
        bindings.len(),
        1,
        "scope=group with no group membership falls back to project-scope (1 result)"
    );
    assert_eq!(bindings[0].tenant_id.as_str(), "t-c");
}
