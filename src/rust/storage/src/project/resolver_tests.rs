//! Tests for `ProjectRegistry` (AC-F10.4, AC-F10.10).
//!
//! File: `wqm-storage/src/project/resolver_tests.rs`
//! Context: sibling test module for `resolver.rs`, split out to keep the main
//!   file under the 500-line budget (arch §9 / coding.md §VIII).

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::PathBuf;
use std::str::FromStr;
use tempfile::NamedTempFile;

use super::{most_specific_match, LocationRow, ProjectRegistry};

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

        // Insert container first, then submodule (intentional order — should not matter).
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
        .execute(&w_pool)
        .await
        .unwrap();

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
        .execute(&w_pool)
        .await
        .unwrap();

        for (name, tenant, db, loc, bn) in &[
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
        ] {
            sqlx::query(
                "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at)
                 VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01')",
            )
            .bind(name)
            .bind(tenant)
            .bind(db)
            .execute(&w_pool)
            .await
            .unwrap();

            sqlx::query(
                "INSERT INTO project_locations
                 (project_id, location, branch_name, branch_id, active, created_at, updated_at)
                 VALUES (
                   (SELECT project_id FROM projects WHERE tenant_id = ?1),
                   ?2, ?3, ?4, 1, '2026-01-01', '2026-01-01'
                 )",
            )
            .bind(tenant)
            .bind(loc)
            .bind(bn)
            .bind(format!("bid-{tenant}"))
            .execute(&w_pool)
            .await
            .unwrap();
        }

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
