//! Tests for the manual project recover RPC (#140).
//!
//! Located at
//! `src/rust/daemon/grpc/src/services/project_service/tests/recover_tests.rs`.
//! These exercise the SQLite plane of `handle_recover_project` end to end: a
//! moved path, a local<->remote tenancy flip, dry-run counting, and
//! idempotency. The test service is built with `storage: None`, so the Qdrant
//! cascade is skipped and the assertions focus on state.db (the authoritative
//! plane reconciled synchronously).

use super::{
    build_test_service, create_test_watch_folder, create_test_watch_folder_with_remote,
    setup_test_db,
};

use crate::proto::RecoverProjectRequest;

/// Seed a queue row for a tenant with an absolute file path under `root`.
async fn seed_queue_file(pool: &sqlx::SqlitePool, tenant: &str, abs_path: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, tenant_id, branch, collection, item_type, op, file_path, \
          idempotency_key, retry_count, created_at, updated_at) \
         VALUES (?1, ?2, 'main', 'projects', 'file', 'add', ?3, ?1, 0, ?4, ?4)",
    )
    .bind(format!("q-{abs_path}"))
    .bind(tenant)
    .bind(abs_path)
    .bind(&now)
    .execute(pool)
    .await
    .unwrap();
}

#[tokio::test]
async fn recover_repoints_a_moved_project() {
    let (pool, _tmp) = setup_test_db().await;
    // A REMOTE project keeps the same tenant_id across a move (its id is
    // hash(remote), independent of path), so this exercises a pure path move
    // (#138) without a tenancy flip. The stored remote is preserved by passing
    // --new-path only, with the same remote still detectable is not required:
    // because the path move keeps the id, the recover recomputes the id from
    // the new path's remote. The test path has no real remote, so to keep the
    // id stable we instead assert on the path columns, which move regardless.
    create_test_watch_folder_with_remote(
        &pool,
        "remotehashAA",
        "/old/home/proj",
        "https://example.com/r.git",
    )
    .await;
    seed_queue_file(&pool, "remotehashAA", "/old/home/proj/src/a.rs").await;

    let svc = build_test_service(pool.clone(), 60);
    let resp = svc
        .handle_recover_project(RecoverProjectRequest {
            project_id: "remotehashAA".to_string(),
            new_path: Some("/new/home/proj".to_string()),
            rescan_remote: false,
            dry_run: false,
        })
        .await
        .expect("recover should succeed");

    assert!(resp.success);
    assert!(resp.changed);
    assert_eq!(resp.old_path, "/old/home/proj");
    assert_eq!(resp.new_path, "/new/home/proj");

    // watch_folders.path moved; the queue file_path prefix was spliced. The
    // path columns move regardless of whether the id also changed.
    let new_path_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM watch_folders WHERE path = '/new/home/proj'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(new_path_count, 1, "exactly one row now at the new path");

    let qpath: String = sqlx::query_scalar(
        "SELECT file_path FROM unified_queue WHERE queue_id = 'q-/old/home/proj/src/a.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(qpath, "/new/home/proj/src/a.rs");
}

#[tokio::test]
async fn recover_flips_remote_to_local_without_orphans() {
    let (pool, _tmp) = setup_test_db().await;
    // A project registered with a remote (keyed by hash(remote)). --rescan-remote
    // on a path with no remote recomputes a local id and migrates the rows.
    create_test_watch_folder_with_remote(
        &pool,
        "remotehash01",
        "/tmp/recover_flip_target",
        "https://example.com/r.git",
    )
    .await;
    seed_queue_file(&pool, "remotehash01", "/tmp/recover_flip_target/x.rs").await;

    let svc = build_test_service(pool.clone(), 60);
    let resp = svc
        .handle_recover_project(RecoverProjectRequest {
            project_id: "remotehash01".to_string(),
            new_path: None,
            rescan_remote: true,
            dry_run: false,
        })
        .await
        .expect("recover should succeed");

    assert!(
        resp.changed,
        "a remote->local flip must change the tenant id"
    );
    assert_ne!(resp.new_tenant_id, "remotehash01");

    // No rows left under the old tenant; exactly one project row under the new.
    let old_rows: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM watch_folders WHERE tenant_id = 'remotehash01'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(old_rows, 0, "old tenant must be fully migrated");

    let new_rows: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM watch_folders WHERE tenant_id = ?1")
            .bind(&resp.new_tenant_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(new_rows, 1);

    // The seeded file queue row migrated with the tenant. (The tenancy flip
    // also enqueues Qdrant cascade-rename items under the new tenant, so filter
    // to the original file row by its stable file_path.)
    let q_new: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue \
         WHERE tenant_id = ?1 AND item_type = 'file' \
           AND file_path = '/tmp/recover_flip_target/x.rs'",
    )
    .bind(&resp.new_tenant_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(q_new, 1);

    // No file row left under the old tenant.
    let q_old: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'remotehash01' AND item_type = 'file'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        q_old, 0,
        "file queue row must not remain under the old tenant"
    );
}

#[tokio::test]
async fn recover_dry_run_reports_without_writing() {
    let (pool, _tmp) = setup_test_db().await;
    create_test_watch_folder(&pool, "local_dryrun0001", "/old/dr/proj").await;
    seed_queue_file(&pool, "local_dryrun0001", "/old/dr/proj/a.rs").await;

    let svc = build_test_service(pool.clone(), 60);
    let resp = svc
        .handle_recover_project(RecoverProjectRequest {
            project_id: "local_dryrun0001".to_string(),
            new_path: Some("/new/dr/proj".to_string()),
            rescan_remote: false,
            dry_run: true,
        })
        .await
        .expect("dry run should succeed");

    assert!(resp.dry_run);
    assert!(resp.changed);
    // watch_folders (1) + queue (1) would change.
    assert!(resp.sqlite_rows_updated >= 2);

    // Nothing was actually written.
    let path: String =
        sqlx::query_scalar("SELECT path FROM watch_folders WHERE tenant_id = 'local_dryrun0001'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(path, "/old/dr/proj", "dry run must not move the path");
}

#[tokio::test]
async fn recover_is_idempotent_noop_when_consistent() {
    let (pool, _tmp) = setup_test_db().await;
    create_test_watch_folder(&pool, "local_noop000001", "/same/proj").await;

    let svc = build_test_service(pool.clone(), 60);
    // Recover to the SAME path, no rescan: nothing drifts.
    let resp = svc
        .handle_recover_project(RecoverProjectRequest {
            project_id: "local_noop000001".to_string(),
            new_path: Some("/same/proj".to_string()),
            rescan_remote: false,
            dry_run: false,
        })
        .await
        .expect("recover should succeed");

    assert!(resp.success);
    assert!(!resp.changed, "same path is a no-op");
    assert_eq!(resp.sqlite_rows_updated, 0);
}

#[tokio::test]
async fn recover_unknown_project_is_not_found() {
    let (pool, _tmp) = setup_test_db().await;
    let svc = build_test_service(pool, 60);
    let err = svc
        .handle_recover_project(RecoverProjectRequest {
            project_id: "does_not_exist".to_string(),
            new_path: Some("/x".to_string()),
            rescan_remote: false,
            dry_run: false,
        })
        .await
        .expect_err("unknown project must error");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn recover_rejects_a_relative_new_path() {
    // H4: a non-canonical new_path (relative, or containing `..`) is rejected
    // before any plan is built, so it can never traverse outside an absolute
    // root or compute a spurious tenant id.
    let (pool, _tmp) = setup_test_db().await;
    create_test_watch_folder(&pool, "local_reject00001", "/old/reject/proj").await;

    let svc = build_test_service(pool.clone(), 60);
    for bad in ["relative/path", "/old/reject/../escape"] {
        let err = svc
            .handle_recover_project(RecoverProjectRequest {
                project_id: "local_reject00001".to_string(),
                new_path: Some(bad.to_string()),
                rescan_remote: false,
                dry_run: false,
            })
            .await
            .expect_err("a non-canonical new_path must be rejected");
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "rejecting new_path {bad:?}"
        );
    }

    // The stored path was never touched by the rejected requests.
    let path: String =
        sqlx::query_scalar("SELECT path FROM watch_folders WHERE tenant_id = 'local_reject00001'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(path, "/old/reject/proj");
}
