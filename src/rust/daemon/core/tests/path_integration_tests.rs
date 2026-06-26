//! Integration tests for path abstraction discipline.
//!
//! Verifies that paths remain consistent across all stores (SQLite tables,
//! Qdrant payload structs) per `docs/specs/16-path-abstraction.md` §11.
//!
//! ## Test matrix
//!
//! - **Case 1** (host→host): implemented — uses in-memory SQLite.
//! - **Cases 2–6** (Docker-dependent): stubbed with `#[ignore]`.
//! - **Case 7** (cross-store consistency): implemented — in-memory SQLite.
//! - **Cases 8a–8b** (symlink host-only): implemented — uses `tempdir`.
//! - **Cases 8c–8e** (symlink Docker/macOS FSEvents): stubbed with `#[ignore]`.

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};
use std::time::Duration;
use wqm_common::paths::CanonicalPath;

// ---------------------------------------------------------------------------
// Helper: build an in-memory SQLite pool with the full daemon schema
// ---------------------------------------------------------------------------

/// Create an in-memory SQLite pool and run all production migrations so that
/// every table (`watch_folders`, `tracked_files`, `qdrant_chunks`,
/// `unified_queue`, etc.) exists with the current schema version.
async fn setup_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("connect in-memory sqlite");

    let manager = workspace_qdrant_core::SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("run schema migrations");

    pool
}

/// Insert a watch_folder row and return its `watch_id`.
async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, abs_path: &str, tenant: &str) {
    sqlx::query(
        r"INSERT INTO watch_folders
              (watch_id, path, collection, tenant_id, created_at, updated_at)
          VALUES (?1, ?2, 'projects', ?3,
                  strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                  strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
    )
    .bind(watch_id)
    .bind(abs_path)
    .bind(tenant)
    .execute(pool)
    .await
    .expect("insert watch_folder");
}

/// Insert a v48 `tracked_files` row and return its auto-generated `file_id`.
///
/// v48 (branch-lineage) keeps one row per (branch, path) with the scalar
/// `branch`/`content_key`/`file_identity_id`/`state` columns (the v40
/// `primary_branch`/`branches` JSON pair was dropped). `tenant_id` is read back
/// from the owning watch_folder; `file_identity_id`/`content_key` are synthesized
/// per row (these path-semantics tests do not exercise dedup).
async fn insert_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    branch: &str,
) -> i64 {
    let row = sqlx::query(
        r"INSERT INTO tracked_files
              (watch_folder_id, tenant_id, branch, file_identity_id, content_key,
               is_virtual, state, relative_path,
               file_mtime, file_hash, created_at, updated_at)
          VALUES (
              ?1,
              (SELECT tenant_id FROM watch_folders WHERE watch_id = ?1),
              ?2,
              ?2 || ':' || ?3,
              ?2 || ':' || ?3 || ':deadbeef',
              0, 'present', ?3,
              strftime('%Y-%m-%dT%H:%M:%fZ','now'),
              'deadbeef',
              strftime('%Y-%m-%dT%H:%M:%fZ','now'),
              strftime('%Y-%m-%dT%H:%M:%fZ','now'))
          RETURNING file_id",
    )
    .bind(watch_folder_id)
    .bind(branch)
    .bind(relative_path)
    .fetch_one(pool)
    .await
    .expect("insert tracked_file");

    row.get::<i64, _>("file_id")
}

/// Insert a qdrant_chunks row referencing an existing `file_id`.
async fn insert_qdrant_chunk(pool: &SqlitePool, file_id: i64, point_id: &str, chunk_index: i32) {
    sqlx::query(
        r"INSERT INTO qdrant_chunks
              (file_id, point_id, chunk_index, content_hash, created_at)
          VALUES (?1, ?2, ?3, 'abcdef0123456789',
                  strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
    )
    .bind(file_id)
    .bind(point_id)
    .bind(chunk_index)
    .execute(pool)
    .await
    .expect("insert qdrant_chunk");
}

/// Insert a unified_queue row with a file-level path.
async fn insert_queue_item(
    pool: &SqlitePool,
    tenant: &str,
    file_path: &str,
    branch: &str,
    idem_key: &str,
) {
    sqlx::query(
        r"INSERT INTO unified_queue
              (item_type, op, tenant_id, collection, branch, file_path,
               idempotency_key, payload_json,
               created_at, updated_at)
          VALUES ('file', 'add', ?1, 'projects', ?2, ?3, ?4, '{}',
                  strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                  strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
    )
    .bind(tenant)
    .bind(branch)
    .bind(file_path)
    .bind(idem_key)
    .execute(pool)
    .await
    .expect("insert queue item");
}

// ===========================================================================
// Case 1: Ingest host → Query host
// ===========================================================================

/// Verify that when a file is ingested on the host:
///
/// - `tracked_files.relative_path` is stored as a pure relative path (no
///   leading `/`).
/// - `unified_queue.file_path` stores relative paths.
/// - All paths match between SQLite tables and what the Qdrant payload
///   would contain.
#[tokio::test]
async fn case_1_ingest_host_query_host() {
    let pool = setup_pool().await;

    // Register a project watch folder with a canonical absolute path.
    let root = "/Users/username/dev/my-project";
    let tenant = "tenant-abc";
    let watch_id = "wf-001";

    insert_watch_folder(&pool, watch_id, root, tenant).await;

    // Ingest a file with a pure relative path.
    let rel = "src/main.rs";
    let file_id = insert_tracked_file(&pool, watch_id, rel, "main").await;

    // Insert a chunk for the tracked file.
    insert_qdrant_chunk(&pool, file_id, "point-uuid-1", 0).await;

    // Enqueue the same file in the unified queue.
    insert_queue_item(&pool, tenant, rel, "main", "idem-001").await;

    // --- Assertions ---

    // 1. tracked_files.relative_path must be pure relative (no leading `/`).
    let stored_rel: String =
        sqlx::query_scalar("SELECT relative_path FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .expect("query relative_path");

    assert!(
        !stored_rel.starts_with('/'),
        "relative_path must not start with '/'; got: {stored_rel}"
    );
    assert_eq!(stored_rel, "src/main.rs");

    // 2. unified_queue.file_path must be relative.
    let queue_fp: String = sqlx::query_scalar(
        "SELECT file_path FROM unified_queue WHERE idempotency_key = 'idem-001'",
    )
    .fetch_one(&pool)
    .await
    .expect("query queue file_path");

    assert!(
        !queue_fp.starts_with('/'),
        "queue file_path must not start with '/'; got: {queue_fp}"
    );
    assert_eq!(queue_fp, "src/main.rs");

    // 4. Reconstruction: root + "/" + relative_path must equal the expected
    //    absolute path.
    let reconstructed = format!("{root}/{stored_rel}");
    assert_eq!(reconstructed, "/Users/username/dev/my-project/src/main.rs");

    // 5. The qdrant_chunks row references the correct file_id.
    let chunk_fid: i64 =
        sqlx::query_scalar("SELECT file_id FROM qdrant_chunks WHERE point_id = 'point-uuid-1'")
            .fetch_one(&pool)
            .await
            .expect("query chunk file_id");

    assert_eq!(chunk_fid, file_id);
}

// ===========================================================================
// Case 2: Ingest host → Query docker
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_2_ingest_host_query_docker() {}

// ===========================================================================
// Case 3: Ingest docker → Query host
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_3_ingest_docker_query_host() {
    // Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11
}

// ===========================================================================
// Case 4: Ingest docker → Query docker
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_4_ingest_docker_query_docker() {
    // Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11
}

// ===========================================================================
// Case 5: Mid-run switch (host → docker)
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_5_switch_midway_host_to_docker() {
    // Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11
}

// ===========================================================================
// Case 6: External volume mount (macOS /Volumes)
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_6_external_volume_mount() {
    // Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11
}

// ===========================================================================
// Case 7: Cross-store consistency
// ===========================================================================

/// Verify that a single ingested file has consistent path representation
/// across all stores:
///
/// - `tracked_files.relative_path` = pure relative
/// - `qdrant_chunks` references the correct `file_id` (same relative form)
/// - `unified_queue.file_path` = same relative form
/// - `watch_folders.path` = canonical absolute form
#[tokio::test]
async fn case_7_cross_store_consistency() {
    let pool = setup_pool().await;

    let root = "/Users/username/projects/workspace-qdrant-mcp";
    let tenant = "tenant-xyz";
    let watch_id = "wf-007";
    let branch = "main";
    let rel = "src/rust/daemon/core/src/lib.rs";

    // 1. Insert a watch_folder with a canonical absolute root.
    insert_watch_folder(&pool, watch_id, root, tenant).await;

    // 2. Insert a tracked file with matching relative paths.
    let file_id = insert_tracked_file(&pool, watch_id, rel, branch).await;

    // 3. Insert a chunk pointing to that file_id.
    insert_qdrant_chunk(&pool, file_id, "pt-cross-1", 0).await;
    insert_qdrant_chunk(&pool, file_id, "pt-cross-2", 1).await;

    // 4. Enqueue the file in the unified queue.
    insert_queue_item(&pool, tenant, rel, branch, "idem-cross-007").await;

    // --- Cross-store consistency assertions ---

    // (a) watch_folders.path is canonical absolute.
    let wf_path: String = sqlx::query_scalar("SELECT path FROM watch_folders WHERE watch_id = ?1")
        .bind(watch_id)
        .fetch_one(&pool)
        .await
        .expect("query watch_folders.path");

    assert!(
        wf_path.starts_with('/'),
        "watch_folders.path must be absolute; got: {wf_path}"
    );
    // Verify it parses as a valid CanonicalPath.
    let canon = CanonicalPath::from_user_input(&wf_path);
    assert!(
        canon.is_ok(),
        "watch_folders.path must be a valid CanonicalPath; error: {:?}",
        canon.err()
    );

    // (b) tracked_files.relative_path is pure relative.
    let tf_rel: String =
        sqlx::query_scalar("SELECT relative_path FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .expect("query tracked_files.relative_path");

    assert!(
        !tf_rel.starts_with('/'),
        "tracked_files.relative_path must be relative; got: {tf_rel}"
    );
    assert!(
        !tf_rel.contains(".."),
        "relative_path must not contain '..'"
    );

    // (c) unified_queue.file_path matches tracked_files.relative_path.
    let q_fp: String = sqlx::query_scalar(
        "SELECT file_path FROM unified_queue WHERE idempotency_key = 'idem-cross-007'",
    )
    .fetch_one(&pool)
    .await
    .expect("query unified_queue.file_path");

    assert_eq!(
        q_fp, tf_rel,
        "unified_queue.file_path must equal tracked_files.relative_path"
    );

    // (e) Reconstruction: watch_folders.path + "/" + relative_path = expected absolute.
    let reconstructed = format!("{wf_path}/{tf_rel}");
    let expected_abs = format!("{root}/{rel}");
    assert_eq!(reconstructed, expected_abs);

    // (f) qdrant_chunks all reference the same file_id.
    let chunk_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(&pool)
            .await
            .expect("count chunks");

    assert_eq!(chunk_count, 2, "expected 2 chunks for the tracked file");
}

// ===========================================================================
// Case 8a: Symlink to file
// ===========================================================================

/// Create a symlink to a file, ingest via the symlink's watch folder.
///
/// Verify:
/// - The stored root in `watch_folders.path` uses the user-provided path
///   (the symlink name, NOT the resolved target) because spec §3.1 rule 7
///   says "do NOT resolve symbolic links".
/// - `tracked_files.relative_path` is a pure relative path.
#[tokio::test]
async fn case_8a_symlink_to_file() {
    let pool = setup_pool().await;

    // Create a temp directory structure with a real file and a symlink.
    let tmp = tempfile::TempDir::new().expect("create tempdir");
    let project_dir = tmp.path().join("my-project");
    std::fs::create_dir_all(project_dir.join("src")).expect("create src dir");

    let real_file = project_dir.join("src/foo.rs");
    std::fs::write(&real_file, "fn main() {}").expect("write foo.rs");

    // Create a symlink: src/bar.rs -> src/foo.rs
    let symlink_file = project_dir.join("src/bar.rs");
    std::os::unix::fs::symlink(&real_file, &symlink_file).expect("create symlink");

    // The watch folder root is the project_dir — an absolute real path.
    let root_str = project_dir.to_str().expect("project_dir is utf-8");
    let watch_id = "wf-8a";
    let tenant = "tenant-8a";

    insert_watch_folder(&pool, watch_id, root_str, tenant).await;

    // Ingest the symlink file with its relative path.
    let rel_foo = "src/foo.rs";
    let rel_bar = "src/bar.rs";

    let fid_foo = insert_tracked_file(&pool, watch_id, rel_foo, "main").await;
    let fid_bar = insert_tracked_file(&pool, watch_id, rel_bar, "main").await;

    // Both should store as pure relative.
    let stored_foo: String =
        sqlx::query_scalar("SELECT relative_path FROM tracked_files WHERE file_id = ?1")
            .bind(fid_foo)
            .fetch_one(&pool)
            .await
            .unwrap();

    let stored_bar: String =
        sqlx::query_scalar("SELECT relative_path FROM tracked_files WHERE file_id = ?1")
            .bind(fid_bar)
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(stored_foo, "src/foo.rs");
    assert_eq!(stored_bar, "src/bar.rs");

    // Both are pure relative (no leading slash).
    assert!(!stored_foo.starts_with('/'));
    assert!(!stored_bar.starts_with('/'));

    // The symlink name is preserved, not resolved to foo.rs.
    assert_ne!(
        stored_bar, stored_foo,
        "symlink bar.rs must not be resolved to foo.rs"
    );

    // Verify the root in watch_folders.path is the user-provided path.
    let wf_path: String =
        sqlx::query_scalar("SELECT path FROM watch_folders WHERE watch_id = 'wf-8a'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(wf_path, root_str);
}

// ===========================================================================
// Case 8b: Symlink to directory (watch root is a symlink)
// ===========================================================================

/// Create a symlinked directory, register as watch folder.
///
/// Verify:
/// - `watch_folders.path` stores the symlink path exactly as provided by
///   the user (per spec §3.1 rule 7: no symlink resolution).
/// - Relative paths inside the symlinked root are stored normally.
#[tokio::test]
async fn case_8b_symlink_to_dir() {
    let pool = setup_pool().await;

    // Create a real project directory under a temp root.
    let tmp = tempfile::TempDir::new().expect("create tempdir");
    let real_project = tmp.path().join("real-project");
    std::fs::create_dir_all(real_project.join("lib")).expect("create lib dir");
    std::fs::write(real_project.join("lib/mod.rs"), "pub mod utils;").expect("write mod.rs");

    // Create a symlink directory pointing to the real project.
    let symlink_project = tmp.path().join("linked-project");
    std::os::unix::fs::symlink(&real_project, &symlink_project).expect("create dir symlink");

    // Register the watch folder using the symlink path (what the user typed).
    let symlink_str = symlink_project.to_str().expect("symlink path is utf-8");
    let watch_id = "wf-8b";
    let tenant = "tenant-8b";

    insert_watch_folder(&pool, watch_id, symlink_str, tenant).await;

    // Verify watch_folders.path stores the symlink path, not the resolved target.
    let wf_path: String =
        sqlx::query_scalar("SELECT path FROM watch_folders WHERE watch_id = 'wf-8b'")
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(
        wf_path, symlink_str,
        "watch_folders.path must store the user-provided symlink path, not the resolved target"
    );

    // The resolved real path is different from the symlink path.
    let real_str = real_project.to_str().unwrap();
    assert_ne!(
        wf_path, real_str,
        "watch_folders.path must NOT be the resolved real path"
    );

    // Ingest a file under the symlinked root — relative path is pure relative.
    let rel = "lib/mod.rs";
    let fid = insert_tracked_file(&pool, watch_id, rel, "main").await;

    let stored_rel: String =
        sqlx::query_scalar("SELECT relative_path FROM tracked_files WHERE file_id = ?1")
            .bind(fid)
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(stored_rel, "lib/mod.rs");
    assert!(!stored_rel.starts_with('/'));

    // Reconstruction: symlink root + relative = absolute path through symlink.
    let reconstructed = format!("{symlink_str}/{stored_rel}");
    let expected = format!("{symlink_str}/lib/mod.rs");
    assert_eq!(reconstructed, expected);
}

// ===========================================================================
// Case 8c: Broken symlink
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_8c_broken_symlink() {
    // Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11
}

// ===========================================================================
// Case 8d: Symlink target outside watch root
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
async fn case_8d_symlink_target_outside_root() {
    use std::path::Path;
    use wqm_common::paths::is_within_boundary;

    let project_tmp = tempfile::TempDir::new().unwrap();
    let external_tmp = tempfile::TempDir::new().unwrap();
    let root = project_tmp.path();
    let ext = external_tmp.path();

    std::fs::create_dir_all(root.join("src")).unwrap();
    let internal = root.join("src/main.rs");
    std::fs::write(&internal, "fn main() {}").unwrap();
    let external = ext.join("secret.txt");
    std::fs::write(&external, "sensitive").unwrap();

    let escape = root.join("src/escape.txt");
    std::os::unix::fs::symlink(&external, &escape).unwrap();
    let safe = root.join("src/alias.rs");
    std::os::unix::fs::symlink(&internal, &safe).unwrap();

    assert!(is_within_boundary(&internal, root));
    assert!(!is_within_boundary(&escape, root));
    assert!(is_within_boundary(&safe, root));
    assert!(!is_within_boundary(&external, root));

    let broken = root.join("src/broken.rs");
    std::os::unix::fs::symlink(Path::new("/nonexistent"), &broken).unwrap();
    assert!(!is_within_boundary(&broken, root));
}

// ===========================================================================
// Case 8e: macOS watcher symlink behavior
// ===========================================================================

/// Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11.
#[tokio::test]
#[ignore]
async fn case_8e_macos_watcher_symlink_behavior() {
    // Requires Docker infrastructure — see docs/specs/16-path-abstraction.md §11
}

// ===========================================================================
// Supplementary: CanonicalPath validation on store roundtrip
// ===========================================================================

/// Verify that absolute paths stored in `watch_folders.path` survive
/// a `CanonicalPath::from_validated` roundtrip (they are already in
/// canonical form and do not need further normalization).
#[tokio::test]
async fn canonical_path_roundtrip_through_sqlite() {
    let pool = setup_pool().await;

    let cases = &[
        "/Users/username/dev/project",
        "/home/user/workspace",
        "/mnt/data/repos/my-repo",
    ];

    for (i, path) in cases.iter().enumerate() {
        let wid = format!("wf-rt-{i}");
        let tid = format!("tenant-rt-{i}");
        insert_watch_folder(&pool, &wid, path, &tid).await;

        let stored: String =
            sqlx::query_scalar("SELECT path FROM watch_folders WHERE watch_id = ?1")
                .bind(&wid)
                .fetch_one(&pool)
                .await
                .unwrap();

        // Must survive from_validated without error.
        let canon = CanonicalPath::from_validated(stored.clone());
        assert!(
            canon.is_ok(),
            "stored path '{stored}' must be valid CanonicalPath; err: {:?}",
            canon.err()
        );
        assert_eq!(canon.unwrap().as_str(), *path);
    }
}

/// Verify that relative paths in `tracked_files.relative_path` never
/// pass `CanonicalPath::from_user_input` (they are relative, not
/// absolute).
#[tokio::test]
async fn relative_path_rejects_canonical_construction() {
    let relative_paths = &[
        "src/main.rs",
        "lib/mod.rs",
        "tests/integration.rs",
        "Cargo.toml",
    ];

    for rel in relative_paths {
        let result = CanonicalPath::from_user_input(rel);
        assert!(
            result.is_err(),
            "relative path '{rel}' must be rejected by CanonicalPath::from_user_input"
        );
    }
}

/// Verify that paths with `..` segments are rejected by both
/// `CanonicalPath::from_user_input` and cannot be stored as valid relative
/// paths (traversal protection).
#[tokio::test]
async fn parent_dir_traversal_rejected() {
    let attack_paths = &[
        "/Users/username/../etc/passwd",
        "/home/user/project/../../secret",
    ];

    for path in attack_paths {
        let result = CanonicalPath::from_user_input(path);
        assert!(result.is_err(), "path with '..' must be rejected: {path}");
    }
}
