//! Registration and validation tests for ProjectService

use tonic::Request;
use wqm_common::paths::CanonicalPath;

use crate::proto::project_service_server::ProjectService;
use crate::proto::RegisterProjectRequest;

use super::{
    create_test_watch_folder, create_test_watch_folder_with_remote, fetch_project_rows,
    setup_test_db, ProjectServiceImpl,
};

use workspace_qdrant_core::project_disambiguation::ProjectIdCalculator;

/// Build the syntactic-canonical UTF-8 string for a path used as a
/// fixture — mirrors the new `normalize_project_path` semantics on the
/// production side (no fs symlink resolution). Used by tests that
/// previously asserted on fs-canonicalized output.
fn syntactic_canonical_str(p: &std::path::Path) -> String {
    CanonicalPath::from_user_input(p.to_str().unwrap())
        .unwrap()
        .into_string()
}

#[tokio::test]
async fn test_register_new_project_with_register_if_new() {
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let project_path = project_dir.to_string_lossy().to_string();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: project_path,
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: true,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.created);
    assert_eq!(response.project_id, "abcd12345678");
    assert_eq!(response.priority, "high");
    assert!(response.is_active);
    assert!(response.newly_registered);
}

#[tokio::test]
async fn test_register_new_project_without_register_if_new() {
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let project_path = project_dir.to_string_lossy().to_string();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: project_path,
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.created);
    assert_eq!(response.project_id, "abcd12345678");
    assert_eq!(response.priority, "none");
    assert!(!response.is_active);
    assert!(!response.newly_registered);
}

#[tokio::test]
async fn test_register_existing_project() {
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    // Pre-store the watch_folder with the same syntactic-canonical form
    // the handler will derive from req.path (spec §16 §3.1 — symlinks
    // are not followed). Previously this used filesystem-level path
    // resolution; the migration to syntactic normalization changes the
    // expected value here.
    let canonical_path = syntactic_canonical_str(&project_dir);

    create_test_watch_folder(&pool, "abcd12345678", &canonical_path).await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: canonical_path.clone(),
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(RegisterProjectRequest {
        path: canonical_path,
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.created);
    assert!(response.is_active);
    assert!(!response.newly_registered);
}

#[tokio::test]
async fn test_invalid_project_id_format() {
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let project_path = project_dir.to_string_lossy().to_string();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: project_path,
        project_id: "short".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let result = service.register_project(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_empty_project_id_generates_local_id() {
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let project_path = project_dir.to_string_lossy().to_string();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: project_path,
        project_id: "".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.created);
    assert!(response.project_id.starts_with("local_"));
    assert_eq!(response.priority, "none");
    assert!(!response.is_active);
    assert!(!response.newly_registered);
}

#[tokio::test]
async fn test_empty_path_and_project_id_returns_error() {
    // Issue #70: empty path is now allowed as long as project_id is set
    // (activation flow). Both missing must still be rejected.
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "".to_string(),
        project_id: "".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let result = service.register_project(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_empty_path_with_project_id_is_allowed_for_activation() {
    // Issue #70: `wqm project activate` sends an empty path and relies on
    // project_id alone. The handler must accept this and return a
    // "not found" response (priority=none) rather than rejecting.
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service
        .register_project(request)
        .await
        .unwrap()
        .into_inner();
    // Project does not exist and register_if_new=false => not found.
    assert_eq!(response.priority, "none");
    assert!(!response.is_active);
    assert!(!response.newly_registered);
}

// ── F-019 path canonicalization tests ──────────────────────────────

#[tokio::test]
async fn test_nonexistent_path_rejected() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/nonexistent/path/that/does/not/exist".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: true,
        priority: Some("high".to_string()),
    });

    let result = service.register_project(request).await;
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("does not exist"));
}

#[tokio::test]
async fn test_file_path_rejected_not_directory() {
    let (pool, temp_dir) = setup_test_db().await;
    let file_path = temp_dir.path().join("not_a_dir.txt");
    std::fs::write(&file_path, "hello").unwrap();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: file_path.to_string_lossy().to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: true,
        priority: Some("high".to_string()),
    });

    let result = service.register_project(request).await;
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("not a directory"));
}

// ── #138/#139 registration reconciliation tests ─────────────────────
//
// These exercise the path-based reconciliation that runs before the
// id-only new/existing decision. All use an empty `project_id` so the
// handler recomputes the id from path + remote, reproducing the real
// `store type:project` / `wqm project register` flow.

/// The recomputed local id the handler derives for a non-git directory.
fn local_id_for(path: &std::path::Path) -> String {
    ProjectIdCalculator::new().calculate(path, None, None)
}

/// #138 — moving a registered project to a new path must UPDATE the stored
/// path in place (matched by unchanged identity), not silently no-op or
/// create a duplicate row.
#[tokio::test]
async fn test_reconcile_moved_path_updates_registration() {
    let (pool, temp_dir) = setup_test_db().await;

    // The project's identity is its git remote, so its id is stable across
    // the move. Pre-store the old path under that remote-derived id.
    let remote = "https://github.com/ChrisGVE/business";
    let tenant_id = ProjectIdCalculator::new().calculate(
        std::path::Path::new("/old/does/not/matter"),
        Some(remote),
        None,
    );
    let old_path = "/Users/chris/dev/business";
    create_test_watch_folder_with_remote(&pool, &tenant_id, old_path, remote).await;

    // The new location exists on disk (a real temp dir).
    let new_dir = temp_dir.path().join("business");
    std::fs::create_dir_all(&new_dir).unwrap();
    let new_path = syntactic_canonical_str(&new_dir);

    let service = ProjectServiceImpl::new(pool.clone());
    let request = Request::new(RegisterProjectRequest {
        path: new_path.clone(),
        project_id: String::new(),
        name: Some("business".to_string()),
        // Supply the remote explicitly so identity is unchanged across the
        // move (the temp dir has no real git remote).
        git_remote: Some(remote.to_string()),
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service
        .register_project(request)
        .await
        .unwrap()
        .into_inner();

    // Treated as existing (not newly created), now active.
    assert!(!response.created);
    assert!(!response.newly_registered);
    assert!(response.is_active);
    assert_eq!(response.project_id, tenant_id);

    // The stored path was updated, and there is still exactly one row.
    let rows = fetch_project_rows(&pool).await;
    assert_eq!(rows.len(), 1, "moved path must not duplicate the row");
    assert_eq!(rows[0].0, tenant_id);
    assert_eq!(rows[0].1, new_path, "stored path should be the new path");
}

/// #139 — a project that LOSES its git remote (remote -> local) must rename
/// the tenant and keep a single row at the same path, not enqueue a new one.
#[tokio::test]
async fn test_reconcile_remote_to_local_renames_tenant() {
    let (pool, temp_dir) = setup_test_db().await;

    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let path = syntactic_canonical_str(&project_dir);

    // Pre-store the row under a remote-derived id at this exact path.
    let remote = "https://github.com/ChrisGVE/project";
    let old_tenant_id = ProjectIdCalculator::new().calculate(&project_dir, Some(remote), None);
    create_test_watch_folder_with_remote(&pool, &old_tenant_id, &path, remote).await;

    // Register the same path with NO remote: the handler recomputes a
    // local id and must reconcile the flip.
    let new_tenant_id = local_id_for(&project_dir);
    assert_ne!(old_tenant_id, new_tenant_id);

    let service = ProjectServiceImpl::new(pool.clone());
    let request = Request::new(RegisterProjectRequest {
        path: path.clone(),
        project_id: String::new(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("normal".to_string()),
    });

    let response = service
        .register_project(request)
        .await
        .unwrap()
        .into_inner();

    assert!(!response.created);
    assert!(!response.newly_registered);
    assert_eq!(response.project_id, new_tenant_id);

    // Single row, now keyed under the new local id, same path.
    let rows = fetch_project_rows(&pool).await;
    assert_eq!(rows.len(), 1, "tenancy flip must not duplicate the row");
    assert_eq!(rows[0].0, new_tenant_id);
    assert_eq!(rows[0].1, path);
}

/// #139 — a project that GAINS a git remote (local -> remote) must rename the
/// tenant the other direction and keep a single row at the same path.
#[tokio::test]
async fn test_reconcile_local_to_remote_renames_tenant() {
    let (pool, temp_dir) = setup_test_db().await;

    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let path = syntactic_canonical_str(&project_dir);

    // Pre-store the row under the LOCAL id at this exact path.
    let old_tenant_id = local_id_for(&project_dir);
    create_test_watch_folder(&pool, &old_tenant_id, &path).await;

    // Register the same path WITH a remote: the handler recomputes a
    // remote-derived id and must reconcile the flip.
    let remote = "https://github.com/ChrisGVE/project";
    let new_tenant_id = ProjectIdCalculator::new().calculate(&project_dir, Some(remote), None);
    assert_ne!(old_tenant_id, new_tenant_id);
    assert!(old_tenant_id.starts_with("local_"));

    let service = ProjectServiceImpl::new(pool.clone());
    let request = Request::new(RegisterProjectRequest {
        path: path.clone(),
        project_id: String::new(),
        name: None,
        git_remote: Some(remote.to_string()),
        register_if_new: false,
        priority: Some("normal".to_string()),
    });

    let response = service
        .register_project(request)
        .await
        .unwrap()
        .into_inner();

    assert!(!response.created);
    assert!(!response.newly_registered);
    assert_eq!(response.project_id, new_tenant_id);

    let rows = fetch_project_rows(&pool).await;
    assert_eq!(rows.len(), 1, "tenancy flip must not duplicate the row");
    assert_eq!(rows[0].0, new_tenant_id);
    assert_eq!(rows[0].1, path);
}

/// Re-registering a project at the SAME path with the SAME identity must
/// remain a no-op (regression guard: reconciliation must not disturb the
/// already-consistent case).
#[tokio::test]
async fn test_reconcile_noop_when_already_consistent() {
    let (pool, temp_dir) = setup_test_db().await;

    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let path = syntactic_canonical_str(&project_dir);

    let tenant_id = local_id_for(&project_dir);
    create_test_watch_folder(&pool, &tenant_id, &path).await;

    let service = ProjectServiceImpl::new(pool.clone());
    let request = Request::new(RegisterProjectRequest {
        path: path.clone(),
        project_id: String::new(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("normal".to_string()),
    });

    let response = service
        .register_project(request)
        .await
        .unwrap()
        .into_inner();
    assert!(!response.created);
    assert!(!response.newly_registered);
    assert_eq!(response.project_id, tenant_id);

    let rows = fetch_project_rows(&pool).await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, tenant_id);
    assert_eq!(rows[0].1, path);
}
