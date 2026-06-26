//! Negative path validation tests for ProjectService handlers.
//!
//! Tests that invalid path inputs are rejected with `InvalidArgument` status
//! at the gRPC handler boundary, per spec 16-path-abstraction.md section 7.

use tonic::Request;

use crate::proto::project_service_server::ProjectService;
use crate::proto::{DeprioritizeProjectRequest, RegisterProjectRequest};

use super::{create_test_watch_folder, setup_test_db, ProjectServiceImpl};

// ── RegisterProjectRequest.path (canonical) ──────────────────────────

#[tokio::test]
async fn test_register_relative_path_rejected() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "relative/path".to_string(),
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
}

#[tokio::test]
async fn test_register_parent_dir_path_rejected() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/Users/username/../other".to_string(),
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
}

// ── DeprioritizeProjectRequest.watch_path (canonical, optional) ──────

#[tokio::test]
async fn test_deprioritize_relative_watch_path_rejected() {
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();

    create_test_watch_folder(&pool, "abcd12345678", project_dir.to_str().unwrap()).await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
        watch_path: Some("relative/path".to_string()),
    });

    let result = service.deprioritize_project(request).await;
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(
        status.message().contains("watch_path"),
        "error should mention field name, got: {}",
        status.message()
    );
}

#[tokio::test]
async fn test_deprioritize_parent_dir_watch_path_rejected() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
        watch_path: Some("/Users/username/../escape".to_string()),
    });

    let result = service.deprioritize_project(request).await;
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_deprioritize_empty_watch_path_allowed() {
    // Empty watch_path means tenant-wide deprioritization — should not be rejected.
    let (pool, temp_dir) = setup_test_db().await;
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();
    let canonical =
        wqm_common::paths::CanonicalPath::from_user_input(project_dir.to_str().unwrap())
            .unwrap()
            .into_string();

    create_test_watch_folder(&pool, "abcd12345678", &canonical).await;

    let service = ProjectServiceImpl::new(pool);

    // Register first so we can deprioritize.
    let request = Request::new(RegisterProjectRequest {
        path: canonical,
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
        watch_path: Some(String::new()), // empty = tenant-wide
    });

    let result = service.deprioritize_project(request).await;
    assert!(result.is_ok(), "empty watch_path should be allowed");
}
