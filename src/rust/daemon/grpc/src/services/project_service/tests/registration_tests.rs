//! Registration and validation tests for ProjectService

use tonic::Request;

use crate::proto::project_service_server::ProjectService;
use crate::proto::RegisterProjectRequest;

use super::{create_test_watch_folder, setup_test_db, ProjectServiceImpl};

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
    let canonical_path = std::fs::canonicalize(&project_dir)
        .unwrap()
        .to_string_lossy()
        .to_string();

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
