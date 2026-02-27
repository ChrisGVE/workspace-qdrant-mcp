//! Registration and validation tests for ProjectService

use tonic::Request;

use crate::proto::project_service_server::ProjectService;
use crate::proto::RegisterProjectRequest;

use super::{create_test_watch_folder, setup_test_db, ProjectServiceImpl};

#[tokio::test]
async fn test_register_new_project_with_register_if_new() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
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
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
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
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
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
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
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
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
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
async fn test_empty_path_returns_error() {
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

    let result = service.register_project(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
}
