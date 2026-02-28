//! Query and listing tests for ProjectService

use tonic::Request;

use crate::proto::project_service_server::ProjectService;
use crate::proto::{
    DeprioritizeProjectRequest, GetProjectStatusRequest, HeartbeatRequest, ListProjectsRequest,
    RegisterProjectRequest,
};

use super::{create_test_watch_folder, setup_test_db, ProjectServiceImpl};

#[tokio::test]
async fn test_deprioritize_project() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
    });

    let response = service.deprioritize_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.success);
    assert!(!response.is_active);
    assert_eq!(response.new_priority, "normal");
}

#[tokio::test]
async fn test_get_project_status() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;
    sqlx::query("UPDATE watch_folders SET git_remote_url = ?1 WHERE tenant_id = ?2")
        .bind("https://github.com/user/repo.git")
        .bind("abcd12345678")
        .execute(&pool)
        .await
        .unwrap();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("My Project".to_string()),
        git_remote: Some("https://github.com/user/repo.git".to_string()),
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(GetProjectStatusRequest {
        project_id: "abcd12345678".to_string(),
    });

    let response = service.get_project_status(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.found);
    assert_eq!(response.project_id, "abcd12345678");
    assert_eq!(response.project_name, "project");
    assert_eq!(response.project_root, "/test/project");
    assert_eq!(response.priority, "high");
    assert!(response.is_active);
    assert_eq!(
        response.git_remote,
        Some("https://github.com/user/repo.git".to_string())
    );
}

#[tokio::test]
async fn test_get_nonexistent_project_status() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(GetProjectStatusRequest {
        project_id: "nonexistent12".to_string(),
    });

    let response = service.get_project_status(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.found);
}

#[tokio::test]
async fn test_list_projects() {
    let (pool, _temp_dir) = setup_test_db().await;

    let project_ids = ["aaa000000001", "bbb000000002", "ccc000000003"];
    for (i, project_id) in project_ids.iter().enumerate() {
        create_test_watch_folder(&pool, project_id, &format!("/test/project{i}")).await;
    }

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(ListProjectsRequest {
        priority_filter: None,
        active_only: false,
    });

    let response = service.list_projects(request).await.unwrap();
    let response = response.into_inner();

    assert_eq!(response.total_count, 3);
    assert_eq!(response.projects.len(), 3);
}

#[tokio::test]
async fn test_list_projects_active_only() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
    });
    service.deprioritize_project(request).await.unwrap();

    let request = Request::new(ListProjectsRequest {
        priority_filter: None,
        active_only: true,
    });

    let response = service.list_projects(request).await.unwrap();
    let response = response.into_inner();

    assert_eq!(response.total_count, 0);
}

#[tokio::test]
async fn test_heartbeat() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(HeartbeatRequest {
        project_id: "abcd12345678".to_string(),
    });

    let response = service.heartbeat(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.acknowledged);
    assert!(response.next_heartbeat_by.is_some());
}
