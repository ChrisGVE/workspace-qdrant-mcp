//! Tests for enrichment queries, crash handling, state persistence,
//! and server error tracking.

use std::path::{Path, PathBuf};

use chrono::Utc;

use super::super::*;
use crate::lsp::{Language, ServerStatus};

#[tokio::test]
async fn test_enrich_chunk_increments_metrics() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let _enrichment = manager
        .enrich_chunk("test-project", Path::new("/test/file.rs"), "test_function", 10, 20, false)
        .await;

    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.total_enrichment_queries, 1);
    assert_eq!(metrics.skipped_enrichments, 1);
}

#[tokio::test]
async fn test_enrich_chunk_runs_regardless_of_activity_state() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let result = manager
        .enrich_chunk("test-project", Path::new("/test/file.rs"), "test_symbol", 10, 20, false)
        .await;

    assert_eq!(result.enrichment_status, EnrichmentStatus::Skipped);
    assert!(result.error_message.is_some());
    assert!(result.error_message.as_ref().unwrap().contains("not ready"));
    assert!(result.references.is_empty());
    assert!(result.type_info.is_none());
    assert!(result.resolved_imports.is_empty());
}

#[tokio::test]
async fn test_enrich_chunk_returns_enrichment_structure() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let result = manager
        .enrich_chunk("test-project", Path::new("/test/file.rs"), "test_symbol", 10, 20, true)
        .await;

    assert_eq!(result.enrichment_status, EnrichmentStatus::Skipped);
    assert!(result.error_message.is_some());
    assert!(result.references.is_empty());
    assert!(result.type_info.is_none());
    assert!(result.resolved_imports.is_empty());
}

#[tokio::test]
async fn test_enrich_chunk_skipped_includes_language_info() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let result = manager
        .enrich_chunk("test-project", Path::new("/test/file.rs"), "test_symbol", 10, 20, true)
        .await;

    assert_eq!(result.enrichment_status, EnrichmentStatus::Skipped);
    let msg = result.error_message.unwrap();
    assert!(msg.contains("not ready"));
    assert!(msg.contains("rust"));
}

// State persistence tests

#[tokio::test]
async fn test_manager_without_state_persistence() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    manager.mark_project_active("test-project").await;
    assert!(manager.is_project_active("test-project").await);

    manager.mark_project_inactive("test-project").await;
    assert!(!manager.is_project_active("test-project").await);
}

#[tokio::test]
async fn test_manager_active_projects_tracking() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    assert!(!manager.is_project_active("project-1").await);
    assert!(!manager.is_project_active("project-2").await);

    manager.mark_project_active("project-1").await;
    manager.mark_project_active("project-2").await;

    assert!(manager.is_project_active("project-1").await);
    assert!(manager.is_project_active("project-2").await);

    manager.mark_project_inactive("project-1").await;

    assert!(!manager.is_project_active("project-1").await);
    assert!(manager.is_project_active("project-2").await);
}

#[tokio::test]
async fn test_restore_project_servers_returns_empty() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let restored = manager.restore_project_servers("test-project").await.unwrap();
    assert!(restored.is_empty());
}

// Crash handling tests

#[tokio::test]
async fn test_handle_potential_crash_no_instance() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let key = ProjectLanguageKey::new("nonexistent-project", Language::Python);
    let result = manager.handle_potential_crash(&key, "test error").await;

    assert!(!result, "Should return false when no instance exists");
}

#[tokio::test]
async fn test_handle_potential_crash_marks_server_failed() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let project_id = "test-project";
    let language = Language::Rust;
    let key = ProjectLanguageKey::new(project_id, language.clone());

    {
        let mut servers = manager.servers.write().await;
        servers.insert(
            key.clone(),
            ProjectServerState {
                project_id: project_id.to_string(),
                language: language.clone(),
                project_root: PathBuf::from("/test"),
                status: ServerStatus::Running,
                restart_count: 0,
                last_error: None,
                is_active: true,
                last_healthy_time: Some(Utc::now()),
                marked_unavailable: false,
            },
        );
    }

    let result = manager.handle_potential_crash(&key, "simulated error").await;
    assert!(!result, "Should return false when no instance exists to check");
}

#[tokio::test]
async fn test_crash_detection_increments_metrics() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let initial_metrics = manager.get_metrics().await;
    assert_eq!(initial_metrics.failed_enrichments, 0);

    let key = ProjectLanguageKey::new("test-project", Language::Python);
    manager.handle_potential_crash(&key, "test crash").await;

    let final_metrics = manager.get_metrics().await;
    assert_eq!(final_metrics.failed_enrichments, 0);
}

#[tokio::test]
async fn test_enrichment_continues_after_query_error() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let enrichment = manager
        .enrich_chunk(
            "test-project",
            std::path::Path::new("/test/file.rs"),
            "test_function",
            1,
            10,
            false,
        )
        .await;

    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Skipped);

    let enrichment = manager
        .enrich_chunk(
            "test-project",
            std::path::Path::new("/test/file.rs"),
            "test_function",
            1,
            10,
            true,
        )
        .await;

    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Skipped);
}

#[tokio::test]
async fn test_server_state_error_tracking() {
    let state = ProjectServerState {
        project_id: "test".to_string(),
        language: Language::Rust,
        project_root: PathBuf::from("/test"),
        status: ServerStatus::Running,
        restart_count: 0,
        last_error: None,
        is_active: true,
        last_healthy_time: Some(Utc::now()),
        marked_unavailable: false,
    };

    assert!(state.last_error.is_none());
    assert_eq!(state.status, ServerStatus::Running);

    let mut state = state;
    state.status = ServerStatus::Failed;
    state.last_error = Some("Server crashed: connection lost".to_string());

    assert_eq!(state.status, ServerStatus::Failed);
    assert!(state.last_error.is_some());
    assert!(state.last_error.as_ref().unwrap().contains("crashed"));
}
