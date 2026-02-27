//! Tests for LanguageServerManager: creation, stats, server lookups,
//! health checks, metrics, and initialization.

use std::path::Path;

use super::super::*;
use crate::lsp::Language;

#[tokio::test]
async fn test_manager_creation() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let stats = manager.stats().await;
    assert_eq!(stats.active_servers, 0);
    assert_eq!(stats.total_servers, 0);
}

#[tokio::test]
async fn test_has_active_servers_empty() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let has_active = manager.has_active_servers("project-1").await;
    assert!(!has_active);
}

#[tokio::test]
async fn test_health_check_no_servers() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let (checked, restarted, failed) = manager.check_all_servers_health().await;
    assert_eq!(checked, 0);
    assert_eq!(restarted, 0);
    assert_eq!(failed, 0);
}

#[tokio::test]
async fn test_project_health_check_no_servers() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let (checked, restarted, failed) =
        manager.check_project_servers_health("non-existent").await;
    assert_eq!(checked, 0);
    assert_eq!(restarted, 0);
    assert_eq!(failed, 0);
}

#[tokio::test]
async fn test_project_lsp_stats_default() {
    let stats = ProjectLspStats::default();
    assert_eq!(stats.active_servers, 0);
    assert_eq!(stats.total_servers, 0);
    assert_eq!(stats.available_languages, 0);
    assert_eq!(stats.cache_entries, 0);
    assert_eq!(stats.metrics.total_enrichment_queries, 0);
    assert_eq!(stats.metrics.cache_hits, 0);
}

#[tokio::test]
async fn test_lsp_metrics_default() {
    let metrics = LspMetrics::default();
    assert_eq!(metrics.total_enrichment_queries, 0);
    assert_eq!(metrics.successful_enrichments, 0);
    assert_eq!(metrics.partial_enrichments, 0);
    assert_eq!(metrics.failed_enrichments, 0);
    assert_eq!(metrics.skipped_enrichments, 0);
    assert_eq!(metrics.cache_hits, 0);
    assert_eq!(metrics.cache_misses, 0);
    assert_eq!(metrics.total_references_queries, 0);
    assert_eq!(metrics.total_type_info_queries, 0);
    assert_eq!(metrics.total_import_queries, 0);
    assert_eq!(metrics.total_server_starts, 0);
    assert_eq!(metrics.total_server_restarts, 0);
    assert_eq!(metrics.total_server_stops, 0);
}

#[tokio::test]
async fn test_lsp_metrics_rates() {
    let mut metrics = LspMetrics::default();

    assert_eq!(metrics.cache_hit_rate(), 0.0);

    metrics.cache_hits = 7;
    metrics.cache_misses = 3;
    assert_eq!(metrics.cache_hit_rate(), 70.0);

    let mut metrics2 = LspMetrics::default();
    assert_eq!(metrics2.enrichment_success_rate(), 0.0);

    metrics2.total_enrichment_queries = 10;
    metrics2.successful_enrichments = 8;
    assert_eq!(metrics2.enrichment_success_rate(), 80.0);
}

#[tokio::test]
async fn test_lsp_metrics_snapshot() {
    let mut metrics = LspMetrics::default();
    metrics.total_enrichment_queries = 100;
    metrics.cache_hits = 50;

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_enrichment_queries, 100);
    assert_eq!(snapshot.cache_hits, 50);

    let metrics2 = LspMetrics::default();
    assert_eq!(metrics2.total_enrichment_queries, 0);
}

#[tokio::test]
async fn test_manager_get_metrics() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.total_enrichment_queries, 0);

    manager.reset_metrics().await;
    let metrics2 = manager.get_metrics().await;
    assert_eq!(metrics2.total_enrichment_queries, 0);
}

#[tokio::test]
async fn test_get_server_not_found() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let server = manager.get_server("project-1", Language::Rust).await;
    assert!(server.is_none());
}

#[tokio::test]
async fn test_get_server_state_not_found() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let state = manager.get_server_state("project-1", Language::Rust).await;
    assert!(state.is_none());
}

#[tokio::test]
async fn test_get_project_servers_empty() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let servers = manager.get_project_servers("project-1").await;
    assert!(servers.is_empty());
}

#[tokio::test]
async fn test_is_server_running_no_server() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let running = manager.is_server_running("project-1", Language::Rust).await;
    assert!(!running);
}

#[tokio::test]
async fn test_manager_health_check_disabled() {
    let config = ProjectLspConfig {
        enable_auto_restart: false,
        ..Default::default()
    };

    let mut manager = LanguageServerManager::new(config).await.unwrap();
    let result = manager.initialize().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_is_server_ready_for_file_no_instances() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    assert!(
        !manager
            .is_server_ready_for_file("test-project", Path::new("/test/file.rs"))
            .await
    );
    assert!(
        !manager
            .is_server_ready_for_file("test-project", Path::new("/test/file.py"))
            .await
    );
    assert!(
        !manager
            .is_server_ready_for_file("test-project", Path::new("/test/Makefile"))
            .await
    );
}
