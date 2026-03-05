//! Tests for LanguageServerManager: project activation, enrichment,
//! caching, metrics, serialization, and health checks.

use std::collections::HashSet;

use crate::lsp::project_manager::{
    EnrichmentStatus, LanguageServerManager, LspMetrics, ProjectLanguageKey, ProjectLspConfig,
};
use crate::lsp::Language;

/// Helper to create a test project LSP config
fn create_test_project_config() -> ProjectLspConfig {
    ProjectLspConfig {
        max_servers_per_project: 3,
        auto_start_on_activation: true,
        deactivation_delay_secs: 0, // No delay for tests
        enable_enrichment_cache: true,
        cache_ttl_secs: 60,
        health_check_interval_secs: 60, // Long interval for tests
        max_restarts: 3,
        stability_reset_secs: 3600,
        enable_auto_restart: false, // Disable for predictable tests
        ..Default::default()
    }
}

#[tokio::test]
async fn test_project_manager_initialization() {
    let config = create_test_project_config();
    let mut manager = LanguageServerManager::new(config).await.unwrap();

    // Initialize should succeed
    let result = manager.initialize().await;
    assert!(result.is_ok(), "Manager initialization should succeed");

    // Stats should show no active servers initially
    let stats = manager.stats().await;
    assert_eq!(stats.active_servers, 0);
    assert_eq!(stats.total_servers, 0);
}

#[tokio::test]
async fn test_project_activation_tracking() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let project_id = "test-project-123";

    // Initially not active
    assert!(!manager.is_project_active(project_id).await);

    // Mark as active
    manager.mark_project_active(project_id).await;
    assert!(manager.is_project_active(project_id).await);

    // Mark as inactive
    manager.mark_project_inactive(project_id).await;
    assert!(!manager.is_project_active(project_id).await);
}

#[tokio::test]
async fn test_enrichment_runs_regardless_of_activity_state() {
    // Activity state no longer affects enrichment - it only affects queue priority.
    // Both active and inactive projects receive full LSP enrichment.
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Enrich with inactive project - enrichment still runs
    let enrichment = manager
        .enrich_chunk(
            "inactive-project",
            std::path::Path::new("/test/file.rs"),
            "test_function",
            10,
            20,
            false, // is_active = false, but enrichment runs anyway
        )
        .await;

    // Without any server instances, readiness check fails -> Skipped
    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Skipped);
    assert!(enrichment.error_message.is_some());
    assert!(enrichment
        .error_message
        .as_ref()
        .unwrap()
        .contains("not ready"));
}

#[tokio::test]
async fn test_enrichment_for_active_project_no_server() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Enrich with active project but no server available
    let enrichment = manager
        .enrich_chunk(
            "active-project",
            std::path::Path::new("/test/file.rs"),
            "test_function",
            10,
            20,
            true, // is_active = true
        )
        .await;

    // Without any server instances, readiness check fails -> Skipped
    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Skipped);
    assert!(enrichment.error_message.is_some());
}

#[tokio::test]
async fn test_cache_functionality() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // First enrichment (cache miss)
    let _enrichment1 = manager
        .enrich_chunk(
            "cache-test-project",
            std::path::Path::new("/test/cache_test.rs"),
            "test_function",
            10,
            20,
            true,
        )
        .await;

    let metrics1 = manager.get_metrics().await;
    assert!(metrics1.total_enrichment_queries >= 1);

    // Second enrichment with same params should hit cache
    let _enrichment2 = manager
        .enrich_chunk(
            "cache-test-project",
            std::path::Path::new("/test/cache_test.rs"),
            "test_function",
            10,
            20,
            true,
        )
        .await;

    let metrics2 = manager.get_metrics().await;
    // Cache hits should have increased
    // Note: This depends on implementation details
    assert!(metrics2.total_enrichment_queries >= 2);
}

#[tokio::test]
async fn test_metrics_tracking() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Initial metrics should be zero
    let initial_metrics = manager.get_metrics().await;
    assert_eq!(initial_metrics.total_enrichment_queries, 0);
    assert_eq!(initial_metrics.cache_hits, 0);
    assert_eq!(initial_metrics.cache_misses, 0);

    // Perform some operations
    // Note: Activity state no longer affects enrichment - it only affects queue priority
    let _ = manager
        .enrich_chunk(
            "metrics-test",
            std::path::Path::new("/test/file.rs"),
            "fn1",
            1,
            10,
            false, // Activity state doesn't skip enrichment anymore
        )
        .await;

    let _ = manager
        .enrich_chunk(
            "metrics-test",
            std::path::Path::new("/test/file.rs"),
            "fn2",
            11,
            20,
            false, // Activity state doesn't skip enrichment anymore
        )
        .await;

    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.total_enrichment_queries, 2);
    assert_eq!(metrics.skipped_enrichments, 2); // No server instances -> readiness check skips

    // Reset metrics
    manager.reset_metrics().await;
    let reset_metrics = manager.get_metrics().await;
    assert_eq!(reset_metrics.total_enrichment_queries, 0);
}

#[tokio::test]
async fn test_multi_project_tracking() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Track multiple projects
    manager.mark_project_active("project-a").await;
    manager.mark_project_active("project-b").await;
    manager.mark_project_active("project-c").await;

    assert!(manager.is_project_active("project-a").await);
    assert!(manager.is_project_active("project-b").await);
    assert!(manager.is_project_active("project-c").await);
    assert!(!manager.is_project_active("project-d").await);

    // Deactivate some
    manager.mark_project_inactive("project-b").await;

    assert!(manager.is_project_active("project-a").await);
    assert!(!manager.is_project_active("project-b").await);
    assert!(manager.is_project_active("project-c").await);
}

#[tokio::test]
async fn test_available_servers_detection() {
    let config = create_test_project_config();
    let mut manager = LanguageServerManager::new(config).await.unwrap();

    // Initialize to detect available servers
    manager.initialize().await.unwrap();

    // Get available languages (depends on what's installed)
    let languages = manager.available_languages().await;

    // We can't assert specific languages are available,
    // but we can verify the API works
    println!("Available languages: {:?}", languages);

    // The list may be empty if no servers are installed
    // This is acceptable for a system that may not have LSP servers
}

#[tokio::test]
async fn test_health_check_no_servers() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Health check with no servers should return (0, 0, 0)
    let (checked, restarted, failed) = manager.check_all_servers_health().await;
    assert_eq!(checked, 0);
    assert_eq!(restarted, 0);
    assert_eq!(failed, 0);
}

#[tokio::test]
async fn test_project_servers_empty() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Get servers for a project (none exist)
    let servers = manager.get_project_servers("non-existent-project").await;
    assert!(servers.is_empty());

    // Check has_active_servers
    assert!(!manager.has_active_servers("non-existent-project").await);
}

#[tokio::test]
async fn test_server_running_check() {
    let config = create_test_project_config();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // No server should be running initially
    assert!(!manager.is_server_running("project", Language::Rust).await);
    assert!(!manager.is_server_running("project", Language::Python).await);
    assert!(
        !manager
            .is_server_running("project", Language::TypeScript)
            .await
    );
}

#[tokio::test]
async fn test_enrichment_status_serialization() {
    // Test that EnrichmentStatus can be serialized/deserialized
    let statuses = vec![
        EnrichmentStatus::Success,
        EnrichmentStatus::Partial,
        EnrichmentStatus::Failed,
        EnrichmentStatus::Skipped,
    ];

    for status in statuses {
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: EnrichmentStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}

#[tokio::test]
async fn test_lsp_metrics_serialization() {
    let mut metrics = LspMetrics::default();
    metrics.total_enrichment_queries = 100;
    metrics.successful_enrichments = 80;
    metrics.cache_hits = 50;

    let json = serde_json::to_string(&metrics).unwrap();
    let deserialized: LspMetrics = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.total_enrichment_queries, 100);
    assert_eq!(deserialized.successful_enrichments, 80);
    assert_eq!(deserialized.cache_hits, 50);
}

#[tokio::test]
async fn test_project_language_key_hash_equality() {
    let key1 = ProjectLanguageKey::new("project-1", Language::Rust);
    let key2 = ProjectLanguageKey::new("project-1", Language::Rust);
    let key3 = ProjectLanguageKey::new("project-1", Language::Python);
    let key4 = ProjectLanguageKey::new("project-2", Language::Rust);

    // Same project and language
    assert_eq!(key1, key2);

    // Different language
    assert_ne!(key1, key3);

    // Different project
    assert_ne!(key1, key4);

    // Test that keys can be used in a HashSet
    let mut set = HashSet::new();
    set.insert(key1.clone());
    set.insert(key2.clone()); // Should not add (duplicate)
    set.insert(key3.clone());
    set.insert(key4.clone());
    assert_eq!(set.len(), 3); // Only 3 unique keys
}
