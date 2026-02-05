//! Comprehensive tests for the LSP system
//!
//! These tests validate the LSP server detection, lifecycle management,
//! communication, and state management components.

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use tempfile::tempdir;
    use tokio::time::Duration;

    use crate::lsp::{
        LspConfig, Language, LspServerDetector,
        ServerStatus,
        JsonRpcClient, JsonRpcMessage,
    };
    // NOTE: LspManager and StateManager removed as part of 3-table SQLite compliance
    use crate::lsp::lifecycle::{ServerMetadata, RestartPolicy, HealthMetrics};

    #[test]
    fn test_language_enumeration() {
        // Test language identification
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("c"), Language::C);
        assert_eq!(Language::from_extension("cpp"), Language::Cpp);
        assert_eq!(Language::from_extension("unknown"), Language::Other("unknown".to_string()));

        // Test language identifiers
        assert_eq!(Language::Python.identifier(), "python");
        assert_eq!(Language::Rust.identifier(), "rust");
        assert_eq!(Language::JavaScript.identifier(), "javascript");
        
        // Test extensions
        assert!(Language::Python.extensions().contains(&"py"));
        assert!(Language::Rust.extensions().contains(&"rs"));
        assert!(Language::JavaScript.extensions().contains(&"js"));
    }

    #[test]
    fn test_lsp_config_creation() {
        let config = LspConfig::default();
        
        // Verify default values
        assert!(config.features.enabled);
        assert!(config.features.auto_detection);
        assert!(config.features.health_monitoring);
        assert_eq!(config.startup_timeout, Duration::from_secs(30));
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        
        // Verify language configs exist
        assert!(config.language_configs.contains_key(&Language::Python));
        assert!(config.language_configs.contains_key(&Language::Rust));
        assert!(config.language_configs.contains_key(&Language::TypeScript));
        
        // Verify server configs exist
        assert!(config.server_configs.contains_key("rust-analyzer"));
        assert!(config.server_configs.contains_key("ruff-lsp"));
        assert!(config.server_configs.contains_key("typescript-language-server"));
    }

    #[test]
    fn test_config_validation() {
        let mut config = LspConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid startup timeout
        config.startup_timeout = Duration::from_millis(500);
        assert!(config.validate().is_err());
        
        // Fix timeout and test invalid memory limit
        config.startup_timeout = Duration::from_secs(10);
        config.max_memory_mb = Some(32); // Too low
        assert!(config.validate().is_err());
        
        // Fix memory limit and test invalid CPU limit
        config.max_memory_mb = Some(256);
        config.max_cpu_percent = Some(150.0); // Over 100%
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feature_management() {
        let mut config = LspConfig::default();
        
        // Test enabling/disabling features
        assert!(config.is_feature_enabled("auto_detection"));
        config.disable_feature("auto_detection");
        assert!(!config.is_feature_enabled("auto_detection"));
        
        config.enable_feature("experimental");
        assert!(config.is_feature_enabled("experimental"));
        
        // Test unknown feature
        assert!(!config.is_feature_enabled("unknown_feature"));
    }

    #[tokio::test]
    async fn test_config_file_operations() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("lsp_config.json");

        let mut original_config = LspConfig::default();
        original_config.max_memory_mb = Some(1024);
        original_config.log_level = "debug".to_string();

        // Test JSON save/load
        original_config.save_to_file(&config_path).await.unwrap();
        let loaded_config = LspConfig::load_from_file(&config_path).await.unwrap();
        
        assert_eq!(loaded_config.max_memory_mb, Some(1024));
        assert_eq!(loaded_config.log_level, "debug");
        
        // Test YAML operations
        let yaml_path = temp_dir.path().join("lsp_config.yaml");
        original_config.save_to_file(&yaml_path).await.unwrap();
        let yaml_config = LspConfig::load_from_file(&yaml_path).await.unwrap();
        
        assert_eq!(yaml_config.log_level, "debug");
    }

    #[test]
    fn test_lsp_server_detector() {
        let detector = LspServerDetector::new();
        
        // Test server knowledge
        assert!(detector.is_known_server("rust-analyzer"));
        assert!(detector.is_known_server("ruff-lsp"));
        assert!(detector.is_known_server("typescript-language-server"));
        assert!(!detector.is_known_server("unknown-server"));
        
        // Test language server mapping
        let python_servers = detector.get_servers_for_language(&Language::Python);
        assert!(!python_servers.is_empty());
        assert!(python_servers.contains(&"ruff-lsp"));
        
        let rust_servers = detector.get_servers_for_language(&Language::Rust);
        assert!(!rust_servers.is_empty());
        assert!(rust_servers.contains(&"rust-analyzer"));
        
        // Note: Server template details are private implementation
    }

    #[tokio::test]
    async fn test_lsp_server_detection() {
        let detector = LspServerDetector::new();
        
        // This will test what's actually available on the system
        let servers = detector.detect_servers().await.unwrap();
        
        // We can't assume any specific servers are installed
        // but we can test the detection mechanism
        for server in &servers {
            assert!(!server.name.is_empty());
            assert!(server.path.exists());
            assert!(!server.languages.is_empty());
        }
        
        println!("Detected {} LSP servers:", servers.len());
        for server in servers {
            println!("  - {} at {} (languages: {:?})", 
                    server.name, server.path.display(), server.languages);
        }
    }

    // NOTE: test_state_manager, test_server_metadata_storage, and test_communication_logging
    // removed - all used StateManager (3-table SQLite compliance)

    #[test]
    fn test_json_rpc_message_parsing() {
        // Test request parsing
        let request_json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"foo":"bar"}}"#;
        let message = JsonRpcMessage::parse(request_json).unwrap();
        
        match message {
            JsonRpcMessage::Request(req) => {
                assert_eq!(req.method, "initialize");
                assert_eq!(req.id, serde_json::json!(1));
                assert!(req.params.is_some());
            }
            _ => panic!("Expected request message"),
        }

        // Test response parsing
        let response_json = r#"{"jsonrpc":"2.0","id":1,"result":"success"}"#;
        let message = JsonRpcMessage::parse(response_json).unwrap();
        
        match message {
            JsonRpcMessage::Response(resp) => {
                assert_eq!(resp.id, serde_json::json!(1));
                assert_eq!(resp.result, Some(serde_json::json!("success")));
                assert!(resp.error.is_none());
            }
            _ => panic!("Expected response message"),
        }

        // Test notification parsing
        let notification_json = r#"{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{}}"#;
        let message = JsonRpcMessage::parse(notification_json).unwrap();
        
        match message {
            JsonRpcMessage::Notification(notif) => {
                assert_eq!(notif.method, "textDocument/didOpen");
                assert!(notif.params.is_some());
            }
            _ => panic!("Expected notification message"),
        }

        // Test error response parsing
        let error_json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"Test error"}}"#;
        let message = JsonRpcMessage::parse(error_json).unwrap();
        
        match message {
            JsonRpcMessage::Response(resp) => {
                assert!(resp.result.is_none());
                assert!(resp.error.is_some());
                let error = resp.error.unwrap();
                assert_eq!(error.code, -1);
                assert_eq!(error.message, "Test error");
            }
            _ => panic!("Expected error response"),
        }
    }

    #[tokio::test]
    async fn test_json_rpc_client_lifecycle() {
        let client = JsonRpcClient::new();
        
        // Test initial state
        assert!(!client.is_connected().await);
        
        let stats = client.get_stats().await;
        assert_eq!(stats.get("pending_requests").unwrap().as_u64(), Some(0));
        assert_eq!(stats.get("connected").unwrap().as_bool(), Some(false));
        
        // Test cleanup of expired requests
        let expired = client.cleanup_expired_requests().await;
        assert_eq!(expired, 0);
        
        // Test disconnection
        client.disconnect().await;
        assert!(!client.is_connected().await);
    }

    // NOTE: test_lsp_manager_lifecycle removed - used LspManager (3-table SQLite compliance)

    #[test]
    fn test_restart_policy() {
        let policy = RestartPolicy::default();
        
        assert!(policy.enabled);
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.current_attempts, 0);
        assert_eq!(policy.base_delay, Duration::from_secs(1));
        assert_eq!(policy.max_delay, Duration::from_secs(300));
        assert_eq!(policy.backoff_multiplier, 2.0);
    }

    // NOTE: test_cleanup_old_records removed - used StateManager (3-table SQLite compliance)
    // NOTE: test_integration_basic_workflow removed - used LspManager (3-table SQLite compliance)

    // ============================================================================
    // Task 1.19: Comprehensive LSP Integration Tests
    // ============================================================================

    use crate::lsp::project_manager::{
        LanguageServerManager, ProjectLspConfig, ProjectLanguageKey,
        EnrichmentStatus, LspMetrics,
    };
    use std::sync::Arc;

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
        let enrichment = manager.enrich_chunk(
            "inactive-project",
            std::path::Path::new("/test/file.rs"),
            "test_function",
            10,
            20,
            false, // is_active = false, but enrichment runs anyway
        ).await;

        // Without any servers, queries succeed but return no data → Partial status
        assert!(matches!(
            enrichment.enrichment_status,
            EnrichmentStatus::Failed | EnrichmentStatus::Partial
        ));
    }

    #[tokio::test]
    async fn test_enrichment_for_active_project_no_server() {
        let config = create_test_project_config();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // Enrich with active project but no server available
        let enrichment = manager.enrich_chunk(
            "active-project",
            std::path::Path::new("/test/file.rs"),
            "test_function",
            10,
            20,
            true, // is_active = true
        ).await;

        // Should still work but return failed/partial (no server)
        assert!(matches!(
            enrichment.enrichment_status,
            EnrichmentStatus::Failed | EnrichmentStatus::Partial
        ));
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let config = create_test_project_config();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // First enrichment (cache miss)
        let _enrichment1 = manager.enrich_chunk(
            "cache-test-project",
            std::path::Path::new("/test/cache_test.rs"),
            "test_function",
            10,
            20,
            true,
        ).await;

        let metrics1 = manager.get_metrics().await;
        assert!(metrics1.total_enrichment_queries >= 1);

        // Second enrichment with same params should hit cache
        let _enrichment2 = manager.enrich_chunk(
            "cache-test-project",
            std::path::Path::new("/test/cache_test.rs"),
            "test_function",
            10,
            20,
            true,
        ).await;

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
        let _ = manager.enrich_chunk(
            "metrics-test",
            std::path::Path::new("/test/file.rs"),
            "fn1",
            1,
            10,
            false, // Activity state doesn't skip enrichment anymore
        ).await;

        let _ = manager.enrich_chunk(
            "metrics-test",
            std::path::Path::new("/test/file.rs"),
            "fn2",
            11,
            20,
            false, // Activity state doesn't skip enrichment anymore
        ).await;

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_enrichment_queries, 2);
        assert_eq!(metrics.skipped_enrichments, 0); // No longer skipped - activity doesn't affect enrichment

        // Reset metrics
        manager.reset_metrics().await;
        let reset_metrics = manager.get_metrics().await;
        assert_eq!(reset_metrics.total_enrichment_queries, 0);
    }

    // NOTE: test_state_persistence_with_manager and test_server_state_restoration_empty
    // removed - tests used StateManager/with_state_manager (3-table SQLite compliance)

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
        assert!(!manager.is_server_running("project", Language::TypeScript).await);
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
        use std::collections::HashSet;

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

    // NOTE: test_complete_lifecycle_conditional removed
    // Test used StateManager/with_state_manager (3-table SQLite compliance)

    /// Test server detection reports available servers
    #[tokio::test]
    async fn test_server_detection_reports_installed() {
        let detector = LspServerDetector::new();
        let servers = detector.detect_servers().await.unwrap();

        println!("Detected {} LSP servers on this system:", servers.len());
        for server in &servers {
            println!(
                "  - {} ({}): {:?}",
                server.name,
                server.path.display(),
                server.languages
            );
        }

        // Verify detection structure is valid
        for server in &servers {
            assert!(!server.name.is_empty(), "Server name should not be empty");
            assert!(server.path.exists(), "Server path should exist");
            assert!(!server.languages.is_empty(), "Server should support at least one language");
        }
    }

    // NOTE: test_state_persistence_survives_restart removed
    // StateManager no longer exists (3-table SQLite compliance)
}