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
        LspConfig, LspManager, Language, LspServerDetector,
        LspStateManager, ServerStatus,
        JsonRpcClient, JsonRpcMessage,
    };
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

    #[tokio::test]
    async fn test_state_manager() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("lsp_test.db");

        let manager = LspStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Test basic stats
        let stats = manager.get_stats().await.unwrap();
        assert!(stats.contains_key("servers_by_status"));
        assert!(stats.contains_key("total_health_records"));

        // Test configuration storage
        let test_value = serde_json::json!({"timeout": 30, "enabled": true});
        manager.set_configuration(None, "test_setting", test_value.clone(), "test").await.unwrap();

        let retrieved = manager.get_configuration(None, "test_setting").await.unwrap();
        assert_eq!(retrieved, Some(test_value));

        // Test non-existent configuration
        let missing = manager.get_configuration(None, "missing_key").await.unwrap();
        assert_eq!(missing, None);

        // Close the manager
        manager.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_server_metadata_storage() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("lsp_metadata_test.db");

        let manager = LspStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create test server metadata
        let server_id = uuid::Uuid::new_v4();
        let metadata = ServerMetadata {
            id: server_id,
            name: "test-server".to_string(),
            executable_path: PathBuf::from("/usr/bin/test-lsp"),
            languages: vec![Language::Python, Language::JavaScript],
            version: Some("1.0.0".to_string()),
            started_at: chrono::Utc::now(),
            process_id: Some(12345),
            working_directory: PathBuf::from("/tmp"),
            environment: std::collections::HashMap::new(),
            arguments: vec!["--stdio".to_string()],
        };

        // Store metadata
        manager.store_server_metadata(&metadata).await.unwrap();

        // Retrieve metadata
        let retrieved = manager.get_server_metadata(&server_id).await.unwrap();
        assert!(retrieved.is_some());
        
        let record = retrieved.unwrap();
        assert_eq!(record.name, "test-server");
        assert_eq!(record.languages.len(), 2);
        assert!(record.languages.contains(&Language::Python));
        assert!(record.languages.contains(&Language::JavaScript));

        // Test health metrics
        let health_metrics = HealthMetrics {
            status: ServerStatus::Running,
            last_healthy: chrono::Utc::now(),
            response_time_ms: 150,
            consecutive_failures: 0,
            requests_processed: 42,
            avg_response_time_ms: 125.5,
            memory_usage_bytes: Some(1024 * 1024),
            cpu_usage_percent: Some(5.2),
        };

        manager.update_health_metrics(&server_id, &health_metrics).await.unwrap();

        // Get health history
        let history = manager.get_health_metrics_history(&server_id, 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].status, ServerStatus::Running);
        assert_eq!(history[0].response_time_ms, 150);

        manager.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_communication_logging() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("lsp_comm_test.db");

        let manager = LspStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create test server
        let server_id = uuid::Uuid::new_v4();
        let metadata = ServerMetadata {
            id: server_id,
            name: "test-lsp".to_string(),
            executable_path: PathBuf::from("/usr/bin/test-lsp"),
            languages: vec![Language::Python],
            version: Some("1.0.0".to_string()),
            started_at: chrono::Utc::now(),
            process_id: Some(12345),
            working_directory: PathBuf::from("/tmp"),
            environment: std::collections::HashMap::new(),
            arguments: vec![],
        };

        manager.store_server_metadata(&metadata).await.unwrap();

        // Log communication events
        manager.log_communication(
            &server_id,
            "outgoing",
            "initialize",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            Some(200),
            true,
            None,
        ).await.unwrap();

        manager.log_communication(
            &server_id,
            "incoming",
            "initialize",
            r#"{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}"#,
            None,
            true,
            None,
        ).await.unwrap();

        manager.log_communication(
            &server_id,
            "outgoing",
            "shutdown",
            r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#,
            Some(100),
            false,
            Some("Server not responding"),
        ).await.unwrap();

        // Verify communication was logged
        let stats = manager.get_stats().await.unwrap();
        let comm_count = stats.get("total_communication_records").unwrap().as_u64().unwrap();
        assert_eq!(comm_count, 3);

        manager.close().await.unwrap();
    }

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

    #[tokio::test]
    async fn test_lsp_manager_lifecycle() {
        let temp_dir = tempdir().unwrap();
        let config = LspConfig {
            database_path: temp_dir.path().join("lsp_manager_test.db"),
            startup_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(5),
            health_check_interval: Duration::from_secs(30),
            ..Default::default()
        };

        // Test manager creation
        let manager = LspManager::new(config.clone()).await.unwrap();
        assert!(manager.get_all_servers().await.is_empty());

        // Test statistics
        let stats = manager.get_stats().await;
        assert!(stats.contains_key("active_servers"));
        assert_eq!(stats.get("active_servers").unwrap().as_u64(), Some(0));
    }

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

    #[tokio::test]
    async fn test_cleanup_old_records() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("lsp_cleanup_test.db");

        let manager = LspStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        // Create some test data
        let server_id = uuid::Uuid::new_v4();
        let metadata = ServerMetadata {
            id: server_id,
            name: "cleanup-test".to_string(),
            executable_path: PathBuf::from("/usr/bin/test"),
            languages: vec![Language::Python],
            version: Some("1.0.0".to_string()),
            started_at: chrono::Utc::now(),
            process_id: Some(12345),
            working_directory: PathBuf::from("/tmp"),
            environment: std::collections::HashMap::new(),
            arguments: vec![],
        };

        manager.store_server_metadata(&metadata).await.unwrap();

        // Add some communication logs
        manager.log_communication(
            &server_id,
            "outgoing",
            "test",
            "test content",
            Some(100),
            true,
            None,
        ).await.unwrap();

        // Cleanup (should not remove anything since data is recent)
        let deleted = manager.cleanup_old_records(1).await.unwrap();
        // Should be 0 since we just created the records
        assert_eq!(deleted, 0);

        manager.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_integration_basic_workflow() {
        let temp_dir = tempdir().unwrap();
        let config = LspConfig {
            database_path: temp_dir.path().join("integration_test.db"),
            startup_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(3),
            health_check_interval: Duration::from_secs(60), // Long interval for test
            ..Default::default()
        };

        // Create LSP manager
        let manager = LspManager::new(config).await.unwrap();

        // Test getting server for a language (will be None since no servers are running)
        let python_server = manager.get_server(&Language::Python).await;
        assert!(python_server.is_none());

        // Test getting stats
        let stats = manager.get_stats().await;
        assert_eq!(stats.get("active_servers").unwrap().as_u64(), Some(0));
        
        println!("Integration test completed successfully");
    }
}