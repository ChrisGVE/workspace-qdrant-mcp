//! Tests for LspConfig creation, validation, feature management, and file I/O.

use tempfile::tempdir;
use tokio::time::Duration;

use crate::lsp::lifecycle::RestartPolicy;
use crate::lsp::{Language, LspConfig};

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
    assert!(config
        .server_configs
        .contains_key("typescript-language-server"));
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
fn test_restart_policy() {
    let policy = RestartPolicy::default();

    assert!(policy.enabled);
    assert_eq!(policy.max_attempts, 5);
    assert_eq!(policy.current_attempts, 0);
    assert_eq!(policy.base_delay, Duration::from_secs(1));
    assert_eq!(policy.max_delay, Duration::from_secs(300));
    assert_eq!(policy.backoff_multiplier, 2.0);
}
