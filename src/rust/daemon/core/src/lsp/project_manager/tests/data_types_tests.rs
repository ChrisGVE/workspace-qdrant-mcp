//! Tests for data types: ProjectLanguageKey, EnrichmentStatus, ProjectLspConfig,
//! Reference, TypeInfo, ResolvedImport, LspEnrichment, and ProjectServerState.

use std::path::PathBuf;

use chrono::Utc;

use super::super::*;
use crate::lsp::{Language, ServerStatus};

#[tokio::test]
async fn test_project_language_key() {
    let key1 = ProjectLanguageKey::new("project-1", Language::Rust);
    let key2 = ProjectLanguageKey::new("project-1", Language::Rust);
    let key3 = ProjectLanguageKey::new("project-2", Language::Rust);

    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[tokio::test]
async fn test_enrichment_status() {
    let status = EnrichmentStatus::Success;
    assert_eq!(status, EnrichmentStatus::Success);
}

#[tokio::test]
async fn test_project_lsp_config_default() {
    let config = ProjectLspConfig::default();
    assert_eq!(config.max_servers_per_project, 3);
    assert!(config.auto_start_on_activation);
    assert_eq!(config.deactivation_delay_secs, 60);
}

#[tokio::test]
async fn test_reference_serialization() {
    let reference = Reference {
        file: "src/main.rs".to_string(),
        line: 10,
        column: 5,
        end_line: Some(10),
        end_column: Some(15),
    };

    let json = serde_json::to_string(&reference).unwrap();
    let deserialized: Reference = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.file, "src/main.rs");
    assert_eq!(deserialized.line, 10);
}

#[tokio::test]
async fn test_lsp_enrichment_skipped() {
    let enrichment = LspEnrichment {
        references: Vec::new(),
        type_info: None,
        resolved_imports: Vec::new(),
        definition: None,
        enrichment_status: EnrichmentStatus::Skipped,
        error_message: Some("Project not active".to_string()),
    };

    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Skipped);
    assert!(enrichment.error_message.is_some());
}

#[tokio::test]
async fn test_type_info_structure() {
    let type_info = TypeInfo {
        type_signature: "fn foo() -> i32".to_string(),
        documentation: Some("Returns a number".to_string()),
        kind: "function".to_string(),
        container: Some("MyModule".to_string()),
    };

    assert_eq!(type_info.type_signature, "fn foo() -> i32");
    assert!(type_info.documentation.is_some());
    assert_eq!(type_info.kind, "function");
    assert!(type_info.container.is_some());
}

#[tokio::test]
async fn test_resolved_import_structure() {
    let import = ResolvedImport {
        import_name: "std::collections::HashMap".to_string(),
        target_file: Some("/usr/lib/rust/std/collections/hash_map.rs".to_string()),
        target_symbol: Some("HashMap".to_string()),
        is_stdlib: true,
        resolved: true,
    };

    assert!(import.is_stdlib);
    assert!(import.resolved);
    assert!(import.target_file.is_some());
}

#[tokio::test]
async fn test_resolved_import_unresolved() {
    let import = ResolvedImport {
        import_name: "unknown_crate::Thing".to_string(),
        target_file: None,
        target_symbol: None,
        is_stdlib: false,
        resolved: false,
    };

    assert!(!import.is_stdlib);
    assert!(!import.resolved);
    assert!(import.target_file.is_none());
}

#[tokio::test]
async fn test_lsp_enrichment_success() {
    let enrichment = LspEnrichment {
        references: vec![Reference {
            file: "src/main.rs".to_string(),
            line: 10,
            column: 5,
            end_line: Some(10),
            end_column: Some(15),
        }],
        type_info: Some(TypeInfo {
            type_signature: "fn main()".to_string(),
            documentation: None,
            kind: "function".to_string(),
            container: None,
        }),
        resolved_imports: vec![],
        definition: None,
        enrichment_status: EnrichmentStatus::Success,
        error_message: None,
    };

    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Success);
    assert_eq!(enrichment.references.len(), 1);
    assert!(enrichment.type_info.is_some());
    assert!(enrichment.error_message.is_none());
}

#[tokio::test]
async fn test_lsp_enrichment_partial() {
    let enrichment = LspEnrichment {
        references: vec![],
        type_info: Some(TypeInfo {
            type_signature: "i32".to_string(),
            documentation: None,
            kind: "type".to_string(),
            container: None,
        }),
        resolved_imports: vec![],
        definition: None,
        enrichment_status: EnrichmentStatus::Partial,
        error_message: None,
    };

    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Partial);
}

#[tokio::test]
async fn test_lsp_enrichment_failed() {
    let enrichment = LspEnrichment {
        references: vec![],
        type_info: None,
        resolved_imports: vec![],
        definition: None,
        enrichment_status: EnrichmentStatus::Failed,
        error_message: Some("LSP server crashed".to_string()),
    };

    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Failed);
    assert!(enrichment.error_message.is_some());
}

#[tokio::test]
async fn test_project_server_state_initial() {
    let state = ProjectServerState {
        project_id: "test-project".to_string(),
        language: Language::Rust,
        project_root: PathBuf::from("/test/path"),
        status: ServerStatus::Initializing,
        restart_count: 0,
        last_error: None,
        is_active: false,
        last_healthy_time: None,
        last_enrichment_at: None,
        marked_unavailable: false,
    };

    assert_eq!(state.status, ServerStatus::Initializing);
    assert!(!state.is_active);
    assert_eq!(state.restart_count, 0);
    assert!(state.last_error.is_none());
}

#[tokio::test]
async fn test_project_server_state_running() {
    let state = ProjectServerState {
        project_id: "test-project".to_string(),
        language: Language::Python,
        project_root: PathBuf::from("/test/path"),
        status: ServerStatus::Running,
        restart_count: 0,
        last_error: None,
        is_active: true,
        last_healthy_time: Some(Utc::now()),
        last_enrichment_at: Some(std::time::Instant::now()),
        marked_unavailable: false,
    };

    assert_eq!(state.status, ServerStatus::Running);
    assert!(state.is_active);
    assert_eq!(state.project_id, "test-project");
}

#[tokio::test]
async fn test_enrichment_status_equality() {
    assert_eq!(EnrichmentStatus::Success, EnrichmentStatus::Success);
    assert_eq!(EnrichmentStatus::Partial, EnrichmentStatus::Partial);
    assert_eq!(EnrichmentStatus::Failed, EnrichmentStatus::Failed);
    assert_eq!(EnrichmentStatus::Skipped, EnrichmentStatus::Skipped);

    assert_ne!(EnrichmentStatus::Success, EnrichmentStatus::Failed);
    assert_ne!(EnrichmentStatus::Partial, EnrichmentStatus::Skipped);
}

#[tokio::test]
async fn test_enrichment_status_display() {
    assert_eq!(format!("{:?}", EnrichmentStatus::Success), "Success");
    assert_eq!(format!("{:?}", EnrichmentStatus::Partial), "Partial");
    assert_eq!(format!("{:?}", EnrichmentStatus::Failed), "Failed");
    assert_eq!(format!("{:?}", EnrichmentStatus::Skipped), "Skipped");
}

#[test]
fn test_enrichment_status_as_str_lowercase() {
    assert_eq!(EnrichmentStatus::Success.as_str(), "success");
    assert_eq!(EnrichmentStatus::Partial.as_str(), "partial");
    assert_eq!(EnrichmentStatus::Failed.as_str(), "failed");
    assert_eq!(EnrichmentStatus::Skipped.as_str(), "skipped");
}

#[tokio::test]
async fn test_lsp_enrichment_serialization() {
    let enrichment = LspEnrichment {
        references: vec![Reference {
            file: "test.rs".to_string(),
            line: 1,
            column: 0,
            end_line: None,
            end_column: None,
        }],
        type_info: None,
        resolved_imports: vec![],
        definition: None,
        enrichment_status: EnrichmentStatus::Success,
        error_message: None,
    };

    let json = serde_json::to_string(&enrichment).unwrap();
    let deserialized: LspEnrichment = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.references.len(), 1);
    assert_eq!(deserialized.enrichment_status, EnrichmentStatus::Success);
}

#[tokio::test]
async fn test_lsp_enrichment_structure_complete() {
    let enrichment = LspEnrichment {
        references: vec![
            Reference {
                file: "src/lib.rs".to_string(),
                line: 10,
                column: 5,
                end_line: Some(10),
                end_column: Some(15),
            },
            Reference {
                file: "src/main.rs".to_string(),
                line: 25,
                column: 8,
                end_line: None,
                end_column: None,
            },
        ],
        type_info: Some(TypeInfo {
            type_signature: "fn process() -> Result<(), Error>".to_string(),
            documentation: Some("Process the data".to_string()),
            kind: "function".to_string(),
            container: Some("MyModule".to_string()),
        }),
        resolved_imports: vec![ResolvedImport {
            import_name: "std::collections::HashMap".to_string(),
            target_file: Some("/rustlib/std/collections/hash_map.rs".to_string()),
            target_symbol: Some("HashMap".to_string()),
            is_stdlib: true,
            resolved: true,
        }],
        definition: None,
        enrichment_status: EnrichmentStatus::Success,
        error_message: None,
    };

    assert_eq!(enrichment.references.len(), 2);
    assert!(enrichment.type_info.is_some());
    assert_eq!(enrichment.resolved_imports.len(), 1);
    assert!(enrichment.resolved_imports[0].is_stdlib);
    assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Success);
    assert!(enrichment.error_message.is_none());
}

#[tokio::test]
async fn test_health_monitoring_config_defaults() {
    let config = ProjectLspConfig::default();

    assert_eq!(config.health_check_interval_secs, 30);
    assert_eq!(config.max_restarts, 3);
    assert_eq!(config.stability_reset_secs, 3600);
    assert!(config.enable_auto_restart);
}

#[tokio::test]
async fn test_project_server_state_health_tracking() {
    let state = ProjectServerState {
        project_id: "test-project".to_string(),
        language: Language::Rust,
        project_root: PathBuf::from("/test/path"),
        status: ServerStatus::Running,
        restart_count: 0,
        last_error: None,
        is_active: true,
        last_healthy_time: Some(Utc::now()),
        last_enrichment_at: Some(std::time::Instant::now()),
        marked_unavailable: false,
    };

    assert!(state.last_healthy_time.is_some());
    assert!(!state.marked_unavailable);
    assert_eq!(state.restart_count, 0);
}

#[tokio::test]
async fn test_project_server_state_restart_tracking() {
    let mut state = ProjectServerState {
        project_id: "test-project".to_string(),
        language: Language::Python,
        project_root: PathBuf::from("/test/path"),
        status: ServerStatus::Failed,
        restart_count: 2,
        last_error: Some("Connection failed".to_string()),
        is_active: true,
        last_healthy_time: None,
        last_enrichment_at: None,
        marked_unavailable: false,
    };

    state.restart_count += 1;
    assert_eq!(state.restart_count, 3);

    state.marked_unavailable = true;
    assert!(state.marked_unavailable);
}

#[tokio::test]
async fn test_project_lsp_config_from_lsp_settings() {
    let settings = crate::config::LspSettings {
        user_path: Some("/usr/local/bin".to_string()),
        max_servers_per_project: 5,
        auto_start_on_activation: false,
        deactivation_delay_secs: 120,
        enable_enrichment_cache: false,
        cache_ttl_secs: 600,
        startup_timeout_secs: 45,
        request_timeout_secs: 15,
        health_check_interval_secs: 90,
        max_restart_attempts: 5,
        restart_backoff_multiplier: 3.0,
        enable_auto_restart: false,
        stability_reset_secs: 7200,
        idle_timeout_secs: 3600,
    };

    let config = ProjectLspConfig::from(settings);

    assert_eq!(config.user_path, Some("/usr/local/bin".to_string()));
    assert_eq!(config.max_servers_per_project, 5);
    assert!(!config.auto_start_on_activation);
    assert_eq!(config.deactivation_delay_secs, 120);
    assert!(!config.enable_enrichment_cache);
    assert_eq!(config.cache_ttl_secs, 600);
    assert_eq!(config.health_check_interval_secs, 90);
    assert_eq!(config.max_restarts, 5);
    assert!(!config.enable_auto_restart);
    assert_eq!(config.stability_reset_secs, 7200);

    assert_eq!(config.lsp_config.startup_timeout.as_secs(), 45);
    assert_eq!(config.lsp_config.request_timeout.as_secs(), 15);
    assert_eq!(config.lsp_config.health_check_interval.as_secs(), 90);
    assert!(!config.lsp_config.enable_auto_restart);
    assert_eq!(config.lsp_config.max_restart_attempts, 5);
    assert_eq!(config.lsp_config.restart_backoff_multiplier, 3.0);
}
