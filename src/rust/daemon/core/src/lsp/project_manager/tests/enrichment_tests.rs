//! Tests for enrichment queries, crash handling, state persistence,
//! and server error tracking.

use std::path::{Path, PathBuf};

use chrono::Utc;

use super::super::*;
use crate::lsp::{Language, ServerStatus};

use super::super::enrichment::symbol_column_in_line;

#[test]
fn symbol_column_resolves_real_offset() {
    // Regression guard for the column-0 bug: enrichment must query at the
    // symbol's real column, not the start of the line.
    assert_eq!(symbol_column_in_line("pub fn add(a: i32) {", "add"), 7);
    assert_eq!(
        symbol_column_in_line("    def compute(self):", "compute"),
        8
    );
    assert_eq!(symbol_column_in_line("class Widget:", "Widget"), 6);
    // First column when the symbol leads the line.
    assert_eq!(symbol_column_in_line("add()", "add"), 0);
}

#[test]
fn symbol_column_handles_missing_and_empty() {
    // Missing symbol → fall back to column 0.
    assert_eq!(symbol_column_in_line("pub fn other() {}", "add"), 0);
    // Empty symbol name → 0 (no crash).
    assert_eq!(symbol_column_in_line("pub fn add() {}", ""), 0);
}

#[test]
fn symbol_column_uses_utf16_units() {
    // LSP positions are UTF-16 code units. A 4-byte emoji (1 UTF-16 surrogate
    // pair = 2 code units) before the symbol must count as 2, not 4 (bytes)
    // and not 1 (char).
    let line = "// 😀 fn add";
    let col = symbol_column_in_line(line, "add");
    // "// " = 3, emoji = 2 (UTF-16), " fn " = 4 → 9
    assert_eq!(col, 9);
}

#[tokio::test]
async fn test_enrich_chunk_increments_metrics() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let _enrichment = manager
        .enrich_chunk(
            "test-project",
            Path::new("/test/file.rs"),
            "test_function",
            10,
            20,
            false,
        )
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
        .enrich_chunk(
            "test-project",
            Path::new("/test/file.rs"),
            "test_symbol",
            10,
            20,
            false,
        )
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
        .enrich_chunk(
            "test-project",
            Path::new("/test/file.rs"),
            "test_symbol",
            10,
            20,
            true,
        )
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
        .enrich_chunk(
            "test-project",
            Path::new("/test/file.rs"),
            "test_symbol",
            10,
            20,
            true,
        )
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

    let restored = manager
        .restore_project_servers("test-project")
        .await
        .unwrap();
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
                last_enrichment_at: Some(std::time::Instant::now()),
                marked_unavailable: false,
            },
        );
    }

    let result = manager
        .handle_potential_crash(&key, "simulated error")
        .await;
    assert!(
        !result,
        "Should return false when no instance exists to check"
    );
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
        last_enrichment_at: Some(std::time::Instant::now()),
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

#[tokio::test]
async fn test_evict_idle_servers_no_servers() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let evicted = manager
        .evict_idle_servers(std::time::Duration::from_secs(60))
        .await;
    assert!(evicted.is_empty());
}

#[tokio::test]
async fn test_evict_idle_servers_evicts_stale() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    // Insert a server state with an old enrichment timestamp
    let key = ProjectLanguageKey::new("test-project", Language::Rust);
    {
        let mut servers = manager.servers.write().await;
        servers.insert(
            key.clone(),
            ProjectServerState {
                project_id: "test-project".to_string(),
                language: Language::Rust,
                project_root: PathBuf::from("/test"),
                status: ServerStatus::Running,
                restart_count: 0,
                last_error: None,
                is_active: true,
                last_healthy_time: Some(Utc::now()),
                // Simulate idle: set to a past time
                last_enrichment_at: Some(
                    std::time::Instant::now() - std::time::Duration::from_secs(3600),
                ),
                marked_unavailable: false,
            },
        );
    }

    // With a 60s timeout, the 3600s-old server should be evicted
    let evicted = manager
        .evict_idle_servers(std::time::Duration::from_secs(60))
        .await;
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0].0, "test-project");

    // Server state should be removed
    let servers = manager.servers.read().await;
    assert!(!servers.contains_key(&key));
}

#[tokio::test]
async fn test_evict_idle_servers_keeps_recent() {
    let config = ProjectLspConfig::default();
    let manager = LanguageServerManager::new(config).await.unwrap();

    let key = ProjectLanguageKey::new("test-project", Language::Python);
    {
        let mut servers = manager.servers.write().await;
        servers.insert(
            key.clone(),
            ProjectServerState {
                project_id: "test-project".to_string(),
                language: Language::Python,
                project_root: PathBuf::from("/test"),
                status: ServerStatus::Running,
                restart_count: 0,
                last_error: None,
                is_active: true,
                last_healthy_time: Some(Utc::now()),
                last_enrichment_at: Some(std::time::Instant::now()),
                marked_unavailable: false,
            },
        );
    }

    // With a 3600s timeout, a recently-used server should not be evicted
    let evicted = manager
        .evict_idle_servers(std::time::Duration::from_secs(3600))
        .await;
    assert!(evicted.is_empty());

    // Server state should still be present
    let servers = manager.servers.read().await;
    assert!(servers.contains_key(&key));
}

#[tokio::test]
async fn test_warmup_grace_defers_then_allows_readiness() {
    use crate::lsp::detection::{DetectedServer, ServerCapabilities};
    use crate::lsp::{LspConfig, ServerInstance};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let manager = LanguageServerManager::new(ProjectLspConfig::default())
        .await
        .unwrap();
    let project = "warmup-proj";
    let file = Path::new("/test/lib.rs"); // → Language::Rust
    let key = ProjectLanguageKey::new(project, Language::Rust);

    // Register a process-less server instance so the existence check passes.
    let detected = DetectedServer {
        name: "rust-analyzer".to_string(),
        path: PathBuf::from("/usr/bin/rust-analyzer"),
        languages: vec![Language::Rust],
        version: None,
        capabilities: ServerCapabilities::default(),
        priority: 1,
    };
    let instance = ServerInstance::new(detected, LspConfig::default())
        .await
        .unwrap();
    manager
        .instances
        .write()
        .await
        .insert(key.clone(), Arc::new(tokio::sync::Mutex::new(instance)));

    // Still inside the warm-up window → enrichment is deferred (not ready).
    manager
        .ready_at
        .write()
        .await
        .insert(key.clone(), Instant::now() + Duration::from_secs(60));
    assert!(
        !manager.is_server_ready_for_file(project, file).await,
        "server should not be ready during warm-up grace"
    );

    // Warm-up elapsed → ready to enrich.
    manager
        .ready_at
        .write()
        .await
        .insert(key.clone(), Instant::now() - Duration::from_secs(1));
    assert!(
        manager.is_server_ready_for_file(project, file).await,
        "server should be ready after warm-up grace elapses"
    );
}

#[tokio::test]
async fn test_ready_signal_promotes_before_grace() {
    use crate::lsp::detection::{DetectedServer, ServerCapabilities};
    use crate::lsp::{LspConfig, ServerInstance};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let manager = LanguageServerManager::new(ProjectLspConfig::default())
        .await
        .unwrap();
    let project = "signal-proj";
    let file = Path::new("/test/lib.rs"); // → Language::Rust
    let key = ProjectLanguageKey::new(project, Language::Rust);

    let detected = DetectedServer {
        name: "rust-analyzer".to_string(),
        path: PathBuf::from("/usr/bin/rust-analyzer"),
        languages: vec![Language::Rust],
        version: None,
        capabilities: ServerCapabilities::default(),
        priority: 1,
    };
    let instance = ServerInstance::new(detected, LspConfig::default())
        .await
        .unwrap();
    manager
        .instances
        .write()
        .await
        .insert(key.clone(), Arc::new(tokio::sync::Mutex::new(instance)));

    // Warm-up far in the FUTURE (grace would say "not ready")…
    manager
        .ready_at
        .write()
        .await
        .insert(key.clone(), Instant::now() + Duration::from_secs(600));
    // …but the server signalled indexing-done → ready now.
    let sig = Arc::new(AtomicBool::new(true));
    manager
        .ready_signals
        .write()
        .await
        .insert(key.clone(), sig.clone());
    assert!(
        manager.is_server_ready_for_file(project, file).await,
        "ready signal should promote readiness ahead of the warm-up grace"
    );

    // Clearing the signal falls back to the (future) grace → not ready.
    sig.store(false, Ordering::Relaxed);
    assert!(
        !manager.is_server_ready_for_file(project, file).await,
        "without the signal, the future grace should keep it not-ready"
    );
}
