//! LSP server lifecycle management for projects
//!
//! Handles starting, stopping, and deferred shutdown of per-project LSP servers.
//! The deferred shutdown monitor runs as a background task, checking queue depth
//! before executing shutdowns.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sqlx::SqlitePool;
use tokio::sync::RwLock;
use tonic::Status;
use tracing::{debug, info, warn};

use workspace_qdrant_core::{Language, LanguageServerManager, ProjectLanguageDetector};

/// All supported languages for LSP server lifecycle
fn all_lsp_languages() -> Vec<Language> {
    vec![
        Language::Python,
        Language::Rust,
        Language::TypeScript,
        Language::JavaScript,
        Language::Go,
        Language::C,
        Language::Cpp,
        Language::Java,
    ]
}

/// Start LSP servers for a project's detected languages
///
/// Uses `ProjectLanguageDetector` which:
/// 1. Checks for marker files (Cargo.toml, package.json, etc.)
/// 2. Falls back to extension scanning
/// 3. Caches results per `project_id`
pub async fn start_project_lsp_servers(
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    language_detector: &ProjectLanguageDetector,
    project_id: &str,
    project_root: &PathBuf,
) -> Result<usize, Status> {
    let Some(lsp_manager) = lsp_manager else {
        debug!("No LSP manager configured, skipping LSP server startup");
        return Ok(0);
    };

    let detection_result = language_detector
        .detect(project_id, project_root)
        .await
        .map_err(|e| {
            warn!(
                project_id = project_id,
                error = %e,
                "Failed to detect project languages"
            );
            Status::internal(format!("Language detection failed: {e}"))
        })?;

    let languages = detection_result.all_languages;

    if languages.is_empty() {
        debug!(
            project_id = project_id,
            "No supported languages detected in project"
        );
        return Ok(0);
    }

    info!(
        project_id = project_id,
        marker_languages = ?detection_result.marker_languages,
        extension_languages = ?detection_result.extension_languages,
        from_cache = detection_result.from_cache,
        "Detected project languages"
    );

    let manager = lsp_manager.read().await;
    let mut started = 0;

    for language in languages {
        match manager
            .start_server(project_id, language.clone(), project_root)
            .await
        {
            Ok(_) => {
                info!(
                    project_id = project_id,
                    language = ?language,
                    "Started LSP server"
                );
                started += 1;
            }
            Err(e) => {
                debug!(
                    project_id = project_id,
                    language = ?language,
                    error = %e,
                    "Failed to start LSP server (non-critical)"
                );
            }
        }
    }

    info!(
        project_id = project_id,
        servers_started = started,
        "LSP server startup complete"
    );

    Ok(started)
}

/// Stop all LSP servers for a project
pub async fn stop_project_lsp_servers(
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    language_detector: &ProjectLanguageDetector,
    project_id: &str,
) -> Result<(), Status> {
    language_detector.invalidate_cache(project_id).await;

    let Some(lsp_manager) = lsp_manager else {
        return Ok(());
    };

    let manager = lsp_manager.read().await;

    for language in all_lsp_languages() {
        if let Err(e) = manager.stop_server(project_id, language.clone()).await {
            debug!(
                project_id = project_id,
                language = ?language,
                error = %e,
                "Error stopping LSP server (may not have been running)"
            );
        }
    }

    info!(project_id = project_id, "LSP servers stopped for project");
    Ok(())
}

/// Check pending items in the unified queue for a project
pub async fn get_project_queue_depth(
    db_pool: &SqlitePool,
    project_id: &str,
) -> Result<i64, Status> {
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND tenant_id = ?1",
    )
    .bind(project_id)
    .fetch_one(db_pool)
    .await
    .map_err(|e| {
        if e.to_string().contains("no such table") {
            debug!(
                project_id = project_id,
                "unified_queue table not found, assuming empty"
            );
            return Status::ok("Queue table not initialized");
        }
        tracing::error!(
            project_id = project_id,
            error = %e,
            "Failed to check queue depth"
        );
        Status::internal(format!("Queue check failed: {e}"))
    })?;

    Ok(count)
}

/// Schedule a deferred LSP shutdown for a project
///
/// Called when `is_active` becomes false but:
/// - `deactivation_delay_secs` > 0, OR
/// - queue has pending items
pub async fn schedule_deferred_shutdown(
    pending_shutdowns: &RwLock<HashMap<String, (Instant, bool)>>,
    deactivation_delay_secs: u64,
    project_id: &str,
    has_queue_items: bool,
) {
    let shutdown_time = Instant::now() + Duration::from_secs(deactivation_delay_secs);

    let mut shutdowns = pending_shutdowns.write().await;
    shutdowns.insert(
        project_id.to_string(),
        (shutdown_time, !has_queue_items), // was_queue_checked = true if queue was empty
    );

    info!(
        project_id = project_id,
        delay_secs = deactivation_delay_secs,
        has_queue_items = has_queue_items,
        "Scheduled deferred LSP shutdown"
    );
}

/// Cancel a pending deferred shutdown (e.g., when project reactivates)
pub async fn cancel_deferred_shutdown(
    pending_shutdowns: &RwLock<HashMap<String, (Instant, bool)>>,
    project_id: &str,
) -> bool {
    let mut shutdowns = pending_shutdowns.write().await;
    if shutdowns.remove(project_id).is_some() {
        info!(project_id = project_id, "Cancelled pending LSP shutdown");
        true
    } else {
        false
    }
}

/// Start the background task that monitors and executes deferred LSP shutdowns
///
/// Spawns an async task that:
/// - Runs every 10 seconds
/// - Checks for expired shutdown entries
/// - Verifies queue is empty before executing shutdown
/// - Removes completed/cancelled entries
pub fn start_deferred_shutdown_monitor(
    pending_shutdowns: Arc<RwLock<HashMap<String, (Instant, bool)>>>,
    db_pool: SqlitePool,
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    language_detector: Arc<ProjectLanguageDetector>,
) {
    tokio::spawn(async move {
        info!("Started deferred LSP shutdown monitor (10s interval)");

        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;

            let pending: Vec<(String, Instant, bool)> = {
                let shutdowns = pending_shutdowns.read().await;
                shutdowns
                    .iter()
                    .map(|(k, (time, checked))| (k.clone(), *time, *checked))
                    .collect()
            };

            if pending.is_empty() {
                continue;
            }

            debug!("Checking {} pending shutdowns", pending.len());

            let now = Instant::now();
            for (project_id, scheduled_time, _was_queue_checked) in pending {
                process_pending_shutdown(
                    &project_id,
                    scheduled_time,
                    now,
                    &db_pool,
                    &pending_shutdowns,
                    &lsp_manager,
                    &language_detector,
                )
                .await;
            }
        }
    });
}

/// Evaluate and potentially execute one deferred shutdown entry.
///
/// Skips if the shutdown time has not yet passed or the queue is non-empty.
/// On execution: invalidates language cache and stops all LSP servers.
async fn process_pending_shutdown(
    project_id: &str,
    scheduled_time: Instant,
    now: Instant,
    db_pool: &SqlitePool,
    pending_shutdowns: &RwLock<HashMap<String, (Instant, bool)>>,
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    language_detector: &ProjectLanguageDetector,
) {
    if scheduled_time > now {
        debug!(
            project_id = %project_id,
            remaining_secs = (scheduled_time - now).as_secs(),
            "Shutdown not yet due"
        );
        return;
    }

    let queue_depth: i64 = match sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND tenant_id = ?1",
    )
    .bind(project_id)
    .fetch_one(db_pool)
    .await
    {
        Ok(count) => count,
        Err(e) => {
            if e.to_string().contains("no such table") {
                0
            } else {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to check queue depth, skipping this iteration"
                );
                return;
            }
        }
    };

    if queue_depth > 0 {
        debug!(
            project_id = %project_id,
            pending_items = queue_depth,
            "Queue not empty, deferring shutdown"
        );
        return;
    }

    {
        let mut shutdowns = pending_shutdowns.write().await;
        shutdowns.remove(project_id);
    }

    info!(
        project_id = %project_id,
        "Executing deferred LSP shutdown (queue empty, delay expired)"
    );

    language_detector.invalidate_cache(project_id).await;

    let Some(lsp_mgr) = lsp_manager else {
        return;
    };

    let manager = lsp_mgr.read().await;
    for language in all_lsp_languages() {
        if let Err(e) = manager.stop_server(project_id, language.clone()).await {
            debug!(
                project_id = %project_id,
                language = ?language,
                error = %e,
                "Error stopping LSP server (may not have been running)"
            );
        }
    }

    info!(project_id = %project_id, "Deferred LSP shutdown complete");
}
