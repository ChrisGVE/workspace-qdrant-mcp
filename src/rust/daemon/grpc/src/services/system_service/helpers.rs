//! Helper methods for SystemServiceImpl
//!
//! Grouped into three areas:
//!   • Embedding-provider probe and health component
//!   • Queue-processor health component and metrics
//!   • Folder scan enqueue and server-notification lifecycle

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use tracing::{debug, error, info, warn};
use workspace_qdrant_core::embedding::EmbeddingError;
use workspace_qdrant_core::lifecycle::WatchFolderLifecycle;
use wqm_common::timestamps;

use crate::proto::{ComponentHealth, Metric, ServerState, ServiceStatus};

use super::service_impl::SystemServiceImpl;

// ─────────────────────────────────────────────────────────────────────────────
// Embedding-provider probe helpers
// ─────────────────────────────────────────────────────────────────────────────

impl SystemServiceImpl {
    /// Run a probe inline with a 3 s timeout and store the outcome in the
    /// shared cache. Used by `GetEmbeddingProviderStatus` so callers see
    /// fresh state on demand.
    ///
    /// Honors `health_probe_cache_secs`: if the cached result is younger
    /// than the TTL it is reused without issuing a network probe.
    ///
    /// Returns `(probe_status, probe_message)`.
    pub(super) async fn probe_embedding_provider(&self) -> (String, String) {
        let provider = match &self.dense_provider {
            Some(p) => Arc::clone(p),
            None => {
                return (
                    "probe_pending".to_string(),
                    "embedding provider not wired".to_string(),
                );
            }
        };

        let ttl = self
            .embedding_settings
            .as_ref()
            .map(|s| s.health_probe_cache_secs)
            .unwrap_or(0);

        if ttl > 0 {
            let cache = self.embedding_probe_cache.lock().await;
            if let (Some(last_at), Some(last_result)) =
                (cache.last_probe_at, cache.last_result.as_ref())
            {
                if last_at.elapsed() < Duration::from_secs(ttl) {
                    return classify_probe_result(last_result);
                }
            }
        }

        let probe_call = tokio::time::timeout(Duration::from_secs(3), provider.probe()).await;
        match probe_call {
            Ok(result) => {
                let pair = classify_probe_result(&result);
                let mut cache = self.embedding_probe_cache.lock().await;
                cache.last_probe_at = Some(Instant::now());
                cache.last_result = Some(result);
                pair
            }
            Err(_elapsed) => {
                let mut cache = self.embedding_probe_cache.lock().await;
                cache.last_probe_at = Some(Instant::now());
                cache.last_result = None;
                ("unhealthy".to_string(), "timeout after 3s".to_string())
            }
        }
    }

    /// Build the `embedding_provider` ComponentHealth for the `Health` RPC
    /// from the cached probe result. The Health endpoint never blocks on a
    /// network probe — fresh probes are issued by `ProviderHealthMonitor`
    /// (background) or by `GetEmbeddingProviderStatus` (on-demand). Until
    /// the cache is warm, the component status is `Degraded` with a
    /// `probe pending` message.
    pub(super) async fn get_embedding_provider_health(&self) -> ComponentHealth {
        let now = SystemTime::now();

        if self.dense_provider.is_none() {
            return ComponentHealth {
                component_name: "embedding_provider".to_string(),
                status: ServiceStatus::Unspecified as i32,
                message: "embedding provider not wired".to_string(),
                last_check: Some(prost_types::Timestamp::from(now)),
            };
        }

        let cache = self.embedding_probe_cache.lock().await;
        match cache.last_result.as_ref() {
            None => ComponentHealth {
                component_name: "embedding_provider".to_string(),
                status: ServiceStatus::Degraded as i32,
                message: "probe pending: background probe not yet completed".to_string(),
                last_check: Some(prost_types::Timestamp::from(now)),
            },
            Some(result) => {
                let (_, probe_message) = classify_probe_result(result);
                let svc_status = match result {
                    Ok(()) => ServiceStatus::Healthy,
                    Err(EmbeddingError::TemporarilyUnavailable { .. }) => ServiceStatus::Degraded,
                    Err(_) => ServiceStatus::Unhealthy,
                };
                ComponentHealth {
                    component_name: "embedding_provider".to_string(),
                    status: svc_status as i32,
                    message: probe_message,
                    last_check: Some(prost_types::Timestamp::from(now)),
                }
            }
        }
    }

    /// Test-only seeding of the embedding probe cache.
    #[cfg(test)]
    pub(super) async fn seed_embedding_probe_cache(&self, result: Result<(), EmbeddingError>) {
        let mut cache = self.embedding_probe_cache.lock().await;
        cache.last_probe_at = Some(Instant::now());
        cache.last_result = Some(result);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Queue-processor health and metrics
// ─────────────────────────────────────────────────────────────────────────────

impl SystemServiceImpl {
    /// Build the `queue_processor` ComponentHealth entry.
    pub(super) fn get_queue_processor_health(&self) -> ComponentHealth {
        if let Some(health) = &self.queue_health {
            let is_running = health.is_running.load(Ordering::SeqCst);
            let secs_since_poll = health.seconds_since_last_poll();
            let secs_since_heartbeat = health.seconds_since_last_heartbeat();
            // Stalled only when both poll AND per-item heartbeat are old.
            let secs_since_activity = secs_since_poll.min(secs_since_heartbeat);

            // NOTE (#131): the health verdict deliberately does NOT consider
            // `error_count`. That field is a lifetime-cumulative counter, so a
            // fixed `> 100` threshold made any long-lived daemon report a
            // permanent "High error count detected" degradation long after the
            // errors stopped mattering — a false positive. `error_count` is
            // retained purely as a metric (see `get_queue_metrics`). A real
            // functional-degradation model (DLQ trend, drain-time budget,
            // ms/KB and embedder-latency regression, component diagnosis) is
            // tracked in #133.
            let (status, message) = if !is_running {
                (ServiceStatus::Unhealthy, "Queue processor is not running")
            } else if secs_since_activity > 60 {
                (
                    ServiceStatus::Degraded,
                    "Queue processor may be stalled (>60s since last activity)",
                )
            } else {
                (ServiceStatus::Healthy, "Running normally")
            };

            ComponentHealth {
                component_name: "queue_processor".to_string(),
                status: status as i32,
                message: message.to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            }
        } else {
            ComponentHealth {
                component_name: "queue_processor".to_string(),
                status: ServiceStatus::Unspecified as i32,
                message: "Health monitoring not connected".to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            }
        }
    }

    /// Build queue-processor metric list.
    pub(super) fn get_queue_metrics(&self) -> Vec<Metric> {
        let now = Some(prost_types::Timestamp::from(SystemTime::now()));

        if let Some(health) = &self.queue_health {
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.queue_depth.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_processed.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_failed.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_errors".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.error_count.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processing_avg_ms".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.avg_processing_time_ms.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processor_running".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: if health.is_running.load(Ordering::SeqCst) {
                        1.0
                    } else {
                        0.0
                    },
                    timestamp: now,
                },
            ]
        } else {
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
            ]
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Folder scan enqueue and server-notification lifecycle
// ─────────────────────────────────────────────────────────────────────────────

impl SystemServiceImpl {
    /// Enqueue scan operations for all enabled watch folders.
    ///
    /// Used by `SendRefreshSignal` to trigger rescans.
    pub(super) async fn enqueue_folder_scans(
        &self,
        pool: &sqlx::SqlitePool,
    ) -> Result<(), tonic::Status> {
        let folders = sqlx::query_as::<_, (String, String, String, String)>(
            "SELECT watch_id, path, collection, tenant_id FROM watch_folders WHERE enabled = 1",
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            error!("Failed to query watch_folders: {}", e);
            tonic::Status::internal(format!("Database error: {}", e))
        })?;

        let mut scans_queued = 0u32;
        let now = timestamps::now_utc();

        for (watch_id, path, collection, tenant_id) in &folders {
            let payload = serde_json::json!({
                "folder_path": path,
                "recursive": true,
                "recursive_depth": 10,
                "patterns": [],
                "ignore_patterns": []
            });
            let payload_json = payload.to_string();

            use sha2::{Digest, Sha256};
            let key_input = format!("folder|scan|{}|{}|{}", tenant_id, collection, payload_json);
            let hash = format!("{:x}", Sha256::digest(key_input.as_bytes()));
            let idempotency_key = &hash[..32];

            let queue_id = uuid::Uuid::new_v4().to_string();
            let result = sqlx::query(
                "INSERT OR IGNORE INTO unified_queue \
                 (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                  status, payload_json, created_at, updated_at) \
                 VALUES (?1, ?2, 'folder', 'scan', ?3, ?4, 'pending', ?5, ?6, ?7)",
            )
            .bind(&queue_id)
            .bind(idempotency_key)
            .bind(tenant_id)
            .bind(collection)
            .bind(&payload_json)
            .bind(&now)
            .bind(&now)
            .execute(pool)
            .await;

            match result {
                Ok(r) if r.rows_affected() > 0 => {
                    scans_queued += 1;
                    debug!(
                        "Queued refresh scan for watch_folder {} (path={})",
                        watch_id, path
                    );
                }
                Ok(_) => {
                    debug!(
                        "Refresh scan already queued for watch_folder {} (deduplicated)",
                        watch_id
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to queue refresh scan for watch_folder {}: {}",
                        watch_id, e
                    );
                }
            }
        }

        info!(
            "Refresh signal processed: {} watch folders found, {} scans queued",
            folders.len(),
            scans_queued
        );

        Ok(())
    }

    /// Handle a server status notification: store the entry, log transitions,
    /// and update watch folder activation via lifecycle manager.
    pub(super) async fn handle_server_notification(
        &self,
        state: ServerState,
        project_name: Option<String>,
        project_root: Option<String>,
    ) {
        let component_key = project_name
            .clone()
            .or_else(|| project_root.clone())
            .unwrap_or_else(|| "unknown".to_string());

        match state {
            ServerState::Up => {
                info!(
                    "Server UP: component={}, project_name={:?}, project_root={:?}",
                    component_key, project_name, project_root
                );
            }
            ServerState::Down => {
                warn!(
                    "Server DOWN: component={}, project_name={:?}, project_root={:?}",
                    component_key, project_name, project_root
                );
            }
            _ => {
                debug!(
                    "Server status unspecified: component={}, state={:?}",
                    component_key, state
                );
            }
        }

        let entry = super::types::ServerStatusEntry {
            state,
            project_name: project_name.clone(),
            project_root: project_root.clone(),
            updated_at: SystemTime::now(),
        };

        let previous_state = {
            let mut store = self.status_store.write().await;
            let prev = store.get(&component_key).map(|e| e.state);
            store.insert(component_key.clone(), entry);
            prev
        };

        if let Some(prev) = previous_state {
            if prev != state {
                info!(
                    "Server state transition: component={}, {:?} -> {:?}",
                    component_key, prev, state
                );
            }
        }

        if let (Some(pool), Some(ref root)) = (&self.db_pool, &project_root) {
            let is_active = matches!(state, ServerState::Up);
            let lifecycle = WatchFolderLifecycle::new(pool.clone());

            match lifecycle.set_active_by_path(root, is_active).await {
                Ok(rows) if rows > 0 => {
                    info!(
                        "Updated watch_folder activation: path={}, is_active={}",
                        root, is_active
                    );
                }
                Ok(_) => {
                    debug!(
                        "No watch_folder found for path={} (may not be registered yet)",
                        root
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to update watch_folder activation for {}: {}",
                        root, e
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Map the raw probe `Result<(), EmbeddingError>` into the
/// `(probe_status, probe_message)` pair surfaced by the
/// `GetEmbeddingProviderStatus` RPC and the `embedding_provider` health
/// component.
pub(super) fn classify_probe_result(result: &Result<(), EmbeddingError>) -> (String, String) {
    match result {
        Ok(()) => ("healthy".to_string(), "Running normally".to_string()),
        Err(EmbeddingError::RemoteError {
            status_code: 401,
            message,
        }) => (
            "unhealthy".to_string(),
            format!("auth failure: 401 {}", message),
        ),
        Err(EmbeddingError::RemoteError {
            status_code: 403,
            message,
        }) => (
            "unhealthy".to_string(),
            format!("auth failure: 403 {}", message),
        ),
        Err(EmbeddingError::TemporarilyUnavailable { retry_after_secs }) => (
            "degraded".to_string(),
            format!("temporarily unavailable, retry in {retry_after_secs}s"),
        ),
        Err(other) => ("unhealthy".to_string(), other.to_string()),
    }
}
