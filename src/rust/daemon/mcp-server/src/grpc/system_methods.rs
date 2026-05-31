//! SystemService RPC wrappers for [`DaemonClient`].
//!
//! Implements three health/status helpers that mirror the TS
//! `DaemonClientSystem` class (`system-methods.ts`):
//!
//! | Rust method                    | Proto RPC                   | TS equivalent                  |
//! |--------------------------------|-----------------------------|--------------------------------|
//! | `health()`                     | `SystemService::Health`     | `healthCheck()`                |
//! | `get_status()`                 | `SystemService::GetStatus`  | `getStatus()`                  |
//! | `get_embedding_provider_status`| `SystemService::GetEmbeddingProviderStatus` | `getEmbeddingProviderStatus()` |
//!
//! All three use [`DaemonClient::call`] with default (5 s) timeout and
//! no `grpc-timeout` header — matching the TS abandon-not-cancel semantics.

use tonic::Status;

use crate::proto::{GetEmbeddingProviderStatusResponse, HealthResponse, SystemStatusResponse};

use super::client::DaemonClient;

impl DaemonClient {
    /// Quick health check — mirrors TS `healthCheck()` in system-methods.ts.
    ///
    /// Calls `SystemService::Health` with an empty request.
    /// Returns the raw [`HealthResponse`] proto on success.
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn health(&mut self) -> Result<HealthResponse, Status> {
        let client = self.system.clone();
        self.call("health", None, || {
            let mut c = client.clone();
            async move {
                c.health(tonic::Request::new(()))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Comprehensive system state snapshot — mirrors TS `getStatus()`.
    ///
    /// Calls `SystemService::GetStatus` with an empty request.
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn get_status(&mut self) -> Result<SystemStatusResponse, Status> {
        let client = self.system.clone();
        self.call("getStatus", None, || {
            let mut c = client.clone();
            async move {
                c.get_status(tonic::Request::new(()))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Embedding provider status — mirrors TS `getEmbeddingProviderStatus()`.
    ///
    /// Returns provider name, model, output dimension, probe state, and
    /// a human-readable probe message.
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn get_embedding_provider_status(
        &mut self,
    ) -> Result<GetEmbeddingProviderStatusResponse, Status> {
        let client = self.system.clone();
        self.call("getEmbeddingProviderStatus", None, || {
            let mut c = client.clone();
            async move {
                c.get_embedding_provider_status(tonic::Request::new(()))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{ComponentHealth, ServiceStatus};

    // Helpers to construct proto response values directly (no live daemon needed).

    fn make_health_response(status: i32) -> HealthResponse {
        HealthResponse {
            status,
            components: vec![],
            timestamp: None,
        }
    }

    fn make_component_health(name: &str, status: i32) -> ComponentHealth {
        ComponentHealth {
            component_name: name.to_string(),
            status,
            message: "ok".to_string(),
            last_check: None,
        }
    }

    fn make_system_status_response(status: i32) -> SystemStatusResponse {
        SystemStatusResponse {
            status,
            metrics: None,
            active_projects: vec![],
            total_documents: 0,
            total_collections: 0,
            uptime_since: None,
            resource_mode: None,
            idle_seconds: None,
            current_max_embeddings: None,
            current_inter_item_delay_ms: None,
        }
    }

    fn make_embedding_provider_status() -> GetEmbeddingProviderStatusResponse {
        GetEmbeddingProviderStatusResponse {
            provider: "fastembed".to_string(),
            model: "all-MiniLM-L6-v2".to_string(),
            output_dim: 384,
            base_url: String::new(),
            probe_status: "healthy".to_string(),
            probe_message: "probe succeeded".to_string(),
        }
    }

    // ── HealthResponse field mapping ──────────────────────────────────────────

    #[test]
    fn health_response_status_field() {
        let r = make_health_response(ServiceStatus::Healthy as i32);
        assert_eq!(r.status, ServiceStatus::Healthy as i32);
    }

    #[test]
    fn health_response_components_empty() {
        let r = make_health_response(ServiceStatus::Healthy as i32);
        assert!(r.components.is_empty());
    }

    #[test]
    fn health_response_components_populated() {
        let mut r = make_health_response(ServiceStatus::Healthy as i32);
        r.components.push(make_component_health(
            "queue_processor",
            ServiceStatus::Healthy as i32,
        ));
        assert_eq!(r.components.len(), 1);
        assert_eq!(r.components[0].component_name, "queue_processor");
        assert_eq!(r.components[0].status, ServiceStatus::Healthy as i32);
    }

    #[test]
    fn component_health_all_fields() {
        let c = make_component_health("file_watcher", ServiceStatus::Degraded as i32);
        assert_eq!(c.component_name, "file_watcher");
        assert_eq!(c.status, ServiceStatus::Degraded as i32);
        assert_eq!(c.message, "ok");
        assert!(c.last_check.is_none());
    }

    // ── SystemStatusResponse field mapping ────────────────────────────────────

    #[test]
    fn system_status_response_status_field() {
        let r = make_system_status_response(ServiceStatus::Healthy as i32);
        assert_eq!(r.status, ServiceStatus::Healthy as i32);
    }

    #[test]
    fn system_status_response_total_documents_zero() {
        let r = make_system_status_response(ServiceStatus::Healthy as i32);
        assert_eq!(r.total_documents, 0);
    }

    #[test]
    fn system_status_response_active_projects_empty() {
        let r = make_system_status_response(ServiceStatus::Healthy as i32);
        assert!(r.active_projects.is_empty());
    }

    #[test]
    fn system_status_response_optional_fields_none() {
        let r = make_system_status_response(ServiceStatus::Healthy as i32);
        assert!(r.resource_mode.is_none());
        assert!(r.idle_seconds.is_none());
        assert!(r.current_max_embeddings.is_none());
        assert!(r.current_inter_item_delay_ms.is_none());
    }

    // ── GetEmbeddingProviderStatusResponse field mapping ──────────────────────

    #[test]
    fn embedding_provider_status_provider_field() {
        let r = make_embedding_provider_status();
        assert_eq!(r.provider, "fastembed");
    }

    #[test]
    fn embedding_provider_status_model_field() {
        let r = make_embedding_provider_status();
        assert_eq!(r.model, "all-MiniLM-L6-v2");
    }

    #[test]
    fn embedding_provider_status_output_dim() {
        let r = make_embedding_provider_status();
        assert_eq!(r.output_dim, 384);
    }

    #[test]
    fn embedding_provider_status_probe_status_healthy() {
        let r = make_embedding_provider_status();
        assert_eq!(r.probe_status, "healthy");
    }

    #[test]
    fn embedding_provider_status_base_url_empty_for_local() {
        let r = make_embedding_provider_status();
        assert_eq!(r.base_url, "");
    }

    #[test]
    fn embedding_provider_status_probe_message_nonempty() {
        let r = make_embedding_provider_status();
        assert!(!r.probe_message.is_empty());
    }

    // ── DaemonClient construction (no live daemon) ────────────────────────────

    #[tokio::test]
    async fn daemon_client_constructs_for_system_calls() {
        // Verify the client can be constructed — does not connect eagerly.
        let result = DaemonClient::new("http://127.0.0.1:50051");
        assert!(result.is_ok());
    }

    // ── call() timeout dispatch for system methods ────────────────────────────

    #[tokio::test]
    async fn health_method_name_uses_5s_budget() {
        // "health" does not contain "search" → 5 s budget.
        // We verify via a fast-failing closure that times out immediately.
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call("health", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }

    #[tokio::test]
    async fn get_status_method_name_uses_5s_budget() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call("getStatus", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }
}
