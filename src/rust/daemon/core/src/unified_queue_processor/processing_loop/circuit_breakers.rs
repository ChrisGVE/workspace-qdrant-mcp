//! Circuit breaker checks in the processing loop.

use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;
use crate::unified_queue_processor::config::UnifiedProcessorConfig;
use crate::unified_queue_processor::UnifiedQueueProcessor;

use super::loop_state::LoopState;

impl UnifiedQueueProcessor {
    /// Probe Qdrant when circuit breaker is open; return `true` (→ `continue`) if still down.
    pub(super) async fn handle_qdrant_circuit_breaker(
        config: &UnifiedProcessorConfig,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        state: &mut LoopState,
    ) -> bool {
        if storage_client.is_qdrant_available() {
            return false;
        }
        match storage_client.test_connection().await {
            Ok(true) => {
                storage_client.circuit_breaker().record_success();
                info!("Qdrant recovered — resuming queue processing");
                state.recovery_ramp_remaining = config.recovery_ramp_cycles;
                match queue_manager
                    .resurrect_failed_transient(config.max_resurrections)
                    .await
                {
                    Ok((r, x)) if r > 0 || x > 0 => info!(
                        "Recovery resurrection: reset {} item(s), exhausted {} item(s)",
                        r, x
                    ),
                    Ok(_) => {}
                    Err(e) => warn!("Recovery resurrection failed: {}", e),
                }
                false
            }
            _ => {
                tokio::time::sleep(Duration::from_secs(5)).await;
                true
            }
        }
    }

    /// Check SQLite circuit breaker; return `true` (→ `continue`) if SQLite is down.
    pub(super) async fn handle_sqlite_circuit_breaker(
        config: &UnifiedProcessorConfig,
        queue_manager: &QueueManager,
        state: &mut LoopState,
    ) -> bool {
        let (can_proceed, _) = state.sqlite_breaker.check();
        if can_proceed {
            return false;
        }

        let probe_secs = config.sqlite_probe_interval_secs.max(1);
        tokio::time::sleep(Duration::from_secs(probe_secs)).await;

        match sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(queue_manager.pool())
            .await
        {
            Ok(_) => {
                state.sqlite_breaker.record_success();
                info!("SQLite recovered — resuming queue processing");
                false
            }
            Err(e) => {
                state.sqlite_breaker.record_failure();
                warn!("SQLite probe failed, staying paused: {}", e);
                true
            }
        }
    }
}

pub(super) fn new_sqlite_breaker(config: &UnifiedProcessorConfig) -> CircuitBreaker {
    CircuitBreaker::new(
        "sqlite-queue",
        CircuitBreakerConfig {
            failure_threshold: config.sqlite_failure_threshold,
            failure_window: config.sqlite_failure_window_secs,
            recovery_timeout: config.sqlite_probe_interval_secs,
            success_threshold: 1,
        },
    )
}
