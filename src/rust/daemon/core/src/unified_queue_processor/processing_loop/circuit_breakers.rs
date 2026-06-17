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
        crate::monitoring::METRICS
            .circuit_breaker_pauses_total
            .with_label_values(&["qdrant"])
            .inc();
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
                // Qdrant is still down: this poll dispatched no batch. Mark it so
                // the next poll's health probe emits a zero throughput sample when
                // a backlog persists, letting the drain ETA decay instead of
                // freezing at a stale healthy rate during the outage (#144).
                state.last_poll_dispatched = false;
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
        crate::monitoring::METRICS
            .circuit_breaker_pauses_total
            .with_label_values(&["sqlite"])
            .inc();

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
                // SQLite is still down: this poll dispatched no batch. Mark it so
                // the next poll's health probe zeroes throughput while a backlog
                // remains, instead of reporting a falsely-Green drain ETA (#144).
                state.last_poll_dispatched = false;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;
    use sqlx::SqlitePool;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        pool
    }

    /// T5 (#144): when the SQLite breaker is open and the probe fails, the handler
    /// must leave `last_poll_dispatched == false` so the next health probe emits a
    /// zero throughput sample instead of holding a stale healthy rate.
    #[tokio::test]
    async fn sqlite_breaker_open_path_clears_last_poll_dispatched() {
        let pool = setup_pool().await;
        let qm = QueueManager::new(pool.clone());
        let config = UnifiedProcessorConfig::default();
        let baseline_ttl_secs = crate::config::queue_health::default_baseline_ttl_secs();
        let mut state = LoopState::new(&config, baseline_ttl_secs);

        // Force the breaker open so the handler takes the probe-and-stay-paused
        // path, and pretend the prior poll dispatched a batch (the #144 trap).
        for _ in 0..config.sqlite_failure_threshold {
            state.sqlite_breaker.record_failure();
        }
        assert!(!state.sqlite_breaker.is_closed());
        state.last_poll_dispatched = true;

        // Close the pool so the `SELECT 1` probe fails and we exercise the
        // still-down branch.
        pool.close().await;

        let paused =
            UnifiedQueueProcessor::handle_sqlite_circuit_breaker(&config, &qm, &mut state).await;

        assert!(paused, "probe failure must keep the loop paused");
        assert!(
            !state.last_poll_dispatched,
            "breaker-open probe-fail path must clear last_poll_dispatched (#144)"
        );
    }
}
