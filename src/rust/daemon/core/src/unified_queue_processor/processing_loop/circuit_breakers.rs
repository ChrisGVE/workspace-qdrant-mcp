//! Circuit breaker checks in the processing loop.

use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;
use crate::unified_queue_processor::config::UnifiedProcessorConfig;
use crate::unified_queue_processor::UnifiedQueueProcessor;

impl UnifiedQueueProcessor {
    /// Probe Qdrant when circuit breaker is open; return `true` (→ `continue`) if still down.
    pub(super) async fn handle_qdrant_circuit_breaker(
        config: &UnifiedProcessorConfig,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
    ) -> bool {
        if storage_client.is_qdrant_available() {
            return false;
        }
        match storage_client.test_connection().await {
            Ok(true) => {
                storage_client.circuit_breaker().record_success();
                info!("Qdrant recovered — resuming queue processing");
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
}
