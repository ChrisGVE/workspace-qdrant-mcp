//! Main processing loop for the unified queue processor.

use chrono::{Duration as ChronoDuration, Utc};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::config::IngestionLimitsConfig;
use crate::fairness_scheduler::FairnessScheduler;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::queue_health::QueueProcessorHealth;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tree_sitter::GrammarManager;
use crate::{DocumentProcessor, EmbeddingGenerator};

use crate::unified_queue_processor::config::{
    UnifiedProcessingMetrics, UnifiedProcessorConfig, WarmupState,
};
use crate::unified_queue_processor::error::UnifiedProcessorResult;
use crate::unified_queue_processor::UnifiedQueueProcessor;

use super::batch_processing::{process_batch, update_tenant_activity};
use super::idle_work::run_idle_work;
use super::loop_state::LoopState;

impl UnifiedQueueProcessor {
    /// Main processing loop (runs in background task)
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn processing_loop(
        queue_manager: QueueManager,
        config: UnifiedProcessorConfig,
        fairness_scheduler: Arc<FairnessScheduler>,
        metrics: Arc<RwLock<UnifiedProcessingMetrics>>,
        cancellation_token: CancellationToken,
        document_processor: Arc<DocumentProcessor>,
        embedding_generator: Arc<EmbeddingGenerator>,
        storage_client: Arc<StorageClient>,
        lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: Arc<tokio::sync::Semaphore>,
        allowed_extensions: Arc<AllowedExtensions>,
        lexicon_manager: Arc<LexiconManager>,
        warmup_state: Arc<WarmupState>,
        queue_health: Option<Arc<QueueProcessorHealth>>,
        resource_profile_rx: Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        queue_depth_counter: Arc<std::sync::atomic::AtomicUsize>,
        search_db: Option<Arc<SearchDbManager>>,
        graph_store: Option<crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>>,
        watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,
        grammar_manager: Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: Arc<IngestionLimitsConfig>,
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mut resource_profile_rx = resource_profile_rx;
        let metrics_log_interval = ChronoDuration::minutes(1);

        let mut state = LoopState::new(&config);

        info!(
            "Unified processing loop started (batch_size={}, worker_id={}, fairness={}, \
             warmup_window={}s, maintenance_tasks={})",
            config.batch_size,
            config.worker_id,
            config.fairness_enabled,
            config.warmup_window_secs,
            state.maintenance_scheduler.task_count()
        );

        loop {
            // Warmup transition: add semaphore permits, log once (Task 577, Task 578)
            if !state.warmup_logged && !warmup_state.is_in_warmup() {
                let permits_to_add = config
                    .max_concurrent_embeddings
                    .saturating_sub(config.warmup_max_concurrent_embeddings);
                if permits_to_add > 0 {
                    embedding_semaphore.add_permits(permits_to_add);
                }
                info!(
                    "Warmup period complete after {}s - switching to normal resource limits \
                     (delay: {}ms -> {}ms, max_embeddings: {} -> {})",
                    warmup_state.elapsed_secs(),
                    config.warmup_inter_item_delay_ms,
                    config.inter_item_delay_ms,
                    config.warmup_max_concurrent_embeddings,
                    config.max_concurrent_embeddings
                );
                state.warmup_logged = true;
            }

            if cancellation_token.is_cancelled() {
                info!("Unified queue shutdown signal received");
                break;
            }

            // Adaptive semaphore scaling on resource-profile change
            if let Some(ref mut rx) = resource_profile_rx {
                if rx.has_changed().unwrap_or(false) {
                    let profile = *rx.borrow_and_update();
                    let target = profile.max_concurrent_embeddings;
                    if state.warmup_logged {
                        let current = state
                            .adaptive_target_permits
                            .unwrap_or(config.max_concurrent_embeddings);
                        if target > current {
                            embedding_semaphore.add_permits(target - current);
                        } else if target < current {
                            embedding_semaphore.forget_permits(current - target);
                        }
                        state.adaptive_target_permits = Some(target);
                    }
                }
            }

            if let Some(ref h) = queue_health {
                h.record_poll();
            }

            // Memory pressure gate (Task 504)
            if Self::check_memory_pressure(config.max_memory_percent).await {
                let rss = Self::current_rss_mb();
                if Self::check_process_rss() {
                    warn!(
                        "Process RSS {}MB exceeds {}MB limit, pausing processing for 10s",
                        rss,
                        Self::DEFAULT_MAX_RSS_MB
                    );
                    tokio::time::sleep(Duration::from_secs(10)).await;
                } else {
                    info!(
                        "System memory pressure detected (<{}% available, RSS={}MB), pausing for 5s",
                        100u8.saturating_sub(config.max_memory_percent),
                        rss
                    );
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
                continue;
            }

            // Qdrant circuit breaker: probe before dequeuing when open
            if !storage_client.is_qdrant_available() {
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
                    }
                    _ => {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                }
            }

            // Update queue depth metrics
            if let Ok(depth) = queue_manager.get_unified_queue_depth(None, None).await {
                let mut m = metrics.write().await;
                m.queue_depth = depth;
                if let Some(ref h) = queue_health {
                    h.set_queue_depth(depth as u64);
                }
                queue_depth_counter.store(depth as usize, std::sync::atomic::Ordering::Relaxed);
            }

            // Update oldest pending age Prometheus gauge (Task 7)
            if let Ok(age) = queue_manager.get_oldest_pending_age_seconds().await {
                crate::monitoring::METRICS
                    .queue_oldest_pending_age_seconds
                    .set(age);
            }

            // Dequeue batch (Task 34)
            match fairness_scheduler
                .dequeue_next_batch(config.batch_size)
                .await
            {
                Ok(items) => {
                    if items.is_empty() {
                        run_idle_work(
                            &mut state,
                            &config,
                            &queue_manager,
                            &storage_client,
                            &lexicon_manager,
                            &grammar_manager,
                            &lsp_manager,
                            &search_db,
                            poll_interval,
                        )
                        .await;
                        continue;
                    }

                    // Non-empty batch: cancel maintenance, reset idle tracker
                    state.maintenance_scheduler.cancel_active();
                    state.idle_since = None;

                    info!(
                        "Dequeued {} unified queue items for processing",
                        items.len()
                    );

                    if cancellation_token.is_cancelled() {
                        warn!("Shutdown requested, stopping unified batch processing");
                        return Ok(());
                    }

                    match process_batch(
                        items,
                        &config,
                        &queue_manager,
                        &document_processor,
                        &embedding_generator,
                        &storage_client,
                        &lsp_manager,
                        &embedding_semaphore,
                        &allowed_extensions,
                        &lexicon_manager,
                        &search_db,
                        &graph_store,
                        &watch_refresh_signal,
                        &grammar_manager,
                        &ingestion_limits,
                        &metrics,
                        &queue_health,
                        &cancellation_token,
                        &resource_profile_rx,
                        &warmup_state,
                    )
                    .await
                    {
                        Err(()) => return Ok(()),
                        Ok(processed_tenants) => {
                            update_tenant_activity(&processed_tenants, &queue_manager).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to dequeue unified batch: {}", e);
                    if let Some(ref h) = queue_health {
                        h.record_error();
                    }
                    tokio::time::sleep(poll_interval).await;
                }
            }

            // Periodic metrics log
            let now = Utc::now();
            if now - state.last_metrics_log >= metrics_log_interval {
                Self::log_metrics(&metrics).await;
                state.last_metrics_log = now;
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }
}
