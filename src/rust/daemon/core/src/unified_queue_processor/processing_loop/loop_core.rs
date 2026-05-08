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
            Self::apply_warmup_transition(&config, &warmup_state, &embedding_semaphore, &mut state);

            if cancellation_token.is_cancelled() {
                info!("Unified queue shutdown signal received");
                break;
            }

            Self::apply_adaptive_scaling(
                &config,
                &mut resource_profile_rx,
                &embedding_semaphore,
                &mut state,
            );

            if let Some(ref h) = queue_health {
                h.record_poll();
            }

            if Self::handle_memory_pressure(&config, poll_interval).await {
                continue;
            }

            if Self::handle_circuit_breaker(&config, &queue_manager, &storage_client).await {
                continue;
            }

            Self::update_queue_depth_metrics(
                &queue_manager,
                &metrics,
                &queue_health,
                &queue_depth_counter,
            )
            .await;

            if let Ok(age) = queue_manager.get_oldest_pending_age_seconds().await {
                crate::monitoring::METRICS
                    .queue_oldest_pending_age_seconds
                    .set(age);
            }

            match fairness_scheduler
                .dequeue_next_batch(config.batch_size)
                .await
            {
                Ok(items) if items.is_empty() => {
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
                Ok(items) => {
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
                        Ok(tenants) => update_tenant_activity(&tenants, &queue_manager).await,
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

            let now = Utc::now();
            if now - state.last_metrics_log >= metrics_log_interval {
                Self::log_metrics(&metrics).await;
                state.last_metrics_log = now;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }

    /// Apply the one-time warmup-complete transition (add semaphore permits, log, mark done).
    fn apply_warmup_transition(
        config: &UnifiedProcessorConfig,
        warmup_state: &Arc<WarmupState>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        state: &mut LoopState,
    ) {
        if state.warmup_logged || warmup_state.is_in_warmup() {
            return;
        }
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

    /// Scale the embedding semaphore when the resource profile changes.
    fn apply_adaptive_scaling(
        config: &UnifiedProcessorConfig,
        resource_profile_rx: &mut Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        state: &mut LoopState,
    ) {
        let Some(ref mut rx) = resource_profile_rx else {
            return;
        };
        if !rx.has_changed().unwrap_or(false) {
            return;
        }
        let profile = *rx.borrow_and_update();
        let target = profile.max_concurrent_embeddings;
        if !state.warmup_logged {
            return;
        }
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

    /// Check memory pressure; sleep and return `true` (→ `continue`) if over limit.
    async fn handle_memory_pressure(
        config: &UnifiedProcessorConfig,
        _poll_interval: Duration,
    ) -> bool {
        if !Self::check_memory_pressure(config.max_memory_percent).await {
            return false;
        }
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
        true
    }

    /// Probe Qdrant when circuit breaker is open; return `true` (→ `continue`) if still down.
    async fn handle_circuit_breaker(
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

    /// Sync queue depth into metrics, health, and the adaptive-resource counter.
    async fn update_queue_depth_metrics(
        queue_manager: &QueueManager,
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        queue_health: &Option<Arc<QueueProcessorHealth>>,
        queue_depth_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) {
        if let Ok(depth) = queue_manager.get_unified_queue_depth(None, None).await {
            let mut m = metrics.write().await;
            m.queue_depth = depth;
            if let Some(ref h) = queue_health {
                h.set_queue_depth(depth as u64);
            }
            queue_depth_counter.store(depth as usize, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
