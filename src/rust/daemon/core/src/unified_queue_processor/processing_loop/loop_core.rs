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
        embedding_health: Option<crate::embedding::EmbeddingHealth>,
        resource_profile_rx: Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        queue_depth_counter: Arc<std::sync::atomic::AtomicUsize>,
        search_db: Option<Arc<SearchDbManager>>,
        graph_store: Option<Arc<dyn crate::graph::GraphStore>>,
        watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,
        grammar_manager: Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: Arc<IngestionLimitsConfig>,
        keyword_embedding_generator: Option<Arc<EmbeddingGenerator>>,
        tier2_tagger: Option<Arc<crate::tagging::Tier2Tagger>>,
        concept_config: Arc<crate::config::ConceptConfig>,
        narrative_config: Arc<crate::config::NarrativeConfig>,
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
            if Self::run_poll_cycle(
                &config,
                &cancellation_token,
                &fairness_scheduler,
                &queue_manager,
                &storage_client,
                &document_processor,
                &embedding_generator,
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
                &embedding_health,
                &queue_depth_counter,
                &warmup_state,
                &mut resource_profile_rx,
                &mut state,
                poll_interval,
                metrics_log_interval,
                &keyword_embedding_generator,
                &tier2_tagger,
                &concept_config,
                &narrative_config,
            )
            .await
            {
                break;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_poll_cycle(
        config: &UnifiedProcessorConfig,
        cancellation_token: &CancellationToken,
        fairness_scheduler: &Arc<FairnessScheduler>,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
        graph_store: &Option<Arc<dyn crate::graph::GraphStore>>,
        watch_refresh_signal: &Option<Arc<tokio::sync::Notify>>,
        grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: &Arc<IngestionLimitsConfig>,
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        queue_health: &Option<Arc<QueueProcessorHealth>>,
        embedding_health: &Option<crate::embedding::EmbeddingHealth>,
        queue_depth_counter: &Arc<std::sync::atomic::AtomicUsize>,
        warmup_state: &Arc<WarmupState>,
        resource_profile_rx: &mut Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        state: &mut LoopState,
        poll_interval: Duration,
        metrics_log_interval: ChronoDuration,
        keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
        tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
        concept_config: &Arc<crate::config::ConceptConfig>,
        narrative_config: &Arc<crate::config::NarrativeConfig>,
    ) -> bool {
        Self::apply_warmup_transition(config, warmup_state, embedding_semaphore, state);
        if cancellation_token.is_cancelled() {
            info!("Unified queue shutdown signal received");
            return true;
        }
        Self::apply_adaptive_scaling(config, resource_profile_rx, embedding_semaphore, state);
        if let Some(ref h) = queue_health {
            h.record_poll();
        }
        if Self::handle_memory_pressure(config, poll_interval).await {
            return false;
        }
        if Self::handle_qdrant_circuit_breaker(config, queue_manager, storage_client, state).await {
            return false;
        }
        if Self::handle_sqlite_circuit_breaker(config, queue_manager, state).await {
            return false;
        }
        Self::update_queue_depth_metrics(queue_manager, metrics, queue_health, queue_depth_counter)
            .await;
        Self::record_oldest_pending_age(queue_manager).await;
        if !Self::run_loop_iteration(
            fairness_scheduler,
            config,
            queue_manager,
            cancellation_token,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager,
            embedding_semaphore,
            allowed_extensions,
            lexicon_manager,
            search_db,
            graph_store,
            watch_refresh_signal,
            grammar_manager,
            ingestion_limits,
            metrics,
            queue_health,
            embedding_health,
            resource_profile_rx,
            warmup_state,
            state,
            poll_interval,
            metrics_log_interval,
            keyword_embedding_generator,
            tier2_tagger,
            concept_config,
            narrative_config,
        )
        .await
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }

    /// Execute one dequeue-dispatch-sleep cycle, returning `true` if the caller should `continue`.
    #[allow(clippy::too_many_arguments)]
    async fn run_loop_iteration(
        fairness_scheduler: &Arc<FairnessScheduler>,
        config: &UnifiedProcessorConfig,
        queue_manager: &QueueManager,
        cancellation_token: &CancellationToken,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
        graph_store: &Option<Arc<dyn crate::graph::GraphStore>>,
        watch_refresh_signal: &Option<Arc<tokio::sync::Notify>>,
        grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: &Arc<IngestionLimitsConfig>,
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        queue_health: &Option<Arc<QueueProcessorHealth>>,
        embedding_health: &Option<crate::embedding::EmbeddingHealth>,
        resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        warmup_state: &Arc<WarmupState>,
        state: &mut LoopState,
        poll_interval: Duration,
        metrics_log_interval: ChronoDuration,
        keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
        tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
        concept_config: &Arc<crate::config::ConceptConfig>,
        narrative_config: &Arc<crate::config::NarrativeConfig>,
    ) -> bool {
        let should_continue = Self::handle_dequeue_result(
            fairness_scheduler,
            config,
            queue_manager,
            cancellation_token,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager,
            embedding_semaphore,
            allowed_extensions,
            lexicon_manager,
            search_db,
            graph_store,
            watch_refresh_signal,
            grammar_manager,
            ingestion_limits,
            metrics,
            queue_health,
            embedding_health,
            resource_profile_rx,
            warmup_state,
            state,
            poll_interval,
            keyword_embedding_generator,
            tier2_tagger,
            concept_config,
            narrative_config,
        )
        .await;
        if should_continue {
            return true;
        }
        Self::maybe_log_metrics(metrics, state, metrics_log_interval).await;
        false
    }

    /// Record the age of the oldest pending queue item into metrics.
    async fn record_oldest_pending_age(queue_manager: &QueueManager) {
        if let Ok(age) = queue_manager.get_oldest_pending_age_seconds().await {
            crate::monitoring::METRICS
                .queue_oldest_pending_age_seconds
                .set(age);
        }
    }

    /// Dequeue the next batch and dispatch to idle work, batch processing, or error handling.
    /// Returns `true` when the main loop should `continue` (skip the metrics/sleep tail).
    #[allow(clippy::too_many_arguments)]
    async fn handle_dequeue_result(
        fairness_scheduler: &Arc<FairnessScheduler>,
        config: &UnifiedProcessorConfig,
        queue_manager: &QueueManager,
        cancellation_token: &CancellationToken,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
        graph_store: &Option<Arc<dyn crate::graph::GraphStore>>,
        watch_refresh_signal: &Option<Arc<tokio::sync::Notify>>,
        grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: &Arc<IngestionLimitsConfig>,
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        queue_health: &Option<Arc<QueueProcessorHealth>>,
        embedding_health: &Option<crate::embedding::EmbeddingHealth>,
        resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        warmup_state: &Arc<WarmupState>,
        state: &mut LoopState,
        poll_interval: Duration,
        keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
        tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
        concept_config: &Arc<crate::config::ConceptConfig>,
        narrative_config: &Arc<crate::config::NarrativeConfig>,
    ) -> bool {
        let effective_batch_size = if state.recovery_ramp_remaining > 0 {
            (config.batch_size / 4).max(1)
        } else {
            config.batch_size
        };
        match fairness_scheduler
            .dequeue_next_batch(effective_batch_size)
            .await
        {
            Ok(items) if items.is_empty() => {
                run_idle_work(
                    state,
                    config,
                    queue_manager,
                    storage_client,
                    lexicon_manager,
                    grammar_manager,
                    lsp_manager,
                    search_db,
                    graph_store,
                    ingestion_limits,
                    poll_interval,
                )
                .await;
                true
            }
            Ok(items) => {
                Self::dispatch_nonempty_batch(
                    items,
                    state,
                    config,
                    queue_manager,
                    cancellation_token,
                    document_processor,
                    embedding_generator,
                    storage_client,
                    lsp_manager,
                    embedding_semaphore,
                    allowed_extensions,
                    lexicon_manager,
                    search_db,
                    graph_store,
                    watch_refresh_signal,
                    grammar_manager,
                    ingestion_limits,
                    metrics,
                    queue_health,
                    embedding_health,
                    resource_profile_rx,
                    warmup_state,
                    keyword_embedding_generator,
                    tier2_tagger,
                    concept_config,
                    narrative_config,
                )
                .await
            }
            Err(e) => {
                error!("Failed to dequeue unified batch: {}", e);
                if let Some(ref h) = queue_health {
                    h.record_error();
                }
                tokio::time::sleep(poll_interval).await;
                false
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_nonempty_batch(
        items: Vec<crate::unified_queue_schema::UnifiedQueueItem>,
        state: &mut LoopState,
        config: &UnifiedProcessorConfig,
        queue_manager: &QueueManager,
        cancellation_token: &CancellationToken,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
        graph_store: &Option<Arc<dyn crate::graph::GraphStore>>,
        watch_refresh_signal: &Option<Arc<tokio::sync::Notify>>,
        grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
        ingestion_limits: &Arc<IngestionLimitsConfig>,
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        queue_health: &Option<Arc<QueueProcessorHealth>>,
        embedding_health: &Option<crate::embedding::EmbeddingHealth>,
        resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        warmup_state: &Arc<WarmupState>,
        keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
        tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
        concept_config: &Arc<crate::config::ConceptConfig>,
        narrative_config: &Arc<crate::config::NarrativeConfig>,
    ) -> bool {
        state.maintenance_scheduler.cancel_active();
        state.idle_since = None;
        info!(
            "Dequeued {} unified queue items for processing",
            items.len()
        );
        if cancellation_token.is_cancelled() {
            warn!("Shutdown requested, stopping unified batch processing");
            return false;
        }
        match process_batch(
            items,
            config,
            queue_manager,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager,
            embedding_semaphore,
            allowed_extensions,
            lexicon_manager,
            search_db,
            graph_store,
            watch_refresh_signal,
            grammar_manager,
            ingestion_limits,
            metrics,
            queue_health,
            embedding_health,
            cancellation_token,
            resource_profile_rx,
            warmup_state,
            keyword_embedding_generator,
            tier2_tagger,
            concept_config,
            narrative_config,
        )
        .await
        {
            Err(()) => false,
            Ok(tenants) => {
                update_tenant_activity(&tenants, queue_manager).await;
                if state.recovery_ramp_remaining > 0 {
                    state.recovery_ramp_remaining -= 1;
                }
                false
            }
        }
    }

    /// Log processing metrics if the interval has elapsed.
    async fn maybe_log_metrics(
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        state: &mut LoopState,
        metrics_log_interval: ChronoDuration,
    ) {
        let now = Utc::now();
        if now - state.last_metrics_log >= metrics_log_interval {
            Self::log_metrics(metrics).await;
            state.last_metrics_log = now;
        }
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
             (max_embeddings: {} -> {})",
            warmup_state.elapsed_secs(),
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
