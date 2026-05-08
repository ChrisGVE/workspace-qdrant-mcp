//! Main processing loop for the unified queue processor.

use chrono::{Duration as ChronoDuration, Utc};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::config::IngestionLimitsConfig;
use crate::fairness_scheduler::FairnessScheduler;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::monitoring::metrics_core::METRICS;
use crate::queue_health::QueueProcessorHealth;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tree_sitter::GrammarManager;
use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus};
use crate::{DocumentProcessor, EmbeddingGenerator};

use crate::unified_queue_processor::config::{
    UnifiedProcessingMetrics, UnifiedProcessorConfig, WarmupState,
};
use crate::unified_queue_processor::error::UnifiedProcessorResult;
use crate::unified_queue_processor::UnifiedQueueProcessor;

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
            "Unified processing loop started (batch_size={}, worker_id={}, fairness={}, warmup_window={}s, maintenance_tasks={})",
            config.batch_size, config.worker_id, config.fairness_enabled,
            config.warmup_window_secs, state.maintenance_scheduler.task_count()
        );

        loop {
            // Log warmup transition and adjust embedding semaphore (Task 577, Task 578)
            if !state.warmup_logged && !warmup_state.is_in_warmup() {
                let permits_to_add = config
                    .max_concurrent_embeddings
                    .saturating_sub(config.warmup_max_concurrent_embeddings);
                if permits_to_add > 0 {
                    embedding_semaphore.add_permits(permits_to_add);
                }
                info!(
                    "Warmup period complete after {}s - switching to normal resource limits (delay: {}ms -> {}ms, max_embeddings: {} -> {})",
                    warmup_state.elapsed_secs(),
                    config.warmup_inter_item_delay_ms,
                    config.inter_item_delay_ms,
                    config.warmup_max_concurrent_embeddings,
                    config.max_concurrent_embeddings
                );
                state.warmup_logged = true;
            }

            // Check for shutdown signal
            if cancellation_token.is_cancelled() {
                info!("Unified queue shutdown signal received");
                break;
            }

            // Adaptive resource scaling: adjust semaphore permits on profile change
            if let Some(ref mut rx) = resource_profile_rx {
                if rx.has_changed().unwrap_or(false) {
                    let profile = *rx.borrow_and_update();
                    let target = profile.max_concurrent_embeddings;
                    if state.warmup_logged {
                        let current_target = state
                            .adaptive_target_permits
                            .unwrap_or(config.max_concurrent_embeddings);
                        if target > current_target {
                            let permits_to_add = target - current_target;
                            embedding_semaphore.add_permits(permits_to_add);
                            debug!(
                                "Adaptive: added {} semaphore permits (target: {})",
                                permits_to_add, target
                            );
                        } else if target < current_target {
                            let excess = current_target - target;
                            let removed = embedding_semaphore.forget_permits(excess);
                            debug!(
                                "Adaptive: removed {} semaphore permits (requested: {}, target: {})",
                                removed, excess, target
                            );
                        }
                        state.adaptive_target_permits = Some(target);
                    }
                }
            }

            // Record poll for health monitoring
            if let Some(ref h) = queue_health {
                h.record_poll();
            }

            // Check memory pressure before dequeuing (Task 504)
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

            // Qdrant circuit breaker: when open, probe once to detect recovery
            if !storage_client.is_qdrant_available() {
                debug!("Qdrant circuit breaker open — probing before dequeue");
                match storage_client.test_connection().await {
                    Ok(true) => {
                        storage_client.circuit_breaker().record_success();
                        info!("Qdrant recovered — resuming queue processing");
                        match queue_manager
                            .resurrect_failed_transient(config.max_resurrections)
                            .await
                        {
                            Ok((resurrected, exhausted)) => {
                                if resurrected > 0 || exhausted > 0 {
                                    info!(
                                        "Recovery resurrection: reset {} item(s), exhausted {} item(s)",
                                        resurrected, exhausted
                                    );
                                }
                            }
                            Err(e) => warn!("Recovery resurrection failed: {}", e),
                        }
                    }
                    _ => {
                        debug!("Qdrant still unavailable, sleeping 5s");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                }
            }

            // Update queue depth in metrics and adaptive resource counter
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

            // Dequeue batch of items using fairness scheduler (Task 34)
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

                    // Queue has items — cancel maintenance and reset idle tracker
                    state.maintenance_scheduler.cancel_active();
                    state.idle_since = None;

                    info!(
                        "Dequeued {} unified queue items for processing",
                        items.len()
                    );

                    // Check shutdown signal before processing batch
                    if cancellation_token.is_cancelled() {
                        warn!("Shutdown requested, stopping unified batch processing");
                        return Ok(());
                    }

                    // Track tenant_ids with successful processing for activity update
                    let mut processed_tenants = std::collections::HashSet::new();

                    // Process items sequentially
                    for item in &items {
                        if cancellation_token.is_cancelled() {
                            warn!("Shutdown requested during item processing");
                            return Ok(());
                        }

                        // Memory pressure check between items to prevent runaway growth
                        if Self::check_memory_pressure(config.max_memory_percent).await {
                            warn!(
                                "Memory pressure during batch processing (<{}% available), \
                                 pausing remaining items",
                                100u8.saturating_sub(config.max_memory_percent)
                            );
                            if let Err(e) = queue_manager.re_lease_item(&item.queue_id, 30).await {
                                warn!("Failed to re-lease item during memory pressure: {}", e);
                            }
                            tokio::time::sleep(Duration::from_secs(10)).await;
                            break;
                        }

                        let start_time = std::time::Instant::now();
                        let item_type_str = format!("{:?}", item.item_type);

                        match Self::process_item(
                            &queue_manager,
                            item,
                            &config,
                            &document_processor,
                            &embedding_generator,
                            &storage_client,
                            &lsp_manager,
                            &embedding_semaphore,
                            &allowed_extensions,
                            &lexicon_manager,
                            &search_db,
                            &graph_store,
                            &grammar_manager,
                            &ingestion_limits,
                        )
                        .await
                        {
                            Ok(()) => {
                                let processing_time = start_time.elapsed().as_millis() as u64;
                                METRICS.unified_queue_item_processed(
                                    &item.item_type.to_string(),
                                    &item.op.to_string(),
                                    "success",
                                    start_time.elapsed().as_secs_f64(),
                                );
                                let _ = queue_manager
                                    .ensure_destinations_resolved(&item.queue_id)
                                    .await;
                                let overall = queue_manager
                                    .check_and_finalize(&item.queue_id)
                                    .await
                                    .unwrap_or(QueueStatus::Done);
                                if overall == QueueStatus::Done {
                                    if let Err(e) =
                                        queue_manager.delete_unified_item(&item.queue_id).await
                                    {
                                        error!(
                                            "Failed to delete item {} from queue: {}",
                                            item.queue_id, e
                                        );
                                    }
                                } else {
                                    debug!(
                                        "Item {} not fully resolved (status={:?}), keeping in queue",
                                        item.queue_id, overall
                                    );
                                }
                                Self::update_metrics_success(
                                    &metrics,
                                    &item_type_str,
                                    processing_time,
                                )
                                .await;
                                if let Some(ref h) = queue_health {
                                    h.record_success(processing_time);
                                    h.record_heartbeat();
                                }
                                processed_tenants.insert(item.tenant_id.clone());
                                info!(
                                    "Successfully processed unified item {} (type={:?}, op={:?}) in {}ms",
                                    item.queue_id, item.item_type, item.op, processing_time
                                );
                                // Task 12: Signal WatchManager to refresh after Tenant/Add.
                                if item.item_type == ItemType::Tenant
                                    && item.op == QueueOperation::Add
                                {
                                    if let Some(ref signal) = watch_refresh_signal {
                                        signal.notify_one();
                                        debug!(
                                            "Signaled WatchManager refresh after Tenant/Add for {}",
                                            item.tenant_id
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                let error_category = Self::classify_error(&e);
                                let is_permanent = Self::is_permanent_category(error_category);
                                METRICS.unified_queue_item_processed(
                                    &item.item_type.to_string(),
                                    &item.op.to_string(),
                                    "failure",
                                    start_time.elapsed().as_secs_f64(),
                                );
                                if error_category == "permanent_gone" {
                                    warn!(
                                        "Item {} gone (type={:?}), removing from queue: {}",
                                        item.queue_id, item.item_type, e
                                    );
                                    if let Err(del_err) =
                                        queue_manager.delete_unified_item(&item.queue_id).await
                                    {
                                        error!(
                                            "Failed to delete gone item {}: {}",
                                            item.queue_id, del_err
                                        );
                                    }
                                } else if error_category == "subsystem_unavailable" {
                                    debug!(
                                        "Item {} parked: embedding subsystem unavailable ({})",
                                        item.queue_id, e
                                    );
                                    if let Err(rel_err) =
                                        queue_manager.re_lease_item(&item.queue_id, 60).await
                                    {
                                        error!(
                                            "Failed to re-lease unavailable item {}: {}",
                                            item.queue_id, rel_err
                                        );
                                    }
                                } else {
                                    error!(
                                        error_category = error_category,
                                        permanent = is_permanent,
                                        "Failed to process unified item {} (type={:?}): {}",
                                        item.queue_id,
                                        item.item_type,
                                        e
                                    );
                                    let categorized_msg = format!("[{}] {}", error_category, e);
                                    if let Err(mark_err) = queue_manager
                                        .mark_unified_failed(
                                            &item.queue_id,
                                            &categorized_msg,
                                            is_permanent,
                                            config.max_retries,
                                        )
                                        .await
                                    {
                                        error!(
                                            "Failed to mark item {} as failed: {}",
                                            item.queue_id, mark_err
                                        );
                                    } else if !is_permanent {
                                        METRICS.unified_queue_retry(&item.item_type.to_string());
                                    }
                                }
                                Self::update_metrics_failure(&metrics, &e).await;
                                if let Some(ref h) = queue_health {
                                    h.record_failure();
                                    h.record_heartbeat();
                                }
                            }
                        }

                        // Inter-item delay (Task 504 / Task 577)
                        // Priority: warmup > adaptive profile > config default
                        let effective_delay_ms = if warmup_state.is_in_warmup() {
                            config.warmup_inter_item_delay_ms
                        } else if let Some(ref rx) = resource_profile_rx {
                            rx.borrow().inter_item_delay_ms
                        } else {
                            config.inter_item_delay_ms
                        };
                        if effective_delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(effective_delay_ms)).await;
                        }
                    }

                    // Implicit activity update for tenants that had items processed
                    if !processed_tenants.is_empty() {
                        let now_str = wqm_common::timestamps::now_utc();
                        for tenant_id in &processed_tenants {
                            if let Err(e) = sqlx::query(
                                "UPDATE watch_folders SET last_activity_at = ?1, updated_at = ?1 \
                                 WHERE tenant_id = ?2 AND collection = 'projects' AND is_active > 0",
                            )
                            .bind(&now_str)
                            .bind(tenant_id)
                            .execute(queue_manager.pool())
                            .await
                            {
                                debug!(
                                    "Failed to update activity for tenant {}: {}",
                                    tenant_id, e
                                );
                            }
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

            // Log metrics periodically
            let now = Utc::now();
            if now - state.last_metrics_log >= metrics_log_interval {
                Self::log_metrics(&metrics).await;
                state.last_metrics_log = now;
            }

            // Brief pause before next batch
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }
}
