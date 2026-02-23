//! Main processing loop and item dispatch for the unified queue processor.

use std::sync::Arc;
use std::time::Duration;
use chrono::{Duration as ChronoDuration, Utc};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::fairness_scheduler::FairnessScheduler;
use crate::lsp::LanguageServerManager;
use crate::queue_health::QueueProcessorHealth;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus, UnifiedQueueItem};
use crate::{DocumentProcessor, EmbeddingGenerator};
use crate::storage::StorageClient;
use crate::lexicon::LexiconManager;

use super::config::{UnifiedProcessingMetrics, UnifiedProcessorConfig, WarmupState};
use super::error::UnifiedProcessorResult;
use super::UnifiedQueueProcessor;

impl UnifiedQueueProcessor {
    /// Check if system memory usage exceeds the configured threshold (Task 504)
    pub(crate) async fn check_memory_pressure(max_memory_percent: u8) -> bool {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        let total = sys.total_memory();
        if total == 0 {
            return false;
        }
        let used = sys.used_memory();
        let usage_percent = (used as f64 / total as f64 * 100.0) as u8;
        usage_percent > max_memory_percent
    }

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
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mut last_metrics_log = Utc::now();
        let mut resource_profile_rx = resource_profile_rx;
        // Track current adaptive target to detect changes
        let mut adaptive_target_permits: Option<usize> = None;
        let metrics_log_interval = ChronoDuration::minutes(1);
        let mut warmup_logged = false;
        // Metadata uplift tracking
        let mut uplift_config = crate::metadata_uplift::UpliftConfig::default();
        let mut last_uplift_attempt = std::time::Instant::now()
            .checked_sub(Duration::from_secs(uplift_config.min_interval_secs))
            .unwrap_or_else(std::time::Instant::now);

        info!(
            "Unified processing loop started (batch_size={}, worker_id={}, fairness={}, warmup_window={}s)",
            config.batch_size, config.worker_id, config.fairness_enabled,
            config.warmup_window_secs
        );

        loop {
            // Log warmup transition and adjust embedding semaphore (Task 577, Task 578)
            if !warmup_logged && !warmup_state.is_in_warmup() {
                // Add permits to embedding semaphore to reach normal limit (Task 578)
                let permits_to_add = config.max_concurrent_embeddings.saturating_sub(config.warmup_max_concurrent_embeddings);
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
                warmup_logged = true;
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

                    // Only adjust after warmup is complete
                    if warmup_logged {
                        let current_target = adaptive_target_permits.unwrap_or(config.max_concurrent_embeddings);
                        if target > current_target {
                            let permits_to_add = target - current_target;
                            embedding_semaphore.add_permits(permits_to_add);
                            debug!("Adaptive: added {} semaphore permits (target: {})", permits_to_add, target);
                        }
                        // Note: tokio::sync::Semaphore doesn't support shrinking permits.
                        // When scaling down, in-flight operations keep their permits and
                        // new acquires will block once available permits drop below the
                        // new target. We track the target so subsequent transitions are correct.
                        adaptive_target_permits = Some(target);
                    }
                }
            }

            // Record poll for health monitoring
            if let Some(ref h) = queue_health {
                h.record_poll();
            }

            // Check memory pressure before dequeuing (Task 504)
            if Self::check_memory_pressure(config.max_memory_percent).await {
                info!("Memory pressure detected (>{}%), pausing processing for 5s", config.max_memory_percent);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
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

            // Dequeue batch of items using fairness scheduler (Task 34)
            // The scheduler handles active project prioritization and starvation prevention
            match fairness_scheduler
                .dequeue_next_batch(config.batch_size)
                .await
            {
                Ok(items) => {
                    if items.is_empty() {
                        // Run metadata uplift when queue is idle (Task 18)
                        let since_last = last_uplift_attempt.elapsed().as_secs();
                        if since_last >= uplift_config.min_interval_secs {
                            debug!("Queue idle — running metadata uplift pass (gen={})", uplift_config.current_generation);
                            let collections = vec!["projects".to_string(), "libraries".to_string()];
                            let stats = crate::metadata_uplift::run_uplift_pass(
                                &storage_client,
                                &lexicon_manager,
                                &collections,
                                &uplift_config,
                            )
                            .await;
                            if stats.scanned > 0 {
                                info!(
                                    "Uplift pass complete: scanned={}, updated={}, skipped={}, errors={}",
                                    stats.scanned, stats.updated, stats.skipped, stats.errors
                                );
                            }
                            if stats.updated == 0 && stats.errors == 0 {
                                // All points uplifted at this generation, advance
                                uplift_config.current_generation += 1;
                            }
                            last_uplift_attempt = std::time::Instant::now();
                        }

                        debug!("Unified queue is empty, waiting {}ms", config.poll_interval_ms);
                        tokio::time::sleep(poll_interval).await;
                        continue;
                    }

                    info!("Dequeued {} unified queue items for processing", items.len());

                    // Check shutdown signal before processing batch
                    if cancellation_token.is_cancelled() {
                        warn!("Shutdown requested, stopping unified batch processing");
                        return Ok(());
                    }

                    // Track tenant_ids with successful processing for activity update
                    let mut processed_tenants = std::collections::HashSet::new();

                    // Process items sequentially
                    for item in items {
                        if cancellation_token.is_cancelled() {
                            warn!("Shutdown requested during item processing");
                            return Ok(());
                        }

                        let start_time = std::time::Instant::now();
                        let item_type_str = format!("{:?}", item.item_type);

                        match Self::process_item(
                            &queue_manager,
                            &item,
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
                        )
                        .await
                        {
                            Ok(()) => {
                                let processing_time = start_time.elapsed().as_millis() as u64;

                                // Per-destination state machine (Task 6):
                                // Resolve any destination statuses that weren't explicitly set
                                // by the handler (orchestration-only items, content items, etc.)
                                // before checking finalization. Without this, items whose handlers
                                // don't call update_destination_status() would stay pending forever.
                                let _ = queue_manager
                                    .ensure_destinations_resolved(&item.queue_id)
                                    .await;

                                // check_and_finalize resolves overall status from qdrant_status + search_status.
                                // Delete item only when fully resolved as done.
                                let overall = queue_manager
                                    .check_and_finalize(&item.queue_id)
                                    .await
                                    .unwrap_or(QueueStatus::Done);

                                if overall == QueueStatus::Done {
                                    if let Err(e) = queue_manager
                                        .delete_unified_item(&item.queue_id)
                                        .await
                                    {
                                        error!("Failed to delete item {} from queue: {}", item.queue_id, e);
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
                                ).await;

                                if let Some(ref h) = queue_health {
                                    h.record_success(processing_time);
                                }

                                // Track tenant for implicit activity update
                                processed_tenants.insert(item.tenant_id.clone());

                                info!(
                                    "Successfully processed unified item {} (type={:?}, op={:?}) in {}ms",
                                    item.queue_id, item.item_type, item.op, processing_time
                                );

                                // Task 12: Signal WatchManager to refresh after creating a new watch_folder.
                                // This enables immediate file + git watcher startup for newly registered projects.
                                if item.item_type == ItemType::Tenant && item.op == QueueOperation::Add {
                                    if let Some(ref signal) = watch_refresh_signal {
                                        signal.notify_one();
                                        debug!("Signaled WatchManager refresh after Tenant/Add for {}", item.tenant_id);
                                    }
                                }
                            }
                            Err(e) => {
                                // Classify error into 5 categories for observability
                                let error_category = Self::classify_error(&e);
                                let is_permanent = Self::is_permanent_category(error_category);

                                if error_category == "permanent_gone" {
                                    // File deleted or inaccessible — nothing to retry.
                                    // Delete from queue silently (the resource is gone).
                                    warn!(
                                        "Item {} gone (type={:?}), removing from queue: {}",
                                        item.queue_id, item.item_type, e
                                    );
                                    if let Err(del_err) = queue_manager
                                        .delete_unified_item(&item.queue_id)
                                        .await
                                    {
                                        error!("Failed to delete gone item {}: {}", item.queue_id, del_err);
                                    }
                                } else {
                                    error!(
                                        error_category = error_category,
                                        permanent = is_permanent,
                                        "Failed to process unified item {} (type={:?}): {}",
                                        item.queue_id, item.item_type, e
                                    );

                                    // Mark item as failed (with exponential backoff for transient errors)
                                    // Prefix error message with category for observability
                                    let categorized_msg = format!("[{}] {}", error_category, e);
                                    if let Err(mark_err) = queue_manager
                                        .mark_unified_failed(&item.queue_id, &categorized_msg, is_permanent)
                                        .await
                                    {
                                        error!("Failed to mark item {} as failed: {}", item.queue_id, mark_err);
                                    }
                                }

                                Self::update_metrics_failure(&metrics, &e).await;
                                if let Some(ref h) = queue_health {
                                    h.record_failure();
                                }
                            }
                        }

                        // Inter-item delay for resource breathing room (Task 504 / Task 577)
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

                    // Implicit activity update: refresh last_activity_at for active projects
                    // that had items processed in this batch. Prevents mid-processing deactivation
                    // by the orphan cleanup without requiring explicit heartbeats.
                    if !processed_tenants.is_empty() {
                        let now_str = wqm_common::timestamps::now_utc();
                        for tenant_id in &processed_tenants {
                            if let Err(e) = sqlx::query(
                                "UPDATE watch_folders SET last_activity_at = ?1, updated_at = ?1 \
                                 WHERE tenant_id = ?2 AND collection = 'projects' AND is_active = 1"
                            )
                            .bind(&now_str)
                            .bind(tenant_id)
                            .execute(queue_manager.pool())
                            .await {
                                debug!("Failed to update activity for tenant {}: {}", tenant_id, e);
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
            if now - last_metrics_log >= metrics_log_interval {
                Self::log_metrics(&metrics).await;
                last_metrics_log = now;
            }

            // Brief pause before next batch
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }

    /// Process a single unified queue item based on its type
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn process_item(
        queue_manager: &QueueManager,
        item: &UnifiedQueueItem,
        _config: &UnifiedProcessorConfig,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
        graph_store: &Option<crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>>,
    ) -> UnifiedProcessorResult<()> {
        debug!(
            "Processing unified item: {} (type={:?}, op={:?}, collection={})",
            item.queue_id, item.item_type, item.op, item.collection
        );

        let mut ctx = crate::context::ProcessingContext::new(
            queue_manager.pool().clone(),
            Arc::new(queue_manager.clone()),
            Arc::clone(storage_client),
            Arc::clone(embedding_generator),
            Arc::clone(document_processor),
            Arc::clone(embedding_semaphore),
            Arc::clone(lexicon_manager),
            lsp_manager.clone(),
            search_db.clone(),
            Arc::clone(allowed_extensions),
        );
        if let Some(gs) = graph_store {
            ctx = ctx.with_graph_store(gs.clone());
        }

        match item.item_type {
            ItemType::Text => {
                crate::strategies::processing::text::TextStrategy::process_content_item(&ctx, item).await
            }
            ItemType::File => {
                crate::strategies::processing::file::FileStrategy::process_file_item(&ctx, item).await
            }
            ItemType::Folder => {
                crate::strategies::processing::folder::FolderStrategy::process_folder_item(&ctx, item).await
            }
            ItemType::Tenant => {
                crate::strategies::processing::tenant::TenantStrategy::process_tenant_item(&ctx, item).await
            }
            ItemType::Doc => {
                crate::strategies::processing::tenant::TenantStrategy::process_doc_item(&ctx, item).await
            }
            ItemType::Url => {
                crate::strategies::processing::url::UrlStrategy::process_url_item(&ctx, item).await
            }
            ItemType::Website => {
                crate::strategies::processing::website::WebsiteStrategy::process_website_item(&ctx, item).await
            }
            ItemType::Collection => {
                crate::strategies::processing::collection::CollectionStrategy::process_collection_item(&ctx, item).await
            }
        }
    }
}
