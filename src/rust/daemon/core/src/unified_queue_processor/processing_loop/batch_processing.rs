//! Batch processing for the unified queue processor.
//!
//! `process_batch` handles a non-empty batch of queue items sequentially.
//! It returns `Err(())` when a cancellation signal is detected mid-batch,
//! signalling the caller to `return Ok(())` from the processing loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::config::IngestionLimitsConfig;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::monitoring::metrics_core::METRICS;
use crate::queue_health::QueueProcessorHealth;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tree_sitter::GrammarManager;
use crate::unified_queue_processor::config::{
    UnifiedProcessingMetrics, UnifiedProcessorConfig, WarmupState,
};
use crate::unified_queue_processor::error::UnifiedProcessorError;
use crate::unified_queue_processor::UnifiedQueueProcessor;
use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus, UnifiedQueueItem};
use crate::{DocumentProcessor, EmbeddingGenerator};

/// Process a non-empty batch of queue items sequentially.
///
/// Returns `Err(())` if a shutdown cancellation is detected during processing
/// (the caller should then `return Ok(())`). Returns `Ok(())` otherwise.
#[allow(clippy::too_many_arguments)]
pub(super) async fn process_batch(
    items: Vec<UnifiedQueueItem>,
    config: &UnifiedProcessorConfig,
    queue_manager: &QueueManager,
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
    cancellation_token: &CancellationToken,
    resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
    warmup_state: &Arc<WarmupState>,
    keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
) -> Result<HashSet<String>, ()> {
    let mut processed_tenants: HashSet<String> = HashSet::new();

    for (item_idx, item) in items.iter().enumerate() {
        if cancellation_token.is_cancelled() {
            warn!("Shutdown requested during item processing");
            return Err(());
        }

        if check_memory_pressure(config).await {
            // Re-lease ALL remaining items in the batch (F-044), not just the
            // current one, so they return to pending instead of being stuck
            // in_progress until lease expiry.
            re_lease_remaining_items(queue_manager, &items, item_idx).await;
            tokio::time::sleep(Duration::from_secs(10)).await;
            break;
        }

        let start_time = std::time::Instant::now();
        let item_type_str = format!("{:?}", item.item_type);

        match UnifiedQueueProcessor::process_item(
            queue_manager,
            item,
            config,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager,
            embedding_semaphore,
            allowed_extensions,
            lexicon_manager,
            search_db,
            graph_store,
            grammar_manager,
            ingestion_limits,
            keyword_embedding_generator,
        )
        .await
        {
            Ok(()) => {
                handle_item_success(
                    item,
                    start_time,
                    queue_manager,
                    metrics,
                    queue_health,
                    watch_refresh_signal,
                    &item_type_str,
                    &mut processed_tenants,
                    config,
                )
                .await;
            }
            Err(e) => {
                handle_item_failure(
                    item,
                    e,
                    start_time,
                    queue_manager,
                    metrics,
                    queue_health,
                    config,
                )
                .await;
            }
        }

        apply_inter_item_delay(config, warmup_state, resource_profile_rx).await;
    }

    Ok(processed_tenants)
}

/// Check memory pressure, returning `true` if the batch should pause.
async fn check_memory_pressure(config: &UnifiedProcessorConfig) -> bool {
    if !UnifiedQueueProcessor::check_memory_pressure(config.max_memory_percent).await {
        return false;
    }

    warn!(
        "Memory pressure during batch processing (<{}% available), pausing remaining items",
        100u8.saturating_sub(config.max_memory_percent)
    );
    true
}

/// Re-lease all items from `from_idx` onward so they return to pending (F-044).
///
/// Without this, items leased as part of the batch but not yet processed would
/// remain `in_progress` until their lease expires, blocking other workers.
async fn re_lease_remaining_items(
    queue_manager: &QueueManager,
    items: &[UnifiedQueueItem],
    from_idx: usize,
) {
    let remaining = &items[from_idx..];
    let count = remaining.len();
    let mut failures = 0;
    for item in remaining {
        if let Err(e) = queue_manager.re_lease_item(&item.queue_id, 30).await {
            warn!(queue_id = %item.queue_id, "Failed to re-lease item during memory pressure: {}", e);
            failures += 1;
        }
    }
    info!(
        total = count,
        released = count - failures,
        "Re-leased remaining batch items due to memory pressure"
    );
}

/// Apply inter-item delay based on warmup state, adaptive profile, or config default.
async fn apply_inter_item_delay(
    config: &UnifiedProcessorConfig,
    warmup_state: &Arc<WarmupState>,
    resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
) {
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

/// Handles the success outcome of a single processed item.
#[allow(clippy::too_many_arguments)]
async fn handle_item_success(
    item: &UnifiedQueueItem,
    start_time: std::time::Instant,
    queue_manager: &QueueManager,
    metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
    queue_health: &Option<Arc<QueueProcessorHealth>>,
    watch_refresh_signal: &Option<Arc<tokio::sync::Notify>>,
    item_type_str: &str,
    processed_tenants: &mut HashSet<String>,
    config: &UnifiedProcessorConfig,
) {
    let processing_time = start_time.elapsed().as_millis() as u64;
    METRICS.unified_queue_item_processed(
        &item.item_type.to_string(),
        &item.op.to_string(),
        "success",
        start_time.elapsed().as_secs_f64(),
    );

    // Per-destination state machine (F-010, F-056): only auto-resolve
    // destination statuses for orchestration-only items (decision_json IS NULL).
    // Items that opted into the state machine via store_queue_decision must
    // set every destination status explicitly — pending sinks remain pending.
    let _ = queue_manager
        .mark_explicit_destination_results(&item.queue_id)
        .await;

    // Resolve overall status. For state-machine items with pending sinks the
    // helper will return InProgress and the item remains in the queue for the
    // next lease cycle.
    //
    // On DB/query error, keep the item in the queue for retry rather than
    // coercing to Done and deleting it -- a transient DB error must not
    // permanently discard unfinished work.
    let overall = match queue_manager.check_and_finalize(&item.queue_id).await {
        Ok(status) => status,
        Err(e) => {
            error!(
                "check_and_finalize failed for item {}, keeping in queue for retry: {}",
                item.queue_id, e
            );
            // Record success metrics despite finalization failure -- the item
            // processing itself succeeded; only the bookkeeping step failed.
            UnifiedQueueProcessor::update_metrics_success(metrics, item_type_str, processing_time)
                .await;
            if let Some(ref h) = queue_health {
                h.record_success(processing_time);
                h.record_heartbeat();
            }
            processed_tenants.insert(item.tenant_id.clone());
            return;
        }
    };

    match overall {
        QueueStatus::Done => {
            if let Err(e) = queue_manager.delete_unified_item(&item.queue_id).await {
                error!("Failed to delete item {} from queue: {}", item.queue_id, e);
            }
        }
        QueueStatus::Failed => {
            // F-033/F-034: a destination explicitly reported failure even though
            // the handler returned Ok. Promote to mark_unified_failed so retry
            // metadata (retry_count, error_message, backoff via lease_until)
            // is populated; without this, the row would stay `failed` with no
            // way to surface or schedule a retry.
            let err_msg = format!(
                "destination failure on success path (qdrant_status/search_status reported failed) for item={} type={:?}",
                item.queue_id, item.item_type
            );
            if let Err(mark_err) = queue_manager
                .mark_unified_failed(&item.queue_id, &err_msg, false, config.max_retries)
                .await
            {
                error!(
                    "Failed to record destination-failure retry metadata for {}: {}",
                    item.queue_id, mark_err
                );
            }
        }
        QueueStatus::InProgress | QueueStatus::Pending => {
            debug!(
                "Item {} not fully resolved (status={:?}), keeping in queue",
                item.queue_id, overall
            );
        }
    }

    UnifiedQueueProcessor::update_metrics_success(metrics, item_type_str, processing_time).await;

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
    if item.item_type == ItemType::Tenant && item.op == QueueOperation::Add {
        if let Some(ref signal) = watch_refresh_signal {
            signal.notify_one();
            debug!(
                "Signaled WatchManager refresh after Tenant/Add for {}",
                item.tenant_id
            );
        }
    }
}

/// Handles the failure outcome of a single processed item.
async fn handle_item_failure(
    item: &UnifiedQueueItem,
    e: UnifiedProcessorError,
    start_time: std::time::Instant,
    queue_manager: &QueueManager,
    metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
    queue_health: &Option<Arc<QueueProcessorHealth>>,
    config: &UnifiedProcessorConfig,
) {
    let error_category = UnifiedQueueProcessor::classify_error(&e);
    let is_permanent = UnifiedQueueProcessor::is_permanent_category(error_category);
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
        if let Err(del_err) = queue_manager.delete_unified_item(&item.queue_id).await {
            error!("Failed to delete gone item {}: {}", item.queue_id, del_err);
        }
    } else if error_category == "subsystem_unavailable" || error_category == "rate_limit" {
        debug!("Item {} parked: {} ({})", item.queue_id, error_category, e);
        if let Err(rel_err) = queue_manager.re_lease_item(&item.queue_id, 60).await {
            error!(
                "Failed to re-lease {} item {}: {}",
                error_category, item.queue_id, rel_err
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
        let should_dlq = is_permanent || item.retry_count + 1 >= config.max_retries;

        if should_dlq {
            if let Err(mark_err) = queue_manager
                .mark_unified_failed(&item.queue_id, &categorized_msg, true, config.max_retries)
                .await
            {
                error!(
                    "Failed to mark item {} as failed: {}",
                    item.queue_id, mark_err
                );
            }
            match queue_manager.move_to_dlq(&item.queue_id).await {
                Ok(dlq_id) => {
                    info!(
                        "Moved exhausted item {} to DLQ {} (category={})",
                        item.queue_id, dlq_id, error_category
                    );
                }
                Err(dlq_err) => {
                    warn!(
                        "Failed to move item {} to DLQ (stays in failed): {}",
                        item.queue_id, dlq_err
                    );
                }
            }
        } else {
            if let Err(mark_err) = queue_manager
                .mark_unified_failed(&item.queue_id, &categorized_msg, false, config.max_retries)
                .await
            {
                error!(
                    "Failed to mark item {} as failed: {}",
                    item.queue_id, mark_err
                );
            } else {
                METRICS.unified_queue_retry(&item.item_type.to_string());
            }
        }
    }

    UnifiedQueueProcessor::update_metrics_failure(metrics, &e).await;
    if let Some(ref h) = queue_health {
        h.record_failure();
        h.record_heartbeat();
    }
}

/// Updates `last_activity_at` for all tenants processed in a batch.
pub(super) async fn update_tenant_activity(
    processed_tenants: &HashSet<String>,
    queue_manager: &QueueManager,
) {
    if processed_tenants.is_empty() {
        return;
    }
    let now_str = wqm_common::timestamps::now_utc();
    for tenant_id in processed_tenants {
        if let Err(e) = sqlx::query(
            "UPDATE watch_folders SET last_activity_at = ?1, updated_at = ?1 \
             WHERE tenant_id = ?2 AND collection = 'projects' AND is_active > 0",
        )
        .bind(&now_str)
        .bind(tenant_id)
        .execute(queue_manager.pool())
        .await
        {
            debug!("Failed to update activity for tenant {}: {}", tenant_id, e);
        }
    }
}
