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
    graph_store: &Option<crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>>,
    watch_refresh_signal: &Option<Arc<tokio::sync::Notify>>,
    grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
    ingestion_limits: &Arc<IngestionLimitsConfig>,
    metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
    queue_health: &Option<Arc<QueueProcessorHealth>>,
    cancellation_token: &CancellationToken,
    resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
    warmup_state: &Arc<WarmupState>,
) -> Result<HashSet<String>, ()> {
    let mut processed_tenants: HashSet<String> = HashSet::new();

    for item in &items {
        if cancellation_token.is_cancelled() {
            warn!("Shutdown requested during item processing");
            return Err(());
        }

        if check_memory_pressure_and_pause(config, queue_manager, item).await {
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

/// Check memory pressure and re-lease the item if too high, returning `true` to break.
async fn check_memory_pressure_and_pause(
    config: &UnifiedProcessorConfig,
    queue_manager: &QueueManager,
    item: &UnifiedQueueItem,
) -> bool {
    if !UnifiedQueueProcessor::check_memory_pressure(config.max_memory_percent).await {
        return false;
    }

    warn!(
        "Memory pressure during batch processing (<{}% available), pausing remaining items",
        100u8.saturating_sub(config.max_memory_percent)
    );
    if let Err(e) = queue_manager.re_lease_item(&item.queue_id, 30).await {
        warn!("Failed to re-lease item during memory pressure: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(10)).await;
    true
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
) {
    let processing_time = start_time.elapsed().as_millis() as u64;
    METRICS.unified_queue_item_processed(
        &item.item_type.to_string(),
        &item.op.to_string(),
        "success",
        start_time.elapsed().as_secs_f64(),
    );

    // Per-destination state machine (Task 6): resolve unset destination statuses.
    let _ = queue_manager
        .ensure_destinations_resolved(&item.queue_id)
        .await;

    // Delete only when fully resolved.
    let overall = queue_manager
        .check_and_finalize(&item.queue_id)
        .await
        .unwrap_or(QueueStatus::Done);
    if overall == QueueStatus::Done {
        if let Err(e) = queue_manager.delete_unified_item(&item.queue_id).await {
            error!("Failed to delete item {} from queue: {}", item.queue_id, e);
        }
    } else {
        debug!(
            "Item {} not fully resolved (status={:?}), keeping in queue",
            item.queue_id, overall
        );
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
    } else if error_category == "subsystem_unavailable" {
        debug!(
            "Item {} parked: embedding subsystem unavailable ({})",
            item.queue_id, e
        );
        if let Err(rel_err) = queue_manager.re_lease_item(&item.queue_id, 60).await {
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
