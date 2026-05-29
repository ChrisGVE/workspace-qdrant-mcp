//! Batch processing for the unified queue processor.
//!
//! `process_batch` processes a non-empty batch of queue items concurrently,
//! gated by `max_concurrent_embeddings`. It returns `Err(())` when a
//! cancellation signal is detected, signalling the caller to shut down.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
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

struct BatchContext {
    config: UnifiedProcessorConfig,
    queue_manager: QueueManager,
    document_processor: Arc<DocumentProcessor>,
    embedding_generator: Arc<EmbeddingGenerator>,
    storage_client: Arc<StorageClient>,
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    embedding_semaphore: Arc<tokio::sync::Semaphore>,
    allowed_extensions: Arc<AllowedExtensions>,
    lexicon_manager: Arc<LexiconManager>,
    search_db: Option<Arc<SearchDbManager>>,
    graph_store: Option<Arc<dyn crate::graph::GraphStore>>,
    watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,
    grammar_manager: Option<Arc<RwLock<GrammarManager>>>,
    ingestion_limits: Arc<IngestionLimitsConfig>,
    metrics: Arc<RwLock<UnifiedProcessingMetrics>>,
    queue_health: Option<Arc<QueueProcessorHealth>>,
    keyword_embedding_generator: Option<Arc<EmbeddingGenerator>>,
    tier2_tagger: Option<Arc<crate::tagging::Tier2Tagger>>,
    concept_config: Arc<crate::config::ConceptConfig>,
}

/// Process a non-empty batch of queue items concurrently.
///
/// Items are processed with up to `max_concurrent_embeddings` in-flight at
/// once. Returns `Err(())` if a shutdown cancellation is detected.
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
    _resource_profile_rx: &Option<tokio::sync::watch::Receiver<ResourceProfile>>,
    _warmup_state: &Arc<WarmupState>,
    keyword_embedding_generator: &Option<Arc<EmbeddingGenerator>>,
    tier2_tagger: &Option<Arc<crate::tagging::Tier2Tagger>>,
    concept_config: &Arc<crate::config::ConceptConfig>,
) -> Result<HashSet<String>, ()> {
    let ctx = Arc::new(BatchContext {
        config: config.clone(),
        queue_manager: queue_manager.clone(),
        document_processor: Arc::clone(document_processor),
        embedding_generator: Arc::clone(embedding_generator),
        storage_client: Arc::clone(storage_client),
        lsp_manager: lsp_manager.clone(),
        embedding_semaphore: Arc::clone(embedding_semaphore),
        allowed_extensions: Arc::clone(allowed_extensions),
        lexicon_manager: Arc::clone(lexicon_manager),
        search_db: search_db.clone(),
        graph_store: graph_store.clone(),
        watch_refresh_signal: watch_refresh_signal.clone(),
        grammar_manager: grammar_manager.clone(),
        ingestion_limits: Arc::clone(ingestion_limits),
        metrics: Arc::clone(metrics),
        queue_health: queue_health.clone(),
        keyword_embedding_generator: keyword_embedding_generator.clone(),
        tier2_tagger: tier2_tagger.clone(),
        concept_config: Arc::clone(concept_config),
    });

    let concurrency = config.max_concurrent_embeddings.max(1);
    let total = items.len();
    let mut futures: FuturesUnordered<tokio::task::JoinHandle<Option<String>>> =
        FuturesUnordered::new();
    let mut processed_tenants: HashSet<String> = HashSet::new();
    let mut next_idx = 0;

    // Seed the pipeline up to concurrency limit
    while next_idx < total && futures.len() < concurrency {
        let item = items[next_idx].clone();
        let batch_ctx = Arc::clone(&ctx);
        futures.push(tokio::spawn(async move {
            process_single_item(item, &batch_ctx).await
        }));
        next_idx += 1;
    }

    // Drain completions, refill pipeline
    while let Some(join_result) = futures.next().await {
        match join_result {
            Ok(Some(tenant_id)) => {
                processed_tenants.insert(tenant_id);
            }
            Ok(None) => {}
            Err(e) => {
                error!("Item processing task panicked: {}", e);
            }
        }

        // Before spawning next item, check cancellation and memory pressure
        if next_idx < total {
            if cancellation_token.is_cancelled() {
                warn!(
                    "Shutdown requested, re-leasing {} unstarted items",
                    total - next_idx
                );
                re_lease_remaining_items(&ctx.queue_manager, &items, next_idx).await;
                // Drain in-flight futures before returning
                while let Some(jr) = futures.next().await {
                    if let Ok(Some(tid)) = jr {
                        processed_tenants.insert(tid);
                    }
                }
                return Err(());
            }

            if check_memory_pressure(config).await {
                re_lease_remaining_items(&ctx.queue_manager, &items, next_idx).await;
                // Let in-flight items finish
                while let Some(jr) = futures.next().await {
                    if let Ok(Some(tid)) = jr {
                        processed_tenants.insert(tid);
                    }
                }
                tokio::time::sleep(Duration::from_secs(10)).await;
                return Ok(processed_tenants);
            }

            let item = items[next_idx].clone();
            let batch_ctx = Arc::clone(&ctx);
            futures.push(tokio::spawn(async move {
                process_single_item(item, &batch_ctx).await
            }));
            next_idx += 1;
        }
    }

    if concurrency > 1 {
        debug!(
            "Batch of {} items processed with concurrency={}",
            total, concurrency
        );
    }

    Ok(processed_tenants)
}

/// Process a single queue item: run the handler, then record success or failure.
/// Returns `Some(tenant_id)` on success for activity tracking.
async fn process_single_item(item: UnifiedQueueItem, ctx: &BatchContext) -> Option<String> {
    let start_time = std::time::Instant::now();
    let item_type_str = format!("{:?}", item.item_type);
    let item_timeout = Duration::from_secs(ctx.config.lease_duration_secs as u64);

    let result = tokio::time::timeout(
        item_timeout,
        UnifiedQueueProcessor::process_item(
            &ctx.queue_manager,
            &item,
            &ctx.config,
            &ctx.document_processor,
            &ctx.embedding_generator,
            &ctx.storage_client,
            &ctx.lsp_manager,
            &ctx.embedding_semaphore,
            &ctx.allowed_extensions,
            &ctx.lexicon_manager,
            &ctx.search_db,
            &ctx.graph_store,
            &ctx.grammar_manager,
            &ctx.ingestion_limits,
            &ctx.keyword_embedding_generator,
            &ctx.tier2_tagger,
            &ctx.concept_config,
        ),
    )
    .await;

    match result {
        Ok(Ok(())) => {
            handle_item_success(&item, start_time, ctx, &item_type_str).await;
            Some(item.tenant_id.clone())
        }
        Ok(Err(e)) => {
            handle_item_failure(&item, e, start_time, ctx).await;
            None
        }
        Err(_elapsed) => {
            warn!(
                queue_id = %item.queue_id,
                file_path = ?item.file_path,
                elapsed_secs = start_time.elapsed().as_secs(),
                "Item processing timed out (lease_duration={}s), failing item",
                ctx.config.lease_duration_secs
            );
            handle_item_failure(
                &item,
                UnifiedProcessorError::ProcessingFailed(format!(
                    "Processing timed out after {}s",
                    ctx.config.lease_duration_secs
                )),
                start_time,
                ctx,
            )
            .await;
            None
        }
    }
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

async fn handle_item_success(
    item: &UnifiedQueueItem,
    start_time: std::time::Instant,
    ctx: &BatchContext,
    item_type_str: &str,
) {
    let processing_time = start_time.elapsed().as_millis() as u64;
    METRICS.unified_queue_item_processed(
        &item.item_type.to_string(),
        &item.op.to_string(),
        "success",
        start_time.elapsed().as_secs_f64(),
    );

    let _ = ctx
        .queue_manager
        .mark_explicit_destination_results(&item.queue_id)
        .await;

    let overall = match ctx.queue_manager.check_and_finalize(&item.queue_id).await {
        Ok(status) => status,
        Err(e) => {
            error!(
                "check_and_finalize failed for item {}, keeping in queue for retry: {}",
                item.queue_id, e
            );
            UnifiedQueueProcessor::update_metrics_success(
                &ctx.metrics,
                item_type_str,
                processing_time,
            )
            .await;
            if let Some(ref h) = ctx.queue_health {
                h.record_success(processing_time);
                h.record_heartbeat();
            }
            return;
        }
    };

    match overall {
        QueueStatus::Done => {
            if let Err(e) = ctx.queue_manager.delete_unified_item(&item.queue_id).await {
                error!("Failed to delete item {} from queue: {}", item.queue_id, e);
            }
        }
        QueueStatus::Failed => {
            let err_msg = format!(
                "destination failure on success path (qdrant_status/search_status reported failed) for item={} type={:?}",
                item.queue_id, item.item_type
            );
            if let Err(mark_err) = ctx
                .queue_manager
                .mark_unified_failed(&item.queue_id, &err_msg, false, ctx.config.max_retries)
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

    UnifiedQueueProcessor::update_metrics_success(&ctx.metrics, item_type_str, processing_time)
        .await;

    if let Some(ref h) = ctx.queue_health {
        h.record_success(processing_time);
        h.record_heartbeat();
    }

    info!(
        "Successfully processed unified item {} (type={:?}, op={:?}) in {}ms",
        item.queue_id, item.item_type, item.op, processing_time
    );

    if item.item_type == ItemType::Tenant && item.op == QueueOperation::Add {
        if let Some(ref signal) = ctx.watch_refresh_signal {
            signal.notify_one();
            debug!(
                "Signaled WatchManager refresh after Tenant/Add for {}",
                item.tenant_id
            );
        }
    }
}

async fn handle_item_failure(
    item: &UnifiedQueueItem,
    e: UnifiedProcessorError,
    start_time: std::time::Instant,
    ctx: &BatchContext,
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
        if let Err(del_err) = ctx.queue_manager.delete_unified_item(&item.queue_id).await {
            error!("Failed to delete gone item {}: {}", item.queue_id, del_err);
        }
    } else if error_category == "subsystem_unavailable" || error_category == "rate_limit" {
        debug!("Item {} parked: {} ({})", item.queue_id, error_category, e);
        if let Err(rel_err) = ctx.queue_manager.re_lease_item(&item.queue_id, 60).await {
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
        let should_dlq = is_permanent || item.retry_count + 1 >= ctx.config.max_retries;

        if should_dlq {
            if let Err(mark_err) = ctx
                .queue_manager
                .mark_unified_failed(
                    &item.queue_id,
                    &categorized_msg,
                    true,
                    ctx.config.max_retries,
                )
                .await
            {
                error!(
                    "Failed to mark item {} as failed: {}",
                    item.queue_id, mark_err
                );
            }
            match ctx.queue_manager.move_to_dlq(&item.queue_id).await {
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
            if let Err(mark_err) = ctx
                .queue_manager
                .mark_unified_failed(
                    &item.queue_id,
                    &categorized_msg,
                    false,
                    ctx.config.max_retries,
                )
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

    UnifiedQueueProcessor::update_metrics_failure(&ctx.metrics, &e).await;
    if let Some(ref h) = ctx.queue_health {
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
