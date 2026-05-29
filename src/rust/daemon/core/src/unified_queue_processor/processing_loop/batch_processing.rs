//! Batch processing for the unified queue processor.
//!
//! `process_batch` dispatches a non-empty batch of queue items either
//! sequentially (when `config.max_concurrent_items == 1`) or concurrently via
//! `FuturesUnordered` (when `> 1`). With `=1` the behavior is byte-identical
//! to the legacy sequential loop. It returns `Err(())` when a cancellation
//! signal is detected mid-batch, signalling the caller to `return Ok(())`
//! from the processing loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
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

use super::concurrent_dispatch::run_dispatch_loop;

/// Owned dependency bundle for a single spawned item future.
///
/// Constructed once per dispatch and cloned per spawn so that each future
/// holds its own `'static` references (Arc-clones are cheap). Required
/// because `FuturesUnordered` requires `'static` futures.
#[derive(Clone)]
struct ItemDeps {
    queue_manager: QueueManager,
    config: UnifiedProcessorConfig,
    document_processor: Arc<DocumentProcessor>,
    embedding_generator: Arc<EmbeddingGenerator>,
    storage_client: Arc<StorageClient>,
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    embedding_semaphore: Arc<tokio::sync::Semaphore>,
    allowed_extensions: Arc<AllowedExtensions>,
    lexicon_manager: Arc<LexiconManager>,
    search_db: Option<Arc<SearchDbManager>>,
    graph_store: Option<crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>>,
    watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,
    grammar_manager: Option<Arc<RwLock<GrammarManager>>>,
    ingestion_limits: Arc<IngestionLimitsConfig>,
    metrics: Arc<RwLock<UnifiedProcessingMetrics>>,
    queue_health: Option<Arc<QueueProcessorHealth>>,
    processed_tenants: Arc<Mutex<HashSet<String>>>,
}

/// Process a non-empty batch of queue items.
///
/// With `config.max_concurrent_items == 1` (the default) the dispatch is
/// strictly sequential and byte-identical to the legacy loop. With values
/// `> 1`, items are dispatched via `FuturesUnordered` and capped by an
/// owned semaphore.
///
/// Returns `Err(())` if a shutdown cancellation is detected during processing
/// (the caller should then `return Ok(())`). Returns `Ok(processed_tenants)`
/// otherwise.
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
    let processed_tenants: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let max_concurrent = config.max_concurrent_items.max(1);
    let item_semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));

    let deps = ItemDeps {
        queue_manager: queue_manager.clone(),
        config: config.clone(),
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
        processed_tenants: Arc::clone(&processed_tenants),
    };

    let deps_for_spawn = deps.clone();
    let cancelled = run_dispatch_loop(
        items,
        Arc::clone(&item_semaphore),
        queue_manager,
        cancellation_token,
        |item, permit| {
            let d = deps_for_spawn.clone();
            tokio::spawn(process_one_item_owned(item, permit, d))
        },
        || async { UnifiedQueueProcessor::check_memory_pressure(config.max_memory_percent).await },
        || async {
            apply_inter_dispatch_delay(config, warmup_state, resource_profile_rx).await;
        },
        config.max_memory_percent,
    )
    .await;

    if cancelled {
        return Err(());
    }

    // Spawned tasks have all completed by this point. The outer `deps` and
    // the spawn closure's captured `deps_for_spawn` still hold Arc clones to
    // `processed_tenants`, so we read it through the mutex rather than
    // try_unwrap (which would always fall through to the lock path anyway).
    let drained = processed_tenants.lock().await.clone();
    Ok(drained)
}

/// Drive a single queue item through dispatch and finalize success/failure.
/// The dispatch permit is dropped automatically when this future returns.
///
/// Cancellation policy: we deliberately do NOT abort `process_item` mid-
/// flight on `cancellation_token.cancelled()`. In-flight items run to
/// completion to avoid half-applied SQLite state (some destination markers
/// committed, others rolled back). The outer dispatch loop reacts to the
/// cancel signal by halting new dispatches and re-leasing the pending
/// remainder; this future's job is to take its single item across the
/// finish line so the queue row settles to a terminal state.
async fn process_one_item_owned(
    item: UnifiedQueueItem,
    _permit: tokio::sync::OwnedSemaphorePermit,
    deps: ItemDeps,
) {
    let start_time = std::time::Instant::now();
    let item_type_str = format!("{:?}", item.item_type);

    let result = UnifiedQueueProcessor::process_item(
        &deps.queue_manager,
        &item,
        &deps.config,
        &deps.document_processor,
        &deps.embedding_generator,
        &deps.storage_client,
        &deps.lsp_manager,
        &deps.embedding_semaphore,
        &deps.allowed_extensions,
        &deps.lexicon_manager,
        &deps.search_db,
        &deps.graph_store,
        &deps.grammar_manager,
        &deps.ingestion_limits,
    )
    .await;

    match result {
        Ok(()) => {
            handle_item_success(&item, start_time, &deps, &item_type_str).await;
        }
        Err(e) => {
            handle_item_failure(&item, e, start_time, &deps).await;
        }
    }
}

/// Apply inter-dispatch delay based on warmup state, adaptive profile, or config default.
///
/// With `max_concurrent_items=1` this acts as a between-items pause exactly
/// like the pre-refactor loop. With larger values it acts as a between-
/// completion pause — backpressure on dispatch rate rather than per-item.
async fn apply_inter_dispatch_delay(
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
async fn handle_item_success(
    item: &UnifiedQueueItem,
    start_time: std::time::Instant,
    deps: &ItemDeps,
    item_type_str: &str,
) {
    let processing_time = start_time.elapsed().as_millis() as u64;
    METRICS.unified_queue_item_processed(
        &item.item_type.to_string(),
        &item.op.to_string(),
        "success",
        start_time.elapsed().as_secs_f64(),
    );

    // Per-destination state machine (F-009, F-010, F-056): resolve destination
    // statuses and overall status atomically. `finalize_after_success` runs the
    // auto-resolve UPDATE (orchestration-only items, decision_json IS NULL) and
    // the finalize SELECT+UPDATE inside one transaction, so the finalize read
    // cannot observe a concurrent mid-flight `failed` write committed between
    // them. Items that opted into the state machine via store_queue_decision
    // still keep pending sinks pending (decision_json IS NOT NULL), so the
    // helper returns InProgress and the item remains in the queue for the next
    // lease cycle.
    let overall = deps
        .queue_manager
        .finalize_after_success(&item.queue_id)
        .await
        .unwrap_or(QueueStatus::Done);
    match overall {
        QueueStatus::Done => {
            if let Err(e) = deps.queue_manager.delete_unified_item(&item.queue_id).await {
                error!("Failed to delete item {} from queue: {}", item.queue_id, e);
            }
        }
        QueueStatus::Failed => {
            // F-033/F-034: a destination explicitly reported failure even though
            // the handler returned Ok. Promote to mark_unified_failed so retry
            // metadata (retry_count, error_message, backoff via lease_until)
            // is populated; without this, the row would stay `failed` with no
            // way to surface or schedule a retry.
            //
            // Read back the actual destination statuses so the persisted
            // error_message names which sink failed — without this, every
            // failure looks identical and you have to grep daemon logs to
            // know whether qdrant or search (FTS5) was the culprit.
            let (qs, ss) = deps
                .queue_manager
                .read_destination_statuses(&item.queue_id)
                .await
                .unwrap_or((None, None));
            let err_msg = format!(
                "destination failure on success path for item={} type={:?} \
                 (qdrant_status={:?}, search_status={:?})",
                item.queue_id, item.item_type, qs, ss
            );
            if let Err(mark_err) = deps
                .queue_manager
                .mark_unified_failed(&item.queue_id, &err_msg, false, deps.config.max_retries)
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

    UnifiedQueueProcessor::update_metrics_success(&deps.metrics, item_type_str, processing_time)
        .await;

    if let Some(ref h) = deps.queue_health {
        h.record_success(processing_time);
        h.record_heartbeat();
    }

    deps.processed_tenants
        .lock()
        .await
        .insert(item.tenant_id.clone());

    info!(
        "Successfully processed unified item {} (type={:?}, op={:?}) in {}ms",
        item.queue_id, item.item_type, item.op, processing_time
    );

    // Task 12: Signal WatchManager to refresh after Tenant/Add.
    if item.item_type == ItemType::Tenant && item.op == QueueOperation::Add {
        if let Some(ref signal) = deps.watch_refresh_signal {
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
    deps: &ItemDeps,
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
        if let Err(del_err) = deps.queue_manager.delete_unified_item(&item.queue_id).await {
            error!("Failed to delete gone item {}: {}", item.queue_id, del_err);
        }
    } else if error_category == "subsystem_unavailable" {
        debug!(
            "Item {} parked: embedding subsystem unavailable ({})",
            item.queue_id, e
        );
        if let Err(rel_err) = deps.queue_manager.re_lease_item(&item.queue_id, 60).await {
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
        if let Err(mark_err) = deps
            .queue_manager
            .mark_unified_failed(
                &item.queue_id,
                &categorized_msg,
                is_permanent,
                deps.config.max_retries,
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

    UnifiedQueueProcessor::update_metrics_failure(&deps.metrics, &e).await;
    if let Some(ref h) = deps.queue_health {
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
