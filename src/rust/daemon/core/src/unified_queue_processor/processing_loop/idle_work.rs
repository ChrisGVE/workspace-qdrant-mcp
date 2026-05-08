//! Idle-time maintenance work for the unified queue processor.
//!
//! `run_idle_work` is called when the fairness scheduler returns an empty batch.
//! It runs uplift, maintenance scheduling, grammar checks, LSP eviction,
//! resurrection, and triage, then sleeps for `poll_interval`.
//!
//! Returns `true` — the caller should always `continue` after this returns.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tree_sitter::GrammarManager;

use crate::unified_queue_processor::config::UnifiedProcessorConfig;
use crate::unified_queue_processor::UnifiedQueueProcessor;

use super::loop_state::LoopState;

/// Run all idle-time maintenance. Always returns `true` so the caller can
/// unconditionally `continue` the outer loop.
pub(super) async fn run_idle_work(
    state: &mut LoopState,
    config: &UnifiedProcessorConfig,
    queue_manager: &QueueManager,
    storage_client: &Arc<StorageClient>,
    lexicon_manager: &Arc<LexiconManager>,
    grammar_manager: &Option<Arc<RwLock<GrammarManager>>>,
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    search_db: &Option<Arc<SearchDbManager>>,
    poll_interval: Duration,
) -> bool {
    // Track continuous idle time for grammar update checks
    if state.idle_since.is_none() {
        state.idle_since = Some(std::time::Instant::now());
    }

    // Run metadata uplift when queue is idle (Task 18)
    let since_last = state.last_uplift_attempt.elapsed().as_secs();
    if since_last >= state.uplift_config.min_interval_secs {
        debug!(
            "Queue idle — running metadata uplift pass (gen={})",
            state.uplift_config.current_generation
        );
        let collections = vec!["projects".to_string(), "libraries".to_string()];
        let stats = crate::metadata_uplift::run_uplift_pass(
            storage_client,
            lexicon_manager,
            &collections,
            &state.uplift_config,
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
            state.uplift_config.current_generation += 1;
        }
        state.last_uplift_attempt = std::time::Instant::now();
    }

    let idle_elapsed = state.idle_since.map_or(0, |t| t.elapsed().as_secs());

    // Maintenance scheduler — run one batch if eligible
    {
        let qdrant_available = storage_client.is_qdrant_available();
        let memory_pressure =
            UnifiedQueueProcessor::check_memory_pressure(config.max_memory_percent).await;
        let idle_state = crate::idle::IdleState::determine(
            0, // queue is empty (we're in the empty branch)
            qdrant_available,
            memory_pressure,
        );
        if idle_state.allows_maintenance() {
            let maint_ctx = crate::idle::MaintenanceContext {
                pool: queue_manager.pool(),
                storage_client,
                search_db: search_db.as_ref(),
                queue_manager,
            };
            let _ = state
                .maintenance_scheduler
                .tick(idle_state, idle_elapsed, &maint_ctx)
                .await;
        }
    }

    // Run grammar update check when idle long enough (Task 5)
    if let Some(ref gm) = grammar_manager {
        run_grammar_idle_work(state, gm, idle_elapsed).await;
    }

    // Evict idle LSP servers to bound resource usage
    if let Some(ref lsm) = lsp_manager {
        let lsm_read = lsm.read().await;
        let lsp_timeout = lsm_read.config.idle_timeout_secs;
        drop(lsm_read);
        if lsp_timeout > 0 && idle_elapsed >= lsp_timeout {
            let timeout = Duration::from_secs(lsp_timeout);
            let lsm_read = lsm.read().await;
            let evicted = lsm_read.evict_idle_servers(timeout).await;
            if !evicted.is_empty() {
                info!(
                    "Evicted {} idle LSP servers: {}",
                    evicted.len(),
                    evicted
                        .iter()
                        .map(|(p, l)| format!("{}:{:?}", p, l))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }

    // Periodic resurrection of transient failed items
    let resurrection_interval_secs = config.failed_resurrection_interval_secs;
    if resurrection_interval_secs > 0
        && state.last_resurrection.elapsed().as_secs() >= resurrection_interval_secs
    {
        match queue_manager
            .resurrect_failed_transient(config.max_resurrections)
            .await
        {
            Ok((resurrected, exhausted)) => {
                if resurrected > 0 || exhausted > 0 {
                    info!(
                        "Resurrection pass: reset {} item(s), exhausted {} item(s)",
                        resurrected, exhausted
                    );
                }
            }
            Err(e) => {
                warn!("Resurrection pass failed (non-fatal): {}", e);
            }
        }
        state.last_resurrection = std::time::Instant::now();
    }

    // Failed item triage
    let triage_interval_secs = config.triage_interval_secs;
    if triage_interval_secs > 0 && state.last_triage.elapsed().as_secs() >= triage_interval_secs {
        if let Err(e) = queue_manager.triage_failed_items().await {
            warn!("Triage pass failed (non-fatal): {}", e);
        }
        state.last_triage = std::time::Instant::now();
    }

    debug!(
        "Unified queue is empty, waiting {}ms",
        poll_interval.as_millis()
    );
    tokio::time::sleep(poll_interval).await;
    true
}

/// Grammar-specific idle work: periodic version check and idle eviction.
async fn run_grammar_idle_work(
    state: &mut LoopState,
    gm: &Arc<RwLock<GrammarManager>>,
    idle_elapsed: u64,
) {
    let gm_read = gm.read().await;
    let cfg = gm_read.config();
    let check_enabled = cfg.idle_update_check_enabled;
    let delay = cfg.idle_update_check_delay_secs;
    let needs_check = gm_read.needs_periodic_check();
    drop(gm_read);

    if check_enabled
        && idle_elapsed >= delay
        && needs_check
        && state.last_grammar_check.elapsed().as_secs() >= delay
    {
        info!(
            "Queue idle for {}s (threshold: {}s) — running grammar update check",
            idle_elapsed, delay
        );
        let mut gm_write = gm.write().await;
        let results = gm_write.periodic_version_check().await;
        drop(gm_write);

        let checked = results.len();
        let updated = results.values().filter(|r| r.is_ok()).count();
        let errors = results.values().filter(|r| r.is_err()).count();
        if checked > 0 {
            info!("Grammar update check: {checked} checked, {updated} up-to-date, {errors} errors");
        }
        state.last_grammar_check = std::time::Instant::now();
    }

    // Evict idle grammars to bound memory usage
    let timeout_secs = gm.read().await.config().grammar_idle_timeout_secs;
    if timeout_secs > 0 && idle_elapsed >= timeout_secs {
        let timeout = Duration::from_secs(timeout_secs);
        let mut gm_write = gm.write().await;
        let evicted = gm_write.evict_idle_grammars(timeout);
        if !evicted.is_empty() {
            info!(
                "Evicted {} idle grammars: {}",
                evicted.len(),
                evicted.join(", ")
            );
        }
    }
}
