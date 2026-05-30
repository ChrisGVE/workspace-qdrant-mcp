//! Tree-sitter grammar availability management for file ingestion.
//!
//! Handles dynamic grammar loading (fast/medium/slow paths) and background
//! grammar download with capability-upgrade triggering.

use std::path::Path;
use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::patterns::GitattributesOverrides;
use crate::tree_sitter::{detect_language_with_overrides, parser::LanguageProvider};

/// Ensure the tree-sitter grammar for this file's language is available.
///
/// **Non-blocking**: If the grammar is already loaded or cached on disk, returns
/// a provider immediately. If the grammar needs downloading, spawns a background
/// task and returns `None` so the file gets text-chunked as a fallback. When the
/// download completes, affected files are re-queued for semantic re-processing
/// via the capability upgrade mechanism (File→Uplift).
///
/// **Static-first**: Languages with statically compiled grammars (Python, Rust,
/// TypeScript, etc.) return `None` so the chunker uses the built-in grammar.
/// Dynamic providers are only used for languages that lack static support.
/// This avoids a memory leak in tree-sitter when mixing dynamic grammar `.dylib`
/// loading with the parser's internal allocator.
pub(super) async fn ensure_grammar_available(
    ctx: &ProcessingContext,
    file_path: &Path,
    relative_path: &str,
    overrides: &GitattributesOverrides,
) -> Option<Arc<dyn LanguageProvider>> {
    let grammar_mgr = ctx.grammar_manager.as_ref()?;
    let language = detect_language_with_overrides(file_path, relative_path, overrides)?;

    // If the language has a *statically compiled* grammar, don't use the dynamic
    // provider — the static path is faster and avoids ABI mismatches.
    //
    // BUG FIX: this previously checked `is_language_supported(language)`, which
    // returns true for any language KNOWN TO THE DOWNLOADABLE REGISTRY (the 44
    // bundled languages) — NOT for languages with a statically compiled grammar.
    // In the v0.1.3 dynamic-grammar architecture the static set is empty
    // (`get_static_language` always returns None, `SUPPORTED_LANGUAGES = &[]`),
    // so the old check short-circuited to `return None` for EVERY registry
    // language. The chunker then ran with no static grammar AND no dynamic
    // provider → parse failed → silent text fallback for ALL code, and the
    // dynamic download below became unreachable dead code. Result: the
    // code-relationship graph never received symbol nodes/edges.
    //
    // Check the real static predicate instead. It is always false today, so we
    // always proceed to dynamic loading/download — which is the intended path.
    if crate::tree_sitter::parser::get_static_language(language).is_some() {
        return None;
    }

    // Fast path: read lock to check if grammar is already loaded
    {
        let mgr = grammar_mgr.read().await;
        if mgr.is_loaded(language) {
            let provider = mgr.create_language_provider();
            if !provider.is_empty() {
                return Some(Arc::new(provider));
            }
        }
    }

    // Medium path: grammar is cached on disk but not loaded into memory.
    // Acquire write lock briefly to load from disk (~40ms).
    {
        use crate::tree_sitter::GrammarStatus;
        let status = grammar_mgr.read().await.grammar_status(language);
        if matches!(
            status,
            GrammarStatus::Cached | GrammarStatus::IncompatibleVersion
        ) {
            let mut mgr = grammar_mgr.write().await;
            if let Err(e) = mgr.get_grammar(language).await {
                warn!(
                    language = language,
                    "Failed to load cached grammar for {}: {}",
                    file_path.display(),
                    e
                );
            }
            let provider = mgr.create_language_provider();
            if !provider.is_empty() {
                return Some(Arc::new(provider));
            }
        }
    }

    // Slow path: grammar needs downloading. Spawn background task instead of
    // blocking the queue processor.
    spawn_background_grammar_download(ctx, language).await;

    // Return None — caller will text-chunk this file as fallback.
    // The background download will trigger File→Uplift when complete.
    None
}

/// Spawn a background grammar download task if one isn't already in flight.
///
/// When the download completes, triggers a capability upgrade sweep to
/// re-process files that were text-chunked due to the missing grammar.
async fn spawn_background_grammar_download(ctx: &ProcessingContext, language: &'static str) {
    let grammar_mgr = match ctx.grammar_manager.as_ref() {
        Some(gm) => gm.clone(),
        None => return,
    };

    // Check and insert into pending set atomically
    {
        let mut pending = ctx.pending_grammar_downloads.lock().await;
        if pending.contains(language) {
            debug!(language = language, "Grammar download already in progress");
            return;
        }
        pending.insert(language.to_string());
    }

    info!(
        language = language,
        "Spawning background grammar download (files will be re-processed after)"
    );

    let pending_downloads = ctx.pending_grammar_downloads.clone();
    let pool = ctx.pool.clone();
    let queue_manager = ctx.queue_manager.clone();

    tokio::spawn(async move {
        // Download the grammar (acquires write lock only during download+insert)
        let download_ok = {
            let mut mgr = grammar_mgr.write().await;
            mgr.get_grammar(language).await.is_ok()
        };

        // Remove from pending set regardless of outcome
        {
            let mut pending = pending_downloads.lock().await;
            pending.remove(language);
        }

        if download_ok {
            info!(
                language = language,
                "Background grammar download complete — triggering capability upgrade"
            );

            // Trigger File→Uplift for files of this language that were text-chunked
            // Query all tenants that have files with treesitter_status = 'none'/'skipped'
            use crate::tracked_files_schema::UpgradeReason;
            let tenants = get_distinct_tenants(&pool).await;
            for tenant_id in &tenants {
                crate::strategies::capability_upgrade::trigger_capability_upgrade(
                    &pool,
                    &queue_manager,
                    tenant_id,
                    UpgradeReason::GrammarAvailable,
                    Some(language),
                )
                .await;
            }
        } else {
            warn!(language = language, "Background grammar download failed");
        }
    });
}

/// Get distinct tenant IDs from watch_folders for capability upgrade triggering.
async fn get_distinct_tenants(pool: &SqlitePool) -> Vec<String> {
    use sqlx::Row;
    sqlx::query("SELECT DISTINCT tenant_id FROM watch_folders WHERE is_active > 0")
        .fetch_all(pool)
        .await
        .unwrap_or_default()
        .iter()
        .map(|r| r.get::<String, _>("tenant_id"))
        .collect()
}
