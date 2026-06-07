//! Free-function helpers for file_watcher_ops event processing.

use std::sync::Arc;

use notify::EventKind;
use tokio::sync::{Mutex, RwLock};
use tracing::error;

use crate::allowed_extensions::{AllowedExtensions, FileRoute};
use crate::patterns::exclusion::should_exclude_file_in_root;
use crate::queue_operations::QueueError;

use super::error_state::WatchErrorTracker;
use super::error_types::ErrorCategory;
use super::types::{CompiledPatterns, FileEvent, WatchConfig, WatchType};

pub(super) async fn should_filter_debounced_event(
    event: &FileEvent,
    config: &Arc<RwLock<WatchConfig>>,
    allowed_extensions: &Arc<AllowedExtensions>,
    patterns: &Arc<RwLock<CompiledPatterns>>,
) -> bool {
    if !matches!(event.event_kind, EventKind::Remove(_)) {
        let (watch_root, collection_for_check) = {
            let config_lock = config.read().await;
            (
                config_lock.path.to_string_lossy().to_string(),
                match config_lock.watch_type {
                    WatchType::Library => "libraries",
                    WatchType::Project => "projects",
                }
                .to_string(),
            )
        };
        // Root-anchored check (#97): hidden components above the watch root
        // (e.g. `.config` in `~/.config/...`) must not exclude.
        if should_exclude_file_in_root(&event.path.to_string_lossy(), &watch_root) {
            return true;
        }
        if matches!(
            allowed_extensions.route_file(&event.path.to_string_lossy(), &collection_for_check, ""),
            FileRoute::Excluded
        ) {
            return true;
        }
    }
    let patterns_lock = patterns.read().await;
    !patterns_lock.should_process(&event.path)
}

pub(super) async fn handle_enqueue_error(
    e: &QueueError,
    error_tracker: &Arc<WatchErrorTracker>,
    watch_id: &str,
    attempt: u32,
    max_retries: u32,
    queue_errors: &Arc<Mutex<u64>>,
) {
    let mut count = queue_errors.lock().await;
    *count += 1;
    let error_msg = e.to_string();
    let error_category = ErrorCategory::categorize_str(&error_msg);
    if attempt == max_retries - 1 || error_category == ErrorCategory::Permanent {
        let backoff_delay = error_tracker.record_error(watch_id, &error_msg).await;
        let health_status = error_tracker.get_health_status(watch_id).await;
        error!(
            "Failed to enqueue: {} (attempt {}/{}, category={}, health={}, backoff_ms={})",
            e,
            attempt + 1,
            max_retries,
            error_category.as_str(),
            health_status.as_str(),
            backoff_delay
        );
    } else {
        error!(
            "Failed to enqueue: {} (attempt {}/{})",
            e,
            attempt + 1,
            max_retries
        );
    }
}

pub(super) async fn record_retry_exhaustion(
    error_tracker: &Arc<WatchErrorTracker>,
    watch_id: &str,
    queue_errors: &Arc<Mutex<u64>>,
    max_retries: u32,
) {
    let error_msg = format!("Failed after {} retries", max_retries);
    let _backoff_delay = error_tracker.record_error(watch_id, &error_msg).await;
    let health_status = error_tracker.get_health_status(watch_id).await;
    let mut count = queue_errors.lock().await;
    *count += 1;
    error!(
        "Enqueue failed after {} retries (watch health: {})",
        max_retries,
        health_status.as_str()
    );
}
