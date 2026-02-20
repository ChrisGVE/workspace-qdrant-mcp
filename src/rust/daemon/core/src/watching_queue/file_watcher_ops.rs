//! FileWatcherQueue event processing operations - loop, filtering, enqueuing.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::EventKind;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::queue_operations::{QueueError, QueueManager};
use crate::unified_queue_schema::{ItemType, QueueOperation as UnifiedOp, FilePayload};
use crate::file_classification::classify_file_type;
use crate::allowed_extensions::{AllowedExtensions, FileRoute};
use crate::patterns::exclusion::should_exclude_file;
use crate::tracked_files_schema;

use wqm_common::constants::COLLECTION_LIBRARIES;

use super::types::{
    WatchConfig, CompiledPatterns, FileEvent, EventDebouncer, WatchType,
    get_current_branch, determine_collection_and_tenant,
};
use super::error_types::ErrorCategory;
use super::error_state::WatchErrorTracker;
use super::throttle::QueueThrottleState;
use super::file_watcher::FileWatcherQueue;

impl FileWatcherQueue {
    /// Main event processing loop
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn event_processing_loop(
        event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
        debouncer: Arc<Mutex<EventDebouncer>>,
        patterns: Arc<RwLock<CompiledPatterns>>,
        config: Arc<RwLock<WatchConfig>>,
        queue_manager: Arc<QueueManager>,
        allowed_extensions: Arc<AllowedExtensions>,
        error_tracker: Arc<WatchErrorTracker>,
        throttle_state: Arc<QueueThrottleState>,
        events_received: Arc<Mutex<u64>>,
        events_processed: Arc<Mutex<u64>>,
        events_filtered: Arc<Mutex<u64>>,
        queue_errors: Arc<Mutex<u64>>,
        events_throttled: Arc<Mutex<u64>>,
    ) {
        let mut debounce_interval = interval(Duration::from_millis(500));
        loop {
            tokio::select! {
                event = async {
                    let mut receiver_lock = event_receiver.lock().await;
                    if let Some(ref mut receiver) = *receiver_lock {
                        receiver.recv().await
                    } else {
                        None
                    }
                } => {
                    if let Some(event) = event {
                        Self::process_file_event(
                            event, &debouncer, &patterns, &config,
                            &queue_manager, &allowed_extensions,
                            &error_tracker, &throttle_state,
                            &events_received, &events_processed,
                            &events_filtered, &queue_errors,
                            &events_throttled,
                        ).await;
                    } else {
                        break;
                    }
                },
                _ = debounce_interval.tick() => {
                    Self::process_debounced_events(
                        &debouncer, &patterns, &config,
                        &queue_manager, &allowed_extensions,
                        &error_tracker, &throttle_state,
                        &events_processed, &queue_errors,
                        &events_throttled,
                    ).await;
                },
            }
        }
        info!("Event processing loop stopped");
    }

    /// Process a single file event
    #[allow(clippy::too_many_arguments)]
    async fn process_file_event(
        event: FileEvent,
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        error_tracker: &Arc<WatchErrorTracker>,
        throttle_state: &Arc<QueueThrottleState>,
        events_received: &Arc<Mutex<u64>>,
        events_processed: &Arc<Mutex<u64>>,
        events_filtered: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        events_throttled: &Arc<Mutex<u64>>,
    ) {
        { let mut count = events_received.lock().await; *count += 1; }

        // Check exclusion patterns FIRST (Task 518)
        if !matches!(event.event_kind, EventKind::Remove(_)) {
            let file_path_str = event.path.to_string_lossy();
            if should_exclude_file(&file_path_str) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Check allowlist via route_file() before pattern matching (Task 511/567)
        if !matches!(event.event_kind, EventKind::Remove(_)) {
            let collection_for_check = {
                let config_lock = config.read().await;
                match config_lock.watch_type {
                    WatchType::Library => "libraries",
                    WatchType::Project => "projects",
                }.to_string()
            };
            let file_path_str = event.path.to_string_lossy();
            if matches!(allowed_extensions.route_file(&file_path_str, &collection_for_check, ""), FileRoute::Excluded) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Check patterns
        {
            let patterns_lock = patterns.read().await;
            if !patterns_lock.should_process(&event.path) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Add to debouncer
        let should_process = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.add_event(event.clone())
        };
        if should_process {
            Self::enqueue_file_operation(
                event, config, queue_manager, allowed_extensions,
                error_tracker, throttle_state,
                events_processed, queue_errors, events_throttled,
            ).await;
        }
    }

    /// Process debounced events
    #[allow(clippy::too_many_arguments)]
    async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        error_tracker: &Arc<WatchErrorTracker>,
        throttle_state: &Arc<QueueThrottleState>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        events_throttled: &Arc<Mutex<u64>>,
    ) {
        let ready_events = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.get_ready_events()
        };
        for event in ready_events {
            // Check exclusion patterns (Task 518)
            if !matches!(event.event_kind, EventKind::Remove(_)) {
                if should_exclude_file(&event.path.to_string_lossy()) {
                    continue;
                }
            }
            // Check allowlist via route_file() (Task 511/567)
            if !matches!(event.event_kind, EventKind::Remove(_)) {
                let collection_for_check = {
                    let config_lock = config.read().await;
                    match config_lock.watch_type {
                        WatchType::Library => "libraries",
                        WatchType::Project => "projects",
                    }.to_string()
                };
                if matches!(allowed_extensions.route_file(&event.path.to_string_lossy(), &collection_for_check, ""), FileRoute::Excluded) {
                    continue;
                }
            }
            // Double-check patterns
            {
                let patterns_lock = patterns.read().await;
                if !patterns_lock.should_process(&event.path) { continue; }
            }
            Self::enqueue_file_operation(
                event, config, queue_manager, allowed_extensions,
                error_tracker, throttle_state,
                events_processed, queue_errors, events_throttled,
            ).await;
        }
    }

    /// Determine operation type based on event and file state
    fn determine_operation_type(event_kind: EventKind, file_path: &Path) -> UnifiedOp {
        match event_kind {
            EventKind::Create(_) => UnifiedOp::Add,
            EventKind::Remove(_) => UnifiedOp::Delete,
            EventKind::Modify(_) => {
                if file_path.exists() { UnifiedOp::Update } else { UnifiedOp::Delete }
            },
            _ => UnifiedOp::Update,
        }
    }

    /// Find project root by looking for .git directory
    fn find_project_root(file_path: &Path) -> PathBuf {
        let mut current = file_path.parent().unwrap_or(file_path);
        while current != current.parent().unwrap_or(Path::new("/")) {
            if current.join(".git").exists() { return current.to_path_buf(); }
            current = current.parent().unwrap_or(Path::new("/"));
        }
        file_path.parent().unwrap_or(file_path).to_path_buf()
    }

    /// Determine collection and tenant_id based on watch type (delegates to types module)
    pub(super) fn determine_collection_and_tenant(
        watch_type: WatchType, project_root: &Path,
        library_name: Option<&str>, legacy_collection: &str,
    ) -> (String, String) {
        determine_collection_and_tenant(watch_type, project_root, library_name, legacy_collection)
    }

    /// Enqueue file operation with retry logic, multi-tenant routing, error tracking,
    /// and queue depth throttling
    #[allow(clippy::too_many_arguments)]
    async fn enqueue_file_operation(
        event: FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        error_tracker: &Arc<WatchErrorTracker>,
        throttle_state: &Arc<QueueThrottleState>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        events_throttled: &Arc<Mutex<u64>>,
    ) {
        if !event.path.is_file() && !matches!(event.event_kind, EventKind::Remove(_)) {
            return;
        }

        let file_path_str = event.path.to_string_lossy();
        if should_exclude_file(&file_path_str) {
            debug!("File excluded by exclusion engine, skipping: {}", event.path.display());
            return;
        }

        let watch_id = { let c = config.read().await; c.id.clone() };

        // Check if this watch is in backoff or disabled (Task 461.5)
        if !error_tracker.can_process(&watch_id).await {
            debug!("Watch {} is in backoff or disabled, skipping file: {}", watch_id, event.path.display());
            return;
        }

        // Refresh queue depth if needed (Task 461.8)
        if throttle_state.needs_refresh().await {
            throttle_state.update_from_queue(queue_manager).await;
        }
        if throttle_state.should_throttle().await {
            let load_level = throttle_state.get_load_level().await;
            debug!("Throttling event due to {} queue load (depth: {}): {}",
                load_level.as_str(), throttle_state.get_depth().await, event.path.display());
            let mut count = events_throttled.lock().await;
            *count += 1;
            return;
        }

        let operation = Self::determine_operation_type(event.event_kind, &event.path);

        // Check incremental flag: skip deletion of files marked as "do not delete"
        if operation == UnifiedOp::Delete {
            let fp = event.path.to_string_lossy().to_string();
            match tracked_files_schema::is_incremental(queue_manager.pool(), &fp).await {
                Ok(true) => { debug!("Skipping delete for incremental file: {}", event.path.display()); return; }
                Ok(false) => {}
                Err(e) => { debug!("Failed to check incremental flag for {}: {}", event.path.display(), e); }
            }
        }

        let project_root = Self::find_project_root(&event.path);
        let branch = get_current_branch(&project_root);
        let file_type = classify_file_type(&event.path);

        let (collection, tenant_id) = {
            let config_lock = config.read().await;
            Self::determine_collection_and_tenant(
                config_lock.watch_type, &project_root,
                config_lock.library_name.as_deref(), &config_lock.collection,
            )
        };

        let file_absolute_path = event.path.to_string_lossy().to_string();

        // Apply format-based routing (Task 567)
        let (final_collection, metadata) = match allowed_extensions.route_file(&file_absolute_path, &collection, &tenant_id) {
            FileRoute::LibraryCollection { source_project_id } if collection != COLLECTION_LIBRARIES => {
                let meta = source_project_id.as_ref().map(|pid| serde_json::json!({"source_project_id": pid}).to_string());
                debug!("Format-based routing override: {} -> libraries (source_project={})", file_absolute_path, tenant_id);
                (COLLECTION_LIBRARIES.to_string(), meta)
            }
            _ => (collection.clone(), None),
        };

        debug!("Multi-tenant routing: file={}, collection={}, tenant={}, file_type={}, branch={}",
            file_absolute_path, final_collection, tenant_id, file_type.as_str(), branch);

        let file_payload = FilePayload {
            file_path: file_absolute_path.clone(),
            file_type: Some(file_type.as_str().to_string()),
            file_hash: None,
            size_bytes: event.path.metadata().ok().map(|m| m.len()),
            old_path: None,
        };
        let payload_json = serde_json::to_string(&file_payload).unwrap_or_else(|_| "{}".to_string());

        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAYS_MS: [u64; 3] = [500, 1000, 2000];

        for attempt in 0..MAX_RETRIES {
            match queue_manager.enqueue_unified(
                ItemType::File, operation, &tenant_id, &final_collection,
                &payload_json, 0, Some(&branch), metadata.as_deref(),
            ).await {
                Ok(_) => {
                    let mut count = events_processed.lock().await;
                    *count += 1;
                    error_tracker.record_success(&watch_id).await;
                    debug!("Enqueued file to unified_queue: {} (operation={:?}, collection={}, tenant={}, branch={}, file_type={})",
                        file_absolute_path, operation, final_collection, tenant_id, branch, file_type.as_str());
                    return;
                },
                Err(QueueError::Database(ref e)) if attempt < MAX_RETRIES - 1 => {
                    let delay = RETRY_DELAYS_MS[attempt as usize];
                    warn!("Database error enqueueing {}: {}. Retrying in {}ms (attempt {}/{})",
                        file_absolute_path, e, delay, attempt + 1, MAX_RETRIES);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                },
                Err(ref e) => {
                    let mut count = queue_errors.lock().await;
                    *count += 1;
                    let error_msg = e.to_string();
                    let error_category = ErrorCategory::categorize_str(&error_msg);
                    if attempt == MAX_RETRIES - 1 || error_category == ErrorCategory::Permanent {
                        let backoff_delay = error_tracker.record_error(&watch_id, &error_msg).await;
                        let health_status = error_tracker.get_health_status(&watch_id).await;
                        error!("Failed to enqueue file {}: {} (attempt {}/{}, category={}, health={}, backoff_ms={})",
                            file_absolute_path, e, attempt + 1, MAX_RETRIES,
                            error_category.as_str(), health_status.as_str(), backoff_delay);
                    } else {
                        error!("Failed to enqueue file {}: {} (attempt {}/{})",
                            file_absolute_path, e, attempt + 1, MAX_RETRIES);
                    }
                    return;
                }
            }
        }

        // All retries failed
        let error_msg = format!("Failed after {} retries", MAX_RETRIES);
        let _backoff_delay = error_tracker.record_error(&watch_id, &error_msg).await;
        let health_status = error_tracker.get_health_status(&watch_id).await;
        let mut count = queue_errors.lock().await;
        *count += 1;
        error!("Failed to enqueue file {} after {} retries (watch health: {})",
            file_absolute_path, MAX_RETRIES, health_status.as_str());
    }
}
