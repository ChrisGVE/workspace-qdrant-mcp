//! FileWatcherQueue event processing operations - loop, filtering, enqueuing.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use notify::EventKind;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::allowed_extensions::{AllowedExtensions, FileRoute};
use crate::file_classification::classify_file_type;
use crate::patterns::exclusion::should_exclude_file;
use crate::patterns::global_ignore::is_globally_ignored;
use crate::queue_operations::{QueueError, QueueManager};
use crate::tracked_files_schema;
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation as UnifiedOp};

use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::paths::{CanonicalPath, RelativePath};

use super::error_state::WatchErrorTracker;
use super::file_watcher::FileWatcherQueue;
use super::file_watcher_ops_helpers::{
    handle_enqueue_error, record_retry_exhaustion, should_filter_debounced_event,
};
use super::throttle::QueueThrottleState;
use super::types::{
    get_current_branch, CompiledPatterns, EventDebouncer, FileEvent, WatchConfig, WatchType,
};

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
        {
            let mut count = events_received.lock().await;
            *count += 1;
        }

        // Intercept .gitignore / .wqmignore changes -> trigger reconciliation
        if Self::handle_ignore_file_if_applicable(&event, config, queue_manager, events_processed)
            .await
        {
            return;
        }

        // Apply pre-enqueue filters (exclusion, allowlist, patterns)
        if Self::should_filter_event(
            &event,
            config,
            allowed_extensions,
            patterns,
            events_filtered,
        )
        .await
        {
            return;
        }

        // Add to debouncer
        let should_process = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.add_event(event.clone())
        };
        if should_process {
            Self::enqueue_file_operation(
                event,
                config,
                queue_manager,
                allowed_extensions,
                error_tracker,
                throttle_state,
                events_processed,
                queue_errors,
                events_throttled,
            )
            .await;
        }
    }

    /// Check if the event targets a .gitignore / .wqmignore file and handle it.
    /// Returns `true` if the event was consumed (caller should return early).
    async fn handle_ignore_file_if_applicable(
        event: &FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_processed: &Arc<Mutex<u64>>,
    ) -> bool {
        if let Some(name) = event.path.file_name().and_then(|n| n.to_str()) {
            if name == ".gitignore" || name == ".wqmignore" {
                if matches!(
                    event.event_kind,
                    EventKind::Create(_) | EventKind::Modify(_)
                ) {
                    Self::handle_ignore_file_change(
                        &event.path,
                        config,
                        queue_manager,
                        events_processed,
                    )
                    .await;
                }
                return true; // Don't process ignore files as regular files
            }
        }
        false
    }

    /// Apply exclusion, allowlist, and pattern filters to an event.
    /// Returns `true` if the event should be filtered out (skipped).
    async fn should_filter_event(
        event: &FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        allowed_extensions: &Arc<AllowedExtensions>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        events_filtered: &Arc<Mutex<u64>>,
    ) -> bool {
        // global.wqmignore applies to every event kind (incl. Remove): a
        // globally-ignored path was never indexed, so no watcher op is wanted.
        if is_globally_ignored(&event.path, false) {
            let mut count = events_filtered.lock().await;
            *count += 1;
            return true;
        }

        if !matches!(event.event_kind, EventKind::Remove(_)) {
            let file_path_str = event.path.to_string_lossy();
            if should_exclude_file(&file_path_str) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return true;
            }
        }

        if !matches!(event.event_kind, EventKind::Remove(_)) {
            let collection_for_check = {
                let config_lock = config.read().await;
                match config_lock.watch_type {
                    WatchType::Library => "libraries",
                    WatchType::Project => "projects",
                }
                .to_string()
            };
            let file_path_str = event.path.to_string_lossy();
            if matches!(
                allowed_extensions.route_file(&file_path_str, &collection_for_check, ""),
                FileRoute::Excluded
            ) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return true;
            }
        }

        {
            let patterns_lock = patterns.read().await;
            if !patterns_lock.should_process(&event.path) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return true;
            }
        }

        false
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
            if should_filter_debounced_event(&event, config, allowed_extensions, patterns).await {
                continue;
            }
            Self::enqueue_file_operation(
                event,
                config,
                queue_manager,
                allowed_extensions,
                error_tracker,
                throttle_state,
                events_processed,
                queue_errors,
                events_throttled,
            )
            .await;
        }
    }

    /// Determine operation type based on event and file state
    fn determine_operation_type(event_kind: EventKind, file_path: &Path) -> UnifiedOp {
        match event_kind {
            EventKind::Create(_) => UnifiedOp::Add,
            EventKind::Remove(_) => UnifiedOp::Delete,
            EventKind::Modify(_) => {
                if file_path.exists() {
                    UnifiedOp::Update
                } else {
                    UnifiedOp::Delete
                }
            }
            _ => UnifiedOp::Update,
        }
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

        if should_exclude_file(&event.path.to_string_lossy()) {
            debug!(
                "File excluded by exclusion engine, skipping: {}",
                event.path.display()
            );
            return;
        }

        // Chokepoint enforcement of global.wqmignore. The directory-walk path
        // applies it via the reconciler/folder-scan matcher, but watcher events
        // (notify) reach the queue through here — without this check the daemon
        // re-indexes its own Qdrant storage under <repo>/state/qdrant/ on every
        // segment rotation, a self-sustaining feedback loop. Gates Add, Update,
        // AND Delete: a globally-ignored path is never indexed, so it never
        // needs a watcher-originated delete (the reconciler handles cleanup of
        // anything indexed before the pattern was added).
        if is_globally_ignored(&event.path, false) {
            debug!(
                "File excluded by global.wqmignore, skipping: {}",
                event.path.display()
            );
            return;
        }

        let watch_id = {
            let c = config.read().await;
            c.id.clone()
        };

        if !error_tracker.can_process(&watch_id).await {
            debug!(
                "Watch {} is in backoff or disabled, skipping file: {}",
                watch_id,
                event.path.display()
            );
            return;
        }

        if Self::should_throttle_event(throttle_state, queue_manager, events_throttled, &event)
            .await
        {
            return;
        }

        let operation = Self::determine_operation_type(event.event_kind, &event.path);

        if operation == UnifiedOp::Delete {
            let fp = event.path.to_string_lossy().to_string();
            match tracked_files_schema::is_incremental(queue_manager.pool(), &fp).await {
                Ok(true) => {
                    debug!(
                        "Skipping delete for incremental file: {}",
                        event.path.display()
                    );
                    return;
                }
                Ok(false) => {}
                Err(e) => {
                    debug!(
                        "Failed to check incremental flag for {}: {}",
                        event.path.display(),
                        e
                    );
                }
            }
        }

        let routed = Self::resolve_routing_and_payload(&event, config, allowed_extensions).await;
        let (final_collection, final_tenant, branch, metadata, payload_json) = match routed {
            Some(r) => r,
            None => {
                debug!(
                    "Skipping event whose path could not be anchored to watch root: {}",
                    event.path.display()
                );
                return;
            }
        };

        Self::enqueue_with_retry(
            queue_manager,
            error_tracker,
            &watch_id,
            operation,
            &final_tenant,
            &final_collection,
            &payload_json,
            &branch,
            metadata.as_deref(),
            events_processed,
            queue_errors,
        )
        .await;
    }

    /// Check and apply queue depth throttling; returns true if the event should be skipped.
    async fn should_throttle_event(
        throttle_state: &Arc<QueueThrottleState>,
        queue_manager: &Arc<QueueManager>,
        events_throttled: &Arc<Mutex<u64>>,
        event: &FileEvent,
    ) -> bool {
        if throttle_state.needs_refresh().await {
            throttle_state.update_from_queue(queue_manager).await;
        }
        if throttle_state.should_throttle().await {
            let load_level = throttle_state.get_load_level().await;
            debug!(
                "Throttling event due to {} queue load (depth: {}): {}",
                load_level.as_str(),
                throttle_state.get_depth().await,
                event.path.display()
            );
            let mut count = events_throttled.lock().await;
            *count += 1;
            return true;
        }
        false
    }

    /// Resolve collection/tenant routing and build the enqueue payload.
    ///
    /// Returns `None` when the event path cannot be anchored to the watch
    /// folder root (non-UTF-8, escapes the root, etc.) — the event is then
    /// dropped rather than enqueued with a malformed payload.
    async fn resolve_routing_and_payload(
        event: &FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> Option<(String, String, String, Option<String>, String)> {
        let file_type = classify_file_type(&event.path);
        let (collection, tenant_id, branch, watch_root_str) = {
            let config_lock = config.read().await;
            let coll = match config_lock.watch_type {
                WatchType::Project => wqm_common::constants::COLLECTION_PROJECTS.to_string(),
                WatchType::Library => COLLECTION_LIBRARIES.to_string(),
            };
            (
                coll,
                config_lock.tenant_id.clone(),
                get_current_branch(&config_lock.path),
                config_lock.path.to_string_lossy().to_string(),
            )
        };

        let file_absolute_path = event.path.to_string_lossy().to_string();

        let (final_collection, final_tenant, metadata) = match allowed_extensions.route_file(
            &file_absolute_path,
            &collection,
            &tenant_id,
        ) {
            FileRoute::LibraryCollection { source_project_id }
                if collection != COLLECTION_LIBRARIES =>
            {
                let meta = source_project_id
                    .as_ref()
                    .map(|pid| crate::format_routing::routing_metadata_json(pid));
                let lib_name = source_project_id
                    .as_ref()
                    .map(|pid| crate::format_routing::generate_library_name(pid))
                    .unwrap_or_else(|| tenant_id.clone());
                debug!("Format-based routing override: {} -> libraries (source_project={}, library_name={})",
                        file_absolute_path, tenant_id, lib_name);
                (COLLECTION_LIBRARIES.to_string(), lib_name, meta)
            }
            _ => (collection.clone(), tenant_id.clone(), None),
        };

        debug!(
            "Multi-tenant routing: file={}, collection={}, tenant={}, file_type={}, branch={}",
            file_absolute_path,
            final_collection,
            final_tenant,
            file_type.as_str(),
            branch
        );

        // Anchor the absolute file path to the watch folder root. The
        // watcher exists because the daemon has a watch_folder row, so
        // `config.path` is the canonical root. If anchoring fails the
        // event is dropped — without a relative payload there is no way
        // to address the file in the storage layer.
        let root = match CanonicalPath::from_user_input(&watch_root_str) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Watch root {} failed canonical validation: {} -- dropping event for {}",
                    watch_root_str, e, file_absolute_path
                );
                return None;
            }
        };
        let abs = match CanonicalPath::from_user_input(&file_absolute_path) {
            Ok(a) => a,
            Err(e) => {
                warn!(
                    "File path {} failed canonical validation: {} -- dropping event",
                    file_absolute_path, e
                );
                return None;
            }
        };
        let relative = match RelativePath::from_absolute_and_root(&abs, &root) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "File {} not under watch root {} ({}); dropping event",
                    abs.as_str(),
                    root.as_str(),
                    e
                );
                return None;
            }
        };

        let file_payload = FilePayload {
            file_path: relative,
            file_type: Some(file_type.as_str().to_string()),
            file_hash: None,
            size_bytes: event.path.metadata().ok().map(|m| m.len()),
            old_path: None,
        };
        let payload_json = match serde_json::to_string(&file_payload) {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    "Failed to serialize FilePayload for {}: {}",
                    file_absolute_path, e
                );
                return None;
            }
        };

        Some((
            final_collection,
            final_tenant,
            branch,
            metadata,
            payload_json,
        ))
    }

    /// Execute enqueue with retry logic and error tracking.
    #[allow(clippy::too_many_arguments)]
    async fn enqueue_with_retry(
        queue_manager: &Arc<QueueManager>,
        error_tracker: &Arc<WatchErrorTracker>,
        watch_id: &str,
        operation: UnifiedOp,
        tenant: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata: Option<&str>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAYS_MS: [u64; 3] = [500, 1000, 2000];

        for attempt in 0..MAX_RETRIES {
            match queue_manager
                .enqueue_unified(
                    ItemType::File,
                    operation,
                    tenant,
                    collection,
                    payload_json,
                    Some(branch),
                    metadata,
                )
                .await
            {
                Ok(_) => {
                    let mut count = events_processed.lock().await;
                    *count += 1;
                    error_tracker.record_success(watch_id).await;
                    debug!("Enqueued file to unified_queue (operation={:?}, collection={}, tenant={}, branch={})",
                        operation, collection, tenant, branch);
                    return;
                }
                Err(QueueError::Database(ref e)) if attempt < MAX_RETRIES - 1 => {
                    let delay = RETRY_DELAYS_MS[attempt as usize];
                    warn!(
                        "Database error enqueueing: {}. Retrying in {}ms (attempt {}/{})",
                        e,
                        delay,
                        attempt + 1,
                        MAX_RETRIES
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                Err(ref e) => {
                    handle_enqueue_error(
                        e,
                        error_tracker,
                        watch_id,
                        attempt,
                        MAX_RETRIES,
                        queue_errors,
                    )
                    .await;
                    return;
                }
            }
        }

        record_retry_exhaustion(error_tracker, watch_id, queue_errors, MAX_RETRIES).await;
    }
}
