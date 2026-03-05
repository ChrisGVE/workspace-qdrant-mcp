//! Event processing methods for FileWatcher (static associated functions)
//!
//! These methods are separated from watcher.rs to keep file sizes manageable.
//! They are all static (don't take `&self`) and operate on Arc-wrapped state.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use notify::{Event, EventKind};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::interval;

use crate::processing::{TaskPayload, TaskPriority, TaskSource, TaskSubmitter};

use super::compiled_patterns::CompiledPatterns;
use super::config::WatcherConfig;
use super::debouncer::{EventBatcher, EventDebouncer};
use super::events::{FileEvent, PausedEventBuffer};
use super::telemetry::{TelemetryTracker, WatchingStats};
use super::watcher::FileWatcher;

impl FileWatcher {
    /// Handle a notify event and convert it to our internal event format
    pub(super) fn handle_notify_event(event: Event, tx: &mpsc::UnboundedSender<FileEvent>) {
        let now = Instant::now();
        let system_time = SystemTime::now();

        for path in event.paths {
            let size = std::fs::metadata(&path).ok().map(|metadata| metadata.len());

            let mut metadata = HashMap::new();
            metadata.insert("event_type".to_string(), format!("{:?}", event.kind));

            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                metadata.insert("file_name".to_string(), file_name.to_string());
            }

            if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                metadata.insert("file_extension".to_string(), extension.to_string());
            }

            let file_event = FileEvent {
                path,
                event_kind: event.kind,
                timestamp: now,
                system_time,
                size,
                metadata,
            };

            if let Err(e) = tx.send(file_event) {
                tracing::error!("Failed to send file event: {}", e);
            }
        }
    }

    /// Main event processing loop
    pub(super) async fn event_processing_loop(
        event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
        debouncer: Arc<Mutex<EventDebouncer>>,
        batcher: Arc<Mutex<EventBatcher>>,
        patterns: Arc<RwLock<CompiledPatterns>>,
        config: Arc<RwLock<WatcherConfig>>,
        task_submitter: TaskSubmitter,
        stats: Arc<Mutex<WatchingStats>>,
        telemetry_tracker: Arc<Mutex<TelemetryTracker>>,
        is_paused: Arc<AtomicBool>,
        paused_event_buffer: Arc<Mutex<PausedEventBuffer>>,
    ) {
        let mut cleanup_interval = interval(Duration::from_secs(300));
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
                        if is_paused.load(Ordering::SeqCst) {
                            let mut buffer = paused_event_buffer.lock().await;
                            buffer.push_event(event);
                            let mut stats_lock = stats.lock().await;
                            stats_lock.events_received += 1;
                        } else {
                            Self::process_file_event(
                                event, &debouncer, &batcher, &patterns,
                                &config, &task_submitter, &stats, &telemetry_tracker,
                            ).await;
                        }
                    } else {
                        break;
                    }
                },
                _ = debounce_interval.tick() => {
                    if !is_paused.load(Ordering::SeqCst) {
                        Self::process_debounced_events(
                            &debouncer, &batcher, &patterns, &config,
                            &task_submitter, &stats, &telemetry_tracker,
                        ).await;
                    }
                },
                _ = cleanup_interval.tick() => {
                    Self::cleanup_old_events(&debouncer).await;
                },
            }
        }

        tracing::info!("Event processing loop stopped");
    }

    /// Process a single file event
    pub(super) async fn process_file_event(
        event: FileEvent,
        debouncer: &Arc<Mutex<EventDebouncer>>,
        batcher: &Arc<Mutex<EventBatcher>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
        telemetry_tracker: &Arc<Mutex<TelemetryTracker>>,
    ) {
        {
            let mut stats_lock = stats.lock().await;
            stats_lock.events_received += 1;
        }

        {
            let patterns_lock = patterns.read().await;
            if !patterns_lock.should_process(&event.path) {
                let mut stats_lock = stats.lock().await;
                stats_lock.events_filtered += 1;
                return;
            }
        }

        {
            let config_lock = config.read().await;
            if let (Some(max_size), Some(file_size)) = (config_lock.max_file_size, event.size) {
                if file_size > max_size {
                    tracing::debug!(
                        "Skipping large file: {} ({} bytes)",
                        event.path.display(),
                        file_size
                    );
                    let mut stats_lock = stats.lock().await;
                    stats_lock.events_filtered += 1;
                    return;
                }
            }
        }

        let (should_process, evicted) = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.add_event(event.clone())
        };

        if let Some(evicted_event) = evicted {
            tracing::info!(
                "Processing evicted event to prevent data loss: {}",
                evicted_event.path.display()
            );
            Self::handle_ready_event(
                evicted_event,
                batcher,
                config,
                task_submitter,
                stats,
                telemetry_tracker,
            )
            .await;
        }

        if should_process {
            Self::handle_ready_event(
                event,
                batcher,
                config,
                task_submitter,
                stats,
                telemetry_tracker,
            )
            .await;
        } else {
            let mut stats_lock = stats.lock().await;
            stats_lock.events_debounced += 1;
        }
    }

    /// Process events that are ready after debouncing
    pub(super) async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        batcher: &Arc<Mutex<EventBatcher>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
        telemetry_tracker: &Arc<Mutex<TelemetryTracker>>,
    ) {
        let ready_events = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.get_ready_events()
        };

        for event in ready_events {
            {
                let patterns_lock = patterns.read().await;
                if !patterns_lock.should_process(&event.path) {
                    continue;
                }
            }
            Self::handle_ready_event(
                event,
                batcher,
                config,
                task_submitter,
                stats,
                telemetry_tracker,
            )
            .await;
        }
    }

    /// Handle an event that's ready for processing
    pub(super) async fn handle_ready_event(
        event: FileEvent,
        batcher: &Arc<Mutex<EventBatcher>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
        telemetry_tracker: &Arc<Mutex<TelemetryTracker>>,
    ) {
        let ready_batch = {
            let mut batcher_lock = batcher.lock().await;
            batcher_lock.add_event(event)
        };

        if let Some(batch) = ready_batch {
            Self::submit_processing_tasks(batch, config, task_submitter, stats, telemetry_tracker)
                .await;
        }
    }

    /// Submit processing tasks for a batch of events
    pub(super) async fn submit_processing_tasks(
        events: Vec<FileEvent>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
        telemetry_tracker: &Arc<Mutex<TelemetryTracker>>,
    ) {
        let config_lock = config.read().await;
        let task_priority = config_lock.task_priority;
        let default_collection = config_lock.default_collection.clone();
        let telemetry_enabled = config_lock.telemetry.enabled;
        drop(config_lock);

        for event in events {
            let start_time = Instant::now();

            match event.event_kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    if !event.path.exists() || !event.path.is_file() {
                        continue;
                    }

                    let source = match task_priority {
                        TaskPriority::ProjectWatching => TaskSource::ProjectWatcher {
                            project_path: event
                                .path
                                .parent()
                                .unwrap_or_else(|| Path::new("/"))
                                .to_string_lossy()
                                .to_string(),
                        },
                        TaskPriority::BackgroundWatching => TaskSource::BackgroundWatcher {
                            folder_path: event
                                .path
                                .parent()
                                .unwrap_or_else(|| Path::new("/"))
                                .to_string_lossy()
                                .to_string(),
                        },
                        _ => TaskSource::Generic {
                            operation: "file_watching".to_string(),
                        },
                    };

                    let branch = event
                        .path
                        .parent()
                        .map(|p| crate::watching_queue::get_current_branch(p))
                        .unwrap_or_else(|| "main".to_string());

                    let payload = TaskPayload::ProcessDocument {
                        file_path: event.path.clone(),
                        collection: default_collection.clone(),
                        branch,
                    };

                    match task_submitter
                        .submit_task(task_priority, source, payload, None)
                        .await
                    {
                        Ok(_) => {
                            let mut stats_lock = stats.lock().await;
                            stats_lock.tasks_submitted += 1;
                            stats_lock.events_processed += 1;
                            tracing::debug!(
                                "Submitted processing task for: {}",
                                event.path.display()
                            );

                            if telemetry_enabled {
                                let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                                let file_size = event.size.unwrap_or(0);
                                drop(stats_lock);
                                let mut telemetry_lock = telemetry_tracker.lock().await;
                                telemetry_lock.record_latency(latency_ms);
                                telemetry_lock.record_file_processed(file_size);
                            }
                        }
                        Err(e) => {
                            let mut stats_lock = stats.lock().await;
                            stats_lock.errors += 1;
                            tracing::error!(
                                "Failed to submit processing task for {}: {}",
                                event.path.display(),
                                e
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Submit a batch of events directly for processing (used in initial scan)
    pub(super) async fn submit_batch_directly(
        events: &[FileEvent],
        batcher: &Arc<Mutex<EventBatcher>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
        telemetry_tracker: &Arc<Mutex<TelemetryTracker>>,
    ) {
        for event in events {
            Self::handle_ready_event(
                event.clone(),
                batcher,
                config,
                task_submitter,
                stats,
                telemetry_tracker,
            )
            .await;
        }
    }

    /// Clean up old events from debouncer
    pub(super) async fn cleanup_old_events(debouncer: &Arc<Mutex<EventDebouncer>>) {
        let mut debouncer_lock = debouncer.lock().await;
        debouncer_lock.cleanup(Duration::from_secs(3600));
    }
}
