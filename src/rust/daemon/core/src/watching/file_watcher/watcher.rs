//! EnhancedFileWatcher implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify::EventKind;
use notify::event::{CreateKind, ModifyKind, RemoveKind, RenameMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, DebouncedEvent};
use tokio::sync::{mpsc, RwLock, Mutex};

use crate::watching::move_detector::{MoveCorrelator, RenameAction};
use crate::watching::path_validator::PathValidator;
use super::{
    EnhancedWatcherConfig, EnhancedWatcherError, WatchEntry, WatchEvent,
};
use super::handle::WatcherHandle;

/// Enhanced file watcher with rename correlation
pub struct EnhancedFileWatcher {
    /// Configuration
    config: EnhancedWatcherConfig,

    /// Move correlator for rename tracking
    move_correlator: Arc<Mutex<MoveCorrelator>>,

    /// Path validator for orphan detection
    path_validator: Arc<PathValidator>,

    /// Channel for sending watch events
    event_sender: mpsc::Sender<WatchEvent>,

    /// Watched paths and their tenant IDs
    watch_entries: Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,

    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl EnhancedFileWatcher {
    /// Create a new enhanced file watcher
    pub fn new(
        config: EnhancedWatcherConfig,
        event_sender: mpsc::Sender<WatchEvent>,
    ) -> Self {
        let move_correlator = MoveCorrelator::with_config(config.move_correlator.clone());
        let path_validator = PathValidator::with_config(config.path_validator.clone());

        Self {
            config,
            move_correlator: Arc::new(Mutex::new(move_correlator)),
            path_validator: Arc::new(path_validator),
            event_sender,
            watch_entries: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the file watcher
    pub async fn start(
        self: Arc<Self>,
    ) -> Result<WatcherHandle, EnhancedWatcherError> {
        let debounce_delay = Duration::from_millis(self.config.debounce_delay_ms);

        let event_sender = self.event_sender.clone();
        let move_correlator = self.move_correlator.clone();
        let watch_entries = self.watch_entries.clone();
        let path_validator = self.path_validator.clone();

        let (tx, mut rx) = mpsc::channel::<Vec<DebouncedEvent>>(1000);

        let debouncer = new_debouncer(
            debounce_delay,
            None,
            move |result: DebounceEventResult| {
                match result {
                    Ok(events) => {
                        let _ = tx.blocking_send(events);
                    }
                    Err(errors) => {
                        for error in errors {
                            tracing::error!("Debouncer error: {:?}", error);
                        }
                    }
                }
            },
        )?;

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let running_flag = self.running.clone();
        let tick_interval = Duration::from_millis(self.config.tick_interval_ms);

        let process_task = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(tick_interval);

            loop {
                tokio::select! {
                    Some(events) = rx.recv() => {
                        Self::process_events(
                            events,
                            &event_sender,
                            &move_correlator,
                            &watch_entries,
                            &path_validator,
                        ).await;
                    }

                    _ = ticker.tick() => {
                        Self::handle_tick_expiry(
                            &move_correlator,
                            &event_sender,
                            &path_validator,
                        ).await;
                    }

                    else => {
                        let running = running_flag.read().await;
                        if !*running {
                            break;
                        }
                    }
                }
            }

            tracing::info!("Enhanced file watcher stopped");
        });

        Ok(WatcherHandle::new(
            debouncer,
            self.running.clone(),
            self.watch_entries.clone(),
            process_task,
        ))
    }

    /// Handle periodic tick: emit expired cross-filesystem moves and check path validation.
    async fn handle_tick_expiry(
        move_correlator: &Arc<Mutex<MoveCorrelator>>,
        event_sender: &mpsc::Sender<WatchEvent>,
        path_validator: &Arc<PathValidator>,
    ) {
        let mut correlator = move_correlator.lock().await;
        let expired = correlator.get_expired_moves();
        drop(correlator);

        for action in expired {
            if let RenameAction::CrossFilesystemMove { deleted_path, is_directory } = action {
                let _ = event_sender.send(WatchEvent::CrossFilesystemMove {
                    deleted_path,
                    is_directory,
                }).await;
            }
        }

        if path_validator.is_validation_due().await {
            tracing::debug!("Path validation due, will be triggered by daemon");
        }
    }

    /// Process debounced events
    async fn process_events(
        events: Vec<DebouncedEvent>,
        event_sender: &mpsc::Sender<WatchEvent>,
        move_correlator: &Arc<Mutex<MoveCorrelator>>,
        watch_entries: &Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,
        path_validator: &Arc<PathValidator>,
    ) {
        for debounced in events {
            let event = debounced.event;
            let paths = &event.paths;

            match event.kind {
                EventKind::Create(kind) => {
                    let is_directory = matches!(kind, CreateKind::Folder);
                    for path in paths {
                        let _ = event_sender.send(WatchEvent::Created {
                            path: path.clone(),
                            is_directory,
                        }).await;
                    }
                }

                EventKind::Modify(kind) => {
                    if matches!(kind, ModifyKind::Metadata(_)) {
                        continue;
                    }

                    let is_directory = paths.first()
                        .map(|p| p.is_dir())
                        .unwrap_or(false);

                    if let ModifyKind::Name(rename_mode) = kind {
                        Self::handle_rename_event(
                            rename_mode,
                            paths,
                            is_directory,
                            event_sender,
                            move_correlator,
                            watch_entries,
                            path_validator,
                        ).await;
                    } else {
                        for path in paths {
                            let _ = event_sender.send(WatchEvent::Modified {
                                path: path.clone(),
                                is_directory,
                            }).await;
                        }
                    }
                }

                EventKind::Remove(kind) => {
                    let is_directory = matches!(kind, RemoveKind::Folder);
                    for path in paths {
                        let _ = event_sender.send(WatchEvent::Deleted {
                            path: path.clone(),
                            is_directory,
                        }).await;
                    }
                }

                EventKind::Access(_) => {}

                EventKind::Other | EventKind::Any => {
                    tracing::trace!("Unhandled event kind: {:?}", event.kind);
                }
            }
        }
    }

    /// Handle rename events
    async fn handle_rename_event(
        rename_mode: RenameMode,
        paths: &[PathBuf],
        is_directory: bool,
        event_sender: &mpsc::Sender<WatchEvent>,
        move_correlator: &Arc<Mutex<MoveCorrelator>>,
        watch_entries: &Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,
        path_validator: &Arc<PathValidator>,
    ) {
        match rename_mode {
            RenameMode::From => {
                if let Some(path) = paths.first() {
                    let mut correlator = move_correlator.lock().await;
                    correlator.handle_moved_from(path.clone(), is_directory, None);

                    if is_directory {
                        path_validator.reset_timer().await;
                    }
                }
            }

            RenameMode::To => {
                if let Some(path) = paths.first() {
                    let mut correlator = move_correlator.lock().await;
                    let action = correlator.handle_moved_to(path.clone(), is_directory, None);
                    drop(correlator);

                    Self::emit_rename_event(action, event_sender, watch_entries, path_validator, is_directory).await;
                }
            }

            RenameMode::Both => {
                if paths.len() >= 2 {
                    let old_path = &paths[0];
                    let new_path = &paths[1];

                    let mut correlator = move_correlator.lock().await;
                    let action = correlator.handle_rename_event(
                        old_path.clone(),
                        new_path.clone(),
                        is_directory,
                    );
                    drop(correlator);

                    Self::emit_rename_event(action, event_sender, watch_entries, path_validator, is_directory).await;
                }
            }

            RenameMode::Any | RenameMode::Other => {
                for path in paths {
                    let _ = event_sender.send(WatchEvent::Modified {
                        path: path.clone(),
                        is_directory,
                    }).await;
                }
            }
        }
    }

    /// Emit a rename event, checking if it's a root folder rename
    async fn emit_rename_event(
        action: RenameAction,
        event_sender: &mpsc::Sender<WatchEvent>,
        watch_entries: &Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,
        path_validator: &Arc<PathValidator>,
        _is_directory: bool,
    ) {
        match action {
            RenameAction::SimpleRename { old_path, new_path, is_directory } |
            RenameAction::IntraFilesystemMove { old_path, new_path, is_directory } => {
                let entries = watch_entries.read().await;
                if let Some(entry) = entries.get(&old_path) {
                    let _ = event_sender.send(WatchEvent::RootRenamed {
                        old_path: old_path.clone(),
                        new_path: new_path.clone(),
                        tenant_id: entry.tenant_id.clone(),
                    }).await;
                } else {
                    let _ = event_sender.send(WatchEvent::Renamed {
                        old_path,
                        new_path,
                        is_directory,
                    }).await;
                }

                if is_directory {
                    path_validator.reset_timer().await;
                }
            }

            RenameAction::CrossFilesystemMove { deleted_path, is_directory } => {
                let _ = event_sender.send(WatchEvent::CrossFilesystemMove {
                    deleted_path,
                    is_directory,
                }).await;
            }

            RenameAction::Pending => {}
        }
    }
}
