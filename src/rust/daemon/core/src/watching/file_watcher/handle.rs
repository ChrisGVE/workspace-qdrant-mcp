//! WatcherHandle for controlling the file watcher.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{Debouncer, RecommendedCache};
use tokio::sync::{Mutex, RwLock};
use wqm_common::paths::CanonicalPath;

use super::{EnhancedWatcherError, WatchEntry};

/// Handle for controlling the file watcher
pub struct WatcherHandle {
    /// The underlying debouncer
    debouncer: Arc<Mutex<Debouncer<RecommendedWatcher, RecommendedCache>>>,

    /// Running flag
    running: Arc<RwLock<bool>>,

    /// Watch entries
    watch_entries: Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,

    /// Processing task handle
    _process_task: tokio::task::JoinHandle<()>,
}

/// Convert a [`Path`] to its syntactic-canonical [`PathBuf`] per spec
/// §16 §3.1. No `std::fs::canonicalize`. Returned `PathBuf` is the
/// in-memory key the watch_entries map uses for fast lookup.
fn syntactic_canonical_pathbuf(path: &Path) -> Result<PathBuf, EnhancedWatcherError> {
    let s = path.to_str().ok_or_else(|| {
        EnhancedWatcherError::Path(format!(
            "Failed to convert path to UTF-8: {}",
            path.display()
        ))
    })?;
    let cp = CanonicalPath::from_user_input(s).map_err(|e| {
        EnhancedWatcherError::Path(format!("Failed to normalize path {}: {e}", path.display()))
    })?;
    Ok(PathBuf::from(cp.into_string()))
}

impl WatcherHandle {
    /// Create a new WatcherHandle (called internally by EnhancedFileWatcher::start)
    pub(super) fn new(
        debouncer: Debouncer<RecommendedWatcher, RecommendedCache>,
        running: Arc<RwLock<bool>>,
        watch_entries: Arc<RwLock<HashMap<PathBuf, WatchEntry>>>,
        process_task: tokio::task::JoinHandle<()>,
    ) -> Self {
        Self {
            debouncer: Arc::new(Mutex::new(debouncer)),
            running,
            watch_entries,
            _process_task: process_task,
        }
    }

    /// Add a path to watch
    pub async fn watch(
        &self,
        path: &Path,
        tenant_id: String,
        recursive: bool,
    ) -> Result<(), EnhancedWatcherError> {
        let canonical = syntactic_canonical_pathbuf(path)?;

        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        {
            let mut debouncer = self.debouncer.lock().await;
            debouncer.watch(&canonical, mode)?;
        }

        {
            let mut entries = self.watch_entries.write().await;
            entries.insert(
                canonical,
                WatchEntry {
                    tenant_id,
                    recursive,
                },
            );
        }

        Ok(())
    }

    /// Remove a path from watching
    pub async fn unwatch(&self, path: &Path) -> Result<(), EnhancedWatcherError> {
        let canonical = syntactic_canonical_pathbuf(path)?;

        {
            let mut debouncer = self.debouncer.lock().await;
            debouncer.unwatch(&canonical)?;
        }

        {
            let mut entries = self.watch_entries.write().await;
            entries.remove(&canonical);
        }

        Ok(())
    }

    /// Update watch path after a rename
    pub async fn update_watch_path(
        &self,
        old_path: &Path,
        new_path: &Path,
    ) -> Result<(), EnhancedWatcherError> {
        let entry = {
            let entries = self.watch_entries.read().await;
            entries.get(old_path).cloned()
        };

        if let Some(entry) = entry {
            self.unwatch(old_path).await.ok();
            self.watch(new_path, entry.tenant_id, entry.recursive)
                .await?;
        }

        Ok(())
    }

    /// Stop the watcher
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Get the list of watched paths
    pub async fn watched_paths(&self) -> Vec<PathBuf> {
        let entries = self.watch_entries.read().await;
        entries.keys().cloned().collect()
    }

    /// Get tenant ID for a path
    pub async fn get_tenant_id(&self, path: &Path) -> Option<String> {
        let entries = self.watch_entries.read().await;

        // Try exact match first
        if let Some(entry) = entries.get(path) {
            return Some(entry.tenant_id.clone());
        }

        // Try to find parent watch that covers this path
        for (watched_path, entry) in entries.iter() {
            if entry.recursive && path.starts_with(watched_path) {
                return Some(entry.tenant_id.clone());
            }
        }

        None
    }
}
