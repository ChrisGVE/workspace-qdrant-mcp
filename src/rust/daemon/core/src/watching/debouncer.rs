//! Event debouncing and batching for the file watching system

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use lru::LruCache;

use super::config::BatchConfig;
use super::events::FileEvent;

/// Event debouncer to prevent duplicate processing with bounded memory
#[derive(Debug)]
pub(super) struct EventDebouncer {
    events: LruCache<std::path::PathBuf, FileEvent>,
    debounce_duration: Duration,
    evictions: u64,
}

impl EventDebouncer {
    pub(super) fn new(debounce_ms: u64, capacity: usize) -> Self {
        Self {
            events: LruCache::new(NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(10_000).unwrap())),
            debounce_duration: Duration::from_millis(debounce_ms),
            evictions: 0,
        }
    }

    /// Add event to debouncer, returns (should_process, evicted_event)
    /// - should_process: true if event should be processed immediately (not within debounce period)
    /// - evicted_event: event that was evicted due to capacity (needs immediate processing to avoid data loss)
    pub(super) fn add_event(&mut self, event: FileEvent) -> (bool, Option<FileEvent>) {
        let now = Instant::now();
        let path = event.path.clone();

        if let Some(existing) = self.events.get(&path) {
            // If the existing event is within debounce period, update and don't process
            if now.duration_since(existing.timestamp) < self.debounce_duration {
                // Use push to update and move to front of LRU
                let evicted = self.events.push(path, event);
                if evicted.is_some() {
                    self.evictions += 1;
                    tracing::warn!("EventDebouncer at capacity, flushing oldest event to prevent data loss");
                }
                return (false, evicted.map(|(_, event)| event));
            }
        }

        // Insert new event, track eviction if cache was full
        let evicted = self.events.push(path, event);
        if evicted.is_some() {
            self.evictions += 1;
            tracing::warn!("EventDebouncer at capacity, flushing oldest event to prevent data loss");
        }
        (true, evicted.map(|(_, event)| event))
    }

    /// Get events that are ready to be processed (past debounce period)
    pub(super) fn get_ready_events(&mut self) -> Vec<FileEvent> {
        let now = Instant::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();

        for (path, event) in self.events.iter() {
            if now.duration_since(event.timestamp) >= self.debounce_duration {
                ready.push(event.clone());
                to_remove.push(path.clone());
            }
        }

        for path in to_remove {
            self.events.pop(&path);
        }

        ready
    }

    /// Clear old events (cleanup)
    pub(super) fn cleanup(&mut self, max_age: Duration) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        for (path, event) in self.events.iter() {
            if now.duration_since(event.timestamp) >= max_age {
                to_remove.push(path.clone());
            }
        }

        for path in to_remove {
            self.events.pop(&path);
        }
    }

    /// Get eviction count
    pub(super) fn eviction_count(&self) -> u64 {
        self.evictions
    }
}

/// Batch processor for grouping and processing file events efficiently with bounded memory
#[derive(Debug)]
pub(super) struct EventBatcher {
    batches: HashMap<String, VecDeque<FileEvent>>,
    config: BatchConfig,
    last_flush: Instant,
    max_total_capacity: usize,
    current_total_size: usize,
    evictions: u64,
}

impl EventBatcher {
    pub(super) fn new(config: BatchConfig, max_total_capacity: usize) -> Self {
        Self {
            batches: HashMap::new(),
            config,
            last_flush: Instant::now(),
            max_total_capacity,
            current_total_size: 0,
            evictions: 0,
        }
    }

    /// Add event to batcher
    /// Returns Some(Vec<FileEvent>) when a batch is ready for immediate processing
    pub(super) fn add_event(&mut self, event: FileEvent) -> Option<Vec<FileEvent>> {
        if !self.config.enabled {
            return Some(vec![event]);
        }

        // Check if we're at capacity - evict oldest event and submit immediately
        let evicted_batch = if self.current_total_size >= self.max_total_capacity {
            self.evict_oldest_event().map(|evicted| {
                tracing::info!("Batcher at capacity, submitting evicted event immediately: {}", evicted.path.display());
                vec![evicted]
            })
        } else {
            None
        };

        let key = if self.config.group_by_type {
            event.path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            "default".to_string()
        };

        let batch = self.batches.entry(key).or_default();
        batch.push_back(event);
        self.current_total_size += 1;

        // If we had an evicted event, return it immediately for processing
        if evicted_batch.is_some() {
            return evicted_batch;
        }

        // Check if batch is full
        if batch.len() >= self.config.max_batch_size {
            let count = batch.len();
            let events = batch.drain(..).collect();
            self.current_total_size -= count;
            return Some(events);
        }

        // Check if max wait time has elapsed
        let now = Instant::now();
        if now.duration_since(self.last_flush) >= Duration::from_millis(self.config.max_batch_wait_ms) {
            return self.flush_all();
        }

        None
    }

    pub(super) fn flush_all(&mut self) -> Option<Vec<FileEvent>> {
        self.last_flush = Instant::now();

        let mut all_events = Vec::new();
        for batch in self.batches.values_mut() {
            let count = batch.len();
            all_events.extend(batch.drain(..));
            self.current_total_size -= count;
        }

        if all_events.is_empty() {
            None
        } else {
            Some(all_events)
        }
    }

    /// Evict the oldest event from the oldest batch to make room
    fn evict_oldest_event(&mut self) -> Option<FileEvent> {
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = Instant::now();

        for (key, batch) in &self.batches {
            if let Some(event) = batch.front() {
                if event.timestamp < oldest_time {
                    oldest_time = event.timestamp;
                    oldest_key = Some(key.clone());
                }
            }
        }

        if let Some(key) = oldest_key {
            if let Some(batch) = self.batches.get_mut(&key) {
                if let Some(evicted_event) = batch.pop_front() {
                    self.current_total_size -= 1;
                    self.evictions += 1;
                    tracing::warn!(
                        "EventBatcher at capacity, evicting oldest event from batch '{}' for immediate processing (total evictions: {})",
                        key,
                        self.evictions
                    );

                    if batch.is_empty() {
                        self.batches.remove(&key);
                    }

                    return Some(evicted_event);
                }
            }
        }

        None
    }

    /// Get eviction count
    pub(super) fn eviction_count(&self) -> u64 {
        self.evictions
    }
}
