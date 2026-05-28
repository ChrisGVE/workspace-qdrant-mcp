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
            events: LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(10_000).unwrap()),
            ),
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
                    tracing::warn!(
                        "EventDebouncer at capacity, flushing oldest event to prevent data loss"
                    );
                }
                return (false, evicted.map(|(_, event)| event));
            }
        }

        // Insert new event, track eviction if cache was full
        let evicted = self.events.push(path, event);
        if evicted.is_some() {
            self.evictions += 1;
            tracing::warn!(
                "EventDebouncer at capacity, flushing oldest event to prevent data loss"
            );
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

#[cfg(test)]
mod debouncer_tests {
    use super::*;
    use notify::event::{CreateKind, EventKind};
    use std::path::PathBuf;
    use std::time::SystemTime;

    /// Build a FileEvent with the given path and timestamp. Used to seed
    /// debounce state with a "known past" event.
    fn mk_event(path: &str, ts: Instant) -> FileEvent {
        FileEvent {
            path: PathBuf::from(path),
            event_kind: EventKind::Create(CreateKind::File),
            timestamp: ts,
            system_time: SystemTime::now(),
            size: None,
            metadata: HashMap::new(),
        }
    }

    fn now_event(path: &str) -> FileEvent {
        mk_event(path, Instant::now())
    }

    #[test]
    fn first_event_for_path_is_accepted() {
        let mut d = EventDebouncer::new(1_000, 10);
        let (process, evicted) = d.add_event(now_event("/a"));
        assert!(process);
        assert!(evicted.is_none());
    }

    #[test]
    fn second_event_within_window_is_debounced() {
        let mut d = EventDebouncer::new(1_000, 10);
        let _ = d.add_event(now_event("/a"));
        let (process, _) = d.add_event(now_event("/a"));
        assert!(!process, "second event for same path within window is suppressed");
    }

    #[test]
    fn second_event_past_window_is_accepted() {
        // Seed the cache with an event whose timestamp is well outside the
        // 20ms debounce window — no real sleep needed, the seed event's
        // backdated `timestamp` is what the debouncer compares against.
        let mut d = EventDebouncer::new(20, 10);
        let past = Instant::now()
            .checked_sub(Duration::from_millis(500))
            .expect("Instant subtraction with 500ms backwards must succeed");
        let _ = d.add_event(mk_event("/a", past));

        let (process, _) = d.add_event(now_event("/a"));
        assert!(process, "event past debounce window must pass through");
    }

    #[test]
    fn distinct_paths_do_not_interfere() {
        let mut d = EventDebouncer::new(1_000, 10);
        let (p1, _) = d.add_event(now_event("/a"));
        let (p2, _) = d.add_event(now_event("/b"));
        assert!(p1 && p2);
    }

    #[test]
    fn capacity_overflow_evicts_oldest_and_counts() {
        let mut d = EventDebouncer::new(60_000, 2);
        let (_, e1) = d.add_event(now_event("/a"));
        let (_, e2) = d.add_event(now_event("/b"));
        assert!(e1.is_none() && e2.is_none(), "no eviction below capacity");

        // Third path forces oldest (/a) out.
        let (_, evicted) = d.add_event(now_event("/c"));
        assert!(evicted.is_some(), "capacity overflow must surface evicted event");
        assert_eq!(evicted.unwrap().path, PathBuf::from("/a"));
        assert_eq!(d.eviction_count(), 1);
    }

    #[test]
    fn get_ready_events_returns_only_past_window_entries() {
        // debounce_ms = 0 → every entry is "past window" the instant it
        // lands. Easier than sleeping to make events ready.
        let mut d = EventDebouncer::new(0, 10);
        let _ = d.add_event(now_event("/a"));
        let _ = d.add_event(now_event("/b"));

        let ready = d.get_ready_events();
        assert_eq!(ready.len(), 2);

        // get_ready_events drains the entries it returns.
        let ready_again = d.get_ready_events();
        assert!(ready_again.is_empty());
    }

    #[test]
    fn get_ready_events_skips_in_window_entries() {
        let mut d = EventDebouncer::new(60_000, 10);
        let _ = d.add_event(now_event("/a"));
        // Window is 60s and event was just added — nothing ready yet.
        assert!(d.get_ready_events().is_empty());
    }

    #[test]
    fn cleanup_removes_events_older_than_max_age() {
        let mut d = EventDebouncer::new(60_000, 10);
        let old = Instant::now()
            .checked_sub(Duration::from_secs(10))
            .expect("Instant subtraction with 10s backwards must succeed");
        let _ = d.add_event(mk_event("/old", old));
        let _ = d.add_event(now_event("/fresh"));

        // Anything older than 1s gets purged; /old qualifies, /fresh doesn't.
        d.cleanup(Duration::from_secs(1));

        // Use get_ready_events with a zero-window debouncer to inspect what
        // remains without depending on internal fields.
        let mut probe = EventDebouncer::new(0, 10);
        std::mem::swap(&mut d.events, &mut probe.events);
        let remaining = probe.get_ready_events();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].path, PathBuf::from("/fresh"));
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
                tracing::info!(
                    "Batcher at capacity, submitting evicted event immediately: {}",
                    evicted.path.display()
                );
                vec![evicted]
            })
        } else {
            None
        };

        let key = if self.config.group_by_type {
            event
                .path
                .extension()
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
        if now.duration_since(self.last_flush)
            >= Duration::from_millis(self.config.max_batch_wait_ms)
        {
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

#[cfg(test)]
mod batcher_tests {
    use super::*;
    use notify::event::{CreateKind, EventKind};
    use std::path::PathBuf;
    use std::time::SystemTime;

    fn mk_event(path: &str) -> FileEvent {
        FileEvent {
            path: PathBuf::from(path),
            event_kind: EventKind::Create(CreateKind::File),
            timestamp: Instant::now(),
            system_time: SystemTime::now(),
            size: None,
            metadata: HashMap::new(),
        }
    }

    fn cfg(enabled: bool, max_batch_size: usize, group_by_type: bool) -> BatchConfig {
        BatchConfig {
            enabled,
            max_batch_size,
            // Large wait so the time-based flush never fires during tests.
            max_batch_wait_ms: 60_000,
            group_by_type,
        }
    }

    #[test]
    fn disabled_batcher_returns_each_event_immediately() {
        let mut b = EventBatcher::new(cfg(false, 10, false), 100);
        let out = b.add_event(mk_event("/a.txt"));
        let batch = out.expect("disabled batcher returns the event immediately");
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].path, PathBuf::from("/a.txt"));
    }

    #[test]
    fn batch_returns_when_max_size_reached() {
        let mut b = EventBatcher::new(cfg(true, 2, false), 100);
        assert!(b.add_event(mk_event("/a.txt")).is_none());
        // Second event hits the max — batch drains and returns both.
        let out = b
            .add_event(mk_event("/b.txt"))
            .expect("max_batch_size reached should drain the batch");
        assert_eq!(out.len(), 2);
        // Subsequent add starts a fresh batch.
        assert!(b.add_event(mk_event("/c.txt")).is_none());
    }

    #[test]
    fn group_by_type_keeps_extensions_separate() {
        let mut b = EventBatcher::new(cfg(true, 2, true), 100);
        // Two .rs and one .txt: only the .rs batch fills.
        assert!(b.add_event(mk_event("/a.rs")).is_none());
        assert!(b.add_event(mk_event("/c.txt")).is_none());
        let out = b
            .add_event(mk_event("/b.rs"))
            .expect("rs batch fills first because max_batch_size=2");
        assert_eq!(out.len(), 2);
        for ev in &out {
            assert_eq!(ev.path.extension().and_then(|e| e.to_str()), Some("rs"));
        }
    }

    #[test]
    fn flush_all_drains_every_group_and_returns_none_on_empty() {
        let mut b = EventBatcher::new(cfg(true, 100, true), 100);
        let _ = b.add_event(mk_event("/a.rs"));
        let _ = b.add_event(mk_event("/b.txt"));
        let _ = b.add_event(mk_event("/c.rs"));

        let drained = b.flush_all().expect("flush_all returns pending events");
        assert_eq!(drained.len(), 3);

        // After a full drain the batcher is empty and flush_all returns None.
        assert!(b.flush_all().is_none());
    }

    #[test]
    fn capacity_overflow_evicts_oldest_for_immediate_processing() {
        let mut b = EventBatcher::new(cfg(true, 100, false), 2);
        let _ = b.add_event(mk_event("/oldest"));
        let _ = b.add_event(mk_event("/middle"));
        // Third event triggers eviction: oldest comes out immediately so
        // it isn't lost while the batch continues to fill.
        let evicted = b
            .add_event(mk_event("/newest"))
            .expect("capacity overflow yields an evicted-event batch");
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].path, PathBuf::from("/oldest"));
        assert_eq!(b.eviction_count(), 1);
    }
}
