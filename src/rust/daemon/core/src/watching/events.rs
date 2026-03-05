//! File event types and pause buffer for the watching system

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

use notify::EventKind;

/// File event with metadata and debouncing information
#[derive(Debug, Clone)]
pub struct FileEvent {
    pub path: PathBuf,
    pub event_kind: EventKind,
    pub timestamp: Instant,
    pub system_time: SystemTime,
    pub size: Option<u64>,
    pub metadata: HashMap<String, String>,
}

/// Maximum capacity for the paused event buffer
pub(crate) const PAUSED_EVENT_BUFFER_CAPACITY: usize = 10_000;

/// Buffer for holding file events received while watchers are paused
#[derive(Debug)]
pub struct PausedEventBuffer {
    /// Buffered events in FIFO order
    events: VecDeque<FileEvent>,
    /// Maximum capacity
    capacity: usize,
    /// Count of events evicted due to buffer overflow
    evictions: u64,
}

impl PausedEventBuffer {
    /// Create a new buffer with the default capacity
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            capacity: PAUSED_EVENT_BUFFER_CAPACITY,
            evictions: 0,
        }
    }

    /// Push an event into the buffer, evicting the oldest if at capacity
    pub fn push_event(&mut self, event: FileEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
            self.evictions += 1;
            if self.evictions % 1000 == 1 {
                tracing::warn!(
                    "Paused event buffer overflow: {} events evicted (capacity={})",
                    self.evictions,
                    self.capacity
                );
            }
        }
        self.events.push_back(event);
    }

    /// Drain all buffered events for processing
    pub fn drain_events(&mut self) -> Vec<FileEvent> {
        self.events.drain(..).collect()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get the number of buffered events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Get the number of evicted events
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Get the buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Create a buffer with a custom capacity (for testing)
    #[cfg(test)]
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            events: VecDeque::new(),
            capacity,
            evictions: 0,
        }
    }
}

impl Default for PausedEventBuffer {
    fn default() -> Self {
        Self::new()
    }
}
