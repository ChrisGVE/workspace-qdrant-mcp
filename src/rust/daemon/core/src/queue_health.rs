//! Queue processor health state for monitoring
//!
//! Shared between the UnifiedQueueProcessor (which updates it) and
//! the gRPC SystemService (which reads it for health reporting).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::SystemTime;

/// Thread-safe queue processor health state using atomics.
///
/// Created in `main.rs`, shared via `Arc<QueueProcessorHealth>` between:
/// - `UnifiedQueueProcessor` (writer): calls `set_running`, `record_poll`, `record_success`, etc.
/// - `SystemServiceImpl` (reader): reads atomic fields for health reporting
#[derive(Debug, Default)]
pub struct QueueProcessorHealth {
    /// Whether the processor is currently running
    pub is_running: AtomicBool,
    /// Last poll timestamp (Unix millis)
    pub last_poll_time: AtomicU64,
    /// Total error count
    pub error_count: AtomicU64,
    /// Items processed total
    pub items_processed: AtomicU64,
    /// Items failed total
    pub items_failed: AtomicU64,
    /// Current queue depth
    pub queue_depth: AtomicU64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: AtomicU64,
}

impl QueueProcessorHealth {
    /// Create new health state
    pub fn new() -> Self {
        Self::default()
    }

    /// Update running status
    pub fn set_running(&self, running: bool) {
        self.is_running.store(running, Ordering::SeqCst);
    }

    /// Update last poll time to now
    pub fn record_poll(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_poll_time.store(now, Ordering::SeqCst);
    }

    /// Increment error count
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Record successful processing
    pub fn record_success(&self, processing_time_ms: u64) {
        let prev_count = self.items_processed.fetch_add(1, Ordering::SeqCst);
        let prev_avg = self.avg_processing_time_ms.load(Ordering::SeqCst);
        // Running average calculation
        let new_avg = if prev_count == 0 {
            processing_time_ms
        } else {
            (prev_avg * prev_count + processing_time_ms) / (prev_count + 1)
        };
        self.avg_processing_time_ms.store(new_avg, Ordering::SeqCst);
    }

    /// Record failed processing
    pub fn record_failure(&self) {
        self.items_failed.fetch_add(1, Ordering::SeqCst);
    }

    /// Update queue depth
    pub fn set_queue_depth(&self, depth: u64) {
        self.queue_depth.store(depth, Ordering::SeqCst);
    }

    /// Get seconds since last poll
    pub fn seconds_since_last_poll(&self) -> u64 {
        let last = self.last_poll_time.load(Ordering::SeqCst);
        if last == 0 {
            return u64::MAX; // Never polled
        }
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        (now.saturating_sub(last)) / 1000
    }
}
