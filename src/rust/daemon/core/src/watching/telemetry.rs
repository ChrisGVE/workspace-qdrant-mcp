//! Telemetry and statistics for the file watching system

use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, System};

use super::config::TelemetryConfig;

/// Telemetry snapshot for comprehensive system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Timestamp of this snapshot
    pub timestamp: SystemTime,

    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: Option<f64>,

    /// Memory usage in MB (Resident Set Size)
    pub memory_rss_mb: Option<f64>,

    /// Memory usage in MB (heap allocation estimate)
    pub memory_heap_mb: Option<f64>,

    /// Average processing latency in milliseconds
    pub avg_latency_ms: Option<f64>,

    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: Option<f64>,

    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: Option<f64>,

    /// Current queue depth (number of items in queue)
    pub queue_depth_current: usize,

    /// Average queue depth over collection interval
    pub queue_depth_avg: f64,

    /// Maximum queue depth observed
    pub queue_depth_max: usize,

    /// Throughput in files per second
    pub throughput_files_per_sec: f64,

    /// Throughput in bytes per second
    pub throughput_bytes_per_sec: f64,
}

impl Default for TelemetrySnapshot {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            cpu_usage_percent: None,
            memory_rss_mb: None,
            memory_heap_mb: None,
            avg_latency_ms: None,
            p95_latency_ms: None,
            p99_latency_ms: None,
            queue_depth_current: 0,
            queue_depth_avg: 0.0,
            queue_depth_max: 0,
            throughput_files_per_sec: 0.0,
            throughput_bytes_per_sec: 0.0,
        }
    }
}

/// Statistics for file watching operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatchingStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_debounced: u64,
    pub events_filtered: u64,
    pub tasks_submitted: u64,
    pub errors: u64,
    pub uptime_seconds: u64,
    pub watched_paths: usize,
    pub current_queue_size: usize,
    pub debouncer_evictions: u64,
    pub batcher_evictions: u64,
    /// Number of unique paths in pattern matching cache
    pub pattern_cache_size: usize,

    /// Whether watchers are currently paused
    pub is_paused: bool,
    /// Number of events currently buffered during pause
    pub buffered_events: usize,
    /// Total events evicted from buffer due to overflow
    pub buffer_evictions: u64,

    /// Current telemetry snapshot (only populated if telemetry enabled)
    pub telemetry: Option<TelemetrySnapshot>,

    /// Historical telemetry snapshots (only populated if telemetry enabled with history)
    pub telemetry_history: Option<Vec<TelemetrySnapshot>>,
}

/// Tracks telemetry data for collection
#[derive(Debug)]
pub(super) struct TelemetryTracker {
    /// Processing latencies for percentile calculation (in milliseconds)
    latencies: VecDeque<f64>,

    /// Historical telemetry snapshots (ring buffer)
    pub(super) history: VecDeque<TelemetrySnapshot>,

    /// Last collection time
    pub(super) last_collection: Instant,

    /// Files processed since last collection
    files_processed_interval: u64,

    /// Bytes processed since last collection
    bytes_processed_interval: u64,

    /// Queue depth samples for averaging
    queue_depth_samples: Vec<usize>,

    /// Maximum queue depth observed
    queue_depth_max: usize,

    /// System info for CPU/memory tracking
    system: System,

    /// Current process PID
    pid: Pid,
}

impl TelemetryTracker {
    pub(super) fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let pid = sysinfo::get_current_pid().unwrap();

        Self {
            latencies: VecDeque::with_capacity(1000),
            history: VecDeque::new(),
            last_collection: Instant::now(),
            files_processed_interval: 0,
            bytes_processed_interval: 0,
            queue_depth_samples: Vec::with_capacity(100),
            queue_depth_max: 0,
            system,
            pid,
        }
    }

    pub(super) fn record_latency(&mut self, latency_ms: f64) {
        if self.latencies.len() >= 1000 {
            self.latencies.pop_front();
        }
        self.latencies.push_back(latency_ms);
    }

    pub(super) fn record_file_processed(&mut self, file_size: u64) {
        self.files_processed_interval += 1;
        self.bytes_processed_interval += file_size;
    }

    pub(super) fn record_queue_depth(&mut self, depth: usize) {
        self.queue_depth_samples.push(depth);
        if depth > self.queue_depth_max {
            self.queue_depth_max = depth;
        }
    }

    fn calculate_percentile(&self, percentile: f64) -> Option<f64> {
        if self.latencies.is_empty() {
            return None;
        }

        let mut sorted: Vec<f64> = self.latencies.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((percentile / 100.0) * sorted.len() as f64).floor() as usize;
        let index = index.min(sorted.len() - 1);

        Some(sorted[index])
    }

    pub(super) fn collect_snapshot(
        &mut self,
        config: &TelemetryConfig,
        current_queue_size: usize,
    ) -> Option<TelemetrySnapshot> {
        if !config.enabled {
            return None;
        }

        let elapsed_secs = self.last_collection.elapsed().as_secs_f64();

        self.system.refresh_process(self.pid);

        let cpu_usage_percent = self.collect_cpu(config);
        let (memory_rss_mb, memory_heap_mb) = self.collect_memory(config);
        let (avg_latency_ms, p95_latency_ms, p99_latency_ms) = self.collect_latency(config);
        let (queue_depth_avg, queue_depth_max) = self.collect_queue_depth(config);
        let (throughput_files_per_sec, throughput_bytes_per_sec) =
            self.collect_throughput(config, elapsed_secs);

        let snapshot = TelemetrySnapshot {
            timestamp: SystemTime::now(),
            cpu_usage_percent,
            memory_rss_mb,
            memory_heap_mb,
            avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            queue_depth_current: current_queue_size,
            queue_depth_avg,
            queue_depth_max,
            throughput_files_per_sec,
            throughput_bytes_per_sec,
        };

        self.reset_interval_counters(config, snapshot)
    }

    /// Collect CPU usage metric.
    fn collect_cpu(&self, config: &TelemetryConfig) -> Option<f64> {
        if config.cpu_usage {
            self.system
                .process(self.pid)
                .map(|process| process.cpu_usage() as f64)
        } else {
            None
        }
    }

    /// Collect memory usage metrics (RSS, heap estimate).
    fn collect_memory(&self, config: &TelemetryConfig) -> (Option<f64>, Option<f64>) {
        if config.memory_usage {
            if let Some(process) = self.system.process(self.pid) {
                let rss_mb = process.memory() as f64 / 1024.0 / 1024.0;
                // sysinfo on Windows can report virtual_memory < memory for short-
                // lived test processes; saturating_sub keeps the heap estimate at
                // zero instead of underflowing.
                let heap_mb = process.virtual_memory().saturating_sub(process.memory()) as f64
                    / 1024.0
                    / 1024.0;
                (Some(rss_mb), Some(heap_mb))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        }
    }

    /// Collect latency metrics (avg, p95, p99).
    fn collect_latency(&self, config: &TelemetryConfig) -> (Option<f64>, Option<f64>, Option<f64>) {
        if config.latency {
            let avg = if !self.latencies.is_empty() {
                Some(self.latencies.iter().sum::<f64>() / self.latencies.len() as f64)
            } else {
                None
            };
            let p95 = self.calculate_percentile(95.0);
            let p99 = self.calculate_percentile(99.0);
            (avg, p95, p99)
        } else {
            (None, None, None)
        }
    }

    /// Collect queue depth metrics (avg, max).
    fn collect_queue_depth(&self, config: &TelemetryConfig) -> (f64, usize) {
        if config.queue_depth {
            let avg = if !self.queue_depth_samples.is_empty() {
                self.queue_depth_samples.iter().sum::<usize>() as f64
                    / self.queue_depth_samples.len() as f64
            } else {
                0.0
            };
            (avg, self.queue_depth_max)
        } else {
            (0.0, 0)
        }
    }

    /// Collect throughput metrics (files/sec, bytes/sec).
    fn collect_throughput(&self, config: &TelemetryConfig, elapsed_secs: f64) -> (f64, f64) {
        if config.throughput {
            let files_per_sec = if elapsed_secs > 0.0 {
                self.files_processed_interval as f64 / elapsed_secs
            } else {
                0.0
            };
            let bytes_per_sec = if elapsed_secs > 0.0 {
                self.bytes_processed_interval as f64 / elapsed_secs
            } else {
                0.0
            };
            (files_per_sec, bytes_per_sec)
        } else {
            (0.0, 0.0)
        }
    }

    /// Reset interval counters and add snapshot to history.
    fn reset_interval_counters(
        &mut self,
        config: &TelemetryConfig,
        snapshot: TelemetrySnapshot,
    ) -> Option<TelemetrySnapshot> {
        self.files_processed_interval = 0;
        self.bytes_processed_interval = 0;
        self.queue_depth_samples.clear();
        self.queue_depth_max = 0;
        self.last_collection = Instant::now();

        // Add to history (ring buffer)
        self.history.push_back(snapshot.clone());
        if self.history.len() > config.history_retention {
            self.history.pop_front();
        }

        Some(snapshot)
    }

    pub(super) fn get_history(&self) -> Vec<TelemetrySnapshot> {
        self.history.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Telemetry config with everything enabled — the metric toggles each
    /// gate a different branch in `collect_snapshot`, so the test default
    /// must turn them all on.
    fn full_config() -> TelemetryConfig {
        TelemetryConfig {
            enabled: true,
            history_retention: 3,
            collection_interval_secs: 60,
            cpu_usage: true,
            memory_usage: true,
            latency: true,
            queue_depth: true,
            throughput: true,
        }
    }

    #[test]
    fn calculate_percentile_empty_latencies_returns_none() {
        let t = TelemetryTracker::new();
        assert!(t.calculate_percentile(50.0).is_none());
        assert!(t.calculate_percentile(99.0).is_none());
    }

    #[test]
    fn calculate_percentile_p50_p95_p99() {
        let mut t = TelemetryTracker::new();
        for i in 1..=100 {
            t.record_latency(i as f64);
        }
        // index = floor(p/100 * len), clamped to len-1.
        //   p50 → index 50 → sorted[50] = 51.0
        //   p95 → index 95 → sorted[95] = 96.0
        //   p99 → index 99 → sorted[99] = 100.0
        assert_eq!(t.calculate_percentile(50.0), Some(51.0));
        assert_eq!(t.calculate_percentile(95.0), Some(96.0));
        assert_eq!(t.calculate_percentile(99.0), Some(100.0));
    }

    #[test]
    fn calculate_percentile_clamps_at_max_index() {
        let mut t = TelemetryTracker::new();
        t.record_latency(7.0);
        // (100/100) * 1 = 1, clamped to len-1 = 0.
        assert_eq!(t.calculate_percentile(100.0), Some(7.0));
    }

    #[test]
    fn record_latency_caps_buffer_at_1000() {
        let mut t = TelemetryTracker::new();
        for i in 0..1500 {
            t.record_latency(i as f64);
        }
        assert_eq!(t.latencies.len(), 1000);
        // FIFO eviction: oldest (0.0..499.0) are gone; min remaining is 500.0.
        let min = t.latencies.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(min, 500.0);
    }

    #[test]
    fn record_queue_depth_tracks_running_max() {
        let mut t = TelemetryTracker::new();
        t.record_queue_depth(10);
        t.record_queue_depth(50);
        t.record_queue_depth(20);
        assert_eq!(t.queue_depth_max, 50);
        assert_eq!(t.queue_depth_samples, vec![10, 50, 20]);
    }

    #[test]
    fn collect_snapshot_returns_none_when_disabled() {
        let mut t = TelemetryTracker::new();
        let cfg = TelemetryConfig {
            enabled: false,
            ..full_config()
        };
        assert!(t.collect_snapshot(&cfg, 42).is_none());
    }

    #[test]
    fn collect_snapshot_populates_metrics_and_resets_interval() {
        let mut t = TelemetryTracker::new();
        t.record_latency(10.0);
        t.record_latency(30.0);
        t.record_file_processed(2048);
        t.record_file_processed(1024);
        t.record_queue_depth(7);
        t.record_queue_depth(13);

        let snap = t.collect_snapshot(&full_config(), 99).unwrap();

        assert_eq!(snap.queue_depth_current, 99);
        // (10 + 30) / 2 = 20.0
        assert_eq!(snap.avg_latency_ms, Some(20.0));
        // p95 of two samples: index = floor(0.95 * 2) = 1 → sorted[1] = 30.0
        assert_eq!(snap.p95_latency_ms, Some(30.0));
        // (7 + 13) / 2 = 10.0
        assert_eq!(snap.queue_depth_avg, 10.0);
        assert_eq!(snap.queue_depth_max, 13);
        assert!(snap.throughput_files_per_sec > 0.0);
        assert!(snap.throughput_bytes_per_sec > 0.0);

        // Interval counters are reset after collection so the next snapshot
        // measures only the next window.
        assert_eq!(t.files_processed_interval, 0);
        assert_eq!(t.bytes_processed_interval, 0);
        assert!(t.queue_depth_samples.is_empty());
        assert_eq!(t.queue_depth_max, 0);
    }

    #[test]
    fn collect_snapshot_honours_individual_toggles() {
        let mut t = TelemetryTracker::new();
        t.record_latency(5.0);
        t.record_queue_depth(3);
        t.record_file_processed(100);

        // Disable everything except cpu (which is still optional and may be
        // None on this host); the disabled metrics must come back unset.
        let cfg = TelemetryConfig {
            enabled: true,
            history_retention: 3,
            collection_interval_secs: 60,
            cpu_usage: false,
            memory_usage: false,
            latency: false,
            queue_depth: false,
            throughput: false,
        };

        let snap = t.collect_snapshot(&cfg, 0).unwrap();
        assert!(snap.cpu_usage_percent.is_none());
        assert!(snap.memory_rss_mb.is_none());
        assert!(snap.memory_heap_mb.is_none());
        assert!(snap.avg_latency_ms.is_none());
        assert!(snap.p95_latency_ms.is_none());
        assert!(snap.p99_latency_ms.is_none());
        assert_eq!(snap.queue_depth_avg, 0.0);
        assert_eq!(snap.queue_depth_max, 0);
        assert_eq!(snap.throughput_files_per_sec, 0.0);
        assert_eq!(snap.throughput_bytes_per_sec, 0.0);
    }

    #[test]
    fn history_ring_buffer_evicts_oldest_past_retention() {
        let mut t = TelemetryTracker::new();
        let cfg = TelemetryConfig {
            history_retention: 2,
            ..full_config()
        };

        // Tag each snapshot via a distinct queue_depth_current so we can
        // identify which ones survive eviction.
        for depth in [1usize, 2, 3, 4] {
            t.collect_snapshot(&cfg, depth);
        }

        let history = t.get_history();
        assert_eq!(history.len(), 2);
        // The oldest two (1, 2) were evicted; (3, 4) remain in order.
        assert_eq!(history[0].queue_depth_current, 3);
        assert_eq!(history[1].queue_depth_current, 4);
    }

    #[test]
    fn empty_latencies_produce_none_avg_but_zero_throughput() {
        let mut t = TelemetryTracker::new();
        let snap = t.collect_snapshot(&full_config(), 0).unwrap();
        assert!(snap.avg_latency_ms.is_none());
        assert!(snap.p95_latency_ms.is_none());
        assert!(snap.p99_latency_ms.is_none());
        // No files/bytes recorded → throughput is exactly 0.0, not NaN.
        assert_eq!(snap.throughput_files_per_sec, 0.0);
        assert_eq!(snap.throughput_bytes_per_sec, 0.0);
    }
}
