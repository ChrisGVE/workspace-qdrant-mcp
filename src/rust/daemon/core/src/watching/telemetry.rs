//! Telemetry and statistics for the file watching system

use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

use serde::{Deserialize, Serialize};
use sysinfo::{System, Pid};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
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

        let elapsed = self.last_collection.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        self.system.refresh_process(self.pid);

        let cpu_usage_percent = if config.cpu_usage {
            self.system.process(self.pid).map(|process| process.cpu_usage() as f64)
        } else {
            None
        };

        let (memory_rss_mb, memory_heap_mb) = if config.memory_usage {
            if let Some(process) = self.system.process(self.pid) {
                let rss_mb = process.memory() as f64 / 1024.0 / 1024.0;
                let heap_mb = (process.virtual_memory() - process.memory()) as f64 / 1024.0 / 1024.0;
                (Some(rss_mb), Some(heap_mb))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let (avg_latency_ms, p95_latency_ms, p99_latency_ms) = if config.latency {
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
        };

        let (queue_depth_avg, queue_depth_max) = if config.queue_depth {
            let avg = if !self.queue_depth_samples.is_empty() {
                self.queue_depth_samples.iter().sum::<usize>() as f64 / self.queue_depth_samples.len() as f64
            } else {
                0.0
            };
            (avg, self.queue_depth_max)
        } else {
            (0.0, 0)
        };

        let (throughput_files_per_sec, throughput_bytes_per_sec) = if config.throughput {
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
        };

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

        // Reset interval counters
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
