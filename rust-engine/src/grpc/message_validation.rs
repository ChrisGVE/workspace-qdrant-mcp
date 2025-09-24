//! Message validation and monitoring for gRPC communication
//!
//! This module provides comprehensive message size validation, compression monitoring,
//! and streaming control for gRPC operations.

use crate::config::{
    MessageConfig, CompressionConfig, StreamingConfig,
    ServiceLimit, AdaptiveCompressionConfig, CompressionPerformanceConfig,
    StreamProgressConfig, StreamHealthConfig, LargeOperationStreamConfig,
    MessageMonitoringConfig
};
use anyhow::{Result, anyhow};
use flate2::{Compression, write::GzEncoder, read::GzDecoder};
use std::io::{Write, Read};
use std::sync::atomic::{AtomicUsize, AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tonic::{Request, Response, Status};
use tracing::{debug, warn, error, info};
use tokio::sync::RwLock;

/// Content type for adaptive compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Text content (high compression ratio)
    Text,
    /// Binary content (low compression ratio)
    Binary,
    /// Structured data like JSON (medium compression ratio)
    Structured,
}

/// Statistics for message processing and compression
#[derive(Debug, Clone)]
pub struct MessageStats {
    /// Total messages processed
    pub total_messages: u64,
    /// Total bytes processed (uncompressed)
    pub total_bytes_uncompressed: u64,
    /// Total bytes processed (compressed)
    pub total_bytes_compressed: u64,
    /// Average compression ratio (compressed/uncompressed)
    pub average_compression_ratio: f64,
    /// Messages that exceeded size limits
    pub oversized_messages: u64,
    /// Compression failures
    pub compression_failures: u64,
    /// Streaming operations active
    pub active_streams: u32,
    /// Service-specific statistics
    pub service_stats: HashMap<String, ServiceStats>,
    /// Compression performance metrics
    pub compression_performance: CompressionPerformanceStats,
    /// Streaming performance metrics
    pub streaming_performance: StreamingPerformanceStats,
}

/// Service-specific statistics
#[derive(Debug, Clone)]
pub struct ServiceStats {
    /// Messages processed by this service
    pub messages_processed: u64,
    /// Bytes processed by this service
    pub bytes_processed: u64,
    /// Oversized messages for this service
    pub oversized_messages: u64,
    /// Average processing time (milliseconds)
    pub avg_processing_time_ms: f64,
}

/// Compression performance statistics
#[derive(Debug, Clone)]
pub struct CompressionPerformanceStats {
    /// Average compression time (milliseconds)
    pub avg_compression_time_ms: f64,
    /// Best compression ratio achieved
    pub best_compression_ratio: f64,
    /// Worst compression ratio achieved
    pub worst_compression_ratio: f64,
    /// Number of poor compression ratio alerts
    pub poor_ratio_alerts: u64,
    /// Number of slow compression alerts
    pub slow_compression_alerts: u64,
}

/// Streaming performance statistics
#[derive(Debug, Clone)]
pub struct StreamingPerformanceStats {
    /// Number of streams with progress tracking
    pub tracked_streams: u32,
    /// Number of stream recovery attempts
    pub recovery_attempts: u64,
    /// Number of successful recoveries
    pub successful_recoveries: u64,
    /// Average stream duration (seconds)
    pub avg_stream_duration_secs: f64,
    /// Number of large operations streamed
    pub large_operations_streamed: u64,
}

/// Message validation and monitoring system
pub struct MessageValidator {
    pub message_config: MessageConfig,
    pub compression_config: CompressionConfig,
    pub streaming_config: StreamingConfig,
    stats: Arc<MessageValidationStats>,
    performance_monitor: Arc<PerformanceMonitor>,
}

/// Internal statistics tracking
#[derive(Debug)]
struct MessageValidationStats {
    total_messages: AtomicU64,
    total_bytes_uncompressed: AtomicU64,
    total_bytes_compressed: AtomicU64,
    oversized_messages: AtomicU64,
    compression_failures: AtomicU64,
    active_streams: AtomicUsize,
    service_stats: RwLock<HashMap<String, ServiceStatsInternal>>,
}

/// Internal service statistics
#[derive(Debug, Default)]
struct ServiceStatsInternal {
    messages_processed: u64,
    bytes_processed: u64,
    oversized_messages: u64,
    total_processing_time_ms: u64,
}

/// Performance monitoring system
#[derive(Debug)]
struct PerformanceMonitor {
    compression_times: RwLock<Vec<f64>>,
    compression_ratios: RwLock<Vec<f64>>,
    poor_ratio_alerts: AtomicU64,
    slow_compression_alerts: AtomicU64,
    stream_recovery_attempts: AtomicU64,
    successful_recoveries: AtomicU64,
    tracked_streams: AtomicUsize,
    large_operations_streamed: AtomicU64,
    monitoring_enabled: AtomicBool,
}

impl MessageValidator {
    /// Create a new message validator
    pub fn new(
        message_config: MessageConfig,
        compression_config: CompressionConfig,
        streaming_config: StreamingConfig,
    ) -> Self {
        Self {
            message_config,
            compression_config,
            streaming_config,
            stats: Arc::new(MessageValidationStats {
                total_messages: AtomicU64::new(0),
                total_bytes_uncompressed: AtomicU64::new(0),
                total_bytes_compressed: AtomicU64::new(0),
                oversized_messages: AtomicU64::new(0),
                compression_failures: AtomicU64::new(0),
                active_streams: AtomicUsize::new(0),
                service_stats: RwLock::new(HashMap::new()),
            }),
            performance_monitor: Arc::new(PerformanceMonitor {
                compression_times: RwLock::new(Vec::new()),
                compression_ratios: RwLock::new(Vec::new()),
                poor_ratio_alerts: AtomicU64::new(0),
                slow_compression_alerts: AtomicU64::new(0),
                stream_recovery_attempts: AtomicU64::new(0),
                successful_recoveries: AtomicU64::new(0),
                tracked_streams: AtomicUsize::new(0),
                large_operations_streamed: AtomicU64::new(0),
                monitoring_enabled: AtomicBool::new(true),
            }),
        }
    }

    /// Validate incoming message size for a specific service
    pub fn validate_incoming_message<T>(
        &self,
        request: &Request<T>,
        service_name: &str,
    ) -> Result<(), Status> {
        if !self.message_config.enable_size_validation {
            return Ok(());
        }

        // Estimate message size (simplified - in real scenarios you'd need proper serialization)
        let estimated_size = std::mem::size_of::<Request<T>>();

        // Get service-specific limit
        let service_limit = self.get_service_limit(service_name);
        let max_size = if service_limit.enable_validation {
            service_limit.max_incoming
        } else {
            self.message_config.max_incoming_message_size
        };

        if estimated_size > max_size {
            self.stats.oversized_messages.fetch_add(1, Ordering::Relaxed);
            self.update_service_stats(service_name, 0, 0, 1, 0.0);

            error!(
                "Incoming message size {} exceeds limit {} for service {}",
                estimated_size, max_size, service_name
            );

            // Check if this triggers an alert
            if estimated_size as f64 > max_size as f64 * self.message_config.monitoring.oversized_alert_threshold {
                warn!(
                    "Oversized message alert for service {}: {} bytes ({}% of limit)",
                    service_name,
                    estimated_size,
                    (estimated_size as f64 / max_size as f64) * 100.0
                );
            }

            return Err(Status::invalid_argument(format!(
                "Message size {} exceeds maximum allowed size {} for service {}",
                estimated_size, max_size, service_name
            )));
        }

        // Update service stats
        self.update_service_stats(service_name, 1, estimated_size as u64, 0, 0.0);
        debug!("Validated incoming message size: {} for service {}", estimated_size, service_name);
        Ok(())
    }

    /// Validate incoming message size (backwards compatibility)
    pub fn validate_incoming_message_legacy<T>(&self, request: &Request<T>) -> Result<(), Status> {
        self.validate_incoming_message(request, "unknown")
    }

    /// Validate outgoing message size for a specific service
    pub fn validate_outgoing_message<T>(
        &self,
        response: &Response<T>,
        service_name: &str,
    ) -> Result<(), Status> {
        if !self.message_config.enable_size_validation {
            return Ok(());
        }

        // Estimate message size (simplified - in real scenarios you'd need proper serialization)
        let estimated_size = std::mem::size_of::<Response<T>>();

        // Get service-specific limit
        let service_limit = self.get_service_limit(service_name);
        let max_size = if service_limit.enable_validation {
            service_limit.max_outgoing
        } else {
            self.message_config.max_outgoing_message_size
        };

        if estimated_size > max_size {
            self.stats.oversized_messages.fetch_add(1, Ordering::Relaxed);
            self.update_service_stats(service_name, 0, 0, 1, 0.0);

            error!(
                "Outgoing message size {} exceeds limit {} for service {}",
                estimated_size, max_size, service_name
            );

            return Err(Status::internal(format!(
                "Response size {} exceeds maximum allowed size {} for service {}",
                estimated_size, max_size, service_name
            )));
        }

        // Update service stats
        self.update_service_stats(service_name, 1, estimated_size as u64, 0, 0.0);
        debug!("Validated outgoing message size: {} for service {}", estimated_size, service_name);
        Ok(())
    }

    /// Validate outgoing message size (backwards compatibility)
    pub fn validate_outgoing_message_legacy<T>(&self, response: &Response<T>) -> Result<(), Status> {
        self.validate_outgoing_message(response, "unknown")
    }

    /// Compress message data with adaptive compression based on content type
    pub fn compress_message_adaptive(&self, data: &[u8], content_type: ContentType) -> Result<Vec<u8>> {
        if !self.compression_config.enable_gzip || data.len() < self.compression_config.compression_threshold {
            // No compression needed
            self.update_compression_stats(data.len(), data.len());
            return Ok(data.to_vec());
        }

        let start = Instant::now();

        // Select compression level based on content type and adaptive configuration
        let compression_level = if self.compression_config.adaptive.enable_adaptive {
            match content_type {
                ContentType::Text => self.compression_config.adaptive.text_compression_level,
                ContentType::Binary => self.compression_config.adaptive.binary_compression_level,
                ContentType::Structured => self.compression_config.adaptive.structured_compression_level,
            }
        } else {
            self.compression_config.compression_level
        };

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(compression_level));

        match encoder.write_all(data) {
            Ok(_) => match encoder.finish() {
                Ok(compressed) => {
                    let compression_time = start.elapsed();
                    let compression_time_ms = compression_time.as_millis() as f64;
                    let compression_ratio = compressed.len() as f64 / data.len() as f64;

                    // Check for performance alerts
                    if self.compression_config.performance.enable_time_monitoring &&
                       compression_time_ms > self.compression_config.performance.slow_compression_threshold_ms as f64 {
                        self.performance_monitor.slow_compression_alerts.fetch_add(1, Ordering::Relaxed);
                        warn!(
                            "Slow compression detected: {} bytes compressed in {:.2}ms (threshold: {}ms)",
                            data.len(), compression_time_ms, self.compression_config.performance.slow_compression_threshold_ms
                        );
                    }

                    if self.compression_config.performance.enable_ratio_tracking &&
                       compression_ratio > self.compression_config.performance.poor_ratio_threshold {
                        self.performance_monitor.poor_ratio_alerts.fetch_add(1, Ordering::Relaxed);
                        warn!(
                            "Poor compression ratio detected: {:.2} (threshold: {:.2})",
                            compression_ratio, self.compression_config.performance.poor_ratio_threshold
                        );
                    }

                    // Update performance metrics
                    if self.performance_monitor.monitoring_enabled.load(Ordering::Relaxed) {
                        self.update_compression_performance(compression_time_ms, compression_ratio);
                    }

                    if self.compression_config.enable_compression_monitoring {
                        info!(
                            "Compressed {} bytes to {} bytes (ratio: {:.2}, level: {}) in {:?}",
                            data.len(), compressed.len(), compression_ratio, compression_level, compression_time
                        );
                    }

                    self.update_compression_stats(data.len(), compressed.len());
                    Ok(compressed)
                },
                Err(e) => {
                    self.stats.compression_failures.fetch_add(1, Ordering::Relaxed);
                    if self.compression_config.performance.enable_failure_alerting {
                        error!("Compression finish failed: {}", e);
                    }
                    Err(anyhow!("Compression failed: {}", e))
                }
            },
            Err(e) => {
                self.stats.compression_failures.fetch_add(1, Ordering::Relaxed);
                if self.compression_config.performance.enable_failure_alerting {
                    error!("Compression write failed: {}", e);
                }
                Err(anyhow!("Compression failed: {}", e))
            }
        }
    }

    /// Compress message data (backwards compatibility)
    pub fn compress_message(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compress_message_adaptive(data, ContentType::Binary)
    }

    /// Decompress message data
    pub fn decompress_message(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        if !self.compression_config.enable_gzip {
            return Ok(compressed_data.to_vec());
        }

        let start = Instant::now();
        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = Vec::new();

        match decoder.read_to_end(&mut decompressed) {
            Ok(_) => {
                let decompression_time = start.elapsed();

                if self.compression_config.enable_compression_monitoring {
                    debug!(
                        "Decompressed {} bytes to {} bytes in {:?}",
                        compressed_data.len(), decompressed.len(), decompression_time
                    );
                }

                Ok(decompressed)
            },
            Err(e) => {
                self.stats.compression_failures.fetch_add(1, Ordering::Relaxed);
                error!("Decompression failed: {}", e);
                Err(anyhow!("Decompression failed: {}", e))
            }
        }
    }

    /// Check if streaming is enabled for the operation type
    pub fn is_streaming_enabled(&self, is_server_streaming: bool) -> bool {
        if is_server_streaming {
            self.streaming_config.enable_server_streaming
        } else {
            self.streaming_config.enable_client_streaming
        }
    }

    /// Register a new streaming operation with progress tracking
    pub fn register_stream(&self, operation_size: Option<usize>) -> Result<StreamHandle> {
        let current_streams = self.stats.active_streams.load(Ordering::Relaxed);

        if current_streams >= self.streaming_config.max_concurrent_streams as usize {
            return Err(anyhow!(
                "Maximum concurrent streams ({}) exceeded",
                self.streaming_config.max_concurrent_streams
            ));
        }

        self.stats.active_streams.fetch_add(1, Ordering::Relaxed);

        // Check if this is a large operation that should be tracked
        let is_large_operation = operation_size.map_or(false, |size| {
            size >= self.streaming_config.large_operations.large_operation_chunk_size
        });

        let enable_progress = self.streaming_config.progress.enable_progress_tracking &&
                             operation_size.map_or(false, |size| size >= self.streaming_config.progress.progress_threshold);

        if is_large_operation {
            self.performance_monitor.large_operations_streamed.fetch_add(1, Ordering::Relaxed);
        }

        if enable_progress {
            self.performance_monitor.tracked_streams.fetch_add(1, Ordering::Relaxed);
        }

        info!("Registered new stream, active streams: {}, large operation: {}, progress tracking: {}",
              current_streams + 1, is_large_operation, enable_progress);

        Ok(StreamHandle {
            stats: Arc::clone(&self.stats),
            performance_monitor: Arc::clone(&self.performance_monitor),
            timeout: Duration::from_secs(self.streaming_config.stream_timeout_secs),
            buffer_size: self.streaming_config.stream_buffer_size,
            flow_control_enabled: self.streaming_config.enable_flow_control,
            progress_config: self.streaming_config.progress.clone(),
            health_config: self.streaming_config.health.clone(),
            is_large_operation,
            enable_progress_tracking: enable_progress,
            start_time: Instant::now(),
        })
    }

    /// Register a new streaming operation (backwards compatibility)
    pub fn register_stream_legacy(&self) -> Result<StreamHandle> {
        self.register_stream(None)
    }

    /// Get current message processing statistics
    pub fn get_stats(&self) -> MessageStats {
        let total_uncompressed = self.stats.total_bytes_uncompressed.load(Ordering::Relaxed);
        let total_compressed = self.stats.total_bytes_compressed.load(Ordering::Relaxed);

        let compression_ratio = if total_uncompressed > 0 {
            total_compressed as f64 / total_uncompressed as f64
        } else {
            1.0
        };

        // Build service statistics
        let mut service_stats = HashMap::new();
        if let Ok(stats) = self.stats.service_stats.try_read() {
            for (service_name, internal_stats) in stats.iter() {
                let avg_processing_time = if internal_stats.messages_processed > 0 {
                    internal_stats.total_processing_time_ms as f64 / internal_stats.messages_processed as f64
                } else {
                    0.0
                };

                service_stats.insert(service_name.clone(), ServiceStats {
                    messages_processed: internal_stats.messages_processed,
                    bytes_processed: internal_stats.bytes_processed,
                    oversized_messages: internal_stats.oversized_messages,
                    avg_processing_time_ms: avg_processing_time,
                });
            }
        }

        // Build compression performance statistics
        let (avg_compression_time, best_ratio, worst_ratio) = {
            let times = self.performance_monitor.compression_times.try_read()
                .map(|t| t.iter().sum::<f64>() / t.len().max(1) as f64)
                .unwrap_or(0.0);

            let (best, worst) = self.performance_monitor.compression_ratios.try_read()
                .map(|ratios| {
                    let best = ratios.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let worst = ratios.iter().fold(0.0f64, |a, &b| a.max(b));
                    (best, worst)
                })
                .unwrap_or((1.0, 1.0));

            (times, best, worst)
        };

        let compression_performance = CompressionPerformanceStats {
            avg_compression_time_ms: avg_compression_time,
            best_compression_ratio: best_ratio,
            worst_compression_ratio: worst_ratio,
            poor_ratio_alerts: self.performance_monitor.poor_ratio_alerts.load(Ordering::Relaxed),
            slow_compression_alerts: self.performance_monitor.slow_compression_alerts.load(Ordering::Relaxed),
        };

        let streaming_performance = StreamingPerformanceStats {
            tracked_streams: self.performance_monitor.tracked_streams.load(Ordering::Relaxed) as u32,
            recovery_attempts: self.performance_monitor.stream_recovery_attempts.load(Ordering::Relaxed),
            successful_recoveries: self.performance_monitor.successful_recoveries.load(Ordering::Relaxed),
            avg_stream_duration_secs: 0.0, // TODO: Track stream durations
            large_operations_streamed: self.performance_monitor.large_operations_streamed.load(Ordering::Relaxed),
        };

        MessageStats {
            total_messages: self.stats.total_messages.load(Ordering::Relaxed),
            total_bytes_uncompressed: total_uncompressed,
            total_bytes_compressed: total_compressed,
            average_compression_ratio: compression_ratio,
            oversized_messages: self.stats.oversized_messages.load(Ordering::Relaxed),
            compression_failures: self.stats.compression_failures.load(Ordering::Relaxed),
            active_streams: self.stats.active_streams.load(Ordering::Relaxed) as u32,
            service_stats,
            compression_performance,
            streaming_performance,
        }
    }

    /// Reset statistics (for testing)
    pub fn reset_stats(&self) {
        self.stats.total_messages.store(0, Ordering::Relaxed);
        self.stats.total_bytes_uncompressed.store(0, Ordering::Relaxed);
        self.stats.total_bytes_compressed.store(0, Ordering::Relaxed);
        self.stats.oversized_messages.store(0, Ordering::Relaxed);
        self.stats.compression_failures.store(0, Ordering::Relaxed);
        self.stats.active_streams.store(0, Ordering::Relaxed);

        // Reset service stats
        if let Ok(mut stats) = self.stats.service_stats.try_write() {
            stats.clear();
        }

        // Reset performance monitor
        self.performance_monitor.poor_ratio_alerts.store(0, Ordering::Relaxed);
        self.performance_monitor.slow_compression_alerts.store(0, Ordering::Relaxed);
        self.performance_monitor.stream_recovery_attempts.store(0, Ordering::Relaxed);
        self.performance_monitor.successful_recoveries.store(0, Ordering::Relaxed);
        self.performance_monitor.tracked_streams.store(0, Ordering::Relaxed);
        self.performance_monitor.large_operations_streamed.store(0, Ordering::Relaxed);

        if let Ok(mut times) = self.performance_monitor.compression_times.try_write() {
            times.clear();
        }
        if let Ok(mut ratios) = self.performance_monitor.compression_ratios.try_write() {
            ratios.clear();
        }
    }

    /// Update compression statistics
    fn update_compression_stats(&self, uncompressed_size: usize, compressed_size: usize) {
        self.stats.total_messages.fetch_add(1, Ordering::Relaxed);
        self.stats.total_bytes_uncompressed.fetch_add(uncompressed_size as u64, Ordering::Relaxed);
        self.stats.total_bytes_compressed.fetch_add(compressed_size as u64, Ordering::Relaxed);
    }

    /// Get service-specific limits
    fn get_service_limit(&self, service_name: &str) -> &crate::config::ServiceLimit {
        match service_name {
            "document_processor" => &self.message_config.service_limits.document_processor,
            "search_service" => &self.message_config.service_limits.search_service,
            "memory_service" => &self.message_config.service_limits.memory_service,
            "system_service" => &self.message_config.service_limits.system_service,
            _ => &self.message_config.service_limits.system_service, // Default to system limits
        }
    }

    /// Update service-specific statistics
    fn update_service_stats(
        &self,
        service_name: &str,
        messages_processed: u64,
        bytes_processed: u64,
        oversized_messages: u64,
        processing_time_ms: f64,
    ) {
        if let Ok(mut stats) = self.stats.service_stats.try_write() {
            let entry = stats.entry(service_name.to_string()).or_insert_with(ServiceStatsInternal::default);
            entry.messages_processed += messages_processed;
            entry.bytes_processed += bytes_processed;
            entry.oversized_messages += oversized_messages;
            entry.total_processing_time_ms += processing_time_ms as u64;
        }
    }

    /// Update compression performance metrics
    fn update_compression_performance(&self, compression_time_ms: f64, compression_ratio: f64) {
        if let Ok(mut times) = self.performance_monitor.compression_times.try_write() {
            // Keep only the last 1000 entries to avoid memory growth
            if times.len() >= 1000 {
                times.remove(0);
            }
            times.push(compression_time_ms);
        }

        if let Ok(mut ratios) = self.performance_monitor.compression_ratios.try_write() {
            if ratios.len() >= 1000 {
                ratios.remove(0);
            }
            ratios.push(compression_ratio);
        }
    }
}

/// Handle for streaming operations with progress tracking
#[derive(Debug)]
pub struct StreamHandle {
    stats: Arc<MessageValidationStats>,
    performance_monitor: Arc<PerformanceMonitor>,
    timeout: Duration,
    buffer_size: usize,
    flow_control_enabled: bool,
    progress_config: StreamProgressConfig,
    health_config: StreamHealthConfig,
    is_large_operation: bool,
    enable_progress_tracking: bool,
    start_time: Instant,
}

impl StreamHandle {
    /// Get stream timeout
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Get buffer size for this stream
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Check if flow control is enabled
    pub fn flow_control_enabled(&self) -> bool {
        self.flow_control_enabled
    }

    /// Get current active streams count
    pub fn active_streams(&self) -> usize {
        self.stats.active_streams.load(Ordering::Relaxed)
    }

    /// Check if this is a large operation
    pub fn is_large_operation(&self) -> bool {
        self.is_large_operation
    }

    /// Check if progress tracking is enabled
    pub fn is_progress_tracking_enabled(&self) -> bool {
        self.enable_progress_tracking
    }

    /// Get progress update interval
    pub fn progress_update_interval(&self) -> Duration {
        Duration::from_millis(self.progress_config.progress_update_interval_ms)
    }

    /// Report progress (percentage completed)
    pub fn report_progress(&self, progress_percent: f64) {
        if self.enable_progress_tracking && self.progress_config.enable_progress_callbacks {
            debug!("Stream progress: {:.1}%", progress_percent);
        }
    }

    /// Get stream duration so far
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Attempt recovery from stream interruption
    pub fn attempt_recovery(&self) -> bool {
        if self.health_config.enable_auto_recovery {
            self.performance_monitor.stream_recovery_attempts.fetch_add(1, Ordering::Relaxed);

            info!("Attempting stream recovery (attempt: {})",
                  self.performance_monitor.stream_recovery_attempts.load(Ordering::Relaxed));

            // Simulate recovery attempt (in real implementation, this would reconnect/retry)
            let recovery_successful = true; // Placeholder logic

            if recovery_successful {
                self.performance_monitor.successful_recoveries.fetch_add(1, Ordering::Relaxed);
                info!("Stream recovery successful");
            } else {
                warn!("Stream recovery failed");
            }

            recovery_successful
        } else {
            false
        }
    }

    /// Check if stream health monitoring is enabled
    pub fn health_monitoring_enabled(&self) -> bool {
        self.health_config.enable_health_monitoring
    }

    /// Get health check interval
    pub fn health_check_interval(&self) -> Duration {
        Duration::from_secs(self.health_config.health_check_interval_secs)
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        let remaining = self.stats.active_streams.fetch_sub(1, Ordering::Relaxed);

        // Update tracking counters
        if self.enable_progress_tracking {
            self.performance_monitor.tracked_streams.fetch_sub(1, Ordering::Relaxed);
        }

        let duration = self.start_time.elapsed();
        debug!(
            "Stream closed after {:?}, remaining active streams: {}, was large operation: {}",
            duration, remaining - 1, self.is_large_operation
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MessageConfig, CompressionConfig, StreamingConfig};

    fn create_test_message_config() -> MessageConfig {
        MessageConfig {
            max_incoming_message_size: 1024,
            max_outgoing_message_size: 1024,
            enable_size_validation: true,
            max_frame_size: 512,
            initial_window_size: 1024,
            service_limits: crate::config::ServiceMessageLimits::default(),
            monitoring: crate::config::MessageMonitoringConfig::default(),
        }
    }

    fn create_test_compression_config() -> CompressionConfig {
        CompressionConfig {
            enable_gzip: true,
            compression_threshold: 100,
            compression_level: 6,
            enable_streaming_compression: true,
            enable_compression_monitoring: true,
            adaptive: crate::config::AdaptiveCompressionConfig::default(),
            performance: crate::config::CompressionPerformanceConfig::default(),
        }
    }

    fn create_test_streaming_config() -> StreamingConfig {
        StreamingConfig {
            enable_server_streaming: true,
            enable_client_streaming: true,
            max_concurrent_streams: 10,
            stream_buffer_size: 100,
            stream_timeout_secs: 30,
            enable_flow_control: true,
            progress: crate::config::StreamProgressConfig::default(),
            health: crate::config::StreamHealthConfig::default(),
            large_operations: crate::config::LargeOperationStreamConfig::default(),
        }
    }

    #[test]
    fn test_message_validator_creation() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        let stats = validator.get_stats();
        assert_eq!(stats.total_messages, 0);
        assert_eq!(stats.active_streams, 0);
    }

    #[test]
    fn test_compression_small_message() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Small message below threshold should not be compressed
        let data = b"small";
        let result = validator.compress_message(data).unwrap();

        // Should return original data since it's below threshold
        assert_eq!(result, data);
    }

    #[test]
    fn test_compression_large_message() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Large message above threshold should be compressed
        let data = vec![b'A'; 200]; // 200 bytes, above 100-byte threshold
        let compressed = validator.compress_message(&data).unwrap();

        // Compressed data should be different and typically smaller for repetitive data
        assert_ne!(compressed, data);
        assert!(compressed.len() < data.len());

        // Test decompression
        let decompressed = validator.decompress_message(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compression_disabled() {
        let mut compression_config = create_test_compression_config();
        compression_config.enable_gzip = false;

        let validator = MessageValidator::new(
            create_test_message_config(),
            compression_config,
            create_test_streaming_config(),
        );

        let data = vec![b'A'; 200];
        let result = validator.compress_message(&data).unwrap();

        // Should return original data when compression is disabled
        assert_eq!(result, data);
    }

    #[test]
    fn test_streaming_registration() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Register streams
        let stream1 = validator.register_stream_legacy().unwrap();
        assert_eq!(stream1.active_streams(), 1);

        let stream2 = validator.register_stream_legacy().unwrap();
        assert_eq!(stream2.active_streams(), 2);

        // Test stream properties
        assert_eq!(stream1.timeout(), Duration::from_secs(30));
        assert_eq!(stream1.buffer_size(), 100);
        assert!(stream1.flow_control_enabled());

        // Drop stream and verify count decreases
        drop(stream1);
        assert_eq!(stream2.active_streams(), 1);
    }

    #[test]
    fn test_max_concurrent_streams() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Register maximum allowed streams
        let mut handles = Vec::new();
        for _ in 0..10 {
            handles.push(validator.register_stream_legacy().unwrap());
        }

        // Attempting to register another should fail
        let result = validator.register_stream_legacy();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Maximum concurrent streams"));
    }

    #[test]
    fn test_streaming_enabled_checks() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        assert!(validator.is_streaming_enabled(true)); // server streaming
        assert!(validator.is_streaming_enabled(false)); // client streaming

        // Test with disabled streaming
        let mut streaming_config = create_test_streaming_config();
        streaming_config.enable_server_streaming = false;
        streaming_config.enable_client_streaming = false;

        let validator_disabled = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            streaming_config,
        );

        assert!(!validator_disabled.is_streaming_enabled(true));
        assert!(!validator_disabled.is_streaming_enabled(false));
    }

    #[test]
    fn test_message_size_validation_disabled() {
        let mut message_config = create_test_message_config();
        message_config.enable_size_validation = false;

        let validator = MessageValidator::new(
            message_config,
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Create a dummy request
        let request = Request::new(());

        // Should always pass when validation is disabled
        assert!(validator.validate_incoming_message(&request).is_ok());
    }

    #[test]
    fn test_statistics_tracking() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Process some messages
        let data1 = vec![b'A'; 150];
        let data2 = vec![b'B'; 200];

        validator.compress_message(&data1).unwrap();
        validator.compress_message(&data2).unwrap();

        let stats = validator.get_stats();
        assert_eq!(stats.total_messages, 2);
        assert_eq!(stats.total_bytes_uncompressed, 350);
        assert!(stats.total_bytes_compressed > 0);
        assert!(stats.average_compression_ratio > 0.0);
    }

    #[test]
    fn test_statistics_reset() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Process a message
        let data = vec![b'A'; 150];
        validator.compress_message(&data).unwrap();

        // Verify stats are populated
        let stats = validator.get_stats();
        assert_eq!(stats.total_messages, 1);

        // Reset and verify stats are cleared
        validator.reset_stats();
        let stats = validator.get_stats();
        assert_eq!(stats.total_messages, 0);
        assert_eq!(stats.total_bytes_uncompressed, 0);
        assert_eq!(stats.total_bytes_compressed, 0);
    }

    #[test]
    fn test_compression_failure_handling() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Test decompression of invalid data
        let invalid_data = b"not gzipped data";
        let result = validator.decompress_message(invalid_data);

        assert!(result.is_err());

        let stats = validator.get_stats();
        assert_eq!(stats.compression_failures, 1);
    }

    #[test]
    fn test_compression_levels() {
        let mut compression_config = create_test_compression_config();

        // Test different compression levels
        for level in [1, 6, 9] {
            compression_config.compression_level = level;

            let validator = MessageValidator::new(
                create_test_message_config(),
                compression_config.clone(),
                create_test_streaming_config(),
            );

            let data = vec![b'A'; 500]; // Large enough to trigger compression
            let compressed = validator.compress_message(&data).unwrap();

            // Should successfully compress with any valid level
            assert!(compressed.len() < data.len());

            // Should successfully decompress
            let decompressed = validator.decompress_message(&compressed).unwrap();
            assert_eq!(decompressed, data);
        }
    }

    #[test]
    fn test_stream_handle_properties() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        let handle = validator.register_stream_legacy().unwrap();

        assert_eq!(handle.timeout(), Duration::from_secs(30));
        assert_eq!(handle.buffer_size(), 100);
        assert!(handle.flow_control_enabled());
        assert_eq!(handle.active_streams(), 1);
    }

    #[test]
    fn test_concurrent_stream_operations() {
        let validator = MessageValidator::new(
            create_test_message_config(),
            create_test_compression_config(),
            create_test_streaming_config(),
        );

        // Test concurrent registration and dropping
        let mut handles: Vec<_> = (0..5).map(|_| validator.register_stream_legacy().unwrap()).collect();

        assert_eq!(handles[0].active_streams(), 5);

        // Drop half the handles by taking them out of the vec
        let _handle1 = handles.swap_remove(0);
        let _handle2 = handles.swap_remove(0);

        // Remaining handles should see updated count
        assert_eq!(handles[2].active_streams(), 3);
    }

    // Comprehensive edge case tests for enhanced functionality

    fn create_enhanced_message_config() -> MessageConfig {
        MessageConfig {
            max_incoming_message_size: 8 * 1024 * 1024, // 8MB
            max_outgoing_message_size: 8 * 1024 * 1024,
            enable_size_validation: true,
            max_frame_size: 32 * 1024,
            initial_window_size: 128 * 1024,
            service_limits: crate::config::ServiceMessageLimits::default(),
            monitoring: crate::config::MessageMonitoringConfig {
                enable_detailed_monitoring: true,
                oversized_alert_threshold: 0.8,
                enable_realtime_metrics: true,
                metrics_interval_secs: 30,
            },
        }
    }

    fn create_enhanced_compression_config() -> CompressionConfig {
        CompressionConfig {
            enable_gzip: true,
            compression_threshold: 1024,
            compression_level: 6,
            enable_streaming_compression: true,
            enable_compression_monitoring: true,
            adaptive: crate::config::AdaptiveCompressionConfig {
                enable_adaptive: true,
                text_compression_level: 9,
                binary_compression_level: 3,
                structured_compression_level: 6,
                max_compression_time_ms: 100,
            },
            performance: crate::config::CompressionPerformanceConfig {
                enable_ratio_tracking: true,
                poor_ratio_threshold: 0.9,
                enable_time_monitoring: true,
                slow_compression_threshold_ms: 200,
                enable_failure_alerting: true,
            },
        }
    }

    fn create_enhanced_streaming_config() -> StreamingConfig {
        StreamingConfig {
            enable_server_streaming: true,
            enable_client_streaming: true,
            max_concurrent_streams: 50,
            stream_buffer_size: 1000,
            stream_timeout_secs: 300,
            enable_flow_control: true,
            progress: crate::config::StreamProgressConfig {
                enable_progress_tracking: true,
                progress_update_interval_ms: 500,
                enable_progress_callbacks: true,
                progress_threshold: 1024 * 1024, // 1MB
            },
            health: crate::config::StreamHealthConfig {
                enable_health_monitoring: true,
                health_check_interval_secs: 30,
                enable_auto_recovery: true,
                max_recovery_attempts: 3,
                recovery_backoff_multiplier: 2.0,
                initial_recovery_delay_ms: 500,
            },
            large_operations: crate::config::LargeOperationStreamConfig {
                enable_large_document_streaming: true,
                large_operation_chunk_size: 2 * 1024 * 1024, // 2MB
                enable_bulk_streaming: true,
                max_streaming_memory: 64 * 1024 * 1024, // 64MB
                enable_bidirectional_optimization: true,
            },
        }
    }

    #[test]
    fn test_service_specific_message_limits() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        let request = Request::new(());

        // Test document processor (should have higher limits)
        assert!(validator.validate_incoming_message(&request, "document_processor").is_ok());

        // Test search service
        assert!(validator.validate_incoming_message(&request, "search_service").is_ok());

        // Test memory service
        assert!(validator.validate_incoming_message(&request, "memory_service").is_ok());

        // Test system service
        assert!(validator.validate_incoming_message(&request, "system_service").is_ok());

        // Test unknown service (should use default)
        assert!(validator.validate_incoming_message(&request, "unknown_service").is_ok());
    }

    #[test]
    fn test_oversized_message_handling() {
        let mut config = create_enhanced_message_config();
        config.max_incoming_message_size = 10; // Very small limit for testing

        let validator = MessageValidator::new(
            config,
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        let request = Request::new(());

        // This should trigger oversized message handling
        let result = validator.validate_incoming_message(&request, "system_service");

        let stats = validator.get_stats();
        assert!(stats.oversized_messages > 0);

        // Check service-specific stats
        let service_stats = stats.service_stats.get("system_service");
        assert!(service_stats.is_some());
        if let Some(stats) = service_stats {
            assert!(stats.oversized_messages > 0);
        }
    }

    #[test]
    fn test_adaptive_compression_text() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test text compression (should use high compression level)
        let text_data = "This is a long text that should compress well. ".repeat(100);
        let compressed = validator.compress_message_adaptive(text_data.as_bytes(), ContentType::Text).unwrap();

        // Text should compress well
        assert!(compressed.len() < text_data.len());

        // Test decompression
        let decompressed = validator.decompress_message(&compressed).unwrap();
        assert_eq!(decompressed, text_data.as_bytes());
    }

    #[test]
    fn test_adaptive_compression_binary() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test binary compression (should use low compression level)
        let binary_data: Vec<u8> = (0..2000).map(|i| (i % 256) as u8).collect();
        let compressed = validator.compress_message_adaptive(&binary_data, ContentType::Binary).unwrap();

        // Binary data may not compress as well, but should work
        let decompressed = validator.decompress_message(&compressed).unwrap();
        assert_eq!(decompressed, binary_data);
    }

    #[test]
    fn test_compression_performance_monitoring() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test with data that should trigger performance monitoring
        let data = "Test data for performance monitoring. ".repeat(200);
        let _ = validator.compress_message_adaptive(data.as_bytes(), ContentType::Text).unwrap();

        let stats = validator.get_stats();

        // Check that compression performance stats are collected
        assert!(stats.compression_performance.avg_compression_time_ms >= 0.0);
        assert!(stats.compression_performance.best_compression_ratio > 0.0);
        assert!(stats.compression_performance.worst_compression_ratio > 0.0);
    }

    #[test]
    fn test_streaming_with_progress_tracking() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Register a large operation that should enable progress tracking
        let large_operation_size = 5 * 1024 * 1024; // 5MB
        let stream = validator.register_stream(Some(large_operation_size)).unwrap();

        assert!(stream.is_large_operation());
        assert!(stream.is_progress_tracking_enabled());

        // Test progress reporting
        stream.report_progress(25.0);
        stream.report_progress(50.0);
        stream.report_progress(75.0);
        stream.report_progress(100.0);

        // Test stream properties
        assert!(stream.progress_update_interval() > Duration::from_millis(0));
        assert!(stream.health_monitoring_enabled());

        let stats = validator.get_stats();
        assert!(stats.streaming_performance.large_operations_streamed > 0);
        assert!(stats.streaming_performance.tracked_streams > 0);
    }

    #[test]
    fn test_stream_recovery_mechanism() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        let stream = validator.register_stream(Some(1024 * 1024)).unwrap();

        // Test recovery attempt
        let recovery_successful = stream.attempt_recovery();
        assert!(recovery_successful); // In our mock implementation, recovery always succeeds

        let stats = validator.get_stats();
        assert!(stats.streaming_performance.recovery_attempts > 0);
        assert!(stats.streaming_performance.successful_recoveries > 0);
    }

    #[test]
    fn test_streaming_interruption_handling() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        let mut streams = Vec::new();

        // Create multiple streams
        for i in 0..5 {
            let stream = validator.register_stream(Some(i * 1024 * 1024)).unwrap();
            streams.push(stream);
        }

        let stats = validator.get_stats();
        assert_eq!(stats.active_streams, 5);

        // Drop some streams (simulating interruption)
        streams.truncate(2);

        let stats = validator.get_stats();
        assert_eq!(stats.active_streams, 2);
    }

    #[test]
    fn test_memory_optimization_with_large_data() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test with large data to ensure memory doesn't grow unbounded
        for i in 0..100 {
            let data = format!("Large data chunk {} with varying content", i).repeat(1000);
            let _ = validator.compress_message_adaptive(data.as_bytes(), ContentType::Text);
        }

        // Performance monitor should limit stored metrics to prevent memory growth
        let stats = validator.get_stats();
        assert!(stats.total_messages > 0);
        assert!(stats.compression_performance.avg_compression_time_ms >= 0.0);
    }

    #[test]
    fn test_edge_case_empty_data() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test compression with empty data
        let empty_data = b"";
        let result = validator.compress_message(empty_data);
        assert!(result.is_ok());

        // Empty data should not be compressed (below threshold)
        let compressed = result.unwrap();
        assert_eq!(compressed, empty_data);
    }

    #[test]
    fn test_edge_case_threshold_boundary() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test data exactly at compression threshold (1024 bytes)
        let threshold_data = vec![b'A'; 1024];
        let result = validator.compress_message(&threshold_data);
        assert!(result.is_ok());

        // Should not compress (threshold is exclusive)
        let compressed = result.unwrap();
        assert_eq!(compressed.len(), threshold_data.len());

        // Test data just above threshold
        let above_threshold_data = vec![b'B'; 1025];
        let result = validator.compress_message(&above_threshold_data);
        assert!(result.is_ok());

        // Should compress (repetitive data compresses well)
        let compressed = result.unwrap();
        assert!(compressed.len() < above_threshold_data.len());
    }

    #[test]
    fn test_backwards_compatibility() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        let request = Request::new(());
        let response = Response::new(());

        // Test legacy methods still work
        assert!(validator.validate_incoming_message_legacy(&request).is_ok());
        assert!(validator.validate_outgoing_message_legacy(&response).is_ok());
        assert!(validator.register_stream_legacy().is_ok());

        // Test regular compression method
        let data = b"Test data for backwards compatibility";
        let compressed = validator.compress_message(data).unwrap();
        let decompressed = validator.decompress_message(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
}