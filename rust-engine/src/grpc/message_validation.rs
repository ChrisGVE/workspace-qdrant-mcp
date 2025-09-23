//! Message validation and monitoring for gRPC communication
//!
//! This module provides comprehensive message size validation, compression monitoring,
//! and streaming control for gRPC operations.

use crate::config::{MessageConfig, CompressionConfig, StreamingConfig};
use anyhow::{Result, anyhow};
use flate2::{Compression, write::GzEncoder, read::GzDecoder};
use std::io::{Write, Read};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};
use tracing::{debug, warn, error, info};

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
}

/// Message validation and monitoring system
pub struct MessageValidator {
    pub message_config: MessageConfig,
    pub compression_config: CompressionConfig,
    pub streaming_config: StreamingConfig,
    stats: Arc<MessageValidationStats>,
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
            }),
        }
    }

    /// Validate incoming message size
    pub fn validate_incoming_message<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if !self.message_config.enable_size_validation {
            return Ok(());
        }

        // Estimate message size (simplified - in real scenarios you'd need proper serialization)
        let estimated_size = std::mem::size_of::<Request<T>>();

        if estimated_size > self.message_config.max_incoming_message_size {
            self.stats.oversized_messages.fetch_add(1, Ordering::Relaxed);
            error!(
                "Incoming message size {} exceeds limit {}",
                estimated_size, self.message_config.max_incoming_message_size
            );
            return Err(Status::invalid_argument(format!(
                "Message size {} exceeds maximum allowed size {}",
                estimated_size, self.message_config.max_incoming_message_size
            )));
        }

        debug!("Validated incoming message size: {}", estimated_size);
        Ok(())
    }

    /// Validate outgoing message size
    pub fn validate_outgoing_message<T>(&self, response: &Response<T>) -> Result<(), Status> {
        if !self.message_config.enable_size_validation {
            return Ok(());
        }

        // Estimate message size (simplified - in real scenarios you'd need proper serialization)
        let estimated_size = std::mem::size_of::<Response<T>>();

        if estimated_size > self.message_config.max_outgoing_message_size {
            self.stats.oversized_messages.fetch_add(1, Ordering::Relaxed);
            error!(
                "Outgoing message size {} exceeds limit {}",
                estimated_size, self.message_config.max_outgoing_message_size
            );
            return Err(Status::internal(format!(
                "Response size {} exceeds maximum allowed size {}",
                estimated_size, self.message_config.max_outgoing_message_size
            )));
        }

        debug!("Validated outgoing message size: {}", estimated_size);
        Ok(())
    }

    /// Compress message data if configured
    pub fn compress_message(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.compression_config.enable_gzip || data.len() < self.compression_config.compression_threshold {
            // No compression needed
            self.update_compression_stats(data.len(), data.len());
            return Ok(data.to_vec());
        }

        let start = Instant::now();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_config.compression_level));

        match encoder.write_all(data) {
            Ok(_) => match encoder.finish() {
                Ok(compressed) => {
                    let compression_time = start.elapsed();
                    let compression_ratio = compressed.len() as f64 / data.len() as f64;

                    if self.compression_config.enable_compression_monitoring {
                        info!(
                            "Compressed {} bytes to {} bytes (ratio: {:.2}) in {:?}",
                            data.len(), compressed.len(), compression_ratio, compression_time
                        );
                    }

                    self.update_compression_stats(data.len(), compressed.len());
                    Ok(compressed)
                },
                Err(e) => {
                    self.stats.compression_failures.fetch_add(1, Ordering::Relaxed);
                    error!("Compression finish failed: {}", e);
                    Err(anyhow!("Compression failed: {}", e))
                }
            },
            Err(e) => {
                self.stats.compression_failures.fetch_add(1, Ordering::Relaxed);
                error!("Compression write failed: {}", e);
                Err(anyhow!("Compression failed: {}", e))
            }
        }
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

    /// Register a new streaming operation
    pub fn register_stream(&self) -> Result<StreamHandle> {
        let current_streams = self.stats.active_streams.load(Ordering::Relaxed);

        if current_streams >= self.streaming_config.max_concurrent_streams as usize {
            return Err(anyhow!(
                "Maximum concurrent streams ({}) exceeded",
                self.streaming_config.max_concurrent_streams
            ));
        }

        self.stats.active_streams.fetch_add(1, Ordering::Relaxed);
        info!("Registered new stream, active streams: {}", current_streams + 1);

        Ok(StreamHandle {
            stats: Arc::clone(&self.stats),
            timeout: Duration::from_secs(self.streaming_config.stream_timeout_secs),
            buffer_size: self.streaming_config.stream_buffer_size,
            flow_control_enabled: self.streaming_config.enable_flow_control,
        })
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

        MessageStats {
            total_messages: self.stats.total_messages.load(Ordering::Relaxed),
            total_bytes_uncompressed: total_uncompressed,
            total_bytes_compressed: total_compressed,
            average_compression_ratio: compression_ratio,
            oversized_messages: self.stats.oversized_messages.load(Ordering::Relaxed),
            compression_failures: self.stats.compression_failures.load(Ordering::Relaxed),
            active_streams: self.stats.active_streams.load(Ordering::Relaxed) as u32,
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
    }

    /// Update compression statistics
    fn update_compression_stats(&self, uncompressed_size: usize, compressed_size: usize) {
        self.stats.total_messages.fetch_add(1, Ordering::Relaxed);
        self.stats.total_bytes_uncompressed.fetch_add(uncompressed_size as u64, Ordering::Relaxed);
        self.stats.total_bytes_compressed.fetch_add(compressed_size as u64, Ordering::Relaxed);
    }
}

/// Handle for streaming operations
pub struct StreamHandle {
    stats: Arc<MessageValidationStats>,
    timeout: Duration,
    buffer_size: usize,
    flow_control_enabled: bool,
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
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        let remaining = self.stats.active_streams.fetch_sub(1, Ordering::Relaxed);
        debug!("Stream closed, remaining active streams: {}", remaining - 1);
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
        }
    }

    fn create_test_compression_config() -> CompressionConfig {
        CompressionConfig {
            enable_gzip: true,
            compression_threshold: 100,
            compression_level: 6,
            enable_streaming_compression: true,
            enable_compression_monitoring: true,
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
        let stream1 = validator.register_stream().unwrap();
        assert_eq!(stream1.active_streams(), 1);

        let stream2 = validator.register_stream().unwrap();
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
            handles.push(validator.register_stream().unwrap());
        }

        // Attempting to register another should fail
        let result = validator.register_stream();
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

        let handle = validator.register_stream().unwrap();

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
        let handles: Vec<_> = (0..5).map(|_| validator.register_stream().unwrap()).collect();

        assert_eq!(handles[0].active_streams(), 5);

        // Drop half the handles by taking them out of the vec
        let _handle1 = handles.swap_remove(0);
        let _handle2 = handles.swap_remove(0);

        // Remaining handles should see updated count
        assert_eq!(handles[2].active_streams(), 3);
    }
}