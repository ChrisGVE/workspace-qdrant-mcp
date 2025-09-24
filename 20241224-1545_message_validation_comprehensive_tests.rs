// Comprehensive unit tests for enhanced message validation
// Test file: 20241224-1545_message_validation_comprehensive_tests.rs

#[cfg(test)]
mod comprehensive_tests {
    use super::*;
    use crate::config::{MessageConfig, CompressionConfig, StreamingConfig};
    use crate::grpc::message_validation::{MessageValidator, ContentType};
    use std::time::Duration;
    use tonic::{Request, Response};

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
    fn test_compression_failure_handling() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        // Test decompression with invalid data
        let invalid_data = b"This is not compressed data";
        let result = validator.decompress_message(invalid_data);

        assert!(result.is_err());

        let stats = validator.get_stats();
        assert!(stats.compression_failures > 0);
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
    fn test_concurrent_stream_limit_enforcement() {
        let mut config = create_enhanced_streaming_config();
        config.max_concurrent_streams = 3; // Low limit for testing

        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            config,
        );

        let mut streams = Vec::new();

        // Register up to the limit
        for _ in 0..3 {
            streams.push(validator.register_stream(None).unwrap());
        }

        // Attempting to register beyond the limit should fail
        let result = validator.register_stream(None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Maximum concurrent streams"));
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
    fn test_serialization_performance() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        let start = std::time::Instant::now();

        // Test multiple compression operations
        for i in 0..50 {
            let data = format!("Performance test data {}", i).repeat(100);
            let _ = validator.compress_message(&data.as_bytes());
        }

        let duration = start.elapsed();
        println!("50 compression operations took: {:?}", duration);

        // Should complete within reasonable time (adjust threshold as needed)
        assert!(duration < Duration::from_secs(1));

        let stats = validator.get_stats();
        assert_eq!(stats.total_messages, 50);
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
    fn test_statistics_accuracy() {
        let validator = MessageValidator::new(
            create_enhanced_message_config(),
            create_enhanced_compression_config(),
            create_enhanced_streaming_config(),
        );

        validator.reset_stats();

        // Perform known operations
        let data1 = vec![b'A'; 2000];
        let data2 = vec![b'B'; 3000];
        let data3 = vec![b'C'; 4000];

        let _ = validator.compress_message(&data1);
        let _ = validator.compress_message(&data2);
        let _ = validator.compress_message(&data3);

        let stats = validator.get_stats();

        assert_eq!(stats.total_messages, 3);
        assert_eq!(stats.total_bytes_uncompressed, 2000 + 3000 + 4000);
        assert!(stats.total_bytes_compressed < stats.total_bytes_uncompressed);
        assert!(stats.average_compression_ratio < 1.0); // Should compress well
        assert_eq!(stats.compression_failures, 0);
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