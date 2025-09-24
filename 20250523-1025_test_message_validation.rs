//! Test script for message validation and compression functionality
//! This tests the core functionality without requiring full gRPC infrastructure

use workspace_qdrant_daemon::config::{MessageConfig, CompressionConfig, StreamingConfig};
use workspace_qdrant_daemon::grpc::message_validation::MessageValidator;
use std::sync::Arc;

fn main() {
    println!("Testing message validation and compression...");

    // Test 1: Create MessageValidator
    let validator = Arc::new(MessageValidator::new(
        MessageConfig {
            max_incoming_message_size: 1024 * 1024, // 1MB
            max_outgoing_message_size: 1024 * 1024,
            enable_size_validation: true,
            max_frame_size: 16384,
            initial_window_size: 65536,
        },
        CompressionConfig {
            enable_gzip: true,
            compression_threshold: 1024,
            compression_level: 6,
            enable_streaming_compression: true,
            enable_compression_monitoring: true,
        },
        StreamingConfig {
            enable_server_streaming: true,
            enable_client_streaming: true,
            max_concurrent_streams: 10,
            stream_buffer_size: 100,
            stream_timeout_secs: 30,
            enable_flow_control: true,
        },
    ));

    println!("✓ MessageValidator created successfully");

    // Test 2: Test compression of small message (should not compress)
    let small_data = b"small message";
    match validator.compress_message(small_data) {
        Ok(result) => {
            if result == small_data {
                println!("✓ Small message correctly not compressed");
            } else {
                println!("✗ Small message was compressed unexpectedly");
            }
        },
        Err(e) => {
            println!("✗ Error compressing small message: {}", e);
        }
    }

    // Test 3: Test compression of large message (should compress)
    let large_data = vec![b'A'; 2048]; // 2KB of repeated data
    match validator.compress_message(&large_data) {
        Ok(compressed) => {
            if compressed.len() < large_data.len() {
                println!("✓ Large message compressed successfully: {} -> {} bytes",
                        large_data.len(), compressed.len());

                // Test decompression
                match validator.decompress_message(&compressed) {
                    Ok(decompressed) => {
                        if decompressed == large_data {
                            println!("✓ Message decompressed correctly");
                        } else {
                            println!("✗ Decompressed data doesn't match original");
                        }
                    },
                    Err(e) => {
                        println!("✗ Error decompressing: {}", e);
                    }
                }
            } else {
                println!("✗ Large message was not compressed efficiently");
            }
        },
        Err(e) => {
            println!("✗ Error compressing large message: {}", e);
        }
    }

    // Test 4: Test streaming registration
    let stream1 = validator.register_stream();
    match stream1 {
        Ok(handle) => {
            println!("✓ Stream registered successfully, timeout: {:?}", handle.timeout());

            // Test statistics
            let stats = validator.get_stats();
            println!("✓ Stats retrieved: {} messages, {} active streams",
                    stats.total_messages, stats.active_streams);
        },
        Err(e) => {
            println!("✗ Error registering stream: {}", e);
        }
    }

    // Test 5: Test streaming capabilities check
    if validator.is_streaming_enabled(true) {
        println!("✓ Server streaming is enabled");
    } else {
        println!("✗ Server streaming should be enabled");
    }

    if validator.is_streaming_enabled(false) {
        println!("✓ Client streaming is enabled");
    } else {
        println!("✗ Client streaming should be enabled");
    }

    // Test 6: Test compression with disabled setting
    let validator_no_compression = Arc::new(MessageValidator::new(
        MessageConfig::default(),
        CompressionConfig {
            enable_gzip: false,
            compression_threshold: 1024,
            compression_level: 6,
            enable_streaming_compression: false,
            enable_compression_monitoring: false,
        },
        StreamingConfig::default(),
    ));

    let data = vec![b'B'; 2048];
    match validator_no_compression.compress_message(&data) {
        Ok(result) => {
            if result == data {
                println!("✓ Compression correctly disabled");
            } else {
                println!("✗ Compression should be disabled but data was modified");
            }
        },
        Err(e) => {
            println!("✗ Error with disabled compression: {}", e);
        }
    }

    println!("\n=== Message Validation Tests Complete ===");

    // Final stats
    let final_stats = validator.get_stats();
    println!("Final Statistics:");
    println!("  Total messages: {}", final_stats.total_messages);
    println!("  Bytes uncompressed: {}", final_stats.total_bytes_uncompressed);
    println!("  Bytes compressed: {}", final_stats.total_bytes_compressed);
    println!("  Compression ratio: {:.2}", final_stats.average_compression_ratio);
    println!("  Oversized messages: {}", final_stats.oversized_messages);
    println!("  Compression failures: {}", final_stats.compression_failures);
    println!("  Active streams: {}", final_stats.active_streams);
}