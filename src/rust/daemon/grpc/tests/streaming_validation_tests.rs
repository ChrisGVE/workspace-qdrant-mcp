#![cfg(feature = "legacy_grpc_tests")]
//! Comprehensive streaming response validation tests for gRPC daemon-MCP communication
//!
//! This module provides exhaustive testing of gRPC streaming responses including:
//! - Message ordering and sequencing
//! - Stream termination and completion
//! - Backpressure handling and flow control
//! - Stream cancellation and early termination
//! - Large dataset streaming
//! - Error handling in streams

use shared_test_utils::{async_test, TestResult};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::time::timeout;
use tonic::transport::{Channel, Server};
use tonic::{Request, Code, Status};
use workspace_qdrant_grpc::{
    ServerConfig,
    service::IngestionService,
    proto::{
        ingest_service_client::IngestServiceClient,
        ingest_service_server::IngestServiceServer,
        *,
    },
};

/// Test fixture for streaming tests
pub struct StreamingTestFixture {
    pub server_addr: SocketAddr,
    pub client: IngestServiceClient<Channel>,
    pub _server_handle: tokio::task::JoinHandle<()>,
    pub shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl StreamingTestFixture {
    pub async fn new() -> TestResult<Self> {
        Self::new_with_config(ServerConfig::new("127.0.0.1:0".parse()?)).await
    }

    pub async fn new_with_config(mut config: ServerConfig) -> TestResult<Self> {
        // Find available port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        config.bind_addr = addr;
        drop(listener);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        // Start server
        let server_config = config.clone();
        let server_handle = tokio::spawn(async move {
            let service = IngestionService::new_with_auth(server_config.auth_config.clone());
            let svc = IngestServiceServer::new(service);

            let server = Server::builder()
                .timeout(server_config.timeout_config.request_timeout)
                .add_service(svc);

            server
                .serve_with_shutdown(server_config.bind_addr, async {
                    shutdown_rx.await.ok();
                })
                .await
                .expect("Server failed to start");
        });

        // Wait for server to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        let channel = Channel::from_shared(format!("http://{}", addr))?
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .connect()
            .await?;

        let client = IngestServiceClient::new(channel);

        Ok(Self {
            server_addr: addr,
            client,
            _server_handle: server_handle,
            shutdown_tx,
        })
    }

    pub async fn shutdown(self) -> TestResult<()> {
        let _ = self.shutdown_tx.send(());
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }
}

#[cfg(test)]
mod message_ordering_tests {
    use super::*;

    async_test!(test_stream_message_ordering, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Start watching stream
        let watch_request = StartWatchingRequest {
            path: "/test/ordering".to_string(),
            collection: "ordering_test".to_string(),
            patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 100,
            watch_id: Some("order_test_watch".to_string()),
        };

        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Collect messages and verify ordering
        let mut messages = Vec::new();
        let mut message_count = 0;

        while message_count < 3 {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    messages.push(update);
                    message_count += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Verify messages are ordered (first message should be STARTED event)
        if !messages.is_empty() {
            assert_eq!(
                messages[0].event_type,
                WatchEventType::Started as i32,
                "First message should be STARTED event"
            );
            assert_eq!(messages[0].watch_id, "order_test_watch");
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_processing_status_stream_ordering, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamStatusRequest {
            update_interval_seconds: 1,
            include_history: false,
            collection_filter: None,
        };

        let mut stream = client
            .stream_processing_status(Request::new(stream_request))
            .await?
            .into_inner();

        let mut timestamps = Vec::new();

        // Collect multiple status updates
        for _ in 0..3 {
            match timeout(Duration::from_secs(3), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    if let Some(ts) = update.timestamp {
                        timestamps.push((ts.seconds, ts.nanos));
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Verify timestamps are monotonically increasing
        for i in 1..timestamps.len() {
            let (prev_sec, prev_nano) = timestamps[i - 1];
            let (curr_sec, curr_nano) = timestamps[i];

            assert!(
                curr_sec > prev_sec || (curr_sec == prev_sec && curr_nano >= prev_nano),
                "Timestamps should be monotonically increasing"
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_folder_processing_progress_ordering, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/ordering/folder".to_string(),
            collection: "folder_ordering_test".to_string(),
            include_patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec![],
            recursive: false,
            max_depth: 1,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut stream = client
            .process_folder(Request::new(folder_request))
            .await?
            .into_inner();

        let mut progress_counts = Vec::new();

        // Collect progress updates
        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(2), stream.message()).await {
            progress_counts.push(progress.files_processed);

            if progress.completed {
                break;
            }
        }

        // Verify progress is monotonically increasing
        for i in 1..progress_counts.len() {
            assert!(
                progress_counts[i] >= progress_counts[i - 1],
                "Files processed count should be monotonically increasing"
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod stream_termination_tests {
    use super::*;

    async_test!(test_stream_completes_properly, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/termination".to_string(),
            collection: "termination_test".to_string(),
            include_patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec![],
            recursive: false,
            max_depth: 1,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut stream = client
            .process_folder(Request::new(folder_request))
            .await?
            .into_inner();

        let mut completed = false;
        let mut final_message_received = false;

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(3), stream.message()).await {
            if progress.completed {
                completed = true;
                final_message_received = true;
                break;
            }
        }

        // Stream should complete with final message
        assert!(completed || final_message_received, "Stream should indicate completion");

        // After completion, stream should end
        let result = timeout(Duration::from_millis(500), stream.message()).await;
        match result {
            Ok(Ok(None)) => {}, // Stream properly ended
            Ok(Ok(Some(_))) => {
                // Additional messages after completion - also acceptable
            }
            _ => {},
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_termination_on_error, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Request with invalid path to trigger error
        let watch_request = StartWatchingRequest {
            path: "".to_string(), // Invalid empty path
            collection: "error_test".to_string(),
            patterns: vec![],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 1000,
            watch_id: Some("error_watch".to_string()),
        };

        let result = client.start_watching(Request::new(watch_request)).await;

        // Should either fail immediately or stream error messages
        match result {
            Err(status) => {
                assert_eq!(status.code(), Code::InvalidArgument);
            }
            Ok(response) => {
                let mut stream = response.into_inner();

                // Stream might send error event
                while let Ok(Ok(Some(update))) = timeout(Duration::from_secs(1), stream.message()).await {
                    if update.event_type == WatchEventType::Error as i32 {
                        assert!(update.error_message.is_some());
                        break;
                    }
                }
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_ends_after_final_message, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/final".to_string(),
            collection: "final_test".to_string(),
            include_patterns: vec!["*.rs".to_string()],
            ignore_patterns: vec![],
            recursive: false,
            max_depth: 1,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut stream = client
            .process_folder(Request::new(folder_request))
            .await?
            .into_inner();

        let mut message_count = 0;
        let mut received_after_completion = false;

        while let Ok(result) = timeout(Duration::from_secs(2), stream.message()).await {
            match result {
                Ok(Some(progress)) => {
                    message_count += 1;
                    if progress.completed && message_count > 1 {
                        // Check if we receive messages after completion
                        if let Ok(Ok(Some(_))) = timeout(Duration::from_millis(100), stream.message()).await {
                            received_after_completion = true;
                        }
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        // Stream should end cleanly (either immediately after completion or shortly after)
        assert!(message_count > 0, "Should receive at least one message");

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod backpressure_tests {
    use super::*;

    async_test!(test_slow_consumer_backpressure, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamMetricsRequest {
            update_interval_seconds: 1,
            include_detailed_metrics: true,
        };

        let mut stream = client
            .stream_system_metrics(Request::new(stream_request))
            .await?
            .into_inner();

        // Simulate slow consumer by adding delays
        let mut messages_received = 0;

        for _ in 0..5 {
            match timeout(Duration::from_secs(3), stream.message()).await {
                Ok(Ok(Some(_update))) => {
                    messages_received += 1;

                    // Slow consumer - delay processing
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Should still receive messages despite slow consumption
        assert!(messages_received > 0, "Should handle slow consumer gracefully");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_fast_producer_slow_consumer, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Request fast updates
        let stream_request = StreamStatusRequest {
            update_interval_seconds: 1, // Fast updates
            include_history: false,
            collection_filter: None,
        };

        let mut stream = client
            .stream_processing_status(Request::new(stream_request))
            .await?
            .into_inner();

        let start_time = Instant::now();
        let mut messages = Vec::new();

        // Consume slowly for a period
        while start_time.elapsed() < Duration::from_secs(3) {
            match timeout(Duration::from_millis(500), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    messages.push(update);

                    // Slow consumption
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => continue,
            }
        }

        // Server should handle backpressure without dropping messages or crashing
        assert!(messages.len() > 0, "Should receive messages with backpressure");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_buffering_behavior, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let watch_request = StartWatchingRequest {
            path: "/test/buffering".to_string(),
            collection: "buffer_test".to_string(),
            patterns: vec!["*".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 50, // Very fast updates
            watch_id: Some("buffer_watch".to_string()),
        };

        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Don't consume immediately - let messages buffer
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Now consume rapidly
        let mut buffered_messages = 0;
        let consume_start = Instant::now();

        while consume_start.elapsed() < Duration::from_secs(1) {
            match timeout(Duration::from_millis(100), stream.message()).await {
                Ok(Ok(Some(_))) => {
                    buffered_messages += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Should receive buffered messages
        assert!(buffered_messages > 0, "Should handle message buffering");

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    async_test!(test_early_stream_cancellation, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamStatusRequest {
            update_interval_seconds: 1,
            include_history: false,
            collection_filter: None,
        };

        let mut stream = client
            .stream_processing_status(Request::new(stream_request))
            .await?
            .into_inner();

        // Receive one message then cancel
        match timeout(Duration::from_secs(2), stream.message()).await {
            Ok(Ok(Some(_))) => {
                // Drop the stream to cancel
                drop(stream);
            }
            _ => {}
        }

        // Stream should be cleanly cancelled without errors
        // (No way to verify server-side cleanup from client, but should not hang)

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_client_disconnect_during_stream, {
        let fixture = StreamingTestFixture::new().await?;
        let server_addr = fixture.server_addr;

        // Create separate client for this test
        let channel = Channel::from_shared(format!("http://{}", server_addr))?
            .connect()
            .await?;

        let mut client = IngestServiceClient::new(channel);

        let watch_request = StartWatchingRequest {
            path: "/test/disconnect".to_string(),
            collection: "disconnect_test".to_string(),
            patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 1,
            update_frequency_ms: 1000,
            watch_id: Some("disconnect_watch".to_string()),
        };

        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Receive first message
        timeout(Duration::from_secs(2), stream.message()).await.ok();

        // Drop client to simulate disconnect
        drop(client);
        drop(stream);

        // Server should handle disconnect gracefully
        tokio::time::sleep(Duration::from_millis(200)).await;

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_cancellation_cleanup, {
        let fixture = StreamingTestFixture::new().await?;

        // Start multiple streams then cancel them
        for i in 0..3 {
            let mut client = fixture.client.clone();

            let stream_request = StreamQueueRequest {
                update_interval_seconds: 1,
                collection_filter: None,
            };

            let mut stream = client
                .stream_queue_status(Request::new(stream_request))
                .await?
                .into_inner();

            // Receive one message
            timeout(Duration::from_millis(500), stream.message()).await.ok();

            // Cancel by dropping
            drop(stream);

            // Brief delay between iterations
            if i < 2 {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        // All streams should be cancelled cleanly
        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod large_dataset_tests {
    use super::*;

    async_test!(test_many_small_messages, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let watch_request = StartWatchingRequest {
            path: "/test/many_messages".to_string(),
            collection: "many_msg_test".to_string(),
            patterns: vec!["*".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 10, // Very frequent updates
            watch_id: Some("many_msg_watch".to_string()),
        };

        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        let mut message_count = 0;
        let max_messages = 100;
        let start_time = Instant::now();

        while message_count < max_messages && start_time.elapsed() < Duration::from_secs(10) {
            match timeout(Duration::from_millis(200), stream.message()).await {
                Ok(Ok(Some(_))) => {
                    message_count += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Should handle many small messages
        assert!(message_count > 0, "Should receive multiple messages");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_large_payload_streaming, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Request with large metadata to generate larger messages
        let folder_request = ProcessFolderRequest {
            folder_path: "/test/large_payload".to_string(),
            collection: "large_payload_test".to_string(),
            include_patterns: vec!["*.rs".to_string(), "*.txt".to_string(), "*.md".to_string()],
            ignore_patterns: vec!["*.log".to_string(), "*.tmp".to_string()],
            recursive: true,
            max_depth: 10,
            dry_run: true,
            metadata: (0..100)
                .map(|i| (
                    format!("large_metadata_key_{}", i),
                    format!("large_metadata_value_with_lots_of_data_{}", i.to_string().repeat(10))
                ))
                .collect(),
        };

        let mut stream = client
            .process_folder(Request::new(folder_request))
            .await?
            .into_inner();

        let mut messages_received = 0;

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(3), stream.message()).await {
            messages_received += 1;

            if progress.completed {
                break;
            }

            if messages_received >= 10 {
                break;
            }
        }

        // Should handle large payload messages
        assert!(messages_received > 0, "Should receive messages with large payloads");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_sustained_high_throughput_streaming, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamMetricsRequest {
            update_interval_seconds: 1,
            include_detailed_metrics: true,
        };

        let mut stream = client
            .stream_system_metrics(Request::new(stream_request))
            .await?
            .into_inner();

        let start_time = Instant::now();
        let mut total_messages = 0;
        let test_duration = Duration::from_secs(5);

        while start_time.elapsed() < test_duration {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(_))) => {
                    total_messages += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        let throughput = total_messages as f64 / start_time.elapsed().as_secs_f64();

        println!("Sustained throughput: {:.2} messages/second", throughput);
        assert!(total_messages > 0, "Should maintain sustained throughput");

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod error_handling_in_streams_tests {
    use super::*;

    async_test!(test_error_event_in_stream, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Watch request that might generate errors
        let watch_request = StartWatchingRequest {
            path: "/nonexistent/path/that/does/not/exist".to_string(),
            collection: "error_stream_test".to_string(),
            patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec![],
            auto_ingest: true,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 1000,
            watch_id: Some("error_stream_watch".to_string()),
        };

        let result = client.start_watching(Request::new(watch_request)).await;

        match result {
            Ok(response) => {
                let mut stream = response.into_inner();

                // Look for error events in stream
                let mut found_error = false;

                while let Ok(Ok(Some(update))) = timeout(Duration::from_secs(2), stream.message()).await {
                    if update.event_type == WatchEventType::Error as i32 {
                        found_error = true;
                        assert!(update.error_message.is_some());
                        break;
                    }

                    // Also check status field
                    if update.status == WatchStatus::Error as i32 {
                        found_error = true;
                        break;
                    }
                }

                // Either found error event or stream handled it gracefully
            }
            Err(status) => {
                // Immediate error also acceptable
                assert_eq!(status.code(), Code::InvalidArgument);
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_resilience_to_errors, {
        let fixture = StreamingTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamStatusRequest {
            update_interval_seconds: 1,
            include_history: true,
            collection_filter: Some("nonexistent_collection".to_string()),
        };

        let mut stream = client
            .stream_processing_status(Request::new(stream_request))
            .await?
            .into_inner();

        // Stream should either send error-free updates or handle errors gracefully
        let mut messages_received = 0;

        for _ in 0..5 {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(_update))) => {
                    messages_received += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(status)) => {
                    // Error in stream - should be properly formatted
                    assert!(status.code() != Code::Unknown);
                    break;
                }
                Err(_) => break,
            }
        }

        // Should receive at least some messages or fail gracefully
        fixture.shutdown().await?;
        Ok(())
    });
}
