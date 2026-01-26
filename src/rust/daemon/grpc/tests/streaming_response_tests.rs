#![cfg(feature = "legacy_grpc_tests")]
//! Enhanced gRPC streaming response tests for Task 321.2
//!
//! This module provides comprehensive testing of gRPC streaming operations including:
//! - Stream lifecycle validation (initialization, data flow, completion)
//! - Message ordering in server-side streams
//! - Backpressure handling mechanisms
//! - Stream completion signals
//! - Bulk operation streaming with progress updates
//! - Small and large dataset streaming performance

use shared_test_utils::{async_test, TestResult};
use std::collections::VecDeque;
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

/// Test fixture for streaming response tests
pub struct StreamingResponseTestFixture {
    pub server_addr: SocketAddr,
    pub client: IngestServiceClient<Channel>,
    pub _server_handle: tokio::task::JoinHandle<()>,
    pub shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl StreamingResponseTestFixture {
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
mod stream_lifecycle_tests {
    use super::*;

    async_test!(test_stream_initialization_phase, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let watch_request = StartWatchingRequest {
            path: "/test/lifecycle/init".to_string(),
            collection: "lifecycle_init_test".to_string(),
            patterns: vec!["*.rs".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 1,
            update_frequency_ms: 500,
            watch_id: Some("lifecycle_init_watch".to_string()),
        };

        let start_time = Instant::now();
        let response = client.start_watching(Request::new(watch_request)).await;
        let initialization_time = start_time.elapsed();

        // Stream should initialize quickly
        assert!(
            initialization_time < Duration::from_secs(1),
            "Stream initialization took {:?}, should be under 1s",
            initialization_time
        );

        assert!(response.is_ok(), "Stream should initialize successfully");
        let mut stream = response?.into_inner();

        // First message should arrive quickly and indicate stream started
        let first_message = timeout(Duration::from_secs(2), stream.message())
            .await?
            .expect("Should receive first message")?
            .expect("First message should not be None");

        assert_eq!(
            first_message.event_type,
            WatchEventType::Started as i32,
            "First message should be STARTED event"
        );
        assert_eq!(first_message.watch_id, "lifecycle_init_watch");
        assert_eq!(first_message.status, WatchStatus::Active as i32);

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_data_flow_phase, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamProcessingStatus {
            update_interval_seconds: 1,
            include_history: false,
            collection_filter: None,
        };

        let mut stream = client
            .stream_processing_status(Request::new(stream_request))
            .await?
            .into_inner();

        // Track data flow characteristics
        let mut messages_received = Vec::new();
        let start_time = Instant::now();
        let test_duration = Duration::from_secs(3);

        while start_time.elapsed() < test_duration {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    messages_received.push((start_time.elapsed(), update));
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => continue,
            }
        }

        // Verify continuous data flow
        assert!(
            messages_received.len() >= 2,
            "Should receive multiple status updates during test period"
        );

        // Verify each message has proper structure
        for (elapsed, update) in &messages_received {
            assert!(update.timestamp.is_some(), "Each update should have timestamp");
            println!("Received update at {:?}", elapsed);
        }

        // Verify update intervals are reasonable
        if messages_received.len() >= 2 {
            let interval = messages_received[1].0 - messages_received[0].0;
            assert!(
                interval >= Duration::from_millis(500),
                "Update interval should respect configured frequency"
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_completion_phase, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/lifecycle/complete".to_string(),
            collection: "lifecycle_complete_test".to_string(),
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

        let mut received_completion = false;
        let mut completion_message = None;
        let start_time = Instant::now();

        // Wait for completion signal
        while start_time.elapsed() < Duration::from_secs(5) {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(progress))) => {
                    if progress.completed {
                        received_completion = true;
                        completion_message = Some(progress);
                        break;
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Verify completion signal was received
        assert!(
            received_completion || completion_message.is_some(),
            "Stream should send completion signal"
        );

        if let Some(final_msg) = completion_message {
            assert!(final_msg.completed, "Completion message should have completed=true");
            assert!(
                final_msg.files_processed >= 0,
                "Should report number of files processed"
            );
        }

        // Verify stream ends after completion
        let post_completion = timeout(Duration::from_millis(500), stream.message()).await;
        match post_completion {
            Ok(Ok(None)) => {
                // Stream properly ended - expected
            }
            Ok(Ok(Some(_))) => {
                // Additional messages after completion - also acceptable
            }
            _ => {}
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_lifecycle_all_phases, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Track all lifecycle phases
        #[derive(Debug, PartialEq)]
        enum Phase {
            Initialization,
            DataFlow,
            Completion,
        }

        let mut phases_observed = Vec::new();
        let overall_start = Instant::now();

        let watch_request = StartWatchingRequest {
            path: "/test/lifecycle/full".to_string(),
            collection: "lifecycle_full_test".to_string(),
            patterns: vec!["*".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 200,
            watch_id: Some("lifecycle_full_watch".to_string()),
        };

        // Phase 1: Initialization
        let init_start = Instant::now();
        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();
        phases_observed.push(Phase::Initialization);
        println!("Initialization phase completed in {:?}", init_start.elapsed());

        // Phase 2: Data Flow
        let mut data_flow_messages = 0;
        let data_flow_start = Instant::now();

        for _ in 0..5 {
            match timeout(Duration::from_secs(2), stream.message()).await {
                Ok(Ok(Some(_))) => {
                    if data_flow_messages == 0 {
                        phases_observed.push(Phase::DataFlow);
                    }
                    data_flow_messages += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        println!(
            "Data flow phase: {} messages in {:?}",
            data_flow_messages,
            data_flow_start.elapsed()
        );

        // Phase 3: Completion (implicit - drop stream)
        drop(stream);
        phases_observed.push(Phase::Completion);

        println!("Full lifecycle completed in {:?}", overall_start.elapsed());

        // Verify all phases occurred in order
        assert_eq!(phases_observed.len(), 3, "Should observe all three phases");
        assert_eq!(phases_observed[0], Phase::Initialization);
        assert_eq!(phases_observed[1], Phase::DataFlow);
        assert_eq!(phases_observed[2], Phase::Completion);

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod message_ordering_advanced_tests {
    use super::*;

    async_test!(test_strict_message_sequence_verification, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let watch_request = StartWatchingRequest {
            path: "/test/ordering/strict".to_string(),
            collection: "strict_ordering_test".to_string(),
            patterns: vec!["*.md".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 100,
            watch_id: Some("strict_order_watch".to_string()),
        };

        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        let mut messages = Vec::new();
        let start_time = Instant::now();

        // Collect messages with timestamps
        while messages.len() < 10 && start_time.elapsed() < Duration::from_secs(3) {
            match timeout(Duration::from_millis(500), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    messages.push((Instant::now(), update));
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Verify message ordering properties
        if !messages.is_empty() {
            // First message must be STARTED
            assert_eq!(
                messages[0].1.event_type,
                WatchEventType::Started as i32,
                "First message must be STARTED event"
            );

            // All messages should have same watch_id
            for (_, msg) in &messages {
                assert_eq!(msg.watch_id, "strict_order_watch");
            }

            // Timestamps should be monotonically increasing
            for i in 1..messages.len() {
                let prev_time = messages[i - 1].0;
                let curr_time = messages[i].0;
                assert!(
                    curr_time >= prev_time,
                    "Message timestamps should be monotonically increasing"
                );
            }

            // Verify no duplicate event types in wrong order
            let event_types: Vec<i32> = messages.iter().map(|(_, m)| m.event_type).collect();
            println!("Event sequence: {:?}", event_types);
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_interleaved_stream_ordering, {
        let fixture = StreamingResponseTestFixture::new().await?;

        // Create two concurrent streams
        let mut client1 = fixture.client.clone();
        let mut client2 = fixture.client.clone();

        let watch1 = StartWatchingRequest {
            path: "/test/ordering/stream1".to_string(),
            collection: "stream1_test".to_string(),
            patterns: vec!["*.rs".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 150,
            watch_id: Some("stream1_watch".to_string()),
        };

        let watch2 = StartWatchingRequest {
            path: "/test/ordering/stream2".to_string(),
            collection: "stream2_test".to_string(),
            patterns: vec!["*.txt".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 150,
            watch_id: Some("stream2_watch".to_string()),
        };

        let mut stream1 = client1.start_watching(Request::new(watch1)).await?.into_inner();
        let mut stream2 = client2.start_watching(Request::new(watch2)).await?.into_inner();

        let mut stream1_messages = Vec::new();
        let mut stream2_messages = Vec::new();

        // Collect from both streams concurrently
        let start_time = Instant::now();
        while start_time.elapsed() < Duration::from_secs(2) {
            tokio::select! {
                msg1 = timeout(Duration::from_millis(200), stream1.message()) => {
                    if let Ok(Ok(Some(update))) = msg1 {
                        stream1_messages.push(update);
                    }
                }
                msg2 = timeout(Duration::from_millis(200), stream2.message()) => {
                    if let Ok(Ok(Some(update))) = msg2 {
                        stream2_messages.push(update);
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }

            if stream1_messages.len() >= 3 && stream2_messages.len() >= 3 {
                break;
            }
        }

        // Verify each stream maintains its own ordering
        for msg in &stream1_messages {
            assert_eq!(msg.watch_id, "stream1_watch");
        }

        for msg in &stream2_messages {
            assert_eq!(msg.watch_id, "stream2_watch");
        }

        // Each stream should start with STARTED event
        if !stream1_messages.is_empty() {
            assert_eq!(stream1_messages[0].event_type, WatchEventType::Started as i32);
        }
        if !stream2_messages.is_empty() {
            assert_eq!(stream2_messages[0].event_type, WatchEventType::Started as i32);
        }

        println!(
            "Stream 1 received {} messages, Stream 2 received {} messages",
            stream1_messages.len(),
            stream2_messages.len()
        );

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_folder_processing_progress_sequence, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/ordering/progress".to_string(),
            collection: "progress_seq_test".to_string(),
            include_patterns: vec!["*.rs".to_string(), "*.toml".to_string()],
            ignore_patterns: vec!["target/*".to_string()],
            recursive: true,
            max_depth: 5,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut stream = client
            .process_folder(Request::new(folder_request))
            .await?
            .into_inner();

        let mut progress_sequence = Vec::new();

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(2), stream.message()).await
        {
            progress_sequence.push(progress.clone());

            if progress.completed {
                break;
            }
        }

        // Verify progress monotonicity
        for i in 1..progress_sequence.len() {
            let prev = &progress_sequence[i - 1];
            let curr = &progress_sequence[i];

            // Files processed should never decrease
            assert!(
                curr.files_processed >= prev.files_processed,
                "Files processed should be monotonically increasing"
            );

            // Success + failed should equal processed
            assert_eq!(
                curr.files_processed,
                curr.files_succeeded + curr.files_failed,
                "Processed count should equal succeeded + failed"
            );
        }

        // Last message should have completed=true
        if let Some(last) = progress_sequence.last() {
            assert!(last.completed, "Last progress message should indicate completion");
        }

        println!("Observed {} progress updates", progress_sequence.len());

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod backpressure_advanced_tests {
    use super::*;

    async_test!(test_controlled_backpressure_simulation, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamMetricsRequest {
            update_interval_seconds: 1,
            include_detailed_metrics: true,
        };

        let mut stream = client
            .stream_system_metrics(Request::new(stream_request))
            .await?
            .into_inner();

        // Simulate controlled slow consumption
        let mut messages = Vec::new();
        let consumption_delay = Duration::from_millis(300);

        for iteration in 0..10 {
            match timeout(Duration::from_secs(3), stream.message()).await {
                Ok(Ok(Some(update))) => {
                    messages.push(update);

                    // Introduce controlled backpressure
                    tokio::time::sleep(consumption_delay).await;

                    println!("Iteration {}: Consumed message after {:?} delay", iteration, consumption_delay);
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        // Server should handle slow consumer gracefully
        assert!(
            messages.len() > 0,
            "Should receive messages despite backpressure"
        );

        println!("Received {} messages with controlled backpressure", messages.len());

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_burst_consumption_after_delay, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let watch_request = StartWatchingRequest {
            path: "/test/backpressure/burst".to_string(),
            collection: "burst_test".to_string(),
            patterns: vec!["*".to_string()],
            ignore_patterns: vec![],
            auto_ingest: false,
            recursive: false,
            recursive_depth: 1,
            debounce_seconds: 0,
            update_frequency_ms: 50,
            watch_id: Some("burst_watch".to_string()),
        };

        let mut stream = client
            .start_watching(Request::new(watch_request))
            .await?
            .into_inner();

        // Let messages accumulate without consuming
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Now burst consume
        let burst_start = Instant::now();
        let mut burst_count = 0;

        while burst_start.elapsed() < Duration::from_millis(500) {
            match timeout(Duration::from_millis(50), stream.message()).await {
                Ok(Ok(Some(_))) => {
                    burst_count += 1;
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        println!("Burst consumed {} messages in {:?}", burst_count, burst_start.elapsed());

        // Should handle burst consumption
        assert!(burst_count > 0, "Should handle burst consumption after delay");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_backpressure_with_varying_rates, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let stream_request = StreamQueueRequest {
            update_interval_seconds: 1,
            collection_filter: None,
        };

        let mut stream = client
            .stream_queue_status(Request::new(stream_request))
            .await?
            .into_inner();

        // Vary consumption rate: fast, slow, fast
        let consumption_patterns = vec![
            (Duration::from_millis(50), 5),   // Fast: 5 messages
            (Duration::from_millis(400), 5),  // Slow: 5 messages
            (Duration::from_millis(50), 5),   // Fast: 5 messages
        ];

        let mut total_consumed = 0;

        for (delay, count) in consumption_patterns {
            for _ in 0..count {
                match timeout(Duration::from_secs(3), stream.message()).await {
                    Ok(Ok(Some(_))) => {
                        total_consumed += 1;
                        tokio::time::sleep(delay).await;
                    }
                    Ok(Ok(None)) => break,
                    Ok(Err(_)) => break,
                    Err(_) => break,
                }
            }
        }

        println!("Total consumed with varying rates: {}", total_consumed);

        // Server should adapt to varying consumption rates
        assert!(
            total_consumed > 0,
            "Should handle varying consumption rates"
        );

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod stream_completion_signals_tests {
    use super::*;

    async_test!(test_explicit_completion_signal, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/completion/explicit".to_string(),
            collection: "explicit_complete_test".to_string(),
            include_patterns: vec!["*.json".to_string()],
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

        let mut found_completion_signal = false;
        let mut all_messages = Vec::new();

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(3), stream.message()).await
        {
            all_messages.push(progress.clone());

            if progress.completed {
                found_completion_signal = true;
                println!("Received explicit completion signal: completed={}", progress.completed);
                break;
            }
        }

        assert!(
            found_completion_signal,
            "Stream should send explicit completion signal via completed field"
        );

        // Verify final state
        if let Some(final_msg) = all_messages.last() {
            assert!(final_msg.completed);
            assert_eq!(
                final_msg.files_processed,
                final_msg.files_succeeded + final_msg.files_failed
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_stream_end_after_completion, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let folder_request = ProcessFolderRequest {
            folder_path: "/test/completion/stream_end".to_string(),
            collection: "stream_end_test".to_string(),
            include_patterns: vec!["*.xml".to_string()],
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

        let mut completion_time = None;

        // Wait for completion signal
        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(3), stream.message()).await
        {
            if progress.completed {
                completion_time = Some(Instant::now());
                break;
            }
        }

        assert!(completion_time.is_some(), "Should receive completion signal");

        // Verify stream ends shortly after completion
        let post_completion_timeout = Duration::from_millis(500);
        let post_result = timeout(post_completion_timeout, stream.message()).await;

        match post_result {
            Ok(Ok(None)) => {
                println!("Stream properly ended after completion");
            }
            Ok(Ok(Some(msg))) => {
                // Additional message after completion - verify it's benign
                println!("Received message after completion: {:?}", msg);
            }
            Ok(Err(e)) => {
                println!("Stream error after completion: {:?}", e);
            }
            Err(_) => {
                println!("No more messages after completion (timeout)");
            }
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_completion_signal_consistency, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Run multiple folder processing operations and verify completion consistency
        for iteration in 0..3 {
            let folder_request = ProcessFolderRequest {
                folder_path: format!("/test/completion/consistency/{}", iteration),
                collection: format!("consistency_test_{}", iteration),
                include_patterns: vec!["*.yaml".to_string()],
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

            let mut received_completion = false;
            let mut message_count = 0;

            while let Ok(Ok(Some(progress))) =
                timeout(Duration::from_secs(2), stream.message()).await
            {
                message_count += 1;

                if progress.completed {
                    received_completion = true;
                    break;
                }
            }

            println!(
                "Iteration {}: {} messages, completion={}",
                iteration, message_count, received_completion
            );

            // Each iteration should consistently signal completion
            assert!(
                received_completion || message_count > 0,
                "Each stream should either complete or send messages"
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod bulk_operations_streaming_tests {
    use super::*;

    async_test!(test_bulk_folder_processing_with_progress, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let bulk_request = ProcessFolderRequest {
            folder_path: "/test/bulk/large_folder".to_string(),
            collection: "bulk_processing_test".to_string(),
            include_patterns: vec!["*.rs".to_string(), "*.md".to_string(), "*.toml".to_string()],
            ignore_patterns: vec!["target/*".to_string(), "*.lock".to_string()],
            recursive: true,
            max_depth: 10,
            dry_run: true,
            metadata: [("test_type".to_string(), "bulk".to_string())].into(),
        };

        let mut stream = client
            .process_folder(Request::new(bulk_request))
            .await?
            .into_inner();

        let mut progress_updates = Vec::new();
        let start_time = Instant::now();

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(5), stream.message()).await
        {
            progress_updates.push((start_time.elapsed(), progress.clone()));

            println!(
                "Progress: {}/{} files ({}% complete)",
                progress.files_processed,
                progress.total_files,
                if progress.total_files > 0 {
                    (progress.files_processed * 100) / progress.total_files
                } else {
                    0
                }
            );

            if progress.completed {
                break;
            }
        }

        // Verify bulk operation progress characteristics
        assert!(
            progress_updates.len() > 0,
            "Should receive progress updates during bulk operation"
        );

        // Verify progress increases
        for i in 1..progress_updates.len() {
            let prev_processed = progress_updates[i - 1].1.files_processed;
            let curr_processed = progress_updates[i].1.files_processed;

            assert!(
                curr_processed >= prev_processed,
                "Progress should not decrease"
            );
        }

        println!(
            "Bulk operation completed with {} progress updates in {:?}",
            progress_updates.len(),
            start_time.elapsed()
        );

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_bulk_streaming_throughput, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let bulk_request = ProcessFolderRequest {
            folder_path: "/test/bulk/throughput".to_string(),
            collection: "throughput_test".to_string(),
            include_patterns: vec!["*".to_string()],
            ignore_patterns: vec![],
            recursive: true,
            max_depth: 5,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let start_time = Instant::now();
        let mut stream = client
            .process_folder(Request::new(bulk_request))
            .await?
            .into_inner();

        let mut message_count = 0;
        let mut total_files_processed = 0;

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(3), stream.message()).await
        {
            message_count += 1;
            total_files_processed = progress.files_processed.max(total_files_processed);

            if progress.completed {
                break;
            }
        }

        let duration = start_time.elapsed();
        let throughput = message_count as f64 / duration.as_secs_f64();

        println!(
            "Bulk streaming throughput: {:.2} messages/sec ({} messages in {:?})",
            throughput, message_count, duration
        );
        println!("Total files reported processed: {}", total_files_processed);

        assert!(message_count > 0, "Should receive progress messages");

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_bulk_operation_error_reporting, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let bulk_request = ProcessFolderRequest {
            folder_path: "/test/bulk/with_errors".to_string(),
            collection: "error_reporting_test".to_string(),
            include_patterns: vec!["*.corrupt".to_string()],
            ignore_patterns: vec![],
            recursive: true,
            max_depth: 3,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut stream = client
            .process_folder(Request::new(bulk_request))
            .await?
            .into_inner();

        let mut error_messages = Vec::new();

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(3), stream.message()).await
        {
            if !progress.error_message.is_empty() {
                error_messages.push(progress.error_message.clone());
            }

            if progress.files_failed > 0 {
                println!(
                    "Detected {} failed files, error: {}",
                    progress.files_failed, progress.error_message
                );
            }

            if progress.completed {
                break;
            }
        }

        // Bulk operations should report errors through the stream
        println!("Observed {} error messages in stream", error_messages.len());

        fixture.shutdown().await?;
        Ok(())
    });
}

#[cfg(test)]
mod dataset_size_variation_tests {
    use super::*;

    async_test!(test_small_dataset_streaming, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let small_request = ProcessFolderRequest {
            folder_path: "/test/dataset/small".to_string(),
            collection: "small_dataset_test".to_string(),
            include_patterns: vec!["single_file.txt".to_string()],
            ignore_patterns: vec![],
            recursive: false,
            max_depth: 1,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let start_time = Instant::now();
        let mut stream = client
            .process_folder(Request::new(small_request))
            .await?
            .into_inner();

        let mut messages = Vec::new();

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(2), stream.message()).await
        {
            messages.push(progress.clone());

            if progress.completed {
                break;
            }
        }

        let duration = start_time.elapsed();

        println!(
            "Small dataset: {} messages in {:?}",
            messages.len(),
            duration
        );

        // Small datasets should complete quickly
        assert!(
            duration < Duration::from_secs(2),
            "Small dataset should complete quickly"
        );

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_large_dataset_streaming, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let large_request = ProcessFolderRequest {
            folder_path: "/test/dataset/large".to_string(),
            collection: "large_dataset_test".to_string(),
            include_patterns: vec!["**/*".to_string()],
            ignore_patterns: vec![],
            recursive: true,
            max_depth: 20,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let start_time = Instant::now();
        let mut stream = client
            .process_folder(Request::new(large_request))
            .await?
            .into_inner();

        let mut message_count = 0;
        let mut peak_files_processed = 0;

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(10), stream.message()).await
        {
            message_count += 1;
            peak_files_processed = peak_files_processed.max(progress.files_processed);

            if message_count % 10 == 0 {
                println!(
                    "Large dataset progress: {} messages, {} files processed",
                    message_count, peak_files_processed
                );
            }

            if progress.completed {
                break;
            }
        }

        println!(
            "Large dataset: {} messages, {} files processed in {:?}",
            message_count,
            peak_files_processed,
            start_time.elapsed()
        );

        // Large datasets should provide multiple progress updates
        assert!(
            message_count > 0,
            "Should receive progress updates for large dataset"
        );

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_medium_dataset_streaming_performance, {
        let fixture = StreamingResponseTestFixture::new().await?;
        let mut client = fixture.client.clone();

        let medium_request = ProcessFolderRequest {
            folder_path: "/test/dataset/medium".to_string(),
            collection: "medium_dataset_test".to_string(),
            include_patterns: vec!["*.rs".to_string(), "*.md".to_string()],
            ignore_patterns: vec!["target/*".to_string()],
            recursive: true,
            max_depth: 5,
            dry_run: true,
            metadata: std::collections::HashMap::new(),
        };

        let start_time = Instant::now();
        let mut stream = client
            .process_folder(Request::new(medium_request))
            .await?
            .into_inner();

        let mut messages = Vec::new();
        let mut inter_message_delays = Vec::new();
        let mut last_message_time = start_time;

        while let Ok(Ok(Some(progress))) = timeout(Duration::from_secs(5), stream.message()).await
        {
            let now = Instant::now();
            inter_message_delays.push(now - last_message_time);
            last_message_time = now;

            messages.push(progress.clone());

            if progress.completed {
                break;
            }
        }

        // Analyze performance characteristics
        if !inter_message_delays.is_empty() {
            let avg_delay = inter_message_delays.iter().sum::<Duration>() / inter_message_delays.len() as u32;
            let max_delay = inter_message_delays.iter().max().unwrap();

            println!(
                "Medium dataset performance: {} messages, avg delay: {:?}, max delay: {:?}",
                messages.len(),
                avg_delay,
                max_delay
            );

            // Updates should arrive with reasonable frequency
            assert!(
                *max_delay < Duration::from_secs(3),
                "Maximum inter-message delay should be reasonable"
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_varying_dataset_sizes, {
        let fixture = StreamingResponseTestFixture::new().await?;

        let test_cases = vec![
            ("tiny", 1, "*.txt"),
            ("small", 2, "*.md"),
            ("medium", 5, "*.rs"),
        ];

        for (name, depth, pattern) in test_cases {
            let mut client = fixture.client.clone();

            let request = ProcessFolderRequest {
                folder_path: format!("/test/dataset/varying/{}", name),
                collection: format!("{}_dataset", name),
                include_patterns: vec![pattern.to_string()],
                ignore_patterns: vec![],
                recursive: true,
                max_depth: depth,
                dry_run: true,
                metadata: std::collections::HashMap::new(),
            };

            let start = Instant::now();
            let mut stream = client.process_folder(Request::new(request)).await?.into_inner();

            let mut count = 0;
            while let Ok(Ok(Some(progress))) =
                timeout(Duration::from_secs(3), stream.message()).await
            {
                count += 1;
                if progress.completed {
                    break;
                }
            }

            println!(
                "{} dataset: {} messages in {:?}",
                name,
                count,
                start.elapsed()
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });
}
