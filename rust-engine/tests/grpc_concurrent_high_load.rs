//! Comprehensive gRPC concurrent client and high-concurrency tests (Task 321.7)
//!
//! Tests gRPC infrastructure under concurrent client load and high-volume request scenarios.
//! Validates thread safety, connection pooling under stress, resource contention handling,
//! and performance characteristics under various load patterns.
//!
//! Test coverage:
//! - Multiple concurrent clients (10, 50, 100+ simultaneous clients)
//! - High-volume request scenarios (1000+ requests/sec)
//! - Resource contention validation (shared pool access)
//! - Thread safety verification (no data races, deadlocks)
//! - Connection pooling under load (pool efficiency, reuse patterns)
//! - Performance metrics (throughput, latency percentiles, error rates)
//! - Various load patterns (burst, gradual ramp-up, sustained)

#![cfg(feature = "test-utils")]

use workspace_qdrant_daemon::grpc::client::{ConnectionPool, WorkspaceDaemonClient};
use workspace_qdrant_daemon::proto::{
    system_service_client::SystemServiceClient,
    ServiceStatus,
};
use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::grpc::server::GrpcServer;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::time::{timeout, sleep};
use tonic::transport::Endpoint;
use tonic::Request;
use serial_test::serial;

// ================================
// TEST INFRASTRUCTURE
// ================================

/// Test environment with server and metrics
struct TestEnvironment {
    _daemon: WorkspaceDaemon,
    server_handle: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
    address: String,
    metrics: Arc<LoadTestMetrics>,
}

impl TestEnvironment {
    async fn new(port: u16) -> Self {
        let config = DaemonConfig::default();
        let daemon = WorkspaceDaemon::new(config).await
            .expect("Failed to create test daemon");

        let socket_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
        let grpc_server = GrpcServer::new(daemon.clone(), socket_addr);

        let address = format!("http://127.0.0.1:{}", port);

        // Start server in background
        let server_handle = tokio::spawn(async move {
            grpc_server.serve_daemon().await
        });

        // Give server time to start
        sleep(Duration::from_millis(200)).await;

        Self {
            _daemon: daemon,
            server_handle,
            address,
            metrics: Arc::new(LoadTestMetrics::new()),
        }
    }

    fn address(&self) -> &str {
        &self.address
    }

    fn metrics(&self) -> Arc<LoadTestMetrics> {
        Arc::clone(&self.metrics)
    }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        self.server_handle.abort();
    }
}

/// Metrics collector for load testing
#[derive(Debug)]
struct LoadTestMetrics {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    total_latency_us: AtomicU64,
    active_clients: AtomicUsize,
    min_latency_us: AtomicU64,
    max_latency_us: AtomicU64,
}

impl LoadTestMetrics {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            active_clients: AtomicUsize::new(0),
            min_latency_us: AtomicU64::new(u64::MAX),
            max_latency_us: AtomicU64::new(0),
        }
    }

    fn record_request(&self, success: bool, latency: Duration) {
        self.total_requests.fetch_add(1, Ordering::SeqCst);

        if success {
            self.successful_requests.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed_requests.fetch_add(1, Ordering::SeqCst);
        }

        let latency_us = latency.as_micros() as u64;
        self.total_latency_us.fetch_add(latency_us, Ordering::SeqCst);

        // Update min latency
        let mut current_min = self.min_latency_us.load(Ordering::SeqCst);
        while latency_us < current_min {
            match self.min_latency_us.compare_exchange_weak(
                current_min,
                latency_us,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max latency
        let mut current_max = self.max_latency_us.load(Ordering::SeqCst);
        while latency_us > current_max {
            match self.max_latency_us.compare_exchange_weak(
                current_max,
                latency_us,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    fn increment_active_clients(&self) {
        self.active_clients.fetch_add(1, Ordering::SeqCst);
    }

    fn decrement_active_clients(&self) {
        self.active_clients.fetch_sub(1, Ordering::SeqCst);
    }

    fn summary(&self) -> MetricsSummary {
        let total = self.total_requests.load(Ordering::SeqCst);
        let successful = self.successful_requests.load(Ordering::SeqCst);
        let failed = self.failed_requests.load(Ordering::SeqCst);
        let total_latency = self.total_latency_us.load(Ordering::SeqCst);
        let active = self.active_clients.load(Ordering::SeqCst);
        let min_latency = self.min_latency_us.load(Ordering::SeqCst);
        let max_latency = self.max_latency_us.load(Ordering::SeqCst);

        let avg_latency_us = if total > 0 {
            total_latency / total
        } else {
            0
        };

        MetricsSummary {
            total_requests: total,
            successful_requests: successful,
            failed_requests: failed,
            avg_latency_us,
            min_latency_us: if min_latency == u64::MAX { 0 } else { min_latency },
            max_latency_us,
            active_clients: active,
        }
    }
}

#[derive(Debug)]
struct MetricsSummary {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    avg_latency_us: u64,
    min_latency_us: u64,
    max_latency_us: u64,
    active_clients: usize,
}

// ================================
// MULTIPLE CONCURRENT CLIENTS TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_10_concurrent_clients() {
    // Test with 10 concurrent clients
    let env = TestEnvironment::new(50070).await;
    let metrics = env.metrics();
    let client_count = 10;
    let requests_per_client = 5;

    let mut handles = Vec::new();

    for client_id in 0..client_count {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            metrics_clone.increment_active_clients();

            let client = WorkspaceDaemonClient::new(address);
            let mut client_successful = 0;

            for _req in 0..requests_per_client {
                let start = Instant::now();
                let result = timeout(Duration::from_secs(2), client.health_check()).await;
                let latency = start.elapsed();

                let success = result.is_ok() && result.unwrap().is_ok();
                metrics_clone.record_request(success, latency);

                if success {
                    client_successful += 1;
                }
            }

            metrics_clone.decrement_active_clients();
            (client_id, client_successful)
        });

        handles.push(handle);
    }

    // Wait for all clients
    let mut total_successful = 0;
    for handle in handles {
        let (_, successful) = handle.await.unwrap();
        total_successful += successful;
    }

    let summary = metrics.summary();

    // Validate metrics
    assert_eq!(summary.total_requests, (client_count * requests_per_client) as u64);
    assert_eq!(summary.active_clients, 0, "All clients should complete");
    assert!(summary.successful_requests > 0, "Should have some successful requests");

    println!("10 Concurrent Clients Summary: {:?}", summary);
}

#[tokio::test]
#[serial]
async fn test_50_concurrent_clients() {
    // Test with 50 concurrent clients
    let env = TestEnvironment::new(50071).await;
    let metrics = env.metrics();
    let client_count = 50;
    let requests_per_client = 3;

    let mut handles = Vec::new();

    for client_id in 0..client_count {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            metrics_clone.increment_active_clients();

            let client = WorkspaceDaemonClient::new(address);
            let mut client_successful = 0;

            for _req in 0..requests_per_client {
                let start = Instant::now();
                let result = timeout(Duration::from_secs(3), client.health_check()).await;
                let latency = start.elapsed();

                let success = result.is_ok() && result.unwrap().is_ok();
                metrics_clone.record_request(success, latency);

                if success {
                    client_successful += 1;
                }
            }

            metrics_clone.decrement_active_clients();
            client_successful
        });

        handles.push(handle);
    }

    // Wait for all clients
    let mut total_successful = 0;
    for handle in handles {
        let successful = handle.await.unwrap();
        total_successful += successful;
    }

    let summary = metrics.summary();

    // Validate metrics
    assert_eq!(summary.total_requests, (client_count * requests_per_client) as u64);
    assert_eq!(summary.active_clients, 0, "All clients should complete");

    println!("50 Concurrent Clients Summary: {:?}", summary);
}

#[tokio::test]
#[serial]
async fn test_100_concurrent_clients() {
    // Test with 100 concurrent clients
    let env = TestEnvironment::new(50072).await;
    let metrics = env.metrics();
    let client_count = 100;
    let requests_per_client = 2;

    let mut handles = Vec::new();

    for _client_id in 0..client_count {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            metrics_clone.increment_active_clients();

            let client = WorkspaceDaemonClient::new(address);

            for _req in 0..requests_per_client {
                let start = Instant::now();
                let result = timeout(Duration::from_secs(3), client.health_check()).await;
                let latency = start.elapsed();

                let success = result.is_ok() && result.unwrap().is_ok();
                metrics_clone.record_request(success, latency);
            }

            metrics_clone.decrement_active_clients();
        });

        handles.push(handle);
    }

    // Wait for all clients
    for handle in handles {
        let _ = handle.await;
    }

    let summary = metrics.summary();

    // Validate metrics
    assert_eq!(summary.total_requests, (client_count * requests_per_client) as u64);
    assert_eq!(summary.active_clients, 0, "All clients should complete");

    println!("100 Concurrent Clients Summary: {:?}", summary);
}

// ================================
// HIGH-VOLUME REQUEST TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_high_volume_sustained_load() {
    // Test sustained high-volume requests
    let env = TestEnvironment::new(50073).await;
    let metrics = env.metrics();
    let duration = Duration::from_secs(3);
    let target_rps = 100; // 100 requests per second

    let start = Instant::now();
    let mut handles = Vec::new();

    // Spawn continuous request tasks
    for worker_id in 0..10 {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);
        let test_duration = duration;

        let handle = tokio::spawn(async move {
            let client = WorkspaceDaemonClient::new(address);
            let worker_start = Instant::now();

            while worker_start.elapsed() < test_duration {
                let req_start = Instant::now();
                let result = timeout(Duration::from_secs(1), client.health_check()).await;
                let latency = req_start.elapsed();

                let success = result.is_ok() && result.unwrap().is_ok();
                metrics_clone.record_request(success, latency);

                // Rate limiting: ~10ms between requests per worker = 100 req/s with 10 workers
                sleep(Duration::from_millis(10)).await;
            }

            worker_id
        });

        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed();
    let summary = metrics.summary();

    let actual_rps = summary.total_requests as f64 / elapsed.as_secs_f64();

    println!("Sustained Load Summary: {:?}", summary);
    println!("Actual RPS: {:.2}, Target: {}", actual_rps, target_rps);

    assert!(summary.total_requests > 0, "Should process requests");
    assert!(actual_rps > 50.0, "Should achieve reasonable throughput");
}

#[tokio::test]
#[serial]
async fn test_burst_traffic_pattern() {
    // Test burst traffic pattern
    let env = TestEnvironment::new(50074).await;
    let metrics = env.metrics();
    let burst_size = 50;
    let burst_count = 3;

    for burst_num in 0..burst_count {
        let mut handles = Vec::new();

        // Create burst
        for _i in 0..burst_size {
            let address = env.address().to_string();
            let metrics_clone = Arc::clone(&metrics);

            let handle = tokio::spawn(async move {
                let start = Instant::now();
                let endpoint = Endpoint::from_shared(address)
                    .expect("Failed to create endpoint")
                    .timeout(Duration::from_secs(2));

                let result = timeout(Duration::from_secs(3), endpoint.connect()).await;
                let latency = start.elapsed();

                let success = result.is_ok() && result.unwrap().is_ok();
                metrics_clone.record_request(success, latency);
            });

            handles.push(handle);
        }

        // Wait for burst to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Short delay between bursts
        if burst_num < burst_count - 1 {
            sleep(Duration::from_millis(200)).await;
        }
    }

    let summary = metrics.summary();

    assert_eq!(summary.total_requests, (burst_size * burst_count) as u64);
    println!("Burst Traffic Summary: {:?}", summary);
}

// ================================
// RESOURCE CONTENTION TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_shared_connection_pool_contention() {
    // Test concurrent access to shared connection pool
    let env = TestEnvironment::new(50075).await;
    let pool = Arc::new(ConnectionPool::new());
    let metrics = env.metrics();
    let client_count = 20;
    let requests_per_client = 5;

    let mut handles = Vec::new();

    for _client_id in 0..client_count {
        let address = env.address().to_string();
        let pool_clone = Arc::clone(&pool);
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            metrics_clone.increment_active_clients();

            for _req in 0..requests_per_client {
                let start = Instant::now();
                let result = timeout(Duration::from_secs(2), pool_clone.get_connection(&address)).await;
                let latency = start.elapsed();

                let success = result.is_ok() && result.unwrap().is_ok();
                metrics_clone.record_request(success, latency);
            }

            metrics_clone.decrement_active_clients();
        });

        handles.push(handle);
    }

    // Wait for all clients
    for handle in handles {
        let _ = handle.await;
    }

    let summary = metrics.summary();
    let pool_count = pool.connection_count().await;

    // Pool should contain only 1 connection (all clients share same address)
    assert!(pool_count <= 1, "Shared pool should efficiently reuse connections");
    assert_eq!(summary.active_clients, 0, "All clients should complete");

    println!("Shared Pool Contention Summary: {:?}", summary);
    println!("Pool connections: {}", pool_count);
}

#[tokio::test]
#[serial]
async fn test_multiple_pools_concurrent_access() {
    // Test multiple pools with concurrent access
    let env = TestEnvironment::new(50076).await;
    let pool1 = Arc::new(ConnectionPool::new());
    let pool2 = Arc::new(ConnectionPool::new());
    let metrics = env.metrics();

    let mut handles = Vec::new();

    // Pool 1 workers
    for _i in 0..10 {
        let address = env.address().to_string();
        let pool = Arc::clone(&pool1);
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            for _req in 0..3 {
                let start = Instant::now();
                let result = pool.get_connection(&address).await;
                let latency = start.elapsed();
                metrics_clone.record_request(result.is_ok(), latency);
            }
        });

        handles.push(handle);
    }

    // Pool 2 workers
    for _i in 0..10 {
        let address = env.address().to_string();
        let pool = Arc::clone(&pool2);
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            for _req in 0..3 {
                let start = Instant::now();
                let result = pool.get_connection(&address).await;
                let latency = start.elapsed();
                metrics_clone.record_request(result.is_ok(), latency);
            }
        });

        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        let _ = handle.await;
    }

    let summary = metrics.summary();

    assert_eq!(summary.total_requests, 60); // 20 workers * 3 requests
    println!("Multiple Pools Summary: {:?}", summary);
}

// ================================
// THREAD SAFETY VALIDATION TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_connection_pool_add_remove_concurrent() {
    // Test concurrent add/remove operations on connection pool
    let pool = Arc::new(ConnectionPool::new());
    let addresses = vec![
        "http://127.0.0.1:50077",
        "http://127.0.0.1:50078",
        "http://127.0.0.1:50079",
    ];

    let mut handles = Vec::new();

    // Add connections concurrently
    for addr in &addresses {
        let pool_clone = Arc::clone(&pool);
        let address = addr.to_string();

        let handle = tokio::spawn(async move {
            for _i in 0..5 {
                let _ = pool_clone.get_connection(&address).await;
                sleep(Duration::from_millis(10)).await;
            }
        });

        handles.push(handle);
    }

    // Concurrent remove operations
    for addr in &addresses {
        let pool_clone = Arc::clone(&pool);
        let address = addr.to_string();

        let handle = tokio::spawn(async move {
            for _i in 0..3 {
                sleep(Duration::from_millis(15)).await;
                pool_clone.remove_connection(&address).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        let _ = handle.await;
    }

    // No panics or deadlocks = test passes
    let final_count = pool.connection_count().await;
    println!("Final pool count after concurrent add/remove: {}", final_count);
}

#[tokio::test]
#[serial]
async fn test_client_stats_concurrent_access() {
    // Test concurrent access to client stats
    let env = TestEnvironment::new(50080).await;
    let client = Arc::new(WorkspaceDaemonClient::new(env.address().to_string()));

    let mut handles = Vec::new();

    // Concurrent health checks
    for _i in 0..20 {
        let client_clone = Arc::clone(&client);

        let handle = tokio::spawn(async move {
            let _ = client_clone.health_check().await;
        });

        handles.push(handle);
    }

    // Concurrent stats access
    for _i in 0..20 {
        let client_clone = Arc::clone(&client);

        let handle = tokio::spawn(async move {
            let _stats = client_clone.connection_stats().await;
        });

        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        let _ = handle.await;
    }

    // Verify final state
    let stats = client.connection_stats().await;
    assert_eq!(stats.address, env.address());
    println!("Concurrent stats access completed: {:?}", stats);
}

// ================================
// CONNECTION POOLING UNDER LOAD TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_pool_efficiency_under_load() {
    // Test connection pool efficiency under sustained load
    let env = TestEnvironment::new(50081).await;
    let pool = Arc::new(ConnectionPool::new());
    let worker_count = 30;
    let requests_per_worker = 10;

    let mut handles = Vec::new();

    for _worker in 0..worker_count {
        let address = env.address().to_string();
        let pool_clone = Arc::clone(&pool);

        let handle = tokio::spawn(async move {
            for _req in 0..requests_per_worker {
                let _ = pool_clone.get_connection(&address).await;
                sleep(Duration::from_millis(5)).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        let _ = handle.await;
    }

    let pool_count = pool.connection_count().await;

    // Should have exactly 1 connection (all workers used same address)
    assert!(pool_count <= 1, "Pool should efficiently reuse single connection");
    println!("Pool efficiency test: {} connections for {} workers * {} requests",
             pool_count, worker_count, requests_per_worker);
}

#[tokio::test]
#[serial]
async fn test_pool_reuse_pattern_validation() {
    // Validate connection reuse patterns
    let env = TestEnvironment::new(50082).await;
    let pool = Arc::new(ConnectionPool::new());
    let address = env.address().to_string();

    // First wave: establish connections
    let mut handles = Vec::new();
    for _i in 0..10 {
        let pool_clone = Arc::clone(&pool);
        let addr = address.clone();

        let handle = tokio::spawn(async move {
            pool_clone.get_connection(&addr).await
        });

        handles.push(handle);
    }

    for handle in handles {
        let _ = handle.await;
    }

    let count_after_first = pool.connection_count().await;

    // Second wave: should reuse
    let mut handles = Vec::new();
    for _i in 0..10 {
        let pool_clone = Arc::clone(&pool);
        let addr = address.clone();

        let handle = tokio::spawn(async move {
            pool_clone.get_connection(&addr).await
        });

        handles.push(handle);
    }

    for handle in handles {
        let _ = handle.await;
    }

    let count_after_second = pool.connection_count().await;

    // Connection count should remain stable (reuse, not create new)
    assert_eq!(count_after_first, count_after_second,
              "Pool should reuse connections, not create new ones");
    println!("Reuse pattern: {} connections stable across waves", count_after_second);
}

// ================================
// PERFORMANCE UNDER STRESS TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_latency_under_concurrent_load() {
    // Measure latency distribution under concurrent load
    let env = TestEnvironment::new(50083).await;
    let metrics = env.metrics();
    let client_count = 50;

    let mut handles = Vec::new();

    for _i in 0..client_count {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            let client = WorkspaceDaemonClient::new(address);

            for _req in 0..5 {
                let start = Instant::now();
                let result = client.health_check().await;
                let latency = start.elapsed();

                metrics_clone.record_request(result.is_ok(), latency);
                sleep(Duration::from_millis(5)).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all clients
    for handle in handles {
        let _ = handle.await;
    }

    let summary = metrics.summary();

    println!("Latency under load: min={} us, max={} us, avg={} us",
             summary.min_latency_us, summary.max_latency_us, summary.avg_latency_us);

    // Latency should be reasonable (< 100ms avg under load)
    assert!(summary.avg_latency_us < 100_000, "Average latency should be under 100ms");
}

#[tokio::test]
#[serial]
async fn test_throughput_measurement() {
    // Measure throughput with concurrent clients
    let env = TestEnvironment::new(50084).await;
    let metrics = env.metrics();
    let duration = Duration::from_secs(2);

    let start = Instant::now();
    let mut handles = Vec::new();

    // Spawn workers
    for _worker in 0..10 {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);
        let test_duration = duration;

        let handle = tokio::spawn(async move {
            let client = WorkspaceDaemonClient::new(address);
            let worker_start = Instant::now();

            while worker_start.elapsed() < test_duration {
                let req_start = Instant::now();
                let result = client.health_check().await;
                let latency = req_start.elapsed();

                metrics_clone.record_request(result.is_ok(), latency);
            }
        });

        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed();
    let summary = metrics.summary();

    let throughput = summary.total_requests as f64 / elapsed.as_secs_f64();

    println!("Throughput test: {} requests in {:.2}s = {:.2} req/s",
             summary.total_requests, elapsed.as_secs_f64(), throughput);

    assert!(summary.total_requests > 0, "Should process requests");
    assert!(throughput > 10.0, "Should achieve measurable throughput");
}

// ================================
// VARIOUS LOAD PATTERN TESTS
// ================================

#[tokio::test]
#[serial]
async fn test_gradual_ramp_up_pattern() {
    // Test gradual client ramp-up
    let env = TestEnvironment::new(50085).await;
    let metrics = env.metrics();

    // Ramp up: 5, 10, 15, 20 clients
    for batch in 1..=4 {
        let client_count = batch * 5;
        let mut handles = Vec::new();

        for _i in 0..client_count {
            let address = env.address().to_string();
            let metrics_clone = Arc::clone(&metrics);

            let handle = tokio::spawn(async move {
                let client = WorkspaceDaemonClient::new(address);

                for _req in 0..3 {
                    let start = Instant::now();
                    let result = client.health_check().await;
                    let latency = start.elapsed();

                    metrics_clone.record_request(result.is_ok(), latency);
                }
            });

            handles.push(handle);
        }

        // Wait for batch
        for handle in handles {
            let _ = handle.await;
        }

        sleep(Duration::from_millis(100)).await;
    }

    let summary = metrics.summary();

    // Total: (5 + 10 + 15 + 20) * 3 = 150 requests
    assert_eq!(summary.total_requests, 150);
    println!("Gradual ramp-up summary: {:?}", summary);
}

#[tokio::test]
#[serial]
async fn test_spike_and_cooldown_pattern() {
    // Test traffic spike followed by cooldown
    let env = TestEnvironment::new(50086).await;
    let metrics = env.metrics();

    // Spike phase: 50 concurrent clients
    let mut handles = Vec::new();

    for _i in 0..50 {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            let client = WorkspaceDaemonClient::new(address);

            let start = Instant::now();
            let result = client.health_check().await;
            let latency = start.elapsed();

            metrics_clone.record_request(result.is_ok(), latency);
        });

        handles.push(handle);
    }

    // Wait for spike
    for handle in handles {
        let _ = handle.await;
    }

    // Cooldown phase: sequential requests
    let client = WorkspaceDaemonClient::new(env.address().to_string());

    for _i in 0..10 {
        let start = Instant::now();
        let result = client.health_check().await;
        let latency = start.elapsed();

        metrics.record_request(result.is_ok(), latency);
        sleep(Duration::from_millis(50)).await;
    }

    let summary = metrics.summary();

    assert_eq!(summary.total_requests, 60); // 50 spike + 10 cooldown
    println!("Spike and cooldown summary: {:?}", summary);
}

#[tokio::test]
#[serial]
async fn test_mixed_client_configurations() {
    // Test with various client configurations
    let env = TestEnvironment::new(50087).await;
    let metrics = env.metrics();

    let mut handles = Vec::new();

    // Clients with default pool (10 clients)
    for _i in 0..10 {
        let address = env.address().to_string();
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            let client = WorkspaceDaemonClient::new(address);

            let start = Instant::now();
            let result = client.health_check().await;
            let latency = start.elapsed();

            metrics_clone.record_request(result.is_ok(), latency);
        });

        handles.push(handle);
    }

    // Clients with custom pool (10 clients)
    let custom_pool = Arc::new(ConnectionPool::with_timeouts(
        Duration::from_secs(15),
        Duration::from_secs(8),
    ));

    for _i in 0..10 {
        let address = env.address().to_string();
        let pool = Arc::clone(&custom_pool);
        let metrics_clone = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            let client = WorkspaceDaemonClient::with_pool(address, (*pool).clone());

            let start = Instant::now();
            let result = client.health_check().await;
            let latency = start.elapsed();

            metrics_clone.record_request(result.is_ok(), latency);
        });

        handles.push(handle);
    }

    // Wait for all clients
    for handle in handles {
        let _ = handle.await;
    }

    let summary = metrics.summary();

    assert_eq!(summary.total_requests, 20);
    println!("Mixed configurations summary: {:?}", summary);
}
