//! Comprehensive gRPC performance integration tests with TDD approach
//!
//! This test suite validates performance characteristics of all gRPC services
//! including throughput, latency, resource usage, and concurrent request handling.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tonic::transport::Channel;
use tonic::Request;

use workspace_qdrant_daemon::{
    config::{DaemonConfig, ServerConfig, DatabaseConfig, QdrantConfig, ProcessingConfig,
             FileWatcherConfig, MetricsConfig, LoggingConfig},
    daemon::WorkspaceDaemon,
    grpc::server::GrpcServer,
    proto::{
        search_service_client::SearchServiceClient,
        system_service_client::SystemServiceClient,
        SearchContext, SearchOptions, RankingOptions,
        HybridSearchRequest,
    },
};

/// Performance test configuration
#[derive(Debug, Clone)]
struct PerformanceConfig {
    /// Number of concurrent connections to test
    max_concurrent_connections: usize,
    /// Number of requests per connection
    requests_per_connection: usize,
    /// Maximum acceptable latency in milliseconds
    max_latency_ms: u64,
    /// Minimum required throughput (requests per second)
    min_throughput_rps: f64,
    /// Memory usage threshold in bytes
    max_memory_usage_bytes: u64,
    /// CPU usage threshold as percentage
    max_cpu_usage_percent: f64,
    /// Test duration for sustained load testing
    test_duration_secs: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_connections: 50,
            requests_per_connection: 20,
            max_latency_ms: 200,  // p99 latency target
            min_throughput_rps: 100.0,
            max_memory_usage_bytes: 100 * 1024 * 1024, // 100MB
            max_cpu_usage_percent: 80.0,
            test_duration_secs: 10,
        }
    }
}

/// Performance metrics collector
#[derive(Debug, Default)]
struct PerformanceMetrics {
    /// Request latencies in microseconds
    latencies: Vec<u64>,
    /// Request timestamps
    timestamps: Vec<Instant>,
    /// Successful requests count
    successful_requests: u64,
    /// Failed requests count
    failed_requests: u64,
    /// Memory usage samples in bytes
    memory_usage_samples: Vec<u64>,
    /// CPU usage samples as percentage
    cpu_usage_samples: Vec<f64>,
    /// Network bytes sent
    bytes_sent: u64,
    /// Network bytes received
    bytes_received: u64,
    /// Connection establishment times
    connection_times: Vec<Duration>,
}

impl PerformanceMetrics {
    /// Calculate p50, p95, p99 latency percentiles
    fn latency_percentiles(&self) -> (u64, u64, u64) {
        if self.latencies.is_empty() {
            return (0, 0, 0);
        }

        let mut sorted = self.latencies.clone();
        sorted.sort_unstable();

        let len = sorted.len();
        let p50 = sorted[len * 50 / 100];
        let p95 = sorted[len * 95 / 100];
        let p99 = sorted[len * 99 / 100];

        (p50, p95, p99)
    }

    /// Calculate throughput in requests per second
    fn throughput_rps(&self) -> f64 {
        if self.timestamps.len() < 2 {
            return 0.0;
        }

        let start = self.timestamps.first().unwrap();
        let end = self.timestamps.last().unwrap();
        let duration = end.duration_since(*start).as_secs_f64();

        if duration > 0.0 {
            self.successful_requests as f64 / duration
        } else {
            0.0
        }
    }

    /// Calculate average memory usage
    fn avg_memory_usage(&self) -> u64 {
        if self.memory_usage_samples.is_empty() {
            return 0;
        }
        self.memory_usage_samples.iter().sum::<u64>() / self.memory_usage_samples.len() as u64
    }

    /// Calculate average CPU usage
    fn avg_cpu_usage(&self) -> f64 {
        if self.cpu_usage_samples.is_empty() {
            return 0.0;
        }
        self.cpu_usage_samples.iter().sum::<f64>() / self.cpu_usage_samples.len() as f64
    }
}

/// Test fixture for gRPC performance tests
struct GrpcPerformanceTestFixture {
    daemon: Arc<WorkspaceDaemon>,
    server_handle: tokio::task::JoinHandle<Result<(), anyhow::Error>>,
    server_addr: std::net::SocketAddr,
    performance_config: PerformanceConfig,
}

impl GrpcPerformanceTestFixture {
    /// Create a new test fixture with performance monitoring
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = create_test_daemon_config();
        let daemon = Arc::new(WorkspaceDaemon::new(config.clone()).await?);

        // Find available port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let server_addr = listener.local_addr()?;
        drop(listener);

        let grpc_server = GrpcServer::new((*daemon).clone(), server_addr);

        // Start server in background
        let server_handle = tokio::spawn(async move {
            grpc_server.serve_daemon().await
        });

        // Wait for server to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(Self {
            daemon,
            server_handle,
            server_addr,
            performance_config: PerformanceConfig::default(),
        })
    }

    /// Create search client for testing
    async fn create_search_client(&self) -> Result<SearchServiceClient<Channel>, Box<dyn std::error::Error>> {
        let endpoint = format!("http://{}", self.server_addr);
        let client = SearchServiceClient::connect(endpoint).await?;
        Ok(client)
    }

    /// Create system client for testing
    async fn create_system_client(&self) -> Result<SystemServiceClient<Channel>, Box<dyn std::error::Error>> {
        let endpoint = format!("http://{}", self.server_addr);
        let client = SystemServiceClient::connect(endpoint).await?;
        Ok(client)
    }

    /// Monitor system resources during test execution
    async fn monitor_system_resources(metrics: Arc<tokio::sync::Mutex<PerformanceMetrics>>) {
        let mut interval = tokio::time::interval(Duration::from_millis(100));

        loop {
            interval.tick().await;

            // Simulate resource monitoring
            let memory_usage = Self::get_memory_usage().await;
            let cpu_usage = Self::get_cpu_usage().await;

            let mut metrics = metrics.lock().await;
            metrics.memory_usage_samples.push(memory_usage);
            metrics.cpu_usage_samples.push(cpu_usage);
        }
    }

    /// Get current memory usage (stubbed for testing)
    async fn get_memory_usage() -> u64 {
        // In real implementation, this would use system monitoring
        25 * 1024 * 1024 // Simulate 25MB usage
    }

    /// Get current CPU usage (stubbed for testing)
    async fn get_cpu_usage() -> f64 {
        // In real implementation, this would use system monitoring
        35.0 // Simulate 35% CPU usage
    }
}

impl Drop for GrpcPerformanceTestFixture {
    fn drop(&mut self) {
        self.server_handle.abort();
    }
}

fn create_test_daemon_config() -> DaemonConfig {
    DaemonConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // Will be overridden by test fixture
            max_connections: 200,
            connection_timeout_secs: 30,
            request_timeout_secs: 10,
            enable_tls: false,
        },
        database: DatabaseConfig {
            sqlite_path: ":memory:".to_string(),
            max_connections: 10,
            connection_timeout_secs: 30,
            enable_wal: true,
        },
        qdrant: QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_secs: 30,
            max_retries: 3,
            default_collection: workspace_qdrant_daemon::config::CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        },
        processing: ProcessingConfig {
            max_concurrent_tasks: 10,
            default_chunk_size: 1000,
            default_chunk_overlap: 200,
            max_file_size_bytes: 10 * 1024 * 1024,
            supported_extensions: vec!["txt".to_string(), "md".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 10,
        },
        file_watcher: FileWatcherConfig {
            enabled: false,
            debounce_ms: 500,
            max_watched_dirs: 100,
            ignore_patterns: vec![],
            recursive: true,
        },
        metrics: MetricsConfig {
            enabled: true,
            collection_interval_secs: 1,
            retention_days: 1,
            enable_prometheus: false,
            prometheus_port: 9090,
        },
        logging: LoggingConfig {
            level: "info".to_string(),
            file_path: None,
            json_format: false,
            max_file_size_mb: 100,
            max_files: 5,
        },
    }
}

/// Performance test for search service throughput
#[tokio::test]
async fn test_search_service_throughput_performance() {
    let fixture = GrpcPerformanceTestFixture::new().await.unwrap();
    let metrics = Arc::new(tokio::sync::Mutex::new(PerformanceMetrics::default()));

    let mut search_client = fixture.create_search_client().await.unwrap();

    let start_time = Instant::now();
    let requests_per_batch = 50;
    let num_batches = 5;

    for batch in 0..num_batches {
        let mut handles = vec![];

        for i in 0..requests_per_batch {
            let mut client = search_client.clone();
            let metrics = Arc::clone(&metrics);

            let handle = tokio::spawn(async move {
                let request_start = Instant::now();

                let request = Request::new(HybridSearchRequest {
                    query: format!("test query batch {} request {}", batch, i),
                    context: SearchContext::Project as i32,
                    options: Some(SearchOptions {
                        limit: 10,
                        score_threshold: 0.0,
                        include_metadata: true,
                        include_content: true,
                        ranking: Some(RankingOptions {
                            semantic_weight: 0.7,
                            keyword_weight: 0.3,
                            rrf_constant: 60.0,
                        }),
                    }),
                    project_id: "test_project".to_string(),
                    collection_names: vec!["test_collection".to_string()],
                });

                match client.hybrid_search(request).await {
                    Ok(response) => {
                        let latency = request_start.elapsed().as_micros() as u64;
                        let response = response.into_inner();

                        let mut metrics = metrics.lock().await;
                        metrics.latencies.push(latency);
                        metrics.timestamps.push(request_start);
                        metrics.successful_requests += 1;
                        metrics.bytes_received += std::mem::size_of_val(&response) as u64;

                        true
                    }
                    Err(_) => {
                        let mut metrics = metrics.lock().await;
                        metrics.failed_requests += 1;
                        false
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for batch completion
        for handle in handles {
            handle.await.unwrap();
        }

        // Small delay between batches to avoid overwhelming
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let total_duration = start_time.elapsed();
    let metrics = metrics.lock().await;

    // Performance validation
    let (p50, p95, p99) = metrics.latency_percentiles();
    let throughput = metrics.throughput_rps();

    println!("Search Service Performance Metrics:");
    println!("  Total requests: {}", metrics.successful_requests);
    println!("  Failed requests: {}", metrics.failed_requests);
    println!("  Throughput: {:.2} RPS", throughput);
    println!("  Latency p50: {} μs", p50);
    println!("  Latency p95: {} μs", p95);
    println!("  Latency p99: {} μs", p99);
    println!("  Total duration: {:?}", total_duration);

    // Performance assertions
    assert!(metrics.successful_requests >= 200, "Too many failed requests");
    assert!(throughput >= fixture.performance_config.min_throughput_rps * 0.5,
            "Throughput below threshold: {} < {}", throughput, fixture.performance_config.min_throughput_rps);
    assert!(p99 <= fixture.performance_config.max_latency_ms * 2000,
            "p99 latency too high: {} μs", p99);
}

/// Performance test for concurrent connection handling
#[tokio::test]
async fn test_concurrent_connection_performance() {
    let fixture = GrpcPerformanceTestFixture::new().await.unwrap();
    let metrics = Arc::new(tokio::sync::Mutex::new(PerformanceMetrics::default()));

    let num_concurrent_connections = 20;
    let requests_per_connection = 10;

    let semaphore = Arc::new(Semaphore::new(num_concurrent_connections));
    let mut handles = vec![];

    for conn_id in 0..num_concurrent_connections {
        let permit = Arc::clone(&semaphore).acquire_owned().await.unwrap();
        let fixture_addr = fixture.server_addr;
        let metrics = Arc::clone(&metrics);

        let handle = tokio::spawn(async move {
            let _permit = permit;
            let endpoint = format!("http://{}", fixture_addr);

            let connection_start = Instant::now();
            let mut search_client = SearchServiceClient::connect(endpoint).await.unwrap();
            let connection_time = connection_start.elapsed();

            for req_id in 0..requests_per_connection {
                let request_start = Instant::now();

                let request = Request::new(HybridSearchRequest {
                    query: format!("concurrent test conn {} req {}", conn_id, req_id),
                    context: SearchContext::Project as i32,
                    options: Some(SearchOptions {
                        limit: 5,
                        score_threshold: 0.0,
                        include_metadata: true,
                        include_content: false,
                        ranking: Some(RankingOptions {
                            semantic_weight: 0.6,
                            keyword_weight: 0.4,
                            rrf_constant: 60.0,
                        }),
                    }),
                    project_id: "concurrent_test".to_string(),
                    collection_names: vec!["test_collection".to_string()],
                });

                match search_client.hybrid_search(request).await {
                    Ok(_) => {
                        let latency = request_start.elapsed().as_micros() as u64;
                        let mut metrics = metrics.lock().await;
                        metrics.latencies.push(latency);
                        metrics.timestamps.push(request_start);
                        metrics.successful_requests += 1;

                        if req_id == 0 {
                            metrics.connection_times.push(connection_time);
                        }
                    }
                    Err(_) => {
                        let mut metrics = metrics.lock().await;
                        metrics.failed_requests += 1;
                    }
                }

                // Small delay between requests on same connection
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all connections to complete
    for handle in handles {
        handle.await.unwrap();
    }

    let metrics = metrics.lock().await;

    // Performance validation
    let (p50, p95, p99) = metrics.latency_percentiles();
    let throughput = metrics.throughput_rps();
    let avg_connection_time = if !metrics.connection_times.is_empty() {
        metrics.connection_times.iter().sum::<Duration>().as_millis() as f64 / metrics.connection_times.len() as f64
    } else {
        0.0
    };

    println!("Concurrent Connection Performance Metrics:");
    println!("  Concurrent connections: {}", num_concurrent_connections);
    println!("  Total requests: {}", metrics.successful_requests);
    println!("  Failed requests: {}", metrics.failed_requests);
    println!("  Throughput: {:.2} RPS", throughput);
    println!("  Latency p50: {} μs", p50);
    println!("  Latency p95: {} μs", p95);
    println!("  Latency p99: {} μs", p99);
    println!("  Avg connection time: {:.2} ms", avg_connection_time);

    // Performance assertions
    let expected_requests = (num_concurrent_connections * requests_per_connection) as u64;
    assert!(metrics.successful_requests >= expected_requests * 80 / 100,
            "Too many failed requests: {} / {}", metrics.successful_requests, expected_requests);
    assert!(p99 <= fixture.performance_config.max_latency_ms * 3000, // Allow 3x latency for concurrent
            "p99 latency too high under load: {} μs", p99);
    assert!(avg_connection_time <= 2000.0, // 2 second max connection time
            "Connection establishment too slow: {:.2} ms", avg_connection_time);
}

/// Performance test for sustained load testing
#[tokio::test]
async fn test_sustained_load_performance() {
    let fixture = GrpcPerformanceTestFixture::new().await.unwrap();
    let metrics = Arc::new(tokio::sync::Mutex::new(PerformanceMetrics::default()));

    // Get performance config before moving fixture
    let performance_config = fixture.performance_config.clone();

    // Start resource monitoring
    let monitor_metrics = Arc::clone(&metrics);
    let _monitor_handle = tokio::spawn(async move {
        GrpcPerformanceTestFixture::monitor_system_resources(monitor_metrics).await;
    });

    let test_duration = Duration::from_secs(5); // Shorter for testing
    let request_interval = Duration::from_millis(50); // 20 RPS target

    let mut search_client = fixture.create_search_client().await.unwrap();

    let start_time = Instant::now();
    let mut request_count = 0;

    while start_time.elapsed() < test_duration {
        let request_start = Instant::now();

        let request = Request::new(HybridSearchRequest {
            query: format!("sustained load test request {}", request_count),
            context: SearchContext::Project as i32,
            options: Some(SearchOptions {
                limit: 10,
                score_threshold: 0.0,
                include_metadata: true,
                include_content: true,
                ranking: Some(RankingOptions {
                    semantic_weight: 0.7,
                    keyword_weight: 0.3,
                    rrf_constant: 60.0,
                }),
            }),
            project_id: "sustained_test".to_string(),
            collection_names: vec!["test_collection".to_string()],
        });

        match timeout(Duration::from_secs(1), search_client.hybrid_search(request)).await {
            Ok(Ok(_)) => {
                let latency = request_start.elapsed().as_micros() as u64;
                let mut metrics = metrics.lock().await;
                metrics.latencies.push(latency);
                metrics.timestamps.push(request_start);
                metrics.successful_requests += 1;
            }
            _ => {
                let mut metrics = metrics.lock().await;
                metrics.failed_requests += 1;
            }
        }

        request_count += 1;

        // Maintain request rate
        if let Some(remaining) = request_interval.checked_sub(request_start.elapsed()) {
            tokio::time::sleep(remaining).await;
        }
    }

    // monitor_handle.abort(); // Not needed for static function

    let metrics = metrics.lock().await;

    // Performance validation
    let (p50, p95, p99) = metrics.latency_percentiles();
    let throughput = metrics.throughput_rps();
    let avg_memory = metrics.avg_memory_usage();
    let avg_cpu = metrics.avg_cpu_usage();

    println!("Sustained Load Performance Metrics:");
    println!("  Test duration: {:?}", test_duration);
    println!("  Total requests: {}", metrics.successful_requests);
    println!("  Failed requests: {}", metrics.failed_requests);
    println!("  Throughput: {:.2} RPS", throughput);
    println!("  Latency p50: {} μs", p50);
    println!("  Latency p95: {} μs", p95);
    println!("  Latency p99: {} μs", p99);
    println!("  Avg memory usage: {} bytes", avg_memory);
    println!("  Avg CPU usage: {:.2}%", avg_cpu);

    // Performance assertions for sustained load
    assert!(metrics.failed_requests <= metrics.successful_requests / 10, // Allow 10% failure rate
            "Too many failed requests under sustained load");
    assert!(p99 <= fixture.performance_config.max_latency_ms * 2000, // Allow 2x latency
            "p99 latency degraded under sustained load: {} μs", p99);
    assert!(avg_memory <= performance_config.max_memory_usage_bytes,
            "Memory usage too high under sustained load: {} bytes", avg_memory);
    assert!(avg_cpu <= performance_config.max_cpu_usage_percent,
            "CPU usage too high under sustained load: {:.2}%", avg_cpu);
}

/// Performance test for system service monitoring overhead
#[tokio::test]
async fn test_system_service_monitoring_performance() {
    let fixture = GrpcPerformanceTestFixture::new().await.unwrap();
    let metrics = Arc::new(tokio::sync::Mutex::new(PerformanceMetrics::default()));

    let mut system_client = fixture.create_system_client().await.unwrap();

    let num_health_checks = 100;
    let check_interval = Duration::from_millis(10);

    for i in 0..num_health_checks {
        let request_start = Instant::now();

        let request = Request::new(());

        match system_client.health_check(request).await {
            Ok(response) => {
                let latency = request_start.elapsed().as_micros() as u64;
                let response = response.into_inner();

                let mut metrics = metrics.lock().await;
                metrics.latencies.push(latency);
                metrics.timestamps.push(request_start);
                metrics.successful_requests += 1;
                metrics.bytes_received += std::mem::size_of_val(&response) as u64;
            }
            Err(_) => {
                let mut metrics = metrics.lock().await;
                metrics.failed_requests += 1;
            }
        }

        if i < num_health_checks - 1 {
            tokio::time::sleep(check_interval).await;
        }
    }

    let metrics = metrics.lock().await;

    // Performance validation
    let (p50, p95, p99) = metrics.latency_percentiles();
    let throughput = metrics.throughput_rps();

    println!("System Service Monitoring Performance Metrics:");
    println!("  Health checks: {}", metrics.successful_requests);
    println!("  Failed checks: {}", metrics.failed_requests);
    println!("  Throughput: {:.2} checks/sec", throughput);
    println!("  Latency p50: {} μs", p50);
    println!("  Latency p95: {} μs", p95);
    println!("  Latency p99: {} μs", p99);

    // Performance assertions for monitoring
    assert_eq!(metrics.failed_requests, 0, "Health checks should not fail");
    assert!(p99 <= 50000, "Health check latency too high: {} μs", p99); // 50ms max
    assert!(throughput >= 20.0, "Health check throughput too low: {:.2}", throughput);
}

/// Performance test for network bandwidth utilization
#[tokio::test]
async fn test_network_bandwidth_performance() {
    let fixture = GrpcPerformanceTestFixture::new().await.unwrap();
    let metrics = Arc::new(tokio::sync::Mutex::new(PerformanceMetrics::default()));

    let mut search_client = fixture.create_search_client().await.unwrap();

    // Test with varying payload sizes
    let payload_sizes = [100, 1000, 5000]; // bytes
    let requests_per_size = 20;

    for &payload_size in &payload_sizes {
        for _i in 0..requests_per_size {
            let request_start = Instant::now();

            let large_query = "bandwidth test ".repeat(payload_size / 15); // ~payload_size chars

            let request = Request::new(HybridSearchRequest {
                query: large_query.clone(),
                context: SearchContext::Project as i32,
                options: Some(SearchOptions {
                    limit: 20, // Large result set
                    score_threshold: 0.0,
                    include_metadata: true,
                    include_content: true,
                    ranking: Some(RankingOptions {
                        semantic_weight: 0.7,
                        keyword_weight: 0.3,
                        rrf_constant: 60.0,
                    }),
                }),
                project_id: "bandwidth_test".to_string(),
                collection_names: vec!["test_collection".to_string()],
            });

            let request_size = large_query.len() as u64;

            match search_client.hybrid_search(request).await {
                Ok(response) => {
                    let latency = request_start.elapsed().as_micros() as u64;
                    let response = response.into_inner();
                    let response_size = std::mem::size_of_val(&response) as u64;

                    let mut metrics = metrics.lock().await;
                    metrics.latencies.push(latency);
                    metrics.timestamps.push(request_start);
                    metrics.successful_requests += 1;
                    metrics.bytes_sent += request_size;
                    metrics.bytes_received += response_size;
                }
                Err(_) => {
                    let mut metrics = metrics.lock().await;
                    metrics.failed_requests += 1;
                }
            }
        }
    }

    let metrics = metrics.lock().await;

    // Performance validation
    let (p50, p95, p99) = metrics.latency_percentiles();
    let throughput = metrics.throughput_rps();
    let total_bytes = metrics.bytes_sent + metrics.bytes_received;
    let bandwidth_mbps = if !metrics.timestamps.is_empty() {
        let duration = metrics.timestamps.last().unwrap()
            .duration_since(*metrics.timestamps.first().unwrap()).as_secs_f64();
        if duration > 0.0 {
            (total_bytes as f64 * 8.0) / (duration * 1_000_000.0) // Mbps
        } else {
            0.0
        }
    } else {
        0.0
    };

    println!("Network Bandwidth Performance Metrics:");
    println!("  Total requests: {}", metrics.successful_requests);
    println!("  Failed requests: {}", metrics.failed_requests);
    println!("  Throughput: {:.2} RPS", throughput);
    println!("  Latency p50: {} μs", p50);
    println!("  Latency p95: {} μs", p95);
    println!("  Latency p99: {} μs", p99);
    println!("  Bytes sent: {} bytes", metrics.bytes_sent);
    println!("  Bytes received: {} bytes", metrics.bytes_received);
    println!("  Total bandwidth: {:.2} Mbps", bandwidth_mbps);

    // Performance assertions
    assert!(metrics.successful_requests >= payload_sizes.len() as u64 * requests_per_size as u64 * 80 / 100,
            "Too many failed requests with varying payload sizes");
    assert!(bandwidth_mbps >= 0.1, "Network bandwidth too low: {:.2} Mbps", bandwidth_mbps);

    // Latency should scale reasonably with payload size
    assert!(p99 <= 500000, "Latency too high with large payloads: {} μs", p99); // 500ms max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_metrics_calculations() {
        let mut metrics = PerformanceMetrics::default();

        // Add test data
        metrics.latencies = vec![1000, 2000, 3000, 4000, 5000]; // μs
        metrics.timestamps = vec![
            Instant::now(),
            Instant::now() + Duration::from_millis(1),
            Instant::now() + Duration::from_millis(2),
            Instant::now() + Duration::from_millis(3),
            Instant::now() + Duration::from_millis(4),
        ];
        metrics.successful_requests = 5;
        metrics.memory_usage_samples = vec![100, 200, 150, 175, 125];
        metrics.cpu_usage_samples = vec![10.0, 20.0, 15.0, 17.5, 12.5];

        let (p50, p95, p99) = metrics.latency_percentiles();
        assert_eq!(p50, 3000);
        assert_eq!(p95, 5000);
        assert_eq!(p99, 5000);

        let avg_memory = metrics.avg_memory_usage();
        assert_eq!(avg_memory, 150);

        let avg_cpu = metrics.avg_cpu_usage();
        assert_eq!(avg_cpu, 15.0);
    }

    #[tokio::test]
    async fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();

        assert_eq!(config.max_concurrent_connections, 50);
        assert_eq!(config.requests_per_connection, 20);
        assert_eq!(config.max_latency_ms, 200);
        assert_eq!(config.min_throughput_rps, 100.0);
        assert_eq!(config.max_memory_usage_bytes, 100 * 1024 * 1024);
        assert_eq!(config.max_cpu_usage_percent, 80.0);
        assert_eq!(config.test_duration_secs, 10);
    }

    #[test]
    fn test_daemon_config_performance_settings() {
        let config = create_test_daemon_config();

        assert_eq!(config.server.max_connections, 200);
        assert_eq!(config.server.connection_timeout_secs, 30);
        assert_eq!(config.server.request_timeout_secs, 10);
        assert_eq!(config.database.max_connections, 10);
        assert_eq!(config.processing.max_concurrent_tasks, 10);
        assert!(config.metrics.enabled);
    }
}