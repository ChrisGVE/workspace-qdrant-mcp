//! Common test helper functions and utilities

use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::time::timeout;

use crate::config::{DEFAULT_TEST_TIMEOUT};
use crate::TestResult;
use std::sync::Once;
use tracing_subscriber::{fmt, EnvFilter};
use tokio_test::{assert_pending, assert_ready, assert_ready_err, assert_ready_ok, task};
use std::pin::Pin;
use std::future::Future;

/// Execute an async operation with timeout
pub async fn with_timeout<F, T>(operation: F) -> TestResult<T>
where
    F: std::future::Future<Output = TestResult<T>>,
{
    timeout(DEFAULT_TEST_TIMEOUT, operation)
        .await
        .map_err(|_| Box::<dyn std::error::Error + Send + Sync>::from("Operation timed out"))?
}

/// Execute an async operation with custom timeout
pub async fn with_custom_timeout<F, T>(operation: F, timeout_duration: Duration) -> TestResult<T>
where
    F: std::future::Future<Output = TestResult<T>>,
{
    timeout(timeout_duration, operation)
        .await
        .map_err(|_| Box::<dyn std::error::Error + Send + Sync>::from("Operation timed out"))?
}

/// Measure execution time of an async operation
pub async fn measure_time<F, T>(operation: F) -> TestResult<(T, Duration)>
where
    F: std::future::Future<Output = TestResult<T>>,
{
    let start = Instant::now();
    let result = operation.await?;
    let duration = start.elapsed();
    Ok((result, duration))
}

/// Retry an operation with exponential backoff
pub async fn retry_with_backoff<F, Fut, T, E>(
    mut operation: F,
    max_attempts: usize,
    initial_delay: Duration,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut delay = initial_delay;

    for attempt in 1..=max_attempts {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_attempts {
                    return Err(e);
                }
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
        }
    }

    unreachable!()
}

/// Wait for a condition to become true with polling
pub async fn wait_for_condition<F>(
    mut condition: F,
    timeout_duration: Duration,
    poll_interval: Duration,
) -> TestResult<()>
where
    F: FnMut() -> bool,
{
    let start = Instant::now();

    while start.elapsed() < timeout_duration {
        if condition() {
            return Ok(());
        }
        tokio::time::sleep(poll_interval).await;
    }

    Err(Box::<dyn std::error::Error + Send + Sync>::from("Condition was not met within timeout"))
}

/// Advanced tokio-test utilities for async operation testing
/// Test that a future is initially pending
pub fn assert_future_pending<F>(future: Pin<&mut F>) -> TestResult<()>
where
    F: Future,
    F::Output: std::fmt::Debug,
{
    let mut task = task::spawn(future);
    assert_pending!(task.poll());
    Ok(())
}

/// Test that a future becomes ready and returns expected value
pub fn assert_future_ready<F, T>(future: Pin<&mut F>) -> TestResult<T>
where
    F: Future<Output = T>,
{
    let mut task = task::spawn(future);
    let result = assert_ready!(task.poll());
    Ok(result)
}

/// Test that a future becomes ready with an error
pub fn assert_future_ready_err<F, T, E>(future: Pin<&mut F>) -> TestResult<E>
where
    F: Future<Output = Result<T, E>>,
    T: std::fmt::Debug,
    E: std::fmt::Debug,
{
    let mut task = task::spawn(future);
    let result = assert_ready_err!(task.poll());
    Ok(result)
}

/// Test that a future becomes ready with success
pub fn assert_future_ready_ok<F, T, E>(future: Pin<&mut F>) -> TestResult<T>
where
    F: Future<Output = Result<T, E>>,
    T: std::fmt::Debug,
    E: std::fmt::Debug,
{
    let mut task = task::spawn(future);
    let result = assert_ready_ok!(task.poll());
    Ok(result)
}

/// Test async operation timing with precise control
pub async fn test_async_timing<F, T>(
    operation: F,
    expected_min_duration: Duration,
    expected_max_duration: Duration,
) -> TestResult<T>
where
    F: Future<Output = T>,
{
    let start = Instant::now();
    let result = operation.await;
    let elapsed = start.elapsed();

    if elapsed < expected_min_duration {
        return Err(format!(
            "Operation completed too quickly: {:?} < {:?}",
            elapsed, expected_min_duration
        ).into());
    }

    if elapsed > expected_max_duration {
        return Err(format!(
            "Operation took too long: {:?} > {:?}",
            elapsed, expected_max_duration
        ).into());
    }

    Ok(result)
}

/// Test concurrent async operations with controlled execution
pub async fn test_concurrent_operations<F, T>(
    operations: Vec<F>,
    max_concurrent: usize,
) -> TestResult<Vec<T>>
where
    F: Future<Output = TestResult<T>> + Send + 'static,
    T: Send + 'static,
{
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    let mut handles = Vec::new();

    for op in operations {
        let sem = semaphore.clone();
        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.map_err(|e|
                Box::<dyn std::error::Error + Send + Sync>::from(format!("Semaphore error: {}", e))
            )?;
            op.await
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(format!("Join error: {}", e)))??;
        results.push(result);
    }

    Ok(results)
}

/// Wait for an async condition to become true
pub async fn wait_for_async_condition<F, Fut>(
    mut condition: F,
    timeout_duration: Duration,
    poll_interval: Duration,
) -> TestResult<()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = Instant::now();

    while start.elapsed() < timeout_duration {
        if condition().await {
            return Ok(());
        }
        tokio::time::sleep(poll_interval).await;
    }

    Err(Box::<dyn std::error::Error + Send + Sync>::from("Async condition was not met within timeout"))
}

/// Generate a unique test identifier
pub fn generate_test_id() -> String {
    format!("test_{}", uuid::Uuid::new_v4().simple())
}

/// Generate a unique collection name for testing
pub fn generate_test_collection() -> String {
    format!("test_collection_{}", uuid::Uuid::new_v4().simple())
}

/// Check if we're running in CI environment
pub fn is_ci() -> bool {
    env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok()
}

/// Check if a specific environment variable is set for testing
pub fn env_test_flag(flag: &str) -> bool {
    env::var(flag).map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
}

/// Skip test if not running in CI
pub fn require_ci() -> TestResult<()> {
    if !is_ci() {
        return Err("Test requires CI environment".into());
    }
    Ok(())
}

/// Skip test if running in CI (for local-only tests)
pub fn skip_in_ci() -> TestResult<()> {
    if is_ci() {
        return Err("Test skipped in CI environment".into());
    }
    Ok(())
}

/// Initialize test tracing/logging (call once per test)
pub fn init_test_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("debug"));

        fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init()
            .ok(); // Ignore error if already initialized
    });
}

/// Get test data directory path
pub fn test_data_dir() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    Path::new(&manifest_dir).join("test_data")
}

/// Get test resource file path
pub fn test_resource(filename: &str) -> PathBuf {
    test_data_dir().join(filename)
}

/// Check if a test resource file exists
pub fn has_test_resource(filename: &str) -> bool {
    test_resource(filename).exists()
}

/// Read test resource file as string
pub async fn read_test_resource(filename: &str) -> TestResult<String> {
    let path = test_resource(filename);
    if !path.exists() {
        return Err(format!("Test resource not found: {}", filename).into());
    }
    Ok(tokio::fs::read_to_string(path).await?)
}

/// Read test resource file as bytes
pub async fn read_test_resource_bytes(filename: &str) -> TestResult<Vec<u8>> {
    let path = test_resource(filename);
    if !path.exists() {
        return Err(format!("Test resource not found: {}", filename).into());
    }
    Ok(tokio::fs::read(path).await?)
}

/// Performance benchmarking helper
pub struct PerformanceBenchmark {
    start_time: Instant,
    operation_name: String,
}

impl PerformanceBenchmark {
    /// Start a new benchmark
    pub fn start(operation_name: &str) -> Self {
        Self {
            start_time: Instant::now(),
            operation_name: operation_name.to_string(),
        }
    }

    /// End the benchmark and return duration
    pub fn end(self) -> Duration {
        let duration = self.start_time.elapsed();
        tracing::info!(
            "Performance benchmark '{}' completed in {:?}",
            self.operation_name,
            duration
        );
        duration
    }

    /// End the benchmark and assert it completed within expected time
    pub fn end_with_assertion(self, max_duration: Duration) -> TestResult<Duration> {
        let operation_name = self.operation_name.clone();
        let duration = self.end();
        if duration > max_duration {
            return Err(format!(
                "Performance benchmark '{}' took {:?}, expected <= {:?}",
                operation_name, duration, max_duration
            ).into());
        }
        Ok(duration)
    }
}

/// Memory usage tracking helper
pub struct MemoryTracker {
    initial_memory: Option<usize>,
    operation_name: String,
}

impl MemoryTracker {
    /// Start memory tracking
    pub fn start(operation_name: &str) -> Self {
        // Note: Getting actual memory usage would require platform-specific code
        // For now, this is a placeholder that could be extended
        Self {
            initial_memory: Self::get_memory_usage(),
            operation_name: operation_name.to_string(),
        }
    }

    /// End memory tracking and log difference
    pub fn end(self) -> Option<isize> {
        if let (Some(initial), Some(final_memory)) = (self.initial_memory, Self::get_memory_usage()) {
            let diff = final_memory as isize - initial as isize;
            tracing::info!(
                "Memory usage for '{}': initial={}, final={}, diff={}",
                self.operation_name, initial, final_memory, diff
            );
            Some(diff)
        } else {
            tracing::warn!("Memory tracking not available for '{}'", self.operation_name);
            None
        }
    }

    /// Get current process RSS (Resident Set Size) in bytes
    fn get_memory_usage() -> Option<usize> {
        Self::get_rss_bytes()
    }

    #[cfg(target_os = "macos")]
    fn get_rss_bytes() -> Option<usize> {
        use std::mem;
        // Use mach task_info to get resident memory
        unsafe {
            let mut info: libc::mach_task_basic_info_data_t = mem::zeroed();
            let mut count = (mem::size_of::<libc::mach_task_basic_info_data_t>()
                / mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
            let result = libc::task_info(
                #[allow(deprecated)]
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as libc::task_info_t,
                &mut count,
            );
            if result == libc::KERN_SUCCESS {
                Some(info.resident_size as usize)
            } else {
                None
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_rss_bytes() -> Option<usize> {
        // Read /proc/self/statm for RSS in pages
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let rss_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size > 0 {
            Some(rss_pages * page_size as usize)
        } else {
            None
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn get_rss_bytes() -> Option<usize> {
        tracing::warn!("Memory tracking not supported on this platform");
        None
    }
}

/// Concurrent execution helper
pub async fn run_concurrent<F, T>(operations: Vec<F>, max_concurrent: usize) -> Vec<Result<T, Box<dyn std::error::Error + Send + Sync>>>
where
    F: std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>> + Send + 'static,
    T: Send + 'static,
{
    use tokio::task::JoinSet;

    let mut set = JoinSet::new();
    let mut results = Vec::new();
    let mut operations = operations.into_iter();

    // Start initial batch
    for _ in 0..max_concurrent {
        if let Some(op) = operations.next() {
            set.spawn(op);
        }
    }

    // Process results and start new operations
    while let Some(result) = set.join_next().await {
        match result {
            Ok(op_result) => results.push(op_result),
            Err(join_error) => results.push(Err(join_error.into())),
        }

        // Start next operation if available
        if let Some(op) = operations.next() {
            set.spawn(op);
        }
    }

    results
}

/// Test data generator for stress testing
pub fn generate_large_text(size_kb: usize) -> String {
    let base_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
    let target_size = size_kb * 1024;
    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str(base_text);
    }

    result.truncate(target_size);
    result
}

/// Generate test vectors for embedding tests
pub fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i + j) as f32 * 0.1).sin())
                .collect()
        })
        .collect()
}

/// Generate test documents with varying content
pub fn generate_test_documents(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            format!(
                "Test document #{}\n\nThis is a test document with unique content.\nDocument index: {}\nGenerated for testing purposes.\n\nContent varies to ensure different embeddings.",
                i + 1,
                i
            )
        })
        .collect()
}

/// Assert that two floating point values are approximately equal
pub fn assert_approx_eq(a: f32, b: f32, epsilon: f32) -> TestResult<()> {
    if (a - b).abs() <= epsilon {
        Ok(())
    } else {
        Err(format!("Values not approximately equal: {} vs {} (epsilon: {})", a, b, epsilon).into())
    }
}

/// Assert that a vector of floats are approximately equal
pub fn assert_vectors_approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> TestResult<()> {
    if a.len() != b.len() {
        return Err(format!("Vectors have different lengths: {} vs {}", a.len(), b.len()).into());
    }

    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if (x - y).abs() > epsilon {
            return Err(format!(
                "Vectors differ at index {}: {} vs {} (epsilon: {})",
                i, x, y, epsilon
            ).into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_with_timeout_success() -> TestResult {
        let result = with_timeout(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<i32, Box<dyn std::error::Error + Send + Sync>>(42)
        }).await?;

        assert_eq!(result, 42);
        Ok(())
    }

    #[tokio::test]
    async fn test_measure_time() -> TestResult {
        let (result, duration) = measure_time(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<i32, Box<dyn std::error::Error + Send + Sync>>(42)
        }).await?;

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(90)); // Allow some variance
        Ok(())
    }

    #[tokio::test]
    async fn test_wait_for_condition() -> TestResult {
        use std::sync::{Arc, Mutex};

        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        // Spawn a task that increments the counter after a delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            *counter_clone.lock().unwrap() = 5;
        });

        wait_for_condition(
            || *counter.lock().unwrap() == 5,
            Duration::from_millis(200),
            Duration::from_millis(10),
        ).await?;

        Ok(())
    }

    #[test]
    fn test_generate_test_id() {
        let id1 = generate_test_id();
        let id2 = generate_test_id();

        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_large_text() {
        let text = generate_large_text(1); // 1KB
        assert!(text.len() >= 1024);
        assert!(text.len() <= 1024 + 200); // Allow some variance due to rounding
    }

    #[test]
    fn test_generate_test_vectors() {
        let vectors = generate_test_vectors(5, 10);
        assert_eq!(vectors.len(), 5);
        assert!(vectors.iter().all(|v| v.len() == 10));
    }

    #[test]
    fn test_assert_approx_eq() -> TestResult {
        assert_approx_eq(1.0, 1.001, 0.01)?;

        let result = assert_approx_eq(1.0, 1.1, 0.01);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_assert_vectors_approx_eq() -> TestResult {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.001, 2.001, 3.001];

        assert_vectors_approx_eq(&a, &b, 0.01)?;

        let c = vec![1.0, 2.0, 3.1];
        let result = assert_vectors_approx_eq(&a, &c, 0.01);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_performance_benchmark() {
        let benchmark = PerformanceBenchmark::start("test_operation");
        std::thread::sleep(Duration::from_millis(10));
        let duration = benchmark.end();

        assert!(duration >= Duration::from_millis(5));
    }

    #[test]
    fn test_memory_tracker_returns_rss() {
        let tracker = MemoryTracker::start("rss_test");
        // On macOS and Linux, get_memory_usage should return Some with non-zero RSS
        let diff = tracker.end();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(diff.is_some(), "MemoryTracker should return RSS on macOS/Linux");
        }
    }

    #[test]
    fn test_memory_tracker_reports_allocation() {
        let tracker = MemoryTracker::start("alloc_test");
        // Allocate a non-trivial amount of memory to detect a change
        let _data: Vec<u8> = vec![0u8; 4 * 1024 * 1024]; // 4MB
        let diff = tracker.end();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let diff_val = diff.expect("should return a value");
            // The diff should be positive since we allocated 4MB
            assert!(diff_val > 0, "Expected positive memory diff after 4MB alloc, got {}", diff_val);
        }
    }
}