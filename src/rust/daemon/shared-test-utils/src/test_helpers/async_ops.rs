//! Async operation helpers for testing: timeouts, retries, waits, concurrency

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use tokio::time::timeout;
use tokio_test::{assert_pending, assert_ready, assert_ready_err, assert_ready_ok, task};

use crate::config::DEFAULT_TEST_TIMEOUT;
use crate::TestResult;

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

    Err(Box::<dyn std::error::Error + Send + Sync>::from(
        "Async condition was not met within timeout",
    ))
}

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
        )
        .into());
    }

    if elapsed > expected_max_duration {
        return Err(format!(
            "Operation took too long: {:?} > {:?}",
            elapsed, expected_max_duration
        )
        .into());
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
            let _permit = sem.acquire().await.map_err(|e| {
                Box::<dyn std::error::Error + Send + Sync>::from(format!(
                    "Semaphore error: {}",
                    e
                ))
            })?;
            op.await
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        let result = handle
            .await
            .map_err(|e| {
                Box::<dyn std::error::Error + Send + Sync>::from(format!("Join error: {}", e))
            })??;
        results.push(result);
    }

    Ok(results)
}

/// Concurrent execution helper
pub async fn run_concurrent<F, T>(
    operations: Vec<F>,
    max_concurrent: usize,
) -> Vec<Result<T, Box<dyn std::error::Error + Send + Sync>>>
where
    F: std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>
        + Send
        + 'static,
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
