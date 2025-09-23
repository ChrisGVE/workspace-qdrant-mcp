//! Simple test binary to verify error handling, retry, and circuit breaker functionality

use std::time::Duration;
use tokio::time::sleep;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};
use workspace_qdrant_daemon::grpc::{RetryStrategy, RetryConfig, CircuitBreaker, CircuitBreakerConfig};

#[tokio::main]
async fn main() -> DaemonResult<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("Testing error handling, retry logic, and circuit breaker functionality...\n");

    // Test 1: Basic error types
    test_error_types().await?;

    // Test 2: Retry logic
    test_retry_logic().await?;

    // Test 3: Circuit breaker
    test_circuit_breaker().await?;

    println!("All tests completed successfully!");
    Ok(())
}

async fn test_error_types() -> DaemonResult<()> {
    println!("=== Testing Error Types ===");

    // Test network errors
    let network_error = DaemonError::NetworkConnection {
        message: "Connection refused".to_string()
    };
    println!("Network error: {}", network_error);

    // Test timeout errors
    let timeout_error = DaemonError::NetworkTimeout { timeout_ms: 5000 };
    println!("Timeout error: {}", timeout_error);

    // Test circuit breaker errors
    let cb_error = DaemonError::CircuitBreakerOpen {
        service: "test-service".to_string()
    };
    println!("Circuit breaker error: {}", cb_error);

    // Test error cloning
    let cloned_error = network_error.clone();
    println!("Cloned error: {}", cloned_error);

    println!("✓ Error types test passed\n");
    Ok(())
}

async fn test_retry_logic() -> DaemonResult<()> {
    println!("=== Testing Retry Logic ===");

    // Create retry configuration
    let config = RetryConfig::new()
        .max_attempts(3)
        .initial_delay(Duration::from_millis(100))
        .backoff_multiplier(2.0)
        .enable_jitter(false);

    let retry_strategy = RetryStrategy::with_config(config)?;

    // Test successful operation on first attempt
    let result = retry_strategy.execute(|| async {
        Ok::<i32, DaemonError>(42)
    }).await?;
    println!("✓ Successful operation: {}", result);

    // Test operation that succeeds after retries
    let attempt_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let result = retry_strategy.execute(|| {
        let count = attempt_count.clone();
        async move {
            let current_attempt = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            if current_attempt < 3 {
                Err(DaemonError::NetworkConnection {
                    message: "Temporary failure".to_string()
                })
            } else {
                Ok::<i32, DaemonError>(100)
            }
        }
    }).await?;
    println!("✓ Operation succeeded after retries: {}", result);

    // Test non-retryable error
    let result = retry_strategy.execute(|| async {
        Err::<i32, DaemonError>(DaemonError::InvalidInput {
            message: "Bad input".to_string()
        })
    }).await;

    match result {
        Err(DaemonError::InvalidInput { .. }) => {
            println!("✓ Non-retryable error handled correctly");
        }
        _ => {
            return Err(DaemonError::Internal {
                message: "Expected non-retryable error".to_string()
            });
        }
    }

    println!("✓ Retry logic test passed\n");
    Ok(())
}

async fn test_circuit_breaker() -> DaemonResult<()> {
    println!("=== Testing Circuit Breaker ===");

    // Create circuit breaker configuration
    let config = CircuitBreakerConfig::new()
        .failure_threshold(2)
        .success_threshold(2)
        .recovery_timeout(Duration::from_millis(100))
        .minimum_requests(1)
        .request_timeout(Duration::from_millis(1000));

    let circuit_breaker = CircuitBreaker::new("test-service".to_string(), config)?;

    // Test successful operation
    let result = circuit_breaker.execute(|| async {
        Ok::<i32, DaemonError>(42)
    }).await?;
    println!("✓ Successful operation through circuit breaker: {}", result);

    // Generate failures to open circuit
    for i in 1..=3 {
        let result = circuit_breaker.execute(|| async {
            Err::<i32, DaemonError>(DaemonError::NetworkConnection {
                message: "Service failure".to_string()
            })
        }).await;

        if result.is_err() {
            println!("✓ Failure {} recorded", i);
        }
    }

    // Check if circuit is open
    let stats = circuit_breaker.stats().await;
    println!("Circuit breaker state: {:?}", stats.state);
    println!("Total failures: {}", stats.total_failures);

    // Wait for recovery timeout
    sleep(Duration::from_millis(150)).await;

    // Test recovery
    let result = circuit_breaker.execute(|| async {
        Ok::<i32, DaemonError>(200)
    }).await?;
    println!("✓ Recovery operation succeeded: {}", result);

    let final_stats = circuit_breaker.stats().await;
    println!("Final circuit breaker state: {:?}", final_stats.state);

    println!("✓ Circuit breaker test passed\n");
    Ok(())
}