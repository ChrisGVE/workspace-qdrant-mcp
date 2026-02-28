use super::*;
use std::time::Duration;

#[test]
fn test_error_creation() {
    let error = WorkspaceError::configuration("Test configuration error");
    assert_eq!(error.category(), "configuration");
    assert!(!error.is_retryable());
    assert_eq!(error.severity(), ErrorSeverity::High);
}

#[test]
fn test_recovery_strategy() {
    let strategy = ErrorRecoveryStrategy::network();
    assert_eq!(strategy.max_retries, 5);
    assert!(strategy.exponential_backoff);

    let delay = strategy.calculate_delay(2);
    assert_eq!(delay, Duration::from_millis(1000));
}

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("Test error: {0}")]
    Test(String),
}

#[tokio::test]
async fn test_circuit_breaker() {
    let mut circuit_breaker = CircuitBreaker::new("test", 2);

    // Simulate failures
    let result1 = circuit_breaker
        .execute(async { Err::<(), TestError>(TestError::Test("test error".to_string())) })
        .await;
    assert!(result1.is_err());

    let result2 = circuit_breaker
        .execute(async { Err::<(), TestError>(TestError::Test("test error".to_string())) })
        .await;
    assert!(result2.is_err());

    // Circuit should be open now
    let status = circuit_breaker.status();
    assert_eq!(status.state, "open");
    assert_eq!(status.failure_count, 2);
}

#[test]
fn test_error_monitor() {
    let monitor = DefaultErrorMonitor::new();
    let error = WorkspaceError::network("Test network error", 1, 3);

    monitor.report_error(&error, Some("test context"));
    monitor.report_recovery("network", 2);

    let stats = monitor.get_error_stats();
    assert_eq!(stats.total_errors, 1);
    assert_eq!(stats.recovery_successes, 1);
    assert!(stats.errors_by_category.contains_key("network"));
}
