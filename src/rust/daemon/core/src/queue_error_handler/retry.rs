// Retry strategy configuration and delay calculation

use rand::Rng;
use tracing::debug;

use super::error_types::{ErrorCategory, ErrorType};

/// Retry strategy configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Base retry delay in seconds
    pub base_delay: f64,
    /// Maximum delay between retries
    pub max_delay: f64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor (0-1) to randomize delays
    pub jitter_factor: f64,
    /// Default max retries (when error type not specified)
    pub default_max_retries: i32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            base_delay: 1.0,
            max_delay: 300.0, // 5 minutes
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            default_max_retries: 3,
        }
    }
}

impl RetryConfig {
    /// Calculate retry delay with exponential backoff and jitter
    pub fn calculate_delay(&self, retry_count: i32, error_type: ErrorType) -> f64 {
        // Base delay with exponential backoff
        let mut delay = self.base_delay * self.backoff_multiplier.powi(retry_count);

        // Apply different multipliers based on error category
        match error_type.category() {
            ErrorCategory::RateLimit => delay *= 3.0, // Longer delays for rate limits
            ErrorCategory::Resource => delay *= 2.0,  // Moderate delays for resource issues
            _ => {}
        }

        // Cap at max delay
        delay = delay.min(self.max_delay);

        // Add jitter to prevent thundering herd
        let mut rng = rand::thread_rng();
        let jitter = delay * self.jitter_factor * rng.gen::<f64>();
        delay += jitter;

        debug!(
            "Calculated retry delay for {}: {:.2}s (attempt {})",
            error_type.as_str(),
            delay,
            retry_count + 1
        );

        delay
    }

    /// Determine if error should be retried given the retry count
    pub fn should_retry(&self, error_type: ErrorType, retry_count: i32) -> bool {
        // Permanent errors never retry
        if error_type.category() == ErrorCategory::Permanent {
            return false;
        }

        // Check if max retries exceeded
        let max_retries = if error_type.max_retries() > 0 {
            error_type.max_retries()
        } else {
            self.default_max_retries
        };

        retry_count < max_retries
    }
}
