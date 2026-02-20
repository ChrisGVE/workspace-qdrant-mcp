//! Watch error state tracking - per-watch and cross-watch error management.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::error_types::{
    BackoffConfig, CircuitBreakerState, WatchErrorSummary, WatchHealthStatus,
};

/// Error state tracking for a single watch folder (Task 461)
///
/// Tracks consecutive errors, backoff state, and health status for coordinated
/// error handling between file watchers and queue processors.
#[derive(Debug, Clone)]
pub struct WatchErrorState {
    /// Number of consecutive errors for this watch
    pub consecutive_errors: u32,
    /// Total errors since watch started
    pub total_errors: u64,
    /// Timestamp of the last error
    pub last_error_time: Option<SystemTime>,
    /// Description of the last error
    pub last_error_message: Option<String>,
    /// Current backoff level (0 = no backoff, increases with each failure)
    pub backoff_level: u8,
    /// Timestamp of last successful processing
    pub last_successful_processing: Option<SystemTime>,
    /// Current health status
    pub health_status: WatchHealthStatus,
    /// Count of consecutive successes (for recovery tracking)
    pub consecutive_successes: u32,
    /// Time when backoff period ends (if in backoff)
    pub backoff_until: Option<SystemTime>,
    // Circuit breaker fields (Task 461.15)
    /// Timestamps of errors within the time window
    pub errors_in_window: Vec<SystemTime>,
    /// When the circuit breaker was opened (if Disabled or HalfOpen)
    pub circuit_opened_at: Option<SystemTime>,
    /// Number of retry attempts in half-open state
    pub half_open_attempts: u32,
    /// Consecutive successes in half-open state
    pub half_open_successes: u32,
}

impl Default for WatchErrorState {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchErrorState {
    /// Create a new error state with all fields initialized to healthy defaults
    pub fn new() -> Self {
        Self {
            consecutive_errors: 0,
            total_errors: 0,
            last_error_time: None,
            last_error_message: None,
            backoff_level: 0,
            last_successful_processing: None,
            health_status: WatchHealthStatus::Healthy,
            consecutive_successes: 0,
            backoff_until: None,
            errors_in_window: Vec::new(),
            circuit_opened_at: None,
            half_open_attempts: 0,
            half_open_successes: 0,
        }
    }

    /// Record an error occurrence
    ///
    /// Increments error counters and updates health status based on thresholds.
    /// Implements circuit breaker pattern with both consecutive error and
    /// time-window thresholds.
    /// Returns the calculated backoff delay in milliseconds (0 if no backoff needed).
    pub fn record_error(&mut self, error_message: &str, config: &BackoffConfig) -> u64 {
        let now = SystemTime::now();

        self.consecutive_errors += 1;
        self.total_errors += 1;
        self.last_error_time = Some(now);
        self.last_error_message = Some(error_message.to_string());
        self.consecutive_successes = 0;

        // Track errors in time window (Task 461.15)
        self.errors_in_window.push(now);

        // Remove errors outside the time window
        let window_start = now - Duration::from_secs(config.window_duration_secs);
        self.errors_in_window.retain(|t| *t >= window_start);

        // Check window-based threshold for circuit breaker
        let errors_in_window = self.errors_in_window.len() as u32;

        // If in half-open state, any error immediately reopens the circuit
        if self.health_status == WatchHealthStatus::HalfOpen {
            self.health_status = WatchHealthStatus::Disabled;
            self.circuit_opened_at = Some(now);
            self.half_open_attempts += 1;
            self.half_open_successes = 0;
            return self.calculate_backoff_delay(config);
        }

        // Update health status based on thresholds
        let should_open_circuit = self.consecutive_errors >= config.disable_threshold
            || errors_in_window >= config.window_error_threshold;

        self.health_status = if should_open_circuit {
            if self.circuit_opened_at.is_none() {
                self.circuit_opened_at = Some(now);
            }
            WatchHealthStatus::Disabled
        } else if self.consecutive_errors >= config.backoff_threshold {
            WatchHealthStatus::Backoff
        } else if self.consecutive_errors >= config.degraded_threshold {
            WatchHealthStatus::Degraded
        } else {
            WatchHealthStatus::Healthy
        };

        // Calculate backoff delay if needed
        let backoff_delay = if self.health_status == WatchHealthStatus::Backoff
            || self.health_status == WatchHealthStatus::Disabled
        {
            self.backoff_level = self.backoff_level.saturating_add(1);
            self.calculate_backoff_delay(config)
        } else {
            0
        };

        // Set backoff_until if there's a delay
        if backoff_delay > 0 {
            self.backoff_until = Some(now + Duration::from_millis(backoff_delay));
        }

        backoff_delay
    }

    /// Record a successful operation
    ///
    /// Resets error state on success, allowing recovery from degraded states.
    /// Handles half-open state for circuit breaker pattern.
    /// Returns true if health status changed (recovered to healthy).
    pub fn record_success(&mut self, config: &BackoffConfig) -> bool {
        let previous_status = self.health_status;

        self.last_successful_processing = Some(SystemTime::now());
        self.consecutive_successes += 1;

        // Handle half-open state (Task 461.15)
        if self.health_status == WatchHealthStatus::HalfOpen {
            self.half_open_successes += 1;

            if self.half_open_successes >= config.half_open_success_threshold {
                self.reset();
                return true;
            }

            return false;
        }

        // Check if we've had enough consecutive successes to reset
        if self.consecutive_successes >= config.success_reset_count {
            self.reset();
            return previous_status != WatchHealthStatus::Healthy;
        }

        // Gradual recovery: decrease backoff level on each success
        if self.backoff_level > 0 {
            self.backoff_level = self.backoff_level.saturating_sub(1);
        }

        // Clear backoff_until if backoff level is 0
        if self.backoff_level == 0 {
            self.backoff_until = None;
        }

        // Update health status based on recovery
        if self.consecutive_errors > 0 && self.consecutive_successes > 0 {
            self.health_status = if self.backoff_level > 0 {
                WatchHealthStatus::Backoff
            } else if self.consecutive_errors >= config.degraded_threshold {
                WatchHealthStatus::Degraded
            } else {
                WatchHealthStatus::Healthy
            };
        }

        previous_status != self.health_status
    }

    /// Reset error state to healthy defaults (close circuit)
    pub fn reset(&mut self) {
        self.consecutive_errors = 0;
        self.backoff_level = 0;
        self.health_status = WatchHealthStatus::Healthy;
        self.consecutive_successes = 0;
        self.backoff_until = None;
        // Circuit breaker reset (Task 461.15)
        self.circuit_opened_at = None;
        self.half_open_attempts = 0;
        self.half_open_successes = 0;
        self.errors_in_window.clear();
        // Note: We keep total_errors, last_error_time, and last_error_message
        // for historical tracking purposes
    }

    /// Check if circuit should transition to half-open (Task 461.15)
    pub fn should_attempt_half_open(&self, config: &BackoffConfig) -> bool {
        if self.health_status != WatchHealthStatus::Disabled {
            return false;
        }

        if let Some(opened_at) = self.circuit_opened_at {
            let elapsed = SystemTime::now()
                .duration_since(opened_at)
                .unwrap_or(Duration::ZERO);
            elapsed >= Duration::from_secs(config.cooldown_secs)
        } else {
            false
        }
    }

    /// Transition to half-open state for retry attempt (Task 461.15)
    pub fn transition_to_half_open(&mut self) {
        if self.health_status == WatchHealthStatus::Disabled {
            self.health_status = WatchHealthStatus::HalfOpen;
            self.half_open_successes = 0;
            self.consecutive_successes = 0;
        }
    }

    /// Manually reset the circuit breaker (for CLI use) (Task 461.15)
    pub fn manual_circuit_reset(&mut self) {
        self.reset();
    }

    /// Get circuit breaker state information (Task 461.15)
    pub fn get_circuit_state(&self) -> CircuitBreakerState {
        CircuitBreakerState {
            is_open: self.health_status == WatchHealthStatus::Disabled,
            is_half_open: self.health_status == WatchHealthStatus::HalfOpen,
            opened_at: self.circuit_opened_at,
            half_open_attempts: self.half_open_attempts,
            half_open_successes: self.half_open_successes,
            errors_in_window: self.errors_in_window.len() as u32,
        }
    }

    /// Calculate backoff delay using exponential backoff with jitter
    pub fn calculate_backoff_delay(&self, config: &BackoffConfig) -> u64 {
        if self.backoff_level == 0 {
            return 0;
        }

        // Exponential backoff: base_delay * 2^(level-1)
        let exponential_delay = config.base_delay_ms
            .saturating_mul(1u64 << (self.backoff_level.saturating_sub(1) as u64).min(10));

        // Cap at max delay
        let capped_delay = exponential_delay.min(config.max_delay_ms);

        // Add jitter (+/-10% of delay) to prevent thundering herd
        let jitter_range = capped_delay / 10;
        let jitter = if jitter_range > 0 {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64;
            (now % (jitter_range * 2)).saturating_sub(jitter_range)
        } else {
            0
        };

        capped_delay.saturating_add(jitter)
    }

    /// Check if currently in backoff period
    pub fn is_in_backoff(&self) -> bool {
        if let Some(backoff_until) = self.backoff_until {
            SystemTime::now() < backoff_until
        } else {
            false
        }
    }

    /// Get remaining backoff time in milliseconds (0 if not in backoff)
    pub fn remaining_backoff_ms(&self) -> u64 {
        if let Some(backoff_until) = self.backoff_until {
            backoff_until
                .duration_since(SystemTime::now())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Check if watch should be disabled (circuit breaker open)
    pub fn should_disable(&self) -> bool {
        self.health_status == WatchHealthStatus::Disabled
    }

    /// Check if watch can process (not in backoff and not disabled)
    pub fn can_process(&self) -> bool {
        !self.should_disable() && !self.is_in_backoff()
    }
}

/// Manager for tracking error states across all watches (Task 461)
///
/// Thread-safe container for WatchErrorState instances keyed by watch_id.
#[derive(Debug)]
pub struct WatchErrorTracker {
    /// Error states keyed by watch_id
    states: Arc<RwLock<HashMap<String, WatchErrorState>>>,
    /// Shared backoff configuration
    config: BackoffConfig,
}

impl WatchErrorTracker {
    /// Create a new error tracker with default configuration
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            config: BackoffConfig::default(),
        }
    }

    /// Create a new error tracker with custom configuration
    pub fn with_config(config: BackoffConfig) -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Get or create error state for a watch_id
    pub async fn get_or_create(&self, watch_id: &str) -> WatchErrorState {
        let states = self.states.read().await;
        states.get(watch_id).cloned().unwrap_or_default()
    }

    /// Record an error for a watch
    ///
    /// Returns the backoff delay in milliseconds.
    pub async fn record_error(&self, watch_id: &str, error_message: &str) -> u64 {
        let mut states = self.states.write().await;
        let state = states.entry(watch_id.to_string()).or_insert_with(WatchErrorState::new);
        let delay = state.record_error(error_message, &self.config);

        debug!(
            "Watch '{}' error recorded: consecutive={}, status={:?}, backoff_ms={}",
            watch_id, state.consecutive_errors, state.health_status, delay
        );

        delay
    }

    /// Record a successful operation for a watch
    ///
    /// Returns true if health status improved.
    pub async fn record_success(&self, watch_id: &str) -> bool {
        let mut states = self.states.write().await;
        let state = states.entry(watch_id.to_string()).or_insert_with(WatchErrorState::new);
        let improved = state.record_success(&self.config);

        if improved {
            debug!(
                "Watch '{}' health improved: status={:?}, consecutive_successes={}",
                watch_id, state.health_status, state.consecutive_successes
            );
        }

        improved
    }

    /// Check if a watch can process (not in backoff and not disabled)
    pub async fn can_process(&self, watch_id: &str) -> bool {
        let states = self.states.read().await;
        states.get(watch_id).map(|s| s.can_process()).unwrap_or(true)
    }

    /// Get health status for a watch
    pub async fn get_health_status(&self, watch_id: &str) -> WatchHealthStatus {
        let states = self.states.read().await;
        states.get(watch_id).map(|s| s.health_status).unwrap_or(WatchHealthStatus::Healthy)
    }

    /// Get all watch health statuses
    pub async fn get_all_health_statuses(&self) -> HashMap<String, WatchHealthStatus> {
        let states = self.states.read().await;
        states.iter().map(|(k, v)| (k.clone(), v.health_status)).collect()
    }

    /// Get error summary for all watches
    pub async fn get_error_summary(&self) -> Vec<WatchErrorSummary> {
        let states = self.states.read().await;
        states.iter().map(|(id, state)| WatchErrorSummary {
            watch_id: id.clone(),
            health_status: state.health_status,
            consecutive_errors: state.consecutive_errors,
            total_errors: state.total_errors,
            backoff_level: state.backoff_level,
            remaining_backoff_ms: state.remaining_backoff_ms(),
            last_error_message: state.last_error_message.clone(),
        }).collect()
    }

    /// Reset error state for a watch (manual recovery)
    pub async fn reset_watch(&self, watch_id: &str) {
        let mut states = self.states.write().await;
        if let Some(state) = states.get_mut(watch_id) {
            state.reset();
            info!("Watch '{}' error state manually reset", watch_id);
        }
    }

    /// Remove error tracking for a watch (when watch is removed)
    pub async fn remove_watch(&self, watch_id: &str) {
        let mut states = self.states.write().await;
        states.remove(watch_id);
    }

    /// Get error state for a specific watch (Task 461.5)
    pub fn get_state(&self, watch_id: &str) -> Option<WatchErrorState> {
        self.states.try_read().ok().and_then(|states| states.get(watch_id).cloned())
    }

    /// Set error state for a specific watch (Task 461.5)
    pub fn set_state(&self, watch_id: &str, state: WatchErrorState) {
        match self.states.try_write() {
            Ok(mut states) => {
                states.insert(watch_id.to_string(), state);
            }
            Err(_) => {
                warn!("Could not set error state for watch {} - lock contention", watch_id);
            }
        }
    }

    /// Get error summary for a specific watch (Task 461.5)
    pub fn get_summary(&self, watch_id: &str) -> Option<WatchErrorSummary> {
        self.states.try_read().ok().and_then(|states| {
            states.get(watch_id).map(|state| WatchErrorSummary {
                watch_id: watch_id.to_string(),
                health_status: state.health_status,
                consecutive_errors: state.consecutive_errors,
                total_errors: state.total_errors,
                backoff_level: state.backoff_level,
                remaining_backoff_ms: state.remaining_backoff_ms(),
                last_error_message: state.last_error_message.clone(),
            })
        })
    }
}

impl Default for WatchErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}
