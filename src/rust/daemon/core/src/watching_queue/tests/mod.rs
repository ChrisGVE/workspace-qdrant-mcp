//! Tests for the watching_queue module.
//!
//! Split into focused submodules:
//! - `types_tests` - WatchType, collection constants, library watch config
//! - `error_state_tests` - WatchErrorState and WatchErrorTracker
//! - `coordinator_tests` - WatchQueueCoordinator capacity management
//! - `circuit_breaker_tests` - Circuit breaker state transitions
//! - `error_feedback_tests` - ProcessingErrorFeedback and ErrorFeedbackManager

mod types_tests;
mod error_state_tests;
mod coordinator_tests;
mod circuit_breaker_tests;
mod error_feedback_tests;
