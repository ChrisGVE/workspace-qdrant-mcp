//! Logging configuration

use serde::{Deserialize, Serialize};

/// Logging configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub info_includes_connection_events: bool,
    pub info_includes_transport_details: bool,
    pub info_includes_retry_attempts: bool,
    pub info_includes_fallback_behavior: bool,
    pub error_includes_stack_trace: bool,
    pub error_includes_connection_state: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            info_includes_connection_events: true,
            info_includes_transport_details: true,
            info_includes_retry_attempts: true,
            info_includes_fallback_behavior: true,
            error_includes_stack_trace: true,
            error_includes_connection_state: true,
        }
    }
}
