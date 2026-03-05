//! Discovery types: strategy enumeration and discovery events.

use super::super::health::HealthStatus;
use super::super::registry::ServiceInfo;

/// Discovery strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryStrategy {
    /// File-based registry lookup
    Registry,
    /// Network multicast discovery
    Network,
    /// Configuration file fallback
    Configuration,
    /// Default endpoint attempts
    Defaults,
}

/// Service discovery events
#[derive(Debug, Clone)]
pub enum ServiceDiscoveryEvent {
    /// Service discovered successfully
    ServiceDiscovered {
        service_name: String,
        service_info: ServiceInfo,
        strategy: DiscoveryStrategy,
    },

    /// Service lost or became unreachable
    ServiceLost {
        service_name: String,
        reason: String,
    },

    /// Service health changed
    HealthChanged {
        service_name: String,
        old_status: HealthStatus,
        new_status: HealthStatus,
    },

    /// Discovery strategy failed
    StrategyFailed {
        strategy: DiscoveryStrategy,
        service_name: String,
        error: String,
    },
}
