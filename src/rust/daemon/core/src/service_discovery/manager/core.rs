//! DiscoveryManager struct definition.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

use super::super::{
    DiscoveryConfig,
    registry::{ServiceRegistry, ServiceInfo},
    network::NetworkDiscovery,
    health::HealthChecker,
};
use super::types::{DiscoveryStrategy, ServiceDiscoveryEvent};
use crate::unified_config::UnifiedConfigManager;

/// Discovery manager orchestrating all discovery mechanisms
pub struct DiscoveryManager {
    /// Service registry for file-based discovery
    pub(super) registry: ServiceRegistry,

    /// Network discovery for multicast discovery
    pub(super) network_discovery: NetworkDiscovery,

    /// Health checker for service monitoring
    pub(super) health_checker: HealthChecker,

    /// Unified configuration manager for fallback
    pub(super) config_manager: UnifiedConfigManager,

    /// Discovery configuration
    pub(super) config: DiscoveryConfig,

    /// Event broadcaster for discovery events
    pub(super) event_sender: broadcast::Sender<ServiceDiscoveryEvent>,

    /// Currently known services
    pub(super) known_services: Arc<RwLock<HashMap<String, (ServiceInfo, DiscoveryStrategy)>>>,

    /// Service monitoring handles
    pub(super) monitoring_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}
