//! DiscoveryManager constructor and start/stop lifecycle methods.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::info;

use super::super::{
    DiscoveryResult, DiscoveryConfig,
    registry::ServiceRegistry,
    network::NetworkDiscovery,
    health::{HealthChecker, HealthConfig},
};
use super::core::DiscoveryManager;
use crate::unified_config::UnifiedConfigManager;
use super::super::registry::ServiceInfo;

impl DiscoveryManager {
    /// Create a new discovery manager
    pub async fn new(config: DiscoveryConfig) -> DiscoveryResult<Self> {
        info!("Initializing service discovery manager");

        let registry = ServiceRegistry::new(config.registry_path.as_ref())?;

        let auth_token = if config.enable_authentication {
            Some(ServiceInfo::generate_auth_token())
        } else {
            None
        };

        let network_discovery = if config.enable_network {
            Some(NetworkDiscovery::new(
                &config.multicast_address,
                config.multicast_port,
                auth_token.clone(),
            )?)
        } else {
            None
        };

        let health_config = HealthConfig {
            request_timeout: config.discovery_timeout,
            check_interval: config.health_check_interval,
            max_failures: 3,
            validate_process: true,
            custom_headers: HashMap::new(),
        };

        let health_checker = HealthChecker::new(health_config)?;
        let config_manager = UnifiedConfigManager::new::<std::path::PathBuf>(None);

        let (event_sender, _) = broadcast::channel(1000);

        let manager = Self {
            registry,
            network_discovery: network_discovery.unwrap_or_else(|| {
                NetworkDiscovery::new("239.255.42.42", 9999, None).unwrap()
            }),
            health_checker,
            config_manager,
            config: config.clone(),
            event_sender,
            known_services: Arc::new(RwLock::new(HashMap::new())),
            monitoring_handles: Arc::new(RwLock::new(Vec::new())),
        };

        info!("Service discovery manager initialized successfully");
        Ok(manager)
    }

    /// Start the discovery manager
    pub async fn start(&self) -> DiscoveryResult<()> {
        info!("Starting service discovery manager");

        if self.config.enable_network {
            self.network_discovery.start().await?;
            self.start_network_event_processing().await;
        }

        self.start_periodic_cleanup().await;

        info!("Service discovery manager started successfully");
        Ok(())
    }

    /// Stop the discovery manager
    pub async fn stop(&self) -> DiscoveryResult<()> {
        info!("Stopping service discovery manager");

        if self.config.enable_network {
            self.network_discovery.stop().await?;
        }

        let mut handles = self.monitoring_handles.write().await;
        for handle in handles.drain(..) {
            handle.abort();
        }

        info!("Service discovery manager stopped");
        Ok(())
    }
}
