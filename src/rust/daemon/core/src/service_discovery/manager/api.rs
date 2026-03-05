//! Public API methods for DiscoveryManager: register, deregister, discover, query.

use std::collections::HashMap;
use tokio::sync::broadcast;
use tracing::{info, warn};

use super::super::{health::HealthStatus, registry::ServiceInfo, DiscoveryResult};
use super::core::DiscoveryManager;
use super::types::{DiscoveryStrategy, ServiceDiscoveryEvent};

impl DiscoveryManager {
    /// Register this service in the discovery system
    pub async fn register_service(
        &self,
        service_name: &str,
        service_info: ServiceInfo,
    ) -> DiscoveryResult<()> {
        info!(
            "Registering service: {} at {}:{}",
            service_name, service_info.host, service_info.port
        );

        self.registry
            .register_service(service_name, service_info.clone())?;

        if self.config.enable_network {
            self.network_discovery
                .announce_service(service_name, &service_info)
                .await?;
        }

        {
            let mut known_services = self.known_services.write().await;
            known_services.insert(
                service_name.to_string(),
                (service_info.clone(), DiscoveryStrategy::Registry),
            );
        }

        let _ = self
            .event_sender
            .send(ServiceDiscoveryEvent::ServiceDiscovered {
                service_name: service_name.to_string(),
                service_info,
                strategy: DiscoveryStrategy::Registry,
            });

        info!("Service {} registered successfully", service_name);
        Ok(())
    }

    /// Deregister this service from the discovery system
    pub async fn deregister_service(&self, service_name: &str) -> DiscoveryResult<()> {
        info!("Deregistering service: {}", service_name);

        let _ = self.registry.deregister_service(service_name)?;

        if self.config.enable_network {
            let _ = self.network_discovery.announce_shutdown(service_name).await;
        }

        {
            let mut known_services = self.known_services.write().await;
            known_services.remove(service_name);
        }

        let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceLost {
            service_name: service_name.to_string(),
            reason: "Deregistered".to_string(),
        });

        info!("Service {} deregistered successfully", service_name);
        Ok(())
    }

    /// Discover a specific service using all available strategies
    pub async fn discover_service(
        &self,
        service_name: &str,
    ) -> DiscoveryResult<Option<ServiceInfo>> {
        info!("Discovering service: {}", service_name);

        if let Some(service_info) = self.try_registry_discovery(service_name).await? {
            return Ok(Some(service_info));
        }

        if self.config.enable_network {
            if let Some(service_info) = self.try_network_discovery(service_name).await? {
                return Ok(Some(service_info));
            }
        }

        if let Some(service_info) = self.try_configuration_discovery(service_name).await? {
            return Ok(Some(service_info));
        }

        if let Some(service_info) = self.try_default_discovery(service_name).await? {
            return Ok(Some(service_info));
        }

        warn!(
            "Service {} not found using any discovery strategy",
            service_name
        );
        Ok(None)
    }

    /// Get all currently known services
    pub async fn get_known_services(&self) -> HashMap<String, ServiceInfo> {
        let known_services = self.known_services.read().await;
        known_services
            .iter()
            .map(|(name, (info, _))| (name.clone(), info.clone()))
            .collect()
    }

    /// Subscribe to discovery events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ServiceDiscoveryEvent> {
        self.event_sender.subscribe()
    }

    /// Check health of a specific service
    pub async fn check_service_health(
        &self,
        service_name: &str,
    ) -> DiscoveryResult<Option<HealthStatus>> {
        let known_services = self.known_services.read().await;

        if let Some((service_info, _)) = known_services.get(service_name) {
            match self
                .health_checker
                .check_service_health(service_name, service_info)
                .await
            {
                Ok(result) => Ok(Some(result.status)),
                Err(e) => {
                    warn!("Health check failed for {}: {}", service_name, e);
                    Ok(Some(HealthStatus::Unreachable))
                }
            }
        } else {
            Ok(None)
        }
    }
}
