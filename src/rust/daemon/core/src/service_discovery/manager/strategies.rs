//! Discovery strategy implementations: registry, network, configuration, defaults.

use tokio::time::timeout;
use tracing::debug;
use tracing::warn;

use super::super::{health::HealthStatus, registry::ServiceInfo, DiscoveryResult};
use super::core::DiscoveryManager;
use super::types::{DiscoveryStrategy, ServiceDiscoveryEvent};

impl DiscoveryManager {
    /// Try discovery using file-based registry
    pub(super) async fn try_registry_discovery(
        &self,
        service_name: &str,
    ) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying registry discovery for: {}", service_name);

        match self.registry.discover_service(service_name) {
            Ok(Some(service_info)) => {
                debug!("Found {} via registry discovery", service_name);
                self.update_known_service(
                    service_name,
                    service_info.clone(),
                    DiscoveryStrategy::Registry,
                )
                .await;

                let _ = self
                    .event_sender
                    .send(ServiceDiscoveryEvent::ServiceDiscovered {
                        service_name: service_name.to_string(),
                        service_info: service_info.clone(),
                        strategy: DiscoveryStrategy::Registry,
                    });

                Ok(Some(service_info))
            }
            Ok(None) => {
                debug!("Service {} not found in registry", service_name);
                Ok(None)
            }
            Err(e) => {
                warn!("Registry discovery failed for {}: {}", service_name, e);

                let _ = self
                    .event_sender
                    .send(ServiceDiscoveryEvent::StrategyFailed {
                        strategy: DiscoveryStrategy::Registry,
                        service_name: service_name.to_string(),
                        error: e.to_string(),
                    });

                Ok(None)
            }
        }
    }

    /// Try discovery using network multicast
    pub(super) async fn try_network_discovery(
        &self,
        service_name: &str,
    ) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying network discovery for: {}", service_name);

        match timeout(
            self.config.discovery_timeout,
            self.network_discovery.discover_services(
                vec![service_name.to_string()],
                self.config.discovery_timeout,
            ),
        )
        .await
        {
            Ok(Ok(mut discovered)) => {
                if let Some(service_info) = discovered.remove(service_name) {
                    debug!("Found {} via network discovery", service_name);
                    self.update_known_service(
                        service_name,
                        service_info.clone(),
                        DiscoveryStrategy::Network,
                    )
                    .await;

                    let _ = self
                        .event_sender
                        .send(ServiceDiscoveryEvent::ServiceDiscovered {
                            service_name: service_name.to_string(),
                            service_info: service_info.clone(),
                            strategy: DiscoveryStrategy::Network,
                        });

                    Ok(Some(service_info))
                } else {
                    debug!("Service {} not found via network discovery", service_name);
                    Ok(None)
                }
            }
            Ok(Err(e)) => {
                warn!("Network discovery failed for {}: {}", service_name, e);

                let _ = self
                    .event_sender
                    .send(ServiceDiscoveryEvent::StrategyFailed {
                        strategy: DiscoveryStrategy::Network,
                        service_name: service_name.to_string(),
                        error: e.to_string(),
                    });

                Ok(None)
            }
            Err(_) => {
                warn!("Network discovery timeout for {}", service_name);

                let _ = self
                    .event_sender
                    .send(ServiceDiscoveryEvent::StrategyFailed {
                        strategy: DiscoveryStrategy::Network,
                        service_name: service_name.to_string(),
                        error: "Timeout".to_string(),
                    });

                Ok(None)
            }
        }
    }

    /// Try discovery using configuration files
    pub(super) async fn try_configuration_discovery(
        &self,
        service_name: &str,
    ) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying configuration discovery for: {}", service_name);

        let daemon_config = match self.config_manager.load_config(None) {
            Ok(config) => config,
            Err(e) => {
                debug!("Configuration load failed: {}", e);
                let _ = self
                    .event_sender
                    .send(ServiceDiscoveryEvent::StrategyFailed {
                        strategy: DiscoveryStrategy::Configuration,
                        service_name: service_name.to_string(),
                        error: format!("Config load failed: {}", e),
                    });
                return Ok(None);
            }
        };

        let endpoint = &daemon_config.daemon_endpoint;

        let service_info = match service_name {
            "rust-daemon" => {
                let mut info = ServiceInfo::new(endpoint.host.clone(), endpoint.grpc_port);
                info.health_endpoint = endpoint.health_endpoint.clone();
                if let Some(ref token) = endpoint.auth_token {
                    info = info.with_auth_token(token.clone());
                }
                Some(info)
            }
            _ => None,
        };

        if let Some(service_info) = service_info {
            match self
                .health_checker
                .check_service_health(service_name, &service_info)
                .await
            {
                Ok(result) if result.status != HealthStatus::Unreachable => {
                    debug!(
                        "Found {} via configuration discovery at {}:{}",
                        service_name, service_info.host, service_info.port
                    );
                    self.update_known_service(
                        service_name,
                        service_info.clone(),
                        DiscoveryStrategy::Configuration,
                    )
                    .await;

                    let _ = self
                        .event_sender
                        .send(ServiceDiscoveryEvent::ServiceDiscovered {
                            service_name: service_name.to_string(),
                            service_info: service_info.clone(),
                            strategy: DiscoveryStrategy::Configuration,
                        });

                    Ok(Some(service_info))
                }
                _ => {
                    debug!(
                        "Configuration endpoint {}:{} not reachable for {}",
                        endpoint.host, endpoint.grpc_port, service_name
                    );
                    let _ = self
                        .event_sender
                        .send(ServiceDiscoveryEvent::StrategyFailed {
                            strategy: DiscoveryStrategy::Configuration,
                            service_name: service_name.to_string(),
                            error: format!(
                                "Endpoint {}:{} not reachable",
                                endpoint.host, endpoint.grpc_port
                            ),
                        });
                    Ok(None)
                }
            }
        } else {
            debug!("No configuration endpoint for service: {}", service_name);
            Ok(None)
        }
    }

    /// Try discovery using default endpoints
    pub(super) async fn try_default_discovery(
        &self,
        service_name: &str,
    ) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying default endpoint discovery for: {}", service_name);

        let default_info = match service_name {
            "rust-daemon" => Some(ServiceInfo::new("127.0.0.1".to_string(), 8080)),
            "python-mcp" => Some(ServiceInfo::new("127.0.0.1".to_string(), 8000)),
            _ => None,
        };

        if let Some(service_info) = default_info {
            match self
                .health_checker
                .check_service_health(service_name, &service_info)
                .await
            {
                Ok(result) if result.is_reachable() => {
                    debug!("Found {} via default endpoints", service_name);
                    self.update_known_service(
                        service_name,
                        service_info.clone(),
                        DiscoveryStrategy::Defaults,
                    )
                    .await;

                    let _ = self
                        .event_sender
                        .send(ServiceDiscoveryEvent::ServiceDiscovered {
                            service_name: service_name.to_string(),
                            service_info: service_info.clone(),
                            strategy: DiscoveryStrategy::Defaults,
                        });

                    Ok(Some(service_info))
                }
                _ => {
                    debug!("Default endpoint for {} not reachable", service_name);

                    let _ = self
                        .event_sender
                        .send(ServiceDiscoveryEvent::StrategyFailed {
                            strategy: DiscoveryStrategy::Defaults,
                            service_name: service_name.to_string(),
                            error: "Service not reachable".to_string(),
                        });

                    Ok(None)
                }
            }
        } else {
            debug!("No default endpoint defined for service: {}", service_name);
            Ok(None)
        }
    }

    /// Update known service information
    pub(super) async fn update_known_service(
        &self,
        service_name: &str,
        service_info: ServiceInfo,
        strategy: DiscoveryStrategy,
    ) {
        let mut known_services = self.known_services.write().await;
        known_services.insert(service_name.to_string(), (service_info, strategy));
    }
}
