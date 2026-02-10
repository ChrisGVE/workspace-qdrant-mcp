//! Service Discovery Manager
//!
//! This module provides the main orchestrator for service discovery, combining
//! file-based registry, network discovery, health checking, and configuration
//! fallback mechanisms into a unified discovery system.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, info, warn};

use super::{
    DiscoveryResult, DiscoveryConfig,
    registry::{ServiceRegistry, ServiceInfo},
    network::{NetworkDiscovery, DiscoveryEvent},
    health::{HealthChecker, HealthConfig, HealthStatus},
};
use crate::unified_config::UnifiedConfigManager;

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

/// Discovery manager orchestrating all discovery mechanisms
pub struct DiscoveryManager {
    /// Service registry for file-based discovery
    registry: ServiceRegistry,
    
    /// Network discovery for multicast discovery
    network_discovery: NetworkDiscovery,
    
    /// Health checker for service monitoring
    health_checker: HealthChecker,

    /// Unified configuration manager for fallback
    config_manager: UnifiedConfigManager,

    /// Discovery configuration
    config: DiscoveryConfig,
    
    /// Event broadcaster for discovery events
    event_sender: broadcast::Sender<ServiceDiscoveryEvent>,
    
    /// Currently known services
    known_services: Arc<RwLock<HashMap<String, (ServiceInfo, DiscoveryStrategy)>>>,
    
    /// Service monitoring handles
    monitoring_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

impl DiscoveryManager {
    /// Create a new discovery manager
    pub async fn new(config: DiscoveryConfig) -> DiscoveryResult<Self> {
        info!("Initializing service discovery manager");
        
        // Initialize components
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
                // Create a dummy network discovery that won't be used
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
        
        // Start network discovery if enabled
        if self.config.enable_network {
            self.network_discovery.start().await?;
            self.start_network_event_processing().await;
        }
        
        // Start periodic tasks
        self.start_periodic_cleanup().await;
        
        info!("Service discovery manager started successfully");
        Ok(())
    }

    /// Stop the discovery manager
    pub async fn stop(&self) -> DiscoveryResult<()> {
        info!("Stopping service discovery manager");
        
        // Stop network discovery
        if self.config.enable_network {
            self.network_discovery.stop().await?;
        }
        
        // Cancel monitoring tasks
        let mut handles = self.monitoring_handles.write().await;
        for handle in handles.drain(..) {
            handle.abort();
        }
        
        info!("Service discovery manager stopped");
        Ok(())
    }

    /// Register this service in the discovery system
    pub async fn register_service(
        &self,
        service_name: &str,
        service_info: ServiceInfo,
    ) -> DiscoveryResult<()> {
        info!("Registering service: {} at {}:{}", service_name, service_info.host, service_info.port);
        
        // Register in file-based registry
        self.registry.register_service(service_name, service_info.clone())?;
        
        // Announce via network discovery if enabled
        if self.config.enable_network {
            self.network_discovery.announce_service(service_name, &service_info).await?;
        }
        
        // Add to known services
        {
            let mut known_services = self.known_services.write().await;
            known_services.insert(service_name.to_string(), (service_info.clone(), DiscoveryStrategy::Registry));
        }
        
        // Send discovery event
        let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
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
        
        // Deregister from file-based registry
        let _ = self.registry.deregister_service(service_name)?;
        
        // Announce shutdown via network discovery if enabled
        if self.config.enable_network {
            let _ = self.network_discovery.announce_shutdown(service_name).await;
        }
        
        // Remove from known services
        {
            let mut known_services = self.known_services.write().await;
            known_services.remove(service_name);
        }
        
        // Send lost event
        let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceLost {
            service_name: service_name.to_string(),
            reason: "Deregistered".to_string(),
        });
        
        info!("Service {} deregistered successfully", service_name);
        Ok(())
    }

    /// Discover a specific service using all available strategies
    pub async fn discover_service(&self, service_name: &str) -> DiscoveryResult<Option<ServiceInfo>> {
        info!("Discovering service: {}", service_name);
        
        // Strategy 1: Check file-based registry
        if let Some(service_info) = self.try_registry_discovery(service_name).await? {
            return Ok(Some(service_info));
        }
        
        // Strategy 2: Network discovery
        if self.config.enable_network {
            if let Some(service_info) = self.try_network_discovery(service_name).await? {
                return Ok(Some(service_info));
            }
        }
        
        // Strategy 3: Configuration fallback
        if let Some(service_info) = self.try_configuration_discovery(service_name).await? {
            return Ok(Some(service_info));
        }
        
        // Strategy 4: Default endpoints
        if let Some(service_info) = self.try_default_discovery(service_name).await? {
            return Ok(Some(service_info));
        }
        
        warn!("Service {} not found using any discovery strategy", service_name);
        Ok(None)
    }

    /// Get all currently known services
    pub async fn get_known_services(&self) -> HashMap<String, ServiceInfo> {
        let known_services = self.known_services.read().await;
        known_services.iter()
            .map(|(name, (info, _))| (name.clone(), info.clone()))
            .collect()
    }

    /// Subscribe to discovery events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ServiceDiscoveryEvent> {
        self.event_sender.subscribe()
    }

    /// Check health of a specific service
    pub async fn check_service_health(&self, service_name: &str) -> DiscoveryResult<Option<HealthStatus>> {
        let known_services = self.known_services.read().await;
        
        if let Some((service_info, _)) = known_services.get(service_name) {
            match self.health_checker.check_service_health(service_name, service_info).await {
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

    /// Try discovery using file-based registry
    async fn try_registry_discovery(&self, service_name: &str) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying registry discovery for: {}", service_name);
        
        match self.registry.discover_service(service_name) {
            Ok(Some(service_info)) => {
                debug!("Found {} via registry discovery", service_name);
                self.update_known_service(service_name, service_info.clone(), DiscoveryStrategy::Registry).await;
                
                let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
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
                
                let _ = self.event_sender.send(ServiceDiscoveryEvent::StrategyFailed {
                    strategy: DiscoveryStrategy::Registry,
                    service_name: service_name.to_string(),
                    error: e.to_string(),
                });
                
                Ok(None)
            }
        }
    }

    /// Try discovery using network multicast
    async fn try_network_discovery(&self, service_name: &str) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying network discovery for: {}", service_name);
        
        match timeout(
            self.config.discovery_timeout,
            self.network_discovery.discover_services(vec![service_name.to_string()], self.config.discovery_timeout)
        ).await {
            Ok(Ok(mut discovered)) => {
                if let Some(service_info) = discovered.remove(service_name) {
                    debug!("Found {} via network discovery", service_name);
                    self.update_known_service(service_name, service_info.clone(), DiscoveryStrategy::Network).await;
                    
                    let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
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
                
                let _ = self.event_sender.send(ServiceDiscoveryEvent::StrategyFailed {
                    strategy: DiscoveryStrategy::Network,
                    service_name: service_name.to_string(),
                    error: e.to_string(),
                });
                
                Ok(None)
            }
            Err(_) => {
                warn!("Network discovery timeout for {}", service_name);
                
                let _ = self.event_sender.send(ServiceDiscoveryEvent::StrategyFailed {
                    strategy: DiscoveryStrategy::Network,
                    service_name: service_name.to_string(),
                    error: "Timeout".to_string(),
                });
                
                Ok(None)
            }
        }
    }

    /// Try discovery using configuration files
    async fn try_configuration_discovery(&self, service_name: &str) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying configuration discovery for: {}", service_name);

        // Load config from unified config search paths (includes env var overrides)
        let daemon_config = match self.config_manager.load_config(None) {
            Ok(config) => config,
            Err(e) => {
                debug!("Configuration load failed: {}", e);
                let _ = self.event_sender.send(ServiceDiscoveryEvent::StrategyFailed {
                    strategy: DiscoveryStrategy::Configuration,
                    service_name: service_name.to_string(),
                    error: format!("Config load failed: {}", e),
                });
                return Ok(None);
            }
        };

        let endpoint = &daemon_config.daemon_endpoint;

        // Only return a result for known service types that match the config
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
            // Verify the service is actually reachable
            match self.health_checker.check_service_health(service_name, &service_info).await {
                Ok(result) if result.status != HealthStatus::Unreachable => {
                    debug!("Found {} via configuration discovery at {}:{}", service_name, service_info.host, service_info.port);
                    self.update_known_service(service_name, service_info.clone(), DiscoveryStrategy::Configuration).await;

                    let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
                        service_name: service_name.to_string(),
                        service_info: service_info.clone(),
                        strategy: DiscoveryStrategy::Configuration,
                    });

                    Ok(Some(service_info))
                }
                _ => {
                    debug!("Configuration endpoint {}:{} not reachable for {}", endpoint.host, endpoint.grpc_port, service_name);
                    let _ = self.event_sender.send(ServiceDiscoveryEvent::StrategyFailed {
                        strategy: DiscoveryStrategy::Configuration,
                        service_name: service_name.to_string(),
                        error: format!("Endpoint {}:{} not reachable", endpoint.host, endpoint.grpc_port),
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
    async fn try_default_discovery(&self, service_name: &str) -> DiscoveryResult<Option<ServiceInfo>> {
        debug!("Trying default endpoint discovery for: {}", service_name);
        
        let default_info = match service_name {
            "rust-daemon" => Some(ServiceInfo::new("127.0.0.1".to_string(), 8080)),
            "python-mcp" => Some(ServiceInfo::new("127.0.0.1".to_string(), 8000)),
            _ => None,
        };
        
        if let Some(service_info) = default_info {
            // Verify the service is actually reachable
            match self.health_checker.check_service_health(service_name, &service_info).await {
                Ok(result) if result.is_reachable() => {
                    debug!("Found {} via default endpoints", service_name);
                    self.update_known_service(service_name, service_info.clone(), DiscoveryStrategy::Defaults).await;
                    
                    let _ = self.event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
                        service_name: service_name.to_string(),
                        service_info: service_info.clone(),
                        strategy: DiscoveryStrategy::Defaults,
                    });
                    
                    Ok(Some(service_info))
                }
                _ => {
                    debug!("Default endpoint for {} not reachable", service_name);
                    
                    let _ = self.event_sender.send(ServiceDiscoveryEvent::StrategyFailed {
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
    async fn update_known_service(&self, service_name: &str, service_info: ServiceInfo, strategy: DiscoveryStrategy) {
        let mut known_services = self.known_services.write().await;
        known_services.insert(service_name.to_string(), (service_info, strategy));
    }

    /// Start processing network discovery events
    async fn start_network_event_processing(&self) {
        let mut event_receiver = self.network_discovery.subscribe_events();
        let event_sender = self.event_sender.clone();
        let known_services = Arc::clone(&self.known_services);
        
        let handle = tokio::spawn(async move {
            while let Ok(event) = event_receiver.recv().await {
                match event {
                    DiscoveryEvent::ServiceDiscovered { service_name, service_info } => {
                        {
                            let mut known = known_services.write().await;
                            known.insert(service_name.clone(), (service_info.clone(), DiscoveryStrategy::Network));
                        }
                        
                        let _ = event_sender.send(ServiceDiscoveryEvent::ServiceDiscovered {
                            service_name,
                            service_info,
                            strategy: DiscoveryStrategy::Network,
                        });
                    }
                    
                    DiscoveryEvent::ServiceLost { service_name } => {
                        {
                            let mut known = known_services.write().await;
                            known.remove(&service_name);
                        }
                        
                        let _ = event_sender.send(ServiceDiscoveryEvent::ServiceLost {
                            service_name,
                            reason: "Network discovery lost".to_string(),
                        });
                    }
                    
                    _ => {
                        // Handle other events as needed
                    }
                }
            }
        });
        
        self.monitoring_handles.write().await.push(handle);
    }

    /// Start periodic cleanup tasks
    async fn start_periodic_cleanup(&self) {
        let registry = ServiceRegistry::new(self.config.registry_path.as_ref()).unwrap();
        let cleanup_interval = self.config.cleanup_interval;
        let event_sender = self.event_sender.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                match registry.cleanup_stale_entries() {
                    Ok(removed_services) => {
                        for service_name in removed_services {
                            let _ = event_sender.send(ServiceDiscoveryEvent::ServiceLost {
                                service_name,
                                reason: "Stale registry entry".to_string(),
                            });
                        }
                    }
                    Err(e) => {
                        warn!("Registry cleanup failed: {}", e);
                    }
                }
            }
        });
        
        self.monitoring_handles.write().await.push(handle);
    }
}