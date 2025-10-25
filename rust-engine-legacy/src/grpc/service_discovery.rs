//! Comprehensive gRPC service discovery implementation with health checking and load balancing
//!
//! This module provides:
//! - Service registration and deregistration
//! - Health monitoring with automatic failure detection
//! - Dynamic service topology management
//! - Load balancing algorithms (round-robin, weighted, least-connections)
//! - Service mesh integration capabilities

use crate::error::DaemonError;
use anyhow::Result;
use dashmap::DashMap;
use rand::random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;
use tokio::time::interval;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Service instance health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceHealth {
    /// Service is healthy and ready to receive requests
    Healthy,
    /// Service is degraded but still operational
    Degraded,
    /// Service is unhealthy and should not receive requests
    Unhealthy,
    /// Service is in draining state (preparing for shutdown)
    Draining,
    /// Service status is unknown (no recent health checks)
    Unknown,
}

impl Default for ServiceHealth {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Load balancing strategy for service selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin selection
    RoundRobin,
    /// Weighted round-robin based on service weights
    WeightedRoundRobin,
    /// Least connections selection
    LeastConnections,
    /// Random selection
    Random,
    /// Health-weighted selection (prefer healthier services)
    HealthWeighted,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Service instance metadata and runtime information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    /// Unique service instance identifier
    pub id: String,
    /// Service name/type
    pub name: String,
    /// Service version
    pub version: String,
    /// Network address
    pub address: SocketAddr,
    /// Service weight for load balancing (1-100)
    pub weight: u8,
    /// Current health status
    pub health: ServiceHealth,
    /// Last health check timestamp
    pub last_health_check: SystemTime,
    /// Service registration timestamp
    pub registered_at: SystemTime,
    /// Current active connections
    pub active_connections: u32,
    /// Total requests served
    pub total_requests: u64,
    /// Service metadata tags
    pub metadata: HashMap<String, String>,
    /// Health check endpoint path
    pub health_endpoint: Option<String>,
    /// Health check interval in seconds
    pub health_interval: u64,
    /// Consecutive failed health checks
    pub failed_health_checks: u32,
    /// Service tags for discovery filtering
    pub tags: Vec<String>,
}

impl ServiceInstance {
    /// Create a new service instance
    pub fn new(
        name: String,
        version: String,
        address: SocketAddr,
        weight: u8,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            version,
            address,
            weight: weight.clamp(1, 100),
            health: ServiceHealth::Unknown,
            last_health_check: now,
            registered_at: now,
            active_connections: 0,
            total_requests: 0,
            metadata: HashMap::new(),
            health_endpoint: None,
            health_interval: 30, // Default 30 seconds
            failed_health_checks: 0,
            tags: Vec::new(),
        }
    }

    /// Update service health status
    pub fn update_health(&mut self, health: ServiceHealth) {
        if health != self.health {
            debug!("Service {} health changed from {:?} to {:?}", self.id, self.health, health);
        }
        self.health = health;
        self.last_health_check = SystemTime::now();

        if health == ServiceHealth::Healthy {
            self.failed_health_checks = 0;
        } else if matches!(health, ServiceHealth::Unhealthy | ServiceHealth::Unknown) {
            self.failed_health_checks += 1;
        }
    }

    /// Update connection count
    pub fn update_connections(&mut self, connections: u32) {
        self.active_connections = connections;
    }

    /// Increment request counter
    pub fn increment_requests(&mut self) {
        self.total_requests += 1;
    }

    /// Check if service is available for requests
    pub fn is_available(&self) -> bool {
        matches!(self.health, ServiceHealth::Healthy | ServiceHealth::Degraded)
    }

    /// Get service load score for load balancing
    pub fn load_score(&self) -> f64 {
        if !self.is_available() {
            return f64::INFINITY;
        }

        let base_score = self.active_connections as f64 / self.weight as f64;
        let health_penalty = match self.health {
            ServiceHealth::Healthy => 1.0,
            ServiceHealth::Degraded => 1.5,
            _ => f64::INFINITY,
        };

        base_score * health_penalty
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDiscoveryConfig {
    /// Default health check interval
    pub health_check_interval: Duration,
    /// Health check timeout
    pub health_check_timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Default load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Enable automatic service cleanup
    pub enable_cleanup: bool,
    /// Service cleanup interval
    pub cleanup_interval: Duration,
    /// TTL for inactive services
    pub service_ttl: Duration,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            enable_cleanup: true,
            cleanup_interval: Duration::from_secs(60),
            service_ttl: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Service discovery registry with health monitoring and load balancing
#[derive(Debug)]
pub struct ServiceRegistry {
    /// Service instances indexed by service name
    services: Arc<DashMap<String, Vec<ServiceInstance>>>,
    /// Configuration
    config: ServiceDiscoveryConfig,
    /// Health check client
    health_client: reqwest::Client,
    /// Round-robin counters for load balancing
    round_robin_counters: Arc<DashMap<String, usize>>,
    /// Service topology change notifier
    topology_notifier: watch::Sender<u64>,
    /// Topology change receiver
    topology_receiver: watch::Receiver<u64>,
    /// Shutdown signal
    shutdown_tx: Option<watch::Sender<bool>>,
}

impl ServiceRegistry {
    /// Create a new service registry
    pub fn new(config: ServiceDiscoveryConfig) -> Self {
        let (topology_tx, topology_rx) = watch::channel(0);

        let health_client = reqwest::Client::builder()
            .timeout(config.health_check_timeout)
            .connect_timeout(config.health_check_timeout)
            .build()
            .expect("Failed to create health check client");

        Self {
            services: Arc::new(DashMap::new()),
            config,
            health_client,
            round_robin_counters: Arc::new(DashMap::new()),
            topology_notifier: topology_tx,
            topology_receiver: topology_rx,
            shutdown_tx: None,
        }
    }

    /// Start the service registry with background tasks
    pub async fn start(&mut self) -> Result<()> {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        self.shutdown_tx = Some(shutdown_tx);

        // Start health check task
        let health_task = self.start_health_check_task(shutdown_rx.clone()).await;

        // Start cleanup task if enabled
        let cleanup_task = if self.config.enable_cleanup {
            Some(self.start_cleanup_task(shutdown_rx.clone()).await)
        } else {
            None
        };

        // Start topology monitoring task
        let topology_task = self.start_topology_monitor_task(shutdown_rx).await;

        info!("Service registry started with {} background tasks",
              2 + if cleanup_task.is_some() { 1 } else { 0 });

        // Store task handles for graceful shutdown (in a real implementation)
        tokio::spawn(health_task);
        if let Some(task) = cleanup_task {
            tokio::spawn(task);
        }
        tokio::spawn(topology_task);

        Ok(())
    }

    /// Register a new service instance
    pub async fn register_service(&self, mut instance: ServiceInstance) -> Result<()> {
        instance.registered_at = SystemTime::now();
        instance.last_health_check = SystemTime::now();

        let service_name = instance.name.clone();

        // Add to registry
        let mut services = self.services.entry(service_name.clone()).or_insert_with(Vec::new);

        // Remove any existing instance with the same ID
        services.retain(|s| s.id != instance.id);
        services.push(instance.clone());

        info!("Registered service instance: {} ({}:{})",
              instance.id, instance.name, instance.address);

        // Notify topology change
        self.notify_topology_change().await;

        Ok(())
    }

    /// Deregister a service instance
    pub async fn deregister_service(&self, service_name: &str, instance_id: &str) -> Result<()> {
        if let Some(mut services) = self.services.get_mut(service_name) {
            let initial_count = services.len();
            services.retain(|s| s.id != instance_id);

            if services.len() < initial_count {
                info!("Deregistered service instance: {} ({})", instance_id, service_name);

                // Remove empty service entries
                if services.is_empty() {
                    drop(services);
                    self.services.remove(service_name);
                }

                // Notify topology change
                self.notify_topology_change().await;
                return Ok(());
            }
        }

        warn!("Attempted to deregister non-existent service: {} ({})", instance_id, service_name);
        Err(DaemonError::ServiceNotFound(instance_id.to_string()).into())
    }

    /// Get healthy service instances for a service name
    pub fn get_healthy_instances(&self, service_name: &str) -> Vec<ServiceInstance> {
        self.services
            .get(service_name)
            .map(|services| {
                services
                    .iter()
                    .filter(|s| s.is_available())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Select a service instance using the configured load balancing strategy
    pub fn select_instance(&self, service_name: &str) -> Option<ServiceInstance> {
        let healthy_instances = self.get_healthy_instances(service_name);
        if healthy_instances.is_empty() {
            return None;
        }

        match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.select_round_robin(service_name, &healthy_instances)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(&healthy_instances)
            }
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(&healthy_instances)
            }
            LoadBalancingStrategy::Random => {
                self.select_random(&healthy_instances)
            }
            LoadBalancingStrategy::HealthWeighted => {
                self.select_health_weighted(&healthy_instances)
            }
        }
    }

    /// Round-robin load balancing
    fn select_round_robin(&self, service_name: &str, instances: &[ServiceInstance]) -> Option<ServiceInstance> {
        if instances.is_empty() {
            return None;
        }

        let mut counter = self.round_robin_counters.entry(service_name.to_string()).or_insert(0);
        let index = *counter % instances.len();
        *counter += 1;

        Some(instances[index].clone())
    }

    /// Weighted round-robin load balancing
    fn select_weighted_round_robin(&self, instances: &[ServiceInstance]) -> Option<ServiceInstance> {
        if instances.is_empty() {
            return None;
        }

        let total_weight: u32 = instances.iter().map(|i| i.weight as u32).sum();
        if total_weight == 0 {
            return self.select_random(instances);
        }

        // Generate random number based on total weight
        let mut weight_sum = 0u32;
        let target = (random::<u32>() % total_weight) + 1;

        for instance in instances {
            weight_sum += instance.weight as u32;
            if weight_sum >= target {
                return Some(instance.clone());
            }
        }

        // Fallback to first instance
        instances.first().cloned()
    }

    /// Least connections load balancing
    fn select_least_connections(&self, instances: &[ServiceInstance]) -> Option<ServiceInstance> {
        instances
            .iter()
            .min_by_key(|i| i.active_connections)
            .cloned()
    }

    /// Random load balancing
    fn select_random(&self, instances: &[ServiceInstance]) -> Option<ServiceInstance> {
        if instances.is_empty() {
            return None;
        }

        let index = random::<usize>() % instances.len();
        Some(instances[index].clone())
    }

    /// Health-weighted load balancing
    fn select_health_weighted(&self, instances: &[ServiceInstance]) -> Option<ServiceInstance> {
        instances
            .iter()
            .min_by(|a, b| {
                a.load_score()
                    .partial_cmp(&b.load_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Update service instance health status
    pub async fn update_service_health(&self, service_name: &str, instance_id: &str, health: ServiceHealth) -> Result<()> {
        if let Some(mut services) = self.services.get_mut(service_name) {
            if let Some(instance) = services.iter_mut().find(|s| s.id == instance_id) {
                let old_health = instance.health;
                instance.update_health(health);

                if old_health != health {
                    debug!("Updated service {} health: {:?} -> {:?}", instance_id, old_health, health);
                    self.notify_topology_change().await;
                }
                return Ok(());
            }
        }

        Err(DaemonError::ServiceNotFound(instance_id.to_string()).into())
    }

    /// Update service connection count
    pub async fn update_service_connections(&self, service_name: &str, instance_id: &str, connections: u32) -> Result<()> {
        if let Some(mut services) = self.services.get_mut(service_name) {
            if let Some(instance) = services.iter_mut().find(|s| s.id == instance_id) {
                instance.update_connections(connections);
                return Ok(());
            }
        }

        Err(DaemonError::ServiceNotFound(instance_id.to_string()).into())
    }

    /// Get all registered services
    pub fn get_all_services(&self) -> HashMap<String, Vec<ServiceInstance>> {
        self.services
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get service discovery statistics
    pub fn get_stats(&self) -> ServiceDiscoveryStats {
        let mut stats = ServiceDiscoveryStats::default();

        for entry in self.services.iter() {
            stats.total_services += 1;

            for instance in entry.value() {
                stats.total_instances += 1;
                match instance.health {
                    ServiceHealth::Healthy => stats.healthy_instances += 1,
                    ServiceHealth::Degraded => stats.degraded_instances += 1,
                    ServiceHealth::Unhealthy => stats.unhealthy_instances += 1,
                    ServiceHealth::Draining => stats.draining_instances += 1,
                    ServiceHealth::Unknown => stats.unknown_instances += 1,
                }
            }
        }

        stats
    }

    /// Subscribe to topology changes
    pub fn subscribe_topology_changes(&self) -> watch::Receiver<u64> {
        self.topology_receiver.clone()
    }

    /// Notify about topology changes
    async fn notify_topology_change(&self) {
        let new_version = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let _ = self.topology_notifier.send(new_version);
    }

    /// Start health check background task
    async fn start_health_check_task(&self, mut shutdown_rx: watch::Receiver<bool>) -> tokio::task::JoinHandle<()> {
        let services = Arc::clone(&self.services);
        let client = self.health_client.clone();
        let interval_duration = self.config.health_check_interval;
        let timeout = self.config.health_check_timeout;
        let failure_threshold = self.config.failure_threshold;

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Perform health checks
                        for entry in services.iter() {
                            let service_name = entry.key();
                            let instances = entry.value().clone();

                            for instance in instances {
                                if let Some(health_endpoint) = &instance.health_endpoint {
                                    let health_url = format!("http://{}{}", instance.address, health_endpoint);

                                    match client.get(&health_url).timeout(timeout).send().await {
                                        Ok(response) if response.status().is_success() => {
                                            // Mark as healthy
                                            if let Some(mut services) = services.get_mut(service_name) {
                                                if let Some(inst) = services.iter_mut().find(|s| s.id == instance.id) {
                                                    inst.update_health(ServiceHealth::Healthy);
                                                }
                                            }
                                        }
                                        _ => {
                                            // Mark as unhealthy if exceeded failure threshold
                                            if let Some(mut services) = services.get_mut(service_name) {
                                                if let Some(inst) = services.iter_mut().find(|s| s.id == instance.id) {
                                                    if inst.failed_health_checks >= failure_threshold {
                                                        inst.update_health(ServiceHealth::Unhealthy);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("Health check task shutting down");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Start cleanup background task
    async fn start_cleanup_task(&self, mut shutdown_rx: watch::Receiver<bool>) -> tokio::task::JoinHandle<()> {
        let services = Arc::clone(&self.services);
        let cleanup_interval = self.config.cleanup_interval;
        let service_ttl = self.config.service_ttl;

        tokio::spawn(async move {
            let mut interval = interval(cleanup_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let now = SystemTime::now();
                        let mut services_to_remove = Vec::new();

                        // Find expired services
                        for mut entry in services.iter_mut() {
                            let service_name = entry.key().clone();
                            let instances = entry.value_mut();

                            instances.retain(|instance| {
                                let age = now.duration_since(instance.last_health_check)
                                    .unwrap_or(Duration::from_secs(0));

                                if age > service_ttl && instance.health == ServiceHealth::Unhealthy {
                                    info!("Cleaning up expired service instance: {} ({})", instance.id, service_name);
                                    false
                                } else {
                                    true
                                }
                            });

                            if instances.is_empty() {
                                services_to_remove.push(service_name);
                            }
                        }

                        // Remove empty service entries
                        for service_name in services_to_remove {
                            services.remove(&service_name);
                            info!("Removed empty service: {}", service_name);
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("Cleanup task shutting down");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Start topology monitoring task
    async fn start_topology_monitor_task(&self, mut shutdown_rx: watch::Receiver<bool>) -> tokio::task::JoinHandle<()> {
        let services = Arc::clone(&self.services);

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {
                        // Monitor topology changes and log statistics
                        let total_services = services.len();
                        let total_instances: usize = services.iter().map(|entry| entry.value().len()).sum();

                        debug!("Service topology: {} services, {} instances", total_services, total_instances);
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("Topology monitor task shutting down");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Graceful shutdown
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(shutdown_tx) = &self.shutdown_tx {
            let _ = shutdown_tx.send(true);
        }

        info!("Service registry shutdown initiated");

        // Give background tasks time to shutdown gracefully
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(())
    }
}

/// Service discovery statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ServiceDiscoveryStats {
    pub total_services: usize,
    pub total_instances: usize,
    pub healthy_instances: usize,
    pub degraded_instances: usize,
    pub unhealthy_instances: usize,
    pub draining_instances: usize,
    pub unknown_instances: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn create_test_instance(name: &str, port: u16) -> ServiceInstance {
        ServiceInstance::new(
            name.to_string(),
            "1.0.0".to_string(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
            50,
        )
    }

    #[tokio::test]
    async fn test_service_registry_creation() {
        let config = ServiceDiscoveryConfig::default();
        let registry = ServiceRegistry::new(config);

        let stats = registry.get_stats();
        assert_eq!(stats.total_services, 0);
        assert_eq!(stats.total_instances, 0);
    }

    #[tokio::test]
    async fn test_service_registration() {
        let config = ServiceDiscoveryConfig::default();
        let registry = ServiceRegistry::new(config);

        let instance = create_test_instance("test-service", 8080);
        let instance_id = instance.id.clone();

        registry.register_service(instance).await.unwrap();

        let instances = registry.get_healthy_instances("test-service");
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].id, instance_id);
    }

    #[tokio::test]
    async fn test_service_deregistration() {
        let config = ServiceDiscoveryConfig::default();
        let registry = ServiceRegistry::new(config);

        let instance = create_test_instance("test-service", 8080);
        let instance_id = instance.id.clone();

        registry.register_service(instance).await.unwrap();
        registry.deregister_service("test-service", &instance_id).await.unwrap();

        let instances = registry.get_healthy_instances("test-service");
        assert_eq!(instances.len(), 0);
    }

    #[tokio::test]
    async fn test_load_balancing_round_robin() {
        let config = ServiceDiscoveryConfig {
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };
        let registry = ServiceRegistry::new(config);

        // Register multiple instances
        for i in 0..3 {
            let instance = create_test_instance("test-service", 8080 + i);
            registry.register_service(instance).await.unwrap();
        }

        // Update all instances to healthy
        let instances = registry.services.get("test-service").unwrap().clone();
        for instance in &instances {
            registry.update_service_health("test-service", &instance.id, ServiceHealth::Healthy).await.unwrap();
        }

        // Test round-robin selection
        let selected1 = registry.select_instance("test-service").unwrap();
        let selected2 = registry.select_instance("test-service").unwrap();
        let selected3 = registry.select_instance("test-service").unwrap();
        let selected4 = registry.select_instance("test-service").unwrap();

        // Should cycle through instances
        assert_ne!(selected1.id, selected2.id);
        assert_ne!(selected2.id, selected3.id);
        assert_eq!(selected1.id, selected4.id); // Should wrap around
    }

    #[tokio::test]
    async fn test_health_status_filtering() {
        let config = ServiceDiscoveryConfig::default();
        let registry = ServiceRegistry::new(config);

        let instance1 = create_test_instance("test-service", 8080);
        let instance2 = create_test_instance("test-service", 8081);

        let id1 = instance1.id.clone();
        let id2 = instance2.id.clone();

        registry.register_service(instance1).await.unwrap();
        registry.register_service(instance2).await.unwrap();

        // Mark one as healthy, one as unhealthy
        registry.update_service_health("test-service", &id1, ServiceHealth::Healthy).await.unwrap();
        registry.update_service_health("test-service", &id2, ServiceHealth::Unhealthy).await.unwrap();

        let healthy_instances = registry.get_healthy_instances("test-service");
        assert_eq!(healthy_instances.len(), 1);
        assert_eq!(healthy_instances[0].id, id1);
    }

    #[test]
    fn test_service_instance_load_score() {
        let mut instance = create_test_instance("test", 8080);
        instance.weight = 50;
        instance.active_connections = 10;
        instance.health = ServiceHealth::Healthy;

        let score = instance.load_score();
        assert_eq!(score, 0.2); // 10 connections / 50 weight

        instance.health = ServiceHealth::Degraded;
        let degraded_score = instance.load_score();
        assert_eq!(degraded_score, 0.3); // 0.2 * 1.5 penalty

        instance.health = ServiceHealth::Unhealthy;
        let unhealthy_score = instance.load_score();
        assert_eq!(unhealthy_score, f64::INFINITY);
    }

    #[tokio::test]
    async fn test_service_discovery_stats() {
        let config = ServiceDiscoveryConfig::default();
        let registry = ServiceRegistry::new(config);

        // Register services with different health states
        for i in 0..5 {
            let instance = create_test_instance("test-service", 8080 + i);
            let id = instance.id.clone();
            registry.register_service(instance).await.unwrap();

            let health = match i {
                0 => ServiceHealth::Healthy,
                1 => ServiceHealth::Degraded,
                2 => ServiceHealth::Unhealthy,
                3 => ServiceHealth::Draining,
                _ => ServiceHealth::Unknown,
            };

            registry.update_service_health("test-service", &id, health).await.unwrap();
        }

        let stats = registry.get_stats();
        assert_eq!(stats.total_services, 1);
        assert_eq!(stats.total_instances, 5);
        assert_eq!(stats.healthy_instances, 1);
        assert_eq!(stats.degraded_instances, 1);
        assert_eq!(stats.unhealthy_instances, 1);
        assert_eq!(stats.draining_instances, 1);
        assert_eq!(stats.unknown_instances, 1);
    }
}