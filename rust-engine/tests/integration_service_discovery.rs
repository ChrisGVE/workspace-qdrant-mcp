//! Comprehensive integration tests for gRPC service discovery
//!
//! This test suite covers:
//! - Service registration and deregistration
//! - Health check implementation and monitoring
//! - Load balancing algorithms and failover
//! - Service topology management
//! - Discovery protocol compliance
//! - Service mesh integration patterns

use workspace_qdrant_daemon::grpc::service_discovery::{
    ServiceRegistry, ServiceInstance, ServiceHealth, LoadBalancingStrategy, ServiceDiscoveryConfig,
};
use workspace_qdrant_daemon::grpc::{ServiceDiscoveryStats};
use anyhow::Result;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout};
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path};

/// Helper function to create test service instances
fn create_test_service(name: &str, port: u16, weight: u8) -> ServiceInstance {
    let mut instance = ServiceInstance::new(
        name.to_string(),
        "1.0.0".to_string(),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
        weight,
    );
    instance.health_endpoint = Some("/health".to_string());
    instance
}

/// Helper function to create test service discovery config
fn create_test_config() -> ServiceDiscoveryConfig {
    ServiceDiscoveryConfig {
        health_check_interval: Duration::from_millis(100), // Fast for testing
        health_check_timeout: Duration::from_millis(50),
        failure_threshold: 2,
        success_threshold: 1,
        load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
        enable_cleanup: true,
        cleanup_interval: Duration::from_millis(200), // Fast for testing
        service_ttl: Duration::from_secs(1), // Short for testing
    }
}

#[tokio::test]
async fn test_service_registry_initialization() {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    let stats = registry.get_stats();
    assert_eq!(stats.total_services, 0);
    assert_eq!(stats.total_instances, 0);
    assert_eq!(stats.healthy_instances, 0);
    assert_eq!(stats.unhealthy_instances, 0);
}

#[tokio::test]
async fn test_service_registration_and_deregistration() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Test service registration
    let instance = create_test_service("test-service", 8080, 50);
    let instance_id = instance.id.clone();
    registry.register_service(instance).await?;

    // Verify registration
    let instances = registry.get_healthy_instances("test-service");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].id, instance_id);

    // Test service deregistration
    registry.deregister_service("test-service", &instance_id).await?;

    // Verify deregistration
    let instances_after = registry.get_healthy_instances("test-service");
    assert_eq!(instances_after.len(), 0);

    Ok(())
}

#[tokio::test]
async fn test_multiple_service_registration() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Register multiple instances of the same service
    let mut instance_ids = Vec::new();
    for i in 0..3 {
        let instance = create_test_service("test-service", 8080 + i, 50);
        instance_ids.push(instance.id.clone());
        registry.register_service(instance).await?;
    }

    // Register instances of different services
    for i in 0..2 {
        let instance = create_test_service(&format!("other-service-{}", i), 9000 + i, 30);
        registry.register_service(instance).await?;
    }

    // Verify registrations
    let test_instances = registry.get_healthy_instances("test-service");
    assert_eq!(test_instances.len(), 3);

    let other_instances_0 = registry.get_healthy_instances("other-service-0");
    assert_eq!(other_instances_0.len(), 1);

    let other_instances_1 = registry.get_healthy_instances("other-service-1");
    assert_eq!(other_instances_1.len(), 1);

    // Test statistics
    let stats = registry.get_stats();
    assert_eq!(stats.total_services, 3);
    assert_eq!(stats.total_instances, 5);

    Ok(())
}

#[tokio::test]
async fn test_health_status_management() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Register service instances
    let mut instances = Vec::new();
    for i in 0..3 {
        let instance = create_test_service("health-test", 8080 + i, 50);
        instances.push(instance.clone());
        registry.register_service(instance).await?;
    }

    // Set different health statuses
    registry.update_service_health("health-test", &instances[0].id, ServiceHealth::Healthy).await?;
    registry.update_service_health("health-test", &instances[1].id, ServiceHealth::Degraded).await?;
    registry.update_service_health("health-test", &instances[2].id, ServiceHealth::Unhealthy).await?;

    // Test healthy instances filtering
    let healthy_instances = registry.get_healthy_instances("health-test");
    assert_eq!(healthy_instances.len(), 2); // Healthy + Degraded

    // Verify health statuses
    let all_services = registry.get_all_services();
    let health_test_instances = all_services.get("health-test").unwrap();

    for instance in health_test_instances {
        match instance.id.as_str() {
            id if id == instances[0].id => assert_eq!(instance.health, ServiceHealth::Healthy),
            id if id == instances[1].id => assert_eq!(instance.health, ServiceHealth::Degraded),
            id if id == instances[2].id => assert_eq!(instance.health, ServiceHealth::Unhealthy),
            _ => panic!("Unexpected instance ID"),
        }
    }

    // Test statistics with health breakdown
    let stats = registry.get_stats();
    assert_eq!(stats.healthy_instances, 1);
    assert_eq!(stats.degraded_instances, 1);
    assert_eq!(stats.unhealthy_instances, 1);

    Ok(())
}

#[tokio::test]
async fn test_round_robin_load_balancing() -> Result<()> {
    let config = ServiceDiscoveryConfig {
        load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
        ..create_test_config()
    };
    let registry = ServiceRegistry::new(config);

    // Register multiple instances
    let mut instance_ids = Vec::new();
    for i in 0..3 {
        let instance = create_test_service("lb-test", 8080 + i, 50);
        instance_ids.push(instance.id.clone());
        registry.register_service(instance).await?;
    }

    // Mark all as healthy
    for id in &instance_ids {
        registry.update_service_health("lb-test", id, ServiceHealth::Healthy).await?;
    }

    // Test round-robin selection
    let mut selected_ids = Vec::new();
    for _ in 0..6 {
        if let Some(instance) = registry.select_instance("lb-test") {
            selected_ids.push(instance.id);
        }
    }

    // Should cycle through all instances
    assert_eq!(selected_ids.len(), 6);

    // Verify round-robin pattern (each instance should be selected twice)
    let mut selection_counts = HashMap::new();
    for id in selected_ids {
        *selection_counts.entry(id).or_insert(0) += 1;
    }

    assert_eq!(selection_counts.len(), 3);
    for count in selection_counts.values() {
        assert_eq!(*count, 2);
    }

    Ok(())
}

#[tokio::test]
async fn test_weighted_round_robin_load_balancing() -> Result<()> {
    let config = ServiceDiscoveryConfig {
        load_balancing_strategy: LoadBalancingStrategy::WeightedRoundRobin,
        ..create_test_config()
    };
    let registry = ServiceRegistry::new(config);

    // Register instances with different weights
    let instance1 = {
        let mut instance = create_test_service("weighted-test", 8080, 20); // Low weight
        let id = instance.id.clone();
        registry.register_service(instance).await?;
        id
    };

    let instance2 = {
        let mut instance = create_test_service("weighted-test", 8081, 80); // High weight
        let id = instance.id.clone();
        registry.register_service(instance).await?;
        id
    };

    // Mark both as healthy
    registry.update_service_health("weighted-test", &instance1, ServiceHealth::Healthy).await?;
    registry.update_service_health("weighted-test", &instance2, ServiceHealth::Healthy).await?;

    // Test weighted selection over many iterations
    let mut selection_counts = HashMap::new();
    for _ in 0..100 {
        if let Some(instance) = registry.select_instance("weighted-test") {
            *selection_counts.entry(instance.id).or_insert(0) += 1;
        }
    }

    // High weight instance should be selected more often
    let count1 = selection_counts.get(&instance1).unwrap_or(&0);
    let count2 = selection_counts.get(&instance2).unwrap_or(&0);

    // Instance2 (weight 80) should be selected more than instance1 (weight 20)
    assert!(count2 > count1);

    // Rough proportion check (not exact due to randomness)
    let total_selections = count1 + count2;
    let instance2_percentage = (*count2 as f64 / total_selections as f64) * 100.0;
    assert!(instance2_percentage > 60.0); // Should be around 80%

    Ok(())
}

#[tokio::test]
async fn test_least_connections_load_balancing() -> Result<()> {
    let config = ServiceDiscoveryConfig {
        load_balancing_strategy: LoadBalancingStrategy::LeastConnections,
        ..create_test_config()
    };
    let registry = ServiceRegistry::new(config);

    // Register instances
    let mut instance_ids = Vec::new();
    for i in 0..3 {
        let instance = create_test_service("lc-test", 8080 + i, 50);
        instance_ids.push(instance.id.clone());
        registry.register_service(instance).await?;
    }

    // Mark all as healthy
    for id in &instance_ids {
        registry.update_service_health("lc-test", id, ServiceHealth::Healthy).await?;
    }

    // Set different connection counts
    registry.update_service_connections("lc-test", &instance_ids[0], 10).await?;
    registry.update_service_connections("lc-test", &instance_ids[1], 5).await?;
    registry.update_service_connections("lc-test", &instance_ids[2], 15).await?;

    // Select instance - should pick the one with least connections (instance_ids[1])
    let selected = registry.select_instance("lc-test").unwrap();
    assert_eq!(selected.id, instance_ids[1]);
    assert_eq!(selected.active_connections, 5);

    Ok(())
}

#[tokio::test]
async fn test_health_weighted_load_balancing() -> Result<()> {
    let config = ServiceDiscoveryConfig {
        load_balancing_strategy: LoadBalancingStrategy::HealthWeighted,
        ..create_test_config()
    };
    let registry = ServiceRegistry::new(config);

    // Register instances
    let mut instance_ids = Vec::new();
    for i in 0..3 {
        let instance = create_test_service("hw-test", 8080 + i, 50);
        instance_ids.push(instance.id.clone());
        registry.register_service(instance).await?;
    }

    // Set different health and connection states
    registry.update_service_health("hw-test", &instance_ids[0], ServiceHealth::Healthy).await?;
    registry.update_service_connections("hw-test", &instance_ids[0], 10).await?;

    registry.update_service_health("hw-test", &instance_ids[1], ServiceHealth::Degraded).await?;
    registry.update_service_connections("hw-test", &instance_ids[1], 5).await?; // Lower connections but degraded

    registry.update_service_health("hw-test", &instance_ids[2], ServiceHealth::Healthy).await?;
    registry.update_service_connections("hw-test", &instance_ids[2], 3).await?; // Lowest connections and healthy

    // Select instance - should pick the healthy one with lowest load score
    let selected = registry.select_instance("hw-test").unwrap();
    assert_eq!(selected.id, instance_ids[2]); // Best load score
    assert_eq!(selected.active_connections, 3);
    assert_eq!(selected.health, ServiceHealth::Healthy);

    Ok(())
}

#[tokio::test]
async fn test_service_discovery_with_health_checks() -> Result<()> {
    // Create mock HTTP server for health checks
    let mock_server = MockServer::start().await;
    let health_url = format!("http://{}/health", mock_server.address());

    // Set up health check endpoints
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_string("OK"))
        .mount(&mock_server)
        .await;

    let config = create_test_config();
    let mut registry = ServiceRegistry::new(config);

    // Register service with health endpoint
    let mut instance = create_test_service("health-check-test", mock_server.address().port(), 50);
    instance.health_endpoint = Some("/health".to_string());
    let instance_id = instance.id.clone();

    registry.register_service(instance).await?;

    // Start the registry (enables health checking)
    registry.start().await?;

    // Wait for health check to complete
    sleep(Duration::from_millis(300)).await;

    // Verify health status was updated to healthy
    let instances = registry.get_healthy_instances("health-check-test");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].health, ServiceHealth::Healthy);

    // Stop mock server to simulate service failure
    mock_server.reset().await;

    // Wait for health check failure detection
    sleep(Duration::from_millis(500)).await;

    // Service should now be marked as unhealthy
    let unhealthy_instances = registry.get_healthy_instances("health-check-test");
    assert_eq!(unhealthy_instances.len(), 0); // No healthy instances

    registry.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_automatic_service_cleanup() -> Result<()> {
    let config = ServiceDiscoveryConfig {
        enable_cleanup: true,
        cleanup_interval: Duration::from_millis(100),
        service_ttl: Duration::from_millis(200),
        ..create_test_config()
    };
    let mut registry = ServiceRegistry::new(config);

    // Register a service
    let instance = create_test_service("cleanup-test", 8080, 50);
    let instance_id = instance.id.clone();
    registry.register_service(instance).await?;

    // Mark as unhealthy
    registry.update_service_health("cleanup-test", &instance_id, ServiceHealth::Unhealthy).await?;

    // Start registry with cleanup enabled
    registry.start().await?;

    // Wait for cleanup to occur
    sleep(Duration::from_millis(400)).await;

    // Service should be cleaned up
    let instances = registry.get_all_services();
    assert!(instances.is_empty() || !instances.contains_key("cleanup-test"));

    registry.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_topology_change_notifications() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Subscribe to topology changes
    let mut topology_rx = registry.subscribe_topology_changes();
    let initial_version = *topology_rx.borrow();

    // Register a service (should trigger topology change)
    let instance = create_test_service("topology-test", 8080, 50);
    registry.register_service(instance.clone()).await?;

    // Wait for topology change notification
    let change_detected = timeout(Duration::from_millis(100), topology_rx.changed()).await;
    assert!(change_detected.is_ok());

    let new_version = *topology_rx.borrow();
    assert!(new_version > initial_version);

    // Another topology change when deregistering
    let before_deregister = *topology_rx.borrow();
    registry.deregister_service("topology-test", &instance.id).await?;

    let change_detected = timeout(Duration::from_millis(100), topology_rx.changed()).await;
    assert!(change_detected.is_ok());

    let after_deregister = *topology_rx.borrow();
    assert!(after_deregister > before_deregister);

    Ok(())
}

#[tokio::test]
async fn test_service_failover_scenarios() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Register multiple instances for failover testing
    let mut primary_instance = create_test_service("failover-test", 8080, 100);
    let mut backup_instance = create_test_service("failover-test", 8081, 50);

    let primary_id = primary_instance.id.clone();
    let backup_id = backup_instance.id.clone();

    registry.register_service(primary_instance).await?;
    registry.register_service(backup_instance).await?;

    // Initially both healthy
    registry.update_service_health("failover-test", &primary_id, ServiceHealth::Healthy).await?;
    registry.update_service_health("failover-test", &backup_id, ServiceHealth::Healthy).await?;

    // Primary should be selected due to higher weight (in round-robin, first healthy)
    let selected = registry.select_instance("failover-test").unwrap();
    // Note: Round-robin doesn't consider weight, so we just verify we get a healthy instance
    assert!(matches!(selected.health, ServiceHealth::Healthy));

    // Simulate primary failure
    registry.update_service_health("failover-test", &primary_id, ServiceHealth::Unhealthy).await?;

    // Now only backup should be available
    let healthy_instances = registry.get_healthy_instances("failover-test");
    assert_eq!(healthy_instances.len(), 1);
    assert_eq!(healthy_instances[0].id, backup_id);

    // Simulate backup recovery
    registry.update_service_health("failover-test", &primary_id, ServiceHealth::Healthy).await?;

    // Both should be available again
    let healthy_instances = registry.get_healthy_instances("failover-test");
    assert_eq!(healthy_instances.len(), 2);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_service_operations() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);
    let registry = Arc::new(registry);

    let mut handles = Vec::new();

    // Concurrent service registrations
    for i in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            let instance = create_test_service(&format!("concurrent-test-{}", i), 8080 + i, 50);
            registry_clone.register_service(instance).await
        });
        handles.push(handle);
    }

    // Concurrent health updates
    for i in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            // Wait a bit for registration to complete
            sleep(Duration::from_millis(10)).await;

            let instances = registry_clone.get_all_services();
            if let Some(service_instances) = instances.get(&format!("concurrent-test-{}", i)) {
                if let Some(instance) = service_instances.first() {
                    let _ = registry_clone.update_service_health(
                        &format!("concurrent-test-{}", i),
                        &instance.id,
                        ServiceHealth::Healthy
                    ).await;
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        let _ = handle.await;
    }

    // Verify all services are registered
    let stats = registry.get_stats();
    assert_eq!(stats.total_services, 10);
    assert_eq!(stats.total_instances, 10);

    Ok(())
}

#[tokio::test]
async fn test_service_mesh_integration_patterns() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Register services with metadata tags (simulating service mesh)
    let mut web_service = create_test_service("web-service", 8080, 75);
    web_service.tags = vec!["frontend".to_string(), "web".to_string()];
    web_service.metadata.insert("zone".to_string(), "us-east-1".to_string());
    web_service.metadata.insert("cluster".to_string(), "prod".to_string());

    let mut api_service = create_test_service("api-service", 8081, 100);
    api_service.tags = vec!["backend".to_string(), "api".to_string()];
    api_service.metadata.insert("zone".to_string(), "us-east-1".to_string());
    api_service.metadata.insert("cluster".to_string(), "prod".to_string());

    let mut db_service = create_test_service("db-service", 5432, 50);
    db_service.tags = vec!["database".to_string(), "postgres".to_string()];
    db_service.metadata.insert("zone".to_string(), "us-east-2".to_string());
    db_service.metadata.insert("cluster".to_string(), "prod".to_string());

    registry.register_service(web_service).await?;
    registry.register_service(api_service).await?;
    registry.register_service(db_service).await?;

    // Mark all as healthy
    let all_services = registry.get_all_services();
    for (service_name, instances) in all_services {
        for instance in instances {
            registry.update_service_health(&service_name, &instance.id, ServiceHealth::Healthy).await?;
        }
    }

    // Test service discovery by tags (simulated filtering)
    let all_services = registry.get_all_services();
    let frontend_services: Vec<_> = all_services.iter()
        .flat_map(|(_, instances)| instances)
        .filter(|instance| instance.tags.contains(&"frontend".to_string()))
        .collect();
    assert_eq!(frontend_services.len(), 1);
    assert_eq!(frontend_services[0].name, "web-service");

    let backend_services: Vec<_> = all_services.iter()
        .flat_map(|(_, instances)| instances)
        .filter(|instance| instance.tags.contains(&"backend".to_string()))
        .collect();
    assert_eq!(backend_services.len(), 1);
    assert_eq!(backend_services[0].name, "api-service");

    // Test metadata-based filtering (zone-aware routing)
    let us_east_1_services: Vec<_> = all_services.iter()
        .flat_map(|(_, instances)| instances)
        .filter(|instance| instance.metadata.get("zone") == Some(&"us-east-1".to_string()))
        .collect();
    assert_eq!(us_east_1_services.len(), 2);

    let us_east_2_services: Vec<_> = all_services.iter()
        .flat_map(|(_, instances)| instances)
        .filter(|instance| instance.metadata.get("zone") == Some(&"us-east-2".to_string()))
        .collect();
    assert_eq!(us_east_2_services.len(), 1);

    Ok(())
}

#[tokio::test]
async fn test_service_discovery_stats_comprehensive() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Create services with various health states
    let health_states = [
        ServiceHealth::Healthy,
        ServiceHealth::Degraded,
        ServiceHealth::Unhealthy,
        ServiceHealth::Draining,
        ServiceHealth::Unknown,
    ];

    for (i, &health) in health_states.iter().enumerate() {
        let instance = create_test_service(&format!("stats-test-{}", i), 8080 + i as u16, 50);
        let instance_id = instance.id.clone();
        registry.register_service(instance).await?;
        registry.update_service_health(&format!("stats-test-{}", i), &instance_id, health).await?;
    }

    // Verify comprehensive statistics
    let stats = registry.get_stats();
    assert_eq!(stats.total_services, 5);
    assert_eq!(stats.total_instances, 5);
    assert_eq!(stats.healthy_instances, 1);
    assert_eq!(stats.degraded_instances, 1);
    assert_eq!(stats.unhealthy_instances, 1);
    assert_eq!(stats.draining_instances, 1);
    assert_eq!(stats.unknown_instances, 1);

    // Verify the sum adds up
    let total_by_health = stats.healthy_instances + stats.degraded_instances +
                         stats.unhealthy_instances + stats.draining_instances + stats.unknown_instances;
    assert_eq!(total_by_health, stats.total_instances);

    Ok(())
}

#[tokio::test]
async fn test_service_discovery_protocol_compliance() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Test standard service discovery operations

    // 1. Service Registration with full metadata
    let mut instance = create_test_service("protocol-test", 8080, 75);
    instance.version = "2.1.0".to_string();
    instance.metadata.insert("protocol".to_string(), "grpc".to_string());
    instance.metadata.insert("encoding".to_string(), "protobuf".to_string());
    instance.tags = vec!["production".to_string(), "v2".to_string()];

    let instance_id = instance.id.clone();
    registry.register_service(instance).await?;

    // 2. Health Check Protocol
    registry.update_service_health("protocol-test", &instance_id, ServiceHealth::Healthy).await?;

    // 3. Service Discovery Query
    let healthy_instances = registry.get_healthy_instances("protocol-test");
    assert_eq!(healthy_instances.len(), 1);

    let discovered_instance = &healthy_instances[0];
    assert_eq!(discovered_instance.name, "protocol-test");
    assert_eq!(discovered_instance.version, "2.1.0");
    assert_eq!(discovered_instance.health, ServiceHealth::Healthy);
    assert!(discovered_instance.metadata.contains_key("protocol"));
    assert!(discovered_instance.tags.contains(&"production".to_string()));

    // 4. Load Balancing Selection
    let selected = registry.select_instance("protocol-test");
    assert!(selected.is_some());
    assert_eq!(selected.unwrap().id, instance_id);

    // 5. Service Deregistration
    registry.deregister_service("protocol-test", &instance_id).await?;
    let empty_instances = registry.get_healthy_instances("protocol-test");
    assert_eq!(empty_instances.len(), 0);

    Ok(())
}

#[tokio::test]
async fn test_dynamic_service_topology_changes() -> Result<()> {
    let config = create_test_config();
    let registry = ServiceRegistry::new(config);

    // Start with initial topology
    let instance1 = create_test_service("dynamic-test", 8080, 50);
    let id1 = instance1.id.clone();
    registry.register_service(instance1).await?;

    // Verify initial state
    let stats = registry.get_stats();
    assert_eq!(stats.total_instances, 1);

    // Dynamic scaling - add more instances
    let mut new_instances = Vec::new();
    for i in 1..4 {
        let instance = create_test_service("dynamic-test", 8080 + i, 50);
        new_instances.push(instance.id.clone());
        registry.register_service(instance).await?;
    }

    // Verify scaled topology
    let stats = registry.get_stats();
    assert_eq!(stats.total_instances, 4);

    // Mark all as healthy
    registry.update_service_health("dynamic-test", &id1, ServiceHealth::Healthy).await?;
    for id in &new_instances {
        registry.update_service_health("dynamic-test", id, ServiceHealth::Healthy).await?;
    }

    // Test load distribution across all instances
    let healthy_instances = registry.get_healthy_instances("dynamic-test");
    assert_eq!(healthy_instances.len(), 4);

    // Simulate rolling update - gradually mark instances as draining
    registry.update_service_health("dynamic-test", &id1, ServiceHealth::Draining).await?;
    registry.update_service_health("dynamic-test", &new_instances[0], ServiceHealth::Draining).await?;

    // Only 2 should be available for new requests
    let available_instances = registry.get_healthy_instances("dynamic-test");
    assert_eq!(available_instances.len(), 2);

    // Complete rolling update - remove old instances and add new ones
    registry.deregister_service("dynamic-test", &id1).await?;
    registry.deregister_service("dynamic-test", &new_instances[0]).await?;

    // Add new version instances
    for i in 0..2 {
        let mut instance = create_test_service("dynamic-test", 9000 + i, 50);
        instance.version = "2.0.0".to_string();
        let id = instance.id.clone();
        registry.register_service(instance).await?;
        registry.update_service_health("dynamic-test", &id, ServiceHealth::Healthy).await?;
    }

    // Verify final topology
    let final_instances = registry.get_healthy_instances("dynamic-test");
    assert_eq!(final_instances.len(), 4); // 2 remaining + 2 new

    // Verify version mix
    let all_services = registry.get_all_services();
    let dynamic_instances = all_services.get("dynamic-test").unwrap();
    let v1_count = dynamic_instances.iter().filter(|i| i.version == "1.0.0").count();
    let v2_count = dynamic_instances.iter().filter(|i| i.version == "2.0.0").count();
    assert_eq!(v1_count, 2);
    assert_eq!(v2_count, 2);

    Ok(())
}