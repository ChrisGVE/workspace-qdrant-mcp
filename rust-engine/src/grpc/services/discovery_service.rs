//! Service discovery gRPC service implementation for daemon service registration and discovery

use crate::daemon::WorkspaceDaemon;
use crate::grpc::service_discovery::{ServiceRegistry, ServiceInstance, ServiceHealth, LoadBalancingStrategy, ServiceDiscoveryConfig};
use crate::proto::{
    service_discovery_server::ServiceDiscovery,
    RegisterServiceRequest, RegisterServiceResponse,
    DeregisterServiceRequest, DeregisterServiceResponse,
    ListServicesRequest, ListServicesResponse,
    GetServiceInstancesRequest, GetServiceInstancesResponse,
    UpdateServiceHealthRequest, UpdateServiceHealthResponse,
    ServiceDiscoveryStatsRequest, ServiceDiscoveryStatsResponse,
    ServiceInstanceProto, ServiceHealthProto,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

/// Service discovery service implementation with gRPC interface
#[derive(Debug)]
pub struct ServiceDiscoveryImpl {
    daemon: Arc<WorkspaceDaemon>,
    registry: Arc<RwLock<ServiceRegistry>>,
}

impl ServiceDiscoveryImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        // Create service discovery configuration from daemon config
        let config = ServiceDiscoveryConfig {
            health_check_interval: std::time::Duration::from_secs(30),
            health_check_timeout: std::time::Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            enable_cleanup: true,
            cleanup_interval: std::time::Duration::from_secs(60),
            service_ttl: std::time::Duration::from_secs(300),
        };

        let registry = ServiceRegistry::new(config);

        Self {
            daemon,
            registry: Arc::new(RwLock::new(registry)),
        }
    }

    /// Start the service discovery registry
    pub async fn start(&self) -> Result<(), tonic::Status> {
        let mut registry = self.registry.write().await;
        registry.start().await.map_err(|e| {
            error!("Failed to start service discovery registry: {}", e);
            Status::internal("Failed to start service discovery")
        })?;

        info!("Service discovery registry started");
        Ok(())
    }

    /// Stop the service discovery registry
    pub async fn stop(&self) -> Result<(), tonic::Status> {
        let mut registry = self.registry.write().await;
        registry.shutdown().await.map_err(|e| {
            error!("Failed to shutdown service discovery registry: {}", e);
            Status::internal("Failed to shutdown service discovery")
        })?;

        info!("Service discovery registry stopped");
        Ok(())
    }

    /// Convert protobuf ServiceInstance to internal ServiceInstance
    fn proto_to_service_instance(&self, proto: ServiceInstanceProto) -> Result<ServiceInstance, Status> {
        let address: SocketAddr = proto.address.parse().map_err(|_| {
            Status::invalid_argument("Invalid socket address format")
        })?;

        let health = match ServiceHealthProto::try_from(proto.health).unwrap_or(ServiceHealthProto::ServiceHealthUnknown) {
            ServiceHealthProto::ServiceHealthHealthy => ServiceHealth::Healthy,
            ServiceHealthProto::ServiceHealthDegraded => ServiceHealth::Degraded,
            ServiceHealthProto::ServiceHealthUnhealthy => ServiceHealth::Unhealthy,
            ServiceHealthProto::ServiceHealthDraining => ServiceHealth::Draining,
            ServiceHealthProto::ServiceHealthUnknown => ServiceHealth::Unknown,
        };

        let mut instance = ServiceInstance::new(
            proto.name,
            proto.version,
            address,
            proto.weight.clamp(1, 100) as u8,
        );

        instance.health = health;
        instance.metadata = proto.metadata;
        instance.tags = proto.tags;
        instance.health_endpoint = if proto.health_endpoint.is_empty() {
            None
        } else {
            Some(proto.health_endpoint)
        };
        instance.health_interval = proto.health_interval;

        Ok(instance)
    }

    /// Convert internal ServiceInstance to protobuf ServiceInstance
    fn service_instance_to_proto(&self, instance: &ServiceInstance) -> ServiceInstanceProto {
        let health = match instance.health {
            ServiceHealth::Healthy => ServiceHealthProto::ServiceHealthHealthy,
            ServiceHealth::Degraded => ServiceHealthProto::ServiceHealthDegraded,
            ServiceHealth::Unhealthy => ServiceHealthProto::ServiceHealthUnhealthy,
            ServiceHealth::Draining => ServiceHealthProto::ServiceHealthDraining,
            ServiceHealth::Unknown => ServiceHealthProto::ServiceHealthUnknown,
        } as i32;

        ServiceInstanceProto {
            id: instance.id.clone(),
            name: instance.name.clone(),
            version: instance.version.clone(),
            address: instance.address.to_string(),
            weight: instance.weight as u32,
            health,
            metadata: instance.metadata.clone(),
            tags: instance.tags.clone(),
            health_endpoint: instance.health_endpoint.clone().unwrap_or_default(),
            health_interval: instance.health_interval,
            active_connections: instance.active_connections,
            total_requests: instance.total_requests,
            registered_at: Some(prost_types::Timestamp {
                seconds: instance.registered_at
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                nanos: 0,
            }),
            last_health_check: Some(prost_types::Timestamp {
                seconds: instance.last_health_check
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                nanos: 0,
            }),
        }
    }
}

#[tonic::async_trait]
impl ServiceDiscovery for ServiceDiscoveryImpl {
    async fn register_service(
        &self,
        request: Request<RegisterServiceRequest>,
    ) -> Result<Response<RegisterServiceResponse>, Status> {
        let req = request.into_inner();
        debug!("Service registration requested: {:?}", req.instance);

        let instance = match req.instance {
            Some(proto_instance) => self.proto_to_service_instance(proto_instance)?,
            None => return Err(Status::invalid_argument("Service instance is required")),
        };

        let instance_id = instance.id.clone();
        let service_name = instance.name.clone();

        let registry = self.registry.read().await;
        registry.register_service(instance).await.map_err(|e| {
            error!("Failed to register service {}: {}", service_name, e);
            Status::internal("Failed to register service")
        })?;

        info!("Registered service instance: {} ({})", instance_id, service_name);

        let response = RegisterServiceResponse {
            success: true,
            instance_id,
            error_message: String::new(),
        };

        Ok(Response::new(response))
    }

    async fn deregister_service(
        &self,
        request: Request<DeregisterServiceRequest>,
    ) -> Result<Response<DeregisterServiceResponse>, Status> {
        let req = request.into_inner();
        debug!("Service deregistration requested: {} ({})", req.instance_id, req.service_name);

        let registry = self.registry.read().await;
        registry.deregister_service(&req.service_name, &req.instance_id).await.map_err(|e| {
            warn!("Failed to deregister service {} ({}): {}", req.instance_id, req.service_name, e);
            Status::not_found("Service instance not found")
        })?;

        info!("Deregistered service instance: {} ({})", req.instance_id, req.service_name);

        let response = DeregisterServiceResponse {
            success: true,
            error_message: String::new(),
        };

        Ok(Response::new(response))
    }

    async fn list_services(
        &self,
        _request: Request<ListServicesRequest>,
    ) -> Result<Response<ListServicesResponse>, Status> {
        debug!("Service list requested");

        let registry = self.registry.read().await;
        let all_services = registry.get_all_services();

        let services: Vec<String> = all_services.keys().cloned().collect();

        let response = ListServicesResponse {
            service_names: services,
        };

        Ok(Response::new(response))
    }

    async fn get_service_instances(
        &self,
        request: Request<GetServiceInstancesRequest>,
    ) -> Result<Response<GetServiceInstancesResponse>, Status> {
        let req = request.into_inner();
        debug!("Service instances requested for: {}", req.service_name);

        let registry = self.registry.read().await;

        let instances = if req.healthy_only {
            registry.get_healthy_instances(&req.service_name)
        } else {
            registry.get_all_services()
                .get(&req.service_name)
                .cloned()
                .unwrap_or_default()
        };

        let proto_instances: Vec<ServiceInstanceProto> = instances
            .iter()
            .map(|instance| self.service_instance_to_proto(instance))
            .collect();

        let response = GetServiceInstancesResponse {
            instances: proto_instances,
        };

        Ok(Response::new(response))
    }

    async fn update_service_health(
        &self,
        request: Request<UpdateServiceHealthRequest>,
    ) -> Result<Response<UpdateServiceHealthResponse>, Status> {
        let req = request.into_inner();
        debug!("Health update requested: {} ({}) -> {:?}", req.instance_id, req.service_name, req.health);

        let health = match ServiceHealthProto::try_from(req.health).unwrap_or(ServiceHealthProto::ServiceHealthUnknown) {
            ServiceHealthProto::ServiceHealthHealthy => ServiceHealth::Healthy,
            ServiceHealthProto::ServiceHealthDegraded => ServiceHealth::Degraded,
            ServiceHealthProto::ServiceHealthUnhealthy => ServiceHealth::Unhealthy,
            ServiceHealthProto::ServiceHealthDraining => ServiceHealth::Draining,
            ServiceHealthProto::ServiceHealthUnknown => ServiceHealth::Unknown,
        };

        let registry = self.registry.read().await;
        registry.update_service_health(&req.service_name, &req.instance_id, health).await.map_err(|e| {
            warn!("Failed to update service health {} ({}): {}", req.instance_id, req.service_name, e);
            Status::not_found("Service instance not found")
        })?;

        debug!("Updated service {} health to {:?}", req.instance_id, health);

        let response = UpdateServiceHealthResponse {
            success: true,
            error_message: String::new(),
        };

        Ok(Response::new(response))
    }

    async fn get_discovery_stats(
        &self,
        _request: Request<ServiceDiscoveryStatsRequest>,
    ) -> Result<Response<ServiceDiscoveryStatsResponse>, Status> {
        debug!("Service discovery stats requested");

        let registry = self.registry.read().await;
        let stats = registry.get_stats();

        let response = ServiceDiscoveryStatsResponse {
            total_services: stats.total_services as u32,
            total_instances: stats.total_instances as u32,
            healthy_instances: stats.healthy_instances as u32,
            degraded_instances: stats.degraded_instances as u32,
            unhealthy_instances: stats.unhealthy_instances as u32,
            draining_instances: stats.draining_instances as u32,
            unknown_instances: stats.unknown_instances as u32,
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use std::net::{IpAddr, Ipv4Addr};
    use tempfile::TempDir;
    use tonic::Request;
    use tokio_test;

    fn create_test_daemon_config() -> DaemonConfig {
        let db_path = ":memory:";

        DaemonConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
                security: crate::config::SecurityConfig::default(),
                transport: crate::config::TransportConfig::default(),
                message: crate::config::MessageConfig::default(),
                compression: crate::config::CompressionConfig::default(),
                streaming: crate::config::StreamingConfig::default(),
            },
            database: DatabaseConfig {
                sqlite_path: db_path.to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_retries: 3,
                default_collection: CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 4,
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: 1024 * 1024,
                supported_extensions: vec!["txt".to_string(), "md".to_string()],
                enable_lsp: false,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: false,
                debounce_ms: 500,
                max_watched_dirs: 10,
                ignore_patterns: vec![],
                recursive: true,
            },
            metrics: MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: false,
                prometheus_port: 9090,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_path: None,
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
        }
    }

    async fn create_test_daemon() -> Arc<WorkspaceDaemon> {
        let config = create_test_daemon_config();
        Arc::new(WorkspaceDaemon::new(config).await.expect("Failed to create daemon"))
    }

    fn create_test_proto_instance() -> ServiceInstanceProto {
        ServiceInstanceProto {
            id: uuid::Uuid::new_v4().to_string(),
            name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            address: "127.0.0.1:8080".to_string(),
            weight: 50,
            health: ServiceHealthProto::ServiceHealthHealthy as i32,
            metadata: std::collections::HashMap::new(),
            tags: vec!["test".to_string()],
            health_endpoint: "/health".to_string(),
            health_interval: 30,
            active_connections: 0,
            total_requests: 0,
            registered_at: Some(prost_types::Timestamp {
                seconds: 1234567890,
                nanos: 0,
            }),
            last_health_check: Some(prost_types::Timestamp {
                seconds: 1234567890,
                nanos: 0,
            }),
        }
    }

    #[tokio::test]
    async fn test_service_discovery_impl_creation() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // Service should be created successfully
        let registry = service.registry.read().await;
        let stats = registry.get_stats();
        assert_eq!(stats.total_services, 0);
    }

    #[tokio::test]
    async fn test_register_service() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        let proto_instance = create_test_proto_instance();
        let request = Request::new(RegisterServiceRequest {
            instance: Some(proto_instance.clone()),
        });

        let result = service.register_service(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.success);
        assert!(!response.instance_id.is_empty());
        assert!(response.error_message.is_empty());

        // Verify service was registered
        let registry = service.registry.read().await;
        let instances = registry.get_healthy_instances("test-service");
        assert_eq!(instances.len(), 1);
    }

    #[tokio::test]
    async fn test_register_service_missing_instance() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        let request = Request::new(RegisterServiceRequest {
            instance: None,
        });

        let result = service.register_service(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_deregister_service() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // First register a service
        let proto_instance = create_test_proto_instance();
        let instance_id = proto_instance.id.clone();
        let register_request = Request::new(RegisterServiceRequest {
            instance: Some(proto_instance),
        });

        service.register_service(register_request).await.unwrap();

        // Then deregister it
        let deregister_request = Request::new(DeregisterServiceRequest {
            service_name: "test-service".to_string(),
            instance_id: instance_id.clone(),
        });

        let result = service.deregister_service(deregister_request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.success);
        assert!(response.error_message.is_empty());

        // Verify service was deregistered
        let registry = service.registry.read().await;
        let instances = registry.get_healthy_instances("test-service");
        assert_eq!(instances.len(), 0);
    }

    #[tokio::test]
    async fn test_deregister_nonexistent_service() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        let request = Request::new(DeregisterServiceRequest {
            service_name: "nonexistent-service".to_string(),
            instance_id: "nonexistent-id".to_string(),
        });

        let result = service.deregister_service(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_list_services() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // Register multiple services
        for i in 0..3 {
            let mut proto_instance = create_test_proto_instance();
            proto_instance.name = format!("test-service-{}", i);
            proto_instance.id = uuid::Uuid::new_v4().to_string();

            let request = Request::new(RegisterServiceRequest {
                instance: Some(proto_instance),
            });

            service.register_service(request).await.unwrap();
        }

        // List services
        let request = Request::new(ListServicesRequest {});
        let result = service.list_services(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.service_names.len(), 3);
        assert!(response.service_names.contains(&"test-service-0".to_string()));
        assert!(response.service_names.contains(&"test-service-1".to_string()));
        assert!(response.service_names.contains(&"test-service-2".to_string()));
    }

    #[tokio::test]
    async fn test_get_service_instances() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // Register service
        let proto_instance = create_test_proto_instance();
        let register_request = Request::new(RegisterServiceRequest {
            instance: Some(proto_instance.clone()),
        });

        service.register_service(register_request).await.unwrap();

        // Get instances
        let request = Request::new(GetServiceInstancesRequest {
            service_name: "test-service".to_string(),
            healthy_only: false,
        });

        let result = service.get_service_instances(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.instances.len(), 1);
        assert_eq!(response.instances[0].name, "test-service");
    }

    #[tokio::test]
    async fn test_get_service_instances_healthy_only() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // Register service
        let proto_instance = create_test_proto_instance();
        let instance_id = proto_instance.id.clone();
        let register_request = Request::new(RegisterServiceRequest {
            instance: Some(proto_instance),
        });

        service.register_service(register_request).await.unwrap();

        // Mark as unhealthy
        let health_request = Request::new(UpdateServiceHealthRequest {
            service_name: "test-service".to_string(),
            instance_id: instance_id.clone(),
            health: ServiceHealthProto::ServiceHealthUnhealthy as i32,
        });

        service.update_service_health(health_request).await.unwrap();

        // Get healthy instances only
        let request = Request::new(GetServiceInstancesRequest {
            service_name: "test-service".to_string(),
            healthy_only: true,
        });

        let result = service.get_service_instances(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.instances.len(), 0); // No healthy instances
    }

    #[tokio::test]
    async fn test_update_service_health() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // Register service
        let proto_instance = create_test_proto_instance();
        let instance_id = proto_instance.id.clone();
        let register_request = Request::new(RegisterServiceRequest {
            instance: Some(proto_instance),
        });

        service.register_service(register_request).await.unwrap();

        // Update health
        let request = Request::new(UpdateServiceHealthRequest {
            service_name: "test-service".to_string(),
            instance_id: instance_id.clone(),
            health: ServiceHealthProto::ServiceHealthDegraded as i32,
        });

        let result = service.update_service_health(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.success);
        assert!(response.error_message.is_empty());

        // Verify health was updated
        let registry = service.registry.read().await;
        let all_services = registry.get_all_services();
        let instances = all_services.get("test-service").unwrap();
        assert_eq!(instances[0].health, ServiceHealth::Degraded);
    }

    #[tokio::test]
    async fn test_update_health_nonexistent_service() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        let request = Request::new(UpdateServiceHealthRequest {
            service_name: "nonexistent-service".to_string(),
            instance_id: "nonexistent-id".to_string(),
            health: ServiceHealthProto::ServiceHealthHealthy as i32,
        });

        let result = service.update_service_health(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_get_discovery_stats() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        // Register services with different health states
        for i in 0..3 {
            let mut proto_instance = create_test_proto_instance();
            proto_instance.name = format!("stats-test-{}", i);
            proto_instance.id = uuid::Uuid::new_v4().to_string();

            let register_request = Request::new(RegisterServiceRequest {
                instance: Some(proto_instance.clone()),
            });

            service.register_service(register_request).await.unwrap();

            // Set different health states
            let health = match i {
                0 => ServiceHealthProto::ServiceHealthHealthy,
                1 => ServiceHealthProto::ServiceHealthDegraded,
                _ => ServiceHealthProto::ServiceHealthUnhealthy,
            };

            let health_request = Request::new(UpdateServiceHealthRequest {
                service_name: proto_instance.name,
                instance_id: proto_instance.id,
                health: health as i32,
            });

            service.update_service_health(health_request).await.unwrap();
        }

        // Get stats
        let request = Request::new(ServiceDiscoveryStatsRequest {});
        let result = service.get_discovery_stats(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.total_services, 3);
        assert_eq!(response.total_instances, 3);
        assert_eq!(response.healthy_instances, 1);
        assert_eq!(response.degraded_instances, 1);
        assert_eq!(response.unhealthy_instances, 1);
    }

    #[tokio::test]
    async fn test_proto_conversion() {
        let daemon = create_test_daemon().await;
        let service = ServiceDiscoveryImpl::new(daemon);

        let proto_instance = create_test_proto_instance();

        // Test proto to internal conversion
        let internal_instance = service.proto_to_service_instance(proto_instance.clone()).unwrap();
        assert_eq!(internal_instance.name, proto_instance.name);
        assert_eq!(internal_instance.version, proto_instance.version);
        assert_eq!(internal_instance.weight, proto_instance.weight as u8);

        // Test internal to proto conversion
        let converted_proto = service.service_instance_to_proto(&internal_instance);
        assert_eq!(converted_proto.name, proto_instance.name);
        assert_eq!(converted_proto.version, proto_instance.version);
        assert_eq!(converted_proto.weight, proto_instance.weight);
        assert_eq!(converted_proto.health, proto_instance.health);
    }

    #[test]
    fn test_invalid_address_conversion() {
        let daemon_config = create_test_daemon_config();
        let daemon = tokio_test::block_on(WorkspaceDaemon::new(daemon_config)).unwrap();
        let service = ServiceDiscoveryImpl::new(Arc::new(daemon));

        let mut proto_instance = create_test_proto_instance();
        proto_instance.address = "invalid-address".to_string();

        let result = service.proto_to_service_instance(proto_instance);
        assert!(result.is_err());
    }
}