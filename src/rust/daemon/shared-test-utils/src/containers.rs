//! Container management for integration tests using testcontainers
//!
//! Provides managed Qdrant containers for integration testing. Uses
//! testcontainers v0.25 `ContainerAsync` (no lifetime parameter) with
//! `GenericImage` since there is no Qdrant module in testcontainers-modules.

use std::collections::HashMap;
use testcontainers::core::IntoContainerPort;
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage};
use tokio::time::{timeout, Duration};
use uuid::Uuid;

use crate::config::CONTAINER_STARTUP_TIMEOUT;
use crate::TestResult;

/// Default Qdrant Docker image
const QDRANT_IMAGE: &str = "qdrant/qdrant";
/// Default Qdrant image tag
const QDRANT_TAG: &str = "latest";
/// Qdrant HTTP API port
const QDRANT_HTTP_PORT: u16 = 6333;
/// Qdrant gRPC port
const QDRANT_GRPC_PORT: u16 = 6334;

/// Wrapper for Qdrant test container with utility methods
pub struct QdrantTestContainer {
    /// Held to keep Docker container alive; Drop stops the container
    #[allow(dead_code)]
    container: ContainerAsync<GenericImage>,
    host: String,
    http_port: u16,
    grpc_port: u16,
}

impl QdrantTestContainer {
    /// Start a new Qdrant container for testing
    pub async fn start() -> TestResult<Self> {
        let image = GenericImage::new(QDRANT_IMAGE, QDRANT_TAG)
            .with_exposed_port(QDRANT_HTTP_PORT.tcp())
            .with_exposed_port(QDRANT_GRPC_PORT.tcp());

        let container = image.start().await
            .map_err(|e| format!("Failed to start Qdrant container: {}", e))?;

        let host = container.get_host().await
            .map_err(|e| format!("Failed to get container host: {}", e))?
            .to_string();
        let http_port = container.get_host_port_ipv4(QDRANT_HTTP_PORT).await
            .map_err(|e| format!("Failed to get HTTP port: {}", e))?;
        let grpc_port = container.get_host_port_ipv4(QDRANT_GRPC_PORT).await
            .map_err(|e| format!("Failed to get gRPC port: {}", e))?;

        // Wait for container to be ready
        timeout(
            CONTAINER_STARTUP_TIMEOUT,
            Self::wait_for_ready(&host, http_port),
        )
        .await??;

        Ok(Self {
            container,
            host,
            http_port,
            grpc_port,
        })
    }

    /// Get the HTTP connection URL for this Qdrant instance
    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.http_port)
    }

    /// Get the gRPC connection URL for this Qdrant instance
    pub fn grpc_url(&self) -> String {
        format!("http://{}:{}", self.host, self.grpc_port)
    }

    /// Get the host
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Get the HTTP port
    pub fn http_port(&self) -> u16 {
        self.http_port
    }

    /// Get the gRPC port
    pub fn grpc_port(&self) -> u16 {
        self.grpc_port
    }

    /// Wait for Qdrant to be ready to accept connections
    async fn wait_for_ready(host: &str, port: u16) -> TestResult<()> {
        let client = reqwest::Client::new();
        let health_url = format!("http://{}:{}/health", host, port);

        for _ in 0..30 {
            match client.get(&health_url).send().await {
                Ok(response) if response.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }

        Err("Qdrant container failed to start within timeout".into())
    }

    /// Create a test collection in this Qdrant instance
    pub async fn create_test_collection(&self, name: &str, vector_size: u64) -> TestResult<()> {
        let client = reqwest::Client::new();
        let url = format!("{}/collections/{}", self.url(), name);

        let create_request = serde_json::json!({
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        });

        let response = client.put(&url).json(&create_request).send().await?;

        if !response.status().is_success() {
            return Err(format!("Failed to create collection: {}", response.status()).into());
        }

        Ok(())
    }

    /// Delete a collection from this Qdrant instance
    pub async fn delete_collection(&self, name: &str) -> TestResult<()> {
        let client = reqwest::Client::new();
        let url = format!("{}/collections/{}", self.url(), name);

        let response = client.delete(&url).send().await?;

        // Accept both success and not found as success for cleanup
        if response.status().is_success() || response.status() == 404 {
            Ok(())
        } else {
            Err(format!("Failed to delete collection: {}", response.status()).into())
        }
    }

    /// Insert test vectors into a collection
    pub async fn insert_test_vectors(
        &self,
        collection: &str,
        vectors: Vec<(String, Vec<f32>)>,
    ) -> TestResult<()> {
        let client = reqwest::Client::new();
        let url = format!("{}/collections/{}/points", self.url(), collection);

        let points: Vec<serde_json::Value> = vectors
            .into_iter()
            .enumerate()
            .map(|(i, (payload, vector))| {
                serde_json::json!({
                    "id": i,
                    "vector": vector,
                    "payload": {"text": payload}
                })
            })
            .collect();

        let upsert_request = serde_json::json!({
            "points": points
        });

        let response = client.put(&url).json(&upsert_request).send().await?;

        if !response.status().is_success() {
            return Err(format!("Failed to insert vectors: {}", response.status()).into());
        }

        Ok(())
    }
}

/// Manager for multiple named test containers
pub struct ContainerManager {
    containers: HashMap<String, QdrantTestContainer>,
}

impl ContainerManager {
    /// Create a new container manager
    pub fn new() -> Self {
        Self {
            containers: HashMap::new(),
        }
    }

    /// Start a named Qdrant container
    pub async fn start_qdrant(&mut self, name: &str) -> TestResult<&QdrantTestContainer> {
        if self.containers.contains_key(name) {
            return Err(format!("Container '{}' already exists", name).into());
        }

        let container = QdrantTestContainer::start().await?;
        self.containers.insert(name.to_string(), container);
        Ok(self.containers.get(name).unwrap())
    }

    /// Get a reference to a running container by name
    pub fn get_container(&self, name: &str) -> TestResult<&QdrantTestContainer> {
        self.containers
            .get(name)
            .ok_or_else(|| format!("Container '{}' not found", name).into())
    }

    /// List all managed container names
    pub fn list_containers(&self) -> Vec<&str> {
        self.containers.keys().map(|s| s.as_str()).collect()
    }

    /// Stop and remove a container by name (Drop handles Docker cleanup)
    pub fn stop_container(&mut self, name: &str) -> TestResult<()> {
        self.containers
            .remove(name)
            .ok_or_else(|| format!("Container '{}' not found", name))?;
        Ok(())
    }

    /// Stop all containers
    pub fn stop_all(&mut self) {
        self.containers.clear();
    }
}

impl Default for ContainerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a unique container name for parallel test isolation
pub fn generate_container_name(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

/// Helper function to create a unique Qdrant container for a test
pub async fn create_test_qdrant() -> TestResult<QdrantTestContainer> {
    QdrantTestContainer::start().await
}

/// Helper function to create a test collection with random name
pub async fn create_test_collection_with_random_name(
    qdrant: &QdrantTestContainer,
    vector_size: u64,
) -> TestResult<String> {
    let collection_name = format!("test_collection_{}", Uuid::new_v4().simple());
    qdrant
        .create_test_collection(&collection_name, vector_size)
        .await?;
    Ok(collection_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_manager_creation() {
        let manager = ContainerManager::new();
        assert!(manager.list_containers().is_empty());
    }

    #[test]
    fn test_generate_container_name() {
        let name1 = generate_container_name("test");
        let name2 = generate_container_name("test");
        assert!(name1.starts_with("test_"));
        assert!(name2.starts_with("test_"));
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_container_manager_get_nonexistent() {
        let manager = ContainerManager::new();
        assert!(manager.get_container("nonexistent").is_err());
    }

    #[test]
    fn test_container_manager_stop_nonexistent() {
        let mut manager = ContainerManager::new();
        assert!(manager.stop_container("nonexistent").is_err());
    }
}
