//! Container management for integration tests using testcontainers

use std::collections::HashMap;
use testcontainers::{clients::Cli, Container, Image};
use testcontainers_modules::qdrant::Qdrant;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

use crate::config::CONTAINER_STARTUP_TIMEOUT;
use crate::TestResult;

/// Wrapper for Qdrant test container with utility methods
pub struct QdrantTestContainer<'a> {
    container: Container<'a, Qdrant>,
    host: String,
    port: u16,
    api_key: Option<String>,
}

impl<'a> QdrantTestContainer<'a> {
    /// Start a new Qdrant container for testing
    pub async fn start(docker: &'a Cli) -> TestResult<Self> {
        let qdrant_image = Qdrant::default();
        let container = docker.run(qdrant_image);

        let host = "localhost".to_string();
        let port = container.get_host_port_ipv4(6333);

        // Wait for container to be ready
        timeout(
            CONTAINER_STARTUP_TIMEOUT,
            Self::wait_for_ready(&host, port)
        ).await??;

        Ok(Self {
            container,
            host,
            port,
            api_key: None,
        })
    }

    /// Get the connection URL for this Qdrant instance
    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Get the host
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Get the API key if set
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
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

        let response = client
            .put(&url)
            .json(&create_request)
            .send()
            .await?;

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
        vectors: Vec<(String, Vec<f32>)>
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

        let response = client
            .put(&url)
            .json(&upsert_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Failed to insert vectors: {}", response.status()).into());
        }

        Ok(())
    }
}

/// Manager for multiple test containers
pub struct ContainerManager {
    docker: Cli,
    containers: HashMap<String, QdrantTestContainer<'static>>,
}

impl ContainerManager {
    /// Create a new container manager
    pub fn new() -> Self {
        Self {
            docker: Cli::default(),
            containers: HashMap::new(),
        }
    }

    /// Start a named Qdrant container
    pub async fn start_qdrant(&mut self, name: &str) -> TestResult<&QdrantTestContainer> {
        // Note: This is simplified - in practice you'd need to handle lifetimes properly
        // For now, we'll use a simpler approach where each test manages its own container
        todo!("Implement proper lifetime management for containers")
    }

    /// Stop and remove a container
    pub async fn stop_container(&mut self, name: &str) -> TestResult<()> {
        self.containers.remove(name);
        Ok(())
    }

    /// Stop all containers
    pub async fn stop_all(&mut self) -> TestResult<()> {
        self.containers.clear();
        Ok(())
    }
}

impl Default for ContainerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a unique Qdrant container for a test
pub async fn create_test_qdrant() -> TestResult<(Cli, QdrantTestContainer<'_>)> {
    let docker = Cli::default();
    let container = QdrantTestContainer::start(&docker).await?;
    Ok((docker, container))
}

/// Helper function to create a test collection with random name
pub async fn create_test_collection_with_random_name(
    qdrant: &QdrantTestContainer<'_>,
    vector_size: u64,
) -> TestResult<String> {
    let collection_name = format!("test_collection_{}", Uuid::new_v4().simple());
    qdrant.create_test_collection(&collection_name, vector_size).await?;
    Ok(collection_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_qdrant_container_startup() -> TestResult {
        let (docker, qdrant) = create_test_qdrant().await?;

        // Test basic connectivity
        let client = reqwest::Client::new();
        let response = client.get(&format!("{}/health", qdrant.url())).send().await?;
        assert!(response.status().is_success());

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_operations() -> TestResult {
        let (docker, qdrant) = create_test_qdrant().await?;

        // Create a test collection
        let collection_name = create_test_collection_with_random_name(&qdrant, 384).await?;

        // Insert some test vectors
        let test_vectors = vec![
            ("test document 1".to_string(), vec![0.1; 384]),
            ("test document 2".to_string(), vec![0.2; 384]),
        ];

        qdrant.insert_test_vectors(&collection_name, test_vectors).await?;

        // Cleanup
        qdrant.delete_collection(&collection_name).await?;

        Ok(())
    }
}