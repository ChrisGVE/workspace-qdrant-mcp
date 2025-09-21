//! Mock implementations for testing

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use uuid::Uuid;
// use wiremock::{Mock, MockServer, ResponseTemplate}; // Temporarily disabled due to dependency conflicts

use crate::TestResult;

/// Mock Qdrant server for testing without real containers
pub struct MockQdrantServer {
    server: MockServer,
    collections: Arc<Mutex<HashMap<String, MockCollection>>>,
}

#[derive(Debug, Clone)]
struct MockCollection {
    vector_size: u64,
    points: HashMap<u64, MockPoint>,
}

#[derive(Debug, Clone)]
struct MockPoint {
    id: u64,
    vector: Vec<f32>,
    payload: serde_json::Value,
}

impl MockQdrantServer {
    /// Start a new mock Qdrant server
    pub async fn start() -> TestResult<Self> {
        let server = MockServer::start().await;
        let collections = Arc::new(Mutex::new(HashMap::new()));

        // Setup default mocks
        let collections_clone = collections.clone();
        Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "status": "ok"
            })))
            .mount(&server)
            .await;

        // Collection creation mock
        let collections_clone = collections.clone();
        Mock::given(wiremock::matchers::method("PUT"))
            .and(wiremock::matchers::path_regex(r"/collections/[^/]+"))
            .respond_with_function(move |req: &wiremock::Request| {
                let path = req.url.path();
                let collection_name = path.split('/').last().unwrap().to_string();

                // Parse the request body to get vector configuration
                let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
                let vector_size = body["vectors"]["size"].as_u64().unwrap_or(384);

                let mut collections = collections_clone.lock().unwrap();
                collections.insert(collection_name.clone(), MockCollection {
                    vector_size,
                    points: HashMap::new(),
                });

                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "result": true,
                    "status": "ok",
                    "time": 0.001
                }))
            })
            .mount(&server)
            .await;

        // Collection deletion mock
        let collections_clone = collections.clone();
        Mock::given(wiremock::matchers::method("DELETE"))
            .and(wiremock::matchers::path_regex(r"/collections/[^/]+"))
            .respond_with_function(move |req: &wiremock::Request| {
                let path = req.url.path();
                let collection_name = path.split('/').last().unwrap().to_string();

                let mut collections = collections_clone.lock().unwrap();
                collections.remove(&collection_name);

                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "result": true,
                    "status": "ok",
                    "time": 0.001
                }))
            })
            .mount(&server)
            .await;

        // Points insertion mock
        let collections_clone = collections.clone();
        Mock::given(wiremock::matchers::method("PUT"))
            .and(wiremock::matchers::path_regex(r"/collections/[^/]+/points"))
            .respond_with_function(move |req: &wiremock::Request| {
                let path = req.url.path();
                let parts: Vec<&str> = path.split('/').collect();
                let collection_name = parts[2].to_string();

                let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
                let points = body["points"].as_array().unwrap_or(&vec![]);

                let mut collections = collections_clone.lock().unwrap();
                if let Some(collection) = collections.get_mut(&collection_name) {
                    for point in points {
                        let id = point["id"].as_u64().unwrap_or(0);
                        let vector = point["vector"].as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                            .collect();
                        let payload = point["payload"].clone();

                        collection.points.insert(id, MockPoint { id, vector, payload });
                    }
                }

                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "result": {
                        "operation_id": 12345,
                        "status": "acknowledged"
                    },
                    "status": "ok",
                    "time": 0.002
                }))
            })
            .mount(&server)
            .await;

        // Search mock
        let collections_clone = collections.clone();
        Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(r"/collections/[^/]+/points/search"))
            .respond_with_function(move |req: &wiremock::Request| {
                let path = req.url.path();
                let parts: Vec<&str> = path.split('/').collect();
                let collection_name = parts[2].to_string();

                let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
                let limit = body["limit"].as_u64().unwrap_or(10);

                let collections = collections_clone.lock().unwrap();
                let results = if let Some(collection) = collections.get(&collection_name) {
                    collection.points.values()
                        .take(limit as usize)
                        .enumerate()
                        .map(|(i, point)| serde_json::json!({
                            "id": point.id,
                            "score": 0.9 - (i as f64 * 0.1),
                            "payload": point.payload,
                            "vector": point.vector
                        }))
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                };

                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "result": results,
                    "status": "ok",
                    "time": 0.003
                }))
            })
            .mount(&server)
            .await;

        Ok(Self {
            server,
            collections,
        })
    }

    /// Get the base URL of the mock server
    pub fn url(&self) -> String {
        self.server.uri()
    }

    /// Get the number of collections
    pub fn collection_count(&self) -> usize {
        self.collections.lock().unwrap().len()
    }

    /// Get the number of points in a collection
    pub fn point_count(&self, collection: &str) -> usize {
        self.collections
            .lock()
            .unwrap()
            .get(collection)
            .map(|c| c.points.len())
            .unwrap_or(0)
    }
}

/// Mock embedding service for testing
pub struct MockEmbeddingService {
    server: MockServer,
    embedding_dim: usize,
    response_delay: Duration,
}

impl MockEmbeddingService {
    /// Create a new mock embedding service
    pub async fn start(embedding_dim: usize) -> TestResult<Self> {
        let server = MockServer::start().await;
        let response_delay = Duration::from_millis(10);

        // Mock embedding endpoint
        Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/embed"))
            .respond_with_function(move |req: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
                let texts = body["texts"].as_array().unwrap_or(&vec![]);

                let embeddings: Vec<Vec<f32>> = texts
                    .iter()
                    .enumerate()
                    .map(|(i, text)| {
                        // Generate deterministic but varied embeddings based on text content
                        let text_str = text.as_str().unwrap_or("");
                        let hash = text_str.len() + i;
                        (0..embedding_dim)
                            .map(|j| ((hash + j) as f32 * 0.1).sin())
                            .collect()
                    })
                    .collect();

                ResponseTemplate::new(200)
                    .set_delay(response_delay)
                    .set_body_json(serde_json::json!({
                        "embeddings": embeddings,
                        "model": "mock-embedding-model",
                        "usage": {
                            "prompt_tokens": texts.len() * 10,
                            "total_tokens": texts.len() * 10
                        }
                    }))
            })
            .mount(&server)
            .await;

        // Health check endpoint
        Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "status": "healthy",
                "model_loaded": true
            })))
            .mount(&server)
            .await;

        Ok(Self {
            server,
            embedding_dim,
            response_delay,
        })
    }

    /// Get the base URL of the mock server
    pub fn url(&self) -> String {
        self.server.uri()
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Set response delay for testing timeout scenarios
    pub fn set_response_delay(&mut self, delay: Duration) {
        self.response_delay = delay;
    }
}

/// Mock file system watcher for testing
pub struct MockFileWatcher {
    events: Arc<Mutex<Vec<MockFileEvent>>>,
}

#[derive(Debug, Clone)]
pub struct MockFileEvent {
    pub path: String,
    pub event_type: MockEventType,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub enum MockEventType {
    Created,
    Modified,
    Deleted,
    Renamed { from: String, to: String },
}

impl MockFileWatcher {
    /// Create a new mock file watcher
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Simulate a file creation event
    pub fn emit_created(&self, path: &str) {
        let event = MockFileEvent {
            path: path.to_string(),
            event_type: MockEventType::Created,
            timestamp: Instant::now(),
        };
        self.events.lock().unwrap().push(event);
    }

    /// Simulate a file modification event
    pub fn emit_modified(&self, path: &str) {
        let event = MockFileEvent {
            path: path.to_string(),
            event_type: MockEventType::Modified,
            timestamp: Instant::now(),
        };
        self.events.lock().unwrap().push(event);
    }

    /// Simulate a file deletion event
    pub fn emit_deleted(&self, path: &str) {
        let event = MockFileEvent {
            path: path.to_string(),
            event_type: MockEventType::Deleted,
            timestamp: Instant::now(),
        };
        self.events.lock().unwrap().push(event);
    }

    /// Get all recorded events
    pub fn events(&self) -> Vec<MockFileEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Clear all recorded events
    pub fn clear_events(&self) {
        self.events.lock().unwrap().clear();
    }

    /// Get event count
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

impl Default for MockFileWatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock metrics collector for testing
pub struct MockMetricsCollector {
    metrics: Arc<Mutex<HashMap<String, f64>>>,
    counters: Arc<Mutex<HashMap<String, u64>>>,
}

impl MockMetricsCollector {
    /// Create a new mock metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a metric value
    pub fn record_metric(&self, name: &str, value: f64) {
        self.metrics.lock().unwrap().insert(name.to_string(), value);
    }

    /// Increment a counter
    pub fn increment_counter(&self, name: &str) {
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Get a metric value
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.lock().unwrap().get(name).copied()
    }

    /// Get a counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters.lock().unwrap().get(name).copied().unwrap_or(0)
    }

    /// Get all metrics
    pub fn all_metrics(&self) -> HashMap<String, f64> {
        self.metrics.lock().unwrap().clone()
    }

    /// Get all counters
    pub fn all_counters(&self) -> HashMap<String, u64> {
        self.counters.lock().unwrap().clone()
    }

    /// Reset all metrics and counters
    pub fn reset(&self) {
        self.metrics.lock().unwrap().clear();
        self.counters.lock().unwrap().clear();
    }
}

impl Default for MockMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_qdrant_server() -> TestResult {
        let mock_server = MockQdrantServer::start().await?;

        // Test health endpoint
        let client = reqwest::Client::new();
        let response = client
            .get(&format!("{}/health", mock_server.url()))
            .send()
            .await?;

        assert!(response.status().is_success());

        // Test collection creation
        let create_response = client
            .put(&format!("{}/collections/test", mock_server.url()))
            .json(&serde_json::json!({
                "vectors": { "size": 384, "distance": "Cosine" }
            }))
            .send()
            .await?;

        assert!(create_response.status().is_success());
        assert_eq!(mock_server.collection_count(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_mock_embedding_service() -> TestResult {
        let mock_service = MockEmbeddingService::start(384).await?;

        let client = reqwest::Client::new();
        let response = client
            .post(&format!("{}/embed", mock_service.url()))
            .json(&serde_json::json!({
                "texts": ["hello world", "test document"]
            }))
            .send()
            .await?;

        assert!(response.status().is_success());

        let body: serde_json::Value = response.json().await?;
        let embeddings = body["embeddings"].as_array().unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].as_array().unwrap().len(), 384);

        Ok(())
    }

    #[test]
    fn test_mock_file_watcher() {
        let watcher = MockFileWatcher::new();

        watcher.emit_created("/test/file.txt");
        watcher.emit_modified("/test/file.txt");

        assert_eq!(watcher.event_count(), 2);

        let events = watcher.events();
        assert!(matches!(events[0].event_type, MockEventType::Created));
        assert!(matches!(events[1].event_type, MockEventType::Modified));
    }

    #[test]
    fn test_mock_metrics_collector() {
        let collector = MockMetricsCollector::new();

        collector.record_metric("latency", 123.45);
        collector.increment_counter("requests");
        collector.increment_counter("requests");

        assert_eq!(collector.get_metric("latency"), Some(123.45));
        assert_eq!(collector.get_counter("requests"), 2);
    }
}