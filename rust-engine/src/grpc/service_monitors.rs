//! Service-level health monitors for document processing, search, and memory operations
//!
//! This module provides specialized health monitoring for each of the main gRPC services
//! defined in workspace_daemon.proto, with service-specific metrics and health indicators.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, error};

use crate::error::DaemonResult;
use crate::grpc::health::{HealthMonitoringSystem, ServiceHealthMonitor, AlertConfig, HealthStatus};

/// Service-specific health monitor for DocumentProcessor service
#[derive(Debug)]
pub struct DocumentProcessorMonitor {
    /// Base health monitor
    health_monitor: Arc<ServiceHealthMonitor>,
    /// Documents currently being processed
    documents_in_progress: Arc<tokio::sync::RwLock<u32>>,
    /// Total documents processed successfully
    total_processed: Arc<tokio::sync::RwLock<u64>>,
    /// Total processing failures
    total_failures: Arc<tokio::sync::RwLock<u64>>,
    /// Queue size for pending documents
    queue_size: Arc<tokio::sync::RwLock<u32>>,
}

impl DocumentProcessorMonitor {
    /// Create a new document processor monitor
    pub async fn new(monitoring_system: &Arc<HealthMonitoringSystem>) -> DaemonResult<Self> {
        let alert_config = AlertConfig {
            max_error_rate: 0.02, // 2% - stricter for document processing
            max_avg_latency_ms: 5000.0, // 5 seconds for document processing
            max_p99_latency_ms: 15000.0, // 15 seconds P99
            min_throughput_rps: 0.5, // 0.5 documents per second minimum
            evaluation_window_secs: 300,
        };

        let monitor = monitoring_system.register_service("document-processor".to_string(), Some(alert_config)).await;

        Ok(Self {
            health_monitor: monitor,
            documents_in_progress: Arc::new(tokio::sync::RwLock::new(0)),
            total_processed: Arc::new(tokio::sync::RwLock::new(0)),
            total_failures: Arc::new(tokio::sync::RwLock::new(0)),
            queue_size: Arc::new(tokio::sync::RwLock::new(0)),
        })
    }

    /// Record the start of document processing
    pub async fn start_processing(&self, document_id: &str) {
        let mut in_progress = self.documents_in_progress.write().await;
        *in_progress += 1;

        debug!("Started processing document {}, {} documents in progress", document_id, *in_progress);

        // Check for overload condition
        if *in_progress > 10 {
            warn!("Document processor overloaded: {} documents in progress", *in_progress);
            self.health_monitor.set_status(
                HealthStatus::NotServing,
                Some(format!("Processing overload: {} documents in progress", *in_progress))
            ).await;
        }
    }

    /// Record successful completion of document processing
    pub async fn complete_processing(&self, document_id: &str, processing_time: Duration) {
        // Update counters
        let mut in_progress = self.documents_in_progress.write().await;
        let mut total_processed = self.total_processed.write().await;

        if *in_progress > 0 {
            *in_progress -= 1;
        }
        *total_processed += 1;

        // Record success in health monitor
        self.health_monitor.record_success(processing_time).await;

        debug!("Completed processing document {} in {:.2}ms, {} documents remaining",
            document_id, processing_time.as_secs_f64() * 1000.0, *in_progress);

        // Check if we recovered from overload
        if *in_progress <= 5 {
            let health = self.health_monitor.get_health().await;
            if health.status == HealthStatus::NotServing && health.message.contains("overload") {
                self.health_monitor.set_status(
                    HealthStatus::Serving,
                    Some("Recovered from processing overload".to_string())
                ).await;
            }
        }
    }

    /// Record failed document processing
    pub async fn fail_processing(&self, document_id: &str, processing_time: Duration, error: &str) {
        // Update counters
        let mut in_progress = self.documents_in_progress.write().await;
        let mut total_failures = self.total_failures.write().await;

        if *in_progress > 0 {
            *in_progress -= 1;
        }
        *total_failures += 1;

        // Record failure in health monitor
        self.health_monitor.record_failure(processing_time).await;

        warn!("Failed processing document {} in {:.2}ms: {}",
            document_id, processing_time.as_secs_f64() * 1000.0, error);
    }

    /// Update queue size for pending documents
    pub async fn update_queue_size(&self, size: u32) {
        let mut queue_size = self.queue_size.write().await;
        *queue_size = size;

        debug!("Document processing queue size: {}", size);

        // Alert on large queue
        if size > 100 {
            warn!("Document processing queue is large: {} documents pending", size);
        }
    }

    /// Get processing statistics
    pub async fn get_processing_stats(&self) -> DocumentProcessingStats {
        let in_progress = *self.documents_in_progress.read().await;
        let total_processed = *self.total_processed.read().await;
        let total_failures = *self.total_failures.read().await;
        let queue_size = *self.queue_size.read().await;

        DocumentProcessingStats {
            in_progress,
            total_processed,
            total_failures,
            queue_size,
            processing_rate: self.calculate_processing_rate().await,
        }
    }

    /// Calculate current processing rate (documents per second)
    async fn calculate_processing_rate(&self) -> f64 {
        let health = self.health_monitor.get_health().await;
        health.metrics.throughput_rps
    }
}

/// Document processing statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentProcessingStats {
    pub in_progress: u32,
    pub total_processed: u64,
    pub total_failures: u64,
    pub queue_size: u32,
    pub processing_rate: f64,
}

/// Service-specific health monitor for SearchService
#[derive(Debug)]
pub struct SearchServiceMonitor {
    /// Base health monitor
    health_monitor: Arc<ServiceHealthMonitor>,
    /// Search cache hit rate tracking
    cache_hits: Arc<tokio::sync::RwLock<u64>>,
    /// Search cache misses
    cache_misses: Arc<tokio::sync::RwLock<u64>>,
    /// Active concurrent searches
    concurrent_searches: Arc<tokio::sync::RwLock<u32>>,
    /// Average result count per search
    avg_results_per_search: Arc<tokio::sync::RwLock<f64>>,
}

impl SearchServiceMonitor {
    /// Create a new search service monitor
    pub async fn new(monitoring_system: &Arc<HealthMonitoringSystem>) -> DaemonResult<Self> {
        let alert_config = AlertConfig {
            max_error_rate: 0.01, // 1% - very strict for search
            max_avg_latency_ms: 500.0, // 500ms for search operations
            max_p99_latency_ms: 2000.0, // 2 seconds P99
            min_throughput_rps: 5.0, // 5 searches per second minimum
            evaluation_window_secs: 180, // 3 minute window
        };

        let monitor = monitoring_system.register_service("search-service".to_string(), Some(alert_config)).await;

        Ok(Self {
            health_monitor: monitor,
            cache_hits: Arc::new(tokio::sync::RwLock::new(0)),
            cache_misses: Arc::new(tokio::sync::RwLock::new(0)),
            concurrent_searches: Arc::new(tokio::sync::RwLock::new(0)),
            avg_results_per_search: Arc::new(tokio::sync::RwLock::new(0.0)),
        })
    }

    /// Record the start of a search operation
    pub async fn start_search(&self, query: &str) -> SearchTracker {
        let mut concurrent = self.concurrent_searches.write().await;
        *concurrent += 1;

        debug!("Started search for '{}', {} concurrent searches", query, *concurrent);

        // Check for overload
        if *concurrent > 50 {
            warn!("Search service overloaded: {} concurrent searches", *concurrent);
            self.health_monitor.set_status(
                HealthStatus::NotServing,
                Some(format!("Search overload: {} concurrent searches", *concurrent))
            ).await;
        }

        SearchTracker {
            monitor: Arc::downgrade(&self.health_monitor),
            concurrent_searches: Arc::downgrade(&self.concurrent_searches),
            avg_results: Arc::downgrade(&self.avg_results_per_search),
            start_time: Instant::now(),
        }
    }

    /// Record a cache hit
    pub async fn record_cache_hit(&self) {
        let mut hits = self.cache_hits.write().await;
        *hits += 1;

        debug!("Search cache hit recorded");
    }

    /// Record a cache miss
    pub async fn record_cache_miss(&self) {
        let mut misses = self.cache_misses.write().await;
        *misses += 1;

        debug!("Search cache miss recorded");
    }

    /// Get search statistics
    pub async fn get_search_stats(&self) -> SearchStats {
        let cache_hits = *self.cache_hits.read().await;
        let cache_misses = *self.cache_misses.read().await;
        let concurrent_searches = *self.concurrent_searches.read().await;
        let avg_results_per_search = *self.avg_results_per_search.read().await;

        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };

        SearchStats {
            cache_hit_rate,
            concurrent_searches,
            avg_results_per_search,
            search_rate: self.calculate_search_rate().await,
        }
    }

    /// Calculate current search rate (searches per second)
    async fn calculate_search_rate(&self) -> f64 {
        let health = self.health_monitor.get_health().await;
        health.metrics.throughput_rps
    }
}

/// Search operation tracker
pub struct SearchTracker {
    monitor: std::sync::Weak<ServiceHealthMonitor>,
    concurrent_searches: std::sync::Weak<tokio::sync::RwLock<u32>>,
    avg_results: std::sync::Weak<tokio::sync::RwLock<f64>>,
    start_time: Instant,
}

impl SearchTracker {
    /// Complete search operation successfully
    pub async fn complete_success(self, result_count: usize) {
        let duration = self.start_time.elapsed();

        if let Some(monitor) = self.monitor.upgrade() {
            monitor.record_success(duration).await;
        }

        if let Some(concurrent) = self.concurrent_searches.upgrade() {
            let mut concurrent_searches = concurrent.write().await;
            if *concurrent_searches > 0 {
                *concurrent_searches -= 1;
            }

            // Check recovery from overload
            if *concurrent_searches <= 25 {
                if let Some(monitor) = self.monitor.upgrade() {
                    let health = monitor.get_health().await;
                    if health.status == HealthStatus::NotServing && health.message.contains("overload") {
                        monitor.set_status(
                            HealthStatus::Serving,
                            Some("Recovered from search overload".to_string())
                        ).await;
                    }
                }
            }
        }

        if let Some(avg_results) = self.avg_results.upgrade() {
            let mut avg = avg_results.write().await;
            // Simple exponential moving average
            *avg = *avg * 0.9 + result_count as f64 * 0.1;
        }

        debug!("Search completed successfully in {:.2}ms with {} results",
            duration.as_secs_f64() * 1000.0, result_count);
    }

    /// Complete search operation with failure
    pub async fn complete_failure(self, error: &str) {
        let duration = self.start_time.elapsed();

        if let Some(monitor) = self.monitor.upgrade() {
            monitor.record_failure(duration).await;
        }

        if let Some(concurrent) = self.concurrent_searches.upgrade() {
            let mut concurrent_searches = concurrent.write().await;
            if *concurrent_searches > 0 {
                *concurrent_searches -= 1;
            }
        }

        warn!("Search failed in {:.2}ms: {}", duration.as_secs_f64() * 1000.0, error);
    }
}

/// Search service statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchStats {
    pub cache_hit_rate: f64,
    pub concurrent_searches: u32,
    pub avg_results_per_search: f64,
    pub search_rate: f64,
}

/// Service-specific health monitor for MemoryService
#[derive(Debug)]
pub struct MemoryServiceMonitor {
    /// Base health monitor
    health_monitor: Arc<ServiceHealthMonitor>,
    /// Memory pool utilization percentage
    memory_utilization: Arc<tokio::sync::RwLock<f64>>,
    /// Number of active collections
    active_collections: Arc<tokio::sync::RwLock<u32>>,
    /// Total documents stored
    total_documents: Arc<tokio::sync::RwLock<u64>>,
    /// Memory operations in progress
    operations_in_progress: Arc<tokio::sync::RwLock<u32>>,
}

impl MemoryServiceMonitor {
    /// Create a new memory service monitor
    pub async fn new(monitoring_system: &Arc<HealthMonitoringSystem>) -> DaemonResult<Self> {
        let alert_config = AlertConfig {
            max_error_rate: 0.005, // 0.5% - strictest for data integrity
            max_avg_latency_ms: 200.0, // 200ms for memory operations
            max_p99_latency_ms: 1000.0, // 1 second P99
            min_throughput_rps: 10.0, // 10 memory operations per second minimum
            evaluation_window_secs: 120, // 2 minute window
        };

        let monitor = monitoring_system.register_service("memory-service".to_string(), Some(alert_config)).await;

        Ok(Self {
            health_monitor: monitor,
            memory_utilization: Arc::new(tokio::sync::RwLock::new(0.0)),
            active_collections: Arc::new(tokio::sync::RwLock::new(0)),
            total_documents: Arc::new(tokio::sync::RwLock::new(0)),
            operations_in_progress: Arc::new(tokio::sync::RwLock::new(0)),
        })
    }

    /// Record the start of a memory operation
    pub async fn start_operation(&self, operation_type: &str) -> MemoryOperationTracker {
        let mut in_progress = self.operations_in_progress.write().await;
        *in_progress += 1;

        debug!("Started {} operation, {} operations in progress", operation_type, *in_progress);

        MemoryOperationTracker {
            monitor: Arc::downgrade(&self.health_monitor),
            operations_in_progress: Arc::downgrade(&self.operations_in_progress),
            start_time: Instant::now(),
            operation_type: operation_type.to_string(),
        }
    }

    /// Update memory utilization percentage
    pub async fn update_memory_utilization(&self, utilization: f64) {
        let mut util = self.memory_utilization.write().await;
        *util = utilization;

        debug!("Memory utilization: {:.2}%", utilization);

        // Alert on high memory usage
        if utilization > 90.0 {
            warn!("High memory utilization: {:.2}%", utilization);
            self.health_monitor.set_status(
                HealthStatus::NotServing,
                Some(format!("High memory utilization: {:.2}%", utilization))
            ).await;
        } else if utilization < 80.0 {
            // Check for recovery from high memory
            let health = self.health_monitor.get_health().await;
            if health.status == HealthStatus::NotServing && health.message.contains("memory utilization") {
                self.health_monitor.set_status(
                    HealthStatus::Serving,
                    Some("Recovered from high memory utilization".to_string())
                ).await;
            }
        }
    }

    /// Update collection count
    pub async fn update_collection_count(&self, count: u32) {
        let mut collections = self.active_collections.write().await;
        *collections = count;

        debug!("Active collections: {}", count);
    }

    /// Update total document count
    pub async fn update_document_count(&self, count: u64) {
        let mut docs = self.total_documents.write().await;
        *docs = count;

        debug!("Total documents: {}", count);
    }

    /// Get memory service statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let memory_utilization = *self.memory_utilization.read().await;
        let active_collections = *self.active_collections.read().await;
        let total_documents = *self.total_documents.read().await;
        let operations_in_progress = *self.operations_in_progress.read().await;

        MemoryStats {
            memory_utilization,
            active_collections,
            total_documents,
            operations_in_progress,
            operation_rate: self.calculate_operation_rate().await,
        }
    }

    /// Calculate current operation rate (operations per second)
    async fn calculate_operation_rate(&self) -> f64 {
        let health = self.health_monitor.get_health().await;
        health.metrics.throughput_rps
    }
}

/// Memory operation tracker
pub struct MemoryOperationTracker {
    monitor: std::sync::Weak<ServiceHealthMonitor>,
    operations_in_progress: std::sync::Weak<tokio::sync::RwLock<u32>>,
    start_time: Instant,
    operation_type: String,
}

impl MemoryOperationTracker {
    /// Complete memory operation successfully
    pub async fn complete_success(self) {
        let duration = self.start_time.elapsed();

        if let Some(monitor) = self.monitor.upgrade() {
            monitor.record_success(duration).await;
        }

        if let Some(in_progress) = self.operations_in_progress.upgrade() {
            let mut operations = in_progress.write().await;
            if *operations > 0 {
                *operations -= 1;
            }
        }

        debug!("{} operation completed successfully in {:.2}ms",
            self.operation_type, duration.as_secs_f64() * 1000.0);
    }

    /// Complete memory operation with failure
    pub async fn complete_failure(self, error: &str) {
        let duration = self.start_time.elapsed();

        if let Some(monitor) = self.monitor.upgrade() {
            monitor.record_failure(duration).await;
        }

        if let Some(in_progress) = self.operations_in_progress.upgrade() {
            let mut operations = in_progress.write().await;
            if *operations > 0 {
                *operations -= 1;
            }
        }

        error!("{} operation failed in {:.2}ms: {}",
            self.operation_type, duration.as_secs_f64() * 1000.0, error);
    }
}

/// Memory service statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryStats {
    pub memory_utilization: f64,
    pub active_collections: u32,
    pub total_documents: u64,
    pub operations_in_progress: u32,
    pub operation_rate: f64,
}

/// Comprehensive service monitoring system
#[derive(Debug)]
pub struct ServiceMonitors {
    /// Document processor monitor
    pub document_processor: DocumentProcessorMonitor,
    /// Search service monitor
    pub search_service: SearchServiceMonitor,
    /// Memory service monitor
    pub memory_service: MemoryServiceMonitor,
    /// Health monitoring system
    pub monitoring_system: Arc<HealthMonitoringSystem>,
}

impl ServiceMonitors {
    /// Create a new service monitoring system
    pub async fn new() -> DaemonResult<Self> {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));

        let document_processor = DocumentProcessorMonitor::new(&monitoring_system).await?;
        let search_service = SearchServiceMonitor::new(&monitoring_system).await?;
        let memory_service = MemoryServiceMonitor::new(&monitoring_system).await?;

        Ok(Self {
            document_processor,
            search_service,
            memory_service,
            monitoring_system,
        })
    }

    /// Create a new service monitoring system with external monitoring
    pub async fn with_external_monitoring(
        external_monitoring: Arc<crate::grpc::health::ExternalMonitoring>,
    ) -> DaemonResult<Self> {
        let monitoring_system = Arc::new(HealthMonitoringSystem::with_external_monitoring(None, external_monitoring));

        let document_processor = DocumentProcessorMonitor::new(&monitoring_system).await?;
        let search_service = SearchServiceMonitor::new(&monitoring_system).await?;
        let memory_service = MemoryServiceMonitor::new(&monitoring_system).await?;

        Ok(Self {
            document_processor,
            search_service,
            memory_service,
            monitoring_system,
        })
    }

    /// Get comprehensive service statistics
    pub async fn get_all_stats(&self) -> ServiceStats {
        let doc_stats = self.document_processor.get_processing_stats().await;
        let search_stats = self.search_service.get_search_stats().await;
        let memory_stats = self.memory_service.get_memory_stats().await;
        let (system_status, system_message) = self.monitoring_system.get_system_health().await;

        ServiceStats {
            system_status,
            system_message,
            document_processing: doc_stats,
            search: search_stats,
            memory: memory_stats,
        }
    }

    /// Perform health check for all services
    pub async fn perform_health_check(&self) -> DaemonResult<()> {
        info!("Performing comprehensive service health check");

        // Get all service health
        let all_health = self.monitoring_system.get_all_health().await;

        for (service_name, health) in all_health.iter() {
            match health.status {
                HealthStatus::Serving => {
                    debug!("Service {} is healthy: {}", service_name, health.message);
                }
                HealthStatus::NotServing => {
                    warn!("Service {} is degraded: {}", service_name, health.message);
                }
                HealthStatus::Unknown => {
                    warn!("Service {} status unknown: {}", service_name, health.message);
                }
            }
        }

        // Attempt recovery if needed
        let recovered = self.monitoring_system.attempt_recovery().await?;
        if !recovered.is_empty() {
            info!("Auto-recovered {} services: {:?}", recovered.len(), recovered);
        }

        Ok(())
    }

    /// Export all metrics for external monitoring
    pub async fn export_all_metrics(&self) -> DaemonResult<serde_json::Value> {
        let service_stats = self.get_all_stats().await;
        let monitoring_metrics = self.monitoring_system.export_metrics().await?;

        let mut all_metrics = serde_json::Map::new();

        // Add service-specific metrics
        all_metrics.insert("service_stats".to_string(), serde_json::to_value(service_stats)?);

        // Add monitoring system metrics
        for (key, value) in monitoring_metrics {
            all_metrics.insert(key, value);
        }

        Ok(serde_json::Value::Object(all_metrics))
    }
}

/// Combined service statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct ServiceStats {
    pub system_status: HealthStatus,
    pub system_message: String,
    pub document_processing: DocumentProcessingStats,
    pub search: SearchStats,
    pub memory: MemoryStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_document_processor_monitor() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = DocumentProcessorMonitor::new(&monitoring_system).await.unwrap();

        // Test processing workflow
        monitor.start_processing("doc1").await;
        monitor.update_queue_size(5).await;

        let stats = monitor.get_processing_stats().await;
        assert_eq!(stats.in_progress, 1);
        assert_eq!(stats.queue_size, 5);

        // Complete processing
        monitor.complete_processing("doc1", Duration::from_millis(500)).await;

        let stats = monitor.get_processing_stats().await;
        assert_eq!(stats.in_progress, 0);
        assert_eq!(stats.total_processed, 1);
    }

    #[tokio::test]
    async fn test_document_processor_overload() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = DocumentProcessorMonitor::new(&monitoring_system).await.unwrap();

        // Start many documents to trigger overload
        for i in 0..12 {
            monitor.start_processing(&format!("doc{}", i)).await;
        }

        let health = monitor.health_monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::NotServing);
        assert!(health.message.contains("overload"));
    }

    #[tokio::test]
    async fn test_search_service_monitor() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = SearchServiceMonitor::new(&monitoring_system).await.unwrap();

        // Test search workflow
        let tracker = monitor.start_search("test query").await;
        monitor.record_cache_hit().await;

        tracker.complete_success(10).await;

        let stats = monitor.get_search_stats().await;
        assert!(stats.cache_hit_rate > 0.0);
        assert_eq!(stats.concurrent_searches, 0);
    }

    #[tokio::test]
    async fn test_search_service_cache_stats() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = SearchServiceMonitor::new(&monitoring_system).await.unwrap();

        // Record cache hits and misses
        for _ in 0..8 {
            monitor.record_cache_hit().await;
        }
        for _ in 0..2 {
            monitor.record_cache_miss().await;
        }

        let stats = monitor.get_search_stats().await;
        assert_eq!(stats.cache_hit_rate, 0.8); // 8 hits out of 10 total
    }

    #[tokio::test]
    async fn test_memory_service_monitor() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = MemoryServiceMonitor::new(&monitoring_system).await.unwrap();

        // Test memory operation workflow
        let tracker = monitor.start_operation("add_document").await;
        monitor.update_memory_utilization(75.0).await;
        monitor.update_collection_count(5).await;
        monitor.update_document_count(1000).await;

        tracker.complete_success().await;

        let stats = monitor.get_memory_stats().await;
        assert_eq!(stats.memory_utilization, 75.0);
        assert_eq!(stats.active_collections, 5);
        assert_eq!(stats.total_documents, 1000);
        assert_eq!(stats.operations_in_progress, 0);
    }

    #[tokio::test]
    async fn test_memory_service_high_utilization_alert() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = MemoryServiceMonitor::new(&monitoring_system).await.unwrap();

        // Trigger high memory utilization alert
        monitor.update_memory_utilization(95.0).await;

        let health = monitor.health_monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::NotServing);
        assert!(health.message.contains("memory utilization"));

        // Test recovery
        monitor.update_memory_utilization(70.0).await;

        let health = monitor.health_monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::Serving);
        assert!(health.message.contains("Recovered"));
    }

    #[tokio::test]
    async fn test_service_monitors_creation() {
        let monitors = ServiceMonitors::new().await.unwrap();

        let stats = monitors.get_all_stats().await;
        assert_eq!(stats.system_status, HealthStatus::Serving);
        assert!(!stats.system_message.is_empty());
    }

    #[tokio::test]
    async fn test_service_monitors_health_check() {
        let monitors = ServiceMonitors::new().await.unwrap();

        let result = monitors.perform_health_check().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_monitors_export_metrics() {
        let monitors = ServiceMonitors::new().await.unwrap();

        // Record some activity
        monitors.document_processor.start_processing("test").await;
        monitors.document_processor.complete_processing("test", Duration::from_millis(100)).await;

        let search_tracker = monitors.search_service.start_search("query").await;
        search_tracker.complete_success(5).await;

        let mem_tracker = monitors.memory_service.start_operation("test").await;
        mem_tracker.complete_success().await;

        let metrics = monitors.export_all_metrics().await.unwrap();
        assert!(metrics.is_object());

        let obj = metrics.as_object().unwrap();
        assert!(obj.contains_key("service_stats"));
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let monitor = Arc::new(SearchServiceMonitor::new(&monitoring_system).await.unwrap());

        let mut handles = vec![];

        // Start multiple concurrent searches
        for i in 0..5 {
            let monitor_clone: Arc<SearchServiceMonitor> = Arc::clone(&monitor);
            let handle = tokio::spawn(async move {
                let tracker = monitor_clone.start_search(&format!("query {}", i)).await;
                tokio::time::sleep(Duration::from_millis(10)).await;
                tracker.complete_success(i * 2).await;
            });
            handles.push(handle);
        }

        // Wait for all searches to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let stats = monitor.get_search_stats().await;
        assert_eq!(stats.concurrent_searches, 0);
    }
}