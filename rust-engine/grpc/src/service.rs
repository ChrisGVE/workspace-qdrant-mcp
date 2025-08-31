//! gRPC service implementation
//!
//! This module contains the actual service implementations

use workspace_qdrant_core::DocumentProcessor;

/// Ingestion service implementation
pub struct IngestionService {
    processor: DocumentProcessor,
}

impl IngestionService {
    pub fn new() -> Self {
        Self {
            processor: DocumentProcessor::new(),
        }
    }
}

impl Default for IngestionService {
    fn default() -> Self {
        Self::new()
    }
}

// Future: Implement actual gRPC service traits here when proto definitions are ready
