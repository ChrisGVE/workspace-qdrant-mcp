//! Python bindings for workspace-qdrant-mcp ingestion engine
//!
//! This crate provides PyO3 bindings that allow Python code to interface
//! with the Rust ingestion engine.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use workspace_qdrant_core::{DocumentProcessor, DocumentResult, ProcessingError};
use workspace_qdrant_grpc::{GrpcServer, ServerConfig};

/// Python-accessible ingestion engine
#[pyclass]
pub struct RustIngestionEngine {
    processor: Arc<DocumentProcessor>,
    grpc_server: Option<Arc<Mutex<GrpcServer>>>,
    grpc_port: u16,
}

#[pymethods]
impl RustIngestionEngine {
    #[new]
    fn new(config: HashMap<String, PyObject>) -> PyResult<Self> {
        // Initialize with default configuration for now
        // In future: parse config dict into proper configuration
        Ok(Self {
            processor: Arc::new(DocumentProcessor::new()),
            grpc_server: None,
            grpc_port: 0,
        })
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        // Placeholder for engine startup
        // In future: start actual gRPC server and return port
        self.grpc_port = 50051; // Default port for testing
        Ok(())
    }

    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        // Placeholder for engine shutdown
        // In future: gracefully stop gRPC server
        self.grpc_port = 0;
        Ok(())
    }

    fn grpc_port(&self) -> u16 {
        self.grpc_port
    }

    fn get_state(&self) -> PyResult<String> {
        Ok("RUNNING".to_string())
    }

    fn process_document(&self, file_path: String, collection: String) -> PyResult<String> {
        // Placeholder for document processing
        // In future: actual async processing with proper error handling
        Ok(format!(
            "Processed {} into collection {}",
            file_path, collection
        ))
    }
}

/// Python-accessible health check function
#[pyfunction]
fn health_check() -> bool {
    workspace_qdrant_core::health_check() && workspace_qdrant_grpc::health_check()
}

/// Module initialization for Python
#[pymodule]
fn workspace_qdrant_python_bindings(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustIngestionEngine>()?;
    m.add_function(wrap_pyfunction!(health_check, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_health_check() {
        assert!(health_check());
    }

    #[test]
    fn test_engine_creation() {
        let config = HashMap::new();
        let engine = RustIngestionEngine::new(config);
        assert!(engine.is_ok());
    }
}
