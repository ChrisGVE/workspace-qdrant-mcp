//! gRPC service implementations

// TODO: These services need to be updated to match the proto
// pub mod document_processor;
// pub mod search_service;
// pub mod memory_service;

pub mod system_service;

// pub use document_processor::DocumentProcessorImpl;
// pub use search_service::SearchServiceImpl;
// pub use memory_service::MemoryServiceImpl;
pub use system_service::SystemServiceImpl;

// Test-only modules (disabled until services are updated)
// #[cfg(test)]
// pub mod discovery_service;
