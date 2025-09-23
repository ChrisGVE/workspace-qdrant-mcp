//! gRPC service implementations

pub mod document_processor;
pub mod search_service;
pub mod memory_service;
pub mod system_service;
pub mod discovery_service;

pub use document_processor::DocumentProcessorImpl;
pub use search_service::SearchServiceImpl;
pub use memory_service::MemoryServiceImpl;
pub use system_service::SystemServiceImpl;
pub use discovery_service::ServiceDiscoveryImpl;