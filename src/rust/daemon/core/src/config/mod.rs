//! Configuration management
//!
//! This module contains configuration management for the priority processing engine.
//! Domain-specific configs are organized into submodules and re-exported here.

mod code_intelligence;
mod concept;
mod embedding;
mod graph_rag;
mod ingestion;
mod integration;
mod narrative;
mod observability;
mod processing;
pub mod queue_health;
mod resource_limits;
mod url_ingestion;

// Config machinery (WI-a2): split from the former monolithic mod.rs and wired
// to the shared `wqm_common::config` primitives. Replaces the deleted
// `unified_config` module.
mod build;
mod env;
mod error;
mod loader;
mod types;
mod validate;

// Re-export all public types for backward compatibility
pub use code_intelligence::{GrammarConfig, LspSettings};
pub use concept::ConceptConfig;
pub use embedding::{EmbeddingSettings, KeywordEmbedderConfig};
pub use graph_rag::GraphRagConfig;
pub use ingestion::{AutoIngestionConfig, IngestionLimitsConfig};
pub use integration::{GitConfig, UpdateChannel, UpdatesConfig};
pub use narrative::NarrativeConfig;
pub use observability::{
    LoggingConfig, MetricsConfig, MonitoringConfig, ObservabilityConfig, OtlpExportConfig,
    OtlpProtocol, PrometheusExportConfig, TelemetryConfig, TracingConfig,
};
pub use processing::{QueueProcessorSettings, StartupConfig};
pub use queue_health::QueueHealthConfig;
pub use resource_limits::{detect_physical_cores, ResourceLimitsConfig};
pub use url_ingestion::UrlIngestionConfig;

// Config view + loader + error (WI-a2).
pub use env::apply_env_overrides;
pub use error::ConfigError;
pub use loader::{config_search_paths, load_config};
pub use types::{Config, DaemonConfig, DaemonEndpointConfig};

#[cfg(test)]
mod tests;
