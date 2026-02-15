//! Shared configuration, types, and utilities for workspace-qdrant-mcp
//!
//! This crate provides canonical implementations used by both the daemon and CLI:
//! - Configuration file search paths
//! - Database and config directory resolution
//! - Environment variable expansion
//! - YAML-derived default configuration values
//! - Queue type enums (ItemType, QueueOperation, QueueStatus)
//! - Hashing utilities (idempotency keys, content/file hashing)
//! - Project ID calculation and disambiguation
//! - Collection name and default constants
//! - Queue payload structs
//! - NLP tokenization utilities

pub mod classification;
pub mod constants;
pub mod env_expand;
pub mod hashing;
pub mod nlp;
pub mod paths;
pub mod payloads;
pub mod project_id;
pub mod queue_types;
pub mod schema;
pub mod timestamps;
pub mod yaml_defaults;
