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
pub mod cli_profiles;
pub mod config;
pub mod constants;
pub mod document_id;
pub mod domain;
pub mod duration_fmt;
pub mod env_expand;
pub mod error;
pub mod exclusion;
pub mod git;
pub mod git_url;
pub mod guard;
pub mod handle;
pub mod hashing;
pub mod language_registry;
pub mod lsp_detection;
pub mod nlp;
pub mod paths;
pub mod payloads;
pub mod project_id;
pub mod qdrant_endpoint;
pub mod queue_types;
pub mod rules_legacy;
pub mod schema;
pub mod search;
pub mod timestamp_fmt;
pub mod timestamps;
pub mod yaml_defaults;
