//! CLI command modules
//!
//! All commands are always available (no feature flags).

// Service & Status
pub mod service;
pub mod status;

// Diagnostics
pub mod debug;

// Content Management
pub mod library;
pub mod memory;
pub mod project;
pub mod scratch;

// Search & Queue
pub mod graph;
pub mod queue;
pub mod search;
pub mod stats;
pub mod tags;

// Language Support
pub mod language;

// System Administration
pub mod admin;
pub mod config_cmd;
pub mod backup;
pub mod collections;
pub mod ingest;
pub mod restore;
pub mod update;
pub mod watch;

// Recovery
pub mod recover_state;

// Shared utilities
pub mod qdrant_helpers;

// Diagnostics & Setup
pub mod hooks;
pub mod init;
pub mod man;

// Benchmarking
pub mod benchmark;
