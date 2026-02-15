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
pub mod queue;
pub mod search;
pub mod stats;

// Language Support
pub mod language;

// System Administration
pub mod admin;
pub mod backup;
pub mod collections;
pub mod ingest;
pub mod restore;
pub mod update;
pub mod watch;

// Diagnostics & Setup
pub mod hooks;
pub mod init;
