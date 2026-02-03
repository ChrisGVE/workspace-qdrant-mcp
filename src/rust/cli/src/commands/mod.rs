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

// Search & Queue
pub mod queue;
pub mod search;

// Language Support
pub mod grammar;
pub mod language;
pub mod lsp;

// System Administration
pub mod admin;
pub mod backup;
pub mod ingest;
pub mod update;
pub mod watch;

// Diagnostics & Setup
pub mod init;
pub mod wizard;
