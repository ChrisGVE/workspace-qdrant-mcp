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
pub mod language;

// System Administration
pub mod backup;
pub mod ingest;
pub mod restore;
pub mod update;
pub mod watch;

// Diagnostics & Setup
pub mod hooks;
pub mod init;
