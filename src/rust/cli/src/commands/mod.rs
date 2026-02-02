//! CLI command modules
//!
//! Organized by implementation phase:
//! - Phase 1 (HIGH): service, admin, status, library, queue, lsp
//! - Phase 2 (MEDIUM): search, ingest, backup, memory, language, project
//! - Phase 3 (LOW): init, help, wizard

// Phase 1 - HIGH priority (always available)
pub mod admin;
pub mod grammar;
pub mod library;
pub mod lsp;
pub mod queue;
pub mod service;
pub mod status;
pub mod update;
pub mod watch;

// Phase 2 - MEDIUM priority (behind feature flag)
#[cfg(feature = "phase2")]
pub mod backup;
#[cfg(feature = "phase2")]
pub mod ingest;
#[cfg(feature = "phase2")]
pub mod language;
#[cfg(feature = "phase2")]
pub mod memory;
#[cfg(feature = "phase2")]
pub mod project;
#[cfg(feature = "phase2")]
pub mod search;

// Phase 3 - LOW priority (behind feature flag)
#[cfg(feature = "phase3")]
pub mod help;
#[cfg(feature = "phase3")]
pub mod init;
#[cfg(feature = "phase3")]
pub mod wizard;
