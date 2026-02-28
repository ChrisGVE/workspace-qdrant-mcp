//! Comprehensive tests for the LSP system
//!
//! These tests validate the LSP server detection, lifecycle management,
//! communication, and state management components.
//!
//! Organized into focused submodules:
//! - `language_tests`: Language enum, extensions, LSP support flags
//! - `config_tests`: LspConfig creation, validation, file I/O
//! - `detection_tests`: LspServerDetector and server discovery
//! - `communication_tests`: JSON-RPC message parsing and client lifecycle
//! - `project_manager_tests`: LanguageServerManager, enrichment, metrics

#[cfg(test)]
mod language_tests;
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod detection_tests;
#[cfg(test)]
mod communication_tests;
#[cfg(test)]
mod project_manager_tests;
