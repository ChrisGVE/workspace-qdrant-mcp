//! Configuration management
//!
//! This module will contain the configuration management implementation

use serde::{Deserialize, Serialize};

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub chunk_size: usize,
    pub enable_lsp: bool,
}

impl EngineConfig {
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            enable_lsp: true,
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::new()
    }
}
