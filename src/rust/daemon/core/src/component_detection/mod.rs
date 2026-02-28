//! Component auto-detection from workspace definition files.
//!
//! Parses Cargo.toml `[workspace]` members and package.json `workspaces`
//! to derive dot-separated hierarchical component names.
//!
//! Examples:
//!   Cargo.toml member `"daemon/core"`  → component `"daemon.core"`
//!   package.json workspace `"packages/ui"` → component `"packages.ui"`

mod detection;
mod persistence;

pub use detection::{
    assign_component, component_matches_filter, detect_components, file_matches_component,
    parse_cargo_members, path_to_component_id,
};
pub use persistence::{backfill_components, load_components, persist_components, BackfillStats};

use std::collections::HashMap;

/// Detected workspace component.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// Dot-separated component ID, e.g. "daemon.core"
    pub id: String,
    /// Base directory relative to project root, e.g. "daemon/core"
    pub base_path: String,
    /// Glob patterns matching files in this component
    pub patterns: Vec<String>,
    /// Detection source
    pub source: ComponentSource,
}

/// How the component was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentSource {
    Cargo,
    Npm,
    Directory,
}

impl ComponentSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cargo => "cargo",
            Self::Npm => "npm",
            Self::Directory => "directory",
        }
    }
}

/// Map from component ID to its info.
pub type ComponentMap = HashMap<String, ComponentInfo>;

#[cfg(test)]
mod tests;
