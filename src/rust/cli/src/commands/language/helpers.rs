//! Shared helper functions for language command submodules

use std::process::Command;

use workspace_qdrant_core::language_registry::providers::bundled::BundledProvider;
use workspace_qdrant_core::language_registry::types::LanguageDefinition;

/// Load all language definitions from the bundled registry.
pub fn load_definitions() -> Vec<LanguageDefinition> {
    match BundledProvider::new() {
        Ok(p) => p.definitions().to_vec(),
        Err(e) => {
            eprintln!("Warning: failed to load language registry: {e}");
            Vec::new()
        }
    }
}

/// Get a language definition by ID or alias.
pub fn find_language(language: &str) -> Option<LanguageDefinition> {
    let defs = load_definitions();
    let normalized = language.to_lowercase();

    // Direct ID match
    if let Some(def) = defs.iter().find(|d| d.id() == normalized) {
        return Some(def.clone());
    }

    // Alias match
    defs.iter()
        .find(|d| d.aliases.iter().any(|a| a.to_lowercase() == normalized))
        .cloned()
}

/// Detect available language servers on PATH using registry data.
///
/// Returns a list of (language name, server name, resolved path).
pub fn detect_available_servers() -> Vec<(String, String, String)> {
    let defs = load_definitions();
    let mut found = Vec::new();

    for def in &defs {
        if def.lsp_servers.is_empty() {
            continue;
        }
        for server in &def.lsp_servers {
            if let Some(path) = which_cmd(&server.binary) {
                found.push((def.language.clone(), server.name.clone(), path));
                break; // Only report first found for each language
            }
        }
    }

    found
}

/// Find an executable on PATH using `which` then the `which` crate as fallback.
pub fn which_cmd(name: &str) -> Option<String> {
    match Command::new("which").arg(name).output() {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
        _ => {}
    }

    // Fallback: use which crate
    match which::which(name) {
        Ok(path) => Some(path.display().to_string()),
        Err(_) => None,
    }
}
