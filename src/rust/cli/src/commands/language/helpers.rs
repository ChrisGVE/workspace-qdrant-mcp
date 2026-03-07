//! Shared helper functions for language command submodules

use workspace_qdrant_core::language_registry::providers::bundled::BundledProvider;
use workspace_qdrant_core::language_registry::types::LanguageDefinition;
use workspace_qdrant_core::lsp::detection::editor_paths::{
    find_lsp_binary, DetectionSource, LspDetectionResult,
};

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

/// Detect available language servers using PATH and editor-managed locations.
///
/// Returns a list of (language name, server name, resolved path, detection source).
pub fn detect_available_servers() -> Vec<(String, String, String, DetectionSource)> {
    let defs = load_definitions();
    let mut found = Vec::new();

    for def in &defs {
        if def.lsp_servers.is_empty() {
            continue;
        }
        for server in &def.lsp_servers {
            if let Some(result) = find_lsp_binary(&server.binary) {
                found.push((
                    def.language.clone(),
                    server.name.clone(),
                    result.path.display().to_string(),
                    result.source,
                ));
                break; // Only report first found for each language
            }
        }
    }

    found
}

/// Find an executable on PATH and editor-managed locations.
///
/// Returns the path string if found, or None.
pub fn which_cmd(name: &str) -> Option<String> {
    find_lsp_binary(name).map(|r| r.path.display().to_string())
}

/// Find an executable with full detection result (path + source).
pub fn which_cmd_detailed(name: &str) -> Option<LspDetectionResult> {
    find_lsp_binary(name)
}
