//! Shared helper functions for language command submodules

use wqm_common::language_registry::types::LanguageDefinition;
use wqm_common::language_registry::RegistryReader;
use wqm_common::lsp_detection::{find_lsp_binary, DetectionSource, LspDetectionResult};

/// Load all language definitions from the bundled registry.
///
/// Pure, daemon-free: reads the embedded `language_registry.yaml` via
/// `wqm_common::language_registry::RegistryReader` — no `workspace-qdrant-core`.
pub fn load_definitions() -> Vec<LanguageDefinition> {
    match RegistryReader::bundled() {
        Ok(defs) => defs,
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

/// Build a `file-extension (no dot, lowercase) → language id` map from the
/// bundled registry, replacing the core `detect_language` engine call for
/// project file-extension scanning.
pub fn build_extension_map(defs: &[LanguageDefinition]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for def in defs {
        let id = def.id();
        for ext in &def.extensions {
            let key = ext.trim_start_matches('.').to_lowercase();
            if !key.is_empty() {
                map.entry(key).or_insert_with(|| id.clone());
            }
        }
    }
    map
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

// ── Daemon grammar-status (best-effort) ──────────────────────────────────────
//
// The grammar engine + on-disk cache live in the daemon (WI-e1/e2, #82). Read
// commands enrich registry display with live cache status when the daemon is
// reachable, but still render registry data offline (status falls back to
// "not available"). Mutations (`ts-install`, `ts-remove`, `warm`) require the
// daemon explicitly via `crate::grpc::ensure_daemon_available`.

use std::collections::{HashMap, HashSet};

/// Best-effort `language id → grammar-status` map from the daemon's
/// `LanguageService::ListGrammars`. Empty when the daemon is unreachable.
pub async fn try_grammar_status_map() -> HashMap<String, String> {
    try_grammar_cached_and_status().await.1
}

/// Best-effort `(cached language ids, language id → status)` from the daemon.
/// Both empty when the daemon is unreachable.
pub async fn try_grammar_cached_and_status() -> (HashSet<String>, HashMap<String, String>) {
    let Ok(mut client) = crate::grpc::connect_default().await else {
        return (HashSet::new(), HashMap::new());
    };
    match client.list_grammars().await {
        Ok(listed) => (
            listed.cached.into_iter().collect(),
            listed
                .known
                .into_iter()
                .map(|g| (g.language, g.status))
                .collect(),
        ),
        Err(_) => (HashSet::new(), HashMap::new()),
    }
}

/// Best-effort single-language grammar status string from the daemon.
/// Returns `"not_available"` when the daemon is unreachable or errors.
pub async fn try_grammar_status(language: &str) -> String {
    let Ok(mut client) = crate::grpc::connect_default().await else {
        return "not_available".to_string();
    };
    match client.query_language(language.to_string()).await {
        Ok(q) => q.grammar_status,
        Err(_) => "not_available".to_string(),
    }
}
