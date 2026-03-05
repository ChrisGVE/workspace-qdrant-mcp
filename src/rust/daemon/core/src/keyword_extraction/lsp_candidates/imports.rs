//! Import/use statement candidate extraction for multiple languages.

use super::normalization::{normalize_identifier, strip_suffix};
use super::types::{CandidateSource, LspCandidate, LspCandidateConfig};

/// Extract candidates from import/use statements in source code.
///
/// # Arguments
/// * `source` - File content
/// * `language` - Language identifier (e.g., "rust", "python")
/// * `config` - Extraction configuration
pub fn extract_import_candidates(
    source: &str,
    language: &str,
    config: &LspCandidateConfig,
) -> Vec<LspCandidate> {
    let mut candidates = Vec::new();

    match language {
        "rust" => extract_rust_imports(source, &mut candidates, config),
        "python" => extract_python_imports(source, &mut candidates, config),
        "javascript" | "typescript" | "tsx" | "jsx" => {
            extract_js_imports(source, &mut candidates, config)
        }
        "go" => extract_go_imports(source, &mut candidates, config),
        _ => {}
    }

    candidates
}

fn extract_rust_imports(
    source: &str,
    candidates: &mut Vec<LspCandidate>,
    config: &LspCandidateConfig,
) {
    for line in source.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("use ") {
            continue;
        }

        // Extract the path after "use "
        let path = trimmed
            .trim_start_matches("use ")
            .trim_end_matches(';')
            .trim();

        // Handle braces: use std::collections::{HashMap, HashSet}
        if let Some(brace_start) = path.find('{') {
            let prefix = &path[..brace_start];
            let inner = path[brace_start + 1..].trim_end_matches('}').trim();
            for item in inner.split(',') {
                let item = item.trim();
                if !item.is_empty() && item.len() >= config.min_identifier_len {
                    let ident = item.split("::").last().unwrap_or(item);
                    add_candidate(candidates, ident, prefix, CandidateSource::Import, config);
                }
            }
            // Also add the crate/module name from brace imports
            let module = prefix
                .trim_end_matches("::")
                .split("::")
                .next()
                .unwrap_or("");
            if !module.is_empty()
                && module.len() >= config.min_identifier_len
                && module != "std"
                && module != "core"
            {
                candidates.push(LspCandidate {
                    phrase: module.to_lowercase(),
                    identifier: module.to_string(),
                    source: CandidateSource::Import,
                    priority_boost: config.priority_boost,
                });
            }
        } else {
            // Single import: use tokio::runtime::Runtime
            let ident = path.split("::").last().unwrap_or(path);
            let module = path.split("::").next().unwrap_or("");
            if ident.len() >= config.min_identifier_len {
                add_candidate(candidates, ident, module, CandidateSource::Import, config);
            }
            // Also add the crate/module name itself
            if !module.is_empty()
                && module.len() >= config.min_identifier_len
                && module != "std"
                && module != "core"
            {
                candidates.push(LspCandidate {
                    phrase: module.to_lowercase(),
                    identifier: module.to_string(),
                    source: CandidateSource::Import,
                    priority_boost: config.priority_boost,
                });
            }
        }
    }
}

fn extract_python_imports(
    source: &str,
    candidates: &mut Vec<LspCandidate>,
    config: &LspCandidateConfig,
) {
    for line in source.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("from ") {
            // from django.db import models
            let parts: Vec<&str> = trimmed.splitn(2, " import ").collect();
            if parts.len() == 2 {
                let module = parts[0].trim_start_matches("from ").trim();
                let imports = parts[1].trim();

                // Add module
                let top_module = module.split('.').next().unwrap_or(module);
                if top_module.len() >= config.min_identifier_len {
                    candidates.push(LspCandidate {
                        phrase: top_module.to_lowercase(),
                        identifier: top_module.to_string(),
                        source: CandidateSource::Import,
                        priority_boost: config.priority_boost,
                    });
                }

                // Add imported names
                for item in imports.split(',') {
                    let item = item.trim().split(" as ").next().unwrap_or("").trim();
                    if item.len() >= config.min_identifier_len && item != "*" {
                        add_candidate(candidates, item, module, CandidateSource::Import, config);
                    }
                }
            }
        } else if trimmed.starts_with("import ") {
            // import os
            let module = trimmed
                .trim_start_matches("import ")
                .trim()
                .split(" as ")
                .next()
                .unwrap_or("")
                .trim();
            if module.len() >= config.min_identifier_len {
                candidates.push(LspCandidate {
                    phrase: module.to_lowercase().replace('.', " "),
                    identifier: module.to_string(),
                    source: CandidateSource::Import,
                    priority_boost: config.priority_boost,
                });
            }
        }
    }
}

fn extract_js_imports(
    source: &str,
    candidates: &mut Vec<LspCandidate>,
    config: &LspCandidateConfig,
) {
    for line in source.lines() {
        let trimmed = line.trim();

        // import X from 'module'
        // import { X, Y } from 'module'
        // const X = require('module')
        if trimmed.starts_with("import ") || trimmed.contains("require(") {
            // Extract module name from quotes
            if let Some(module) = extract_quoted_string(trimmed) {
                // Strip @ scope prefix for display
                let display = module
                    .trim_start_matches('@')
                    .split('/')
                    .next()
                    .unwrap_or(&module);
                if display.len() >= config.min_identifier_len {
                    candidates.push(LspCandidate {
                        phrase: display.to_lowercase(),
                        identifier: module.to_string(),
                        source: CandidateSource::Import,
                        priority_boost: config.priority_boost,
                    });
                }
            }

            // Extract named imports: { X, Y }
            if let Some(brace_start) = trimmed.find('{') {
                if let Some(brace_end) = trimmed.find('}') {
                    let inner = &trimmed[brace_start + 1..brace_end];
                    for item in inner.split(',') {
                        let item = item.trim().split(" as ").next().unwrap_or("").trim();
                        if item.len() >= config.min_identifier_len {
                            add_candidate(candidates, item, "", CandidateSource::Import, config);
                        }
                    }
                }
            }
        }
    }
}

fn extract_go_imports(
    source: &str,
    candidates: &mut Vec<LspCandidate>,
    config: &LspCandidateConfig,
) {
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(module) = extract_quoted_string(trimmed) {
            // Use last path segment as identifier
            let ident = module.split('/').last().unwrap_or(&module);
            if ident.len() >= config.min_identifier_len {
                candidates.push(LspCandidate {
                    phrase: ident.to_lowercase().replace('-', " "),
                    identifier: ident.to_string(),
                    source: CandidateSource::Import,
                    priority_boost: config.priority_boost,
                });
            }
        }
    }
}

fn extract_quoted_string(s: &str) -> Option<String> {
    // Try single quotes
    if let Some(start) = s.find('\'') {
        if let Some(end) = s[start + 1..].find('\'') {
            return Some(s[start + 1..start + 1 + end].to_string());
        }
    }
    // Try double quotes
    if let Some(start) = s.find('"') {
        if let Some(end) = s[start + 1..].find('"') {
            return Some(s[start + 1..start + 1 + end].to_string());
        }
    }
    None
}

fn add_candidate(
    candidates: &mut Vec<LspCandidate>,
    ident: &str,
    _context: &str,
    source: CandidateSource,
    config: &LspCandidateConfig,
) {
    let stripped = strip_suffix(ident, config);
    let phrase = normalize_identifier(&stripped);
    if phrase.len() >= config.min_identifier_len {
        candidates.push(LspCandidate {
            phrase,
            identifier: ident.to_string(),
            source,
            priority_boost: config.priority_boost,
        });
    }
}
