//! LSP symbol-based candidate extraction.
//!
//! Extracts high-quality keyword candidates from code structure:
//! - Symbol normalization (camelCase/snake_case splitting)
//! - Import/dependency detection
//! - Symbol context string generation for embedding

use super::lexical_candidates::LexicalCandidate;

/// A candidate extracted from LSP or code structure analysis.
#[derive(Debug, Clone)]
pub struct LspCandidate {
    /// Normalized phrase (for semantic search)
    pub phrase: String,
    /// Original identifier (for exact search)
    pub identifier: String,
    /// Source of the candidate
    pub source: CandidateSource,
    /// Priority boost (LSP candidates rank higher)
    pub priority_boost: f64,
}

/// Source of an LSP candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateSource {
    /// Extracted from public symbol definition
    PublicSymbol,
    /// Extracted from import/use statement
    Import,
    /// Extracted from frequently referenced name
    Reference,
}

/// Configuration for LSP candidate extraction.
#[derive(Debug, Clone)]
pub struct LspCandidateConfig {
    /// Boost factor for LSP candidates in combined scoring
    pub priority_boost: f64,
    /// Minimum identifier length to consider
    pub min_identifier_len: usize,
    /// Suffixes to strip from identifiers
    pub strip_suffixes: Vec<String>,
}

impl Default for LspCandidateConfig {
    fn default() -> Self {
        Self {
            priority_boost: 1.5,
            min_identifier_len: 3,
            strip_suffixes: vec![
                "Impl".to_string(),
                "Manager".to_string(),
                "Handler".to_string(),
                "Helper".to_string(),
                "Util".to_string(),
                "Utils".to_string(),
                "Factory".to_string(),
                "Builder".to_string(),
            ],
        }
    }
}

/// Normalize a symbol identifier into a readable phrase.
///
/// Splits camelCase and snake_case into separate words.
/// Examples:
/// - `PrimeSieve` → `prime sieve`
/// - `find_n_primes` → `find n primes`
/// - `HTMLParser` → `html parser`
/// - `getHTTPResponse` → `get http response`
pub fn normalize_identifier(ident: &str) -> String {
    let mut words = Vec::new();
    let mut current = String::new();

    // Handle snake_case first: split on underscores
    for part in ident.split('_') {
        if part.is_empty() {
            continue;
        }

        // Split camelCase within each part
        let chars: Vec<char> = part.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if i > 0 && ch.is_uppercase() {
                // Check for acronym: consecutive uppercase (e.g., HTTP)
                let prev_upper = chars[i - 1].is_uppercase();
                let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();

                if !prev_upper || next_lower {
                    // Start of new word
                    if !current.is_empty() {
                        words.push(current.to_lowercase());
                        current.clear();
                    }
                }
            }
            current.push(ch);
        }
        if !current.is_empty() {
            words.push(current.to_lowercase());
            current.clear();
        }
    }

    words.join(" ")
}

/// Strip known trivial suffixes from an identifier.
fn strip_suffix(ident: &str, config: &LspCandidateConfig) -> String {
    let mut result = ident.to_string();
    for suffix in &config.strip_suffixes {
        if result.ends_with(suffix.as_str()) && result.len() > suffix.len() {
            result.truncate(result.len() - suffix.len());
            break;
        }
    }
    result
}

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

fn extract_rust_imports(source: &str, candidates: &mut Vec<LspCandidate>, config: &LspCandidateConfig) {
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
            let inner = path[brace_start + 1..]
                .trim_end_matches('}')
                .trim();
            for item in inner.split(',') {
                let item = item.trim();
                if !item.is_empty() && item.len() >= config.min_identifier_len {
                    let ident = item.split("::").last().unwrap_or(item);
                    add_candidate(candidates, ident, prefix, CandidateSource::Import, config);
                }
            }
            // Also add the crate/module name from brace imports
            let module = prefix.trim_end_matches("::").split("::").next().unwrap_or("");
            if !module.is_empty() && module.len() >= config.min_identifier_len && module != "std" && module != "core" {
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
            if !module.is_empty() && module.len() >= config.min_identifier_len && module != "std" && module != "core" {
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

fn extract_python_imports(source: &str, candidates: &mut Vec<LspCandidate>, config: &LspCandidateConfig) {
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
            let module = trimmed.trim_start_matches("import ").trim().split(" as ").next().unwrap_or("").trim();
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

fn extract_js_imports(source: &str, candidates: &mut Vec<LspCandidate>, config: &LspCandidateConfig) {
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

fn extract_go_imports(source: &str, candidates: &mut Vec<LspCandidate>, config: &LspCandidateConfig) {
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

/// Merge LSP candidates with lexical candidates.
///
/// LSP candidates get a priority boost. Duplicates (by normalized phrase)
/// are deduplicated, keeping the higher-scored version.
pub fn merge_candidates(
    lexical: Vec<LexicalCandidate>,
    lsp: &[LspCandidate],
    boost: f64,
) -> Vec<LexicalCandidate> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut merged = Vec::new();

    // Add LSP candidates first (higher priority)
    for c in lsp {
        let key = c.phrase.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        merged.push(LexicalCandidate {
            phrase: c.phrase.clone(),
            raw_tf: 1,
            tf_score: boost,
            ngram_size: c.phrase.split(' ').count() as u8,
        });
    }

    // Add lexical candidates that aren't already present
    for c in lexical {
        let key = c.phrase.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        merged.push(c);
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_camel_case() {
        assert_eq!(normalize_identifier("PrimeSieve"), "prime sieve");
        assert_eq!(normalize_identifier("findPrimes"), "find primes");
        assert_eq!(normalize_identifier("getHTTPResponse"), "get http response");
        assert_eq!(normalize_identifier("HTMLParser"), "html parser");
    }

    #[test]
    fn test_normalize_snake_case() {
        assert_eq!(normalize_identifier("find_n_primes"), "find n primes");
        assert_eq!(normalize_identifier("async_runtime"), "async runtime");
        assert_eq!(normalize_identifier("__init__"), "init");
    }

    #[test]
    fn test_normalize_mixed() {
        assert_eq!(normalize_identifier("get_HTTPClient"), "get http client");
    }

    #[test]
    fn test_strip_suffix() {
        let config = LspCandidateConfig::default();
        assert_eq!(strip_suffix("PrimeSieveImpl", &config), "PrimeSieve");
        assert_eq!(strip_suffix("AuthManager", &config), "Auth");
        assert_eq!(strip_suffix("RequestHandler", &config), "Request");
        // Don't strip if it would leave nothing
        assert_eq!(strip_suffix("Impl", &config), "Impl");
    }

    #[test]
    fn test_extract_rust_imports() {
        let source = r#"
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
"#;
        let config = LspCandidateConfig::default();
        let candidates = extract_import_candidates(source, "rust", &config);

        let phrases: Vec<&str> = candidates.iter().map(|c| c.phrase.as_str()).collect();
        assert!(phrases.contains(&"runtime"), "Should extract Runtime: {:?}", phrases);
        assert!(phrases.contains(&"serialize"), "Should extract Serialize: {:?}", phrases);
        assert!(phrases.contains(&"deserialize"), "Should extract Deserialize: {:?}", phrases);
        assert!(phrases.contains(&"hash map"), "Should extract HashMap: {:?}", phrases);
        assert!(phrases.contains(&"tokio"), "Should extract tokio module: {:?}", phrases);
        assert!(phrases.contains(&"serde"), "Should extract serde module: {:?}", phrases);
    }

    #[test]
    fn test_extract_python_imports() {
        let source = r#"
from django.db import models
import pandas as pd
from typing import Optional, List
"#;
        let config = LspCandidateConfig::default();
        let candidates = extract_import_candidates(source, "python", &config);

        let phrases: Vec<&str> = candidates.iter().map(|c| c.phrase.as_str()).collect();
        assert!(phrases.contains(&"django"), "Should extract django: {:?}", phrases);
        assert!(phrases.contains(&"models"), "Should extract models: {:?}", phrases);
        assert!(phrases.contains(&"pandas"), "Should extract pandas: {:?}", phrases);
        assert!(phrases.contains(&"optional"), "Should extract Optional: {:?}", phrases);
    }

    #[test]
    fn test_extract_js_imports() {
        let source = r#"
import React from 'react';
import { useState, useEffect } from 'react';
const express = require('express');
"#;
        let config = LspCandidateConfig::default();
        let candidates = extract_import_candidates(source, "typescript", &config);

        let phrases: Vec<&str> = candidates.iter().map(|c| c.phrase.as_str()).collect();
        assert!(phrases.contains(&"react"), "Should extract react: {:?}", phrases);
        assert!(phrases.contains(&"express"), "Should extract express: {:?}", phrases);
        assert!(
            phrases.contains(&"use state"),
            "Should extract useState: {:?}",
            phrases
        );
        assert!(
            phrases.contains(&"use effect"),
            "Should extract useEffect: {:?}",
            phrases
        );
    }

    #[test]
    fn test_extract_go_imports() {
        let source = r#"
import (
    "fmt"
    "github.com/gin-gonic/gin"
    "net/http"
)
"#;
        let config = LspCandidateConfig::default();
        let candidates = extract_import_candidates(source, "go", &config);

        let phrases: Vec<&str> = candidates.iter().map(|c| c.phrase.as_str()).collect();
        assert!(phrases.contains(&"fmt"), "Should extract fmt: {:?}", phrases);
        assert!(phrases.contains(&"gin"), "Should extract gin: {:?}", phrases);
        assert!(phrases.contains(&"http"), "Should extract http: {:?}", phrases);
    }

    #[test]
    fn test_merge_candidates_dedup() {
        let lexical = vec![
            LexicalCandidate {
                phrase: "vector search".to_string(),
                raw_tf: 5,
                tf_score: 2.0,
                ngram_size: 2,
            },
            LexicalCandidate {
                phrase: "database".to_string(),
                raw_tf: 3,
                tf_score: 1.5,
                ngram_size: 1,
            },
        ];
        let lsp = vec![LspCandidate {
            phrase: "vector search".to_string(), // duplicate
            identifier: "VectorSearch".to_string(),
            source: CandidateSource::Import,
            priority_boost: 1.5,
        }];

        let merged = merge_candidates(lexical, &lsp, 1.5);
        // Should have 2 entries (deduped)
        assert_eq!(merged.len(), 2, "Should dedup: {:?}", merged);
        // LSP version should win (comes first)
        assert_eq!(merged[0].phrase, "vector search");
        assert!((merged[0].tf_score - 1.5).abs() < 1e-6, "LSP boost should be applied");
    }

    #[test]
    fn test_candidate_source_types() {
        assert_ne!(CandidateSource::PublicSymbol, CandidateSource::Import);
        assert_ne!(CandidateSource::Import, CandidateSource::Reference);
    }

    #[test]
    fn test_config_defaults() {
        let config = LspCandidateConfig::default();
        assert!((config.priority_boost - 1.5).abs() < 1e-6);
        assert_eq!(config.min_identifier_len, 3);
        assert!(!config.strip_suffixes.is_empty());
    }

    #[test]
    fn test_extract_empty_source() {
        let config = LspCandidateConfig::default();
        let candidates = extract_import_candidates("", "rust", &config);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_short_identifiers_filtered() {
        let source = "use std::io;\n";
        let config = LspCandidateConfig::default();
        let candidates = extract_import_candidates(source, "rust", &config);
        // "io" is only 2 chars, should be filtered
        assert!(
            !candidates.iter().any(|c| c.identifier == "io"),
            "Short identifiers should be filtered: {:?}",
            candidates
        );
    }
}
