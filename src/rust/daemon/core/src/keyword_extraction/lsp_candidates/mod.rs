//! LSP symbol-based candidate extraction.
//!
//! Extracts high-quality keyword candidates from code structure:
//! - Symbol normalization (camelCase/snake_case splitting)
//! - Import/dependency detection
//! - Symbol context string generation for embedding

mod imports;
mod merge;
mod normalization;
mod types;

// Re-export all public items so callers use the same paths as before.
pub use imports::extract_import_candidates;
pub use merge::merge_candidates;
pub use normalization::normalize_identifier;
pub use types::{CandidateSource, LspCandidate, LspCandidateConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword_extraction::lexical_candidates::LexicalCandidate;
    use normalization::strip_suffix;

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
