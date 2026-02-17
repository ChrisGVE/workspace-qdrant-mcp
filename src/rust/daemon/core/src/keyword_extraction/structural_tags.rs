//! Structural tag extraction for code files.
//!
//! Auto-derives structural tags from file metadata:
//! - Language (from tree-sitter detection)
//! - Framework (from import/use statements)
//! - Build system (from manifest files)
//! - Layer (from path patterns)

use std::path::Path;

use super::tag_selector::{SelectedTag, TagType};

/// Extract structural tags from a code file.
///
/// # Arguments
/// * `file_path` - Path to the source file
/// * `source` - File content as string
/// * `language` - Language identifier from tree-sitter (e.g., "rust", "python")
pub fn extract_structural_tags(
    file_path: &Path,
    source: &str,
    language: Option<&str>,
) -> Vec<SelectedTag> {
    let mut tags = Vec::new();

    // Language tag
    if let Some(lang) = language {
        tags.push(make_structural_tag(&format!("language:{}", lang)));
    }

    // Framework tags from imports
    if let Some(lang) = language {
        let frameworks = detect_frameworks(source, lang);
        for fw in frameworks {
            tags.push(make_structural_tag(&format!("framework:{}", fw)));
        }
    }

    // Build system tag from manifest
    if let Some(build) = detect_build_system(file_path) {
        tags.push(make_structural_tag(&format!("build:{}", build)));
    }

    // Layer tag from path
    if let Some(layer) = detect_layer(file_path) {
        tags.push(make_structural_tag(&format!("layer:{}", layer)));
    }

    tags
}

fn make_structural_tag(phrase: &str) -> SelectedTag {
    SelectedTag {
        phrase: phrase.to_string(),
        tag_type: TagType::Structural,
        score: 1.0,
        diversity_score: 1.0,
        semantic_score: 1.0,
        ngram_size: 1,
    }
}

/// Detect frameworks from import/use statements.
fn detect_frameworks(source: &str, language: &str) -> Vec<&'static str> {
    let mut found = Vec::new();

    match language {
        "rust" => {
            for (pattern, name) in RUST_FRAMEWORKS {
                if source.contains(pattern) {
                    found.push(*name);
                }
            }
        }
        "python" => {
            for (pattern, name) in PYTHON_FRAMEWORKS {
                if source.contains(pattern) {
                    found.push(*name);
                }
            }
        }
        "javascript" | "typescript" | "tsx" | "jsx" => {
            for (pattern, name) in JS_FRAMEWORKS {
                if source.contains(pattern) {
                    found.push(*name);
                }
            }
        }
        "go" => {
            for (pattern, name) in GO_FRAMEWORKS {
                if source.contains(pattern) {
                    found.push(*name);
                }
            }
        }
        "java" => {
            for (pattern, name) in JAVA_FRAMEWORKS {
                if source.contains(pattern) {
                    found.push(*name);
                }
            }
        }
        _ => {}
    }

    found
}

// Framework detection patterns: (import pattern, framework name)
const RUST_FRAMEWORKS: &[(&str, &str)] = &[
    ("use tokio", "tokio"),
    ("use axum", "axum"),
    ("use actix", "actix"),
    ("use serde", "serde"),
    ("use diesel", "diesel"),
    ("use sqlx", "sqlx"),
    ("use tonic", "tonic"),
    ("use warp", "warp"),
    ("use reqwest", "reqwest"),
    ("use clap", "clap"),
    ("use tracing", "tracing"),
    ("use hyper", "hyper"),
    ("use tower", "tower"),
    ("use qdrant_client", "qdrant"),
    ("use fastembed", "fastembed"),
];

const PYTHON_FRAMEWORKS: &[(&str, &str)] = &[
    ("import django", "django"),
    ("from django", "django"),
    ("import flask", "flask"),
    ("from flask", "flask"),
    ("import fastapi", "fastapi"),
    ("from fastapi", "fastapi"),
    ("import pandas", "pandas"),
    ("import numpy", "numpy"),
    ("import torch", "pytorch"),
    ("import tensorflow", "tensorflow"),
    ("import sqlalchemy", "sqlalchemy"),
    ("import pytest", "pytest"),
    ("import asyncio", "asyncio"),
    ("import aiohttp", "aiohttp"),
];

const JS_FRAMEWORKS: &[(&str, &str)] = &[
    ("from 'react'", "react"),
    ("from \"react\"", "react"),
    ("require('react')", "react"),
    ("from 'vue'", "vue"),
    ("from \"vue\"", "vue"),
    ("from 'express'", "express"),
    ("require('express')", "express"),
    ("from 'next", "nextjs"),
    ("from '@angular", "angular"),
    ("from 'svelte'", "svelte"),
    ("from '@nestjs", "nestjs"),
    ("from 'axios'", "axios"),
    ("from 'prisma'", "prisma"),
    ("@modelcontextprotocol", "mcp"),
];

const GO_FRAMEWORKS: &[(&str, &str)] = &[
    ("\"net/http\"", "net-http"),
    ("\"github.com/gin-gonic/gin\"", "gin"),
    ("\"github.com/gorilla/mux\"", "gorilla"),
    ("\"github.com/labstack/echo\"", "echo"),
    ("\"gorm.io/gorm\"", "gorm"),
    ("\"google.golang.org/grpc\"", "grpc"),
];

const JAVA_FRAMEWORKS: &[(&str, &str)] = &[
    ("import org.springframework", "spring"),
    ("import javax.persistence", "jpa"),
    ("import io.quarkus", "quarkus"),
    ("import jakarta.", "jakarta"),
    ("import org.junit", "junit"),
    ("import org.hibernate", "hibernate"),
];

/// Detect build system from file path context.
///
/// Checks if the file resides in a project with a known build manifest.
fn detect_build_system(file_path: &Path) -> Option<&'static str> {
    let path_str = file_path.to_string_lossy();
    let file_name = file_path.file_name()?.to_str()?;

    // Check if the file itself is a build manifest
    match file_name {
        "Cargo.toml" => return Some("cargo"),
        "package.json" => return Some("npm"),
        "pyproject.toml" => return Some("poetry"),
        "setup.py" | "setup.cfg" => return Some("setuptools"),
        "go.mod" => return Some("go-mod"),
        "pom.xml" => return Some("maven"),
        "build.gradle" | "build.gradle.kts" => return Some("gradle"),
        "Makefile" | "makefile" | "GNUmakefile" => return Some("make"),
        "CMakeLists.txt" => return Some("cmake"),
        "Dockerfile" => return Some("docker"),
        _ => {}
    }

    // Infer build system from path components
    if path_str.contains("/src/rust/") || path_str.contains(".rs") {
        return Some("cargo");
    }

    None
}

/// Detect architectural layer from file path patterns.
fn detect_layer(file_path: &Path) -> Option<&'static str> {
    let path_str = file_path.to_string_lossy().to_lowercase();
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Test layer
    if path_str.contains("/test")
        || path_str.contains("/tests/")
        || path_str.contains("/spec/")
        || path_str.contains("/bench")
        || file_name.starts_with("test_")
        || file_name.ends_with("_test.rs")
        || file_name.ends_with("_test.go")
        || file_name.ends_with(".test.ts")
        || file_name.ends_with(".test.js")
        || file_name.ends_with(".spec.ts")
        || file_name.ends_with(".spec.js")
    {
        return Some("test");
    }

    // API layer
    if path_str.contains("/api/")
        || path_str.contains("/routes/")
        || path_str.contains("/controllers/")
        || path_str.contains("/handlers/")
        || path_str.contains("/endpoints/")
        || path_str.contains("/grpc/")
    {
        return Some("api");
    }

    // Data access layer
    if path_str.contains("/models/")
        || path_str.contains("/schema/")
        || path_str.contains("/migrations/")
        || path_str.contains("/storage/")
        || path_str.contains("/repository/")
        || path_str.contains("/dao/")
        || file_name.contains("storage")
        || file_name.contains("repository")
    {
        return Some("data-access");
    }

    // Configuration layer
    if path_str.contains("/config/")
        || path_str.contains("/settings/")
        || file_name.contains("config")
    {
        return Some("config");
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_tag() {
        let tags = extract_structural_tags(Path::new("src/main.rs"), "", Some("rust"));
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"language:rust"));
    }

    #[test]
    fn test_framework_rust_tokio() {
        let source = r#"
use tokio::runtime::Runtime;
use serde::Serialize;

fn main() {
    let rt = Runtime::new().unwrap();
}
"#;
        let tags = extract_structural_tags(Path::new("src/main.rs"), source, Some("rust"));
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"framework:tokio"), "Should detect tokio: {:?}", phrases);
        assert!(phrases.contains(&"framework:serde"), "Should detect serde: {:?}", phrases);
    }

    #[test]
    fn test_framework_python_django() {
        let source = "from django.db import models\nimport pandas as pd\n";
        let tags = extract_structural_tags(Path::new("app/models.py"), source, Some("python"));
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"framework:django"));
        assert!(phrases.contains(&"framework:pandas"));
    }

    #[test]
    fn test_framework_js_react() {
        let source = "import React from 'react';\nimport axios from 'axios';\n";
        let tags =
            extract_structural_tags(Path::new("src/App.tsx"), source, Some("typescript"));
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"framework:react"));
        assert!(phrases.contains(&"framework:axios"));
    }

    #[test]
    fn test_build_system_cargo() {
        let tags = extract_structural_tags(Path::new("Cargo.toml"), "", None);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"build:cargo"));
    }

    #[test]
    fn test_build_system_npm() {
        let tags = extract_structural_tags(Path::new("package.json"), "", None);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"build:npm"));
    }

    #[test]
    fn test_layer_test() {
        let tags = extract_structural_tags(
            Path::new("src/tests/test_auth.py"),
            "",
            Some("python"),
        );
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"layer:test"), "Should detect test layer: {:?}", phrases);
    }

    #[test]
    fn test_layer_api() {
        let tags = extract_structural_tags(
            Path::new("src/api/routes.rs"),
            "",
            Some("rust"),
        );
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"layer:api"), "Should detect api layer: {:?}", phrases);
    }

    #[test]
    fn test_layer_grpc() {
        let tags = extract_structural_tags(
            Path::new("src/grpc/service.rs"),
            "",
            Some("rust"),
        );
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"layer:api"), "gRPC should map to api layer: {:?}", phrases);
    }

    #[test]
    fn test_layer_data_access() {
        let tags = extract_structural_tags(
            Path::new("src/storage/database.rs"),
            "",
            Some("rust"),
        );
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(
            phrases.contains(&"layer:data-access"),
            "Should detect data-access layer: {:?}",
            phrases
        );
    }

    #[test]
    fn test_no_language_no_source() {
        let tags = extract_structural_tags(Path::new("README.md"), "", None);
        // Should have no tags (no language, no build system, no layer)
        assert!(tags.is_empty(), "README.md should have no structural tags: {:?}", tags);
    }

    #[test]
    fn test_all_tags_structural_type() {
        let source = "use tokio::runtime::Runtime;\n";
        let tags = extract_structural_tags(Path::new("src/main.rs"), source, Some("rust"));
        for tag in &tags {
            assert_eq!(
                tag.tag_type,
                TagType::Structural,
                "All structural tags should have Structural type"
            );
        }
    }

    #[test]
    fn test_test_file_patterns() {
        // Rust test
        let tags = extract_structural_tags(Path::new("src/auth_test.rs"), "", Some("rust"));
        assert!(tags.iter().any(|t| t.phrase == "layer:test"));

        // JS spec
        let tags = extract_structural_tags(
            Path::new("src/auth.spec.ts"),
            "",
            Some("typescript"),
        );
        assert!(tags.iter().any(|t| t.phrase == "layer:test"));

        // Go test
        let tags = extract_structural_tags(Path::new("auth_test.go"), "", Some("go"));
        assert!(tags.iter().any(|t| t.phrase == "layer:test"));
    }

    #[test]
    fn test_java_spring() {
        let source = "import org.springframework.boot.SpringApplication;\n";
        let tags = extract_structural_tags(
            Path::new("src/main/java/App.java"),
            source,
            Some("java"),
        );
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"framework:spring"));
    }

    #[test]
    fn test_go_gin() {
        let source = "import \"github.com/gin-gonic/gin\"\n";
        let tags = extract_structural_tags(Path::new("main.go"), source, Some("go"));
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"framework:gin"));
    }
}
