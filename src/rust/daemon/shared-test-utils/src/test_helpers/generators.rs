//! Test data generators: IDs, text, vectors, documents, and environment helpers

use std::env;
use std::path::{Path, PathBuf};

use crate::TestResult;

/// Generate a unique test identifier
pub fn generate_test_id() -> String {
    format!("test_{}", uuid::Uuid::new_v4().simple())
}

/// Generate a unique collection name for testing
pub fn generate_test_collection() -> String {
    format!("test_collection_{}", uuid::Uuid::new_v4().simple())
}

/// Check if we're running in CI environment
pub fn is_ci() -> bool {
    env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok()
}

/// Check if a specific environment variable is set for testing
pub fn env_test_flag(flag: &str) -> bool {
    env::var(flag)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Skip test if not running in CI
pub fn require_ci() -> TestResult<()> {
    if !is_ci() {
        return Err("Test requires CI environment".into());
    }
    Ok(())
}

/// Skip test if running in CI (for local-only tests)
pub fn skip_in_ci() -> TestResult<()> {
    if is_ci() {
        return Err("Test skipped in CI environment".into());
    }
    Ok(())
}

/// Get test data directory path
pub fn test_data_dir() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    Path::new(&manifest_dir).join("test_data")
}

/// Get test resource file path
pub fn test_resource(filename: &str) -> PathBuf {
    test_data_dir().join(filename)
}

/// Check if a test resource file exists
pub fn has_test_resource(filename: &str) -> bool {
    test_resource(filename).exists()
}

/// Read test resource file as string
pub async fn read_test_resource(filename: &str) -> TestResult<String> {
    let path = test_resource(filename);
    if !path.exists() {
        return Err(format!("Test resource not found: {}", filename).into());
    }
    Ok(tokio::fs::read_to_string(path).await?)
}

/// Read test resource file as bytes
pub async fn read_test_resource_bytes(filename: &str) -> TestResult<Vec<u8>> {
    let path = test_resource(filename);
    if !path.exists() {
        return Err(format!("Test resource not found: {}", filename).into());
    }
    Ok(tokio::fs::read(path).await?)
}

/// Test data generator for stress testing
pub fn generate_large_text(size_kb: usize) -> String {
    let base_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
    let target_size = size_kb * 1024;
    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str(base_text);
    }

    result.truncate(target_size);
    result
}

/// Generate test vectors for embedding tests
pub fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i + j) as f32 * 0.1).sin())
                .collect()
        })
        .collect()
}

/// Generate test documents with varying content
pub fn generate_test_documents(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            format!(
                "Test document #{}\n\nThis is a test document with unique content.\nDocument index: {}\nGenerated for testing purposes.\n\nContent varies to ensure different embeddings.",
                i + 1,
                i
            )
        })
        .collect()
}
