//! Test fixtures and sample data generators

use std::path::PathBuf;
use tempfile::{NamedTempFile, TempDir};
use tokio::fs;
use tokio::io::AsyncWriteExt;
// use uuid::Uuid; // Unused

use crate::config::TEST_EMBEDDING_DIM;
use crate::TestResult;

/// Test document content generator
pub struct DocumentFixtures;

impl DocumentFixtures {
    /// Generate a sample Markdown document
    pub fn markdown_content() -> String {
        include_str!("fixture_data/sample_markdown.md").to_string()
    }

    /// Generate a sample Python code file
    pub fn python_content() -> String {
        include_str!("fixture_data/sample_python.py").to_string()
    }

    /// Generate a sample Rust code file
    pub fn rust_content() -> String {
        include_str!("fixture_data/sample_rust.rs").to_string()
    }

    /// Generate a sample JSON configuration file
    pub fn json_config() -> String {
        serde_json::json!({
            "app_name": "workspace-qdrant-mcp",
            "version": "0.2.1",
            "config": {
                "qdrant": {
                    "url": "http://localhost:6333",
                    "api_key": null,
                    "timeout_ms": 5000,
                    "max_retries": 3
                },
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384,
                    "batch_size": 32,
                    "cache_size": 1000
                },
                "processing": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "max_workers": 4,
                    "queue_size": 100
                },
                "logging": {
                    "level": "info",
                    "format": "json",
                    "file": null
                }
            },
            "collections": [
                "default",
                "documents",
                "code"
            ],
            "features": {
                "auto_embedding": true,
                "file_watching": true,
                "hybrid_search": true,
                "caching": true
            }
        }).to_string()
    }
}

/// File system test utilities
pub struct TempFileFixtures;

impl TempFileFixtures {
    /// Create a temporary file with given content and extension
    pub async fn create_temp_file(content: &str, extension: &str) -> TestResult<NamedTempFile> {
        let temp_file = NamedTempFile::with_suffix(format!(".{}", extension))?;

        let mut file = fs::File::create(temp_file.path()).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;

        Ok(temp_file)
    }

    /// Create a temporary directory with test files
    pub async fn create_temp_project() -> TestResult<(TempDir, Vec<PathBuf>)> {
        let temp_dir = TempDir::new()?;
        let mut file_paths = Vec::new();

        // Create a basic project structure
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&src_dir).await?;

        let docs_dir = temp_dir.path().join("docs");
        fs::create_dir(&docs_dir).await?;

        // Create sample files
        let files = vec![
            ("README.md", DocumentFixtures::markdown_content()),
            ("src/main.py", DocumentFixtures::python_content()),
            ("src/lib.rs", DocumentFixtures::rust_content()),
            ("config.json", DocumentFixtures::json_config()),
        ];

        for (relative_path, content) in files {
            let file_path = temp_dir.path().join(relative_path);

            // Ensure parent directory exists
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).await?;
            }

            fs::write(&file_path, content).await?;
            file_paths.push(file_path);
        }

        Ok((temp_dir, file_paths))
    }

    /// Create a temporary file with random content for stress testing
    pub async fn create_large_temp_file(size_kb: usize) -> TestResult<NamedTempFile> {
        let temp_file = NamedTempFile::with_suffix(".txt")?;

        let chunk = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(10);
        let chunks_needed = (size_kb * 1024) / chunk.len() + 1;

        let mut file = fs::File::create(temp_file.path()).await?;
        for _ in 0..chunks_needed {
            file.write_all(chunk.as_bytes()).await?;
        }
        file.flush().await?;

        Ok(temp_file)
    }
}

/// Vector and embedding test utilities
pub struct EmbeddingFixtures;

impl EmbeddingFixtures {
    /// Generate a random embedding vector
    pub fn random_embedding() -> Vec<f32> {
        use std::f32::consts::PI;

        (0..TEST_EMBEDDING_DIM)
            .map(|i| (i as f32 * PI / TEST_EMBEDDING_DIM as f32).sin() * 0.5)
            .collect()
    }

    /// Generate embeddings for test documents
    pub fn test_document_embeddings() -> Vec<(String, Vec<f32>)> {
        vec![
            ("Machine learning is a subset of artificial intelligence".to_string(), Self::random_embedding()),
            ("Natural language processing enables computers to understand text".to_string(), Self::random_embedding()),
            ("Vector databases store high-dimensional data efficiently".to_string(), Self::random_embedding()),
            ("Semantic search finds meaning rather than exact matches".to_string(), Self::random_embedding()),
            ("Document chunking splits text into manageable pieces".to_string(), Self::random_embedding()),
        ]
    }

    /// Generate sparse vector for BM25 testing
    pub fn test_sparse_vector() -> (Vec<u32>, Vec<f32>) {
        let indices = vec![0, 5, 12, 23, 45, 67, 89, 123];
        let values = vec![0.8, 0.6, 0.9, 0.4, 0.7, 0.3, 0.5, 0.2];
        (indices, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temp_file_creation() -> TestResult {
        let content = "test content";
        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let read_content = fs::read_to_string(temp_file.path()).await?;
        assert_eq!(read_content, content);

        Ok(())
    }

    #[tokio::test]
    async fn test_temp_project_creation() -> TestResult {
        let (_temp_dir, file_paths) = TempFileFixtures::create_temp_project().await?;

        assert_eq!(file_paths.len(), 4);

        for file_path in &file_paths {
            assert!(file_path.exists());
        }

        Ok(())
    }

    #[test]
    fn test_embedding_generation() {
        let embedding = EmbeddingFixtures::random_embedding();
        assert_eq!(embedding.len(), TEST_EMBEDDING_DIM);
    }

    #[test]
    fn test_document_fixtures() {
        let markdown = DocumentFixtures::markdown_content();
        assert!(markdown.contains("# Test Document"));
        assert!(markdown.contains("```rust"));

        let python = DocumentFixtures::python_content();
        assert!(python.contains("import asyncio"));
        assert!(python.contains("async def"));

        let rust = DocumentFixtures::rust_content();
        assert!(rust.contains("pub struct"));
        assert!(rust.contains("impl"));
    }
}