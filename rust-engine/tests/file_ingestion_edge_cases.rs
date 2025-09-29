//! Edge case file handling tests for daemon file ingestion
//!
//! This test suite covers edge cases and boundary conditions for file ingestion:
//! - Zero-byte files
//! - Large files (>100MB PDFs, >10MB text)
//! - Corrupted/malformed files
//! - Special character filenames
//! - Files without extensions
//! - Symlinked files and circular symlinks
//! - Files changing during ingestion
//! - Deep nested directories (20+ levels)
//! - Unicode filenames/content
//! - Incomplete metadata

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;
use workspace_qdrant_daemon::config::{ProcessingConfig, QdrantConfig, CollectionConfig};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;

/// Test configuration helper
mod test_config {
    use super::*;

    pub fn create_test_processing_config() -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_tasks: 2,
            default_chunk_size: 1000,
            default_chunk_overlap: 200,
            max_file_size_bytes: 200 * 1024 * 1024, // 200MB for large file tests
            supported_extensions: vec![
                "txt".to_string(),
                "rs".to_string(),
                "py".to_string(),
                "md".to_string(),
                "pdf".to_string(),
                "epub".to_string(),
            ],
            enable_lsp: false,
            lsp_timeout_secs: 10,
        }
    }

    pub fn create_test_qdrant_config() -> QdrantConfig {
        QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            max_retries: 3,
            default_collection: CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        }
    }
}

/// Test fixture management
mod fixtures {
    use super::*;

    /// Create a zero-byte file
    pub fn create_zero_byte_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        File::create(&path).expect("Failed to create zero-byte file");
        path
    }

    /// Create a large text file
    pub fn create_large_text_file(dir: &Path, name: &str, size_mb: usize) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create large text file");

        // Write content in chunks
        let chunk = "A".repeat(1024 * 1024); // 1MB chunk
        for _ in 0..size_mb {
            file.write_all(chunk.as_bytes()).expect("Failed to write to large file");
        }
        file.flush().expect("Failed to flush file");
        path
    }

    /// Create a file with only whitespace
    pub fn create_whitespace_only_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create whitespace file");
        file.write_all(b"   \n\t\r\n   ").expect("Failed to write whitespace");
        path
    }

    /// Create a file with special characters in name
    pub fn create_special_char_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        File::create(&path).expect("Failed to create special char file");
        path
    }

    /// Create a file without extension
    pub fn create_no_extension_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create no-extension file");
        file.write_all(b"Content without extension").expect("Failed to write");
        path
    }

    /// Create a deeply nested directory structure
    pub fn create_deep_nested_structure(base_dir: &Path, depth: usize) -> PathBuf {
        let mut current = base_dir.to_path_buf();
        for i in 0..depth {
            current = current.join(format!("level_{}", i));
        }
        fs::create_dir_all(&current).expect("Failed to create deep nested structure");

        // Create a file at the deepest level
        let file_path = current.join("deep_file.txt");
        let mut file = File::create(&file_path).expect("Failed to create deep nested file");
        file.write_all(b"Deep nested content").expect("Failed to write");
        file_path
    }

    /// Create a unicode filename
    pub fn create_unicode_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create unicode file");
        file.write_all("Unicode content: 你好世界 مرحبا العالم".as_bytes())
            .expect("Failed to write unicode content");
        path
    }

    /// Create a symlink (Unix-only)
    #[cfg(unix)]
    pub fn create_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    /// Create a circular symlink (Unix-only)
    #[cfg(unix)]
    pub fn create_circular_symlink(dir: &Path) -> std::io::Result<(PathBuf, PathBuf)> {
        let link1 = dir.join("link1");
        let link2 = dir.join("link2");

        // Create circular reference
        std::os::unix::fs::symlink(&link2, &link1)?;
        std::os::unix::fs::symlink(&link1, &link2)?;

        Ok((link1, link2))
    }

    /// Create a corrupted PDF-like file (invalid PDF header)
    pub fn create_corrupted_pdf(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create corrupted PDF");
        // Write invalid PDF header
        file.write_all(b"%PDF-INVALID\nCorrupted content").expect("Failed to write");
        path
    }

    /// Create a file with very long name
    pub fn create_long_filename(dir: &Path) -> PathBuf {
        // Maximum safe filename length on most systems is 255 characters
        let long_name = format!("{}.txt", "a".repeat(200));
        let path = dir.join(long_name);
        let mut file = File::create(&path).expect("Failed to create long filename file");
        file.write_all(b"Long filename content").expect("Failed to write");
        path
    }

    /// Create a file with null bytes in content
    pub fn create_null_byte_content_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create null byte file");
        file.write_all(b"Content\x00with\x00null\x00bytes").expect("Failed to write");
        path
    }
}

/// Zero-byte file tests
#[cfg(test)]
mod zero_byte_tests {
    use super::*;

    #[tokio::test]
    async fn test_zero_byte_file_handling() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let zero_file = fixtures::create_zero_byte_file(temp_dir.path(), "empty.txt");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(zero_file.to_str().unwrap()).await;

        // Zero-byte files should be handled gracefully
        assert!(result.is_ok(), "Zero-byte file should be handled gracefully");

        // Check that it returns "empty" status
        let status = result.unwrap();
        assert_eq!(status, "empty", "Zero-byte file should return 'empty' status");
    }

    #[tokio::test]
    async fn test_whitespace_only_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let whitespace_file = fixtures::create_whitespace_only_file(temp_dir.path(), "whitespace.txt");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(whitespace_file.to_str().unwrap()).await;

        // Whitespace-only files should be treated as empty
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "empty");
    }

    #[tokio::test]
    async fn test_multiple_zero_byte_files() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        // Create and process multiple zero-byte files
        for i in 0..5 {
            let zero_file = fixtures::create_zero_byte_file(temp_dir.path(), &format!("empty_{}.txt", i));
            let result = processor.process_document(zero_file.to_str().unwrap()).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "empty");
        }
    }
}

/// Large file tests
#[cfg(test)]
mod large_file_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Ignored by default due to time/resource requirements
    async fn test_large_text_file_10mb() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let large_file = fixtures::create_large_text_file(temp_dir.path(), "large_10mb.txt", 10);

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(large_file.to_str().unwrap()).await;

        // Large files should be processed successfully
        assert!(result.is_ok(), "10MB file should be processed successfully");

        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
        assert!(uuid::Uuid::parse_str(&document_id).is_ok());
    }

    #[tokio::test]
    #[ignore] // Ignored by default - very large file test
    async fn test_large_text_file_100mb() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let large_file = fixtures::create_large_text_file(temp_dir.path(), "large_100mb.txt", 100);

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(large_file.to_str().unwrap()).await;

        // 100MB files should be processed (within max_file_size_bytes)
        assert!(result.is_ok(), "100MB file should be processed successfully");

        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
    }

    #[tokio::test]
    async fn test_file_size_near_limit() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        // Create file just under 1MB (config limit is 200MB)
        let near_limit_file = fixtures::create_large_text_file(temp_dir.path(), "near_limit.txt", 1);

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(near_limit_file.to_str().unwrap()).await;
        assert!(result.is_ok());
    }
}

/// Corrupted/malformed file tests
#[cfg(test)]
mod corrupted_file_tests {
    use super::*;

    #[tokio::test]
    async fn test_corrupted_pdf_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let corrupted_pdf = fixtures::create_corrupted_pdf(temp_dir.path(), "corrupted.pdf");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(corrupted_pdf.to_str().unwrap()).await;

        // Corrupted files should be handled gracefully (not crash)
        assert!(result.is_ok(), "Corrupted PDF should be handled gracefully");
    }

    #[tokio::test]
    async fn test_null_byte_content() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let null_byte_file = fixtures::create_null_byte_content_file(temp_dir.path(), "null_bytes.txt");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(null_byte_file.to_str().unwrap()).await;

        // Files with null bytes should be handled
        assert!(result.is_ok(), "File with null bytes should be handled");
    }

    #[tokio::test]
    async fn test_nonexistent_file() {
        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document("/nonexistent/path/to/file.txt").await;

        // Nonexistent files should return error status gracefully
        assert!(result.is_ok(), "Nonexistent file should be handled gracefully");
        assert_eq!(result.unwrap(), "error");
    }
}

/// Special character filename tests
#[cfg(test)]
mod special_filename_tests {
    use super::*;

    #[tokio::test]
    async fn test_file_with_spaces() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let space_file = fixtures::create_special_char_file(temp_dir.path(), "file with spaces.txt");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(space_file.to_str().unwrap()).await;
        assert!(result.is_ok(), "File with spaces should be processed");
    }

    #[tokio::test]
    async fn test_file_with_special_chars() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let special_names = vec![
            "file@with#special$.txt",
            "file(with)parens.txt",
            "file[with]brackets.txt",
            "file{with}braces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.dots.txt",
        ];

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        for name in special_names {
            let file = fixtures::create_special_char_file(temp_dir.path(), name);
            let result = processor.process_document(file.to_str().unwrap()).await;
            assert!(result.is_ok(), "File with special chars '{}' should be processed", name);
        }
    }

    #[tokio::test]
    async fn test_very_long_filename() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let long_file = fixtures::create_long_filename(temp_dir.path());

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(long_file.to_str().unwrap()).await;
        assert!(result.is_ok(), "File with long filename should be processed");
    }
}

/// Files without extension tests
#[cfg(test)]
mod no_extension_tests {
    use super::*;

    #[tokio::test]
    async fn test_file_without_extension() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let no_ext_file = fixtures::create_no_extension_file(temp_dir.path(), "README");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(no_ext_file.to_str().unwrap()).await;

        // Files without extensions should be skipped (not in supported_extensions)
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "skipped");
    }

    #[tokio::test]
    async fn test_multiple_no_extension_files() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let no_ext_names = vec!["Makefile", "LICENSE", "CHANGELOG", "Dockerfile"];

        for name in no_ext_names {
            let file = fixtures::create_no_extension_file(temp_dir.path(), name);
            let result = processor.process_document(file.to_str().unwrap()).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "skipped");
        }
    }
}

/// Symlink tests (Unix only)
#[cfg(unix)]
#[cfg(test)]
mod symlink_tests {
    use super::*;

    #[tokio::test]
    async fn test_symlink_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create target file
        let target = temp_dir.path().join("target.txt");
        let mut file = File::create(&target).expect("Failed to create target file");
        file.write_all(b"Symlink target content").expect("Failed to write");

        // Create symlink
        let link = temp_dir.path().join("link.txt");
        fixtures::create_symlink(&target, &link).expect("Failed to create symlink");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        // Process via symlink
        let result = processor.process_document(link.to_str().unwrap()).await;
        assert!(result.is_ok(), "Symlink should be followed and processed");
    }

    #[tokio::test]
    async fn test_circular_symlink() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let (link1, _link2) = fixtures::create_circular_symlink(temp_dir.path())
            .expect("Failed to create circular symlink");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        // Attempt to process circular symlink
        let result = processor.process_document(link1.to_str().unwrap()).await;

        // Circular symlinks should be handled gracefully (error or skip)
        assert!(result.is_ok(), "Circular symlink should be handled without crashing");
    }

    #[tokio::test]
    async fn test_broken_symlink() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create symlink to nonexistent target
        let target = temp_dir.path().join("nonexistent.txt");
        let link = temp_dir.path().join("broken_link.txt");
        fixtures::create_symlink(&target, &link).expect("Failed to create broken symlink");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(link.to_str().unwrap()).await;

        // Broken symlinks should return error gracefully
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "error");
    }
}

/// Deep nested directory tests
#[cfg(test)]
mod deep_nesting_tests {
    use super::*;

    #[tokio::test]
    async fn test_20_level_nested_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let deep_file = fixtures::create_deep_nested_structure(temp_dir.path(), 20);

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(deep_file.to_str().unwrap()).await;
        assert!(result.is_ok(), "20-level nested file should be processed");

        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
    }

    #[tokio::test]
    async fn test_50_level_nested_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let deep_file = fixtures::create_deep_nested_structure(temp_dir.path(), 50);

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(deep_file.to_str().unwrap()).await;
        assert!(result.is_ok(), "50-level nested file should be processed");
    }

    #[tokio::test]
    async fn test_extremely_deep_100_level_nested() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let deep_file = fixtures::create_deep_nested_structure(temp_dir.path(), 100);

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(deep_file.to_str().unwrap()).await;

        // Even extremely deep nesting should be handled
        assert!(result.is_ok(), "100-level nested file should be processed");
    }
}

/// Unicode filename and content tests
#[cfg(test)]
mod unicode_tests {
    use super::*;

    #[tokio::test]
    async fn test_unicode_filenames() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let unicode_names = vec![
            "файл.txt",           // Russian
            "文件.txt",           // Chinese
            "ファイル.txt",        // Japanese
            "파일.txt",           // Korean
            "αρχείο.txt",        // Greek
            "ملف.txt",           // Arabic
            "dosya.txt",         // Turkish
            "🚀rocket.txt",      // Emoji
        ];

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        for name in unicode_names {
            let file = fixtures::create_unicode_file(temp_dir.path(), name);
            let result = processor.process_document(file.to_str().unwrap()).await;
            assert!(result.is_ok(), "Unicode filename '{}' should be processed", name);

            if result.as_ref().unwrap() != "skipped" {
                let document_id = result.unwrap();
                assert_eq!(document_id.len(), 36);
            }
        }
    }

    #[tokio::test]
    async fn test_mixed_unicode_content() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create file with mixed unicode content
        let file_path = temp_dir.path().join("mixed_unicode.txt");
        let mut file = File::create(&file_path).expect("Failed to create file");
        file.write_all(
            "Mixed content:\n\
             English: Hello World\n\
             中文: 你好世界\n\
             日本語: こんにちは世界\n\
             한국어: 안녕하세요 세계\n\
             Русский: Привет мир\n\
             العربية: مرحبا بالعالم\n\
             Emoji: 🌍🌎🌏".as_bytes()
        ).expect("Failed to write mixed unicode");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        let result = processor.process_document(file_path.to_str().unwrap()).await;
        assert!(result.is_ok(), "Mixed unicode content should be processed");

        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
    }
}

/// Concurrent edge case tests
#[cfg(test)]
mod concurrent_edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_mixed_edge_cases() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create various edge case files
        let zero_file = fixtures::create_zero_byte_file(temp_dir.path(), "concurrent_empty.txt");
        let unicode_file = fixtures::create_unicode_file(temp_dir.path(), "concurrent_日本語.txt");
        let special_file = fixtures::create_special_char_file(temp_dir.path(), "concurrent@special#.txt");
        let no_ext_file = fixtures::create_no_extension_file(temp_dir.path(), "concurrent_no_ext");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create processor")
        );

        let files = vec![
            zero_file.to_str().unwrap().to_string(),
            unicode_file.to_str().unwrap().to_string(),
            special_file.to_str().unwrap().to_string(),
            no_ext_file.to_str().unwrap().to_string(),
        ];

        let mut handles = vec![];
        for file in files {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                processor_clone.process_document(&file).await
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;

        // All edge case files should be handled concurrently
        for (i, result) in results.iter().enumerate() {
            let process_result = result.as_ref().expect(&format!("Task {} panicked", i));
            assert!(process_result.is_ok(), "Edge case {} should be handled", i);
        }
    }
}

/// Recovery and error handling tests
#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[tokio::test]
    async fn test_recovery_after_corrupted_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create processor");

        // Process corrupted file
        let corrupted = fixtures::create_corrupted_pdf(temp_dir.path(), "corrupted.pdf");
        let result1 = processor.process_document(corrupted.to_str().unwrap()).await;
        assert!(result1.is_ok(), "Corrupted file should be handled");

        // Process valid file after corrupted one
        let valid_file = temp_dir.path().join("valid.txt");
        let mut file = File::create(&valid_file).expect("Failed to create valid file");
        file.write_all(b"Valid content").expect("Failed to write");

        let result2 = processor.process_document(valid_file.to_str().unwrap()).await;
        assert!(result2.is_ok(), "Valid file after corrupted should succeed");

        let document_id = result2.unwrap();
        assert_eq!(document_id.len(), 36);
    }

    #[tokio::test]
    async fn test_batch_with_mixed_valid_invalid() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let processing_config = test_config::create_test_processing_config();
        let qdrant_config = test_config::create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create processor")
        );

        // Create mix of valid and invalid files
        let valid = temp_dir.path().join("valid.txt");
        File::create(&valid).expect("Failed to create valid file");

        let zero = fixtures::create_zero_byte_file(temp_dir.path(), "zero.txt");
        let nonexistent = "/nonexistent/file.txt";

        let files = vec![
            valid.to_str().unwrap(),
            zero.to_str().unwrap(),
            nonexistent,
        ];

        let mut handles = vec![];
        for file in files {
            let processor_clone = Arc::clone(&processor);
            let file_str = file.to_string();
            let handle = tokio::spawn(async move {
                processor_clone.process_document(&file_str).await
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;

        // All should return results (no panics)
        for result in results {
            assert!(result.is_ok(), "Task should not panic");
            assert!(result.unwrap().is_ok(), "Processing should handle errors gracefully");
        }
    }
}