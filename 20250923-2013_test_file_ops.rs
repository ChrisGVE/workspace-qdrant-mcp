//! Test runner for async file operations
//! This file tests the file_ops module in isolation

use std::path::PathBuf;
use tempfile::tempdir;
use tokio;

#[path = "rust-engine/src/daemon/file_ops.rs"]
mod file_ops;

#[path = "rust-engine/src/error.rs"]
mod error;

#[path = "rust-engine/src/config.rs"]
mod config;

use file_ops::*;

#[tokio::test]
async fn test_async_file_processor_basic_operations() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.txt");
    let content = b"Hello, async world!";

    let processor = AsyncFileProcessor::default();

    // Test write
    processor.write_file(&file_path, content).await.unwrap();
    assert!(file_path.exists());

    // Test read
    let read_content = processor.read_file(&file_path).await.unwrap();
    assert_eq!(read_content, content);

    // Test hash
    let hash = processor.calculate_hash(&file_path).await.unwrap();
    assert_eq!(hash.len(), 64); // BLAKE3 hash length

    // Test file info
    let file_info = processor.validate_file(&file_path).await.unwrap();
    assert_eq!(file_info.size, content.len() as u64);
    assert!(file_info.is_file);
    assert!(!file_info.is_dir);

    println!("✅ Basic async file operations test passed");
}

#[tokio::test]
async fn test_async_file_processor_error_handling() {
    let temp_dir = tempdir().unwrap();
    let nonexistent_path = temp_dir.path().join("nonexistent.txt");

    let processor = AsyncFileProcessor::default();

    // Test read nonexistent file
    let result = processor.read_file(&nonexistent_path).await;
    assert!(result.is_err());

    // Test file too large
    let large_processor = AsyncFileProcessor::new(100, 1024, true);
    let large_file_path = temp_dir.path().join("large.txt");
    let large_content = vec![0u8; 200];

    std::fs::write(&large_file_path, &large_content).unwrap();
    let result = large_processor.read_file(&large_file_path).await;
    assert!(result.is_err());

    println!("✅ Error handling test passed");
}

#[tokio::test]
async fn test_async_file_processor_chunked_processing() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("chunks.txt");
    let content = b"0123456789abcdefghijklmnopqrstuvwxyz"; // 36 bytes

    std::fs::write(&file_path, content).unwrap();

    let processor = AsyncFileProcessor::default();
    let results = processor.process_chunks(&file_path, 10, |chunk, index| {
        Ok(format!("Chunk {}: {} bytes", index, chunk.len()))
    }).await.unwrap();

    assert_eq!(results.len(), 4);
    assert_eq!(results[0], "Chunk 0: 10 bytes");
    assert_eq!(results[3], "Chunk 3: 6 bytes");

    println!("✅ Chunked processing test passed");
}

#[tokio::test]
async fn test_async_file_processor_copy_operations() {
    let temp_dir = tempdir().unwrap();
    let src_path = temp_dir.path().join("source.txt");
    let dst_path = temp_dir.path().join("destination.txt");
    let content = b"Copy test content";

    std::fs::write(&src_path, content).unwrap();

    let processor = AsyncFileProcessor::default();
    let bytes_copied = processor.copy_file(&src_path, &dst_path).await.unwrap();

    assert_eq!(bytes_copied, content.len() as u64);
    assert!(dst_path.exists());

    let copied_content = std::fs::read(&dst_path).unwrap();
    assert_eq!(copied_content, content);

    println!("✅ Copy operations test passed");
}

#[tokio::test]
async fn test_async_file_processor_atomic_operations() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("atomic.txt");
    let original_content = b"Original content";
    let new_content = b"New atomic content";

    // Create original file
    std::fs::write(&file_path, original_content).unwrap();

    let processor = AsyncFileProcessor::default();

    // Test atomic replacement
    processor.atomic_replace(&file_path, new_content).await.unwrap();

    let read_content = std::fs::read(&file_path).unwrap();
    assert_eq!(read_content, new_content);

    println!("✅ Atomic operations test passed");
}

#[tokio::test]
async fn test_async_file_stream() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("stream_test.txt");
    let content = b"0123456789abcdefghijklmnopqrstuvwxyz";

    std::fs::write(&file_path, content).unwrap();

    let processor = AsyncFileProcessor::default();
    let stream_processor = AsyncFileStream::new(processor);
    let mut stream = stream_processor.process_stream(&file_path, 10).await.unwrap();

    let mut chunks = Vec::new();
    use futures_util::stream::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        chunks.push(chunk_result.unwrap());
    }

    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].len(), 10);
    assert_eq!(chunks[3].len(), 6);

    let reconstructed: Vec<u8> = chunks.into_iter().flatten().collect();
    assert_eq!(reconstructed, content);

    println!("✅ Streaming test passed");
}

#[tokio::main]
async fn main() {
    println!("🚀 Running async file operations tests...\n");

    test_async_file_processor_basic_operations().await;
    test_async_file_processor_error_handling().await;
    test_async_file_processor_chunked_processing().await;
    test_async_file_processor_copy_operations().await;
    test_async_file_processor_atomic_operations().await;
    test_async_file_stream().await;

    println!("\n🎉 All async file operations tests passed!");
}