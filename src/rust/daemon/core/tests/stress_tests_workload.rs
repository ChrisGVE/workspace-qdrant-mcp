//! Stress tests: code analysis, queue depth, and mixed workloads
//!
//! Tests are marked with #[ignore] to prevent running in normal CI runs.
//! Run with: cargo test --test stress_tests_workload -- --ignored --test-threads=1
//!
//! Updated per Task 21 to use unified_queue instead of legacy ingestion_queue.

#[allow(dead_code)]
#[path = "common/stress.rs"]
mod stress;

use serde_json;
use shared_test_utils::TestResult;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tempfile::tempdir;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use workspace_qdrant_core::{
    queue_operations::QueueManager,
    unified_queue_schema::{FilePayload, ItemType, QueueOperation as UnifiedOp},
    DocumentProcessor,
};

use stress::{create_test_file_with_size, generate_code_content, setup_test_db, StressMetrics};

/// Spawn processing tasks for a batch of files, recording metrics.
fn spawn_process_batch(
    file_paths: &[PathBuf],
    processor: &Arc<DocumentProcessor>,
    metrics: &StressMetrics,
    collection: &str,
) -> Vec<tokio::task::JoinHandle<()>> {
    file_paths
        .iter()
        .map(|file_path| {
            let processor = processor.clone();
            let path = file_path.clone();
            let metrics_clone = metrics.clone();
            let coll = collection.to_string();

            tokio::spawn(async move {
                let start = Instant::now();
                match processor.process_file(&path, &coll).await {
                    Ok(_) => {
                        let size = tokio::fs::metadata(&path)
                            .await
                            .ok()
                            .map(|m| m.len() as usize)
                            .unwrap_or(0);
                        metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                    }
                    Err(_) => metrics_clone.record_failure(),
                }
            })
        })
        .collect()
}

/// Test 6: Code files with heavy LSP analysis
#[tokio::test]
#[ignore]
async fn stress_test_code_analysis() -> TestResult {
    const FILE_COUNT: usize = 100;
    const FUNCTIONS_PER_FILE: usize = 50;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!(
        "\nStarting code analysis stress test: {} Rust files",
        FILE_COUNT
    );

    println!(
        "Creating code files with {} functions each...",
        FUNCTIONS_PER_FILE
    );
    let mut file_paths = Vec::new();
    for i in 0..FILE_COUNT {
        let content = generate_code_content(FUNCTIONS_PER_FILE);
        let file_path = temp_dir.path().join(format!("code_{:03}.rs", i));

        let mut file = fs::File::create(&file_path).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
        drop(file);

        file_paths.push(file_path);

        if (i + 1) % 20 == 0 {
            println!("  Created {}/{} code files", i + 1, FILE_COUNT);
        }
    }

    println!("Analyzing code files...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let tasks = spawn_process_batch(
        &file_paths,
        &document_processor,
        &metrics,
        "code_analysis_test",
    );

    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("Code Analysis");

    assert!(
        report.files_processed >= (FILE_COUNT * 90 / 100),
        "Should analyze at least 90% of code files"
    );
    assert!(report.error_rate < 0.1, "Error rate should be below 10%");

    Ok(())
}

/// Test 7: Queue depth stress test
#[tokio::test]
#[ignore]
async fn stress_test_queue_depth() -> TestResult {
    const QUEUE_SIZE: usize = 10000;

    let (pool, _db_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());
    let temp_dir = tempdir()?;

    println!("\nStarting queue depth stress test: {} items", QUEUE_SIZE);

    let test_file = create_test_file_with_size(temp_dir.path(), "queue_test.txt", 5).await?;
    let base_path = test_file.to_string_lossy().to_string();

    println!("Enqueueing {} items...", QUEUE_SIZE);
    let start = Instant::now();

    for i in 0..QUEUE_SIZE {
        let unique_path = format!("{}.{}", base_path, i);
        let payload = FilePayload {
            file_path: unique_path,
            file_type: Some("text".to_string()),
            file_hash: None,
            size_bytes: Some(5 * 1024),
            old_path: None,
        };
        let payload_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

        queue_manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "tenant",
                "queue_depth_test",
                &payload_json,
                Some("main"),
                None,
            )
            .await?;

        if (i + 1) % 1000 == 0 {
            println!("  Enqueued {}/{} items", i + 1, QUEUE_SIZE);
        }
    }

    let enqueue_time = start.elapsed();
    println!(
        "Enqueued {} items in {:.2}s",
        QUEUE_SIZE,
        enqueue_time.as_secs_f64()
    );

    let depth = queue_manager.get_unified_queue_depth(None, None).await?;
    assert_eq!(
        depth, QUEUE_SIZE as i64,
        "Queue depth should match enqueued items"
    );

    println!("Queue depth verified: {}", depth);

    Ok(())
}

/// Test 8: Mixed workload stress test
#[tokio::test]
#[ignore]
async fn stress_test_mixed_workload() -> TestResult {
    const SMALL_FILES: usize = 200;
    const MEDIUM_FILES: usize = 50;
    const LARGE_FILES: usize = 10;
    const CODE_FILES: usize = 30;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!("\nStarting mixed workload stress test");
    println!(
        "   {} small, {} medium, {} large, {} code files",
        SMALL_FILES, MEDIUM_FILES, LARGE_FILES, CODE_FILES
    );

    let document_processor = Arc::new(DocumentProcessor::new());
    let mut tasks = Vec::new();

    // Small files (1-5KB)
    println!("Creating small files...");
    let small_paths = create_sized_file_batch(temp_dir.path(), "small", SMALL_FILES, 1..6).await?;
    tasks.extend(spawn_process_batch(
        &small_paths,
        &document_processor,
        &metrics,
        "mixed_test",
    ));

    // Medium files (50-100KB)
    println!("Creating medium files...");
    let medium_paths =
        create_sized_file_batch(temp_dir.path(), "medium", MEDIUM_FILES, 50..101).await?;
    tasks.extend(spawn_process_batch(
        &medium_paths,
        &document_processor,
        &metrics,
        "mixed_test",
    ));

    // Large files (1-5MB)
    println!("Creating large files...");
    let large_paths =
        create_sized_file_batch(temp_dir.path(), "large", LARGE_FILES, 1024..5120).await?;
    tasks.extend(spawn_process_batch(
        &large_paths,
        &document_processor,
        &metrics,
        "mixed_test",
    ));

    // Code files
    println!("Creating code files...");
    let mut code_paths = Vec::new();
    for i in 0..CODE_FILES {
        let content = generate_code_content(fastrand::usize(10..100));
        let file_path = temp_dir.path().join(format!("code_{:03}.rs", i));
        let mut file = fs::File::create(&file_path).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
        drop(file);
        code_paths.push(file_path);
    }
    tasks.extend(spawn_process_batch(
        &code_paths,
        &document_processor,
        &metrics,
        "mixed_test",
    ));

    println!("Processing mixed workload...");
    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("Mixed Workload");

    let total_expected = SMALL_FILES + MEDIUM_FILES + LARGE_FILES + CODE_FILES;
    assert!(
        report.files_processed >= (total_expected * 90 / 100),
        "Should process at least 90% of mixed workload"
    );

    Ok(())
}

/// Create a batch of test files with random sizes in the given KB range.
async fn create_sized_file_batch(
    dir: &std::path::Path,
    prefix: &str,
    count: usize,
    size_range_kb: std::ops::Range<usize>,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
    let mut paths = Vec::with_capacity(count);
    for i in 0..count {
        let size_kb = fastrand::usize(size_range_kb.clone());
        let path =
            create_test_file_with_size(dir, &format!("{}_{:03}.txt", prefix, i), size_kb).await?;
        paths.push(path);
    }
    Ok(paths)
}
