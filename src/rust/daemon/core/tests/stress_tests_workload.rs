//! Stress tests: code analysis, queue depth, and mixed workloads
//!
//! Tests are marked with #[ignore] to prevent running in normal CI runs.
//! Run with: cargo test --test stress_tests_workload -- --ignored --test-threads=1
//!
//! Updated per Task 21 to use unified_queue instead of legacy ingestion_queue.

mod common;

use serde_json;
use shared_test_utils::TestResult;
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

use common::stress::{
    create_test_file_with_size, generate_code_content, setup_test_db, StressMetrics,
};

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
    let mut tasks = Vec::new();

    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor
                .process_file(&path, "code_analysis_test")
                .await
            {
                Ok(_) => {
                    let size = tokio::fs::metadata(&path)
                        .await
                        .ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone
                        .record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => {
                    metrics_clone.record_failure();
                }
            }
        });
        tasks.push(task);
    }

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

    println!(
        "\nStarting queue depth stress test: {} items",
        QUEUE_SIZE
    );

    let test_file =
        create_test_file_with_size(temp_dir.path(), "queue_test.txt", 5).await?;
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
        let payload_json =
            serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

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

    let depth = queue_manager
        .get_unified_queue_depth(None, None)
        .await?;
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
    for i in 0..SMALL_FILES {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("small_{:03}.txt", i),
            fastrand::usize(1..6),
        )
        .await?;

        let processor = document_processor.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&file_path, "mixed_test").await {
                Ok(_) => {
                    let size = tokio::fs::metadata(&file_path)
                        .await
                        .ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone
                        .record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    // Medium files (50-100KB)
    println!("Creating medium files...");
    for i in 0..MEDIUM_FILES {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("medium_{:03}.txt", i),
            fastrand::usize(50..101),
        )
        .await?;

        let processor = document_processor.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&file_path, "mixed_test").await {
                Ok(_) => {
                    let size = tokio::fs::metadata(&file_path)
                        .await
                        .ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone
                        .record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    // Large files (1-5MB)
    println!("Creating large files...");
    for i in 0..LARGE_FILES {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("large_{:03}.txt", i),
            fastrand::usize(1024..5120),
        )
        .await?;

        let processor = document_processor.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&file_path, "mixed_test").await {
                Ok(_) => {
                    let size = tokio::fs::metadata(&file_path)
                        .await
                        .ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone
                        .record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    // Code files
    println!("Creating code files...");
    for i in 0..CODE_FILES {
        let content = generate_code_content(fastrand::usize(10..100));
        let file_path = temp_dir.path().join(format!("code_{:03}.rs", i));

        let mut file = fs::File::create(&file_path).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
        drop(file);

        let processor = document_processor.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&file_path, "mixed_test").await {
                Ok(_) => {
                    let size = tokio::fs::metadata(&file_path)
                        .await
                        .ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone
                        .record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

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
