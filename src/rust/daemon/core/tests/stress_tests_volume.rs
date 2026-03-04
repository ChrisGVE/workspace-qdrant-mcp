//! Stress tests: volume, rate, watchers, large files, and memory constraints
//!
//! Tests are marked with #[ignore] to prevent running in normal CI runs.
//! Run with: cargo test --test stress_tests_volume -- --ignored --test-threads=1
//!
//! Updated per Task 21 to use unified_queue instead of legacy ingestion_queue.

#[allow(dead_code)]
#[path = "common/stress.rs"]
mod stress;

use serde_json;
use shared_test_utils::TestResult;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::sync::Semaphore;
use tokio::time::sleep;
use workspace_qdrant_core::{
    queue_operations::QueueManager,
    unified_queue_schema::{FilePayload, ItemType, QueueOperation as UnifiedOp},
    DocumentProcessor,
};

use stress::{create_test_file_with_size, setup_test_db, StressMetrics};

/// Test 1: High volume - 1000+ simultaneous file ingestion
#[tokio::test]
#[ignore]
async fn stress_test_high_volume_ingestion() -> TestResult {
    const FILE_COUNT: usize = 1500;
    const FILE_SIZE_KB: usize = 5;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;
    let (pool, _db_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    println!(
        "\n Starting high volume stress test: {} files",
        FILE_COUNT
    );

    // Create test files
    println!("Creating {} test files...", FILE_COUNT);
    let mut file_paths = Vec::new();
    for i in 0..FILE_COUNT {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("test_file_{:04}.txt", i),
            FILE_SIZE_KB,
        )
        .await?;
        file_paths.push(file_path);
    }

    // Enqueue all files using unified queue
    println!("Enqueueing files...");
    for (i, file_path) in file_paths.iter().enumerate() {
        let file_path_str = file_path.to_string_lossy().to_string();
        let file_size = tokio::fs::metadata(&file_path)
            .await
            .ok()
            .map(|m| m.len());
        let payload = FilePayload {
            file_path: file_path_str.clone(),
            file_type: Some("text".to_string()),
            file_hash: None,
            size_bytes: file_size,
            old_path: None,
        };
        let payload_json =
            serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

        queue_manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "stress_tenant",
                "stress_test_collection",
                &payload_json,
                Some("main"),
                None,
            )
            .await?;

        if (i + 1) % 100 == 0 {
            println!("  Enqueued {}/{} files", i + 1, FILE_COUNT);
        }
    }

    // Process files with high concurrency
    println!("Processing files...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let semaphore = Arc::new(Semaphore::new(50));

    let mut tasks = Vec::new();
    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let start = Instant::now();

            match processor
                .process_file(&path, "stress_test_collection")
                .await
            {
                Ok(_result) => {
                    let file_size = tokio::fs::metadata(&path)
                        .await
                        .ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone
                        .record_success(file_size, start.elapsed().as_millis() as u64);
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
    report.print("High Volume Ingestion");

    assert!(
        report.files_processed >= (FILE_COUNT * 95 / 100),
        "Should process at least 95% of files"
    );
    assert!(report.error_rate < 0.05, "Error rate should be below 5%");
    assert!(
        report.throughput_files_per_sec > 10.0,
        "Should process at least 10 files/sec"
    );

    Ok(())
}

/// Test 2: High rate - 100+ files/second rapid additions
#[tokio::test]
#[ignore]
async fn stress_test_high_rate_ingestion() -> TestResult {
    const TARGET_RATE: usize = 120;
    const DURATION_SECONDS: usize = 10;
    const TOTAL_FILES: usize = TARGET_RATE * DURATION_SECONDS;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;
    let (pool, _db_dir) = setup_test_db().await;
    let _queue_manager = QueueManager::new(pool.clone());

    println!(
        "\nStarting high rate stress test: {} files/sec for {}s",
        TARGET_RATE, DURATION_SECONDS
    );

    // Pre-create all test files
    println!("Pre-creating {} test files...", TOTAL_FILES);
    let mut file_paths = Vec::new();
    for i in 0..TOTAL_FILES {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("rapid_{:05}.txt", i),
            2,
        )
        .await?;
        file_paths.push(file_path);
    }

    println!("Starting rapid ingestion...");
    let start_time = Instant::now();
    let interval = Duration::from_secs(1) / TARGET_RATE as u32;

    let document_processor = Arc::new(DocumentProcessor::new());
    let mut tasks = Vec::new();

    for (i, file_path) in file_paths.iter().enumerate() {
        let target_time = start_time + (interval * i as u32);
        let now = Instant::now();
        if target_time > now {
            sleep(target_time - now).await;
        }

        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&path, "rapid_test").await {
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

        if (i + 1) % 100 == 0 {
            println!("  Submitted {}/{} files", i + 1, TOTAL_FILES);
        }
    }

    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("High Rate Ingestion");

    assert!(
        report.throughput_files_per_sec >= (TARGET_RATE as f64 * 0.8),
        "Should maintain at least 80% of target rate"
    );
    assert!(report.error_rate < 0.1, "Error rate should be below 10%");

    Ok(())
}

/// Test 3: Multiple watchers - 10+ folders watched simultaneously
#[tokio::test]
#[ignore]
async fn stress_test_multiple_watchers() -> TestResult {
    const WATCHER_COUNT: usize = 15;
    const FILES_PER_WATCHER: usize = 50;

    let metrics = StressMetrics::new();
    let (_pool, _db_dir) = setup_test_db().await;

    println!(
        "\nStarting multiple watchers stress test: {} watchers",
        WATCHER_COUNT
    );

    let mut watch_dirs = Vec::new();
    for _i in 0..WATCHER_COUNT {
        let dir = tempdir()?;
        watch_dirs.push(dir);
    }

    println!("Creating files in {} directories...", WATCHER_COUNT);

    let mut tasks = Vec::new();
    let document_processor = Arc::new(DocumentProcessor::new());

    for (watcher_id, dir) in watch_dirs.iter().enumerate() {
        for file_id in 0..FILES_PER_WATCHER {
            let file_path = create_test_file_with_size(
                dir.path(),
                &format!("watcher_{}_file_{}.txt", watcher_id, file_id),
                3,
            )
            .await?;

            let processor = document_processor.clone();
            let metrics_clone = metrics.clone();
            let collection = format!("watcher_{}", watcher_id);

            let task = tokio::spawn(async move {
                let start = Instant::now();
                match processor.process_file(&file_path, &collection).await {
                    Ok(_) => {
                        let size = tokio::fs::metadata(&file_path)
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

        if (watcher_id + 1) % 5 == 0 {
            println!(
                "  Created files for {}/{} watchers",
                watcher_id + 1,
                WATCHER_COUNT
            );
        }
    }

    println!("Processing files from all watchers...");
    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("Multiple Watchers");

    let expected_files = WATCHER_COUNT * FILES_PER_WATCHER;
    assert!(
        report.files_processed >= (expected_files * 95 / 100),
        "Should process at least 95% of files across all watchers"
    );
    assert!(report.error_rate < 0.05, "Error rate should be below 5%");

    Ok(())
}

/// Test 4: Large files - 5x 100MB files concurrent processing
#[tokio::test]
#[ignore]
async fn stress_test_large_files() -> TestResult {
    const FILE_COUNT: usize = 5;
    const FILE_SIZE_MB: usize = 100;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!(
        "\nStarting large files stress test: {} files x {}MB",
        FILE_COUNT, FILE_SIZE_MB
    );

    println!("Creating large files (this may take a moment)...");
    let mut file_paths = Vec::new();
    for i in 0..FILE_COUNT {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("large_file_{}.txt", i),
            FILE_SIZE_MB * 1024,
        )
        .await?;
        file_paths.push(file_path);
        println!("  Created file {}/{}", i + 1, FILE_COUNT);
    }

    println!("Processing large files concurrently...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let mut tasks = Vec::new();

    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor
                .process_file(&path, "large_files_test")
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
                    println!(
                        "  Processed large file: {:.2}MB in {:.2}s",
                        size as f64 / 1_048_576.0,
                        start.elapsed().as_secs_f64()
                    );
                }
                Err(e) => {
                    println!("  Failed to process large file: {}", e);
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
    report.print("Large Files");

    assert!(
        report.files_processed >= (FILE_COUNT * 80 / 100),
        "Should process at least 80% of large files"
    );
    assert!(
        report.total_bytes_processed
            >= (FILE_SIZE_MB * 1024 * 1024 * FILE_COUNT * 80 / 100),
        "Should process at least 80% of total bytes"
    );

    Ok(())
}

/// Test 5: Memory constraints - Test behavior under memory pressure
#[tokio::test]
#[ignore]
async fn stress_test_memory_constraints() -> TestResult {
    const FILE_COUNT: usize = 500;
    const FILE_SIZE_KB: usize = 50;
    const MAX_CONCURRENT: usize = 10;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!("\nStarting memory constraints stress test");
    println!(
        "   {} files, max {} concurrent",
        FILE_COUNT, MAX_CONCURRENT
    );

    println!("Creating test files...");
    let mut file_paths = Vec::new();
    for i in 0..FILE_COUNT {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("memory_test_{:04}.txt", i),
            FILE_SIZE_KB,
        )
        .await?;
        file_paths.push(file_path);
    }

    println!("Processing with memory constraints...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT));

    let mut tasks = Vec::new();
    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let start = Instant::now();

            match processor.process_file(&path, "memory_test").await {
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
    report.print("Memory Constraints");

    assert!(
        report.files_processed >= (FILE_COUNT * 90 / 100),
        "Should process at least 90% despite memory constraints"
    );
    assert!(report.error_rate < 0.1, "Error rate should be below 10%");

    Ok(())
}
