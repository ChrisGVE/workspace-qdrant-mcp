//! Stress and load testing for Rust daemon file ingestion
//!
//! This module contains comprehensive stress tests for the daemon's file ingestion
//! pipeline under various extreme conditions including high volume, high rate,
//! multiple watchers, large files, memory pressure, and network failures.
//!
//! Tests are marked with #[ignore] to prevent running in normal CI runs.
//! Run with: cargo test --test stress_tests -- --ignored --test-threads=1
//!
//! Updated per Task 21 to use unified_queue instead of legacy ingestion_queue.

use serde_json;
use shared_test_utils::{
    config::*, fixtures::*, test_helpers::*, TestResult,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::{tempdir, TempDir};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Semaphore, RwLock};
use tokio::time::{sleep, timeout};
use workspace_qdrant_core::{
    queue_config::QueueConnectionConfig,
    queue_operations::QueueManager,
    queue_types::ProcessorConfig,
    DocumentProcessor, EmbeddingGenerator, EmbeddingConfig,
    storage::{StorageClient, StorageConfig},
    unified_queue_schema::{ItemType, QueueOperation as UnifiedOp, FilePayload},
};

/// Stress test metrics collector
#[derive(Debug, Clone)]
struct StressMetrics {
    files_processed: Arc<AtomicUsize>,
    files_failed: Arc<AtomicUsize>,
    total_bytes_processed: Arc<AtomicUsize>,
    processing_times_ms: Arc<RwLock<Vec<u64>>>,
    start_time: Instant,
}

impl StressMetrics {
    fn new() -> Self {
        Self {
            files_processed: Arc::new(AtomicUsize::new(0)),
            files_failed: Arc::new(AtomicUsize::new(0)),
            total_bytes_processed: Arc::new(AtomicUsize::new(0)),
            processing_times_ms: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    fn record_success(&self, bytes: usize, processing_time_ms: u64) {
        self.files_processed.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_processed.fetch_add(bytes, Ordering::SeqCst);
        let times = self.processing_times_ms.clone();
        tokio::spawn(async move {
            times.write().await.push(processing_time_ms);
        });
    }

    fn record_failure(&self) {
        self.files_failed.fetch_add(1, Ordering::SeqCst);
    }

    async fn report(&self) -> StressTestReport {
        let elapsed = self.start_time.elapsed();
        let processed = self.files_processed.load(Ordering::SeqCst);
        let failed = self.files_failed.load(Ordering::SeqCst);
        let bytes = self.total_bytes_processed.load(Ordering::SeqCst);
        let times = self.processing_times_ms.read().await;

        let throughput_files_per_sec = if elapsed.as_secs_f64() > 0.0 {
            processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let throughput_mb_per_sec = if elapsed.as_secs_f64() > 0.0 {
            (bytes as f64 / 1_048_576.0) / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let avg_latency_ms = if !times.is_empty() {
            times.iter().sum::<u64>() as f64 / times.len() as f64
        } else {
            0.0
        };

        let mut sorted_times = times.clone();
        sorted_times.sort_unstable();
        let p50_latency_ms = sorted_times.get(sorted_times.len() / 2).copied().unwrap_or(0);
        let p95_latency_ms = sorted_times.get(sorted_times.len() * 95 / 100).copied().unwrap_or(0);
        let p99_latency_ms = sorted_times.get(sorted_times.len() * 99 / 100).copied().unwrap_or(0);

        StressTestReport {
            total_files: processed + failed,
            files_processed: processed,
            files_failed: failed,
            total_bytes_processed: bytes,
            elapsed_seconds: elapsed.as_secs_f64(),
            throughput_files_per_sec,
            throughput_mb_per_sec,
            avg_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            error_rate: if processed + failed > 0 {
                failed as f64 / (processed + failed) as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug)]
struct StressTestReport {
    total_files: usize,
    files_processed: usize,
    files_failed: usize,
    total_bytes_processed: usize,
    elapsed_seconds: f64,
    throughput_files_per_sec: f64,
    throughput_mb_per_sec: f64,
    avg_latency_ms: f64,
    p50_latency_ms: u64,
    p95_latency_ms: u64,
    p99_latency_ms: u64,
    error_rate: f64,
}

impl StressTestReport {
    fn print(&self, test_name: &str) {
        println!("\n=== Stress Test Report: {} ===", test_name);
        println!("Total Files:        {}", self.total_files);
        println!("Files Processed:    {}", self.files_processed);
        println!("Files Failed:       {}", self.files_failed);
        println!("Total Bytes:        {} MB", self.total_bytes_processed / 1_048_576);
        println!("Elapsed Time:       {:.2}s", self.elapsed_seconds);
        println!("Throughput:         {:.2} files/s", self.throughput_files_per_sec);
        println!("Throughput:         {:.2} MB/s", self.throughput_mb_per_sec);
        println!("Avg Latency:        {:.2}ms", self.avg_latency_ms);
        println!("P50 Latency:        {}ms", self.p50_latency_ms);
        println!("P95 Latency:        {}ms", self.p95_latency_ms);
        println!("P99 Latency:        {}ms", self.p99_latency_ms);
        println!("Error Rate:         {:.2}%", self.error_rate * 100.0);
        println!("=======================================\n");
    }
}

/// Helper to create test database with unified_queue schema
async fn setup_test_db() -> (sqlx::SqlitePool, TempDir) {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("stress_test.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.expect("Failed to create pool");

    // Create unified_queue table (spec-compliant)
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS unified_queue (
            queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
            item_type TEXT NOT NULL CHECK (item_type IN (
                'content', 'file', 'folder', 'project', 'library',
                'delete_tenant', 'delete_document', 'rename'
            )),
            op TEXT NOT NULL CHECK (op IN ('ingest', 'update', 'delete', 'scan')),
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 0 AND priority <= 10),
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                'pending', 'in_progress', 'done', 'failed'
            )),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            lease_until TEXT,
            worker_id TEXT,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL DEFAULT '{}',
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            error_message TEXT,
            last_error_at TEXT,
            branch TEXT DEFAULT 'main',
            metadata TEXT DEFAULT '{}',
            file_path TEXT UNIQUE
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create unified_queue table");

    // Create watch_folders table for priority calculation JOIN
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            is_active INTEGER DEFAULT 0 CHECK (is_active IN (0, 1)),
            patterns TEXT NOT NULL,
            ignore_patterns TEXT NOT NULL,
            auto_ingest BOOLEAN NOT NULL DEFAULT 1,
            recursive BOOLEAN NOT NULL DEFAULT 1,
            recursive_depth INTEGER NOT NULL DEFAULT 10,
            debounce_seconds REAL NOT NULL DEFAULT 2.0,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TEXT,
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folders table");

    (pool, temp_dir)
}

/// Helper to create test file with specified size
async fn create_test_file_with_size(
    dir: &Path,
    name: &str,
    size_kb: usize,
) -> TestResult<PathBuf> {
    let file_path = dir.join(name);
    let mut file = fs::File::create(&file_path).await?;

    // Generate content with repeated pattern
    let pattern = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    let pattern_bytes = pattern.as_bytes();
    let target_bytes = size_kb * 1024;
    let mut written = 0;

    while written < target_bytes {
        let to_write = std::cmp::min(pattern_bytes.len(), target_bytes - written);
        file.write_all(&pattern_bytes[..to_write]).await?;
        written += to_write;
    }

    file.flush().await?;
    drop(file);

    Ok(file_path)
}

/// Helper to generate code file content
fn generate_code_content(function_count: usize) -> String {
    let mut content = String::from("// Auto-generated test code\n\n");

    for i in 0..function_count {
        content.push_str(&format!(
            r#"
/// Function {i} documentation
pub fn function_{i}(param: u32) -> u32 {{
    let result = param * {i};
    println!("Function {i} called with {{}}", param);
    result
}}

#[test]
fn test_function_{i}() {{
    assert_eq!(function_{i}(10), {});
}}
"#,
            i * 10
        ));
    }

    content
}

//
// STRESS TESTS
//

/// Test 1: High volume - 1000+ simultaneous file ingestion
#[tokio::test]
#[ignore] // Run with --ignored flag
async fn stress_test_high_volume_ingestion() -> TestResult {
    const FILE_COUNT: usize = 1500;
    const FILE_SIZE_KB: usize = 5;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;
    let (pool, _db_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    println!("\n🚀 Starting high volume stress test: {} files", FILE_COUNT);

    // Create test files
    println!("📁 Creating {} test files...", FILE_COUNT);
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
    println!("📤 Enqueueing files...");
    for (i, file_path) in file_paths.iter().enumerate() {
        let file_path_str = file_path.to_string_lossy().to_string();
        let file_size = tokio::fs::metadata(&file_path).await.ok().map(|m| m.len());
        let payload = FilePayload {
            file_path: file_path_str.clone(),
            file_type: Some("text".to_string()),
            file_hash: None,
            size_bytes: file_size,
        };
        let payload_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

        queue_manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "stress_tenant",
                "stress_test_collection",
                &payload_json,
                0,
                Some("main"),
                None,
            )
            .await?;

        if (i + 1) % 100 == 0 {
            println!("  Enqueued {}/{} files", i + 1, FILE_COUNT);
        }
    }

    // Process files with high concurrency
    println!("⚙️  Processing files...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let semaphore = Arc::new(Semaphore::new(50)); // Limit concurrent processing

    let mut tasks = Vec::new();
    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let start = Instant::now();

            match processor.process_file(&path, "stress_test_collection").await {
                Ok(_result) => {
                    let file_size = tokio::fs::metadata(&path).await.ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone.record_success(file_size, start.elapsed().as_millis() as u64);
                }
                Err(_) => {
                    metrics_clone.record_failure();
                }
            }
        });
        tasks.push(task);
    }

    // Wait for all tasks to complete
    for task in tasks {
        let _ = task.await;
    }

    // Generate report
    let report = metrics.report().await;
    report.print("High Volume Ingestion");

    // Assertions
    assert!(report.files_processed >= (FILE_COUNT * 95 / 100),
        "Should process at least 95% of files");
    assert!(report.error_rate < 0.05,
        "Error rate should be below 5%");
    assert!(report.throughput_files_per_sec > 10.0,
        "Should process at least 10 files/sec");

    Ok(())
}

/// Test 2: High rate - 100+ files/second rapid additions
#[tokio::test]
#[ignore]
async fn stress_test_high_rate_ingestion() -> TestResult {
    const TARGET_RATE: usize = 120; // files per second
    const DURATION_SECONDS: usize = 10;
    const TOTAL_FILES: usize = TARGET_RATE * DURATION_SECONDS;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;
    let (pool, _db_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    println!("\n🚀 Starting high rate stress test: {} files/sec for {}s",
        TARGET_RATE, DURATION_SECONDS);

    // Pre-create all test files
    println!("📁 Pre-creating {} test files...", TOTAL_FILES);
    let mut file_paths = Vec::new();
    for i in 0..TOTAL_FILES {
        let file_path = create_test_file_with_size(
            temp_dir.path(),
            &format!("rapid_{:05}.txt", i),
            2, // Small files for rapid processing
        )
        .await?;
        file_paths.push(file_path);
    }

    println!("⚡ Starting rapid ingestion...");
    let start_time = Instant::now();
    let interval = Duration::from_secs(1) / TARGET_RATE as u32;

    // Spawn ingestion task
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
                    let size = tokio::fs::metadata(&path).await.ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
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

    // Wait for all to complete
    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("High Rate Ingestion");

    // Assertions
    assert!(report.throughput_files_per_sec >= (TARGET_RATE as f64 * 0.8),
        "Should maintain at least 80% of target rate");
    assert!(report.error_rate < 0.1,
        "Error rate should be below 10%");

    Ok(())
}

/// Test 3: Multiple watchers - 10+ folders watched simultaneously
#[tokio::test]
#[ignore]
async fn stress_test_multiple_watchers() -> TestResult {
    const WATCHER_COUNT: usize = 15;
    const FILES_PER_WATCHER: usize = 50;

    let metrics = StressMetrics::new();
    let (pool, _db_dir) = setup_test_db().await;

    println!("\n🚀 Starting multiple watchers stress test: {} watchers", WATCHER_COUNT);

    // Create multiple watched directories
    let mut watch_dirs = Vec::new();
    for i in 0..WATCHER_COUNT {
        let dir = tempdir()?;
        watch_dirs.push(dir);
    }

    println!("📁 Creating files in {} directories...", WATCHER_COUNT);

    // Process files from all watchers concurrently
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
                        let size = tokio::fs::metadata(&file_path).await.ok()
                            .map(|m| m.len() as usize)
                            .unwrap_or(0);
                        metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                    }
                    Err(_) => {
                        metrics_clone.record_failure();
                    }
                }
            });
            tasks.push(task);
        }

        if (watcher_id + 1) % 5 == 0 {
            println!("  Created files for {}/{} watchers", watcher_id + 1, WATCHER_COUNT);
        }
    }

    println!("⚙️  Processing files from all watchers...");
    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("Multiple Watchers");

    // Assertions
    let expected_files = WATCHER_COUNT * FILES_PER_WATCHER;
    assert!(report.files_processed >= (expected_files * 95 / 100),
        "Should process at least 95% of files across all watchers");
    assert!(report.error_rate < 0.05,
        "Error rate should be below 5%");

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

    println!("\n🚀 Starting large files stress test: {} files x {}MB",
        FILE_COUNT, FILE_SIZE_MB);

    println!("📁 Creating large files (this may take a moment)...");
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

    println!("⚙️  Processing large files concurrently...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let mut tasks = Vec::new();

    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&path, "large_files_test").await {
                Ok(_) => {
                    let size = tokio::fs::metadata(&path).await.ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                    println!("  ✓ Processed large file: {:.2}MB in {:.2}s",
                        size as f64 / 1_048_576.0,
                        start.elapsed().as_secs_f64());
                }
                Err(e) => {
                    println!("  ✗ Failed to process large file: {}", e);
                    metrics_clone.record_failure();
                }
            }
        });
        tasks.push(task);
    }

    // Wait for all large file processing
    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("Large Files");

    // Assertions
    assert!(report.files_processed >= (FILE_COUNT * 80 / 100),
        "Should process at least 80% of large files");
    assert!(report.total_bytes_processed >= (FILE_SIZE_MB * 1024 * 1024 * FILE_COUNT * 80 / 100),
        "Should process at least 80% of total bytes");

    Ok(())
}

/// Test 5: Memory constraints - Test behavior under memory pressure
#[tokio::test]
#[ignore]
async fn stress_test_memory_constraints() -> TestResult {
    const FILE_COUNT: usize = 500;
    const FILE_SIZE_KB: usize = 50;
    const MAX_CONCURRENT: usize = 10; // Limit to constrain memory

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!("\n🚀 Starting memory constraints stress test");
    println!("   {} files, max {} concurrent", FILE_COUNT, MAX_CONCURRENT);

    // Create test files
    println!("📁 Creating test files...");
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

    println!("⚙️  Processing with memory constraints...");
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
                    let size = tokio::fs::metadata(&path).await.ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
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

    // Assertions - more lenient due to constraints
    assert!(report.files_processed >= (FILE_COUNT * 90 / 100),
        "Should process at least 90% despite memory constraints");
    assert!(report.error_rate < 0.1,
        "Error rate should be below 10%");

    Ok(())
}

/// Test 6: Code files with heavy LSP analysis
#[tokio::test]
#[ignore]
async fn stress_test_code_analysis() -> TestResult {
    const FILE_COUNT: usize = 100;
    const FUNCTIONS_PER_FILE: usize = 50;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!("\n🚀 Starting code analysis stress test: {} Rust files", FILE_COUNT);

    // Create Rust code files
    println!("📁 Creating code files with {} functions each...", FUNCTIONS_PER_FILE);
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

    println!("⚙️  Analyzing code files...");
    let document_processor = Arc::new(DocumentProcessor::new());
    let mut tasks = Vec::new();

    for file_path in file_paths.iter() {
        let processor = document_processor.clone();
        let path = file_path.clone();
        let metrics_clone = metrics.clone();

        let task = tokio::spawn(async move {
            let start = Instant::now();
            match processor.process_file(&path, "code_analysis_test").await {
                Ok(_) => {
                    let size = tokio::fs::metadata(&path).await.ok()
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
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

    // Assertions
    assert!(report.files_processed >= (FILE_COUNT * 90 / 100),
        "Should analyze at least 90% of code files");
    assert!(report.error_rate < 0.1,
        "Error rate should be below 10%");

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

    println!("\n🚀 Starting queue depth stress test: {} items", QUEUE_SIZE);

    // Create a single test file for reference
    let test_file = create_test_file_with_size(
        temp_dir.path(),
        "queue_test.txt",
        5,
    )
    .await?;
    let base_path = test_file.to_string_lossy().to_string();

    println!("📤 Enqueueing {} items...", QUEUE_SIZE);
    let start = Instant::now();

    // Each item gets a unique file path in payload to ensure unique idempotency keys
    for i in 0..QUEUE_SIZE {
        let unique_path = format!("{}.{}", base_path, i);
        let payload = FilePayload {
            file_path: unique_path,
            file_type: Some("text".to_string()),
            file_hash: None,
            size_bytes: Some(5 * 1024),
        };
        let payload_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

        queue_manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "tenant",
                "queue_depth_test",
                &payload_json,
                0,
                Some("main"),
                None,
            )
            .await?;

        if (i + 1) % 1000 == 0 {
            println!("  Enqueued {}/{} items", i + 1, QUEUE_SIZE);
        }
    }

    let enqueue_time = start.elapsed();
    println!("✓ Enqueued {} items in {:.2}s", QUEUE_SIZE, enqueue_time.as_secs_f64());

    // Verify queue depth using unified queue method
    let depth = queue_manager.get_unified_queue_depth(None, None).await?;
    assert_eq!(depth, QUEUE_SIZE as i64, "Queue depth should match enqueued items");

    println!("✓ Queue depth verified: {}", depth);

    Ok(())
}

#[tokio::test]
#[ignore]
async fn stress_test_mixed_workload() -> TestResult {
    const SMALL_FILES: usize = 200;
    const MEDIUM_FILES: usize = 50;
    const LARGE_FILES: usize = 10;
    const CODE_FILES: usize = 30;

    let metrics = StressMetrics::new();
    let temp_dir = tempdir()?;

    println!("\n🚀 Starting mixed workload stress test");
    println!("   {} small, {} medium, {} large, {} code files",
        SMALL_FILES, MEDIUM_FILES, LARGE_FILES, CODE_FILES);

    let document_processor = Arc::new(DocumentProcessor::new());
    let mut tasks = Vec::new();

    // Small files (1-5KB)
    println!("📁 Creating small files...");
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
                    let size = tokio::fs::metadata(&file_path).await.ok()
                        .map(|m| m.len() as usize).unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    // Medium files (50-100KB)
    println!("📁 Creating medium files...");
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
                    let size = tokio::fs::metadata(&file_path).await.ok()
                        .map(|m| m.len() as usize).unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    // Large files (1-5MB)
    println!("📁 Creating large files...");
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
                    let size = tokio::fs::metadata(&file_path).await.ok()
                        .map(|m| m.len() as usize).unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    // Code files
    println!("📁 Creating code files...");
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
                    let size = tokio::fs::metadata(&file_path).await.ok()
                        .map(|m| m.len() as usize).unwrap_or(0);
                    metrics_clone.record_success(size, start.elapsed().as_millis() as u64);
                }
                Err(_) => metrics_clone.record_failure(),
            }
        });
        tasks.push(task);
    }

    println!("⚙️  Processing mixed workload...");
    for task in tasks {
        let _ = task.await;
    }

    let report = metrics.report().await;
    report.print("Mixed Workload");

    let total_expected = SMALL_FILES + MEDIUM_FILES + LARGE_FILES + CODE_FILES;
    assert!(report.files_processed >= (total_expected * 90 / 100),
        "Should process at least 90% of mixed workload");

    Ok(())
}
