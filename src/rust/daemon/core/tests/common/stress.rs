//! Stress test shared helpers: metrics, reporting, database setup, and file generation
//!
//! Extracted from stress_tests.rs to be reused across split test modules.

use shared_test_utils::TestResult;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tempfile::{tempdir, TempDir};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use workspace_qdrant_core::queue_config::QueueConnectionConfig;

/// Stress test metrics collector
#[derive(Debug, Clone)]
pub struct StressMetrics {
    pub files_processed: Arc<AtomicUsize>,
    pub files_failed: Arc<AtomicUsize>,
    pub total_bytes_processed: Arc<AtomicUsize>,
    pub processing_times_ms: Arc<RwLock<Vec<u64>>>,
    pub start_time: Instant,
}

impl StressMetrics {
    pub fn new() -> Self {
        Self {
            files_processed: Arc::new(AtomicUsize::new(0)),
            files_failed: Arc::new(AtomicUsize::new(0)),
            total_bytes_processed: Arc::new(AtomicUsize::new(0)),
            processing_times_ms: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    pub fn record_success(&self, bytes: usize, processing_time_ms: u64) {
        self.files_processed.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_processed.fetch_add(bytes, Ordering::SeqCst);
        let times = self.processing_times_ms.clone();
        tokio::spawn(async move {
            times.write().await.push(processing_time_ms);
        });
    }

    pub fn record_failure(&self) {
        self.files_failed.fetch_add(1, Ordering::SeqCst);
    }

    pub async fn report(&self) -> StressTestReport {
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
        let p50_latency_ms = sorted_times
            .get(sorted_times.len() / 2)
            .copied()
            .unwrap_or(0);
        let p95_latency_ms = sorted_times
            .get(sorted_times.len() * 95 / 100)
            .copied()
            .unwrap_or(0);
        let p99_latency_ms = sorted_times
            .get(sorted_times.len() * 99 / 100)
            .copied()
            .unwrap_or(0);

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
pub struct StressTestReport {
    pub total_files: usize,
    pub files_processed: usize,
    pub files_failed: usize,
    pub total_bytes_processed: usize,
    pub elapsed_seconds: f64,
    pub throughput_files_per_sec: f64,
    pub throughput_mb_per_sec: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub error_rate: f64,
}

impl StressTestReport {
    pub fn print(&self, test_name: &str) {
        println!("\n=== Stress Test Report: {} ===", test_name);
        println!("Total Files:        {}", self.total_files);
        println!("Files Processed:    {}", self.files_processed);
        println!("Files Failed:       {}", self.files_failed);
        println!(
            "Total Bytes:        {} MB",
            self.total_bytes_processed / 1_048_576
        );
        println!("Elapsed Time:       {:.2}s", self.elapsed_seconds);
        println!(
            "Throughput:         {:.2} files/s",
            self.throughput_files_per_sec
        );
        println!(
            "Throughput:         {:.2} MB/s",
            self.throughput_mb_per_sec
        );
        println!("Avg Latency:        {:.2}ms", self.avg_latency_ms);
        println!("P50 Latency:        {}ms", self.p50_latency_ms);
        println!("P95 Latency:        {}ms", self.p95_latency_ms);
        println!("P99 Latency:        {}ms", self.p99_latency_ms);
        println!("Error Rate:         {:.2}%", self.error_rate * 100.0);
        println!("=======================================\n");
    }
}

/// Helper to create test database with unified_queue schema
pub async fn setup_test_db() -> (sqlx::SqlitePool, TempDir) {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("stress_test.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.expect("Failed to create pool");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS unified_queue (
            queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
            item_type TEXT NOT NULL CHECK (item_type IN (
                'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
            )),
            op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
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
            error_message TEXT,
            last_error_at TEXT,
            branch TEXT DEFAULT 'main',
            metadata TEXT DEFAULT '{}',
            file_path TEXT UNIQUE
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create unified_queue table");

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
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folders table");

    (pool, temp_dir)
}

/// Helper to create test file with specified size
pub async fn create_test_file_with_size(
    dir: &Path,
    name: &str,
    size_kb: usize,
) -> TestResult<PathBuf> {
    let file_path = dir.join(name);
    let mut file = fs::File::create(&file_path).await?;

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
pub fn generate_code_content(function_count: usize) -> String {
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
