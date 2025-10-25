use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use std::sync::Arc;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::runtime::Runtime;
use workspace_qdrant_daemon::daemon::watcher::{FileWatcher, EventDebouncer, EventFilter, DebouncedEvent};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig, CollectionConfig};
use notify::{EventKind, CreateKind, ModifyKind};
use std::time::Instant;
use futures_util::future::join_all;

/// Sample benchmark for document processing performance
fn benchmark_document_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_processing");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("small_document", |b| {
        let content = "Hello world! ".repeat(100);
        b.iter(|| {
            // Simulate document processing
            black_box(content.len())
        });
    });

    group.bench_function("large_document", |b| {
        let content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(1000);
        b.iter(|| {
            // Simulate large document processing
            black_box(content.len())
        });
    });

    group.finish();
}

/// Benchmark for search operations
fn benchmark_search_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_operations");

    group.bench_function("vector_search", |b| {
        let query_vector = vec![0.1f32; 384]; // Typical embedding dimension
        b.iter(|| {
            // Simulate vector similarity calculation
            let sum: f32 = black_box(query_vector.iter().sum());
            black_box(sum)
        });
    });

    group.bench_function("hybrid_search", |b| {
        let dense_scores = vec![0.8, 0.7, 0.6, 0.5, 0.4];
        let sparse_scores = vec![0.9, 0.6, 0.8, 0.3, 0.7];

        b.iter(|| {
            // Simulate reciprocal rank fusion
            let mut combined_scores = Vec::new();
            for (i, (&dense, &sparse)) in dense_scores.iter().zip(sparse_scores.iter()).enumerate() {
                let rrf_score = 1.0 / (60.0 + i as f32 + 1.0) + 1.0 / (60.0 + i as f32 + 1.0);
                combined_scores.push(black_box(dense + sparse + rrf_score));
            }
            black_box(combined_scores)
        });
    });

    group.finish();
}

/// Benchmark for concurrent operations
fn benchmark_concurrent_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_processing");

    group.bench_function("parallel_document_processing", |b| {
        let documents: Vec<String> = (0..100).map(|i| format!("Document {}", i)).collect();

        b.iter(|| {
            use std::thread;
            let handles: Vec<_> = documents
                .chunks(10)
                .map(|chunk| {
                    let chunk = chunk.to_vec();
                    thread::spawn(move || {
                        chunk.iter().map(|doc| black_box(doc.len())).sum::<usize>()
                    })
                })
                .collect();

            let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            black_box(results)
        });
    });

    group.finish();
}

/// File watcher performance benchmarks
fn benchmark_watcher_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("watcher_initialization", |b| {
        b.to_async(&rt).iter(|| async {
            let config = create_performance_config();
            let processor = create_test_processor();

            let start = Instant::now();
            let watcher = FileWatcher::new(&config, processor).await.unwrap();
            let init_time = start.elapsed();

            black_box((watcher, init_time))
        });
    });
}

/// Benchmark watching multiple directories
fn benchmark_watch_multiple_directories(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let sizes = vec![10, 50, 100, 200, 500];

    for size in sizes {
        c.bench_with_input(
            BenchmarkId::new("watch_directories", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = create_performance_config();
                    let processor = create_test_processor();
                    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

                    // Create temporary directories
                    let temp_dirs: Vec<TempDir> = (0..size)
                        .map(|_| TempDir::new().unwrap())
                        .collect();

                    let start = Instant::now();
                    for temp_dir in &temp_dirs {
                        let _ = watcher.watch_directory(temp_dir.path()).await;
                    }
                    let watch_time = start.elapsed();

                    black_box(watch_time)
                });
            },
        );
    }
}

/// Benchmark event processing throughput
fn benchmark_event_processing_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let event_counts = vec![100, 500, 1000, 2000, 5000];

    for count in event_counts {
        c.bench_with_input(
            BenchmarkId::new("event_processing", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let processor = create_test_processor();
                    let events = create_test_events(count);

                    let start = Instant::now();
                    for event in &events {
                        let _ = processor.process_document(&event.path.to_string_lossy()).await;
                    }
                    let processing_time = start.elapsed();

                    black_box(processing_time)
                });
            },
        );
    }
}

/// Benchmark concurrent watcher performance
fn benchmark_concurrent_watchers(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let watcher_counts = vec![5, 10, 20, 50];

    for count in watcher_counts {
        c.bench_with_input(
            BenchmarkId::new("concurrent_watchers", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let config = create_performance_config();
                    let processor = create_test_processor();

                    let start = Instant::now();
                    let mut handles = Vec::new();

                    for _ in 0..count {
                        let config_clone = config.clone();
                        let processor_clone = Arc::clone(&processor);

                        let handle = tokio::spawn(async move {
                            let mut watcher = FileWatcher::new(&config_clone, processor_clone).await.unwrap();
                            watcher.start().await.unwrap();

                            // Simulate some work
                            tokio::time::sleep(Duration::from_millis(10)).await;

                            watcher.stop().await.unwrap();
                        });
                        handles.push(handle);
                    }

                    join_all(handles).await;
                    let concurrent_time = start.elapsed();

                    black_box(concurrent_time)
                });
            },
        );
    }
}

/// Benchmark event debouncing performance
fn benchmark_event_debouncing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let event_counts = vec![1000, 5000, 10000, 20000];

    for count in event_counts {
        c.bench_with_input(
            BenchmarkId::new("event_debouncing", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let debouncer = EventDebouncer::new(Duration::from_millis(100));
                    let events = create_test_events(count);

                    let start = Instant::now();
                    let processed = debouncer.process_events(events).await;
                    let debounce_time = start.elapsed();

                    black_box((processed.len(), debounce_time))
                });
            },
        );
    }
}

/// Benchmark event filtering performance
fn benchmark_event_filtering(c: &mut Criterion) {
    let event_counts = vec![1000, 5000, 10000, 20000];

    for count in event_counts {
        c.bench_with_input(
            BenchmarkId::new("event_filtering", count),
            &count,
            |b, &count| {
                let config = create_performance_config();
                let filter = EventFilter::new(&config);
                let events = create_notify_events(count);

                b.iter(|| {
                    let mut filtered_count = 0;
                    for event in &events {
                        if !filter.should_ignore(event) {
                            filtered_count += 1;
                        }
                    }
                    black_box(filtered_count)
                });
            },
        );
    }
}

/// Benchmark pattern matching performance
fn benchmark_pattern_matching(c: &mut Criterion) {
    let path_counts = vec![1000, 5000, 10000];

    for count in path_counts {
        c.bench_with_input(
            BenchmarkId::new("pattern_matching", count),
            &count,
            |b, &count| {
                let patterns = vec![
                    "*.tmp".to_string(),
                    "*.log".to_string(),
                    "target/**".to_string(),
                    "node_modules/**".to_string(),
                    ".git/**".to_string(),
                ];

                let test_paths = create_test_paths(count);

                b.iter(|| {
                    let mut matches = 0;
                    for path in &test_paths {
                        for pattern in &patterns {
                            if path_matches_pattern(&path.to_string_lossy(), pattern) {
                                matches += 1;
                                break;
                            }
                        }
                    }
                    black_box(matches)
                });
            },
        );
    }
}

/// Benchmark memory usage under load
fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("memory_usage_stress", |b| {
        b.to_async(&rt).iter(|| async {
            let config = create_large_scale_config();
            let processor = create_test_processor();
            let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

            // Create many temporary directories
            let temp_dirs: Vec<TempDir> = (0..500)
                .map(|_| TempDir::new().unwrap())
                .collect();

            let initial_memory = get_memory_usage();

            for temp_dir in &temp_dirs {
                let _ = watcher.watch_directory(temp_dir.path()).await;
            }

            let peak_memory = get_memory_usage();
            let memory_growth = peak_memory - initial_memory;

            black_box(memory_growth)
        });
    });
}

/// Benchmark scalability limits
fn benchmark_scalability_limits(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let sizes = vec![100, 500, 1000, 2000, 5000];

    for size in sizes {
        c.bench_with_input(
            BenchmarkId::new("scalability_test", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = create_large_scale_config();
                    let processor = create_test_processor();
                    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

                    let temp_dirs: Vec<TempDir> = (0..size)
                        .map(|_| TempDir::new().unwrap())
                        .collect();

                    let start = Instant::now();
                    let mut successful_watches = 0;

                    for temp_dir in &temp_dirs {
                        if watcher.watch_directory(temp_dir.path()).await.is_ok() {
                            successful_watches += 1;
                        } else {
                            break; // Hit the limit
                        }
                    }

                    let scaling_time = start.elapsed();

                    black_box((scaling_time, successful_watches))
                });
            },
        );
    }
}

/// Benchmark rapid file operations
fn benchmark_rapid_file_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("rapid_file_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let temp_dir = TempDir::new().unwrap();

            let start = Instant::now();

            // Create many files rapidly
            let mut handles = Vec::new();
            for i in 0..100 {
                let file_path = temp_dir.path().join(format!("test_file_{}.txt", i));
                let handle = tokio::spawn(async move {
                    tokio::fs::write(&file_path, "test content").await.unwrap();
                    tokio::fs::remove_file(&file_path).await.unwrap();
                });
                handles.push(handle);
            }

            join_all(handles).await;
            let operation_time = start.elapsed();

            black_box(operation_time)
        });
    });
}

/// Benchmark watcher start/stop cycles
fn benchmark_watcher_lifecycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("watcher_start_stop_cycles", |b| {
        b.to_async(&rt).iter(|| async {
            let config = create_performance_config();
            let processor = create_test_processor();

            let start = Instant::now();

            for _ in 0..10 {
                let mut watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();
                watcher.start().await.unwrap();
                watcher.stop().await.unwrap();
            }

            let lifecycle_time = start.elapsed();

            black_box(lifecycle_time)
        });
    });
}

// Helper functions for benchmarks

fn create_performance_config() -> FileWatcherConfig {
    FileWatcherConfig {
        enabled: true,
        debounce_ms: 50,
        max_watched_dirs: 1000,
        ignore_patterns: vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            "target/**".to_string(),
        ],
        recursive: true,
    }
}

fn create_large_scale_config() -> FileWatcherConfig {
    FileWatcherConfig {
        enabled: true,
        debounce_ms: 10,
        max_watched_dirs: 10000,
        ignore_patterns: vec![],
        recursive: true,
    }
}

fn create_test_processor() -> Arc<DocumentProcessor> {
    Arc::new(DocumentProcessor::test_instance())
}

fn create_test_events(count: usize) -> Vec<DebouncedEvent> {
    (0..count)
        .map(|i| DebouncedEvent {
            path: PathBuf::from(format!("/tmp/test_file_{}.txt", i)),
            event_type: EventKind::Modify(ModifyKind::Data),
            timestamp: Instant::now(),
        })
        .collect()
}

fn create_notify_events(count: usize) -> Vec<notify::Event> {
    (0..count)
        .map(|i| notify::Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![PathBuf::from(format!("/tmp/notify_test_{}.txt", i))],
            attrs: Default::default(),
        })
        .collect()
}

fn create_test_paths(count: usize) -> Vec<PathBuf> {
    let extensions = ["txt", "log", "tmp", "rs", "py", "js"];
    let dirs = ["target", "node_modules", ".git", "src", "tests"];

    (0..count)
        .map(|i| {
            let ext = extensions[i % extensions.len()];
            let dir = dirs[i % dirs.len()];
            PathBuf::from(format!("{}/test_file_{}.{}", dir, i, ext))
        })
        .collect()
}

fn path_matches_pattern(path: &str, pattern: &str) -> bool {
    if pattern.ends_with("/**") {
        let prefix = &pattern[..pattern.len() - 3];
        return path.starts_with(prefix);
    }
    if pattern.starts_with("*.") {
        let extension = &pattern[2..];
        return path.ends_with(extension);
    }
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return path.starts_with(parts[0]) && path.ends_with(parts[1]);
        }
    }
    path == pattern
}

fn get_memory_usage() -> u64 {
    // Simplified memory usage calculation
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert to bytes
                        }
                    }
                }
            }
        }
    }

    // Fallback estimation
    1024 * 1024 // 1MB as placeholder
}

criterion_group!(
    document_benches,
    benchmark_document_processing,
    benchmark_search_operations,
    benchmark_concurrent_processing
);

criterion_group!(
    watcher_benches,
    benchmark_watcher_initialization,
    benchmark_watch_multiple_directories,
    benchmark_event_processing_throughput,
    benchmark_concurrent_watchers,
    benchmark_event_debouncing,
    benchmark_event_filtering,
    benchmark_pattern_matching,
    benchmark_memory_usage,
    benchmark_scalability_limits,
    benchmark_rapid_file_operations,
    benchmark_watcher_lifecycle
);

criterion_main!(document_benches, watcher_benches);