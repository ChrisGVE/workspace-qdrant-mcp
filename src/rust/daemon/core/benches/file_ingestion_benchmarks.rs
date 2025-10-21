//! Benchmark tests for file ingestion performance
//!
//! Measures file ingestion throughput (files/second) for various file sizes
//! and types, testing both single-file and batch scenarios.
//!
//! Run with: cargo bench --bench file_ingestion_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use shared_test_utils::fixtures::*;
use std::fs::{File, self};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

/// Generate test file content of specified size
fn generate_test_content(size_kb: usize, content_type: &str) -> String {
    match content_type {
        "python" => {
            let mut content = String::new();
            let mut bytes = 0;
            let target = size_kb * 1024;
            let mut idx = 0;

            while bytes < target {
                let code = format!(
                    "def function_{}(param1: str, param2: int) -> bool:\n\
                     \"\"\"Docstring for function_{}.\"\"\"\n\
                     result = param1 + str(param2)\n\
                     return len(result) > 0\n\n",
                    idx, idx
                );
                bytes += code.len();
                content.push_str(&code);
                idx += 1;
            }
            content
        }
        "markdown" => {
            let mut content = String::new();
            let mut bytes = 0;
            let target = size_kb * 1024;
            let mut idx = 0;

            while bytes < target {
                let md = format!(
                    "# Heading Level 1\n\n\
                     ## Heading Level {}\n\n\
                     This is a paragraph with **bold text** and *italic text*. \
                     It contains [links](https://example.com) and `inline code`.\n\n\
                     - List item 1\n\
                     - List item 2\n\n",
                    idx
                );
                bytes += md.len();
                content.push_str(&md);
                idx += 1;
            }
            content
        }
        "json" => {
            let items: Vec<_> = (0..size_kb * 10).map(|i| {
                serde_json::json!({
                    "id": i,
                    "name": format!("Item {}", i),
                    "description": format!("This is item number {} for benchmarking", i),
                    "value": i as f64 * 1.5,
                    "active": i % 2 == 0,
                })
            }).collect();
            serde_json::to_string_pretty(&serde_json::json!({"items": items})).unwrap()
        }
        _ => {
            let mut content = String::new();
            let mut bytes = 0;
            let target = size_kb * 1024;

            while bytes < target {
                let line = "This is a sample line of text for benchmarking purposes.\n";
                bytes += line.len();
                content.push_str(line);
            }
            content
        }
    }
}

/// Create a temporary file with specified content
fn create_temp_test_file(content: &str, extension: &str) -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join(format!("test_file{}", extension));
    let mut file = File::create(&file_path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();
    (temp_dir, file_path)
}

/// Benchmark reading and processing small files (1KB)
fn bench_ingest_small_files(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_small_1kb");
    group.throughput(Throughput::Bytes(1024));

    for file_type in &["txt", "py", "md", "json"] {
        let content = generate_test_content(1, file_type);
        let (_temp_dir, file_path) = create_temp_test_file(&content, &format!(".{}", file_type));

        group.bench_with_input(
            BenchmarkId::from_parameter(file_type),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let content = fs::read_to_string(black_box(path)).unwrap();
                    // Simulate basic processing (line counting)
                    let lines = content.lines().count();
                    black_box(lines)
                })
            }
        );
    }

    group.finish();
}

/// Benchmark reading and processing medium files (100KB)
fn bench_ingest_medium_files(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_medium_100kb");
    group.throughput(Throughput::Bytes(100 * 1024));

    for file_type in &["txt", "py", "md", "json"] {
        let content = generate_test_content(100, file_type);
        let (_temp_dir, file_path) = create_temp_test_file(&content, &format!(".{}", file_type));

        group.bench_with_input(
            BenchmarkId::from_parameter(file_type),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let content = fs::read_to_string(black_box(path)).unwrap();
                    let lines = content.lines().count();
                    black_box(lines)
                })
            }
        );
    }

    group.finish();
}

/// Benchmark reading and processing large files (1MB)
fn bench_ingest_large_files(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_large_1mb");
    group.throughput(Throughput::Bytes(1024 * 1024));
    group.sample_size(20); // Reduce sample size for large files

    for file_type in &["txt", "py", "md"] {
        let content = generate_test_content(1024, file_type);
        let (_temp_dir, file_path) = create_temp_test_file(&content, &format!(".{}", file_type));

        group.bench_with_input(
            BenchmarkId::from_parameter(file_type),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let content = fs::read_to_string(black_box(path)).unwrap();
                    let lines = content.lines().count();
                    black_box(lines)
                })
            }
        );
    }

    group.finish();
}

/// Benchmark reading and processing very large files (10MB)
fn bench_ingest_very_large_files(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_very_large_10mb");
    group.throughput(Throughput::Bytes(10 * 1024 * 1024));
    group.sample_size(10); // Reduce sample size for very large files
    group.measurement_time(Duration::from_secs(10));

    for file_type in &["txt", "py"] {
        let content = generate_test_content(10 * 1024, file_type);
        let (_temp_dir, file_path) = create_temp_test_file(&content, &format!(".{}", file_type));

        group.bench_with_input(
            BenchmarkId::from_parameter(file_type),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let content = fs::read_to_string(black_box(path)).unwrap();
                    let lines = content.lines().count();
                    black_box(lines)
                })
            }
        );
    }

    group.finish();
}

/// Benchmark batch file ingestion (10 files)
fn bench_batch_ingestion_10_files(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let mut file_paths = Vec::new();

    // Create 10 small test files
    for i in 0..10 {
        let content = generate_test_content(1, "txt");
        let file_path = temp_dir.path().join(format!("test_{}.txt", i));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_paths.push(file_path);
    }

    c.bench_function("batch_ingest_10_small", |b| {
        b.iter(|| {
            let mut total_lines = 0;
            for path in &file_paths {
                let content = fs::read_to_string(black_box(path)).unwrap();
                total_lines += content.lines().count();
            }
            black_box(total_lines)
        })
    });
}

/// Benchmark batch file ingestion (50 files)
fn bench_batch_ingestion_50_files(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let mut file_paths = Vec::new();

    // Create 50 small test files
    for i in 0..50 {
        let content = generate_test_content(1, "txt");
        let file_path = temp_dir.path().join(format!("test_{}.txt", i));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_paths.push(file_path);
    }

    c.bench_function("batch_ingest_50_small", |b| {
        b.iter(|| {
            let mut total_lines = 0;
            for path in &file_paths {
                let content = fs::read_to_string(black_box(path)).unwrap();
                total_lines += content.lines().count();
            }
            black_box(total_lines)
        })
    });
}

/// Benchmark batch ingestion with mixed file types
fn bench_batch_mixed_types(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let mut file_paths = Vec::new();

    // Create mixed file types: 5 of each type
    for file_type in &["txt", "py", "md", "json"] {
        for i in 0..5 {
            let content = generate_test_content(10, file_type);
            let file_path = temp_dir.path().join(format!("test_{}_{}.{}", file_type, i, file_type));
            let mut file = File::create(&file_path).unwrap();
            file.write_all(content.as_bytes()).unwrap();
            file_paths.push(file_path);
        }
    }

    c.bench_function("batch_ingest_20_mixed", |b| {
        b.iter(|| {
            let mut total_lines = 0;
            for path in &file_paths {
                let content = fs::read_to_string(black_box(path)).unwrap();
                total_lines += content.lines().count();
            }
            black_box(total_lines)
        })
    });
}

/// Benchmark throughput for small files (files/second)
fn bench_throughput_small_files(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let mut file_paths = Vec::new();

    // Create 100 small files
    for i in 0..100 {
        let content = generate_test_content(1, "txt");
        let file_path = temp_dir.path().join(format!("test_{}.txt", i));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_paths.push(file_path);
    }

    c.bench_function("throughput_100_files_1kb", |b| {
        b.iter(|| {
            let mut total_lines = 0;
            for path in &file_paths {
                let content = fs::read_to_string(black_box(path)).unwrap();
                total_lines += content.lines().count();
            }
            black_box(total_lines)
        })
    });
}

/// Benchmark throughput for medium files (MB/second)
fn bench_throughput_medium_files(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let mut file_paths = Vec::new();

    // Create 20 medium files (100KB each = 2MB total)
    for i in 0..20 {
        let content = generate_test_content(100, "txt");
        let file_path = temp_dir.path().join(format!("test_{}.txt", i));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_paths.push(file_path);
    }

    let mut group = c.benchmark_group("throughput_medium");
    group.throughput(Throughput::Bytes(20 * 100 * 1024)); // 2MB total

    group.bench_function("throughput_20_files_100kb", |b| {
        b.iter(|| {
            let mut total_lines = 0;
            for path in &file_paths {
                let content = fs::read_to_string(black_box(path)).unwrap();
                total_lines += content.lines().count();
            }
            black_box(total_lines)
        })
    });

    group.finish();
}

/// Benchmark markdown file creation (legacy test)
fn bench_markdown_file_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("create_markdown_file", |b| {
        b.iter(|| {
            rt.block_on(async {
                let content = DocumentFixtures::markdown_content();
                let temp_file = TempFileFixtures::create_temp_file(&content, "md")
                    .await
                    .unwrap();
                black_box(temp_file)
            })
        })
    });
}

/// Benchmark Python code file creation
fn bench_python_file_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("create_python_file", |b| {
        b.iter(|| {
            rt.block_on(async {
                let content = DocumentFixtures::python_content();
                let temp_file = TempFileFixtures::create_temp_file(&content, "py")
                    .await
                    .unwrap();
                black_box(temp_file)
            })
        })
    });
}

/// Benchmark Rust code file creation
fn bench_rust_file_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("create_rust_file", |b| {
        b.iter(|| {
            rt.block_on(async {
                let content = DocumentFixtures::rust_content();
                let temp_file = TempFileFixtures::create_temp_file(&content, "rs")
                    .await
                    .unwrap();
                black_box(temp_file)
            })
        })
    });
}

/// Benchmark large file creation with varying sizes
fn bench_large_file_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("large_file_creation");

    for size_kb in [10, 50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Bytes(*size_kb as u64 * 1024));
        group.bench_with_input(BenchmarkId::from_parameter(size_kb), size_kb, |b, &size| {
            b.iter(|| {
                rt.block_on(async {
                    let temp_file = TempFileFixtures::create_large_temp_file(size)
                        .await
                        .unwrap();
                    black_box(temp_file)
                })
            })
        });
    }

    group.finish();
}

/// Benchmark project structure creation
fn bench_project_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("create_test_project", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (temp_dir, file_paths) = TempFileFixtures::create_temp_project()
                    .await
                    .unwrap();
                black_box((temp_dir, file_paths))
            })
        })
    });
}

/// Benchmark embedding generation
fn bench_embedding_generation(c: &mut Criterion) {
    c.bench_function("generate_random_embedding", |b| {
        b.iter(|| {
            let embedding = EmbeddingFixtures::random_embedding();
            black_box(embedding)
        })
    });
}

/// Benchmark multiple embedding generation
fn bench_multiple_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_embeddings");

    for count in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| {
                let embeddings: Vec<_> = (0..count)
                    .map(|_| EmbeddingFixtures::random_embedding())
                    .collect();
                black_box(embeddings)
            })
        });
    }

    group.finish();
}

/// Benchmark concurrent file operations
fn bench_concurrent_file_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_file_ops");
    group.sample_size(10); // Reduce sample size for expensive operation

    for concurrent in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrent),
            concurrent,
            |b, &concurrent| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::new();

                        for i in 0..concurrent {
                            let handle = tokio::spawn(async move {
                                let content = format!("Document {}", i);
                                TempFileFixtures::create_temp_file(&content, "txt")
                                    .await
                                    .unwrap()
                            });
                            handles.push(handle);
                        }

                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    })
                })
            }
        );
    }

    group.finish();
}

criterion_group!(
    ingestion_benches,
    // New file ingestion throughput benchmarks
    bench_ingest_small_files,
    bench_ingest_medium_files,
    bench_ingest_large_files,
    bench_ingest_very_large_files,
    bench_batch_ingestion_10_files,
    bench_batch_ingestion_50_files,
    bench_batch_mixed_types,
    bench_throughput_small_files,
    bench_throughput_medium_files,
);

criterion_group!(
    legacy_benches,
    // Legacy file creation benchmarks (kept for compatibility)
    bench_markdown_file_creation,
    bench_python_file_creation,
    bench_rust_file_creation,
    bench_large_file_creation,
    bench_project_creation,
    bench_embedding_generation,
    bench_multiple_embeddings,
    bench_concurrent_file_operations,
);

criterion_main!(ingestion_benches, legacy_benches);