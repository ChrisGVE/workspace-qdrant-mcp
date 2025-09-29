//! Benchmark tests for file ingestion performance
//!
//! Run with: cargo bench --bench file_ingestion_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use shared_test_utils::fixtures::*;
use std::time::Duration;
use tempfile::NamedTempFile;

/// Benchmark markdown file creation
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
    benches,
    bench_markdown_file_creation,
    bench_python_file_creation,
    bench_rust_file_creation,
    bench_large_file_creation,
    bench_project_creation,
    bench_embedding_generation,
    bench_multiple_embeddings,
    bench_concurrent_file_operations,
);

criterion_main!(benches);