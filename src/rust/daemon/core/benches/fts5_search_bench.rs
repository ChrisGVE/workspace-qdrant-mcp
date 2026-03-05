//! FTS5 Search Pipeline Performance Benchmarks (Task 60)
//!
//! Measures:
//! 1. Single-file update latency (diff + FTS5 write)
//! 2. Batch update throughput (lines/sec)
//! 3. Query latency (exact, regex, with/without path_glob, with/without context)
//! 4. search.db size vs corpus size
//!
//! Run: cargo bench --manifest-path src/rust/Cargo.toml --package workspace-qdrant-core --bench fts5_search_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use workspace_qdrant_core::fts_batch_processor::{FileChange, FtsBatchConfig, FtsBatchProcessor};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, search_regex, SearchOptions};

/// Generate realistic Rust-like source code for a given file index.
fn generate_source_file(index: usize, lines: usize) -> String {
    let mut content = Vec::with_capacity(lines);
    content.push(format!(
        "//! Module {} - auto-generated benchmark data",
        index
    ));
    content.push(String::new());
    content.push(format!("use std::collections::HashMap;"));
    content.push(format!("use std::sync::Arc;"));
    content.push(String::new());

    let funcs_per_file = (lines - 5) / 8; // ~8 lines per function
    for f in 0..funcs_per_file {
        content.push(format!("/// Function {} in module {}", f, index));
        content.push(format!(
            "pub fn func_{}_{}(input: &str) -> Result<String, Box<dyn std::error::Error>> {{",
            index, f
        ));
        content.push(format!("    let mut result = HashMap::new();"));
        content.push(format!(
            "    result.insert(\"key_{}\", input.to_string());",
            f
        ));
        content.push(format!(
            "    let processed = Arc::new(result.get(\"key_{}\").cloned());",
            f
        ));
        content.push(format!(
            "    Ok(processed.as_ref().cloned().unwrap_or_default())"
        ));
        content.push(format!("}}"));
        content.push(String::new());
    }
    content.join("\n")
}

/// Setup a database with pre-populated content for search benchmarks.
async fn setup_populated_db(
    file_count: usize,
    lines_per_file: usize,
) -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let db = SearchDbManager::new(&db_path).await.unwrap();

    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    for i in 0..file_count {
        let content = generate_source_file(i, lines_per_file);
        processor.add_change(FileChange {
            file_id: (i + 1) as i64,
            old_content: String::new(),
            new_content: content,
            tenant_id: "bench-proj".to_string(),
            branch: Some("main".to_string()),
            file_path: format!("src/module_{}.rs", i),
        });
    }
    processor.flush(file_count).await.unwrap();

    (tmp, db)
}

// ---------------------------------------------------------------------------
// 1. Single-file update latency
// ---------------------------------------------------------------------------

fn bench_single_file_update(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("fts5_single_file_update");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Pre-populate with 100 files of 100 lines each
    let (_tmp, db) = rt.block_on(setup_populated_db(100, 100));

    // Benchmark: update one line in a 100-line file
    group.bench_function("100_line_file_1_line_change", |b| {
        let original = generate_source_file(0, 100);
        let mut modified = original.clone();
        // Change one line near the middle
        modified = modified.replace(
            "let mut result = HashMap::new();",
            "let mut result = HashMap::with_capacity(16);",
        );

        let mut toggle = false;
        b.to_async(&rt).iter(|| {
            let db = &db;
            let (old, new) = if toggle {
                (&modified, &original)
            } else {
                (&original, &modified)
            };
            toggle = !toggle;

            async move {
                let mut processor = FtsBatchProcessor::new(db, FtsBatchConfig::default());
                processor.add_change(FileChange {
                    file_id: 1,
                    old_content: old.clone(),
                    new_content: new.clone(),
                    tenant_id: "bench-proj".to_string(),
                    branch: Some("main".to_string()),
                    file_path: "src/module_0.rs".to_string(),
                });
                black_box(processor.flush(0).await.unwrap());
            }
        });
    });

    // Benchmark: update a 300-line file
    let large_original = generate_source_file(200, 300);
    let mut large_modified = large_original.clone();
    large_modified = large_modified.replace(
        "let mut result = HashMap::new();",
        "let mut result = HashMap::with_capacity(32);",
    );

    group.bench_function("300_line_file_1_line_change", |b| {
        let mut toggle = false;

        b.to_async(&rt).iter(|| {
            let db = &db;
            let (old, new) = if toggle {
                (&large_modified, &large_original)
            } else {
                (&large_original, &large_modified)
            };
            toggle = !toggle;

            async move {
                let mut processor = FtsBatchProcessor::new(db, FtsBatchConfig::default());
                processor.add_change(FileChange {
                    file_id: 201,
                    old_content: old.clone(),
                    new_content: new.clone(),
                    tenant_id: "bench-proj".to_string(),
                    branch: Some("main".to_string()),
                    file_path: "src/module_200.rs".to_string(),
                });
                black_box(processor.flush(0).await.unwrap());
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Batch update throughput
// ---------------------------------------------------------------------------

fn bench_batch_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("fts5_batch_throughput");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for file_count in [50, 100, 200] {
        let lines_per_file = 100;
        let total_lines = file_count * lines_per_file;
        group.throughput(Throughput::Elements(total_lines as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "batch_ingest",
                format!("{}_files_{}lines", file_count, lines_per_file),
            ),
            &file_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let tmp = TempDir::new().unwrap();
                    let db_path = tmp.path().join("search.db");
                    let db = SearchDbManager::new(&db_path).await.unwrap();

                    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
                    for i in 0..count {
                        let content = generate_source_file(i, lines_per_file);
                        processor.add_change(FileChange {
                            file_id: (i + 1) as i64,
                            old_content: String::new(),
                            new_content: content,
                            tenant_id: "bench-proj".to_string(),
                            branch: Some("main".to_string()),
                            file_path: format!("src/module_{}.rs", i),
                        });
                    }
                    black_box(processor.flush(count).await.unwrap());
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Query latency
// ---------------------------------------------------------------------------

fn bench_query_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("fts5_query_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    // Pre-populate: 200 files × 100 lines = 20,000 lines
    let (_tmp, db) = rt.block_on(setup_populated_db(200, 100));

    // Exact search: common term (many matches)
    group.bench_function("exact_common_term", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(db, "HashMap", &SearchOptions::default())
                        .await
                        .unwrap(),
                );
            }
        });
    });

    // Exact search: rare term (few matches)
    group.bench_function("exact_rare_term", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(db, "func_199_0", &SearchOptions::default())
                        .await
                        .unwrap(),
                );
            }
        });
    });

    // Exact search: no matches
    group.bench_function("exact_no_match", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(db, "nonexistent_pattern_xyz_999", &SearchOptions::default())
                        .await
                        .unwrap(),
                );
            }
        });
    });

    // Exact search with tenant filter
    group.bench_function("exact_with_tenant", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(
                        db,
                        "HashMap",
                        &SearchOptions {
                            tenant_id: Some("bench-proj".to_string()),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap(),
                );
            }
        });
    });

    // Exact search with path glob
    group.bench_function("exact_with_path_glob", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(
                        db,
                        "HashMap",
                        &SearchOptions {
                            path_glob: Some("src/module_1*.rs".to_string()),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap(),
                );
            }
        });
    });

    // Exact search with context lines
    group.bench_function("exact_with_context_2", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(
                        db,
                        "func_50_0",
                        &SearchOptions {
                            context_lines: 2,
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap(),
                );
            }
        });
    });

    // Regex search: simple pattern
    group.bench_function("regex_simple", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_regex(db, "func_\\d+_0", &SearchOptions::default())
                        .await
                        .unwrap(),
                );
            }
        });
    });

    // Regex search: complex pattern
    group.bench_function("regex_complex", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_regex(
                        db,
                        "pub fn \\w+\\(input: &str\\)",
                        &SearchOptions::default(),
                    )
                    .await
                    .unwrap(),
                );
            }
        });
    });

    // Case-insensitive search
    group.bench_function("exact_case_insensitive", |b| {
        b.to_async(&rt).iter(|| {
            let db = &db;
            async move {
                black_box(
                    search_exact(
                        db,
                        "hashmap",
                        &SearchOptions {
                            case_insensitive: true,
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap(),
                );
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. search.db size measurement (printed, not graphed)
// ---------------------------------------------------------------------------

fn bench_db_size(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("fts5_db_size");
    group.sample_size(10);

    for file_count in [100, 500] {
        let lines_per_file = 100;

        group.bench_with_input(
            BenchmarkId::new("measure_size", format!("{}_files", file_count)),
            &file_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let tmp = TempDir::new().unwrap();
                    let db_path = tmp.path().join("search.db");
                    let db = SearchDbManager::new(&db_path).await.unwrap();

                    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

                    let mut total_source_bytes: usize = 0;
                    for i in 0..count {
                        let content = generate_source_file(i, lines_per_file);
                        total_source_bytes += content.len();
                        processor.add_change(FileChange {
                            file_id: (i + 1) as i64,
                            old_content: String::new(),
                            new_content: content,
                            tenant_id: "bench-proj".to_string(),
                            branch: Some("main".to_string()),
                            file_path: format!("src/module_{}.rs", i),
                        });
                    }
                    processor.flush(count).await.unwrap();

                    // Measure db file size
                    let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
                    let ratio = db_size as f64 / total_source_bytes as f64;

                    eprintln!(
                        "\n  [DB Size] {} files × {} lines: source={:.1}KB, db={:.1}KB, ratio={:.2}x",
                        count,
                        lines_per_file,
                        total_source_bytes as f64 / 1024.0,
                        db_size as f64 / 1024.0,
                        ratio,
                    );

                    black_box((db_size, ratio));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_file_update,
    bench_batch_throughput,
    bench_query_latency,
    bench_db_size,
);
criterion_main!(benches);
