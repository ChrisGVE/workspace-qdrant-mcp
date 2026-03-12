//! Benchmarks for the perf-extract optimisations (Tasks 1-5).
//!
//! Measures the performance of the components changed in the perf-extract work:
//!
//! - Task 1: Embedding cache reuse — `resolve_embeddings` cache-hit path
//! - Task 2: Hapax eviction       — `BM25::evict_hapax` throughput
//! - Task 3: LRU phrase cache     — LRU hit/miss throughput
//! - Task 4: Background persist   — BM25 `add_document` hot path (no blocking persist)
//! - Task 5: IDF drift correction — IDF math throughput
//!
//! Run with:
//!   ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib \
//!   cargo bench --manifest-path src/rust/Cargo.toml \
//!               --package workspace-qdrant-core \
//!               --bench perf_extract_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::Duration;
use workspace_qdrant_core::embedding::{tokenize_for_bm25, BM25};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a BM25 corpus of `n_docs` with `unique_terms_per_doc` unique terms
/// and `shared_terms_per_doc` terms repeated across all documents.
fn build_corpus(n_docs: usize, unique_terms_per_doc: usize, shared_terms_per_doc: usize) -> BM25 {
    let mut bm25 = BM25::new(1.2);
    let shared: Vec<String> = (0..shared_terms_per_doc)
        .map(|i| format!("shared_term_{i}"))
        .collect();

    for doc in 0..n_docs {
        let mut tokens: Vec<String> = shared.clone();
        for t in 0..unique_terms_per_doc {
            tokens.push(format!("unique_doc{doc}_term{t}"));
        }
        bm25.add_document(&tokens);
    }
    bm25
}

/// BM25 IDF — mirrors the formula in `rebalance_idf.rs`.
#[inline]
fn bm25_idf(n: u64, df: u64) -> f64 {
    if n == 0 || df == 0 {
        return 0.0;
    }
    let n = n as f64;
    let df = df as f64;
    ((n - df + 0.5) / (df + 0.5)).ln().max(0.0)
}

/// IDF correction factor — mirrors `idf_correction` in `rebalance_idf.rs`.
#[inline]
fn idf_correction(old_n: u64, new_n: u64, df: u64) -> f32 {
    if old_n == new_n || df == 0 {
        return 1.0;
    }
    let old_idf = bm25_idf(old_n, df);
    let new_idf = bm25_idf(new_n, df);
    if old_idf < 1e-10 {
        return 1.0;
    }
    (new_idf / old_idf) as f32
}

// ---------------------------------------------------------------------------
// Group 1: tokenize_for_bm25
// ---------------------------------------------------------------------------

fn bench_tokenize_for_bm25(c: &mut Criterion) {
    let short_code = "fn foo(x: i32) -> bool { x > 0 }";
    let medium_code = "pub async fn process_document(path: &Path, collection: &str) \
        -> Result<Vec<f32>, Error> { let content = fs::read_to_string(path).await?; \
        let tokens = tokenize(&content); embed(tokens).await }"
        .repeat(5);
    let large_code = "impl LexiconManager { pub async fn add_document(&self, \
        collection: &str, tokens: &[String]) -> Result<(), LexiconError> { \
        let mut instances = self.instances.write().await; \
        let bm25 = instances.entry(collection.to_string()).or_insert_with(|| BM25::new(1.2)); \
        bm25.add_document(tokens); Ok(()) } }"
        .repeat(20);

    let mut group = c.benchmark_group("tokenize_for_bm25");
    group.throughput(Throughput::Bytes(short_code.len() as u64));
    group.bench_function("short_32chars", |b| {
        b.iter(|| tokenize_for_bm25(black_box(short_code)))
    });
    group.throughput(Throughput::Bytes(medium_code.len() as u64));
    group.bench_function("medium_~1kb", |b| {
        b.iter(|| tokenize_for_bm25(black_box(&medium_code)))
    });
    group.throughput(Throughput::Bytes(large_code.len() as u64));
    group.bench_function("large_~8kb", |b| {
        b.iter(|| tokenize_for_bm25(black_box(&large_code)))
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: BM25 add_document throughput
// (Task 4 — hot path; persist is off-path via background channel)
// ---------------------------------------------------------------------------

fn bench_bm25_add_document(c: &mut Criterion) {
    // Representative token set for a Rust source file chunk (~30 tokens)
    let tokens: Vec<String> = vec![
        "fn",
        "async",
        "process",
        "document",
        "path",
        "collection",
        "result",
        "vec",
        "error",
        "content",
        "read",
        "string",
        "tokens",
        "tokenize",
        "embed",
        "await",
        "return",
        "impl",
        "pub",
        "self",
        "let",
        "mut",
        "bm25",
        "add",
        "ok",
        "lexicon",
        "manager",
        "instances",
        "write",
        "entry",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut group = c.benchmark_group("bm25_add_document");
    group.measurement_time(Duration::from_secs(5));

    for corpus_docs in [100u32, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("docs_already_in_corpus", corpus_docs),
            &corpus_docs,
            |b, &n| {
                // Pre-warm corpus then benchmark adding one more document
                let mut bm25 = build_corpus(n as usize, 0, 30);
                b.iter(|| {
                    bm25.add_document(black_box(&tokens));
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: BM25 generate_sparse_vector
// (IDF computation varies with corpus size — larger N → cheaper math branch)
// ---------------------------------------------------------------------------

fn bench_bm25_generate_sparse_vector(c: &mut Criterion) {
    let query_tokens: Vec<String> = vec![
        "async",
        "fn",
        "process",
        "document",
        "collection",
        "result",
        "error",
        "tokens",
        "embed",
        "lexicon",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut group = c.benchmark_group("bm25_generate_sparse_vector");
    group.measurement_time(Duration::from_secs(5));

    for corpus_docs in [100usize, 1_000, 10_000] {
        let bm25 = build_corpus(corpus_docs, 0, 30);
        group.bench_with_input(
            BenchmarkId::new("corpus_size", corpus_docs),
            &bm25,
            |b, bm25| {
                b.iter(|| bm25.generate_sparse_vector(black_box(&query_tokens)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: BM25 hapax eviction (Task 2)
// ---------------------------------------------------------------------------

fn bench_bm25_hapax_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_hapax_eviction");
    group.measurement_time(Duration::from_secs(5));

    // Test at varying hapax fractions of the vocabulary
    for (label, n_docs, unique_per_doc, shared_per_doc) in [
        // (label,     docs,   unique, shared)  → hapax fraction ≈ unique/(unique+shared)
        ("10pct_hapax", 1_000, 5, 45), // 10% hapax: 5 unique (df=1), 45 shared (df>1)
        ("50pct_hapax", 1_000, 25, 25), // 50% hapax
        ("90pct_hapax", 1_000, 45, 5), // 90% hapax (worst-case for vocabulary bloat)
    ] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || build_corpus(n_docs, unique_per_doc, shared_per_doc),
                |mut bm25| {
                    let evicted = bm25.evict_hapax();
                    black_box(evicted)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: LRU cache hit/miss throughput (approximates PhraseCache, Task 3)
// ---------------------------------------------------------------------------

fn bench_lru_cache_hit_miss(c: &mut Criterion) {
    const CAPACITY: usize = 4096;
    const EMBEDDING_DIM: usize = 384;

    let mut group = c.benchmark_group("lru_phrase_cache");
    group.measurement_time(Duration::from_secs(5));

    // Pre-build a warm cache (all entries filled)
    let mut warm_cache: LruCache<String, Vec<f32>> =
        LruCache::new(NonZeroUsize::new(CAPACITY).unwrap());
    for i in 0..CAPACITY {
        let key = format!("phrase_{i}");
        warm_cache.put(key, vec![0.1f32; EMBEDDING_DIM]);
    }
    // Keys that are guaranteed hits (within the last CAPACITY insertions)
    let hit_key = "phrase_2048".to_string();
    // Key that is guaranteed a miss
    let miss_key = "phrase_99999".to_string();

    group.bench_function("cache_hit", |b| {
        b.iter_batched(
            || warm_cache.clone(),
            |mut cache| {
                let found = cache.get(black_box(&hit_key)).is_some();
                black_box(found)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("cache_miss", |b| {
        b.iter_batched(
            || warm_cache.clone(),
            |mut cache| {
                let found = cache.get(black_box(&miss_key)).is_some();
                black_box(found)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("cache_put_evict", |b| {
        b.iter_batched(
            || warm_cache.clone(),
            |mut cache| {
                // Inserting into a full cache evicts the LRU entry
                cache.put(black_box(miss_key.clone()), vec![0.5f32; EMBEDDING_DIM]);
                black_box(cache.len())
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 6: Embedding cache resolution — cache-hit path (Task 1)
// (Benchmarks the resolve_embeddings logic without a live embedding model)
// ---------------------------------------------------------------------------

fn bench_embedding_cache_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_cache_resolution");
    group.measurement_time(Duration::from_secs(5));

    // Build a cache of 200 phrases with 384-dim embeddings
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
    for i in 0..200usize {
        cache.insert(format!("phrase_{i}"), vec![0.1f32; 384]);
    }

    // All-hit scenario: 50 phrases, all in cache
    let all_hit_phrases: Vec<String> = (0..50).map(|i| format!("phrase_{i}")).collect();

    // Partial-hit scenario: 25 in cache, 25 not
    let partial_hit_phrases: Vec<String> = (0..50)
        .map(|i| {
            if i < 25 {
                format!("phrase_{i}")
            } else {
                format!("unknown_{i}")
            }
        })
        .collect();

    group.bench_function("all_cache_hits_50_phrases", |b| {
        b.iter(|| {
            // Simulate the resolve_embeddings cache-hit fast path
            let result: Vec<Vec<f32>> = all_hit_phrases
                .iter()
                .map(|p| cache.get(p.as_str()).cloned().unwrap_or_default())
                .collect();
            black_box(result)
        })
    });

    group.bench_function("miss_identification_50_phrases", |b| {
        b.iter(|| {
            // Simulate identifying which phrases need embedding (the miss detection step)
            let miss_indices: Vec<usize> = partial_hit_phrases
                .iter()
                .enumerate()
                .filter(|(_, p)| !cache.contains_key(p.as_str()))
                .map(|(i, _)| i)
                .collect();
            black_box(miss_indices)
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 7: IDF drift correction math throughput (Task 5)
// ---------------------------------------------------------------------------

fn bench_idf_correction_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("idf_drift_correction");
    group.measurement_time(Duration::from_secs(5));

    // Simulate the per-term inner loop of rebalance-idf over a sparse vector
    // with 200 terms (typical for a code chunk).
    let sparse_vector: Vec<(u32, f32, u64)> = (0..200u32)
        .map(|i| {
            let df = (i as u64 % 50) + 1; // df in 1..=50
            (i, 0.5f32 + (i as f32) * 0.001, df)
        })
        .collect();

    for (label, old_n, new_n) in [
        ("10pct_growth", 1_000u64, 1_100u64),
        ("50pct_growth", 1_000u64, 1_500u64),
        ("10x_growth", 1_000u64, 10_000u64),
    ] {
        group.bench_function(label, |b| {
            b.iter(|| {
                let corrected: Vec<f32> = sparse_vector
                    .iter()
                    .map(|&(_, val, df)| val * idf_correction(old_n, new_n, df))
                    .collect();
                black_box(corrected)
            })
        });
    }

    group.bench_function("bm25_idf_single", |b| {
        b.iter(|| bm25_idf(black_box(10_000), black_box(5)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    perf_extract,
    bench_tokenize_for_bm25,
    bench_bm25_add_document,
    bench_bm25_generate_sparse_vector,
    bench_bm25_hapax_eviction,
    bench_lru_cache_hit_miss,
    bench_embedding_cache_resolution,
    bench_idf_correction_throughput,
);

criterion_main!(perf_extract);
