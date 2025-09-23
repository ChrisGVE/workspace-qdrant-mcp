use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

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

criterion_group!(
    benches,
    benchmark_document_processing,
    benchmark_search_operations,
    benchmark_concurrent_processing
);
criterion_main!(benches);