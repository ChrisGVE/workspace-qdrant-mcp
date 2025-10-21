//! Example benchmark demonstrating criterion usage.
//!
//! This file serves as a template for writing Rust performance benchmarks.
//! Run with: cargo bench --bench example_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

/// Simple fibonacci implementation for benchmarking
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

/// Iterative fibonacci implementation
fn fibonacci_iterative(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0;
    let mut b = 1;
    for _ in 1..n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

/// Benchmark recursive fibonacci
fn bench_fibonacci_recursive(c: &mut Criterion) {
    c.bench_function("fibonacci_recursive_20", |b| {
        b.iter(|| fibonacci(black_box(20)))
    });
}

/// Benchmark iterative fibonacci
fn bench_fibonacci_iterative(c: &mut Criterion) {
    c.bench_function("fibonacci_iterative_20", |b| {
        b.iter(|| fibonacci_iterative(black_box(20)))
    });
}

/// Benchmark hashmap operations
fn bench_hashmap_operations(c: &mut Criterion) {
    c.bench_function("hashmap_insert_100", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for i in 0..100 {
                map.insert(black_box(i), black_box(i * 2));
            }
            map
        })
    });

    c.bench_function("hashmap_lookup_100", |b| {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i * 2);
        }
        b.iter(|| {
            for i in 0..100 {
                black_box(map.get(&black_box(i)));
            }
        })
    });
}

/// Benchmark string operations
fn bench_string_operations(c: &mut Criterion) {
    c.bench_function("string_concatenation_100", |b| {
        b.iter(|| {
            let mut s = String::new();
            for i in 0..100 {
                s.push_str(&black_box(i).to_string());
            }
            s
        })
    });

    c.bench_function("string_builder_100", |b| {
        b.iter(|| {
            let parts: Vec<String> = (0..100).map(|i| black_box(i).to_string()).collect();
            parts.join("")
        })
    });
}

/// Benchmark vector operations
fn bench_vector_operations(c: &mut Criterion) {
    c.bench_function("vec_push_1000", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            for i in 0..1000 {
                vec.push(black_box(i));
            }
            vec
        })
    });

    c.bench_function("vec_with_capacity_1000", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(1000);
            for i in 0..1000 {
                vec.push(black_box(i));
            }
            vec
        })
    });
}

criterion_group!(
    benches,
    bench_fibonacci_recursive,
    bench_fibonacci_iterative,
    bench_hashmap_operations,
    bench_string_operations,
    bench_vector_operations
);
criterion_main!(benches);
