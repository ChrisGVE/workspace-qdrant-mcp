//! Comprehensive cross-platform benchmarks for performance regression testing
//!
//! This benchmark suite provides baseline performance measurements across platforms
//! and detects performance regressions in critical paths.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
    PlotConfiguration, AxisScale
};
use std::time::Duration;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::path::Path;
use tempfile::TempDir;

// Test modules - we'll import from our test modules
// Since we can't import test modules directly, we'll redefine key structures

/// Cross-platform benchmark configuration
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub data_sizes: Vec<usize>,
    pub iteration_counts: Vec<usize>,
    pub thread_counts: Vec<usize>,
    pub enable_memory_profiling: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            data_sizes: vec![64, 256, 1024, 4096, 16384, 65536],
            iteration_counts: vec![100, 1000, 10000],
            thread_counts: vec![1, 2, 4, 8],
            enable_memory_profiling: true,
        }
    }
}

/// Memory allocation and deallocation benchmarks
fn bench_memory_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("memory_operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &config.data_sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark allocation
        group.bench_with_input(
            BenchmarkId::new("allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = black_box(vec![0u8; size]);
                    data
                });
            },
        );

        // Benchmark allocation and deallocation
        group.bench_with_input(
            BenchmarkId::new("allocation_deallocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = vec![0u8; size];
                    black_box(data);
                    // Data is dropped here
                });
            },
        );

        // Benchmark cloning
        group.bench_with_input(
            BenchmarkId::new("clone", size),
            &size,
            |b, &size| {
                let original_data = vec![0u8; size];
                b.iter(|| {
                    black_box(original_data.clone())
                });
            },
        );

        // Benchmark memory copy
        group.bench_with_input(
            BenchmarkId::new("memory_copy", size),
            &size,
            |b, &size| {
                let source_data = vec![0u8; size];
                b.iter(|| {
                    let mut dest = Vec::with_capacity(size);
                    dest.extend_from_slice(&source_data);
                    black_box(dest);
                });
            },
        );
    }

    group.finish();
}

/// File system operation benchmarks
fn bench_filesystem_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("filesystem_operations");

    // Create temporary directory for benchmarks
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    for &size in &config.data_sizes {
        let test_data = vec![0u8; size];

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark file creation and writing
        group.bench_with_input(
            BenchmarkId::new("file_write", size),
            &size,
            |b, &_size| {
                b.iter_batched(
                    || {
                        let file_path = temp_path.join(format!("test_file_{}.dat", fastrand::u64(..)));
                        (file_path, test_data.clone())
                    },
                    |(file_path, data)| {
                        std::fs::write(&file_path, &data).unwrap();
                        black_box(file_path);
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Benchmark file reading
        group.bench_with_input(
            BenchmarkId::new("file_read", size),
            &size,
            |b, &_size| {
                // Pre-create file for reading
                let file_path = temp_path.join(format!("read_test_{}.dat", size));
                std::fs::write(&file_path, &test_data).unwrap();

                b.iter(|| {
                    let content = std::fs::read(&file_path).unwrap();
                    black_box(content);
                });
            },
        );
    }

    group.finish();
}

/// Serialization and deserialization benchmarks
fn bench_serialization_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization_operations");

    // Create test data structure
    let test_data = create_test_serialization_data();

    // JSON serialization benchmarks
    group.bench_function("json_serialize", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&test_data).unwrap();
            black_box(serialized);
        });
    });

    group.bench_function("json_deserialize", |b| {
        let serialized = serde_json::to_string(&test_data).unwrap();
        b.iter(|| {
            let deserialized: TestSerializationData = serde_json::from_str(&serialized).unwrap();
            black_box(deserialized);
        });
    });

    // JSON roundtrip benchmark
    group.bench_function("json_roundtrip", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&test_data).unwrap();
            let deserialized: TestSerializationData = serde_json::from_str(&serialized).unwrap();
            black_box(deserialized);
        });
    });

    group.finish();
}

/// Threading and concurrency benchmarks
fn bench_concurrency_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("concurrency_operations");

    for &thread_count in &config.thread_counts {
        // Benchmark thread creation and joining
        group.bench_with_input(
            BenchmarkId::new("thread_spawn_join", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|i| {
                            thread::spawn(move || {
                                // Simulate some work
                                let mut sum = 0;
                                for j in 0..100 {
                                    sum += i * j;
                                }
                                sum
                            })
                        })
                        .collect();

                    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                    black_box(results);
                });
            },
        );

        // Benchmark shared state access
        group.bench_with_input(
            BenchmarkId::new("shared_state_access", thread_count),
            &thread_count,
            |b, &thread_count| {
                let shared_counter = Arc::new(Mutex::new(0));

                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let counter = Arc::clone(&shared_counter);
                            thread::spawn(move || {
                                for _ in 0..100 {
                                    let mut num = counter.lock().unwrap();
                                    *num += 1;
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let final_count = *shared_counter.lock().unwrap();
                    black_box(final_count);

                    // Reset counter for next iteration
                    *shared_counter.lock().unwrap() = 0;
                });
            },
        );
    }

    group.finish();
}

/// String processing benchmarks
fn bench_string_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("string_operations");

    for &size in &config.data_sizes {
        let test_string = "a".repeat(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark string cloning
        group.bench_with_input(
            BenchmarkId::new("string_clone", size),
            &test_string,
            |b, test_string| {
                b.iter(|| {
                    black_box(test_string.clone())
                });
            },
        );

        // Benchmark string concatenation
        group.bench_with_input(
            BenchmarkId::new("string_concat", size),
            &size,
            |b, &size| {
                let part1 = "a".repeat(size / 2);
                let part2 = "b".repeat(size / 2);
                b.iter(|| {
                    let result = format!("{}{}", part1, part2);
                    black_box(result);
                });
            },
        );

        // Benchmark UTF-8 validation
        group.bench_with_input(
            BenchmarkId::new("utf8_validation", size),
            &test_string,
            |b, test_string| {
                let bytes = test_string.as_bytes();
                b.iter(|| {
                    let result = std::str::from_utf8(bytes).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark UTF-16 conversion (Windows-specific optimization)
        #[cfg(target_os = "windows")]
        group.bench_with_input(
            BenchmarkId::new("utf16_conversion", size),
            &test_string,
            |b, test_string| {
                b.iter(|| {
                    let utf16: Vec<u16> = test_string.encode_utf16().collect();
                    black_box(utf16);
                });
            },
        );
    }

    group.finish();
}

/// Platform-specific benchmarks
fn bench_platform_specific_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("platform_specific_operations");

    // File path operations - different performance characteristics on different platforms
    group.bench_function("path_operations", |b| {
        let paths = vec![
            "/home/user/documents/test.txt",
            "/var/log/application.log",
            "/tmp/temporary_file.dat",
            "/usr/share/applications/app.desktop",
        ];

        b.iter(|| {
            for path_str in &paths {
                let path = Path::new(path_str);
                let parent = path.parent();
                let filename = path.file_name();
                let extension = path.extension();
                black_box((parent, filename, extension));
            }
        });
    });

    // Environment variable access
    group.bench_function("env_var_access", |b| {
        // Set test environment variable
        std::env::set_var("BENCH_TEST_VAR", "test_value");

        b.iter(|| {
            let value = std::env::var("BENCH_TEST_VAR").unwrap_or_default();
            black_box(value);
        });
    });

    // Process spawning (minimal test)
    group.bench_function("process_spawn", |b| {
        b.iter(|| {
            #[cfg(unix)]
            let output = std::process::Command::new("echo")
                .arg("test")
                .output()
                .unwrap();

            #[cfg(windows)]
            let output = std::process::Command::new("cmd")
                .args(&["/C", "echo test"])
                .output()
                .unwrap();

            black_box(output);
        });
    });

    group.finish();
}

/// Network operation benchmarks (basic)
fn bench_network_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_operations");

    // TCP socket creation benchmark
    group.bench_function("tcp_socket_creation", |b| {
        b.iter(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            black_box((listener, addr));
        });
    });

    // UDP socket creation benchmark
    group.bench_function("udp_socket_creation", |b| {
        b.iter(|| {
            let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
            let addr = socket.local_addr().unwrap();
            black_box((socket, addr));
        });
    });

    group.finish();
}

/// Data structure operation benchmarks
fn bench_data_structure_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    let mut group = c.benchmark_group("data_structure_operations");

    for &size in &config.data_sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Vector operations
        group.bench_with_input(
            BenchmarkId::new("vec_push", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = Vec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    black_box(vec);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec_with_capacity", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = Vec::with_capacity(size);
                    for i in 0..size {
                        vec.push(i);
                    }
                    black_box(vec);
                });
            },
        );

        // HashMap operations
        group.bench_with_input(
            BenchmarkId::new("hashmap_insert", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut map = HashMap::new();
                    for i in 0..size {
                        map.insert(i, i * 2);
                    }
                    black_box(map);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashmap_with_capacity", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut map = HashMap::with_capacity(size);
                    for i in 0..size {
                        map.insert(i, i * 2);
                    }
                    black_box(map);
                });
            },
        );
    }

    group.finish();
}

/// CPU-intensive computation benchmarks
fn bench_computation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("computation_operations");

    // Fibonacci calculation
    group.bench_function("fibonacci_recursive", |b| {
        b.iter(|| {
            black_box(fibonacci_recursive(black_box(20)));
        });
    });

    group.bench_function("fibonacci_iterative", |b| {
        b.iter(|| {
            black_box(fibonacci_iterative(black_box(20)));
        });
    });

    // Prime number calculation
    group.bench_function("prime_sieve", |b| {
        b.iter(|| {
            black_box(sieve_of_eratosthenes(black_box(1000)));
        });
    });

    // Hash computation
    group.bench_function("hash_computation", |b| {
        let data = "test data for hashing".repeat(100);
        b.iter(|| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            data.hash(&mut hasher);
            black_box(hasher.finish());
        });
    });

    group.finish();
}

// Helper functions for benchmarks

fn fibonacci_recursive(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2),
    }
}

fn fibonacci_iterative(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }

    let mut a = 0;
    let mut b = 1;

    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }

    b
}

fn sieve_of_eratosthenes(limit: usize) -> Vec<usize> {
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit > 0 {
        is_prime[1] = false;
    }

    for i in 2..=((limit as f64).sqrt() as usize) {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
    }

    is_prime
        .iter()
        .enumerate()
        .filter(|(_, &prime)| prime)
        .map(|(i, _)| i)
        .collect()
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct TestSerializationData {
    id: u64,
    name: String,
    content: String,
    metadata: HashMap<String, String>,
    tags: Vec<String>,
    timestamps: Vec<u64>,
}

fn create_test_serialization_data() -> TestSerializationData {
    TestSerializationData {
        id: 12345,
        name: "test_document".to_string(),
        content: "This is test content for serialization benchmarking. ".repeat(50),
        metadata: (0..20)
            .map(|i| (format!("key_{}", i), format!("value_{}", i)))
            .collect(),
        tags: (0..10).map(|i| format!("tag_{}", i)).collect(),
        timestamps: (0..100).map(|i| i * 1000).collect(),
    }
}

// Benchmark configuration
fn configure_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(100)
        .with_plots()
}

// Group all benchmarks
criterion_group!(
    name = benches;
    config = configure_criterion();
    targets =
        bench_memory_operations,
        bench_filesystem_operations,
        bench_serialization_operations,
        bench_concurrency_operations,
        bench_string_operations,
        bench_platform_specific_operations,
        bench_network_operations,
        bench_data_structure_operations,
        bench_computation_operations
);

criterion_main!(benches);