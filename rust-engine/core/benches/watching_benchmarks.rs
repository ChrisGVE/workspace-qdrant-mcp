//! File watching system benchmarks
//! 
//! Benchmarks for file system monitoring and event processing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use tempfile::TempDir;

// Mock file event structure for benchmarking
#[derive(Debug, Clone)]
struct MockFileEvent {
    path: PathBuf,
    event_type: String,
    timestamp: Instant,
    system_time: SystemTime,
    size: Option<u64>,
    metadata: HashMap<String, String>,
}

impl MockFileEvent {
    fn new(path: PathBuf, event_type: &str) -> Self {
        Self {
            path,
            event_type: event_type.to_string(),
            timestamp: Instant::now(),
            system_time: SystemTime::now(),
            size: Some(1024),
            metadata: HashMap::new(),
        }
    }
}

// Mock pattern matching for file filtering
struct MockPatternMatcher {
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
}

impl MockPatternMatcher {
    fn new(include: Vec<String>, exclude: Vec<String>) -> Self {
        Self {
            include_patterns: include,
            exclude_patterns: exclude,
        }
    }
    
    fn should_process(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        // Check exclude patterns first
        for pattern in &self.exclude_patterns {
            if path_str.contains(pattern) {
                return false;
            }
        }
        
        // Check include patterns
        if self.include_patterns.is_empty() {
            return true;
        }
        
        for pattern in &self.include_patterns {
            if path_str.ends_with(pattern.trim_start_matches('*')) {
                return true;
            }
        }
        
        false
    }
}

// Mock event debouncer
struct MockEventDebouncer {
    events: HashMap<PathBuf, MockFileEvent>,
    debounce_duration: Duration,
}

impl MockEventDebouncer {
    fn new(debounce_ms: u64) -> Self {
        Self {
            events: HashMap::new(),
            debounce_duration: Duration::from_millis(debounce_ms),
        }
    }
    
    fn add_event(&mut self, event: MockFileEvent) -> bool {
        let now = Instant::now();
        
        if let Some(existing) = self.events.get(&event.path) {
            if now.duration_since(existing.timestamp) < self.debounce_duration {
                self.events.insert(event.path.clone(), event);
                return false;
            }
        }
        
        self.events.insert(event.path.clone(), event);
        true
    }
    
    fn get_ready_events(&mut self) -> Vec<MockFileEvent> {
        let now = Instant::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();
        
        for (path, event) in &self.events {
            if now.duration_since(event.timestamp) >= self.debounce_duration {
                ready.push(event.clone());
                to_remove.push(path.clone());
            }
        }
        
        for path in to_remove {
            self.events.remove(&path);
        }
        
        ready
    }
}

fn benchmark_pattern_matching(c: &mut Criterion) {
    let matcher = MockPatternMatcher::new(
        vec!["*.txt".to_string(), "*.md".to_string(), "*.rs".to_string()],
        vec!["*.tmp".to_string(), ".git".to_string(), "node_modules".to_string()],
    );
    
    let test_paths = vec![
        PathBuf::from("document.txt"),
        PathBuf::from("README.md"),
        PathBuf::from("src/main.rs"),
        PathBuf::from("temp.tmp"),
        PathBuf::from(".git/config"),
        PathBuf::from("node_modules/package.json"),
        PathBuf::from("project/nested/file.txt"),
    ];

    let mut group = c.benchmark_group("pattern_matching");
    
    group.bench_function("single_path_match", |b| {
        b.iter(|| {
            for path in &test_paths {
                black_box(matcher.should_process(black_box(path)));
            }
        })
    });
    
    // Benchmark with many paths
    let many_paths: Vec<PathBuf> = (0..1000)
        .map(|i| PathBuf::from(format!("file_{}.txt", i)))
        .collect();
    
    group.bench_with_input(
        BenchmarkId::new("batch_path_match", many_paths.len()),
        &many_paths,
        |b, paths| {
            b.iter(|| {
                let mut matches = 0;
                for path in paths {
                    if matcher.should_process(black_box(path)) {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        }
    );
    
    group.finish();
}

fn benchmark_event_debouncing(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_debouncing");
    
    group.bench_function("add_unique_events", |b| {
        b.iter(|| {
            let mut debouncer = MockEventDebouncer::new(1000);
            for i in 0..100 {
                let event = MockFileEvent::new(
                    PathBuf::from(format!("file_{}.txt", i)),
                    "create"
                );
                black_box(debouncer.add_event(event));
            }
        })
    });
    
    group.bench_function("add_duplicate_events", |b| {
        b.iter(|| {
            let mut debouncer = MockEventDebouncer::new(1000);
            let path = PathBuf::from("duplicate.txt");
            for _ in 0..100 {
                let event = MockFileEvent::new(path.clone(), "modify");
                black_box(debouncer.add_event(event));
            }
        })
    });
    
    group.bench_function("get_ready_events", |b| {
        // Pre-populate debouncer with old events
        let mut debouncer = MockEventDebouncer::new(10); // Short debounce for testing
        for i in 0..100 {
            let event = MockFileEvent::new(
                PathBuf::from(format!("file_{}.txt", i)),
                "create"
            );
            debouncer.add_event(event);
        }
        
        // Wait for events to be ready
        std::thread::sleep(Duration::from_millis(20));
        
        b.iter(|| {
            let ready = debouncer.get_ready_events();
            black_box(ready);
        })
    });
    
    group.finish();
}

fn benchmark_event_processing_pipeline(c: &mut Criterion) {
    let matcher = MockPatternMatcher::new(
        vec!["*.txt".to_string(), "*.md".to_string()],
        vec!["*.tmp".to_string()],
    );
    
    let mut group = c.benchmark_group("event_processing_pipeline");
    
    group.bench_function("full_pipeline_processing", |b| {
        b.iter(|| {
            let mut debouncer = MockEventDebouncer::new(100);
            let mut processed_count = 0;
            
            // Simulate incoming events
            for i in 0..50 {
                let event = MockFileEvent::new(
                    PathBuf::from(format!("document_{}.txt", i)),
                    "create"
                );
                
                // Pattern matching
                if matcher.should_process(&event.path) {
                    // Debouncing
                    if debouncer.add_event(event) {
                        processed_count += 1;
                    }
                }
            }
            
            black_box(processed_count);
        })
    });
    
    group.finish();
}

fn benchmark_concurrent_event_processing(c: &mut Criterion) {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let mut group = c.benchmark_group("concurrent_event_processing");
    
    group.bench_function("sequential_event_processing", |b| {
        b.iter(|| {
            let matcher = MockPatternMatcher::new(
                vec!["*.txt".to_string()],
                vec![],
            );
            
            for i in 0..1000 {
                let path = PathBuf::from(format!("file_{}.txt", i));
                black_box(matcher.should_process(&path));
            }
        })
    });
    
    group.bench_function("parallel_event_processing", |b| {
        b.iter(|| {
            let matcher = Arc::new(MockPatternMatcher::new(
                vec!["*.txt".to_string()],
                vec![],
            ));
            
            let processed_count = Arc::new(Mutex::new(0));
            let handles: Vec<_> = (0..4)
                .map(|thread_id| {
                    let matcher = Arc::clone(&matcher);
                    let count = Arc::clone(&processed_count);
                    thread::spawn(move || {
                        let start = thread_id * 250;
                        let end = start + 250;
                        let mut local_count = 0;
                        
                        for i in start..end {
                            let path = PathBuf::from(format!("file_{}.txt", i));
                            if matcher.should_process(&path) {
                                local_count += 1;
                            }
                        }
                        
                        *count.lock().unwrap() += local_count;
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let final_count = *processed_count.lock().unwrap();
            black_box(final_count);
        })
    });
    
    group.finish();
}

fn benchmark_file_system_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_system_operations");
    
    group.bench_function("directory_traversal", |b| {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        
        // Create test directory structure
        for i in 0..100 {
            let file_path = temp_path.join(format!("test_file_{}.txt", i));
            std::fs::write(&file_path, format!("Content of file {}", i)).unwrap();
        }
        
        b.iter(|| {
            let mut file_count = 0;
            for entry in std::fs::read_dir(black_box(temp_path)).unwrap() {
                if let Ok(entry) = entry {
                    if entry.path().is_file() {
                        file_count += 1;
                    }
                }
            }
            black_box(file_count);
        })
    });
    
    group.bench_function("recursive_directory_scan", |b| {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        
        // Create nested directory structure
        for i in 0..10 {
            let dir_path = temp_path.join(format!("dir_{}", i));
            std::fs::create_dir_all(&dir_path).unwrap();
            
            for j in 0..10 {
                let file_path = dir_path.join(format!("file_{}_{}.txt", i, j));
                std::fs::write(&file_path, format!("Content {}_{}", i, j)).unwrap();
            }
        }
        
        b.iter(|| {
            fn scan_directory(path: &Path) -> usize {
                let mut count = 0;
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries {
                        if let Ok(entry) = entry {
                            let path = entry.path();
                            if path.is_dir() {
                                count += scan_directory(&path);
                            } else {
                                count += 1;
                            }
                        }
                    }
                }
                count
            }
            
            let file_count = scan_directory(black_box(temp_path));
            black_box(file_count);
        })
    });
    
    group.finish();
}

criterion_group!(
    watching_benches,
    benchmark_pattern_matching,
    benchmark_event_debouncing,
    benchmark_event_processing_pipeline,
    benchmark_concurrent_event_processing,
    benchmark_file_system_operations
);

criterion_main!(watching_benches);