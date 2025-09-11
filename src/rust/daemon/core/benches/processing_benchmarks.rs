//! Processing engine benchmarks
//! 
//! Benchmarks for core document processing functionality

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::PathBuf;
use std::time::Duration;
use tempfile::NamedTempFile;
use std::io::Write;

// Mock processing functions for benchmarking
// In a real implementation, these would import from the actual crate

fn mock_process_text_document(content: &str) -> usize {
    // Simulate text processing
    content.lines().count()
}

fn mock_process_json_document(content: &str) -> Result<usize, String> {
    // Simulate JSON parsing
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(value) => Ok(value.as_object().map_or(0, |obj| obj.len())),
        Err(e) => Err(e.to_string()),
    }
}

fn mock_calculate_embeddings(text: &str) -> Vec<f32> {
    // Mock embedding calculation (normally would use ML model)
    text.chars()
        .enumerate()
        .map(|(i, c)| (c as u8 as f32) * (i as f32 + 1.0) / 1000.0)
        .take(384) // Standard embedding size
        .collect()
}

fn benchmark_text_processing(c: &mut Criterion) {
    let small_text = "Hello, world!";
    let medium_text = "This is a medium-sized text document with multiple sentences. ".repeat(50);
    let large_text = "This is a large text document with many paragraphs and sentences. ".repeat(1000);

    let mut group = c.benchmark_group("text_processing");
    
    group.bench_with_input(
        BenchmarkId::new("small_text", small_text.len()),
        small_text,
        |b, text| b.iter(|| mock_process_text_document(black_box(text)))
    );
    
    group.bench_with_input(
        BenchmarkId::new("medium_text", medium_text.len()),
        &medium_text,
        |b, text| b.iter(|| mock_process_text_document(black_box(text)))
    );
    
    group.bench_with_input(
        BenchmarkId::new("large_text", large_text.len()),
        &large_text,
        |b, text| b.iter(|| mock_process_text_document(black_box(text)))
    );
    
    group.finish();
}

fn benchmark_json_processing(c: &mut Criterion) {
    let simple_json = r#"{"name": "test", "value": 42}"#;
    let complex_json = serde_json::json!({
        "users": (0..100).map(|i| serde_json::json!({
            "id": i,
            "name": format!("user_{}", i),
            "email": format!("user_{}@example.com", i),
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["tag1", "tag2", "tag3"]
            }
        })).collect::<Vec<_>>()
    }).to_string();

    let mut group = c.benchmark_group("json_processing");
    
    group.bench_with_input(
        BenchmarkId::new("simple_json", simple_json.len()),
        simple_json,
        |b, json| b.iter(|| mock_process_json_document(black_box(json)))
    );
    
    group.bench_with_input(
        BenchmarkId::new("complex_json", complex_json.len()),
        &complex_json,
        |b, json| b.iter(|| mock_process_json_document(black_box(json)))
    );
    
    group.finish();
}

fn benchmark_embedding_generation(c: &mut Criterion) {
    let short_text = "Short text";
    let medium_text = "This is a medium-length text that would be typical for document processing.";
    let long_text = "This is a much longer text that represents a more typical document or paragraph that would need to be processed for embedding generation in a real-world scenario.".repeat(10);

    let mut group = c.benchmark_group("embedding_generation");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_with_input(
        BenchmarkId::new("short_text", short_text.len()),
        short_text,
        |b, text| b.iter(|| mock_calculate_embeddings(black_box(text)))
    );
    
    group.bench_with_input(
        BenchmarkId::new("medium_text", medium_text.len()),
        medium_text,
        |b, text| b.iter(|| mock_calculate_embeddings(black_box(text)))
    );
    
    group.bench_with_input(
        BenchmarkId::new("long_text", long_text.len()),
        &long_text,
        |b, text| b.iter(|| mock_calculate_embeddings(black_box(text)))
    );
    
    group.finish();
}

fn benchmark_file_io(c: &mut Criterion) {
    let test_data = "Test file content for I/O benchmarking. ".repeat(1000);
    
    let mut group = c.benchmark_group("file_io");
    
    group.bench_function("write_temp_file", |b| {
        b.iter(|| {
            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(black_box(test_data.as_bytes())).unwrap();
            temp_file.flush().unwrap();
        })
    });
    
    group.bench_function("read_temp_file", |b| {
        // Create a temp file for reading
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(test_data.as_bytes()).unwrap();
        temp_file.flush().unwrap();
        let path = temp_file.path().to_path_buf();
        
        b.iter(|| {
            let content = std::fs::read_to_string(black_box(&path)).unwrap();
            black_box(content);
        })
    });
    
    group.finish();
}

fn benchmark_concurrent_processing(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;
    
    let test_texts: Vec<String> = (0..100)
        .map(|i| format!("Test document {} with some content to process. ", i).repeat(10))
        .collect();
    let test_texts = Arc::new(test_texts);
    
    let mut group = c.benchmark_group("concurrent_processing");
    
    group.bench_function("sequential_processing", |b| {
        b.iter(|| {
            for text in test_texts.iter() {
                black_box(mock_process_text_document(text));
            }
        })
    });
    
    group.bench_function("parallel_processing", |b| {
        b.iter(|| {
            let texts = Arc::clone(&test_texts);
            let handles: Vec<_> = (0..4)
                .map(|thread_id| {
                    let texts = Arc::clone(&texts);
                    thread::spawn(move || {
                        let chunk_size = texts.len() / 4;
                        let start = thread_id * chunk_size;
                        let end = if thread_id == 3 { texts.len() } else { start + chunk_size };
                        
                        for text in &texts[start..end] {
                            black_box(mock_process_text_document(text));
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        })
    });
    
    group.finish();
}

criterion_group!(
    processing_benches,
    benchmark_text_processing,
    benchmark_json_processing,
    benchmark_embedding_generation,
    benchmark_file_io,
    benchmark_concurrent_processing
);

criterion_main!(processing_benches);