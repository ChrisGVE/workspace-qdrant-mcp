//! Property-based testing generators using proptest

use proptest::prelude::*;
use std::collections::HashMap;

use crate::config::TEST_EMBEDDING_DIM;

/// Generate random text content for testing
pub fn arbitrary_text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        "[a-zA-Z0-9 .,!?-]{1,50}",
        1..20
    ).prop_map(|words| words.join(" "))
}

/// Generate document-like text with structure
pub fn arbitrary_document() -> impl Strategy<Value = String> {
    (
        arbitrary_text(), // title
        prop::collection::vec(arbitrary_text(), 1..10), // paragraphs
    ).prop_map(|(title, paragraphs)| {
        let mut doc = format!("# {}\n\n", title);
        for paragraph in paragraphs {
            doc.push_str(&paragraph);
            doc.push_str("\n\n");
        }
        doc
    })
}

/// Generate valid collection names
pub fn arbitrary_collection_name() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_-]{2,30}"
}

/// Generate embedding vectors
pub fn arbitrary_embedding() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1.0..1.0f32,
        TEST_EMBEDDING_DIM
    )
}

/// Generate normalized embedding vectors (unit length)
pub fn arbitrary_normalized_embedding() -> impl Strategy<Value = Vec<f32>> {
    arbitrary_embedding().prop_map(|mut vec| {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vec {
                *x /= norm;
            }
        }
        vec
    })
}

/// Generate sparse vector indices and values
pub fn arbitrary_sparse_vector() -> impl Strategy<Value = (Vec<u32>, Vec<f32>)> {
    (
        prop::collection::vec(0u32..1000, 1..50),
        prop::collection::vec(0.0..1.0f32, 1..50),
    ).prop_map(|(mut indices, values)| {
        indices.sort_unstable();
        indices.dedup();
        let len = indices.len().min(values.len());
        (indices[..len].to_vec(), values[..len].to_vec())
    })
}

/// Generate file paths with various extensions
pub fn arbitrary_file_path() -> impl Strategy<Value = String> {
    (
        prop::collection::vec("[a-zA-Z0-9_-]{1,20}", 1..5), // path components
        "[a-zA-Z0-9]{2,4}", // extension
    ).prop_map(|(components, ext)| {
        format!("{}.{}", components.join("/"), ext)
    })
}

/// Generate document metadata
pub fn arbitrary_metadata() -> impl Strategy<Value = HashMap<String, String>> {
    prop::collection::hash_map(
        "[a-zA-Z_][a-zA-Z0-9_]{0,19}", // key
        "[a-zA-Z0-9 .,!?-]{1,100}",    // value
        0..10 // number of entries
    )
}

/// Generate BM25 term frequency data
pub fn arbitrary_term_frequencies() -> impl Strategy<Value = HashMap<String, u32>> {
    prop::collection::hash_map(
        "[a-zA-Z]{3,15}", // terms
        1u32..100,        // frequencies
        1..50             // number of terms
    )
}

/// Generate search queries
pub fn arbitrary_search_query() -> impl Strategy<Value = String> {
    prop::collection::vec(
        "[a-zA-Z]{3,15}",
        1..10
    ).prop_map(|terms| terms.join(" "))
}

/// Generate processing configurations
#[derive(Debug, Clone)]
pub struct ArbitraryProcessingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_workers: usize,
    pub timeout_ms: u64,
    pub enable_caching: bool,
}

pub fn arbitrary_processing_config() -> impl Strategy<Value = ArbitraryProcessingConfig> {
    (
        100usize..2000,  // chunk_size
        0usize..100,     // chunk_overlap
        1usize..16,      // max_workers
        100u64..30000,   // timeout_ms
        any::<bool>(),   // enable_caching
    ).prop_map(|(chunk_size, chunk_overlap, max_workers, timeout_ms, enable_caching)| {
        ArbitraryProcessingConfig {
            chunk_size,
            chunk_overlap: chunk_overlap.min(chunk_size / 2), // Ensure overlap < chunk_size
            max_workers,
            timeout_ms,
            enable_caching,
        }
    })
}

/// Generate embedding configurations
#[derive(Debug, Clone)]
pub struct ArbitraryEmbeddingConfig {
    pub model_name: String,
    pub dimension: usize,
    pub batch_size: usize,
    pub cache_size: usize,
}

pub fn arbitrary_embedding_config() -> impl Strategy<Value = ArbitraryEmbeddingConfig> {
    (
        "[a-zA-Z0-9_-]{5,30}",     // model_name
        prop::sample::select(vec![128, 256, 384, 512, 768, 1024, 1536]), // dimension
        1usize..100,               // batch_size
        0usize..10000,             // cache_size
    ).prop_map(|(model_name, dimension, batch_size, cache_size)| {
        ArbitraryEmbeddingConfig {
            model_name,
            dimension,
            batch_size,
            cache_size,
        }
    })
}

/// Generate document chunks with metadata
#[derive(Debug, Clone)]
pub struct ArbitraryDocumentChunk {
    pub id: String,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub chunk_index: usize,
    pub metadata: HashMap<String, String>,
}

pub fn arbitrary_document_chunk() -> impl Strategy<Value = ArbitraryDocumentChunk> {
    (
        "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", // UUID-like id
        arbitrary_text(),          // content
        0usize..10000,            // start_offset
        100usize..10000,          // end_offset (will be adjusted)
        0usize..1000,             // chunk_index
        arbitrary_metadata(),      // metadata
    ).prop_map(|(id, content, start_offset, end_offset, chunk_index, metadata)| {
        let actual_end_offset = start_offset + content.len().max(1);
        ArbitraryDocumentChunk {
            id,
            content,
            start_offset,
            end_offset: actual_end_offset,
            chunk_index,
            metadata,
        }
    })
}

/// Generate search results with scores
#[derive(Debug, Clone)]
pub struct ArbitrarySearchResult {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

pub fn arbitrary_search_result() -> impl Strategy<Value = ArbitrarySearchResult> {
    (
        "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", // id
        0.0f32..1.0,              // score
        arbitrary_text(),          // content
        arbitrary_metadata(),      // metadata
    ).prop_map(|(id, score, content, metadata)| {
        ArbitrarySearchResult {
            id,
            score,
            content,
            metadata,
        }
    })
}

/// Generate collections of search results (sorted by score descending)
pub fn arbitrary_search_results() -> impl Strategy<Value = Vec<ArbitrarySearchResult>> {
    prop::collection::vec(arbitrary_search_result(), 0..50)
        .prop_map(|mut results| {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            results
        })
}

/// Generate network addresses for testing
pub fn arbitrary_socket_addr() -> impl Strategy<Value = String> {
    (
        prop::collection::vec(0u8..255, 4), // IP octets
        1024u16..65535,                     // port
    ).prop_map(|(octets, port)| {
        format!("{}.{}.{}.{}:{}", octets[0], octets[1], octets[2], octets[3], port)
    })
}

/// Generate HTTP status codes
pub fn arbitrary_http_status() -> impl Strategy<Value = u16> {
    prop::sample::select(vec![
        200, 201, 204, 400, 401, 403, 404, 409, 422, 429, 500, 502, 503, 504
    ])
}

/// Generate timing data for performance tests
#[derive(Debug, Clone)]
pub struct ArbitraryTimingData {
    pub operation: String,
    pub duration_ms: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

pub fn arbitrary_timing_data() -> impl Strategy<Value = ArbitraryTimingData> {
    (
        prop::sample::select(vec![
            "embedding", "search", "indexing", "chunking", "processing"
        ]).prop_map(|s| s.to_string()),
        1u64..10000,              // duration_ms
        any::<bool>(),            // success
        prop::option::of("[a-zA-Z0-9 .,!?-]{10,100}"), // error_message
    ).prop_map(|(operation, duration_ms, success, error_message)| {
        ArbitraryTimingData {
            operation,
            duration_ms,
            success,
            error_message,
        }
    })
}

/// Composite strategy for generating test scenarios
#[derive(Debug, Clone)]
pub struct ArbitraryTestScenario {
    pub documents: Vec<String>,
    pub queries: Vec<String>,
    pub config: ArbitraryProcessingConfig,
    pub expected_results: usize,
}

pub fn arbitrary_test_scenario() -> impl Strategy<Value = ArbitraryTestScenario> {
    (
        prop::collection::vec(arbitrary_document(), 1..10), // documents
        prop::collection::vec(arbitrary_search_query(), 1..5), // queries
        arbitrary_processing_config(),                      // config
        0usize..100,                                       // expected_results
    ).prop_map(|(documents, queries, config, expected_results)| {
        ArbitraryTestScenario {
            documents,
            queries,
            config,
            expected_results,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::test_runner::TestRunner;

    #[test]
    fn test_arbitrary_text_generation() {
        let mut runner = TestRunner::default();
        let strategy = arbitrary_text();

        for _ in 0..10 {
            let text = strategy.new_tree(&mut runner).unwrap().current();
            assert!(!text.is_empty());
            assert!(text.len() < 1000); // Reasonable upper bound
        }
    }

    #[test]
    fn test_arbitrary_embedding_generation() {
        let mut runner = TestRunner::default();
        let strategy = arbitrary_embedding();

        for _ in 0..10 {
            let embedding = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(embedding.len(), TEST_EMBEDDING_DIM);
            assert!(embedding.iter().all(|&x| x >= -1.0 && x <= 1.0));
        }
    }

    #[test]
    fn test_arbitrary_normalized_embedding() {
        let mut runner = TestRunner::default();
        let strategy = arbitrary_normalized_embedding();

        for _ in 0..10 {
            let embedding = strategy.new_tree(&mut runner).unwrap().current();
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001); // Should be approximately unit length
        }
    }

    #[test]
    fn test_arbitrary_sparse_vector() {
        let mut runner = TestRunner::default();
        let strategy = arbitrary_sparse_vector();

        for _ in 0..10 {
            let (indices, values) = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(indices.len(), values.len());
            assert!(indices.windows(2).all(|w| w[0] < w[1])); // Should be sorted
            assert!(values.iter().all(|&x| x >= 0.0 && x <= 1.0));
        }
    }

    #[test]
    fn test_arbitrary_processing_config() {
        let mut runner = TestRunner::default();
        let strategy = arbitrary_processing_config();

        for _ in 0..10 {
            let config = strategy.new_tree(&mut runner).unwrap().current();
            assert!(config.chunk_overlap < config.chunk_size);
            assert!(config.max_workers > 0);
            assert!(config.timeout_ms >= 100);
        }
    }

    #[test]
    fn test_arbitrary_search_results_sorted() {
        let mut runner = TestRunner::default();
        let strategy = arbitrary_search_results();

        for _ in 0..5 {
            let results = strategy.new_tree(&mut runner).unwrap().current();
            if results.len() > 1 {
                let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
                assert!(scores.windows(2).all(|w| w[0] >= w[1])); // Should be sorted descending
            }
        }
    }
}