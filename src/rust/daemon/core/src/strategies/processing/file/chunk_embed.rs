//! Per-chunk embedding, payload construction, and LSP enrichment.
//!
//! Processes each chunk from the document processor: generates dense/sparse
//! embeddings, builds Qdrant payload metadata, applies LSP enrichment when
//! available, and returns assembled `DocumentPoint`s with chunk tracking records.

use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

use crate::context::ProcessingContext;
use crate::embedding::SparseEmbedding;
use crate::file_classification::is_test_file;
use crate::lsp::EnrichmentStatus;
use crate::storage::DocumentPoint;
use crate::tracked_files_schema::{self, ChunkType as TrackedChunkType, ProcessingStatus};
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::UnifiedQueueItem;
use crate::DocumentContent;

use super::lsp_payload;

/// Metadata for a single chunk tracked in SQLite `qdrant_chunks`.
pub(super) struct ChunkRecord {
    pub point_id: String,
    pub chunk_index: i32,
    pub content_hash: String,
    pub chunk_type: Option<TrackedChunkType>,
    pub symbol_name: Option<String>,
    pub start_line: Option<i32>,
    pub end_line: Option<i32>,
}

/// Result of embedding all chunks for a file.
pub(super) struct EmbedResult {
    pub points: Vec<DocumentPoint>,
    pub chunk_records: Vec<ChunkRecord>,
    pub lsp_status: ProcessingStatus,
    pub treesitter_status: ProcessingStatus,
}

/// Embed all chunks from a document, building Qdrant points and chunk records.
///
/// For each chunk: generates dense + sparse embeddings, constructs the full
/// payload map, applies LSP enrichment when available, and tracks chunk metadata
/// for SQLite `qdrant_chunks`.
#[allow(clippy::too_many_arguments)]
pub(super) async fn embed_chunks(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    document_content: &DocumentContent,
    file_path: &Path,
    file_document_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_type: Option<&str>,
) -> Result<EmbedResult, UnifiedProcessorError> {
    // Check if LSP enrichment is available for this project
    let (is_project_active, lsp_mgr_guard) = if let Some(lsp_mgr) = &ctx.lsp_manager {
        let mgr = lsp_mgr.read().await;
        let is_active = mgr.has_active_servers(&item.tenant_id).await;
        if is_active {
            debug!(
                "LSP enrichment available for project {} on file {}",
                item.tenant_id,
                file_path.display()
            );
        }
        (is_active, Some(lsp_mgr.clone()))
    } else {
        (false, None)
    };

    let mut lsp_status = ProcessingStatus::None;
    let mut treesitter_status = ProcessingStatus::None;
    let mut points = Vec::new();
    let mut chunk_records = Vec::new();
    let embedding_start = std::time::Instant::now();

    for (chunk_idx, chunk) in document_content.chunks.iter().enumerate() {
        // Semaphore-gated embedding generation (Task 504)
        let _permit = ctx.embedding_semaphore.acquire().await.map_err(|e| {
            UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e))
        })?;
        let embedding_result = ctx
            .embedding_generator
            .generate_embedding(&chunk.content, "bge-small-en-v1.5")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
        drop(_permit);

        let mut point_payload = build_chunk_payload(
            &chunk.content,
            chunk.chunk_index,
            item,
            document_content,
            file_path,
            file_document_id,
            relative_path,
            base_point,
            file_hash,
            file_type,
            &chunk.metadata,
        );

        // Extract chunk metadata for tracked record
        let symbol_name = chunk.metadata.get("symbol_name").cloned();
        let start_line = chunk
            .metadata
            .get("start_line")
            .and_then(|s| s.parse::<i32>().ok());
        let end_line = chunk
            .metadata
            .get("end_line")
            .and_then(|s| s.parse::<i32>().ok());
        let chunk_type_str = chunk.metadata.get("chunk_type");
        let chunk_type = chunk_type_str.and_then(|s| TrackedChunkType::from_str(s));

        // Detect tree-sitter status from chunk metadata
        if chunk.metadata.contains_key("chunk_type") {
            treesitter_status = ProcessingStatus::Done;
        }

        // LSP enrichment (if available and file language has LSP support)
        if let Some(lsp_mgr) = &lsp_mgr_guard {
            let file_lang = file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(crate::lsp::Language::from_extension);

            if file_lang
                .as_ref()
                .map_or(false, |l| l.has_lsp_support())
            {
                let mgr = lsp_mgr.read().await;

                let sym_name = chunk
                    .metadata
                    .get("symbol_name")
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                let sl = chunk
                    .metadata
                    .get("start_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(chunk_idx as u32 * 20);

                let el = chunk
                    .metadata
                    .get("end_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(sl + 20);

                let enrichment = mgr
                    .enrich_chunk(
                        &item.tenant_id,
                        file_path,
                        sym_name,
                        sl,
                        el,
                        is_project_active,
                    )
                    .await;

                if enrichment.enrichment_status == EnrichmentStatus::Skipped {
                    // LSP server not ready -- mark as pending for metadata_uplift retry
                    point_payload.insert(
                        "lsp_enrichment_status".to_string(),
                        serde_json::json!("pending"),
                    );
                    // Keep lsp_status as None (not Done) so tracked_files reflects incomplete state
                } else {
                    lsp_payload::add_lsp_enrichment_to_payload(
                        &mut point_payload,
                        &enrichment,
                    );
                    lsp_status = ProcessingStatus::Done;
                }
            } else {
                // Non-code file (markdown, config, etc.) -- skip LSP enrichment
                point_payload.insert(
                    "lsp_enrichment_status".to_string(),
                    serde_json::json!("skipped"),
                );
                lsp_status = ProcessingStatus::Skipped;
            }
        }

        let point_id =
            wqm_common::hashing::compute_point_id(base_point, chunk_idx as u32);
        let content_hash = tracked_files_schema::compute_content_hash(&chunk.content);

        // Use lexicon-backed BM25 for sparse vectors (Task 19: true BM25 with persisted IDF)
        let chunk_tokens: Vec<String> = chunk
            .content
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let lexicon_sparse = ctx
            .lexicon_manager
            .generate_sparse_vector(&item.collection, &chunk_tokens)
            .await;
        // Fall back to embedding generator's sparse vector if lexicon has no corpus stats yet
        let sparse = if !lexicon_sparse.indices.is_empty() {
            lexicon_sparse
        } else {
            embedding_result.sparse.clone()
        };

        let point = DocumentPoint {
            id: point_id.clone(),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: sparse_embedding_to_map(&sparse),
            payload: point_payload,
        };

        points.push(point);
        chunk_records.push(ChunkRecord {
            point_id,
            chunk_index: chunk_idx as i32,
            content_hash,
            chunk_type,
            symbol_name,
            start_line,
            end_line,
        });
    }

    info!(
        "Embedding generation completed: {} chunks in {}ms",
        chunk_records.len(),
        embedding_start.elapsed().as_millis()
    );

    Ok(EmbedResult {
        points,
        chunk_records,
        lsp_status,
        treesitter_status,
    })
}

/// Build the Qdrant payload map for a single chunk.
#[allow(clippy::too_many_arguments)]
fn build_chunk_payload(
    content: &str,
    chunk_index: usize,
    item: &UnifiedQueueItem,
    document_content: &DocumentContent,
    file_path: &Path,
    file_document_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_type: Option<&str>,
    chunk_metadata: &HashMap<String, String>,
) -> HashMap<String, serde_json::Value> {
    let file_path_str = file_path.to_string_lossy();
    let mut payload = HashMap::new();

    payload.insert("content".to_string(), serde_json::json!(content));
    payload.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
    payload.insert("file_path".to_string(), serde_json::json!(file_path_str));
    payload.insert(
        "document_id".to_string(),
        serde_json::json!(file_document_id),
    );
    payload.insert(
        "tenant_id".to_string(),
        serde_json::json!(item.tenant_id),
    );
    payload.insert("branch".to_string(), serde_json::json!(item.branch));
    payload.insert("base_point".to_string(), serde_json::json!(base_point));
    payload.insert(
        "relative_path".to_string(),
        serde_json::json!(relative_path),
    );
    payload.insert(
        "absolute_path".to_string(),
        serde_json::json!(file_path_str),
    );
    payload.insert("file_hash".to_string(), serde_json::json!(file_hash));
    payload.insert(
        "document_type".to_string(),
        serde_json::json!(document_content.document_type.as_str()),
    );
    if let Some(lang) = document_content.document_type.language() {
        payload.insert("language".to_string(), serde_json::json!(lang));
    }
    // Add file extension as metadata (e.g., "rs", "py", "md") -- lowercase for consistency
    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
        payload.insert(
            "file_extension".to_string(),
            serde_json::json!(ext.to_lowercase()),
        );
    }
    payload.insert("item_type".to_string(), serde_json::json!("file"));

    if let Some(ft) = file_type {
        payload.insert(
            "file_type".to_string(),
            serde_json::json!(ft.to_lowercase()),
        );
    }

    // Build tags array from static metadata for filtering/aggregation
    {
        let mut tags = Vec::new();
        if let Some(ft) = file_type {
            tags.push(ft.to_lowercase());
        }
        if let Some(lang) = document_content.document_type.language() {
            tags.push(lang.to_string());
        }
        if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
            tags.push(ext.to_lowercase());
        }
        if is_test_file(file_path) {
            tags.push("test".to_string());
        }
        if !tags.is_empty() {
            payload.insert("tags".to_string(), serde_json::json!(tags));
        }
    }

    // Add chunk-level metadata (symbol_name, start_line, etc.)
    for (key, value) in chunk_metadata {
        payload.insert(format!("chunk_{}", key), serde_json::json!(value));
    }

    payload
}

/// Convert a `SparseEmbedding` to the `HashMap` format expected by `DocumentPoint`.
fn sparse_embedding_to_map(sparse: &SparseEmbedding) -> Option<HashMap<u32, f32>> {
    crate::shared::embedding_pipeline::sparse_embedding_to_map(sparse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::DocumentType;
    use crate::unified_queue_schema::{ItemType, QueueOperation};
    use wqm_common::queue_types::QueueStatus;

    /// Helper: build a minimal UnifiedQueueItem for tests.
    fn test_queue_item() -> UnifiedQueueItem {
        UnifiedQueueItem {
            queue_id: "q-test-1".into(),
            idempotency_key: "key-1".into(),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "tenant-abc".into(),
            collection: "projects".into(),
            status: QueueStatus::InProgress,
            branch: "main".into(),
            payload_json: "{}".into(),
            metadata: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        }
    }

    /// Helper: build a DocumentContent with a given DocumentType.
    fn test_doc_content(doc_type: DocumentType) -> DocumentContent {
        DocumentContent {
            raw_text: "fn main() {}".into(),
            metadata: HashMap::new(),
            document_type: doc_type,
            chunks: vec![],
        }
    }

    // ---- build_chunk_payload tests ----

    #[test]
    fn test_build_chunk_payload_required_fields() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Code("rust".into()));
        let path = PathBuf::from("/project/src/main.rs");
        let metadata = HashMap::new();

        let payload = build_chunk_payload(
            "fn main() {}",
            0,
            &item,
            &doc,
            &path,
            "doc-123",
            "src/main.rs",
            "bp-abc",
            "hash-xyz",
            None,
            &metadata,
        );

        assert_eq!(payload["content"], serde_json::json!("fn main() {}"));
        assert_eq!(payload["chunk_index"], serde_json::json!(0));
        assert_eq!(payload["tenant_id"], serde_json::json!("tenant-abc"));
        assert_eq!(payload["branch"], serde_json::json!("main"));
        assert_eq!(payload["base_point"], serde_json::json!("bp-abc"));
        assert_eq!(payload["relative_path"], serde_json::json!("src/main.rs"));
        assert_eq!(payload["file_hash"], serde_json::json!("hash-xyz"));
        assert_eq!(payload["document_id"], serde_json::json!("doc-123"));
        assert_eq!(payload["item_type"], serde_json::json!("file"));
        assert_eq!(payload["document_type"], serde_json::json!("code"));
    }

    #[test]
    fn test_build_chunk_payload_code_file_has_language() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Code("rust".into()));
        let path = PathBuf::from("/project/src/lib.rs");

        let payload = build_chunk_payload(
            "pub struct Foo;",
            1,
            &item,
            &doc,
            &path,
            "doc-1",
            "src/lib.rs",
            "bp-1",
            "hash-1",
            None,
            &HashMap::new(),
        );

        assert_eq!(payload["language"], serde_json::json!("rust"));
        assert_eq!(payload["file_extension"], serde_json::json!("rs"));
        assert_eq!(payload["chunk_index"], serde_json::json!(1));
    }

    #[test]
    fn test_build_chunk_payload_non_code_file_no_language() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Pdf);
        let path = PathBuf::from("/docs/report.pdf");

        let payload = build_chunk_payload(
            "Report content",
            0,
            &item,
            &doc,
            &path,
            "doc-2",
            "report.pdf",
            "bp-2",
            "hash-2",
            None,
            &HashMap::new(),
        );

        assert!(!payload.contains_key("language"), "PDF should not have language field");
        assert_eq!(payload["document_type"], serde_json::json!("pdf"));
        assert_eq!(payload["file_extension"], serde_json::json!("pdf"));
    }

    #[test]
    fn test_build_chunk_payload_with_file_type() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Code("python".into()));
        let path = PathBuf::from("/project/src/app.py");

        let payload = build_chunk_payload(
            "import os",
            0,
            &item,
            &doc,
            &path,
            "doc-3",
            "src/app.py",
            "bp-3",
            "hash-3",
            Some("Code"),
            &HashMap::new(),
        );

        // file_type stored lowercase
        assert_eq!(payload["file_type"], serde_json::json!("code"));

        // Tags should include file_type, language, and extension
        let tags = payload["tags"].as_array().unwrap();
        assert!(tags.contains(&serde_json::json!("code")));
        assert!(tags.contains(&serde_json::json!("python")));
        assert!(tags.contains(&serde_json::json!("py")));
    }

    #[test]
    fn test_build_chunk_payload_test_file_gets_test_tag() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Code("rust".into()));
        // Path pattern recognized as a test file
        let path = PathBuf::from("/project/tests/test_main.rs");

        let payload = build_chunk_payload(
            "#[test] fn it_works() {}",
            0,
            &item,
            &doc,
            &path,
            "doc-4",
            "tests/test_main.rs",
            "bp-4",
            "hash-4",
            None,
            &HashMap::new(),
        );

        let tags = payload["tags"].as_array().unwrap();
        assert!(
            tags.contains(&serde_json::json!("test")),
            "Test file should have 'test' tag, got: {:?}",
            tags
        );
    }

    #[test]
    fn test_build_chunk_payload_chunk_metadata_prefixed() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Code("rust".into()));
        let path = PathBuf::from("/project/src/lib.rs");

        let mut metadata = HashMap::new();
        metadata.insert("symbol_name".to_string(), "MyStruct".to_string());
        metadata.insert("start_line".to_string(), "10".to_string());
        metadata.insert("end_line".to_string(), "25".to_string());
        metadata.insert("chunk_type".to_string(), "function".to_string());

        let payload = build_chunk_payload(
            "struct MyStruct {}",
            2,
            &item,
            &doc,
            &path,
            "doc-5",
            "src/lib.rs",
            "bp-5",
            "hash-5",
            None,
            &metadata,
        );

        // Metadata keys get prefixed with "chunk_"
        assert_eq!(payload["chunk_symbol_name"], serde_json::json!("MyStruct"));
        assert_eq!(payload["chunk_start_line"], serde_json::json!("10"));
        assert_eq!(payload["chunk_end_line"], serde_json::json!("25"));
        assert_eq!(payload["chunk_chunk_type"], serde_json::json!("function"));
    }

    #[test]
    fn test_build_chunk_payload_no_extension() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Text);
        let path = PathBuf::from("/project/Makefile");

        let payload = build_chunk_payload(
            "all: build",
            0,
            &item,
            &doc,
            &path,
            "doc-6",
            "Makefile",
            "bp-6",
            "hash-6",
            None,
            &HashMap::new(),
        );

        assert!(
            !payload.contains_key("file_extension"),
            "File without extension should not have file_extension field"
        );
    }

    #[test]
    fn test_build_chunk_payload_absolute_path_matches_file_path() {
        let item = test_queue_item();
        let doc = test_doc_content(DocumentType::Code("rust".into()));
        let path = PathBuf::from("/home/user/project/src/main.rs");

        let payload = build_chunk_payload(
            "fn main() {}",
            0,
            &item,
            &doc,
            &path,
            "doc-7",
            "src/main.rs",
            "bp-7",
            "hash-7",
            None,
            &HashMap::new(),
        );

        // Both file_path and absolute_path should be the same (full path)
        assert_eq!(payload["file_path"], payload["absolute_path"]);
        assert_eq!(
            payload["absolute_path"],
            serde_json::json!("/home/user/project/src/main.rs")
        );
    }

    #[test]
    fn test_build_chunk_payload_feature_branch() {
        let mut item = test_queue_item();
        item.branch = "feature/auth".into();

        let doc = test_doc_content(DocumentType::Code("typescript".into()));
        let path = PathBuf::from("/project/src/auth.ts");

        let payload = build_chunk_payload(
            "export class Auth {}",
            0,
            &item,
            &doc,
            &path,
            "doc-8",
            "src/auth.ts",
            "bp-8",
            "hash-8",
            None,
            &HashMap::new(),
        );

        assert_eq!(payload["branch"], serde_json::json!("feature/auth"));
    }

    // ---- ChunkRecord construction tests ----

    #[test]
    fn test_chunk_record_basic_construction() {
        let record = ChunkRecord {
            point_id: "point-1".into(),
            chunk_index: 0,
            content_hash: "abc123".into(),
            chunk_type: Some(TrackedChunkType::Function),
            symbol_name: Some("my_function".into()),
            start_line: Some(10),
            end_line: Some(25),
        };

        assert_eq!(record.point_id, "point-1");
        assert_eq!(record.chunk_index, 0);
        assert_eq!(record.content_hash, "abc123");
        assert!(record.chunk_type.is_some());
        assert_eq!(record.symbol_name.as_deref(), Some("my_function"));
        assert_eq!(record.start_line, Some(10));
        assert_eq!(record.end_line, Some(25));
    }

    #[test]
    fn test_chunk_record_optional_fields_none() {
        let record = ChunkRecord {
            point_id: "point-2".into(),
            chunk_index: 3,
            content_hash: "def456".into(),
            chunk_type: None,
            symbol_name: None,
            start_line: None,
            end_line: None,
        };

        assert!(record.chunk_type.is_none());
        assert!(record.symbol_name.is_none());
        assert!(record.start_line.is_none());
        assert!(record.end_line.is_none());
    }

    #[test]
    fn test_chunk_record_to_tuple_conversion() {
        // Mirrors the conversion in store_track::upsert_and_track
        let records = vec![
            ChunkRecord {
                point_id: "p1".into(),
                chunk_index: 0,
                content_hash: "h1".into(),
                chunk_type: Some(TrackedChunkType::Function),
                symbol_name: Some("foo".into()),
                start_line: Some(1),
                end_line: Some(10),
            },
            ChunkRecord {
                point_id: "p2".into(),
                chunk_index: 1,
                content_hash: "h2".into(),
                chunk_type: None,
                symbol_name: None,
                start_line: None,
                end_line: None,
            },
        ];

        let tuples: Vec<_> = records
            .iter()
            .map(|cr| {
                (
                    cr.point_id.clone(),
                    cr.chunk_index,
                    cr.content_hash.clone(),
                    cr.chunk_type,
                    cr.symbol_name.clone(),
                    cr.start_line,
                    cr.end_line,
                )
            })
            .collect();

        assert_eq!(tuples.len(), 2);
        assert_eq!(tuples[0].0, "p1");
        assert_eq!(tuples[0].1, 0);
        assert_eq!(tuples[0].2, "h1");
        assert!(tuples[0].3.is_some());
        assert_eq!(tuples[0].4.as_deref(), Some("foo"));
        assert_eq!(tuples[1].3, None);
        assert_eq!(tuples[1].4, None);
    }

    // ---- EmbedResult construction test ----

    #[test]
    fn test_embed_result_construction() {
        let result = EmbedResult {
            points: vec![],
            chunk_records: vec![],
            lsp_status: ProcessingStatus::None,
            treesitter_status: ProcessingStatus::Done,
        };

        assert!(result.points.is_empty());
        assert!(result.chunk_records.is_empty());
        assert_eq!(result.lsp_status, ProcessingStatus::None);
        assert_eq!(result.treesitter_status, ProcessingStatus::Done);
    }
}
