//! Tests for chunk embedding types and payload construction.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::DocumentType;
use crate::unified_queue_schema::{ItemType, QueueOperation, UnifiedQueueItem};
use crate::tracked_files_schema::{ChunkType as TrackedChunkType, ProcessingStatus};
use wqm_common::queue_types::QueueStatus;

use super::payload::build_chunk_payload;
use super::types::{ChunkRecord, EmbedResult};

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
        error_message: None,
        last_error_at: None,
        file_path: None,
        qdrant_status: None,
        search_status: None,
        decision_json: None,
    }
}

/// Helper: build a DocumentContent with a given DocumentType.
fn test_doc_content(doc_type: DocumentType) -> crate::DocumentContent {
    crate::DocumentContent {
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
