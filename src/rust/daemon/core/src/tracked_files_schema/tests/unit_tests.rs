use super::super::*;

#[test]
fn test_processing_status_display() {
    assert_eq!(ProcessingStatus::None.to_string(), "none");
    assert_eq!(ProcessingStatus::Done.to_string(), "done");
    assert_eq!(ProcessingStatus::Failed.to_string(), "failed");
    assert_eq!(ProcessingStatus::Skipped.to_string(), "skipped");
}

#[test]
fn test_processing_status_from_str() {
    assert_eq!(
        ProcessingStatus::from_str("none"),
        Some(ProcessingStatus::None)
    );
    assert_eq!(
        ProcessingStatus::from_str("done"),
        Some(ProcessingStatus::Done)
    );
    assert_eq!(
        ProcessingStatus::from_str("FAILED"),
        Some(ProcessingStatus::Failed)
    );
    assert_eq!(
        ProcessingStatus::from_str("Skipped"),
        Some(ProcessingStatus::Skipped)
    );
    assert_eq!(ProcessingStatus::from_str("invalid"), Option::None);
}

#[test]
fn test_chunk_type_display() {
    assert_eq!(ChunkType::Function.to_string(), "function");
    assert_eq!(ChunkType::Method.to_string(), "method");
    assert_eq!(ChunkType::Class.to_string(), "class");
    assert_eq!(ChunkType::Module.to_string(), "module");
    assert_eq!(ChunkType::Struct.to_string(), "struct");
    assert_eq!(ChunkType::Enum.to_string(), "enum");
    assert_eq!(ChunkType::Interface.to_string(), "interface");
    assert_eq!(ChunkType::Trait.to_string(), "trait");
    assert_eq!(ChunkType::Impl.to_string(), "impl");
    assert_eq!(ChunkType::TextChunk.to_string(), "text_chunk");
}

#[test]
fn test_chunk_type_from_str() {
    assert_eq!(
        ChunkType::from_str("function"),
        Some(ChunkType::Function)
    );
    assert_eq!(ChunkType::from_str("METHOD"), Some(ChunkType::Method));
    assert_eq!(
        ChunkType::from_str("text_chunk"),
        Some(ChunkType::TextChunk)
    );
    assert_eq!(ChunkType::from_str("impl"), Some(ChunkType::Impl));
    assert_eq!(ChunkType::from_str("invalid"), Option::None);
}

#[test]
fn test_tracked_files_sql_is_valid() {
    assert!(CREATE_TRACKED_FILES_SQL.contains("CREATE TABLE"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("tracked_files"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("file_id INTEGER PRIMARY KEY AUTOINCREMENT"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("watch_folder_id TEXT NOT NULL"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("file_path TEXT NOT NULL"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("file_hash TEXT NOT NULL"));
    assert!(CREATE_TRACKED_FILES_SQL.contains(
        "FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id)"
    ));
    assert!(CREATE_TRACKED_FILES_SQL.contains("UNIQUE(watch_folder_id, file_path, branch)"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("lsp_status"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("treesitter_status"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("needs_reconcile INTEGER DEFAULT 0"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("reconcile_reason TEXT"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("collection TEXT NOT NULL DEFAULT 'projects'"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("extension TEXT"));
    assert!(CREATE_TRACKED_FILES_SQL.contains("is_test INTEGER DEFAULT 0"));
}

#[test]
fn test_tracked_files_indexes_sql() {
    assert_eq!(CREATE_TRACKED_FILES_INDEXES_SQL.len(), 3);
    for idx_sql in CREATE_TRACKED_FILES_INDEXES_SQL {
        assert!(idx_sql.contains("CREATE INDEX"));
        assert!(idx_sql.contains("tracked_files"));
    }
    // Verify specific indexes exist
    let all_sql = CREATE_TRACKED_FILES_INDEXES_SQL.join(" ");
    assert!(all_sql.contains("idx_tracked_files_watch"));
    assert!(all_sql.contains("idx_tracked_files_path"));
    assert!(all_sql.contains("idx_tracked_files_branch"));
}

#[test]
fn test_qdrant_chunks_sql_is_valid() {
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("CREATE TABLE"));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("qdrant_chunks"));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("chunk_id INTEGER PRIMARY KEY AUTOINCREMENT"));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("file_id INTEGER NOT NULL"));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("point_id TEXT NOT NULL"));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("content_hash TEXT NOT NULL"));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains(
        "FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE"
    ));
    assert!(CREATE_QDRANT_CHUNKS_SQL.contains("UNIQUE(file_id, chunk_index)"));
}

#[test]
fn test_qdrant_chunks_indexes_sql() {
    assert_eq!(CREATE_QDRANT_CHUNKS_INDEXES_SQL.len(), 2);
    for idx_sql in CREATE_QDRANT_CHUNKS_INDEXES_SQL {
        assert!(idx_sql.contains("CREATE INDEX"));
        assert!(idx_sql.contains("qdrant_chunks"));
    }
    let all_sql = CREATE_QDRANT_CHUNKS_INDEXES_SQL.join(" ");
    assert!(all_sql.contains("idx_qdrant_chunks_point"));
    assert!(all_sql.contains("idx_qdrant_chunks_file"));
}

#[test]
fn test_tracked_file_struct_serde() {
    let file = TrackedFile {
        file_id: 1,
        watch_folder_id: "watch_abc".to_string(),
        file_path: "src/main.rs".to_string(),
        branch: Some("main".to_string()),
        file_type: Some("code".to_string()),
        language: Some("rust".to_string()),
        file_mtime: "2025-01-01T00:00:00Z".to_string(),
        file_hash: "abc123".to_string(),
        chunk_count: 5,
        chunking_method: Some("tree_sitter".to_string()),
        lsp_status: ProcessingStatus::Done,
        treesitter_status: ProcessingStatus::Done,
        last_error: None,
        needs_reconcile: false,
        reconcile_reason: None,
        extension: Some("rs".to_string()),
        is_test: false,
        collection: "projects".to_string(),
        base_point: None,
        relative_path: None,
        incremental: false,
        component: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
    };

    let json = serde_json::to_string(&file).expect("Failed to serialize TrackedFile");
    let deserialized: TrackedFile =
        serde_json::from_str(&json).expect("Failed to deserialize TrackedFile");

    assert_eq!(deserialized.file_id, 1);
    assert_eq!(deserialized.watch_folder_id, "watch_abc");
    assert_eq!(deserialized.file_path, "src/main.rs");
    assert_eq!(deserialized.branch, Some("main".to_string()));
    assert_eq!(deserialized.chunk_count, 5);
    assert_eq!(deserialized.lsp_status, ProcessingStatus::Done);
    assert!(!deserialized.needs_reconcile);
    assert_eq!(deserialized.extension, Some("rs".to_string()));
    assert!(!deserialized.is_test);
}

#[test]
fn test_qdrant_chunk_struct_serde() {
    let chunk = QdrantChunk {
        chunk_id: 1,
        file_id: 42,
        point_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        chunk_index: 0,
        content_hash: "def456".to_string(),
        chunk_type: Some(ChunkType::Function),
        symbol_name: Some("process_item".to_string()),
        start_line: Some(10),
        end_line: Some(50),
        created_at: "2025-01-01T00:00:00Z".to_string(),
    };

    let json = serde_json::to_string(&chunk).expect("Failed to serialize QdrantChunk");
    let deserialized: QdrantChunk =
        serde_json::from_str(&json).expect("Failed to deserialize QdrantChunk");

    assert_eq!(deserialized.chunk_id, 1);
    assert_eq!(deserialized.file_id, 42);
    assert_eq!(
        deserialized.point_id,
        "550e8400-e29b-41d4-a716-446655440000"
    );
    assert_eq!(deserialized.chunk_index, 0);
    assert_eq!(deserialized.chunk_type, Some(ChunkType::Function));
    assert_eq!(
        deserialized.symbol_name,
        Some("process_item".to_string())
    );
}

#[test]
fn test_tracked_file_nullable_fields() {
    let file = TrackedFile {
        file_id: 1,
        watch_folder_id: "w1".to_string(),
        file_path: "doc.pdf".to_string(),
        branch: None,
        file_type: None,
        language: None,
        file_mtime: "2025-01-01T00:00:00Z".to_string(),
        file_hash: "hash".to_string(),
        chunk_count: 0,
        chunking_method: None,
        lsp_status: ProcessingStatus::Skipped,
        treesitter_status: ProcessingStatus::Skipped,
        last_error: None,
        needs_reconcile: false,
        reconcile_reason: None,
        extension: None,
        is_test: false,
        collection: "projects".to_string(),
        base_point: None,
        relative_path: None,
        incremental: false,
        component: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-01T00:00:00Z".to_string(),
    };

    let json = serde_json::to_string(&file).expect("Failed to serialize");
    assert!(json.contains("\"branch\":null"));
    assert!(json.contains("\"language\":null"));
    assert!(json.contains("\"extension\":null"));
}

#[test]
fn test_qdrant_chunk_nullable_fields() {
    let chunk = QdrantChunk {
        chunk_id: 1,
        file_id: 1,
        point_id: "uuid".to_string(),
        chunk_index: 0,
        content_hash: "hash".to_string(),
        chunk_type: None,
        symbol_name: None,
        start_line: None,
        end_line: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
    };

    let json = serde_json::to_string(&chunk).expect("Failed to serialize");
    assert!(json.contains("\"chunk_type\":null"));
    assert!(json.contains("\"symbol_name\":null"));
}

#[test]
fn test_compute_content_hash() {
    let hash1 = compute_content_hash("hello world");
    let hash2 = compute_content_hash("hello world");
    let hash3 = compute_content_hash("different content");

    assert_eq!(hash1, hash2, "Same content should produce same hash");
    assert_ne!(hash1, hash3, "Different content should produce different hash");
    assert_eq!(hash1.len(), 64, "SHA256 hex string should be 64 chars");
}

#[test]
fn test_compute_relative_path() {
    assert_eq!(
        compute_relative_path("/home/user/project/src/main.rs", "/home/user/project"),
        Some("src/main.rs".to_string())
    );
    assert_eq!(
        compute_relative_path("/different/path/file.rs", "/home/user/project"),
        None
    );
    assert_eq!(
        compute_relative_path("/home/user/project/file.rs", "/home/user/project"),
        Some("file.rs".to_string())
    );
}
