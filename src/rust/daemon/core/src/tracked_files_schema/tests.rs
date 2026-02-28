use super::*;

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

// --- Async database tests ---

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn setup_tables(pool: &SqlitePool) {
    // Enable foreign keys
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();

    // Create watch_folders (needed for FK)
    sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(pool)
        .await
        .unwrap();

    // Create tracked_files
    sqlx::query(CREATE_TRACKED_FILES_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }

    // Create qdrant_chunks
    sqlx::query(CREATE_QDRANT_CHUNKS_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_QDRANT_CHUNKS_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }

    // Insert a test watch_folder
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/home/user/project', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(pool)
    .await
    .unwrap();
}

#[tokio::test]
async fn test_insert_and_lookup_tracked_file() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool,
        "w1",
        "src/main.rs",
        Some("main"),
        Some("code"),
        Some("rust"),
        "2025-01-01T00:00:00Z",
        "abc123hash",
        3,
        Some("tree_sitter"),
        ProcessingStatus::Done,
        ProcessingStatus::Done,
        None,
        None,
        false,
        None,
        None,
        None,
    )
    .await
    .expect("Insert failed");

    assert!(file_id > 0);

    let found = lookup_tracked_file(&pool, "w1", "src/main.rs", Some("main"))
        .await
        .expect("Lookup failed");

    assert!(found.is_some());
    let f = found.unwrap();
    assert_eq!(f.file_id, file_id);
    assert_eq!(f.file_path, "src/main.rs");
    assert_eq!(f.file_hash, "abc123hash");
    assert_eq!(f.chunk_count, 3);
    assert_eq!(f.lsp_status, ProcessingStatus::Done);
}

#[tokio::test]
async fn test_lookup_tracked_file_null_branch() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "doc.pdf", None, None, None, "2025-01-01T00:00:00Z", "hash1", 0, None,
        ProcessingStatus::None, ProcessingStatus::Skipped, None, None, false, None, None, None,
    )
    .await
    .expect("Insert failed");

    // Lookup with None branch
    let found = lookup_tracked_file(&pool, "w1", "doc.pdf", None)
        .await
        .expect("Lookup failed");
    assert!(found.is_some());
    assert_eq!(found.unwrap().file_id, file_id);

    // Lookup with "main" branch should NOT find it
    let not_found = lookup_tracked_file(&pool, "w1", "doc.pdf", Some("main"))
        .await
        .expect("Lookup failed");
    assert!(not_found.is_none());
}

#[tokio::test]
async fn test_update_tracked_file() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/main.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 3, Some("text"), ProcessingStatus::None,
        ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .expect("Insert failed");

    update_tracked_file(
        &pool, file_id, "2025-01-02T00:00:00Z", "hash2", 5, Some("tree_sitter"),
        ProcessingStatus::Done, ProcessingStatus::Done, None, None,
    )
    .await
    .expect("Update failed");

    let found = lookup_tracked_file(&pool, "w1", "src/main.rs", Some("main"))
        .await
        .expect("Lookup failed")
        .unwrap();

    assert_eq!(found.file_hash, "hash2");
    assert_eq!(found.chunk_count, 5);
    assert_eq!(found.chunking_method, Some("tree_sitter".to_string()));
    assert_eq!(found.lsp_status, ProcessingStatus::Done);
    assert!(found.last_error.is_none(), "Update should clear last_error");
}

#[tokio::test]
async fn test_insert_and_get_qdrant_chunks() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/lib.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 2, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let chunks = vec![
        (
            "point-1".to_string(), 0, "chash1".to_string(), Some(ChunkType::Function),
            Some("main".to_string()), Some(1), Some(20),
        ),
        (
            "point-2".to_string(), 1, "chash2".to_string(), Some(ChunkType::Struct),
            Some("Config".to_string()), Some(22), Some(40),
        ),
    ];

    insert_qdrant_chunks(&pool, file_id, &chunks)
        .await
        .expect("Insert chunks failed");

    let point_ids = get_chunk_point_ids(&pool, file_id)
        .await
        .expect("Get points failed");
    assert_eq!(point_ids.len(), 2);
    assert!(point_ids.contains(&"point-1".to_string()));
    assert!(point_ids.contains(&"point-2".to_string()));
}

#[tokio::test]
async fn test_delete_tracked_file_cascades_chunks() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/main.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 1, None, ProcessingStatus::None,
        ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let chunks = vec![(
        "point-1".to_string(), 0, "chash1".to_string(), None, None, None, None,
    )];
    insert_qdrant_chunks(&pool, file_id, &chunks).await.unwrap();

    let points_before = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(points_before.len(), 1);

    delete_tracked_file(&pool, file_id)
        .await
        .expect("Delete failed");

    let points_after = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(points_after.len(), 0, "Chunks should be deleted via CASCADE");
}

#[tokio::test]
async fn test_get_tracked_file_paths() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    insert_tracked_file(
        &pool, "w1", "src/main.rs", Some("main"), None, None, "2025-01-01T00:00:00Z", "h1", 0,
        None, ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    insert_tracked_file(
        &pool, "w1", "src/lib.rs", Some("main"), None, None, "2025-01-01T00:00:00Z", "h2", 0,
        None, ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let paths = get_tracked_file_paths(&pool, "w1")
        .await
        .expect("Query failed");
    assert_eq!(paths.len(), 2);

    let file_names: Vec<&str> = paths.iter().map(|(_, p, _)| p.as_str()).collect();
    assert!(file_names.contains(&"src/main.rs"));
    assert!(file_names.contains(&"src/lib.rs"));
}

#[tokio::test]
async fn test_lookup_watch_folder() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let result = lookup_watch_folder(&pool, "t1", "projects")
        .await
        .expect("Lookup failed");
    assert!(result.is_some());
    let (wid, path) = result.unwrap();
    assert_eq!(wid, "w1");
    assert_eq!(path, "/home/user/project");

    let missing = lookup_watch_folder(&pool, "nonexistent", "projects")
        .await
        .expect("Lookup failed");
    assert!(missing.is_none());
}

#[tokio::test]
async fn test_delete_qdrant_chunks_explicit() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "file.rs", Some("main"), None, None, "2025-01-01T00:00:00Z", "h1", 2, None,
        ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let chunks = vec![
        ("p1".to_string(), 0, "c1".to_string(), None, None, None, None),
        ("p2".to_string(), 1, "c2".to_string(), None, None, None, None),
    ];
    insert_qdrant_chunks(&pool, file_id, &chunks).await.unwrap();

    delete_qdrant_chunks(&pool, file_id)
        .await
        .expect("Delete chunks failed");

    let points = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(points.len(), 0);

    let file = lookup_tracked_file(&pool, "w1", "file.rs", Some("main"))
        .await
        .unwrap();
    assert!(file.is_some());
}

// --- Transaction-aware function tests ---

#[tokio::test]
async fn test_insert_tracked_file_tx_commit() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mut tx = pool.begin().await.unwrap();
    let file_id = insert_tracked_file_tx(
        &mut tx, "w1", "src/tx_test.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "txhash1", 2, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .expect("Tx insert failed");
    tx.commit().await.unwrap();

    assert!(file_id > 0);
    let found = lookup_tracked_file(&pool, "w1", "src/tx_test.rs", Some("main"))
        .await
        .unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().file_hash, "txhash1");
}

#[tokio::test]
async fn test_insert_tracked_file_tx_rollback() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    {
        let mut tx = pool.begin().await.unwrap();
        let _file_id = insert_tracked_file_tx(
            &mut tx, "w1", "src/rollback.rs", Some("main"), Some("code"), Some("rust"),
            "2025-01-01T00:00:00Z", "rollback_hash", 1, None, ProcessingStatus::None,
            ProcessingStatus::None, None, None, false, None, None, None,
        )
        .await
        .expect("Tx insert failed");
        // Drop tx without committing = implicit rollback
    }

    let found = lookup_tracked_file(&pool, "w1", "src/rollback.rs", Some("main"))
        .await
        .unwrap();
    assert!(
        found.is_none(),
        "Rolled-back insert should not be visible"
    );
}

#[tokio::test]
async fn test_transaction_atomicity_insert_and_chunks() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mut tx = pool.begin().await.unwrap();
    let file_id = insert_tracked_file_tx(
        &mut tx, "w1", "src/atomic.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "atomic_hash", 2, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let chunks = vec![
        (
            "pt-1".to_string(), 0, "ch1".to_string(), Some(ChunkType::Function),
            Some("main".to_string()), Some(1), Some(20),
        ),
        (
            "pt-2".to_string(), 1, "ch2".to_string(), Some(ChunkType::Struct),
            Some("Config".to_string()), Some(22), Some(40),
        ),
    ];
    insert_qdrant_chunks_tx(&mut tx, file_id, &chunks)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let found = lookup_tracked_file(&pool, "w1", "src/atomic.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(found.chunk_count, 2);
    let point_ids = get_chunk_point_ids(&pool, found.file_id).await.unwrap();
    assert_eq!(point_ids.len(), 2);
}

#[tokio::test]
async fn test_transaction_atomicity_rollback_both() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/base.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "base_hash", 0, None, ProcessingStatus::None,
        ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    {
        let mut tx = pool.begin().await.unwrap();
        update_tracked_file_tx(
            &mut tx, file_id, "2025-02-01T00:00:00Z", "new_hash", 3, Some("tree_sitter"),
            ProcessingStatus::Done, ProcessingStatus::Done, None, None,
        )
        .await
        .unwrap();

        let chunks = vec![(
            "p1".to_string(), 0, "c1".to_string(), None, None, None, None,
        )];
        insert_qdrant_chunks_tx(&mut tx, file_id, &chunks)
            .await
            .unwrap();
        // Drop tx = rollback
    }

    let found = lookup_tracked_file(&pool, "w1", "src/base.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        found.file_hash, "base_hash",
        "Hash should not have changed after rollback"
    );
    assert_eq!(
        found.chunk_count, 0,
        "Chunk count should not have changed after rollback"
    );

    let point_ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(
        point_ids.len(),
        0,
        "No chunks should exist after rollback"
    );
}

#[tokio::test]
async fn test_delete_tracked_file_tx() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/delete_tx.rs", Some("main"), None, None, "2025-01-01T00:00:00Z", "h1",
        1, None, ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None,
        None,
    )
    .await
    .unwrap();

    let chunks = vec![(
        "p1".to_string(), 0, "c1".to_string(), None, None, None, None,
    )];
    insert_qdrant_chunks(&pool, file_id, &chunks).await.unwrap();

    let mut tx = pool.begin().await.unwrap();
    delete_tracked_file_tx(&mut tx, file_id).await.unwrap();
    tx.commit().await.unwrap();

    let found = lookup_tracked_file(&pool, "w1", "src/delete_tx.rs", Some("main"))
        .await
        .unwrap();
    assert!(found.is_none(), "File should be deleted");

    let point_ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(
        point_ids.len(),
        0,
        "Chunks should be deleted via CASCADE"
    );
}

#[tokio::test]
async fn test_mark_and_query_needs_reconcile() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/reconcile.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 3, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let reconcile_files = get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(reconcile_files.len(), 0);

    mark_needs_reconcile(&pool, file_id, "test_reason: sqlite_commit_failed")
        .await
        .unwrap();

    let reconcile_files = get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(reconcile_files.len(), 1);
    assert_eq!(reconcile_files[0].file_id, file_id);
    assert!(reconcile_files[0].needs_reconcile);
    assert_eq!(
        reconcile_files[0].reconcile_reason.as_deref(),
        Some("test_reason: sqlite_commit_failed")
    );

    let mut tx = pool.begin().await.unwrap();
    clear_reconcile_flag_tx(&mut tx, file_id).await.unwrap();
    tx.commit().await.unwrap();

    let reconcile_files = get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(reconcile_files.len(), 0, "Flag should be cleared");

    let found = lookup_tracked_file(&pool, "w1", "src/reconcile.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert!(!found.needs_reconcile);
    assert!(found.reconcile_reason.is_none());
}

#[tokio::test]
async fn test_update_tracked_file_tx_clears_reconcile_flag() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/reconcile_clear.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 1, None, ProcessingStatus::None,
        ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    mark_needs_reconcile(&pool, file_id, "test_failure")
        .await
        .unwrap();

    let mut tx = pool.begin().await.unwrap();
    update_tracked_file_tx(
        &mut tx, file_id, "2025-02-01T00:00:00Z", "hash2", 5, Some("tree_sitter"),
        ProcessingStatus::Done, ProcessingStatus::Done, None, None,
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let found = lookup_tracked_file(&pool, "w1", "src/reconcile_clear.rs", Some("main"))
        .await
        .unwrap()
        .unwrap();
    assert!(
        !found.needs_reconcile,
        "Update should clear needs_reconcile"
    );
    assert!(
        found.reconcile_reason.is_none(),
        "Update should clear reconcile_reason"
    );
}

#[tokio::test]
async fn test_batch_insert_large_chunk_count() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/large.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 250, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    // Generate 250 chunks (spans 3 batches: 100 + 100 + 50)
    let chunks: Vec<_> = (0..250)
        .map(|i| {
            (
                format!("point-{}", i),
                i as i32,
                format!("hash-{}", i),
                Some(ChunkType::Function),
                Some(format!("func_{}", i)),
                Some(i as i32 * 10),
                Some(i as i32 * 10 + 9),
            )
        })
        .collect();

    insert_qdrant_chunks(&pool, file_id, &chunks)
        .await
        .expect("Batch insert failed");

    let point_ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(point_ids.len(), 250, "All 250 chunks should be inserted");
    assert!(point_ids.contains(&"point-0".to_string()));
    assert!(point_ids.contains(&"point-124".to_string()));
    assert!(point_ids.contains(&"point-249".to_string()));
}

#[tokio::test]
async fn test_batch_insert_boundary_sizes() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    for count in [1usize, 99, 100, 101] {
        let path = format!("src/boundary_{}.rs", count);
        let file_id = insert_tracked_file(
            &pool, "w1", &path, Some("main"), None, None, "2025-01-01T00:00:00Z", "h1",
            count as i32, None, ProcessingStatus::None, ProcessingStatus::None, None, None, false,
            None, None, None,
        )
        .await
        .unwrap();

        let chunks: Vec<_> = (0..count)
            .map(|i| {
                (
                    format!("p-{}-{}", count, i),
                    i as i32,
                    format!("c-{}", i),
                    None,
                    None,
                    None,
                    None,
                )
            })
            .collect();

        insert_qdrant_chunks(&pool, file_id, &chunks)
            .await
            .unwrap_or_else(|e| panic!("Failed for count={}: {}", count, e));

        let ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
        assert_eq!(
            ids.len(),
            count,
            "Expected {} chunks, got {}",
            count,
            ids.len()
        );
    }
}

#[tokio::test]
async fn test_batch_insert_empty_chunks() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let file_id = insert_tracked_file(
        &pool, "w1", "src/empty.rs", Some("main"), None, None, "2025-01-01T00:00:00Z", "h1", 0,
        None, ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    insert_qdrant_chunks(&pool, file_id, &[])
        .await
        .expect("Empty insert should succeed");

    let ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(ids.len(), 0);
}

#[tokio::test]
async fn test_batch_insert_tx_large_count() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mut tx = pool.begin().await.unwrap();
    let file_id = insert_tracked_file_tx(
        &mut tx, "w1", "src/tx_large.rs", Some("main"), Some("code"), Some("rust"),
        "2025-01-01T00:00:00Z", "hash1", 150, Some("tree_sitter"), ProcessingStatus::Done,
        ProcessingStatus::Done, None, None, false, None, None, None,
    )
    .await
    .unwrap();

    let chunks: Vec<_> = (0..150)
        .map(|i| {
            (
                format!("tp-{}", i),
                i as i32,
                format!("th-{}", i),
                Some(ChunkType::Method),
                None,
                Some(i as i32),
                Some(i as i32 + 5),
            )
        })
        .collect();

    insert_qdrant_chunks_tx(&mut tx, file_id, &chunks)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let ids = get_chunk_point_ids(&pool, file_id).await.unwrap();
    assert_eq!(ids.len(), 150, "All 150 tx chunks should be inserted");
}

#[tokio::test]
async fn test_get_tracked_files_by_prefix() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    for (path, hash) in [
        ("src/core/main.rs", "h1"),
        ("src/core/lib.rs", "h2"),
        ("src/core/utils/helpers.rs", "h3"),
        ("src/cli/main.rs", "h4"),
        ("README.md", "h5"),
    ] {
        insert_tracked_file(
            &pool, "w1", path, Some("main"), None, None, "2025-01-01T00:00:00Z", hash, 0, None,
            ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
        )
        .await
        .unwrap();
    }

    let result = get_tracked_files_by_prefix(&pool, "w1", "src/core")
        .await
        .unwrap();
    assert_eq!(
        result.len(),
        3,
        "Should match all 3 files under src/core/"
    );
    let paths: Vec<&str> = result.iter().map(|(_, p, _)| p.as_str()).collect();
    assert!(paths.contains(&"src/core/main.rs"));
    assert!(paths.contains(&"src/core/lib.rs"));
    assert!(paths.contains(&"src/core/utils/helpers.rs"));

    let result2 = get_tracked_files_by_prefix(&pool, "w1", "src/core/")
        .await
        .unwrap();
    assert_eq!(
        result2.len(),
        3,
        "Trailing slash should not affect result"
    );

    let result3 = get_tracked_files_by_prefix(&pool, "w1", "src")
        .await
        .unwrap();
    assert_eq!(result3.len(), 4, "Should match all 4 files under src/");

    let result4 = get_tracked_files_by_prefix(&pool, "w1", "src/cli")
        .await
        .unwrap();
    assert_eq!(result4.len(), 1);

    let result5 = get_tracked_files_by_prefix(&pool, "w1", "nonexistent")
        .await
        .unwrap();
    assert_eq!(result5.len(), 0);
}

#[tokio::test]
async fn test_get_tracked_files_by_prefix_no_false_positives() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    for (path, hash) in [
        ("src/core/main.rs", "h1"),
        ("src/core_utils/helpers.rs", "h2"),
    ] {
        insert_tracked_file(
            &pool, "w1", path, Some("main"), None, None, "2025-01-01T00:00:00Z", hash, 0, None,
            ProcessingStatus::None, ProcessingStatus::None, None, None, false, None, None, None,
        )
        .await
        .unwrap();
    }

    // "src/core" should NOT match "src/core_utils/helpers.rs"
    let result = get_tracked_files_by_prefix(&pool, "w1", "src/core")
        .await
        .unwrap();
    assert_eq!(
        result.len(),
        1,
        "Should only match src/core/ not src/core_utils/"
    );
    assert_eq!(result[0].1, "src/core/main.rs");
}
