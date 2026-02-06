//! Tracked Files and Qdrant Chunks Schema Definitions
//!
//! This module defines the types and schema for the `tracked_files` and `qdrant_chunks`
//! tables. Together they form the authoritative file inventory, replacing the need to
//! scroll Qdrant for file listings, recovery, and cleanup operations.
//!
//! Per WORKSPACE_QDRANT_MCP.md spec:
//! - `tracked_files` is written by the daemon, read by CLI
//! - `qdrant_chunks` is daemon-only (write and read)
//! - `qdrant_chunks` is a child of `tracked_files` with CASCADE delete

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Processing status for LSP and Tree-sitter enrichment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingStatus {
    /// Not yet attempted
    None,
    /// Successfully processed
    Done,
    /// Processing failed (details in `last_error`)
    Failed,
    /// Skipped (language not supported, etc.)
    Skipped,
}

impl fmt::Display for ProcessingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessingStatus::None => write!(f, "none"),
            ProcessingStatus::Done => write!(f, "done"),
            ProcessingStatus::Failed => write!(f, "failed"),
            ProcessingStatus::Skipped => write!(f, "skipped"),
        }
    }
}

impl ProcessingStatus {
    /// Parse from string (as stored in SQLite)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(ProcessingStatus::None),
            "done" => Some(ProcessingStatus::Done),
            "failed" => Some(ProcessingStatus::Failed),
            "skipped" => Some(ProcessingStatus::Skipped),
            _ => Option::None,
        }
    }
}

/// Chunk type from tree-sitter semantic chunking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    Function,
    Method,
    Class,
    Module,
    Struct,
    Enum,
    Interface,
    Trait,
    Impl,
    /// Fallback for text-based chunking (no tree-sitter)
    TextChunk,
}

impl fmt::Display for ChunkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkType::Function => write!(f, "function"),
            ChunkType::Method => write!(f, "method"),
            ChunkType::Class => write!(f, "class"),
            ChunkType::Module => write!(f, "module"),
            ChunkType::Struct => write!(f, "struct"),
            ChunkType::Enum => write!(f, "enum"),
            ChunkType::Interface => write!(f, "interface"),
            ChunkType::Trait => write!(f, "trait"),
            ChunkType::Impl => write!(f, "impl"),
            ChunkType::TextChunk => write!(f, "text_chunk"),
        }
    }
}

impl ChunkType {
    /// Parse from string (as stored in SQLite)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "function" => Some(ChunkType::Function),
            "method" => Some(ChunkType::Method),
            "class" => Some(ChunkType::Class),
            "module" => Some(ChunkType::Module),
            "struct" => Some(ChunkType::Struct),
            "enum" => Some(ChunkType::Enum),
            "interface" => Some(ChunkType::Interface),
            "trait" => Some(ChunkType::Trait),
            "impl" => Some(ChunkType::Impl),
            "text_chunk" => Some(ChunkType::TextChunk),
            _ => Option::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Rust structs
// ---------------------------------------------------------------------------

/// A tracked file entry representing an ingested file in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedFile {
    /// Auto-incremented primary key
    pub file_id: i64,
    /// FK to watch_folders.watch_id
    pub watch_folder_id: String,
    /// Relative path within project/library root
    pub file_path: String,
    /// Git branch (NULL for libraries or non-git contexts)
    pub branch: Option<String>,
    /// Detected file type (e.g., "code", "markdown", "config")
    pub file_type: Option<String>,
    /// Detected programming language (e.g., "rust", "python")
    pub language: Option<String>,
    /// Filesystem modification time at ingestion (ISO 8601)
    pub file_mtime: String,
    /// SHA256 hash of file content at ingestion
    pub file_hash: String,
    /// Number of Qdrant points for this file
    pub chunk_count: i32,
    /// Chunking method used (e.g., "tree_sitter", "text")
    pub chunking_method: Option<String>,
    /// LSP enrichment status
    pub lsp_status: ProcessingStatus,
    /// Tree-sitter parsing status
    pub treesitter_status: ProcessingStatus,
    /// Last error message (NULL on success)
    pub last_error: Option<String>,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Last update timestamp (ISO 8601)
    pub updated_at: String,
}

/// A Qdrant chunk entry tracking an individual point per file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantChunk {
    /// Auto-incremented primary key
    pub chunk_id: i64,
    /// FK to tracked_files.file_id
    pub file_id: i64,
    /// Qdrant point UUID
    pub point_id: String,
    /// Position within file (0-based)
    pub chunk_index: i32,
    /// SHA256 of chunk content (for surgical updates)
    pub content_hash: String,
    /// Semantic chunk type (function, class, etc.)
    pub chunk_type: Option<ChunkType>,
    /// Symbol name if from semantic chunking
    pub symbol_name: Option<String>,
    /// Start line in source file (1-based)
    pub start_line: Option<i32>,
    /// End line in source file (1-based)
    pub end_line: Option<i32>,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// SQL constants — tracked_files
// ---------------------------------------------------------------------------

/// SQL to create the tracked_files table
pub const CREATE_TRACKED_FILES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tracked_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    branch TEXT,
    file_type TEXT,
    language TEXT,
    file_mtime TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    chunking_method TEXT,
    lsp_status TEXT DEFAULT 'none' CHECK (lsp_status IN ('none', 'done', 'failed', 'skipped')),
    treesitter_status TEXT DEFAULT 'none' CHECK (treesitter_status IN ('none', 'done', 'failed', 'skipped')),
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, file_path, branch)
)
"#;

/// SQL to create indexes for the tracked_files table
pub const CREATE_TRACKED_FILES_INDEXES_SQL: &[&str] = &[
    // Index for recovery: walk all files for a project
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_watch
       ON tracked_files(watch_folder_id)"#,
    // Index for finding files by path (e.g., file watcher events)
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_path
       ON tracked_files(file_path)"#,
    // Index for branch operations
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_branch
       ON tracked_files(watch_folder_id, branch)"#,
];

// ---------------------------------------------------------------------------
// SQL constants — qdrant_chunks
// ---------------------------------------------------------------------------

/// SQL to create the qdrant_chunks table
pub const CREATE_QDRANT_CHUNKS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS qdrant_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    point_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_type TEXT,
    symbol_name TEXT,
    start_line INTEGER,
    end_line INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE,
    UNIQUE(file_id, chunk_index)
)
"#;

/// SQL to create indexes for the qdrant_chunks table
pub const CREATE_QDRANT_CHUNKS_INDEXES_SQL: &[&str] = &[
    // Index for looking up chunks by Qdrant point ID
    r#"CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_point
       ON qdrant_chunks(point_id)"#,
    // Index for file's chunks
    r#"CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_file
       ON qdrant_chunks(file_id)"#,
];

// ---------------------------------------------------------------------------
// Database operations
// ---------------------------------------------------------------------------

use sha2::{Sha256, Digest};
use sqlx::{SqlitePool, Row};
use std::path::Path;

/// Compute SHA256 hash of file content
pub fn compute_file_hash(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA256 hash of a byte slice (for chunk content)
pub fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Get file modification time as ISO 8601 string
pub fn get_file_mtime(path: &Path) -> std::io::Result<String> {
    let metadata = std::fs::metadata(path)?;
    let mtime = metadata.modified()?;
    let datetime: chrono::DateTime<chrono::Utc> = mtime.into();
    Ok(datetime.to_rfc3339())
}

/// Look up a watch_folder by tenant_id and collection, return (watch_id, path)
pub async fn lookup_watch_folder(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
) -> Result<Option<(String, String)>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT watch_id, path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 AND enabled = 1 LIMIT 1"
    )
    .bind(tenant_id)
    .bind(collection)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| (r.get("watch_id"), r.get("path"))))
}

/// Look up a tracked file by (watch_folder_id, relative_path, branch)
pub async fn lookup_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    file_path: &str,
    branch: Option<&str>,
) -> Result<Option<TrackedFile>, sqlx::Error> {
    let row = match branch {
        Some(b) => {
            sqlx::query(
                "SELECT file_id, watch_folder_id, file_path, branch, file_type, language,
                        file_mtime, file_hash, chunk_count, chunking_method,
                        lsp_status, treesitter_status, last_error, created_at, updated_at
                 FROM tracked_files
                 WHERE watch_folder_id = ?1 AND file_path = ?2 AND branch = ?3"
            )
            .bind(watch_folder_id)
            .bind(file_path)
            .bind(b)
            .fetch_optional(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT file_id, watch_folder_id, file_path, branch, file_type, language,
                        file_mtime, file_hash, chunk_count, chunking_method,
                        lsp_status, treesitter_status, last_error, created_at, updated_at
                 FROM tracked_files
                 WHERE watch_folder_id = ?1 AND file_path = ?2 AND branch IS NULL"
            )
            .bind(watch_folder_id)
            .bind(file_path)
            .fetch_optional(pool)
            .await?
        }
    };

    Ok(row.map(|r| TrackedFile {
        file_id: r.get("file_id"),
        watch_folder_id: r.get("watch_folder_id"),
        file_path: r.get("file_path"),
        branch: r.get("branch"),
        file_type: r.get("file_type"),
        language: r.get("language"),
        file_mtime: r.get("file_mtime"),
        file_hash: r.get("file_hash"),
        chunk_count: r.get("chunk_count"),
        chunking_method: r.get("chunking_method"),
        lsp_status: ProcessingStatus::from_str(
            r.get::<Option<String>, _>("lsp_status").as_deref().unwrap_or("none")
        ).unwrap_or(ProcessingStatus::None),
        treesitter_status: ProcessingStatus::from_str(
            r.get::<Option<String>, _>("treesitter_status").as_deref().unwrap_or("none")
        ).unwrap_or(ProcessingStatus::None),
        last_error: r.get("last_error"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
    }))
}

/// Insert a new tracked file record, returning the file_id
pub async fn insert_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    file_path: &str,
    branch: Option<&str>,
    file_type: Option<&str>,
    language: Option<&str>,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: i32,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
) -> Result<i64, sqlx::Error> {
    let now = chrono::Utc::now().to_rfc3339();
    let result = sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_type, language,
         file_mtime, file_hash, chunk_count, chunking_method, lsp_status, treesitter_status,
         created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
    )
    .bind(watch_folder_id)
    .bind(file_path)
    .bind(branch)
    .bind(file_type)
    .bind(language)
    .bind(file_mtime)
    .bind(file_hash)
    .bind(chunk_count)
    .bind(chunking_method)
    .bind(lsp_status.to_string())
    .bind(treesitter_status.to_string())
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;

    Ok(result.last_insert_rowid())
}

/// Update an existing tracked file record
pub async fn update_tracked_file(
    pool: &SqlitePool,
    file_id: i64,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: i32,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
) -> Result<(), sqlx::Error> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "UPDATE tracked_files SET file_mtime = ?1, file_hash = ?2, chunk_count = ?3,
         chunking_method = ?4, lsp_status = ?5, treesitter_status = ?6,
         last_error = NULL, updated_at = ?7
         WHERE file_id = ?8"
    )
    .bind(file_mtime)
    .bind(file_hash)
    .bind(chunk_count)
    .bind(chunking_method)
    .bind(lsp_status.to_string())
    .bind(treesitter_status.to_string())
    .bind(&now)
    .bind(file_id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Delete a tracked file by file_id (CASCADE deletes qdrant_chunks)
pub async fn delete_tracked_file(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Insert qdrant_chunks for a file (batch)
pub async fn insert_qdrant_chunks(
    pool: &SqlitePool,
    file_id: i64,
    chunks: &[(String, i32, String, Option<ChunkType>, Option<String>, Option<i32>, Option<i32>)],
    // Each tuple: (point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line)
) -> Result<(), sqlx::Error> {
    let now = chrono::Utc::now().to_rfc3339();
    for (point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line) in chunks {
        sqlx::query(
            "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash,
             chunk_type, symbol_name, start_line, end_line, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
        )
        .bind(file_id)
        .bind(point_id)
        .bind(chunk_index)
        .bind(content_hash)
        .bind(chunk_type.as_ref().map(|ct| ct.to_string()))
        .bind(symbol_name.as_deref())
        .bind(start_line)
        .bind(end_line)
        .bind(&now)
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// Delete all qdrant_chunks for a file_id
pub async fn delete_qdrant_chunks(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM qdrant_chunks WHERE file_id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Get all point_ids for a tracked file's chunks
pub async fn get_chunk_point_ids(pool: &SqlitePool, file_id: i64) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query("SELECT point_id FROM qdrant_chunks WHERE file_id = ?1")
        .bind(file_id)
        .fetch_all(pool)
        .await?;

    Ok(rows.iter().map(|r| r.get("point_id")).collect())
}

/// Get all tracked file paths for a watch_folder (for cleanup/recovery)
pub async fn get_tracked_file_paths(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<Vec<(i64, String, Option<String>)>, sqlx::Error> {
    // Returns (file_id, file_path, branch)
    let rows = sqlx::query(
        "SELECT file_id, file_path, branch FROM tracked_files WHERE watch_folder_id = ?1"
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(|r| (r.get("file_id"), r.get("file_path"), r.get("branch"))).collect())
}

/// Compute relative path from absolute file path and watch folder base path
pub fn compute_relative_path(abs_path: &str, base_path: &str) -> Option<String> {
    let abs = Path::new(abs_path);
    let base = Path::new(base_path);
    abs.strip_prefix(base)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
        assert_eq!(ProcessingStatus::from_str("none"), Some(ProcessingStatus::None));
        assert_eq!(ProcessingStatus::from_str("done"), Some(ProcessingStatus::Done));
        assert_eq!(ProcessingStatus::from_str("FAILED"), Some(ProcessingStatus::Failed));
        assert_eq!(ProcessingStatus::from_str("Skipped"), Some(ProcessingStatus::Skipped));
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
        assert_eq!(ChunkType::from_str("function"), Some(ChunkType::Function));
        assert_eq!(ChunkType::from_str("METHOD"), Some(ChunkType::Method));
        assert_eq!(ChunkType::from_str("text_chunk"), Some(ChunkType::TextChunk));
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
        assert!(CREATE_TRACKED_FILES_SQL.contains("FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id)"));
        assert!(CREATE_TRACKED_FILES_SQL.contains("UNIQUE(watch_folder_id, file_path, branch)"));
        assert!(CREATE_TRACKED_FILES_SQL.contains("lsp_status"));
        assert!(CREATE_TRACKED_FILES_SQL.contains("treesitter_status"));
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
        assert!(CREATE_QDRANT_CHUNKS_SQL.contains("FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE"));
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
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&file).expect("Failed to serialize TrackedFile");
        let deserialized: TrackedFile = serde_json::from_str(&json).expect("Failed to deserialize TrackedFile");

        assert_eq!(deserialized.file_id, 1);
        assert_eq!(deserialized.watch_folder_id, "watch_abc");
        assert_eq!(deserialized.file_path, "src/main.rs");
        assert_eq!(deserialized.branch, Some("main".to_string()));
        assert_eq!(deserialized.chunk_count, 5);
        assert_eq!(deserialized.lsp_status, ProcessingStatus::Done);
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
        let deserialized: QdrantChunk = serde_json::from_str(&json).expect("Failed to deserialize QdrantChunk");

        assert_eq!(deserialized.chunk_id, 1);
        assert_eq!(deserialized.file_id, 42);
        assert_eq!(deserialized.point_id, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(deserialized.chunk_index, 0);
        assert_eq!(deserialized.chunk_type, Some(ChunkType::Function));
        assert_eq!(deserialized.symbol_name, Some("process_item".to_string()));
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
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&file).expect("Failed to serialize");
        assert!(json.contains("\"branch\":null"));
        assert!(json.contains("\"language\":null"));
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
        sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await.unwrap();

        // Create watch_folders (needed for FK)
        sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
            .execute(pool).await.unwrap();

        // Create tracked_files
        sqlx::query(CREATE_TRACKED_FILES_SQL).execute(pool).await.unwrap();
        for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }

        // Create qdrant_chunks
        sqlx::query(CREATE_QDRANT_CHUNKS_SQL).execute(pool).await.unwrap();
        for idx in CREATE_QDRANT_CHUNKS_INDEXES_SQL {
            sqlx::query(idx).execute(pool).await.unwrap();
        }

        // Insert a test watch_folder
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/home/user/project', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(pool).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_and_lookup_tracked_file() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_tracked_file(
            &pool, "w1", "src/main.rs", Some("main"),
            Some("code"), Some("rust"),
            "2025-01-01T00:00:00Z", "abc123hash",
            3, Some("tree_sitter"),
            ProcessingStatus::Done, ProcessingStatus::Done,
        ).await.expect("Insert failed");

        assert!(file_id > 0);

        let found = lookup_tracked_file(&pool, "w1", "src/main.rs", Some("main"))
            .await.expect("Lookup failed");

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
            &pool, "w1", "doc.pdf", None,
            None, None,
            "2025-01-01T00:00:00Z", "hash1",
            0, None,
            ProcessingStatus::None, ProcessingStatus::Skipped,
        ).await.expect("Insert failed");

        // Lookup with None branch
        let found = lookup_tracked_file(&pool, "w1", "doc.pdf", None)
            .await.expect("Lookup failed");
        assert!(found.is_some());
        assert_eq!(found.unwrap().file_id, file_id);

        // Lookup with "main" branch should NOT find it
        let not_found = lookup_tracked_file(&pool, "w1", "doc.pdf", Some("main"))
            .await.expect("Lookup failed");
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_update_tracked_file() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_tracked_file(
            &pool, "w1", "src/main.rs", Some("main"),
            Some("code"), Some("rust"),
            "2025-01-01T00:00:00Z", "hash1",
            3, Some("text"),
            ProcessingStatus::None, ProcessingStatus::None,
        ).await.expect("Insert failed");

        update_tracked_file(
            &pool, file_id,
            "2025-01-02T00:00:00Z", "hash2",
            5, Some("tree_sitter"),
            ProcessingStatus::Done, ProcessingStatus::Done,
        ).await.expect("Update failed");

        let found = lookup_tracked_file(&pool, "w1", "src/main.rs", Some("main"))
            .await.expect("Lookup failed").unwrap();

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
            &pool, "w1", "src/lib.rs", Some("main"),
            Some("code"), Some("rust"),
            "2025-01-01T00:00:00Z", "hash1",
            2, Some("tree_sitter"),
            ProcessingStatus::Done, ProcessingStatus::Done,
        ).await.unwrap();

        let chunks = vec![
            ("point-1".to_string(), 0, "chash1".to_string(), Some(ChunkType::Function), Some("main".to_string()), Some(1), Some(20)),
            ("point-2".to_string(), 1, "chash2".to_string(), Some(ChunkType::Struct), Some("Config".to_string()), Some(22), Some(40)),
        ];

        insert_qdrant_chunks(&pool, file_id, &chunks).await.expect("Insert chunks failed");

        let point_ids = get_chunk_point_ids(&pool, file_id).await.expect("Get points failed");
        assert_eq!(point_ids.len(), 2);
        assert!(point_ids.contains(&"point-1".to_string()));
        assert!(point_ids.contains(&"point-2".to_string()));
    }

    #[tokio::test]
    async fn test_delete_tracked_file_cascades_chunks() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_tracked_file(
            &pool, "w1", "src/main.rs", Some("main"),
            Some("code"), Some("rust"),
            "2025-01-01T00:00:00Z", "hash1",
            1, None,
            ProcessingStatus::None, ProcessingStatus::None,
        ).await.unwrap();

        let chunks = vec![
            ("point-1".to_string(), 0, "chash1".to_string(), None, None, None, None),
        ];
        insert_qdrant_chunks(&pool, file_id, &chunks).await.unwrap();

        // Verify chunk exists
        let points_before = get_chunk_point_ids(&pool, file_id).await.unwrap();
        assert_eq!(points_before.len(), 1);

        // Delete the tracked file
        delete_tracked_file(&pool, file_id).await.expect("Delete failed");

        // Verify chunk is also gone (CASCADE)
        let points_after = get_chunk_point_ids(&pool, file_id).await.unwrap();
        assert_eq!(points_after.len(), 0, "Chunks should be deleted via CASCADE");
    }

    #[tokio::test]
    async fn test_get_tracked_file_paths() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        insert_tracked_file(
            &pool, "w1", "src/main.rs", Some("main"),
            None, None, "2025-01-01T00:00:00Z", "h1", 0, None,
            ProcessingStatus::None, ProcessingStatus::None,
        ).await.unwrap();

        insert_tracked_file(
            &pool, "w1", "src/lib.rs", Some("main"),
            None, None, "2025-01-01T00:00:00Z", "h2", 0, None,
            ProcessingStatus::None, ProcessingStatus::None,
        ).await.unwrap();

        let paths = get_tracked_file_paths(&pool, "w1").await.expect("Query failed");
        assert_eq!(paths.len(), 2);

        let file_names: Vec<&str> = paths.iter().map(|(_, p, _)| p.as_str()).collect();
        assert!(file_names.contains(&"src/main.rs"));
        assert!(file_names.contains(&"src/lib.rs"));
    }

    #[tokio::test]
    async fn test_lookup_watch_folder() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let result = lookup_watch_folder(&pool, "t1", "projects").await.expect("Lookup failed");
        assert!(result.is_some());
        let (wid, path) = result.unwrap();
        assert_eq!(wid, "w1");
        assert_eq!(path, "/home/user/project");

        // Non-existent tenant
        let missing = lookup_watch_folder(&pool, "nonexistent", "projects").await.expect("Lookup failed");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_delete_qdrant_chunks_explicit() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let file_id = insert_tracked_file(
            &pool, "w1", "file.rs", Some("main"),
            None, None, "2025-01-01T00:00:00Z", "h1", 2, None,
            ProcessingStatus::None, ProcessingStatus::None,
        ).await.unwrap();

        let chunks = vec![
            ("p1".to_string(), 0, "c1".to_string(), None, None, None, None),
            ("p2".to_string(), 1, "c2".to_string(), None, None, None, None),
        ];
        insert_qdrant_chunks(&pool, file_id, &chunks).await.unwrap();

        // Explicit delete (not CASCADE)
        delete_qdrant_chunks(&pool, file_id).await.expect("Delete chunks failed");

        let points = get_chunk_point_ids(&pool, file_id).await.unwrap();
        assert_eq!(points.len(), 0);

        // But the tracked_file record should still exist
        let file = lookup_tracked_file(&pool, "w1", "file.rs", Some("main")).await.unwrap();
        assert!(file.is_some());
    }
}
