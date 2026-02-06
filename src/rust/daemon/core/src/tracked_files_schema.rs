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
}
