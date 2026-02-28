//! Type definitions for tracked files and Qdrant chunks

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
    /// Whether this file needs reconciliation (Qdrant/SQLite mismatch)
    pub needs_reconcile: bool,
    /// Reason for reconciliation (e.g., "sqlite_commit_failed: ...")
    pub reconcile_reason: Option<String>,
    /// File extension (lowercase, no dot, e.g. "rs", "py", "d.ts")
    pub extension: Option<String>,
    /// Whether this is a test file
    pub is_test: bool,
    /// Target Qdrant collection this file was routed to (e.g., "projects" or "libraries")
    pub collection: String,
    /// Content-addressed identity: SHA256(tenant_id|branch|relative_path|file_hash)[:32]
    pub base_point: Option<String>,
    /// Normalized relative path within project root (for stable identity)
    pub relative_path: Option<String>,
    /// Whether this file supports incremental (chunk-level) updates
    pub incremental: bool,
    /// Dot-separated component ID (e.g. "daemon.core", "cli")
    pub component: Option<String>,
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
