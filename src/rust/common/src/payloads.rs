//! Queue payload structs shared between daemon and CLI
//!
//! These structs represent the JSON payloads for different queue item types.

use serde::{Deserialize, Serialize};

/// Payload for text items (was "content" in old taxonomy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPayload {
    /// The actual text content
    pub content: String,
    /// Source type: scratchbook, mcp, clipboard
    pub source_type: String,
    /// Primary categorization tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_tag: Option<String>,
    /// Full hierarchical tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_tag: Option<String>,
}

/// Payload for memory rule items (queued via MCP memory tool)
///
/// Memory rules have their own payload type because they carry metadata
/// (label, scope, title, tags, priority) that must be persisted in the
/// Qdrant point payload for filtering and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPayload {
    /// Rule content text
    pub content: String,
    /// Source type (always "memory_rule")
    pub source_type: String,
    /// Rule label (identifier, max 15 chars, e.g. "prefer-uv")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Action: add, update, remove
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Scope: global or project
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Project ID for project-scoped rules
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Rule title (max 50 chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags for categorization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Priority (higher = more important)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
}

/// Payload for file items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePayload {
    /// Absolute path to the file
    pub file_path: String,
    /// File type classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_type: Option<String>,
    /// SHA256 hash for change detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,
    /// File size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Previous path before rename (used when op=Rename)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_path: Option<String>,
}

/// Payload for folder items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderPayload {
    /// Absolute path to the folder
    pub folder_path: String,
    /// Whether to scan recursively
    #[serde(default = "default_true")]
    pub recursive: bool,
    /// Maximum recursion depth
    #[serde(default = "default_recursive_depth")]
    pub recursive_depth: u32,
    /// File patterns to include
    #[serde(default)]
    pub patterns: Vec<String>,
    /// Patterns to ignore
    #[serde(default)]
    pub ignore_patterns: Vec<String>,
    /// Previous path before rename (used when op=Rename)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_path: Option<String>,
}

fn default_true() -> bool { true }
fn default_recursive_depth() -> u32 { 10 }

/// Payload for tenant items with collection="projects"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectPayload {
    /// Absolute path to project root
    pub project_root: String,
    /// Git remote URL (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_remote: Option<String>,
    /// Project type classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_type: Option<String>,
    /// Previous tenant_id before rename (used when op=Rename)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_tenant_id: Option<String>,
    /// Whether to set is_active=1 on watch_folder creation (used when op=Add)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_active: Option<bool>,
}

/// Payload for tenant items with collection="libraries"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryPayload {
    /// Library name
    pub library_name: String,
    /// Library version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_version: Option<String>,
    /// Source URL for documentation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
}

/// Payload for library document ingestion queue items.
///
/// Used when a library document (PDF, EPUB, DOCX, etc.) is enqueued for
/// processing by the daemon. The daemon uses `document_type` to select the
/// extraction pipeline and `source_format` to select the specific extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryDocumentPayload {
    /// Absolute path to the library document
    pub document_path: String,
    /// Library name (tenant_id for libraries collection)
    pub library_name: String,
    /// Processing family: "page_based" or "stream_based"
    pub document_type: String,
    /// Actual file format: "pdf", "docx", "pptx", "odt", "epub", "mobi", "html", "markdown", "text"
    pub source_format: String,
    /// Unique document identifier (UUID v5 from library_name + path)
    pub doc_id: String,
    /// SHA256 hash of file bytes for change detection and idempotency
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_fingerprint: Option<String>,
    /// Chunking configuration override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunking_config: Option<ChunkingConfigPayload>,
}

/// Chunking configuration for library document ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfigPayload {
    /// Target tokens per chunk (default: 105, range 90-120)
    pub chunk_target_tokens: usize,
    /// Overlap tokens between chunks (default: 12, ~10-15%)
    pub chunk_overlap_tokens: usize,
}

/// Payload for tenant delete operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteTenantPayload {
    /// Tenant ID to delete
    pub tenant_id_to_delete: String,
    /// Reason for deletion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Payload for doc delete operations (delete by document ID)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteDocumentPayload {
    /// Document identifier (UUID or path)
    pub document_id: String,
    /// Specific point IDs to delete (optional)
    #[serde(default)]
    pub point_ids: Vec<String>,
}

/// Payload for scratchpad items (persistent LLM scratch space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchpadPayload {
    /// The text content
    pub content: String,
    /// Optional title for the entry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// Source type (always "scratchpad")
    #[serde(default = "default_scratchpad_source")]
    pub source_type: String,
}

fn default_scratchpad_source() -> String { "scratchpad".to_string() }

/// Payload for URL fetch and ingestion items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlPayload {
    /// The URL to fetch
    pub url: String,
    /// Whether to crawl linked pages (same domain only)
    #[serde(default)]
    pub crawl: bool,
    /// Maximum crawl depth (0 = single page, default: 2)
    #[serde(default = "default_crawl_depth")]
    pub max_depth: u32,
    /// Maximum pages to crawl (default: 50)
    #[serde(default = "default_max_pages")]
    pub max_pages: u32,
    /// Content type hint from HTTP HEAD (populated by fetcher)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Library name (when storing to libraries collection)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_name: Option<String>,
    /// Title extracted from the page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

fn default_crawl_depth() -> u32 { 2 }
fn default_max_pages() -> u32 { 50 }

/// Payload for website items (multi-page crawl)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsitePayload {
    /// Root URL of the website
    pub url: String,
    /// Maximum crawl depth from root (default: 2)
    #[serde(default = "default_crawl_depth")]
    pub max_depth: u32,
    /// Maximum pages to crawl (default: 50)
    #[serde(default = "default_max_pages")]
    pub max_pages: u32,
}

/// Payload for collection-level operations (uplift, reset)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionPayload {
    /// Name of the Qdrant collection
    pub collection_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_payload_serde() {
        let payload = ContentPayload {
            content: "test content".to_string(),
            source_type: "cli".to_string(),
            main_tag: Some("tag1".to_string()),
            full_tag: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("test content"));
        assert!(json.contains("cli"));
        assert!(json.contains("tag1"));
        assert!(!json.contains("full_tag"));

        let back: ContentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "test content");
    }

    #[test]
    fn test_file_payload_serde() {
        let payload = FilePayload {
            file_path: "/src/main.rs".to_string(),
            file_type: Some("code".to_string()),
            file_hash: None,
            size_bytes: Some(1024),
            old_path: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("/src/main.rs"));
        assert!(!json.contains("file_hash"));
        assert!(!json.contains("old_path"));

        let back: FilePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.file_path, "/src/main.rs");
        assert_eq!(back.size_bytes, Some(1024));
    }

    #[test]
    fn test_file_payload_with_rename() {
        let payload = FilePayload {
            file_path: "/src/new_name.rs".to_string(),
            file_type: None,
            file_hash: None,
            size_bytes: None,
            old_path: Some("/src/old_name.rs".to_string()),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("old_path"));
        assert!(json.contains("old_name.rs"));

        let back: FilePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_path, Some("/src/old_name.rs".to_string()));
    }

    #[test]
    fn test_folder_payload_with_rename() {
        let payload = FolderPayload {
            folder_path: "/src/new_dir".to_string(),
            recursive: true,
            recursive_depth: 10,
            patterns: vec![],
            ignore_patterns: vec![],
            old_path: Some("/src/old_dir".to_string()),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("old_dir"));

        let back: FolderPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_path, Some("/src/old_dir".to_string()));
    }

    #[test]
    fn test_project_payload_with_rename() {
        let payload = ProjectPayload {
            project_root: "/home/user/project".to_string(),
            git_remote: None,
            project_type: None,
            old_tenant_id: Some("old_abc123".to_string()),
            is_active: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("old_tenant_id"));
        assert!(json.contains("old_abc123"));

        let back: ProjectPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_tenant_id, Some("old_abc123".to_string()));
    }

    #[test]
    fn test_memory_payload_full_serde() {
        let payload = MemoryPayload {
            content: "always use bun".to_string(),
            source_type: "memory_rule".to_string(),
            label: Some("prefer-bun".to_string()),
            action: Some("add".to_string()),
            scope: Some("global".to_string()),
            project_id: None,
            title: Some("Prefer bun over npm".to_string()),
            tags: Some(vec!["tooling".to_string(), "workflow".to_string()]),
            priority: Some(8),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("prefer-bun"));
        assert!(json.contains("global"));
        assert!(json.contains("tooling"));
        assert!(!json.contains("project_id"));

        let back: MemoryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.label, Some("prefer-bun".to_string()));
        assert_eq!(back.tags, Some(vec!["tooling".to_string(), "workflow".to_string()]));
        assert_eq!(back.priority, Some(8));
    }

    #[test]
    fn test_memory_payload_minimal_serde() {
        let json = r#"{"content":"test rule","source_type":"memory_rule"}"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.content, "test rule");
        assert_eq!(payload.label, None);
        assert_eq!(payload.scope, None);
        assert_eq!(payload.tags, None);
    }

    #[test]
    fn test_memory_payload_from_mcp_json() {
        // Simulate the JSON the MCP server actually sends
        let json = r#"{
            "content": "deploy after build",
            "source_type": "memory_rule",
            "label": "deploy-after-build",
            "action": "add",
            "scope": "project",
            "project_id": "abc123",
            "title": "Deploy binaries after changes",
            "tags": ["workflow", "deployment"],
            "priority": 9
        }"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.scope, Some("project".to_string()));
        assert_eq!(payload.project_id, Some("abc123".to_string()));
        assert_eq!(payload.priority, Some(9));
    }

    #[test]
    fn test_url_payload_full_serde() {
        let payload = UrlPayload {
            url: "https://example.com/docs".to_string(),
            crawl: true,
            max_depth: 3,
            max_pages: 100,
            content_type: Some("text/html".to_string()),
            library_name: Some("example-docs".to_string()),
            title: Some("Example Documentation".to_string()),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("https://example.com/docs"));
        assert!(json.contains("\"crawl\":true"));
        assert!(json.contains("\"max_depth\":3"));

        let back: UrlPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, "https://example.com/docs");
        assert!(back.crawl);
        assert_eq!(back.max_depth, 3);
        assert_eq!(back.library_name, Some("example-docs".to_string()));
    }

    #[test]
    fn test_url_payload_minimal_serde() {
        let json = r#"{"url":"https://example.com"}"#;
        let payload: UrlPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.url, "https://example.com");
        assert!(!payload.crawl);
        assert_eq!(payload.max_depth, 2);
        assert_eq!(payload.max_pages, 50);
        assert_eq!(payload.content_type, None);
        assert_eq!(payload.library_name, None);
    }

    #[test]
    fn test_scratchpad_payload_full_serde() {
        let payload = ScratchpadPayload {
            content: "design decision: use RRF for fusion".to_string(),
            title: Some("Search Architecture".to_string()),
            tags: vec!["architecture".to_string(), "search".to_string()],
            source_type: "scratchpad".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("design decision"));
        assert!(json.contains("Search Architecture"));
        assert!(json.contains("architecture"));

        let back: ScratchpadPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "design decision: use RRF for fusion");
        assert_eq!(back.title, Some("Search Architecture".to_string()));
        assert_eq!(back.tags, vec!["architecture", "search"]);
    }

    #[test]
    fn test_scratchpad_payload_minimal_serde() {
        let json = r#"{"content":"quick note"}"#;
        let payload: ScratchpadPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.content, "quick note");
        assert_eq!(payload.title, None);
        assert!(payload.tags.is_empty());
        assert_eq!(payload.source_type, "scratchpad");
    }

    #[test]
    fn test_website_payload_serde() {
        let payload = WebsitePayload {
            url: "https://docs.rs/tokio".to_string(),
            max_depth: 3,
            max_pages: 100,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("https://docs.rs/tokio"));

        let back: WebsitePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.url, "https://docs.rs/tokio");
        assert_eq!(back.max_depth, 3);
        assert_eq!(back.max_pages, 100);
    }

    #[test]
    fn test_website_payload_defaults() {
        let json = r#"{"url":"https://example.com"}"#;
        let payload: WebsitePayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.max_depth, 2);
        assert_eq!(payload.max_pages, 50);
    }

    #[test]
    fn test_collection_payload_serde() {
        let payload = CollectionPayload {
            collection_name: "projects".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("projects"));

        let back: CollectionPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.collection_name, "projects");
    }

    #[test]
    fn test_library_document_payload_page_based() {
        let payload = LibraryDocumentPayload {
            document_path: "/docs/report.pdf".to_string(),
            library_name: "internal-docs".to_string(),
            document_type: "page_based".to_string(),
            source_format: "pdf".to_string(),
            doc_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            doc_fingerprint: Some("abc123def456".to_string()),
            chunking_config: Some(ChunkingConfigPayload {
                chunk_target_tokens: 105,
                chunk_overlap_tokens: 12,
            }),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("page_based"));
        assert!(json.contains("pdf"));
        assert!(json.contains("internal-docs"));
        assert!(json.contains("chunk_target_tokens"));

        let back: LibraryDocumentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.document_type, "page_based");
        assert_eq!(back.source_format, "pdf");
        assert_eq!(back.chunking_config.unwrap().chunk_target_tokens, 105);
    }

    #[test]
    fn test_library_document_payload_stream_based() {
        let payload = LibraryDocumentPayload {
            document_path: "/docs/book.epub".to_string(),
            library_name: "reference-books".to_string(),
            document_type: "stream_based".to_string(),
            source_format: "epub".to_string(),
            doc_id: "661e8400-e29b-41d4-a716-446655440001".to_string(),
            doc_fingerprint: None,
            chunking_config: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("stream_based"));
        assert!(json.contains("epub"));
        assert!(!json.contains("doc_fingerprint"));
        assert!(!json.contains("chunking_config"));

        let back: LibraryDocumentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.document_type, "stream_based");
        assert_eq!(back.source_format, "epub");
        assert_eq!(back.doc_fingerprint, None);
    }

    #[test]
    fn test_library_document_payload_docx() {
        let payload = LibraryDocumentPayload {
            document_path: "/docs/proposal.docx".to_string(),
            library_name: "team-docs".to_string(),
            document_type: "page_based".to_string(),
            source_format: "docx".to_string(),
            doc_id: "771e8400-e29b-41d4-a716-446655440002".to_string(),
            doc_fingerprint: Some("deadbeef".to_string()),
            chunking_config: None,
        };
        let json = serde_json::to_string(&payload).unwrap();

        let back: LibraryDocumentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_format, "docx");
        assert_eq!(back.document_type, "page_based");
    }
}
