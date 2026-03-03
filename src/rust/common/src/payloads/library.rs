//! Payloads for project and library tenant management

use serde::{Deserialize, Serialize};

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

/// Chunking configuration for library document ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfigPayload {
    /// Target tokens per chunk (default: 105, range 90-120)
    pub chunk_target_tokens: usize,
    /// Overlap tokens between chunks (default: 12, ~10-15%)
    pub chunk_overlap_tokens: usize,
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
    /// Relative path within the library (e.g., "cs/design_patterns").
    /// Empty string for root-level documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_path: Option<String>,
    /// Source project ID when routed from a project via format-based routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_project_id: Option<String>,
    /// Chunking configuration override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunking_config: Option<ChunkingConfigPayload>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_library_document_payload_page_based() {
        let payload = LibraryDocumentPayload {
            document_path: "/docs/report.pdf".to_string(),
            library_name: "internal-docs".to_string(),
            document_type: "page_based".to_string(),
            source_format: "pdf".to_string(),
            doc_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            doc_fingerprint: Some("abc123def456".to_string()),
            library_path: None,
            source_project_id: None,
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
            library_path: Some("fiction/classics".to_string()),
            source_project_id: None,
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
            library_path: None,
            source_project_id: Some("proj-abc".to_string()),
            chunking_config: None,
        };
        let json = serde_json::to_string(&payload).unwrap();

        let back: LibraryDocumentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_format, "docx");
        assert_eq!(back.document_type, "page_based");
    }
}
