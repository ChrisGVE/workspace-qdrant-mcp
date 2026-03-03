/// Cross-modal image search: text query → CLIP text encoding → image results.
///
/// Encodes a text query with the CLIP text encoder to produce a 512-dim
/// vector, then searches the `images` Qdrant collection using dense-only
/// cosine similarity. Results include thumbnails and source provenance.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, info};

use crate::clip::{ClipEncoder, ClipError};
use crate::storage::{HybridSearchMode, SearchParams, SearchResult, StorageError, StorageClient};
use wqm_common::constants::{field, COLLECTION_IMAGES};

// ─── Errors ─────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum ImageSearchError {
    #[error("CLIP text encoding failed: {0}")]
    ClipError(#[from] ClipError),

    #[error("Qdrant search failed: {0}")]
    StorageError(#[from] StorageError),

    #[error("Empty query")]
    EmptyQuery,
}

// ─── Search types ───────────────────────────────────────────────────────

/// Parameters for cross-modal image search.
#[derive(Debug, Clone)]
pub struct ImageSearchParams {
    /// Text query to encode with CLIP text encoder.
    pub query: String,
    /// Maximum number of image results.
    pub limit: usize,
    /// Minimum cosine similarity score (0.0–1.0).
    pub score_threshold: Option<f32>,
    /// Filter by tenant_id (project scope).
    pub tenant_id: Option<String>,
    /// Filter by source collection (projects or libraries).
    pub source_collection: Option<String>,
}

/// A single image search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSearchResult {
    /// Point ID in the images collection.
    pub id: String,
    /// Cosine similarity score.
    pub score: f32,
    /// Source document this image was extracted from.
    pub source_document_id: String,
    /// Collection the source document lives in.
    pub source_collection: String,
    /// Tenant ID (project_id or library_name).
    pub tenant_id: String,
    /// Base64-encoded 64x64 JPEG thumbnail.
    pub thumbnail: String,
    /// Original image width.
    pub width: u32,
    /// Original image height.
    pub height: u32,
    /// Image format (JPEG, PNG, etc.).
    pub format: String,
    /// Page number within source document (if applicable).
    pub page_number: Option<u32>,
    /// Image index within page/section.
    pub image_index: Option<u32>,
    /// OCR-extracted text (if available).
    pub ocr_text: Option<String>,
    /// Alt text from source markup (if available).
    pub alt_text: Option<String>,
    /// Source file path.
    pub file_path: String,
}

// ─── Search implementation ──────────────────────────────────────────────

/// Execute a cross-modal image search.
///
/// 1. Encodes the text query with CLIP text encoder (blocking, via spawn_blocking).
/// 2. Searches the images collection with the resulting 512-dim vector.
/// 3. Converts raw Qdrant results into structured `ImageSearchResult`.
pub async fn search_images(
    query: &str,
    params: &ImageSearchParams,
    clip_encoder: Arc<ClipEncoder>,
    storage: &StorageClient,
) -> Result<Vec<ImageSearchResult>, ImageSearchError> {
    if query.is_empty() {
        return Err(ImageSearchError::EmptyQuery);
    }

    debug!(query, limit = params.limit, "Starting cross-modal image search");

    // Step 1: Encode text query with CLIP (CPU-bound → spawn_blocking)
    let query_owned = query.to_string();
    let text_vector = tokio::task::spawn_blocking(move || {
        clip_encoder.encode_text(&query_owned)
    })
    .await
    .expect("spawn_blocking join failed")?;

    debug!(dim = text_vector.len(), "CLIP text query encoded");

    // Step 2: Build filter from params
    let filter = build_image_filter(params);

    // Step 3: Search images collection (dense-only)
    let search_params = SearchParams {
        dense_vector: Some(text_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: params.limit,
        score_threshold: params.score_threshold,
        filter,
    };

    let raw_results = storage
        .search(COLLECTION_IMAGES, search_params)
        .await?;

    // Step 4: Convert to structured results
    let results: Vec<ImageSearchResult> = raw_results
        .into_iter()
        .filter_map(|r| convert_search_result(r))
        .collect();

    info!(
        query,
        results = results.len(),
        "Cross-modal image search complete"
    );

    Ok(results)
}

/// Build a Qdrant filter from image search params.
fn build_image_filter(
    params: &ImageSearchParams,
) -> Option<HashMap<String, Value>> {
    let mut filter = HashMap::new();

    if let Some(ref tid) = params.tenant_id {
        filter.insert(
            field::TENANT_ID.to_string(),
            serde_json::json!(tid),
        );
    }

    if let Some(ref sc) = params.source_collection {
        filter.insert(
            field::SOURCE_COLLECTION.to_string(),
            serde_json::json!(sc),
        );
    }

    if filter.is_empty() {
        None
    } else {
        Some(filter)
    }
}

/// Extract optional per-page and annotation fields from a payload.
fn extract_optional_image_fields(
    payload: &HashMap<String, Value>,
) -> (Option<u32>, Option<u32>, Option<String>, Option<String>) {
    let page_number = payload
        .get(field::PAGE_NUMBER)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let image_index = payload
        .get(field::IMAGE_INDEX)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let ocr_text = payload
        .get(field::OCR_TEXT)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let alt_text = payload
        .get(field::ALT_TEXT)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    (page_number, image_index, ocr_text, alt_text)
}

/// Convert a raw Qdrant SearchResult into a structured ImageSearchResult.
fn convert_search_result(result: SearchResult) -> Option<ImageSearchResult> {
    let payload = &result.payload;

    let source_document_id = payload
        .get(field::SOURCE_DOCUMENT_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let source_collection = payload
        .get(field::SOURCE_COLLECTION)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let tenant_id = payload
        .get(field::TENANT_ID)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let thumbnail = payload
        .get(field::THUMBNAIL_B64)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let width = payload
        .get(field::IMAGE_WIDTH)
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let height = payload
        .get(field::IMAGE_HEIGHT)
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let format = payload
        .get(field::IMAGE_FORMAT)
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();

    let file_path = payload
        .get(field::FILE_PATH)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let (page_number, image_index, ocr_text, alt_text) =
        extract_optional_image_fields(payload);

    Some(ImageSearchResult {
        id: result.id,
        score: result.score,
        source_document_id,
        source_collection,
        tenant_id,
        thumbnail,
        width,
        height,
        format,
        page_number,
        image_index,
        ocr_text,
        alt_text,
        file_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_filter_empty() {
        let params = ImageSearchParams {
            query: "test".to_string(),
            limit: 10,
            score_threshold: None,
            tenant_id: None,
            source_collection: None,
        };
        assert!(build_image_filter(&params).is_none());
    }

    #[test]
    fn test_build_filter_tenant_only() {
        let params = ImageSearchParams {
            query: "test".to_string(),
            limit: 10,
            score_threshold: None,
            tenant_id: Some("proj-123".to_string()),
            source_collection: None,
        };
        let filter = build_image_filter(&params).unwrap();
        assert_eq!(filter.len(), 1);
        assert_eq!(filter[field::TENANT_ID], serde_json::json!("proj-123"));
    }

    #[test]
    fn test_build_filter_both() {
        let params = ImageSearchParams {
            query: "test".to_string(),
            limit: 10,
            score_threshold: None,
            tenant_id: Some("proj-123".to_string()),
            source_collection: Some("projects".to_string()),
        };
        let filter = build_image_filter(&params).unwrap();
        assert_eq!(filter.len(), 2);
        assert_eq!(filter[field::TENANT_ID], serde_json::json!("proj-123"));
        assert_eq!(filter[field::SOURCE_COLLECTION], serde_json::json!("projects"));
    }

    #[test]
    fn test_convert_search_result_full() {
        let mut payload = HashMap::new();
        payload.insert(field::SOURCE_DOCUMENT_ID.to_string(), serde_json::json!("doc-1"));
        payload.insert(field::SOURCE_COLLECTION.to_string(), serde_json::json!("projects"));
        payload.insert(field::TENANT_ID.to_string(), serde_json::json!("tenant-a"));
        payload.insert(field::THUMBNAIL_B64.to_string(), serde_json::json!("dGh1bWI="));
        payload.insert(field::IMAGE_WIDTH.to_string(), serde_json::json!(800));
        payload.insert(field::IMAGE_HEIGHT.to_string(), serde_json::json!(600));
        payload.insert(field::IMAGE_FORMAT.to_string(), serde_json::json!("Jpeg"));
        payload.insert(field::FILE_PATH.to_string(), serde_json::json!("/path/doc.pdf"));
        payload.insert(field::PAGE_NUMBER.to_string(), serde_json::json!(3));
        payload.insert(field::IMAGE_INDEX.to_string(), serde_json::json!(0));
        payload.insert(field::OCR_TEXT.to_string(), serde_json::json!("Figure 1: Architecture"));
        payload.insert(field::ALT_TEXT.to_string(), serde_json::json!("arch diagram"));

        let raw = SearchResult {
            id: "point-123".to_string(),
            score: 0.85,
            payload,
            dense_vector: None,
            sparse_vector: None,
        };

        let result = convert_search_result(raw).unwrap();
        assert_eq!(result.id, "point-123");
        assert_eq!(result.score, 0.85);
        assert_eq!(result.source_document_id, "doc-1");
        assert_eq!(result.source_collection, "projects");
        assert_eq!(result.tenant_id, "tenant-a");
        assert_eq!(result.thumbnail, "dGh1bWI=");
        assert_eq!(result.width, 800);
        assert_eq!(result.height, 600);
        assert_eq!(result.format, "Jpeg");
        assert_eq!(result.file_path, "/path/doc.pdf");
        assert_eq!(result.page_number, Some(3));
        assert_eq!(result.image_index, Some(0));
        assert_eq!(result.ocr_text.as_deref(), Some("Figure 1: Architecture"));
        assert_eq!(result.alt_text.as_deref(), Some("arch diagram"));
    }

    #[test]
    fn test_convert_search_result_minimal() {
        let payload = HashMap::new();
        let raw = SearchResult {
            id: "point-456".to_string(),
            score: 0.5,
            payload,
            dense_vector: None,
            sparse_vector: None,
        };

        let result = convert_search_result(raw).unwrap();
        assert_eq!(result.id, "point-456");
        assert_eq!(result.source_document_id, "");
        assert_eq!(result.width, 0);
        assert!(result.page_number.is_none());
        assert!(result.ocr_text.is_none());
    }

    #[test]
    fn test_image_search_params_defaults() {
        let params = ImageSearchParams {
            query: "architecture diagram".to_string(),
            limit: 5,
            score_threshold: Some(0.3),
            tenant_id: None,
            source_collection: None,
        };
        assert_eq!(params.limit, 5);
        assert_eq!(params.score_threshold, Some(0.3));
    }
}
