/// Types for the image ingestion pipeline.
///
/// Configuration, error, and data types used throughout the pipeline.

use thiserror::Error;

use crate::clip::ClipError;
use crate::storage::DocumentPoint;
use crate::thumbnail::ThumbnailError;

// ─── Configuration ──────────────────────────────────────────────────────

/// Configuration for the image ingestion pipeline.
#[derive(Debug, Clone)]
pub struct ImageIngestionConfig {
    /// Minimum width to process (skip smaller images).
    pub min_width: u32,
    /// Minimum height to process (skip smaller images).
    pub min_height: u32,
    /// Maximum images per document (prevent excessive storage).
    pub max_images_per_document: usize,
    /// CLIP encoding batch size.
    pub batch_size: usize,
}

impl Default for ImageIngestionConfig {
    fn default() -> Self {
        Self {
            min_width: 256,
            min_height: 256,
            max_images_per_document: 100,
            batch_size: 10,
        }
    }
}

// ─── Errors ─────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum ImageIngestionError {
    #[error("CLIP encoding failed: {0}")]
    ClipError(#[from] ClipError),

    #[error("Thumbnail generation failed: {0}")]
    ThumbnailError(#[from] ThumbnailError),
}

// ─── Pipeline types ─────────────────────────────────────────────────────

/// Metadata about the source document for image provenance.
#[derive(Debug, Clone)]
pub struct ImageSourceInfo {
    /// UUID of the source document point.
    pub document_id: String,
    /// Collection the source lives in (projects or libraries).
    pub source_collection: String,
    /// Tenant ID (project_id or library_name).
    pub tenant_id: String,
    /// File path of the source document.
    pub file_path: String,
}

/// Result of processing a single image.
#[derive(Debug)]
pub struct ProcessedImage {
    /// Qdrant point ready for insertion.
    pub point: DocumentPoint,
    /// Whether the image had OCR text.
    pub had_ocr: bool,
}

/// Summary of a batch image ingestion run.
#[derive(Debug, Default)]
pub struct ImageIngestionStats {
    /// Total images extracted from the document.
    pub extracted: usize,
    /// Images skipped (too small or over limit).
    pub skipped: usize,
    /// Images successfully embedded and ready for storage.
    pub embedded: usize,
    /// Images that failed during embedding or thumbnail.
    pub failed: usize,
}
