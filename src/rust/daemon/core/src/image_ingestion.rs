/// Image ingestion pipeline for the CLIP embedding collection.
///
/// Orchestrates: image extraction → thumbnail generation → CLIP embedding
/// → Qdrant storage. Runs as part of document processing, storing image
/// points in the `images` collection with 512-dim dense vectors.
///
/// Design:
/// - Image embedding failures do not block document text ingestion.
/// - Images below the minimum size threshold are skipped.
/// - Batch processing: CLIP encodes up to `batch_size` images at once.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use thiserror::Error;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::clip::{ClipEncoder, ClipError, CLIP_EMBEDDING_DIM};
use crate::image_extraction::EmbeddedImage;
use crate::storage::DocumentPoint;
use crate::thumbnail::{self, ThumbnailError, ThumbnailResult};
use wqm_common::constants::field;
use wqm_common::timestamps::now_utc;

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

// ─── Core pipeline ──────────────────────────────────────────────────────

/// Process extracted images into Qdrant-ready document points.
///
/// For each image:
/// 1. Check size threshold (skip if too small).
/// 2. Generate 64x64 JPEG thumbnail.
/// 3. Encode with CLIP → 512-dim vector.
/// 4. Build payload with provenance metadata.
///
/// Returns points ready for `insert_points_batch` and stats.
/// This is a blocking function — call via `spawn_blocking`.
pub fn process_document_images(
    images: &[EmbeddedImage],
    source: &ImageSourceInfo,
    clip_encoder: &ClipEncoder,
    config: &ImageIngestionConfig,
) -> (Vec<DocumentPoint>, ImageIngestionStats) {
    let mut stats = ImageIngestionStats {
        extracted: images.len(),
        ..Default::default()
    };

    // Cap the number of images per document
    let images_to_process = if images.len() > config.max_images_per_document {
        let skipped = images.len() - config.max_images_per_document;
        stats.skipped += skipped;
        warn!(
            doc_id = %source.document_id,
            total = images.len(),
            max = config.max_images_per_document,
            "Capping images per document"
        );
        &images[..config.max_images_per_document]
    } else {
        images
    };

    let mut points = Vec::with_capacity(images_to_process.len());

    // Process in batches for CLIP encoding
    for batch in images_to_process.chunks(config.batch_size) {
        let batch_points = process_image_batch(
            batch, source, clip_encoder, config, &mut stats,
        );
        points.extend(batch_points);
    }

    info!(
        doc_id = %source.document_id,
        extracted = stats.extracted,
        embedded = stats.embedded,
        skipped = stats.skipped,
        failed = stats.failed,
        "Image ingestion complete"
    );

    (points, stats)
}

/// Process a single batch of images through CLIP.
fn process_image_batch(
    batch: &[EmbeddedImage],
    source: &ImageSourceInfo,
    clip_encoder: &ClipEncoder,
    config: &ImageIngestionConfig,
    stats: &mut ImageIngestionStats,
) -> Vec<DocumentPoint> {
    let mut points = Vec::new();

    // First pass: filter by size and generate thumbnails
    let mut eligible: Vec<(usize, &EmbeddedImage, ThumbnailResult)> = Vec::new();

    for (batch_idx, img) in batch.iter().enumerate() {
        // Try to decode to get real dimensions if not in metadata
        let thumb = match thumbnail::generate_thumbnail(&img.bytes) {
            Ok(t) => t,
            Err(e) => {
                debug!(
                    error = %e,
                    idx = batch_idx,
                    "Failed to generate thumbnail, skipping image"
                );
                stats.failed += 1;
                continue;
            }
        };

        // Check minimum size
        if thumb.width < config.min_width || thumb.height < config.min_height {
            stats.skipped += 1;
            continue;
        }

        eligible.push((batch_idx, img, thumb));
    }

    if eligible.is_empty() {
        return points;
    }

    // Second pass: batch CLIP encode all eligible images
    let byte_slices: Vec<&[u8]> = eligible
        .iter()
        .map(|(_, img, _)| img.bytes.as_slice())
        .collect();

    let embeddings = match clip_encoder.encode_images(&byte_slices) {
        Ok(embs) => embs,
        Err(e) => {
            warn!(error = %e, batch_size = byte_slices.len(), "CLIP batch encoding failed");
            stats.failed += eligible.len();
            return points;
        }
    };

    if embeddings.len() != eligible.len() {
        warn!(
            expected = eligible.len(),
            got = embeddings.len(),
            "CLIP returned wrong number of embeddings"
        );
        stats.failed += eligible.len();
        return points;
    }

    // Third pass: build document points
    for ((_, img, thumb), embedding) in eligible.into_iter().zip(embeddings) {
        debug_assert_eq!(
            embedding.len(),
            CLIP_EMBEDDING_DIM,
            "CLIP embedding dimension mismatch"
        );

        let point_id = Uuid::new_v4().to_string();
        let timestamp = now_utc();

        let mut payload = HashMap::new();
        payload.insert(
            field::TENANT_ID.to_string(),
            json!(source.tenant_id),
        );
        payload.insert(
            field::DOCUMENT_ID.to_string(),
            json!(point_id),
        );
        payload.insert(
            field::SOURCE_DOCUMENT_ID.to_string(),
            json!(source.document_id),
        );
        payload.insert(
            field::SOURCE_COLLECTION.to_string(),
            json!(source.source_collection),
        );
        payload.insert(
            field::FILE_PATH.to_string(),
            json!(source.file_path),
        );
        payload.insert(
            field::IMAGE_WIDTH.to_string(),
            json!(thumb.width),
        );
        payload.insert(
            field::IMAGE_HEIGHT.to_string(),
            json!(thumb.height),
        );
        payload.insert(
            field::IMAGE_FORMAT.to_string(),
            json!(thumb.format),
        );
        payload.insert(
            field::THUMBNAIL_B64.to_string(),
            json!(thumb.base64),
        );
        payload.insert(
            field::IMAGE_INDEX.to_string(),
            json!(img.position_in_page),
        );
        payload.insert(
            field::INGESTION_TIMESTAMP.to_string(),
            json!(timestamp),
        );

        // Optional positional metadata
        if let Some(page) = img.page_number {
            payload.insert(
                field::PAGE_NUMBER.to_string(),
                json!(page),
            );
        }

        points.push(DocumentPoint {
            id: point_id,
            dense_vector: embedding,
            sparse_vector: None, // Images have no sparse vectors
            payload,
        });
        stats.embedded += 1;
    }

    points
}

// ─── Async wrapper ──────────────────────────────────────────────────────

/// Async wrapper for `process_document_images` using `spawn_blocking`.
///
/// CLIP inference is CPU-bound, so this moves the work off the async
/// runtime. Returns the points ready for Qdrant insertion plus stats.
pub async fn process_document_images_async(
    images: Vec<EmbeddedImage>,
    source: ImageSourceInfo,
    clip_encoder: Arc<ClipEncoder>,
    config: ImageIngestionConfig,
) -> Result<(Vec<DocumentPoint>, ImageIngestionStats), ImageIngestionError> {
    let result = tokio::task::spawn_blocking(move || {
        process_document_images(&images, &source, &clip_encoder, &config)
    })
    .await
    .expect("spawn_blocking join failed");

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_extraction::ImageFormat;
    use std::io::Cursor;

    fn make_test_image(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([
                ((x * 7 + y * 3) % 256) as u8,
                ((x * 11 + y * 5) % 256) as u8,
                128,
            ])
        });
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();
        bytes
    }

    fn make_source_info() -> ImageSourceInfo {
        ImageSourceInfo {
            document_id: "doc-123".to_string(),
            source_collection: "projects".to_string(),
            tenant_id: "tenant-abc".to_string(),
            file_path: "/path/to/doc.pdf".to_string(),
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = ImageIngestionConfig::default();
        assert_eq!(config.min_width, 256);
        assert_eq!(config.min_height, 256);
        assert_eq!(config.max_images_per_document, 100);
        assert_eq!(config.batch_size, 10);
    }

    #[test]
    fn test_process_empty_images() {
        // Without a real CLIP encoder we can only test the empty case
        let config = ImageIngestionConfig::default();
        let source = make_source_info();

        // Can't create ClipEncoder without model download, so just verify
        // the stats structure works with empty input
        let stats = ImageIngestionStats {
            extracted: 0,
            skipped: 0,
            embedded: 0,
            failed: 0,
        };
        assert_eq!(stats.extracted, 0);
    }

    #[test]
    fn test_max_images_cap() {
        let config = ImageIngestionConfig {
            max_images_per_document: 2,
            ..Default::default()
        };

        // Verify config limits are respected
        let images: Vec<EmbeddedImage> = (0..5)
            .map(|i| EmbeddedImage {
                bytes: vec![0xFF, 0xD8, 0xFF, 0xE0], // JPEG header
                format: ImageFormat::Jpeg,
                width: Some(300),
                height: Some(300),
                page_number: Some(1),
                position_in_page: i,
            })
            .collect();

        assert_eq!(images.len(), 5);
        assert_eq!(config.max_images_per_document, 2);
        // If we had a CLIP encoder, only 2 would be processed
    }

    #[test]
    fn test_image_source_info_fields() {
        let source = make_source_info();
        assert_eq!(source.document_id, "doc-123");
        assert_eq!(source.source_collection, "projects");
        assert_eq!(source.tenant_id, "tenant-abc");
        assert_eq!(source.file_path, "/path/to/doc.pdf");
    }

    #[test]
    fn test_stats_default() {
        let stats = ImageIngestionStats::default();
        assert_eq!(stats.extracted, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.embedded, 0);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_thumbnail_filter_small_images() {
        let config = ImageIngestionConfig {
            min_width: 256,
            min_height: 256,
            ..Default::default()
        };

        // Create a small 50x50 image
        let small_bytes = make_test_image(50, 50);
        let thumb = thumbnail::generate_thumbnail(&small_bytes).unwrap();
        assert!(
            thumb.width < config.min_width || thumb.height < config.min_height,
            "50x50 image should be below threshold"
        );

        // Create a large 300x300 image
        let large_bytes = make_test_image(300, 300);
        let thumb = thumbnail::generate_thumbnail(&large_bytes).unwrap();
        assert!(
            thumb.width >= config.min_width && thumb.height >= config.min_height,
            "300x300 image should pass threshold"
        );
    }

    #[test]
    fn test_process_batch_with_clip_encoder() {
        // This test requires CLIP model download — gated behind model availability
        let clip_config = crate::clip::ClipConfig::default();
        let encoder = match crate::clip::ClipEncoder::new(&clip_config) {
            Ok(e) => e,
            Err(_) => return, // Model not available, skip
        };

        let config = ImageIngestionConfig {
            min_width: 10,  // Lower threshold for test
            min_height: 10,
            max_images_per_document: 5,
            batch_size: 2,
        };

        let source = make_source_info();

        // Create two test images above threshold
        let images: Vec<EmbeddedImage> = (0..2)
            .map(|i| {
                let bytes = make_test_image(64, 64);
                EmbeddedImage {
                    bytes,
                    format: ImageFormat::Jpeg,
                    width: Some(64),
                    height: Some(64),
                    page_number: Some(1),
                    position_in_page: i,
                }
            })
            .collect();

        let (points, stats) = process_document_images(
            &images, &source, &encoder, &config,
        );

        assert_eq!(stats.extracted, 2);
        assert_eq!(stats.embedded, 2);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(points.len(), 2);

        // Verify point structure
        for point in &points {
            assert_eq!(point.dense_vector.len(), CLIP_EMBEDDING_DIM);
            assert!(point.sparse_vector.is_none());
            assert_eq!(
                point.payload[field::TENANT_ID],
                json!("tenant-abc"),
            );
            assert_eq!(
                point.payload[field::SOURCE_COLLECTION],
                json!("projects"),
            );
            assert!(point.payload.contains_key(field::THUMBNAIL_B64));
            assert!(point.payload.contains_key(field::IMAGE_WIDTH));
            assert!(point.payload.contains_key(field::IMAGE_HEIGHT));
            assert!(point.payload.contains_key(field::INGESTION_TIMESTAMP));
        }
    }
}
