/// Core synchronous image processing pipeline.
///
/// Orchestrates: image extraction → thumbnail generation → CLIP embedding
/// → Qdrant storage. Processes images in batches for efficient CLIP encoding.
use std::collections::HashMap;

use serde_json::json;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::clip::{ClipEncoder, CLIP_EMBEDDING_DIM};
use crate::image_extraction::EmbeddedImage;
use crate::storage::DocumentPoint;
use crate::thumbnail;
use wqm_common::constants::field;
use wqm_common::timestamps::now_utc;

use super::types::{ImageIngestionConfig, ImageIngestionStats, ImageSourceInfo};

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
        let batch_points = process_image_batch(batch, source, clip_encoder, config, &mut stats);
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
pub(super) fn process_image_batch(
    batch: &[EmbeddedImage],
    source: &ImageSourceInfo,
    clip_encoder: &ClipEncoder,
    config: &ImageIngestionConfig,
    stats: &mut ImageIngestionStats,
) -> Vec<DocumentPoint> {
    let mut points = Vec::new();

    // First pass: filter by size and generate thumbnails
    let mut eligible: Vec<(usize, &EmbeddedImage, crate::thumbnail::ThumbnailResult)> = Vec::new();

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
        payload.insert(field::TENANT_ID.to_string(), json!(source.tenant_id));
        payload.insert(field::DOCUMENT_ID.to_string(), json!(point_id));
        payload.insert(
            field::SOURCE_DOCUMENT_ID.to_string(),
            json!(source.document_id),
        );
        payload.insert(
            field::SOURCE_COLLECTION.to_string(),
            json!(source.source_collection),
        );
        payload.insert(field::FILE_PATH.to_string(), json!(source.file_path));
        payload.insert(field::IMAGE_WIDTH.to_string(), json!(thumb.width));
        payload.insert(field::IMAGE_HEIGHT.to_string(), json!(thumb.height));
        payload.insert(field::IMAGE_FORMAT.to_string(), json!(thumb.format));
        payload.insert(field::THUMBNAIL_B64.to_string(), json!(thumb.base64));
        payload.insert(field::IMAGE_INDEX.to_string(), json!(img.position_in_page));
        payload.insert(field::INGESTION_TIMESTAMP.to_string(), json!(timestamp));

        // Optional positional metadata
        if let Some(page) = img.page_number {
            payload.insert(field::PAGE_NUMBER.to_string(), json!(page));
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
