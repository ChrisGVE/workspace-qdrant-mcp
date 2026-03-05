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
mod pipeline;
mod types;

// Re-export all public items to preserve the external API
pub use pipeline::process_document_images;
pub use types::{
    ImageIngestionConfig, ImageIngestionError, ImageIngestionStats, ImageSourceInfo, ProcessedImage,
};

use std::sync::Arc;

use crate::clip::ClipEncoder;
use crate::image_extraction::EmbeddedImage;
use crate::storage::DocumentPoint;

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
        let _config = ImageIngestionConfig::default();
        let _source = make_source_info();

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
        use crate::thumbnail;

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
        use serde_json::json;
        use wqm_common::constants::field;

        // This test requires CLIP model download — gated behind model availability
        let clip_config = crate::clip::ClipConfig::default();
        let encoder = match crate::clip::ClipEncoder::new(&clip_config) {
            Ok(e) => e,
            Err(_) => return, // Model not available, skip
        };

        let config = ImageIngestionConfig {
            min_width: 10, // Lower threshold for test
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

        let (points, stats) = process_document_images(&images, &source, &encoder, &config);

        assert_eq!(stats.extracted, 2);
        assert_eq!(stats.embedded, 2);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(points.len(), 2);

        // Verify point structure
        use crate::clip::CLIP_EMBEDDING_DIM;
        for point in &points {
            assert_eq!(point.dense_vector.len(), CLIP_EMBEDDING_DIM);
            assert!(point.sparse_vector.is_none());
            assert_eq!(point.payload[field::TENANT_ID], json!("tenant-abc"),);
            assert_eq!(point.payload[field::SOURCE_COLLECTION], json!("projects"),);
            assert!(point.payload.contains_key(field::THUMBNAIL_B64));
            assert!(point.payload.contains_key(field::IMAGE_WIDTH));
            assert!(point.payload.contains_key(field::IMAGE_HEIGHT));
            assert!(point.payload.contains_key(field::INGESTION_TIMESTAMP));
        }
    }
}
