/// Thumbnail generation for the images collection.
///
/// Generates 64x64 pixel thumbnails from raw image bytes, then
/// base64-encodes for storage in Qdrant payload fields.
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::imageops::FilterType;
use image::ImageReader;
use std::io::Cursor;
use thiserror::Error;

/// Thumbnail dimension (square).
pub const THUMBNAIL_SIZE: u32 = 64;

/// Maximum base64 thumbnail size (~6KB).
pub const MAX_THUMBNAIL_BYTES: usize = 8192;

#[derive(Error, Debug)]
pub enum ThumbnailError {
    #[error("Failed to decode image: {0}")]
    DecodeFailed(String),

    #[error("Failed to encode thumbnail: {0}")]
    EncodeFailed(String),

    #[error("Empty image data")]
    EmptyInput,
}

/// Result of thumbnail generation.
#[derive(Debug)]
pub struct ThumbnailResult {
    /// Base64-encoded JPEG thumbnail (64x64).
    pub base64: String,
    /// Original image width in pixels.
    pub width: u32,
    /// Original image height in pixels.
    pub height: u32,
    /// Detected image format name.
    pub format: String,
}

/// Generate a 64x64 JPEG thumbnail from raw image bytes.
///
/// Returns the base64-encoded thumbnail along with original dimensions
/// and detected format. The thumbnail is always JPEG regardless of
/// input format, to minimize payload size.
pub fn generate_thumbnail(image_bytes: &[u8]) -> Result<ThumbnailResult, ThumbnailError> {
    if image_bytes.is_empty() {
        return Err(ThumbnailError::EmptyInput);
    }

    let reader = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| ThumbnailError::DecodeFailed(e.to_string()))?;

    let format_name = reader
        .format()
        .map(|f| format!("{:?}", f))
        .unwrap_or_else(|| "Unknown".to_string());

    let img = reader
        .decode()
        .map_err(|e| ThumbnailError::DecodeFailed(e.to_string()))?;

    let (width, height) = (img.width(), img.height());

    // Resize to 64x64 using Lanczos3 for quality
    let thumbnail = img.resize_exact(THUMBNAIL_SIZE, THUMBNAIL_SIZE, FilterType::Lanczos3);

    // Encode as JPEG with quality 75
    let mut jpeg_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut jpeg_bytes);
    thumbnail
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .map_err(|e| ThumbnailError::EncodeFailed(e.to_string()))?;

    let base64_str = BASE64.encode(&jpeg_bytes);

    Ok(ThumbnailResult {
        base64: base64_str,
        width,
        height,
        format: format_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let err = generate_thumbnail(&[]).unwrap_err();
        assert!(matches!(err, ThumbnailError::EmptyInput));
    }

    #[test]
    fn test_invalid_image_bytes() {
        let err = generate_thumbnail(b"not an image").unwrap_err();
        assert!(matches!(err, ThumbnailError::DecodeFailed(_)));
    }

    #[test]
    fn test_thumbnail_from_png() {
        // Create a minimal 2x2 red PNG
        let img = image::RgbImage::from_fn(2, 2, |_, _| image::Rgb([255, 0, 0]));
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let result = generate_thumbnail(&bytes).unwrap();
        assert_eq!(result.width, 2);
        assert_eq!(result.height, 2);
        assert!(!result.base64.is_empty());
        assert!(result.format.contains("Png"));

        // Verify base64 is valid
        let decoded = BASE64.decode(&result.base64).unwrap();
        assert!(!decoded.is_empty());
        // Should be under 6KB for a 64x64 JPEG
        assert!(decoded.len() < MAX_THUMBNAIL_BYTES);
    }

    #[test]
    fn test_thumbnail_from_jpeg() {
        // Create a 100x50 gradient image
        let img = image::RgbImage::from_fn(100, 50, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();

        let result = generate_thumbnail(&bytes).unwrap();
        assert_eq!(result.width, 100);
        assert_eq!(result.height, 50);
        assert!(result.format.contains("Jpeg"));
    }

    #[test]
    fn test_thumbnail_base64_size_reasonable() {
        // Create a 512x512 noisy image (worst case for compression)
        let img = image::RgbImage::from_fn(512, 512, |x, y| {
            image::Rgb([
                ((x * 7 + y * 13) % 256) as u8,
                ((x * 11 + y * 3) % 256) as u8,
                ((x * 5 + y * 17) % 256) as u8,
            ])
        });
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let result = generate_thumbnail(&bytes).unwrap();

        // 64x64 JPEG at q75 should be well under 8KB
        let decoded = BASE64.decode(&result.base64).unwrap();
        assert!(
            decoded.len() < MAX_THUMBNAIL_BYTES,
            "Thumbnail too large: {} bytes",
            decoded.len()
        );
    }
}
