/// HTML image extraction for local image references.
///
/// Only extracts images from local file paths (no remote URLs).
/// Validates that paths don't traverse outside the HTML file's directory.

use std::path::Path;

use tracing::{debug, warn};

use super::types::{EmbeddedImage, ImageFormat};

/// Extract local image references from an HTML file.
///
/// Only extracts images from local file paths (no remote URLs).
/// Validates that paths don't traverse outside the HTML file's directory.
pub fn extract_html_images(
    html_path: &Path,
    html_content: &str,
) -> Vec<EmbeddedImage> {
    let base_dir = match html_path.parent() {
        Some(dir) => dir,
        None => return Vec::new(),
    };

    // Simple regex-free extraction of img src attributes
    let mut images = Vec::new();
    let mut position = 0;

    for segment in html_content.split("<img") {
        if let Some(src_start) = segment.find("src=") {
            let after_src = &segment[src_start + 4..];
            let (quote, rest) = match after_src.chars().next() {
                Some('"') => ('"', &after_src[1..]),
                Some('\'') => ('\'', &after_src[1..]),
                _ => continue,
            };
            if let Some(end) = rest.find(quote) {
                let src = &rest[..end];

                // Skip remote URLs
                if src.starts_with("http://")
                    || src.starts_with("https://")
                    || src.starts_with("data:")
                    || src.starts_with("//")
                {
                    continue;
                }

                let image_path = base_dir.join(src);

                // Security: canonicalize and verify the path is within base_dir
                let canonical = match image_path.canonicalize() {
                    Ok(p) => p,
                    Err(_) => continue, // File doesn't exist
                };
                let canonical_base = match base_dir.canonicalize() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !canonical.starts_with(&canonical_base) {
                    warn!(
                        src,
                        "Skipping HTML image: path traversal detected"
                    );
                    continue;
                }

                let bytes = match std::fs::read(&canonical) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                if bytes.is_empty() {
                    continue;
                }

                let format = ImageFormat::from_bytes(&bytes);

                images.push(EmbeddedImage {
                    bytes,
                    format,
                    width: None,
                    height: None,
                    page_number: None,
                    position_in_page: position,
                });
                position += 1;
            }
        }
    }

    debug!(
        path = %html_path.display(),
        count = images.len(),
        "Extracted images from HTML"
    );
    images
}
