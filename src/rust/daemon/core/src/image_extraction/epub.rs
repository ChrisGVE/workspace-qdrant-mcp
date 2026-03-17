/// EPUB image extraction from EPUB manifest.
///
/// Images are referenced in the EPUB manifest with image MIME types.
use std::path::Path;

use tracing::{debug, warn};

use super::types::{EmbeddedImage, ImageFormat};

/// Extract embedded images from an EPUB document.
pub fn extract_epub_images(file_path: &Path) -> Vec<EmbeddedImage> {
    let epub = match rbook::Epub::open(file_path) {
        Ok(e) => e,
        Err(e) => {
            warn!(path = %file_path.display(), error = %e, "Failed to open EPUB for image extraction");
            return Vec::new();
        }
    };

    let mut images = Vec::new();

    for (idx, entry) in epub.manifest().images().enumerate() {
        let bytes = match entry.read_bytes() {
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
            position_in_page: idx,
        });
    }

    debug!(
        path = %file_path.display(),
        count = images.len(),
        "Extracted images from EPUB"
    );
    images
}
