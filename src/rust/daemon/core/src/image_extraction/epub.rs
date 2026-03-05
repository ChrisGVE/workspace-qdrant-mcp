/// EPUB image extraction from EPUB manifest.
///
/// Images are referenced in the EPUB manifest with image MIME types.
use std::path::Path;

use tracing::{debug, warn};

use super::types::{EmbeddedImage, ImageFormat};

/// Extract embedded images from an EPUB document.
pub fn extract_epub_images(file_path: &Path) -> Vec<EmbeddedImage> {
    let mut doc = match epub::doc::EpubDoc::new(file_path) {
        Ok(d) => d,
        Err(e) => {
            warn!(path = %file_path.display(), error = %e, "Failed to open EPUB for image extraction");
            return Vec::new();
        }
    };

    // Collect image resource IDs
    // epub resources: HashMap<String, ResourceItem> where ResourceItem has path, mime, properties
    let image_ids: Vec<String> = doc
        .resources
        .iter()
        .filter(|(_id, item)| item.mime.starts_with("image/"))
        .map(|(id, _item)| id.clone())
        .collect();

    let mut images = Vec::new();

    for (idx, id) in image_ids.iter().enumerate() {
        // get_resource returns Option<(Vec<u8>, String)> — (bytes, mime_type)
        let (bytes, _mime_type) = match doc.get_resource(id) {
            Some(data) => data,
            None => continue,
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
