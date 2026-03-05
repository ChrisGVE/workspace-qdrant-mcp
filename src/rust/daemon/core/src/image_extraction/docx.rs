/// DOCX image extraction from ZIP archive.
///
/// Images are stored in the `word/media/` directory within the DOCX ZIP.
use std::io::Read;
use std::path::Path;

use tracing::{debug, warn};

use super::types::{EmbeddedImage, ImageFormat};

/// Extract embedded images from a DOCX document (ZIP archive).
pub fn extract_docx_images(file_path: &Path) -> Vec<EmbeddedImage> {
    let file = match std::fs::File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            warn!(path = %file_path.display(), error = %e, "Failed to open DOCX for image extraction");
            return Vec::new();
        }
    };

    let mut archive = match zip::ZipArchive::new(file) {
        Ok(a) => a,
        Err(e) => {
            warn!(path = %file_path.display(), error = %e, "Failed to open DOCX ZIP");
            return Vec::new();
        }
    };

    let image_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"];

    // Collect media file names first (borrow checker)
    let media_names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.name_for_index(i).map(|s| s.to_string()))
        .filter(|name| {
            name.starts_with("word/media/")
                && image_extensions
                    .iter()
                    .any(|ext| name.to_lowercase().ends_with(ext))
        })
        .collect();

    let mut images = Vec::new();

    for (idx, name) in media_names.iter().enumerate() {
        let mut entry = match archive.by_name(name) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut bytes = Vec::new();
        if entry.read_to_end(&mut bytes).is_err() {
            continue;
        }

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
        "Extracted images from DOCX"
    );
    images
}
