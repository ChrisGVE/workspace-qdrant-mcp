/// Image extraction from documents (PDF, DOCX, EPUB, HTML).
///
/// Extracts embedded images for OCR and/or CLIP embedding pipelines.
/// Each format has a dedicated extraction function returning `EmbeddedImage`.

use std::io::Read;
use std::path::Path;

use tracing::{debug, warn};

/// Detected image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Tiff,
    Bmp,
    Gif,
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        if data.len() < 4 {
            return Self::Unknown;
        }
        match &data[..4] {
            [0xFF, 0xD8, 0xFF, ..] => Self::Jpeg,
            [0x89, 0x50, 0x4E, 0x47] => Self::Png,
            [0x49, 0x49, 0x2A, 0x00] | [0x4D, 0x4D, 0x00, 0x2A] => Self::Tiff,
            [0x42, 0x4D, ..] => Self::Bmp,
            [0x47, 0x49, 0x46, 0x38] => Self::Gif,
            _ => Self::Unknown,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Tiff => "tiff",
            Self::Bmp => "bmp",
            Self::Gif => "gif",
            Self::Unknown => "bin",
        }
    }
}

/// An image extracted from a document.
#[derive(Debug, Clone)]
pub struct EmbeddedImage {
    /// Raw image bytes.
    pub bytes: Vec<u8>,
    /// Detected image format.
    pub format: ImageFormat,
    /// Image width (if known from metadata).
    pub width: Option<u32>,
    /// Image height (if known from metadata).
    pub height: Option<u32>,
    /// Page number (1-based, for page-based formats like PDF).
    pub page_number: Option<u32>,
    /// Position index within the page/section (0-based).
    pub position_in_page: usize,
}

/// Minimum image dimension to extract (skip tiny icons/decorations).
const MIN_IMAGE_DIMENSION: u32 = 100;

// ─── PDF Image Extraction ───────────────────────────────────────────────

/// Extract embedded images from a PDF document using lopdf.
///
/// Supports DCTDecode (JPEG) and FlateDecode (raw pixel data) streams.
/// Skips images smaller than 100x100 pixels.
pub fn extract_pdf_images(file_path: &Path) -> Vec<EmbeddedImage> {
    let doc = match lopdf::Document::load(file_path) {
        Ok(d) => d,
        Err(e) => {
            warn!(path = %file_path.display(), error = %e, "Failed to load PDF for image extraction");
            return Vec::new();
        }
    };

    let mut images = Vec::new();

    for (page_num, page_id) in doc.get_pages() {
        let mut page_position = 0;

        let page_obj = match doc.get_object(page_id) {
            Ok(obj) => obj,
            Err(_) => continue,
        };

        let resources = match page_obj
            .as_dict()
            .and_then(|d| d.get(b"Resources"))
            .and_then(|r| doc.dereference(r).map(|(_, obj)| obj))
            .and_then(|obj| obj.as_dict())
        {
            Ok(r) => r,
            Err(_) => continue,
        };

        let xobjects = match resources
            .get(b"XObject")
            .and_then(|x| doc.dereference(x).map(|(_, obj)| obj))
            .and_then(|obj| obj.as_dict())
        {
            Ok(x) => x,
            Err(_) => continue,
        };

        for (_name, xobj_ref) in xobjects.iter() {
            let (_, xobj) = match doc.dereference(xobj_ref) {
                Ok(pair) => pair,
                Err(_) => continue,
            };

            let stream = match xobj.as_stream() {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Must be an Image XObject
            let is_image = stream
                .dict
                .get(b"Subtype")
                .ok()
                .and_then(|v| v.as_name_str().ok())
                == Some("Image");

            if !is_image {
                continue;
            }

            // Read dimensions
            let width = stream.dict.get(b"Width")
                .ok()
                .and_then(|v| v.as_i64().ok())
                .map(|v| v as u32);
            let height = stream.dict.get(b"Height")
                .ok()
                .and_then(|v| v.as_i64().ok())
                .map(|v| v as u32);

            // Skip tiny images
            if let (Some(w), Some(h)) = (width, height) {
                if w < MIN_IMAGE_DIMENSION || h < MIN_IMAGE_DIMENSION {
                    continue;
                }
            }

            // Determine compression filter
            let filter = stream.dict.get(b"Filter")
                .ok()
                .and_then(|v| v.as_name_str().ok())
                .unwrap_or("");

            let image_bytes = match filter {
                "DCTDecode" => {
                    // JPEG — raw stream content is the JPEG file
                    stream.content.clone()
                }
                "FlateDecode" => {
                    // Compressed raw pixels — decompress
                    match stream.decompressed_content() {
                        Ok(data) => data,
                        Err(_) => continue,
                    }
                }
                _ => {
                    // Other filters (JPXDecode, CCITTFaxDecode, etc.) — skip for now
                    debug!(filter, "Skipping PDF image with unsupported filter");
                    continue;
                }
            };

            if image_bytes.is_empty() {
                continue;
            }

            let format = if filter == "DCTDecode" {
                ImageFormat::Jpeg
            } else {
                ImageFormat::from_bytes(&image_bytes)
            };

            images.push(EmbeddedImage {
                bytes: image_bytes,
                format,
                width,
                height,
                page_number: Some(page_num),
                position_in_page: page_position,
            });
            page_position += 1;
        }
    }

    debug!(
        path = %file_path.display(),
        count = images.len(),
        "Extracted images from PDF"
    );
    images
}

// ─── DOCX Image Extraction ─────────────────────────────────────────────

/// Extract embedded images from a DOCX document (ZIP archive).
///
/// Images are stored in `word/media/` directory within the ZIP.
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

// ─── EPUB Image Extraction ──────────────────────────────────────────────

/// Extract embedded images from an EPUB document.
///
/// Images are referenced in the EPUB manifest with image MIME types.
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

// ─── HTML Image Extraction ──────────────────────────────────────────────

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

// ─── Dispatch helper ────────────────────────────────────────────────────

/// Extract images from a document based on its file extension.
pub fn extract_images(file_path: &Path) -> Vec<EmbeddedImage> {
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf" => extract_pdf_images(file_path),
        "docx" => extract_docx_images(file_path),
        "epub" => extract_epub_images(file_path),
        "html" | "htm" => {
            match std::fs::read_to_string(file_path) {
                Ok(content) => extract_html_images(file_path, &content),
                Err(_) => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_detection() {
        assert_eq!(ImageFormat::from_bytes(&[0xFF, 0xD8, 0xFF, 0xE0]), ImageFormat::Jpeg);
        assert_eq!(ImageFormat::from_bytes(&[0x89, 0x50, 0x4E, 0x47]), ImageFormat::Png);
        assert_eq!(ImageFormat::from_bytes(&[0x42, 0x4D, 0x00, 0x00]), ImageFormat::Bmp);
        assert_eq!(ImageFormat::from_bytes(&[0x47, 0x49, 0x46, 0x38]), ImageFormat::Gif);
        assert_eq!(ImageFormat::from_bytes(&[0x49, 0x49, 0x2A, 0x00]), ImageFormat::Tiff);
        assert_eq!(ImageFormat::from_bytes(&[0x00, 0x00]), ImageFormat::Unknown);
        assert_eq!(ImageFormat::from_bytes(&[]), ImageFormat::Unknown);
    }

    #[test]
    fn test_image_format_extension() {
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Unknown.extension(), "bin");
    }

    #[test]
    fn test_extract_images_unsupported_format() {
        let path = Path::new("/tmp/test.txt");
        let images = extract_images(path);
        assert!(images.is_empty());
    }

    #[test]
    fn test_extract_pdf_images_missing_file() {
        let images = extract_pdf_images(Path::new("/nonexistent/file.pdf"));
        assert!(images.is_empty());
    }

    #[test]
    fn test_extract_docx_images_missing_file() {
        let images = extract_docx_images(Path::new("/nonexistent/file.docx"));
        assert!(images.is_empty());
    }

    #[test]
    fn test_extract_epub_images_missing_file() {
        let images = extract_epub_images(Path::new("/nonexistent/file.epub"));
        assert!(images.is_empty());
    }

    #[test]
    fn test_extract_html_images_no_images() {
        let html = "<html><body><p>No images here</p></body></html>";
        let images = extract_html_images(Path::new("/tmp/test.html"), html);
        assert!(images.is_empty());
    }

    #[test]
    fn test_extract_html_images_skips_remote_urls() {
        let html = r#"<html><body>
            <img src="https://example.com/image.png">
            <img src="http://example.com/image.jpg">
            <img src="//cdn.example.com/image.gif">
            <img src="data:image/png;base64,abc123">
        </body></html>"#;
        let images = extract_html_images(Path::new("/tmp/test.html"), html);
        assert!(images.is_empty());
    }

    #[test]
    fn test_embedded_image_struct() {
        let img = EmbeddedImage {
            bytes: vec![0xFF, 0xD8, 0xFF, 0xE0],
            format: ImageFormat::Jpeg,
            width: Some(800),
            height: Some(600),
            page_number: Some(1),
            position_in_page: 0,
        };
        assert_eq!(img.format, ImageFormat::Jpeg);
        assert_eq!(img.width, Some(800));
        assert_eq!(img.page_number, Some(1));
    }
}
