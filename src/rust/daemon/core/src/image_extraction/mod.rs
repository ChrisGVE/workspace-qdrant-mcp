/// Image extraction from documents (PDF, DOCX, EPUB, HTML).
///
/// Extracts embedded images for OCR and/or CLIP embedding pipelines.
/// Each format has a dedicated extraction function returning `EmbeddedImage`.

mod docx;
mod epub;
mod html;
mod pdf;
mod types;

pub use types::{EmbeddedImage, ImageFormat};

pub use docx::extract_docx_images;
pub use epub::extract_epub_images;
pub use html::extract_html_images;
pub use pdf::extract_pdf_images;

use std::path::Path;

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
    use std::path::Path;

    use super::{
        extract_docx_images, extract_epub_images, extract_html_images, extract_images,
        extract_pdf_images, EmbeddedImage, ImageFormat,
    };

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
