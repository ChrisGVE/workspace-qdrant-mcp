/// PDF image extraction using lopdf.
///
/// Supports DCTDecode (JPEG) and FlateDecode (raw pixel data) streams.
/// Skips images smaller than 100x100 pixels.

use std::path::Path;

use tracing::{debug, warn};

use super::types::{EmbeddedImage, ImageFormat, MIN_IMAGE_DIMENSION};

/// Extract embedded images from a PDF document using lopdf.
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
