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
        extract_page_images(&doc, page_num, page_id, &mut images);
    }

    debug!(
        path = %file_path.display(),
        count = images.len(),
        "Extracted images from PDF"
    );
    images
}

/// Extract all images from a single PDF page and append them to `out`.
fn extract_page_images(
    doc: &lopdf::Document,
    page_num: u32,
    page_id: lopdf::ObjectId,
    out: &mut Vec<EmbeddedImage>,
) {
    let page_obj = match doc.get_object(page_id) {
        Ok(obj) => obj,
        Err(_) => return,
    };

    let resources = match page_obj
        .as_dict()
        .and_then(|d| d.get(b"Resources"))
        .and_then(|r| doc.dereference(r).map(|(_, obj)| obj))
        .and_then(|obj| obj.as_dict())
    {
        Ok(r) => r,
        Err(_) => return,
    };

    let xobjects = match resources
        .get(b"XObject")
        .and_then(|x| doc.dereference(x).map(|(_, obj)| obj))
        .and_then(|obj| obj.as_dict())
    {
        Ok(x) => x,
        Err(_) => return,
    };

    let mut page_position = 0usize;
    for (_name, xobj_ref) in xobjects.iter() {
        if let Some(image) = extract_xobject_image(doc, xobj_ref, page_num, page_position) {
            out.push(image);
            page_position += 1;
        }
    }
}

/// Attempt to extract a single image from an XObject reference.
///
/// Returns `None` if the XObject is not an image, has unsupported encoding,
/// has dimensions below the minimum threshold, or produces empty bytes.
fn extract_xobject_image(
    doc: &lopdf::Document,
    xobj_ref: &lopdf::Object,
    page_num: u32,
    page_position: usize,
) -> Option<EmbeddedImage> {
    let (_, xobj) = doc.dereference(xobj_ref).ok()?;
    let stream = xobj.as_stream().ok()?;

    let is_image = stream
        .dict
        .get(b"Subtype")
        .ok()
        .and_then(|v| v.as_name_str().ok())
        == Some("Image");
    if !is_image {
        return None;
    }

    let width = stream.dict.get(b"Width").ok()
        .and_then(|v| v.as_i64().ok())
        .map(|v| v as u32);
    let height = stream.dict.get(b"Height").ok()
        .and_then(|v| v.as_i64().ok())
        .map(|v| v as u32);

    if let (Some(w), Some(h)) = (width, height) {
        if w < MIN_IMAGE_DIMENSION || h < MIN_IMAGE_DIMENSION {
            return None;
        }
    }

    let filter = stream.dict.get(b"Filter").ok()
        .and_then(|v| v.as_name_str().ok())
        .unwrap_or("");

    let image_bytes = decode_image_stream(stream, filter)?;
    if image_bytes.is_empty() {
        return None;
    }

    let format = if filter == "DCTDecode" {
        ImageFormat::Jpeg
    } else {
        ImageFormat::from_bytes(&image_bytes)
    };

    Some(EmbeddedImage {
        bytes: image_bytes,
        format,
        width,
        height,
        page_number: Some(page_num),
        position_in_page: page_position,
    })
}

/// Decode raw image bytes from a stream according to its compression filter.
///
/// Returns `None` for unsupported filters.
fn decode_image_stream(stream: &lopdf::Stream, filter: &str) -> Option<Vec<u8>> {
    match filter {
        "DCTDecode" => Some(stream.content.clone()),
        "FlateDecode" => stream.decompressed_content().ok(),
        _ => {
            debug!(filter, "Skipping PDF image with unsupported filter");
            None
        }
    }
}
