//! Apple iWork format (.pages, .key) text extraction.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::xml_utils::{clean_extracted_text, extract_text_from_xml_tags};
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// First four bytes of every ZIP archive (local file header).
const ZIP_LOCAL_HEADER: [u8; 4] = *b"PK\x03\x04";

/// Extract text from Apple iWork formats (.pages, .key) -- ZIP-based bundles
pub fn extract_iwork(
    file_path: &Path,
    format_name: &str,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), format_name.to_lowercase());

    if file_path.is_dir() {
        return Err(DocumentProcessorError::UnsupportedFormat(format!(
            "{} package bundle (directory): modern iWork documents are not supported",
            format_name
        )));
    }

    // Sniff the ZIP signature before trusting the extension: `.key` is also
    // the extension of TLS/SSH private keys and of Claude Code session key
    // files (JSON carrying a token). Those are rejected, never read as text,
    // so a secret cannot reach the index through the fallback path (#302).
    let mut file = std::fs::File::open(file_path)?;
    let mut signature = [0u8; 4];
    let has_zip_signature =
        file.read_exact(&mut signature).is_ok() && signature == ZIP_LOCAL_HEADER;
    if !has_zip_signature {
        return Err(DocumentProcessorError::UnsupportedFormat(format!(
            "not a {} document (no ZIP signature); not indexed",
            format_name
        )));
    }

    let mut archive = zip::ZipArchive::new(file).map_err(|e| {
        DocumentProcessorError::UnsupportedFormat(format!(
            "{} document is not a readable ZIP archive: {}",
            format_name, e
        ))
    })?;

    let mut text = String::new();

    // Try QuickLook preview text first (most reliable for iWork)
    if let Ok(mut preview) = archive.by_name("QuickLook/Preview.txt") {
        preview.read_to_string(&mut text)?;
    }

    // Try index.xml or Index/Document.iwa
    if text.is_empty() {
        // Try extracting from any XML files in the archive
        let xml_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive.by_index(i).ok().and_then(|f| {
                    let name = f.name().to_string();
                    if name.ends_with(".xml") {
                        Some(name)
                    } else {
                        None
                    }
                })
            })
            .collect();

        for name in &xml_names {
            if let Ok(mut f) = archive.by_name(name) {
                let mut content = String::new();
                if f.read_to_string(&mut content).is_ok() {
                    let extracted = extract_text_from_xml_tags(&content, "sf:p");
                    if !extracted.is_empty() {
                        text.push_str(&extracted);
                        text.push('\n');
                    }
                }
            }
        }
    }

    if text.is_empty() {
        return Err(DocumentProcessorError::UnsupportedFormat(format!(
            "{} document has no QuickLook preview or XML text (the modern .iwa \
             format is not supported); export it as PDF or DOCX to index it",
            format_name
        )));
    }

    Ok((clean_extracted_text(&text), metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unsupported(result: DocumentProcessorResult<(String, HashMap<String, String>)>) -> String {
        match result {
            Err(e @ DocumentProcessorError::UnsupportedFormat(_)) => e.to_string(),
            other => panic!(
                "expected UnsupportedFormat, got {:?}",
                other.map(|(t, _)| t)
            ),
        }
    }

    fn zip_with(path: &Path, entries: &[(&str, &str)]) {
        let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        for (name, body) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
    }

    #[test]
    fn json_key_file_is_rejected_not_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("82430.abc.key");
        std::fs::write(&path, r#"{"peerToken":"secret"}"#).unwrap();

        let msg = unsupported(extract_iwork(&path, "Keynote"));
        assert!(msg.contains("not a Keynote document"), "{msg}");
        assert!(!msg.contains("secret"));
    }

    #[test]
    fn package_bundle_directory_is_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("Deck.key");
        std::fs::create_dir_all(bundle.join("Index")).unwrap();

        let msg = unsupported(extract_iwork(&bundle, "Keynote"));
        assert!(msg.contains("package bundle"), "{msg}");
    }

    #[test]
    fn zip_without_text_names_the_modern_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Doc.pages");
        zip_with(&path, &[("Index/Document.iwa", "binary")]);

        let msg = unsupported(extract_iwork(&path, "Pages"));
        assert!(msg.contains(".iwa"), "{msg}");
        assert!(!msg.contains("DOCX extraction"), "{msg}");
    }

    #[test]
    fn quicklook_preview_text_is_extracted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Old.pages");
        zip_with(&path, &[("QuickLook/Preview.txt", "Hello iWork")]);

        let (text, meta) = extract_iwork(&path, "Pages").unwrap();
        assert!(text.contains("Hello iWork"));
        assert_eq!(meta.get("source_format").map(String::as_str), Some("pages"));
    }
}
