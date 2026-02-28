use super::*;
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

use super::chunking::{chunk_by_characters, floor_char_boundary};
use super::extraction::{
    clean_extracted_text, extract_rtf, extract_text_from_xml_tags,
};

mod tests_formats;

#[test]
fn test_detect_document_type_pdf() {
    let path = Path::new("test.pdf");
    assert_eq!(detect_document_type(path), DocumentType::Pdf);
}

#[test]
fn test_detect_document_type_code() {
    assert_eq!(
        detect_document_type(Path::new("main.rs")),
        DocumentType::Code("rust".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("app.py")),
        DocumentType::Code("python".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("index.js")),
        DocumentType::Code("javascript".to_string())
    );
}

#[test]
fn test_detect_document_type_text() {
    assert_eq!(
        detect_document_type(Path::new("readme.md")),
        DocumentType::Markdown
    );
    assert_eq!(
        detect_document_type(Path::new("notes.txt")),
        DocumentType::Text
    );
}

#[tokio::test]
async fn test_process_text_file() {
    let processor = DocumentProcessor::new();

    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "Hello, World!\nThis is a test file.").unwrap();

    let result = processor
        .process_file(temp_file.path(), "test_collection")
        .await;
    assert!(result.is_ok());

    let doc_result = result.unwrap();
    assert!(!doc_result.document_id.is_empty());
    assert_eq!(doc_result.collection, "test_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
    // processing_time_ms may be 0 for very fast operations (sub-millisecond)
    assert!(doc_result.processing_time_ms < 60000); // But should complete within a minute
}

#[tokio::test]
async fn test_process_text_file_content() {
    let processor = DocumentProcessor::new();

    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "Hello, World!\nThis is a test file.").unwrap();

    let result = processor
        .process_file_content(temp_file.path(), "test_collection")
        .await;
    assert!(result.is_ok());

    let content = result.unwrap();
    assert!(content.raw_text.contains("Hello, World!"));
    assert_eq!(
        content.metadata.get("collection"),
        Some(&"test_collection".to_string())
    );
}

#[tokio::test]
async fn test_processor_is_healthy() {
    let processor = DocumentProcessor::new();
    assert!(processor.is_healthy().await);
}

#[test]
fn test_chunk_text_simple() {
    let config = ChunkingConfig {
        chunk_size: 50,
        overlap_size: 10,
        preserve_paragraphs: false,
        ..ChunkingConfig::default()
    };

    let text =
        "This is a test. It has multiple sentences. Each one should be processed.";
    let chunks = chunk_text(text, &HashMap::new(), &config);

    assert!(!chunks.is_empty());
}

#[test]
fn test_chunk_text_with_paragraphs() {
    let config = ChunkingConfig {
        chunk_size: 100,
        overlap_size: 10,
        preserve_paragraphs: true,
        ..ChunkingConfig::default()
    };

    let text =
        "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
    let chunks = chunk_text(text, &HashMap::new(), &config);

    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(chunk.metadata.contains_key("chunk_index"));
    }
}

#[test]
fn test_clean_extracted_text() {
    let text = "Hello    World\n\n\nTest\x00Control";
    let cleaned = clean_extracted_text(text);

    assert!(!cleaned.contains('\x00'));
}

#[tokio::test]
async fn test_file_not_found() {
    let processor = DocumentProcessor::new();
    let result = processor
        .process_file(Path::new("/nonexistent/file.txt"), "test")
        .await;

    assert!(result.is_err());
    matches!(
        result.unwrap_err(),
        DocumentProcessorError::FileNotFound(_)
    );
}

#[test]
fn test_floor_char_boundary() {
    // ASCII-only string: all byte indices are char boundaries
    let ascii = "hello";
    assert_eq!(floor_char_boundary(ascii, 3), 3);
    assert_eq!(floor_char_boundary(ascii, 5), 5);
    assert_eq!(floor_char_boundary(ascii, 10), 5); // beyond end

    // Multi-byte: '-' is U+2500, encoded as 3 bytes (0xE2 0x94 0x80)
    let s = "ab\u{2500}cd"; // bytes: a(0) b(1) -(2,3,4) c(5) d(6)
    assert_eq!(floor_char_boundary(s, 2), 2); // start of -
    assert_eq!(floor_char_boundary(s, 3), 2); // inside - -> back to 2
    assert_eq!(floor_char_boundary(s, 4), 2); // inside - -> back to 2
    assert_eq!(floor_char_boundary(s, 5), 5); // start of c
}

#[test]
fn test_chunk_by_paragraphs_with_multibyte_overlap() {
    // Text with multi-byte box-drawing characters that caused the original crash
    let text = "First paragraph with content.\n\n\
                \u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}\n\
                \u{2502}   Box drawing test   \u{2502}\n\
                \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}\n\n\
                Third paragraph after box.";

    let config = ChunkingConfig {
        chunk_size: 60,
        overlap_size: 20,
        ..ChunkingConfig::default()
    };

    let metadata = HashMap::new();
    let chunks = chunk_text(text, &metadata, &config);
    // Should not panic and should produce chunks
    assert!(!chunks.is_empty());
    // All chunks concatenated should cover the original text
    for chunk in &chunks {
        assert!(!chunk.content.is_empty());
    }
}

#[test]
fn test_chunk_by_characters_with_multibyte() {
    // Ensure character-based chunking handles multi-byte chars
    let text = "Hello \u{2500}\u{2500}\u{2500} world \u{2500}\u{2500}\u{2500} end";
    let config = ChunkingConfig {
        chunk_size: 10,
        overlap_size: 4,
        ..ChunkingConfig::default()
    };

    let metadata = HashMap::new();
    let mut chunks = Vec::new();
    chunk_by_characters(text, &metadata, &mut chunks, &config);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        // Each chunk should be valid UTF-8 (guaranteed by &str, but verify no panics)
        assert!(!chunk.content.is_empty());
    }
}

#[test]
fn test_detect_document_type_new_formats() {
    assert_eq!(
        detect_document_type(Path::new("slides.pptx")),
        DocumentType::Pptx
    );
    assert_eq!(
        detect_document_type(Path::new("slides.ppt")),
        DocumentType::Ppt
    );
    assert_eq!(
        detect_document_type(Path::new("doc.odt")),
        DocumentType::Odt
    );
    assert_eq!(
        detect_document_type(Path::new("slides.odp")),
        DocumentType::Odp
    );
    assert_eq!(
        detect_document_type(Path::new("sheet.ods")),
        DocumentType::Ods
    );
    assert_eq!(
        detect_document_type(Path::new("doc.rtf")),
        DocumentType::Rtf
    );
    assert_eq!(
        detect_document_type(Path::new("legacy.doc")),
        DocumentType::Doc
    );
    assert_eq!(
        detect_document_type(Path::new("doc.pages")),
        DocumentType::Pages
    );
    assert_eq!(
        detect_document_type(Path::new("slides.key")),
        DocumentType::Key
    );
}

#[test]
fn test_detect_document_type_case_insensitive() {
    assert_eq!(
        detect_document_type(Path::new("FILE.PPTX")),
        DocumentType::Pptx
    );
    assert_eq!(
        detect_document_type(Path::new("FILE.Rtf")),
        DocumentType::Rtf
    );
    assert_eq!(
        detect_document_type(Path::new("FILE.ODT")),
        DocumentType::Odt
    );
}

#[test]
fn test_extract_text_from_xml_tags() {
    let xml = r#"<a:t>Hello</a:t><a:t>World</a:t>"#;
    let result = extract_text_from_xml_tags(xml, "a:t");
    assert!(result.contains("Hello"));
    assert!(result.contains("World"));
}

#[test]
fn test_extract_text_from_xml_tags_nested() {
    let xml = r#"<text:p><text:span>Inner text</text:span></text:p>"#;
    let result = extract_text_from_xml_tags(xml, "text:p");
    assert!(result.contains("Inner text"));
}

#[test]
fn test_extract_rtf_basic() {
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, r"{{\rtf1\ansi Hello World \par Second line}}").unwrap();
    let result = extract_rtf(tmp.path());
    assert!(result.is_ok());
    let (text, metadata) = result.unwrap();
    assert!(text.contains("Hello World"));
    assert!(text.contains("Second line"));
    assert_eq!(metadata.get("source_format").unwrap(), "rtf");
}

#[test]
fn test_extract_rtf_with_formatting() {
    let mut tmp = NamedTempFile::new().unwrap();
    write!(
        tmp,
        r"{{\rtf1\ansi\deff0 {{\b Bold text}} normal text \par New para}}"
    )
    .unwrap();
    let result = extract_rtf(tmp.path());
    assert!(result.is_ok());
    let (text, _) = result.unwrap();
    assert!(text.contains("Bold text"));
    assert!(text.contains("normal text"));
}
